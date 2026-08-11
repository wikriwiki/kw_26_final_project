"""Safe background orchestration for DASOL reports.

The web layer owns request validation, snapshot/lock checks, output paths and
job state.  ``scripts/report/menu.py`` is the only executable it starts, with
an argument vector built from validated fields.  No user-provided shell
command or output path crosses this boundary.
"""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from scripts.report import llm as report_llm
from scripts.report.catalog import catalog_payload, v2_catalog_payload
from scripts.report.snapshot import SnapshotError, build_manifest, snapshot_readiness, write_manifest

from .runner import Runner
from .store import ArtifactStore, StoreError


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_bytes_atomic(path: Path, payload: bytes) -> Path:
    """Freeze an input beside the job artifacts without exposing a partial file."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as fp:
        temporary = Path(fp.name)
        fp.write(payload)
    temporary.replace(path)
    return path


def _neo4j_source_status(repo_root: Path, *, run_id: str) -> dict[str, Any]:
    """Expose configuration readiness without leaking credentials.

    The report engine reads the same ``.env``/environment contract as the
    protected Neo4j helpers.  This check is deliberately configuration-only;
    it does not invent a report or probe a database during a catalog request.
    """
    values: dict[str, str] = {}
    env_path = repo_root / "data" / "neo4j_load" / ".env"
    try:
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_DATABASE", "NEO4J_PASSWORD", "DASOL_NEO4J_RUN_ID"):
        value = os.environ.get(key)
        if value:
            values[key] = value
    configured = bool(values.get("NEO4J_PASSWORD"))
    binding = values.get("DASOL_NEO4J_RUN_ID")
    binding_declared = configured and binding == run_id
    if not configured:
        reason = "NEO4J_PASSWORD가 없어 DASOL 원본 엔진을 시작할 수 없습니다."
    elif not binding:
        reason = (
            "DASOL_NEO4J_RUN_ID가 없어 Neo4j 결과를 선택한 run snapshot에 묶을 수 없습니다. "
            "동일 snapshot을 적재한 뒤 명시적으로 설정해야 합니다."
        )
    elif binding != run_id:
        reason = f"Neo4j snapshot binding이 {binding}으로 확인되어 선택한 run {run_id}와 다릅니다."
    else:
        reason = (
            "DASOL_NEO4J_RUN_ID 환경값은 일치하지만 Neo4j 내부 원본과 run snapshot의 "
            "hash를 대조하는 검증 계약이 아직 없습니다."
        )
    return {
        "configured": configured,
        # 환경변수 자기선언은 원본 결합 증명이 아니다. 실제 DB 내부 manifest/hash
        # 대조가 구현되기 전까지 서버와 UI 모두 fail-closed로 유지한다.
        "snapshot_bound": False,
        "binding_declared": binding_declared,
        "binding_verified": False,
        "verification_level": "environment_only" if binding_declared else "unconfigured",
        "binding_run_id": binding,
        "uri": values.get("NEO4J_URI", "bolt://localhost:7687"),
        "reason": reason,
        "unknown": ["neo4j_snapshot_proof"] if binding_declared else (["NEO4J_PASSWORD"] if not configured else ["DASOL_NEO4J_RUN_ID"]),
    }


class ReportLock:
    """Atomic cross-request lock for one report process at a time."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreError(500, "보고서 lock을 읽지 못했습니다", str(exc)) from exc
        return value if isinstance(value, dict) else None

    def status(self) -> dict[str, Any]:
        owner = self.read()
        return {"locked": owner is not None, "owner": owner, "unknown": []}

    def acquire(self, *, job_id: str, run_id: str, policy_id: str) -> dict[str, Any]:
        payload = {
            "job_id": job_id,
            "run_id": run_id,
            "policy_id": policy_id,
            "pid": os.getpid(),
            "started_at": _now(),
        }
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise StoreError(409, "이미 다른 보고서 생성 job이 실행 중입니다", json.dumps(self.status(), ensure_ascii=False)) from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
                fp.write("\n")
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        return payload

    def release(self, job_id: str) -> None:
        current = self.read()
        if current and current.get("job_id") == job_id:
            self.path.unlink(missing_ok=True)


class ReportJobManager:
    """In-process job registry with a physical report lock."""

    def __init__(self, *, repo_root: Path, store: ArtifactStore, runner: Runner):
        self.repo_root = repo_root.resolve()
        self.store = store
        self.runner = runner
        self.lock = ReportLock(self.repo_root / "web" / ".report.lock")
        self._mutex = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def catalog(self, *, run_id: str, policy_id: str) -> dict[str, Any]:
        run = self.store.run_detail(run_id)
        policy = self.store.policy_detail(policy_id)
        try:
            artifacts = self.store.artifact_index().get("items", [])
        except StoreError:
            artifacts = []
        # 다른 실행에서 만든 보고서를 이 실행의 목록에 섞지 않는다.
        # 어느 실행 것인지 알 수 없는 파일도 올리지 않는다 — 고르는 사람이
        # 그게 무엇인지 확인할 방법이 없다.
        report_artifacts = []
        for item in artifacts:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", ""))
            if not path.lower().startswith("report/") or not path.lower().endswith(".html"):
                continue
            meta_path = (self.store.output_root / path).with_suffix(".meta.json")
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if meta.get("run_id") != run_id:
                continue
            report_artifacts.append({**item, **meta})
        policy_body = policy.get("policy", policy)
        payload = catalog_payload(policy_body)
        snapshot = snapshot_readiness(run_id=run_id, run=run, data_root=self.store.data_root)
        run_manifest = run.get("manifest") if isinstance(run.get("manifest"), dict) else {}
        run_plan = run.get("plan") if isinstance(run.get("plan"), dict) else {}
        artifacts = run.get("artifacts") if isinstance(run.get("artifacts"), dict) else {}
        v2_payload = v2_catalog_payload(
            policy_body,
            run_artifacts={
                "events": artifacts.get("events", artifacts.get("events_jsonl", True)),
                "metrics": artifacts.get("metrics", True),
            },
        )
        engine_v2 = self._v2_engine_status(run)
        return {
            "run": {
                "run_id": run.get("run_id", run_id),
                "status": run.get("status"),
                "root": run.get("root"),
                "days_present": run.get("days_present", []),
                "plan": run.get("plan", {}),
                "policy_id": run.get("policy_id") or run_manifest.get("policy_id") or run_plan.get("policy_id"),
                "policy_sha256": run.get("policy_sha256") or run_manifest.get("policy_sha256") or run_plan.get("policy_sha256"),
                "manifest_sha256": run.get("manifest_sha256") or run_manifest.get("sha256"),
                "unknown": run.get("unknown", []),
            },
            "policy": {
                "id": policy.get("policy", {}).get("id", policy_id),
                "file": policy.get("file"),
                "effective_from": policy.get("policy", {}).get("effective_from"),
                "name": policy.get("policy", {}).get("name", policy_id),
                "unknown": policy.get("unknown", []),
            },
            "analyses": payload["items"],
            "v2_sections": v2_payload["items"],
            "v2_required": v2_payload["required"],
            "report_artifacts": report_artifacts,
            "report_lock": self.lock.status(),
            "snapshot": snapshot,
            "engine": _neo4j_source_status(self.repo_root, run_id=run_id),
            "engine_v2": engine_v2,
            "engines": [
                {
                    "id": "v2",
                    "label": "상세 분석 보고서 (v2)",
                    "description": (
                        "run snapshot 파일만 읽어 시계열·전후 겹쳐보기·업종별 이중차분·분위별 효과와 "
                        "일관성 검증까지 담은 단일 HTML 을 만듭니다. Neo4j 가 필요 없습니다."
                    ),
                    "available": engine_v2["available"],
                    "reason": engine_v2["reason"],
                },
                {
                    "id": "dasol",
                    "label": "기존 DASOL 엔진",
                    "description": (
                        "origin/dasol 의 분석 함수를 그대로 호출합니다. Neo4j 원본이 선택한 run snapshot 에 "
                        "binding 되어 있어야 실행됩니다."
                    ),
                    "available": False,
                    "reason": _neo4j_source_status(self.repo_root, run_id=run_id)["reason"],
                },
            ],
            "llm": report_llm.provider_status(),
            "policy_binding": self.policy_binding(run_id=run_id, policy_id=policy_id, engine="v2"),
            "unknown": [],
        }

    def policy_binding(self, *, run_id: str, policy_id: str, engine: str) -> dict[str, Any]:
        """선택한 정책이 **정말 그 run 에서 돌았는지**를 확인한다.

        기존 경로는 run manifest 의 ``policy_id``/``policy_sha256`` 만 인정했는데,
        실제 산출물에는 그 manifest 가 없어 어떤 run 으로도 보고서를 만들 수 없었다.
        v2 는 대신 **결제 기록**을 근거로 쓴다. ``events.jsonl`` 의 정책 지급 내역에
        그 정책 ID 가 실제로 남아 있으면, manifest 보다 강한 실물 증거다.

        지급이 아예 없는 run(대조군·비지급형 정책)은 막지 않되, 근거의 종류를
        ``source`` 로 남겨 보고서와 API 응답이 그 사실을 숨기지 않게 한다.
        """
        manifest_policy = None
        try:
            run = self.store.run_detail(run_id)
            manifest = run.get("manifest") if isinstance(run.get("manifest"), dict) else {}
            manifest_policy = run.get("policy_id") or manifest.get("policy_id")
        except StoreError:
            run = {}
        if manifest_policy:
            if manifest_policy != policy_id:
                return {
                    "bound": False,
                    "source": "run_manifest",
                    "error": "run manifest 의 실행 정책과 선택한 정책이 다릅니다",
                    "reason": f"run={manifest_policy} · 선택={policy_id}",
                }
            return {"bound": True, "source": "run_manifest", "error": None, "reason": None}

        if engine == "dasol":
            return {
                "bound": False,
                "source": None,
                "error": "run manifest에서 실행 정책 ID와 정책 SHA-256을 검증할 수 없어 보고서를 생성하지 않습니다",
                "reason": "기존 DASOL 엔진은 manifest binding 없이는 실행하지 않습니다",
            }

        try:
            events = self.store.events_summary(run_id)
        except StoreError as exc:
            return {
                "bound": False,
                "source": None,
                "error": "run 의 결제 기록을 읽지 못해 정책 결합을 확인할 수 없습니다",
                "reason": str(exc),
            }
        if not events.get("available"):
            return {
                "bound": False,
                "source": None,
                "error": "run 에 결제 기록(events.jsonl)이 없어 정책 결합을 확인할 수 없습니다",
                "reason": events.get("reason"),
            }
        paid = events.get("policy_paid_by_policy_id") or {}
        if policy_id in paid:
            return {
                "bound": True,
                "source": "events.jsonl policy payments",
                "paid": paid.get(policy_id),
                "error": None,
                "reason": None,
            }
        if not paid:
            # 지급 실적이 전혀 없는 run — 대조군이거나 비지급형 정책이다.
            return {
                "bound": True,
                "source": "no_policy_payments",
                "error": None,
                "reason": "이 run 에는 정책 지급 기록이 없습니다. 보고서는 지급 근거 없이 관측치만 비교합니다.",
            }
        return {
            "bound": False,
            "source": "events.jsonl policy payments",
            "error": "이 run 의 결제 기록에 선택한 정책이 없습니다",
            "reason": f"기록된 정책: {', '.join(sorted(paid))} · 선택: {policy_id}",
        }

    def _v2_engine_status(self, run: dict[str, Any]) -> dict[str, Any]:
        """v2 엔진은 파일만 읽는다 — 필요한 것은 run root 와 events.jsonl 뿐이다."""
        if self.store.fixture_dir is not None:
            return {
                "available": False,
                "run_root": None,
                "events_present": False,
                "reason": "fixture 모드에서는 실제 보고서를 생성하지 않습니다.",
                "unknown": ["fixture_mode"],
            }
        root_value = run.get("root")
        if not root_value:
            return {
                "available": False,
                "run_root": None,
                "events_present": False,
                "reason": "run 산출물 경로를 알 수 없습니다.",
                "unknown": ["run_root"],
            }
        root = Path(str(root_value))
        events = root / "events.jsonl"
        if not events.is_file():
            return {
                "available": False,
                "run_root": str(root),
                "events_present": False,
                "reason": (
                    f"events.jsonl 이 없습니다 ({events}). run 종료 후 export 단계를 마쳐야 "
                    "소비 이벤트를 집계할 수 있습니다."
                ),
                "unknown": ["events.jsonl"],
            }
        return {
            "available": True,
            "run_root": str(root),
            "events_present": True,
            "events_bytes": events.stat().st_size,
            "reason": None,
            "unknown": [],
        }

    def list_jobs(self, *, run_id: str | None = None) -> dict[str, Any]:
        with self._mutex:
            jobs = list(self._jobs.values())
        if run_id:
            jobs = [job for job in jobs if job.get("run_id") == run_id]
        jobs.sort(key=lambda job: str(job.get("created_at", "")), reverse=True)
        return {"total": len(jobs), "items": [self._public(job) for job in jobs], "unknown": []}

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._mutex:
            job = self._jobs.get(job_id)
        if job is None:
            raise StoreError(404, f"보고서 job을 찾을 수 없습니다: {job_id}")
        return self._public(job)

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = str(payload.get("run_id", ""))
        policy_id = str(payload.get("policy_id", ""))
        try:
            start = date.fromisoformat(str(payload.get("start", "")))
        except ValueError as exc:
            raise StoreError(422, "보고서 시작일은 YYYY-MM-DD여야 합니다") from exc
        try:
            days = int(payload.get("days", 0))
        except (TypeError, ValueError) as exc:
            raise StoreError(422, "보고서 기간은 정수여야 합니다") from exc
        if not 1 <= days <= 365:
            raise StoreError(422, "보고서 기간은 1~365일이어야 합니다")

        policy_from = payload.get("policy_from")
        if policy_from not in (None, ""):
            try:
                date.fromisoformat(str(policy_from))
            except ValueError as exc:
                raise StoreError(422, "정책 시행일은 YYYY-MM-DD여야 합니다") from exc
            policy_from = str(policy_from)
        else:
            policy_from = None

        analyses = payload.get("analyses", [])
        if analyses is None:
            analyses = []
        if not isinstance(analyses, list) or not all(isinstance(item, str) for item in analyses):
            raise StoreError(422, "분석 항목은 문자열 배열이어야 합니다")
        include_interview = bool(payload.get("include_interview", False))
        engine = str(payload.get("engine") or "v2").lower()
        if engine not in {"v2", "dasol"}:
            raise StoreError(422, "engine 은 v2 또는 dasol 이어야 합니다")
        use_llm = bool(payload.get("use_llm", True))

        catalog = self.catalog(run_id=run_id, policy_id=policy_id)
        if catalog["run"].get("status") != "completed":
            raise StoreError(409, "완료된 run snapshot에서만 보고서를 생성할 수 있습니다")
        source_items = catalog["v2_sections"] if engine == "v2" else catalog["analyses"]
        by_id = {item["id"]: item for item in source_items}
        selected = list(dict.fromkeys(analyses))
        if not selected:
            selected = [item["id"] for item in source_items if item["applicable"]]
        if engine == "v2":
            # 항상 포함되는 절은 사용자가 빼도 되돌린다 — 근거·검증 없는 보고서를 만들지 않는다.
            selected = [item["id"] for item in source_items if item["id"] in set(selected) | set(catalog["v2_required"])]
        invalid = [item for item in selected if item not in by_id or not by_id[item]["applicable"]]
        if invalid:
            raise StoreError(422, f"적용할 수 없는 분석 항목입니다: {', '.join(invalid)}")

        if self.runner.lock.status().get("locked"):
            raise StoreError(409, "시뮬레이션 실행 lock이 있어 보고서 snapshot을 만들 수 없습니다")
        if self.store.fixture_dir is not None:
            raise StoreError(409, "fixture 모드에서는 실제 DASOL report job을 실행하지 않습니다")
        binding = self.policy_binding(run_id=run_id, policy_id=policy_id, engine=engine)
        if not binding["bound"]:
            raise StoreError(409, binding["error"], binding["reason"])
        if engine == "dasol":
            # 기존 엔진은 Neo4j 원본을 읽으므로 snapshot binding 증명 없이는 실행하지 않는다.
            if not catalog["engine"].get("configured"):
                raise StoreError(
                    503, "DASOL 원본 엔진 설정이 없어 report job을 실행할 수 없습니다", catalog["engine"].get("reason")
                )
            if not catalog["engine"].get("snapshot_bound"):
                raise StoreError(
                    503,
                    "Neo4j 원본 결과가 선택한 run snapshot에 binding되지 않아 report job을 실행할 수 없습니다",
                    catalog["engine"].get("reason"),
                )
        elif not catalog["engine_v2"].get("available"):
            # v2 는 파일만 읽는다. 읽을 파일이 없으면 그 사실을 그대로 돌려준다.
            raise StoreError(
                503,
                "v2 보고서 엔진이 읽을 run 산출물이 없습니다",
                catalog["engine_v2"].get("reason"),
            )
        if not catalog["snapshot"].get("ready"):
            raise StoreError(
                409,
                "선택한 run의 immutable snapshot을 만들 수 없습니다",
                catalog["snapshot"].get("reason"),
            )

        policy_path = (self.store.policy_dir / f"{policy_id}.json").resolve()
        if not policy_path.is_file():
            raise StoreError(404, f"정책 원본 JSON을 찾을 수 없습니다: {policy_id}")
        policy_bytes = policy_path.read_bytes()
        policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
        manifest_sha = catalog["run"].get("policy_sha256")
        if manifest_sha and policy_sha256 != manifest_sha:
            raise StoreError(409, "현재 정책 JSON이 run manifest의 정책 hash와 다릅니다")
        output_dir = (self.store.output_root / "report").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        job_id = f"rpt-{uuid.uuid4().hex[:12]}"
        out_path = output_dir / f"FINAL_REPORT_{job_id}.html"
        owner = self.lock.acquire(job_id=job_id, run_id=run_id, policy_id=policy_id)
        snapshot_path = output_dir / f"{out_path.stem}.snapshot.json"
        policy_snapshot_path = output_dir / f"{out_path.stem}.policy.json"
        try:
            # Refresh after both physical locks are held.  A catalog response
            # may have been rendered from an older run detail.
            latest_run = self.store.run_detail(run_id)
            manifest = build_manifest(
                run_id=run_id,
                run=latest_run,
                data_root=self.store.data_root,
                requested_start=start,
                requested_days=days,
            )
            manifest = {
                **manifest,
                "report_policy": {
                    "id": policy_id,
                    "sha256": policy_sha256,
                    "snapshot_path": policy_snapshot_path.name,
                },
            }
            write_manifest(snapshot_path, manifest)
            _write_bytes_atomic(policy_snapshot_path, policy_bytes)
        except SnapshotError as exc:
            self.lock.release(job_id)
            raise StoreError(409, "immutable run snapshot 검증을 통과하지 못했습니다", str(exc)) from exc
        except Exception:
            self.lock.release(job_id)
            raise
        if engine == "v2":
            command = [
                sys.executable,
                str(self.repo_root / "scripts" / "report" / "build_report_v2.py"),
                "--run-id", run_id,
                "--run-root", str(catalog["engine_v2"]["run_root"]),
                "--policy-json", str(policy_snapshot_path),
                "--start", start.isoformat(),
                "--days", str(days),
                "--out", str(out_path),
                "--snapshot-id", str(manifest["snapshot_id"]),
            ]
            if policy_from:
                command.extend(["--policy-from", policy_from])
            for section_id in selected:
                command.extend(["--section", section_id])
            if not use_llm:
                command.append("--no-llm")
        else:
            command = [
                sys.executable,
                str(self.repo_root / "scripts" / "report" / "menu.py"),
                "--run-id", run_id,
                "--policy-id", policy_id,
                "--start", start.isoformat(),
                "--days", str(days),
                "--policy-json", str(policy_snapshot_path),
                "--snapshot-manifest", str(snapshot_path),
                "--data-root", str(self.store.data_root),
                "--out", str(out_path),
            ]
            if policy_from:
                command.extend(["--policy-from", policy_from])
            for analysis_id in selected:
                command.extend(["--analysis", analysis_id])
            if include_interview:
                command.append("--include-interview")

        job: dict[str, Any] = {
            "job_id": job_id,
            "state": "queued",
            "stage": "queued",
            "engine": engine,
            "use_llm": use_llm,
            "run_id": run_id,
            "policy_id": policy_id,
            "start": start.isoformat(),
            "days": days,
            "policy_from": policy_from,
            "analyses": selected,
            "include_interview": include_interview,
            "output_path": str(out_path.relative_to(self.store.output_root).as_posix()),
            "snapshot_manifest_path": str(snapshot_path.relative_to(self.store.output_root).as_posix()),
            "snapshot_id": manifest["snapshot_id"],
            "policy_snapshot_path": str(policy_snapshot_path.relative_to(self.store.output_root).as_posix()),
            "policy_sha256": policy_sha256,
            "policy_binding": binding,
            "command": command,
            "lock": owner,
            "logs": [],
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "artifacts": [],
            "unknown": [],
        }
        with self._mutex:
            self._jobs[job_id] = job
            job["state"] = "running"
            job["stage"] = "starting"
            job["started_at"] = _now()
        thread = threading.Thread(target=self._run, args=(job_id, command), name=f"report-{job_id}", daemon=True)
        thread.start()
        return self._public(job)

    def _run(self, job_id: str, command: list[str]) -> None:
        process: subprocess.Popen[str] | None = None
        try:
            # 자식 프로세스의 stdout 도 UTF-8 로 맞춘다.
            # 윈도우에서는 파이썬이 콘솔 코드페이지(cp949)로 인코딩해서 내보내는데
            # 이쪽은 UTF-8 로 읽으므로 한글 로그가 전부 깨진 채 화면에 그대로 실린다.
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with self._mutex:
                job = self._jobs[job_id]
                job["pid"] = process.pid
                job["stage"] = "running"
            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(job_id, line.rstrip())
            exit_code = process.wait()
            with self._mutex:
                job = self._jobs[job_id]
                output = self.store.output_root / str(job["output_path"])
                markdown = output.with_suffix(".md")
                snapshot_manifest = self.store.output_root / str(job["snapshot_manifest_path"])
                policy_snapshot = self.store.output_root / str(job["policy_snapshot_path"])
                job["exit_code"] = exit_code
                data_json = output.with_suffix(".data.json")
                # v2 는 계산 결과 원본(.data.json)까지 내야 완료로 본다. 그림만 나온 보고서는
                # 재현·검증이 불가능하므로 완료로 표시하지 않는다.
                needs_data = job.get("engine") == "v2"
                complete = (
                    exit_code in (0, 3)
                    and output.is_file()
                    and markdown.is_file()
                    and snapshot_manifest.is_file()
                    and policy_snapshot.is_file()
                    and (data_json.is_file() or not needs_data)
                )
                if complete:
                    job["state"] = "completed"
                    job["stage"] = "ready"
                    # 목록에서 보고서를 고를 때 사람이 보는 것은 파일 이름이 아니라
                    # "언제 만든 · 어느 기간 · 어느 정책" 이다. 그걸 옆에 적어 둔다.
                    output.with_suffix(".meta.json").write_text(
                        json.dumps(
                            {
                                "run_id": job.get("run_id"),
                                "policy_id": job.get("policy_id"),
                                "start": job.get("start"),
                                "days": job.get("days"),
                                "policy_from": job.get("policy_from"),
                                "created_at": job.get("finished_at") or _now(),
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    # exit code 3 = 보고서는 만들어졌지만 일관성 검사가 실패한 상태.
                    job["consistent"] = exit_code == 0
                    if exit_code == 3:
                        job["unknown"] = ["consistency"]
                        job["error"] = (
                            "보고서는 생성되었지만 일관성 검증에서 어긋난 항등식이 있습니다. "
                            "보고서 마지막 절의 검증 표를 확인하세요."
                        )
                    artifacts = [
                        output.relative_to(self.store.output_root).as_posix(),
                        markdown.relative_to(self.store.output_root).as_posix(),
                        snapshot_manifest.relative_to(self.store.output_root).as_posix(),
                        policy_snapshot.relative_to(self.store.output_root).as_posix(),
                    ]
                    if data_json.is_file():
                        artifacts.insert(2, data_json.relative_to(self.store.output_root).as_posix())
                    job["artifacts"] = artifacts
                else:
                    job["state"] = "failed"
                    job["stage"] = "failed"
                    job["error"] = "보고서 엔진이 완료되지 않았습니다" if exit_code == 0 else f"보고서 엔진 exit code {exit_code}"
                    job["unknown"] = ["report_output"]
                job["finished_at"] = _now()
        except Exception as exc:  # noqa: BLE001
            with self._mutex:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["state"] = "failed"
                    job["stage"] = "failed"
                    job["error"] = str(exc)
                    job["unknown"] = ["report_runtime"]
                    job["finished_at"] = _now()
        finally:
            self.lock.release(job_id)

    def _append_log(self, job_id: str, line: str) -> None:
        if not line:
            return
        with self._mutex:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["logs"] = [*job.get("logs", []), line][-200:]
            stages_v2 = {"1": "scanning", "2": "verifying", "3": "narrating", "4": "rendering", "5": "writing"}
            stages_dasol = {"1": "conditions", "2": "analysis", "3": "interview", "4": "writing"}
            table = stages_v2 if job.get("engine") == "v2" else stages_dasol
            if line.startswith("[") and "/" in line[:5]:
                step = line[1 : line.index("/")]
                if step in table:
                    job["stage"] = table[step]

    @staticmethod
    def _public(job: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in job.items() if key != "command"}
