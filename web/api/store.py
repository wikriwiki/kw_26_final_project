"""Read-only artifact access and server-side aggregation for S2.

The store deliberately has no mock fallback. Production mode reads the three
documented run roots and the repository policy directory. Tests may pass the
checked-in S1 fixtures explicitly; that mode is opt-in and never selected by
the production environment.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
from datetime import date
from pathlib import Path
from typing import Any

from web.fixtures import _build_fixtures as builder


RUN_IDS = ("BASE", "FINAL", "BASE7500")
POLICY_ID_RE = re.compile(r"^P[0-9]{3,}$")
DAY_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


class StoreError(Exception):
    """An expected data or request error that can be returned as JSON."""

    def __init__(self, status_code: int, message: str, detail: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.detail = detail


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreError(500, f"산출물을 읽지 못했습니다: {path.name}", str(exc)) from exc


def _safe_policy_id(policy_id: str) -> str:
    if not POLICY_ID_RE.fullmatch(policy_id):
        raise StoreError(400, "정책 ID 형식이 올바르지 않습니다")
    return policy_id


def _safe_day(day: str) -> str:
    """Keep day lookups inside the contract's YYYY-MM-DD namespace."""
    if not DAY_RE.fullmatch(day):
        raise StoreError(400, "일자 형식은 YYYY-MM-DD여야 합니다")
    try:
        date.fromisoformat(day)
    except ValueError as exc:
        raise StoreError(400, "존재하지 않는 날짜입니다") from exc
    return day


class ArtifactStore:
    """Repository for actual runs and explicit S1 fixture mode."""

    def __init__(
        self,
        *,
        repo_root: Path,
        data_root: Path,
        policy_dir: Path | None = None,
        output_root: Path | None = None,
        fixture_dir: Path | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.data_root = data_root.resolve()
        self.policy_dir = (policy_dir or self.repo_root / "data" / "neo4j_load" / "policies").resolve()
        self.output_root = (output_root or self.repo_root / "output" / "sim").resolve()
        self.fixture_dir = fixture_dir.resolve() if fixture_dir else None
        self._cache_lock = threading.Lock()
        self._day_cache: dict[tuple[str, str, int, int], dict] = {}
        self._events_cache: dict[tuple[str, int, int], dict] = {}

        # Keep the S1 reference implementation and S2 on exactly the same
        # source-root rules. This only changes module configuration in memory;
        # it never writes to scripts/sim or scripts/neo4j_load.
        builder.DATA_ROOT = self.data_root
        builder.POLICY_DIR = self.policy_dir
        builder.PREFLIGHT = self.repo_root / "scripts" / "sim" / "policy_preflight.py"
        builder.RUNS = {
            "BASE": (self.data_root / "out_BASE", self.data_root / "logs_scripts" / "run_BASE.log"),
            "FINAL": (self.data_root / "out_FINAL", self.data_root / "logs_scripts" / "run_FINAL.log"),
            "BASE7500": (
                self.data_root / "rescue" / "out_BASE7500",
                self.data_root / "logs_scripts" / "run_BASE7500.log",
            ),
        }

    @classmethod
    def from_environment(cls, repo_root: Path) -> "ArtifactStore":
        data_root = Path(os.environ.get("SIM_DATA_ROOT", r"C:\Users\srdyh\gpu_exp_data\20260802"))
        fixture_dir = None
        # Explicit fixture mode is for local contract/API tests only. It is
        # never an implicit fallback when real data is unavailable.
        if os.environ.get("SIM_USE_FIXTURES", "").lower() in {"1", "true", "yes"}:
            fixture_dir = Path(os.environ.get("SIM_FIXTURES_DIR", str(repo_root / "web" / "fixtures")))
        return cls(repo_root=repo_root, data_root=data_root, fixture_dir=fixture_dir)

    def _require_run(self, run_id: str) -> tuple[Path, Path]:
        if run_id not in RUN_IDS:
            raise StoreError(404, f"알 수 없는 run: {run_id}")
        root, log = builder.RUNS[run_id]
        if self.fixture_dir is None and not root.exists():
            raise StoreError(503, f"실제 run 산출물을 찾을 수 없습니다: {root}")
        return root, log

    def _fixture(self, name: str) -> dict | None:
        if self.fixture_dir is None:
            return None
        path = self.fixture_dir / name
        if not path.exists():
            return None
        payload = _json(path)
        if not isinstance(payload, dict):
            raise StoreError(500, f"픽스처 응답이 객체가 아닙니다: {name}")
        return payload

    def _actual_run(self, run_id: str) -> dict:
        self._require_run(run_id)
        return builder.scan_run(run_id)

    def runs_index(self) -> dict:
        fixture = self._fixture("runs.index.json")
        if fixture is not None:
            return fixture
        runs = {run_id: self._actual_run(run_id) for run_id in RUN_IDS}
        items = []
        for run_id, run in runs.items():
            days = run["days_present"]
            items.append(
                {
                    "run_id": run_id,
                    "root": run["root"],
                    "status": run["status"],
                    "first_day": days[0] if days else None,
                    "last_day": days[-1] if days else None,
                    "days_present": len(days),
                    "days_planned": run["plan"]["planned_days"],
                    "agents_target": run["plan"]["agents_target"],
                    "completed_at": run["completed_at"],
                    "artifacts": run["artifacts"],
                    "unknown": [
                        key
                        for key, value in (
                            ("days_planned", run["plan"]["planned_days"]),
                            ("agents_target", run["plan"]["agents_target"]),
                            ("completed_at", run["completed_at"]),
                        )
                        if value is None
                    ],
                }
            )
        return {"total": len(items), "items": items, "unknown": []}

    def run_detail(self, run_id: str) -> dict:
        fixture = self._fixture(f"run.{run_id}.detail.json")
        if fixture is not None:
            return fixture
        run = self._actual_run(run_id)
        detail = {
            key: run[key]
            for key in (
                "run_id",
                "root",
                "status",
                "artifacts",
                "days_present",
                "days_with_timing",
                "days_with_done_checkpoint",
                "days_with_failed_checkpoint",
                "plan",
                "completed_at",
                "updated_at",
                "log_hint",
            )
        }
        detail["day_summaries"] = [
            {key: value for key, value in summary.items() if key != "timing_top"}
            for summary in ((run.get("summary") or {}).get("summary") or [])
        ]
        detail["unknown"] = [
            key
            for key, value in (
                ("plan.planned_days", run["plan"]["planned_days"]),
                ("plan.agents_target", run["plan"]["agents_target"]),
                ("plan.start_day", run["plan"]["start_day"]),
                ("completed_at", run["completed_at"]),
            )
            if value is None
        ]
        return detail

    def run_days(self, run_id: str) -> dict:
        fixture = self._fixture(f"run.{run_id}.days.json")
        if fixture is not None:
            return fixture
        run = self._actual_run(run_id)
        root = Path(run["root"])
        items = []
        for day in run["days_present"]:
            metrics_path = root / "metrics" / f"day_{day}.jsonl"
            counts = builder.status_scan(metrics_path)
            done = builder.read_json(root / "checkpoints" / f"done_{day}.json")
            failed = builder.read_json(root / "checkpoints" / f"failed_{day}.json")
            timing = builder.read_json(root / "timing" / f"day_{day}.json")
            summary = next(
                (
                    item
                    for item in ((run.get("summary") or {}).get("summary") or [])
                    if item.get("day") == day
                ),
                None,
            )
            target = run["plan"]["agents_target"]
            items.append(
                {
                    "day": day,
                    "agents_ok": counts["agents_ok"],
                    "agents_error": counts["agents_error"],
                    "metrics_rows": counts["rows"],
                    "counts_source": "status_scan",
                    "checkpoint_done_count": len(done) if isinstance(done, list) else None,
                    "checkpoint_failed_count": len(failed) if isinstance(failed, list) else None,
                    "agents_target": target,
                    "progress_ratio": round(counts["agents_ok"] / target, 6) if target else None,
                    "day_complete": bool(summary),
                    "elapsed_sec": (summary or {}).get("elapsed_sec"),
                    "agent_elapsed_sec": (summary or {}).get("agent_elapsed_sec"),
                    "night2_elapsed_sec": (summary or {}).get("night2_elapsed_sec"),
                    "timing_report_present": timing is not None,
                    "policy_payment": (timing or {}).get("policy_payment"),
                    "metrics_bytes": metrics_path.stat().st_size,
                    "unknown": builder._unknown_flags(run, summary, timing, target),
                }
            )
        return {
            "run_id": run_id,
            "total": len(items),
            "items": items,
            "unknown": ["agents_target"] if run["plan"]["agents_target"] is None else [],
        }

    def day_detail(self, run_id: str, day: str) -> dict:
        day = _safe_day(day)
        fixture = self._fixture(f"run.{run_id}.day.{day}.json")
        if fixture is not None:
            return fixture
        root, _ = self._require_run(run_id)
        source = root / "metrics" / f"day_{day}.jsonl"
        if not source.is_file():
            raise StoreError(404, f"일자 산출물을 찾을 수 없습니다: {run_id}/{day}")
        stat = source.stat()
        cache_key = (run_id, day, stat.st_mtime_ns, stat.st_size)
        completed = (root / "timing" / f"day_{day}.json").is_file()
        if completed:
            with self._cache_lock:
                cached = self._day_cache.get(cache_key)
            if cached is not None:
                return cached
        payload = {
            "run_id": run_id,
            "day": day,
            "source_file": f"metrics/day_{day}.jsonl",
            "source_bytes": stat.st_size,
            "aggregated_server_side": True,
            **builder.aggregate_day(source),
        }
        if completed:
            with self._cache_lock:
                self._day_cache[cache_key] = payload
        return payload

    def bottlenecks(self, run_id: str, day: str) -> dict:
        day = _safe_day(day)
        fixture = self._fixture(f"run.{run_id}.day.{day}.bottlenecks.json")
        if fixture is not None:
            return fixture
        run = self._actual_run(run_id)
        root = Path(run["root"])
        source = root / "metrics" / f"day_{day}.jsonl"
        if not source.is_file():
            raise StoreError(404, f"일자 산출물을 찾을 수 없습니다: {run_id}/{day}")
        aggregate = None
        if not (root / "timing" / f"day_{day}.json").is_file():
            aggregate = self.day_detail(run_id, day)
            aggregate = {key: value for key, value in aggregate.items() if key not in {"run_id", "day", "source_file", "source_bytes", "aggregated_server_side"}}
        return builder.bottlenecks(run, day, aggregate)

    def slow(self, run_id: str, day: str, limit: int = 15, offset: int = 0) -> dict:
        day = _safe_day(day)
        limit = max(1, min(limit, 100))
        offset = max(0, offset)
        fixture = self._fixture(f"run.{run_id}.day.{day}.slow.json")
        if fixture is not None:
            if offset:
                fixture = {**fixture, "items": fixture["items"][offset : offset + limit], "limit": limit}
            else:
                fixture = {**fixture, "items": fixture["items"][:limit], "limit": limit}
            return fixture
        run = self._actual_run(run_id)
        payload = builder.slow_page(run, day, limit + offset)
        payload["items"] = payload["items"][offset : offset + limit]
        payload["limit"] = limit
        return payload

    def failed(self, run_id: str, day: str) -> dict:
        day = _safe_day(day)
        fixture = self._fixture(f"run.{run_id}.day.{day}.failed.json")
        if fixture is not None:
            return fixture
        run = self._actual_run(run_id)
        return builder.failed_page(run, day)

    def failures(self, run_id: str, *, day: str | None = None, limit: int = 12) -> dict:
        if day is not None:
            day = _safe_day(day)
        limit = max(1, min(limit, 100))
        fixture = self._fixture(f"run.{run_id}.failures.json")
        if fixture is not None and not day:
            return {**fixture, "limit": limit, "items": fixture.get("items", [])[:limit]}
        run = self._actual_run(run_id)
        payload = builder.failures_page(run, limit)
        if day and payload.get("items") is not None:
            payload["items"] = [item for item in payload["items"] if str(item.get("day")) == day]
        return payload

    def events_summary(self, run_id: str) -> dict:
        fixture = self._fixture(f"run.{run_id}.events.summary.json")
        if fixture is not None:
            return fixture
        run = self._actual_run(run_id)
        path = Path(run["root"]) / "events.jsonl"
        if path.is_file():
            stat = path.stat()
            key = (run_id, stat.st_mtime_ns, stat.st_size)
            with self._cache_lock:
                cached = self._events_cache.get(key)
            if cached is not None:
                return cached
            payload = builder.events_summary(run)
            with self._cache_lock:
                self._events_cache[key] = payload
            return payload
        return builder.events_summary(run)

    def policy_index(self) -> dict:
        fixture = self._fixture("policies.index.json")
        if fixture is not None:
            return fixture
        if not self.policy_dir.exists():
            raise StoreError(503, f"정책 디렉터리를 찾을 수 없습니다: {self.policy_dir}")
        return builder.policy_index()

    def policy_detail(self, policy_id: str) -> dict:
        policy_id = _safe_policy_id(policy_id)
        fixture = self._fixture(f"policy.{policy_id}.detail.json")
        if fixture is not None:
            return fixture
        path = self.policy_dir / f"{policy_id}.json"
        policy = _json(path)
        if policy is None:
            raise StoreError(404, f"정책을 찾을 수 없습니다: {policy_id}")
        return {
            "file": path.name,
            "source_dir": "data/neo4j_load/policies",
            "policy": policy,
            "grant_key_effective": builder.effective_grant_key(policy),
            "grant_key_source": "file" if policy.get("grant_key") else "default",
            "unknown": [],
        }

    def validate_policy(self, policy_id: str, payload: dict | None = None) -> dict:
        policy_id = _safe_policy_id(policy_id)
        if payload is None:
            fixture = self._fixture(f"policy.{policy_id}.validate.json")
            if fixture is not None:
                return fixture
            path = self.policy_dir / f"{policy_id}.json"
            if not path.is_file():
                raise StoreError(404, f"정책을 찾을 수 없습니다: {policy_id}")
            return builder.run_preflight(path)

        if not isinstance(payload, dict):
            raise StoreError(422, "정책 본문은 JSON 객체여야 합니다")
        with tempfile.TemporaryDirectory(prefix="sim-policy-") as temp_dir:
            path = Path(temp_dir) / f"{policy_id}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            result = builder.run_preflight(path)
        return result

    def save_policy(self, policy_id: str, payload: dict) -> dict:
        policy_id = _safe_policy_id(policy_id)
        if payload.get("id") not in (None, policy_id):
            raise StoreError(422, "URL의 정책 ID와 JSON의 id가 다릅니다")
        payload = {**payload, "id": policy_id}
        result = self.validate_policy(policy_id, payload)
        if not result.get("ok"):
            raise StoreError(422, "정책 preflight를 통과하지 못했습니다", json.dumps(result, ensure_ascii=False))
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        target = self.policy_dir / f"{policy_id}.json"
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.policy_dir, prefix=f".{policy_id}.", suffix=".tmp", delete=False
        ) as fp:
            temp_path = Path(fp.name)
            json.dump(payload, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        temp_path.replace(target)
        return self.policy_detail(policy_id)

    def delete_policy(self, policy_id: str) -> dict:
        policy_id = _safe_policy_id(policy_id)
        if self.fixture_dir is not None:
            raise StoreError(405, "fixture 모드에서는 정책을 삭제할 수 없습니다")
        target = self.policy_dir / f"{policy_id}.json"
        if not target.is_file():
            raise StoreError(404, f"정책을 찾을 수 없습니다: {policy_id}")
        target.unlink()
        return {
            "deleted": True,
            "policy_id": policy_id,
            "file": target.name,
            "unknown": [],
        }

    def artifact(self, relative_path: str, *, run_id: str | None = None) -> Path:
        if self.fixture_dir is not None:
            raise StoreError(404, "fixture 모드에는 원본 산출물 서빙이 없습니다")
        if run_id is None:
            root = self.output_root
        else:
            root, _ = self._require_run(run_id)
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise StoreError(400, "산출물 경로가 허용 범위를 벗어났습니다") from exc
        if not candidate.is_file():
            raise StoreError(404, f"산출물을 찾을 수 없습니다: {relative_path}")
        return candidate

    def artifact_index(self) -> dict:
        if self.fixture_dir is not None:
            raise StoreError(404, "fixture 모드에는 원본 산출물 목록이 없습니다")
        if not self.output_root.is_dir():
            raise StoreError(503, f"시각화 산출물 디렉터리를 찾을 수 없습니다: {self.output_root}")
        items = []
        for path in sorted(self.output_root.rglob("*.html")):
            if path.is_file():
                items.append({"path": path.relative_to(self.output_root).as_posix(), "bytes": path.stat().st_size})
        return {"total": len(items), "items": items, "unknown": []}
