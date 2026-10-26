"""실험 레지스트리와 실제 실행 데이터(로컬 미러) — 콘솔이 여러 실험을 한곳에서 다룬다.

무엇을 읽는가
-------------
* 실험 정의: ``web/experiments/*.json``(저장소) + ``<SIM_EXPERIMENTS_ROOT>/<id>/experiment.json``(이 PC 에만 있는 실험)
* 실행 데이터: 실험 서버에서 **읽기 전용으로 받아 온 미러**(``experiment_sync.py``).
  콘솔은 실험 서버에 아무것도 쓰지 않는다.

지키는 규칙
-----------
* metrics 한 줄마다 봉인(``integrity_sha256``)을 검사한다. 깨진 줄은 집계에서 빼고 그 수를 보고한다.
* 지금 쓰고 있는 날짜 파일의 마지막 줄이 덜 써졌으면 그 줄은 다음 동기화까지 미룬다.
* 건너뛴 대상자-일(skipped)은 결측이다. 0으로 세지 않는다.
* 하루 집계는 파일 크기·수정 시각이 같으면 다시 계산하지 않는다(메모리 + ``_console`` 디스크 캐시).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import statistics
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from scripts.sim.evidence_integrity import EvidenceError, verify

from .store import StoreError

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO_ROOT / "web" / "experiments"
EXPERIMENT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
ARM_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# 대상자 id 에는 연령대 한글이 들어간다 (AGT_11350560_M_30대_003)
AGENT_ID = re.compile(r"^[A-Za-z0-9_\-가-힣]{1,96}$")
ROLES = {"single", "shared_pre", "control", "treatment"}
CACHE_VERSION = 2  # 2: 정책 실험(policy_paired)은 대상자별 업종 금액·정책 지갑 사용액도 요약한다
CONSOLE_DIR = "_console"


def experiments_root() -> Path:
    configured = os.environ.get("SIM_EXPERIMENTS_ROOT", "").strip()
    if configured:
        return Path(configured)
    data_root = os.environ.get("SIM_DATA_ROOT", "").strip()
    if data_root:
        return Path(data_root) / "experiments"
    return Path.home() / "sim_data" / "experiments"


@dataclass(frozen=True)
class Arm:
    arm_id: str
    role: str
    label: str
    policy: str
    directory: str
    start: str
    days: int
    parent: str | None = None

    def day_list(self) -> list[str]:
        first = date.fromisoformat(self.start)
        return [(first + timedelta(days=offset)).isoformat() for offset in range(self.days)]

    def public(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "role": self.role,
            "label": self.label,
            "policy": self.policy,
            "start": self.start,
            "days": self.days,
            "parent": self.parent,
        }


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    title: str
    kind: str
    summary: str
    interpretation: str
    window: dict[str, Any]
    cohort_size: int
    arms: tuple[Arm, ...]
    analysis: dict[str, Any] = field(default_factory=dict)
    bundle: dict[str, Any] = field(default_factory=dict)
    remote: dict[str, Any] = field(default_factory=dict)
    source_branch: str | None = None
    origin: str = "repo"
    # 시민 문답에서 누를 질문 예시, 실측 연구와의 비교(평가_통합성능에서 옮김)
    questions: tuple[str, ...] = ()
    # 정책 있는 쪽 시민이 시행 뒤 알고 있는 정책 안내 (시민 문답 프롬프트에만 쓴다)
    interview_context: str = ""
    validation: dict[str, Any] = field(default_factory=dict)

    def arm(self, arm_id: str) -> Arm:
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        raise StoreError(404, f"이 실험에 없는 팔입니다: {arm_id}")

    @property
    def local_dir(self) -> Path:
        return experiments_root() / self.experiment_id

    def arm_dir(self, arm: Arm) -> Path:
        return self.local_dir / "arms" / arm.arm_id

    @property
    def bundle_dir(self) -> Path:
        return self.local_dir / "bundle"

    def lineage(self, arm: Arm) -> list[Arm]:
        """부모(공통 사전) → 이 팔 순서. 분기 실험에서 한 사람의 전 기간을 잇는 데 쓴다."""
        chain = [arm]
        seen = {arm.arm_id}
        while chain[0].parent:
            parent = self.arm(chain[0].parent)
            if parent.arm_id in seen:
                raise StoreError(500, f"팔 계보가 순환합니다: {self.experiment_id}")
            seen.add(parent.arm_id)
            chain.insert(0, parent)
        return chain

    def public(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "kind": self.kind,
            "summary": self.summary,
            "interpretation": self.interpretation,
            "window": self.window,
            "cohort_size": self.cohort_size,
            "arms": [arm.public() for arm in self.arms],
            "analysis": self.analysis,
            "source_branch": self.source_branch,
            "origin": self.origin,
            # 원격 접속 정보는 이 PC 의 환경변수에만 있다. 프로필 이름만 밝힌다.
            "remote_profile": self.remote.get("profile"),
            "questions": list(self.questions),
            "validation": self.validation or None,
        }


def _parse_experiment(raw: dict[str, Any], origin: str) -> Experiment:
    experiment_id = str(raw.get("experiment_id") or "")
    if not EXPERIMENT_ID.match(experiment_id):
        raise ValueError(f"잘못된 실험 id: {experiment_id!r}")
    arms = []
    for item in raw.get("arms") or []:
        arm = Arm(
            arm_id=str(item.get("arm_id") or ""),
            role=str(item.get("role") or "single"),
            label=str(item.get("label") or item.get("arm_id") or ""),
            policy=str(item.get("policy") or ""),
            directory=str(item.get("directory") or ""),
            start=str(item.get("start") or ""),
            days=int(item.get("days") or 0),
            parent=item.get("parent"),
        )
        if not ARM_ID.match(arm.arm_id) or arm.role not in ROLES or not DAY.match(arm.start) or arm.days <= 0:
            raise ValueError(f"잘못된 팔 정의: {experiment_id}/{arm.arm_id}")
        if not arm.directory or "/" in arm.directory or "\\" in arm.directory or arm.directory in {".", ".."}:
            raise ValueError(f"팔 폴더 이름이 잘못됐습니다: {experiment_id}/{arm.arm_id}")
        arms.append(arm)
    if not arms or len({arm.arm_id for arm in arms}) != len(arms):
        raise ValueError(f"팔이 없거나 겹칩니다: {experiment_id}")
    ids = {arm.arm_id for arm in arms}
    if any(arm.parent and arm.parent not in ids for arm in arms):
        raise ValueError(f"부모 팔이 없습니다: {experiment_id}")
    return Experiment(
        experiment_id=experiment_id,
        title=str(raw.get("title") or experiment_id),
        kind=str(raw.get("kind") or "generic"),
        summary=str(raw.get("summary") or ""),
        interpretation=str(raw.get("interpretation") or ""),
        window=dict(raw.get("window") or {}),
        cohort_size=int(raw.get("cohort_size") or 0),
        arms=tuple(arms),
        analysis=dict(raw.get("analysis") or {}),
        bundle=dict(raw.get("bundle") or {}),
        remote=dict(raw.get("remote") or {}),
        source_branch=raw.get("source_branch"),
        origin=origin,
        questions=tuple(str(q) for q in raw.get("questions") or [] if str(q).strip()),
        interview_context=str(raw.get("interview_context") or "").strip(),
        validation=dict(raw.get("validation") or {}),
    )


def load_registry(registry_dir: Path | None = None) -> tuple[dict[str, Experiment], list[dict[str, str]]]:
    """정의를 모두 읽는다. 잘못된 정의는 건너뛰고 그 이유를 함께 돌려준다."""
    found: dict[str, Experiment] = {}
    problems: list[dict[str, str]] = []
    sources: list[tuple[Path, str]] = [(path, "repo") for path in sorted((registry_dir or REGISTRY_DIR).glob("*.json"))]
    root = experiments_root()
    if root.is_dir():
        sources += [(path, "local") for path in sorted(root.glob("*/experiment.json"))]
    for path, origin in sources:
        try:
            experiment = _parse_experiment(json.loads(path.read_text(encoding="utf-8")), origin)
        except (OSError, ValueError, TypeError) as exc:
            problems.append({"file": path.name, "error": str(exc)})
            continue
        if experiment.experiment_id in found:
            # 저장소 정의가 이긴다. 같은 id 를 이 PC 에서 몰래 바꿔치기하지 못하게 한다.
            problems.append({"file": path.name, "error": f"이미 있는 실험 id: {experiment.experiment_id}"})
            continue
        found[experiment.experiment_id] = experiment
    return found, problems


# ---------------------------------------------------------------------------
# metrics 한 날짜 파일 → 봉인 검사 → 하루 집계(+ 대상자별 요약, 줄 위치 색인)
# ---------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _int(value: Any) -> int:
    number = _number(value)
    return int(round(number)) if number is not None else 0


def read_day_rows(path: Path) -> dict[str, Any]:
    """한 날짜 파일을 줄 단위로 읽고 봉인을 검사한다.

    같은 대상자의 같은 줄이 두 번 있으면 하나로 본다. 내용이 다르면 충돌로 세고 그 대상자-일을 뺀다.
    """
    data = path.read_bytes()
    rows: dict[str, dict[str, Any]] = {}
    offsets: dict[str, list[int]] = {}
    conflicts: set[str] = set()
    failures = 0
    position = 0
    partial = 0
    while position < len(data):
        newline = data.find(b"\n", position)
        if newline < 0:
            # 시뮬레이터가 아직 쓰는 중인 마지막 줄 — 다음 동기화 때 완성된다
            partial = len(data) - position
            break
        line = data[position:newline]
        if line.strip():
            try:
                row = verify(json.loads(line))
                aid = row["aid"]
                if not isinstance(aid, str):
                    raise EvidenceError("aid missing")
            except (ValueError, KeyError, EvidenceError):
                failures += 1
            else:
                previous = rows.get(aid)
                if previous is None:
                    rows[aid] = row
                    offsets[aid] = [position, newline - position]
                elif previous.get("integrity_sha256") != row.get("integrity_sha256"):
                    conflicts.add(aid)
        position = newline + 1
    for aid in conflicts:
        rows.pop(aid, None)
        offsets.pop(aid, None)
    return {
        "rows": rows,
        "offsets": offsets,
        "integrity_failures": failures,
        "conflicting_agents": sorted(conflicts),
        "partial_tail_bytes": partial,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def summarize_rows(rows: Iterable[dict[str, Any]], experiment: Experiment) -> dict[str, Any]:
    """하루치 집계와 대상자별 압축 요약.

    대상자별 요약(``agents``)은 집단 통계와 짝 비교에 쓴다. 원문 줄은 색인으로 다시 찾는다.
    """
    analysis = experiment.analysis or {}
    aliases = analysis.get("facility_aliases") or {}
    status: Counter[str] = Counter()
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    totals = defaultdict(float)
    means: dict[str, list[float]] = defaultdict(list)
    agents: dict[str, dict[str, Any]] = {}
    target_total = {"revenue_krw": 0, "payment_count": 0}
    target_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"revenue_krw": 0, "payment_count": 0})
    for row in rows:
        aid = row["aid"]
        state = str(row.get("status") or "unknown")
        status[state] += 1
        no_smoking = row.get("no_smoking") if isinstance(row.get("no_smoking"), dict) else {}
        compact: dict[str, Any] = {"s": state}
        if no_smoking.get("smoking_status"):
            compact["smoking"] = no_smoking["smoking_status"]
        if state != "ok":
            agents[aid] = compact
            continue
        purchased = [
            receipt
            for receipt in row.get("execution_receipts") or []
            if isinstance(receipt, dict) and receipt.get("kind") == "purchase_receipt"
            and receipt.get("purchase_status") == "purchased"
        ]
        amount = 0
        for receipt in purchased:
            value = _int(receipt.get("amount"))
            amount += value
            bucket = by_category[str(receipt.get("category") or "기타")]
            bucket[0] += 1
            bucket[1] += value
        spend = _number(row.get("cm_today_total"))
        spend_all = _number(row.get("cm_today_total_incl_online"))
        totals["receipt_count"] += len(purchased)
        totals["receipt_amount"] += amount
        if spend is not None:
            totals["spend_offline"] += spend
        if spend_all is not None:
            totals["spend_incl_online"] += spend_all
        for key in ("tokens_in", "tokens_out"):
            totals[key] += _int(row.get(key))
        for key in ("mood", "fatigue", "avg_sat", "elapsed", "balance"):
            value = _number(row.get(key))
            if value is not None:
                means[key].append(value)
        compact.update(
            {
                "n": len(purchased),
                "amt": amount,
                "spend": spend,
                "spend_all": spend_all,
                "mood": _number(row.get("mood")),
                "fatigue": _number(row.get("fatigue")),
                "sat": _number(row.get("avg_sat")),
                "balance": _number(row.get("balance")),
            }
        )
        if experiment.kind == "policy_paired":
            # 짝 비교(같은 시민·같은 날, 정책 있음 − 없음)를 업종 단위로도 하려면 대상자별 업종 금액이 필요하다.
            # 정책 지갑 몫은 영수증 금액에서 본인 부담을 뺀 값이다
            categories: dict[str, int] = defaultdict(int)
            wallet = 0
            for receipt in purchased:
                value = _int(receipt.get("amount"))
                categories[str(receipt.get("category") or "기타")] += value
                own = receipt.get("own_paid")
                if own is not None:
                    wallet += max(0, value - _int(own))
            compact["cat"] = dict(categories)
            compact["pol"] = wallet
        if experiment.kind == "no_smoking_zone":
            hits = []
            smoking = compact.get("smoking") or "unknown"
            for poi in no_smoking.get("by_poi") or []:
                if not isinstance(poi, dict):
                    continue
                facility = aliases.get(poi.get("facility_type"), poi.get("facility_type")) or "unknown"
                district = str(poi.get("district_code") or "")
                revenue = _int(poi.get("revenue_krw"))
                count = _int(poi.get("payment_count"))
                hits.append([facility, district, revenue, count])
                target_total["revenue_krw"] += revenue
                target_total["payment_count"] += count
                for group in (facility, f"{facility}:{district}", f"{facility}:{smoking}"):
                    target_groups[group]["revenue_krw"] += revenue
                    target_groups[group]["payment_count"] += count
            if hits:
                compact["t"] = hits
            compact["policy_active"] = bool(no_smoking.get("policy_active"))
        agents[aid] = compact

    summary: dict[str, Any] = {
        "status": dict(status),
        "unique_agents": len(agents),
        "totals": {key: round(value, 2) for key, value in totals.items()},
        "means": {key: round(statistics.fmean(values), 4) for key, values in means.items() if values},
        "by_category": {
            category: {"count": count, "amount": amount}
            for category, (count, amount) in sorted(by_category.items(), key=lambda item: -item[1][1])
        },
        "agents": agents,
    }
    if experiment.kind == "no_smoking_zone":
        summary["target"] = {"total": target_total, "groups": dict(sorted(target_groups.items()))}
    return summary


class DayCache:
    """하루 집계 캐시. 파일 크기·수정 시각이 같으면 메모리 → 디스크 순으로 재사용한다."""

    def __init__(self) -> None:
        self._memory: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, experiment: Experiment, arm: Arm, day: str) -> dict[str, Any] | None:
        path = experiment.arm_dir(arm) / "metrics" / f"day_{day}.jsonl"
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        with self._lock:
            cached = self._memory.get(key)
        if cached is not None:
            return cached
        disk = experiment.arm_dir(arm) / CONSOLE_DIR / f"day_{day}.json"
        loaded = self._read_disk(disk, stat)
        if loaded is None:
            parsed = read_day_rows(path)
            loaded = {
                "version": CACHE_VERSION,
                "day": day,
                "source": {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": parsed["sha256"]},
                "integrity_failures": parsed["integrity_failures"],
                "conflicting_agents": parsed["conflicting_agents"],
                "partial_tail_bytes": parsed["partial_tail_bytes"],
                "offsets": parsed["offsets"],
                "summary": summarize_rows(parsed["rows"].values(), experiment),
            }
            self._write_disk(disk, loaded)
        with self._lock:
            # 같은 파일의 예전 판은 버린다 (진행 중인 날은 동기화마다 커진다)
            for old in [k for k in self._memory if k[0] == key[0]]:
                del self._memory[old]
            self._memory[key] = loaded
        return loaded

    @staticmethod
    def _read_disk(path: Path, stat: os.stat_result) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        source = value.get("source") or {}
        if (
            value.get("version") != CACHE_VERSION
            or source.get("bytes") != stat.st_size
            or source.get("mtime_ns") != stat.st_mtime_ns
        ):
            return None
        return value

    @staticmethod
    def _write_disk(path: Path, value: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + f".tmp{os.getpid()}")
            temp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            os.replace(temp, path)
        except OSError:
            pass  # 캐시를 못 써도 집계는 그대로 돌려준다


def _json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class ExperimentStore:
    def __init__(self, registry_dir: Path | None = None) -> None:
        self.registry_dir = registry_dir
        self.cache = DayCache()
        self._roster_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    # -- 정의 -----------------------------------------------------------------

    def registry(self) -> tuple[dict[str, Experiment], list[dict[str, str]]]:
        # 매 요청 다시 읽는다. 새 실험 정의를 넣으면 서버를 다시 띄우지 않아도 보인다.
        return load_registry(self.registry_dir)

    def experiment(self, experiment_id: str) -> Experiment:
        if not EXPERIMENT_ID.match(experiment_id or ""):
            raise StoreError(400, "잘못된 실험 id 입니다")
        experiments, _ = self.registry()
        if experiment_id not in experiments:
            raise StoreError(404, f"등록되지 않은 실험입니다: {experiment_id}")
        return experiments[experiment_id]

    # -- 팔 상태 ---------------------------------------------------------------

    def arm_manifest(self, experiment: Experiment, arm: Arm) -> dict[str, Any] | None:
        manifest = _json_file(experiment.arm_dir(arm) / "experiment_run.json")
        if not isinstance(manifest, dict):
            return None
        cohort = manifest.get("cohort_ids") if isinstance(manifest.get("cohort_ids"), list) else []
        return {
            "status": manifest.get("status"),
            "arm": manifest.get("arm"),
            "phase": manifest.get("phase"),
            "start": manifest.get("start"),
            "days": manifest.get("days"),
            "model": manifest.get("model"),
            "workers": manifest.get("workers"),
            "assignment_seed": manifest.get("assignment_seed"),
            "simulation_seed": manifest.get("simulation_seed"),
            "interpretation": manifest.get("interpretation"),
            "cohort_size": len(cohort),
            "runtime_sha256": manifest.get("runtime_sha256"),
        }

    def arm_status(self, experiment: Experiment, arm: Arm) -> dict[str, Any]:
        directory = experiment.arm_dir(arm)
        manifest = self.arm_manifest(experiment, arm)
        cohort_size = (manifest or {}).get("cohort_size") or experiment.cohort_size
        sync = _json_file(directory / "_sync.json") or {}
        days = []
        for day in arm.day_list():
            entry: dict[str, Any] = {"day": day, "present": False}
            loaded = self.cache.get(experiment, arm, day)
            if loaded is not None:
                summary = loaded["summary"]
                status = summary["status"]
                entry.update(
                    {
                        "present": True,
                        "rows": summary["unique_agents"],
                        "ok": status.get("ok", 0),
                        "skipped": status.get("skipped", 0),
                        "complete": bool(cohort_size) and summary["unique_agents"] >= cohort_size,
                        "integrity_failures": loaded["integrity_failures"],
                        "partial_tail": loaded["partial_tail_bytes"] > 0,
                    }
                )
            entry["night_done"] = (directory / f"night2_completed_{day}.json").is_file()
            entry["backup_done"] = (directory / f"backup_completed_{day}.json").is_file()
            days.append(entry)
        present = [day for day in days if day["present"]]
        complete = [day for day in present if day.get("complete")]
        if not directory.is_dir():
            phase = "not_synced"
        elif not present:
            phase = "not_started"
        elif len(complete) == arm.days:
            phase = "complete"
        else:
            phase = "running" if (manifest or {}).get("status") != "complete" else "complete"
        current = next((day for day in reversed(present) if not day.get("complete")), None)
        return {
            **arm.public(),
            "phase": phase,
            "manifest": manifest,
            "days_present": len(present),
            "days_complete": len(complete),
            "current_day": current,
            "skipped_total": sum(day.get("skipped", 0) for day in present),
            "integrity_failures": sum(day.get("integrity_failures", 0) for day in present),
            "day_status": days,
            "synced_at": sync.get("synced_at"),
        }

    def list(self) -> dict[str, Any]:
        experiments, problems = self.registry()
        items = []
        for experiment in experiments.values():
            arms = [self.arm_status(experiment, arm) for arm in experiment.arms]
            items.append(
                {
                    **experiment.public(),
                    "arms": [{key: value for key, value in arm.items() if key != "day_status"} for arm in arms],
                    "mirror": experiment.local_dir.is_dir(),
                }
            )
        unknown = [] if experiments_root().is_dir() else ["experiments_root"]
        return {"total": len(items), "items": items, "problems": problems, "unknown": unknown}

    def detail(self, experiment_id: str) -> dict[str, Any]:
        experiment = self.experiment(experiment_id)
        return {
            **experiment.public(),
            "arms": [self.arm_status(experiment, arm) for arm in experiment.arms],
            "mirror": experiment.local_dir.is_dir(),
            "bundle_present": (experiment.bundle_dir / "personas.json").is_file(),
        }

    # -- 일별 시계열 ------------------------------------------------------------

    def daily(self, experiment_id: str) -> dict[str, Any]:
        experiment = self.experiment(experiment_id)
        series = []
        for arm in experiment.arms:
            points = []
            for day in arm.day_list():
                loaded = self.cache.get(experiment, arm, day)
                if loaded is None:
                    continue
                summary = {key: value for key, value in loaded["summary"].items() if key != "agents"}
                points.append(
                    {
                        "day": day,
                        "partial_tail": loaded["partial_tail_bytes"] > 0,
                        "integrity_failures": loaded["integrity_failures"],
                        **summary,
                    }
                )
            series.append({"arm_id": arm.arm_id, "label": arm.label, "role": arm.role, "points": points})
        return {"experiment_id": experiment_id, "kind": experiment.kind, "series": series}

    # -- 대상자 ---------------------------------------------------------------

    def roster(self, experiment_id: str) -> dict[str, Any]:
        experiment = self.experiment(experiment_id)
        personas_path = experiment.bundle_dir / "personas.json"
        runtime_path = experiment.bundle_dir / "runtime.json"
        stamp = tuple(
            (path.stat().st_size, path.stat().st_mtime_ns) if path.is_file() else None
            for path in (personas_path, runtime_path)
        )
        cached = self._roster_cache.get(experiment_id)
        if cached and cached[0] == stamp and stamp != (None, None):
            return cached[1]
        smoking: dict[str, str] = {}
        runtime = _json_file(runtime_path)
        if isinstance(runtime, dict):
            for row in runtime.get("cohort") or []:
                if isinstance(row, dict) and row.get("id"):
                    smoking[str(row["id"])] = str(row.get("smoking_status") or "unknown")
        agents: list[dict[str, Any]] = []
        personas = _json_file(personas_path)
        if isinstance(personas, list):
            for persona in personas:
                if not isinstance(persona, dict) or not persona.get("agent_id"):
                    continue
                residence = persona.get("residence") or {}
                personal = persona.get("personal") or {}
                spending = persona.get("spending") or {}
                workplace = persona.get("workplace") or {}
                personality = persona.get("personality") or {}
                aid = str(persona["agent_id"])
                agents.append(
                    {
                        "id": aid,
                        "gu": residence.get("gu"),
                        "dong": residence.get("dong"),
                        "district_code": str(residence.get("dong_code") or "")[:5] or None,
                        "age": personal.get("age"),
                        "age_group": personal.get("age_group"),
                        "sex": personal.get("gender"),
                        "job": personal.get("job"),
                        "income_level": personal.get("income_level"),
                        "life_stage": personal.get("life_stage"),
                        "daily_spending_weekday": spending.get("daily_spending_weekday"),
                        "smoking_status": smoking.get(aid),
                        # 인터뷰에서 그 사람답게 말하는 데 쓴다
                        "work_dong": workplace.get("dong"),
                        "commute_min": workplace.get("commute_min"),
                        "spending_tendency": personality.get("spending_tendency"),
                        "lifestyle": personality.get("lifestyle"),
                    }
                )
            source = "bundle"
        else:
            agents = self._roster_from_metrics(experiment)
            source = "metrics"
        result = {"experiment_id": experiment_id, "source": source, "total": len(agents), "items": agents}
        if source == "bundle":
            self._roster_cache[experiment_id] = (stamp, result)
        return result

    def _roster_from_metrics(self, experiment: Experiment) -> list[dict[str, Any]]:
        """번들을 아직 받지 않았을 때: id 와 metrics 에서 알 수 있는 것만 쓴다(지어내지 않는다)."""
        seen: dict[str, dict[str, Any]] = {}
        for arm in experiment.arms:
            for day in arm.day_list():
                loaded = self.cache.get(experiment, arm, day)
                if loaded is None:
                    continue
                for aid, compact in loaded["summary"]["agents"].items():
                    if aid in seen:
                        continue
                    parts = aid.split("_")
                    seen[aid] = {
                        "id": aid,
                        "district_code": parts[1][:5] if len(parts) > 1 and parts[1][:5].isdigit() else None,
                        "sex": parts[2] if len(parts) > 2 and parts[2] in {"M", "F"} else None,
                        "age_group": parts[3] if len(parts) > 3 else None,
                        "smoking_status": compact.get("smoking"),
                    }
                break  # 하루치면 명단 전체가 있다
            if seen:
                break
        districts = (experiment.analysis or {}).get("districts") or {}
        for agent in seen.values():
            agent["gu"] = districts.get(agent.get("district_code") or "")
        return sorted(seen.values(), key=lambda item: item["id"])

    def agent_row(self, experiment: Experiment, arm: Arm, day: str, agent_id: str) -> dict[str, Any] | None:
        """색인으로 한 줄만 다시 읽고 봉인을 확인한다."""
        loaded = self.cache.get(experiment, arm, day)
        if loaded is None:
            return None
        location = loaded["offsets"].get(agent_id)
        if not location:
            return None
        path = experiment.arm_dir(arm) / "metrics" / f"day_{day}.jsonl"
        with path.open("rb") as handle:
            handle.seek(location[0])
            line = handle.read(location[1])
        try:
            return verify(json.loads(line))
        except (ValueError, EvidenceError):
            return None

    def agent(self, experiment_id: str, agent_id: str) -> dict[str, Any]:
        experiment = self.experiment(experiment_id)
        if not AGENT_ID.match(agent_id or ""):
            raise StoreError(400, "잘못된 대상자 id 입니다")
        roster = {item["id"]: item for item in self.roster(experiment_id)["items"]}
        if roster and agent_id not in roster:
            raise StoreError(404, f"이 실험의 대상자가 아닙니다: {agent_id}")
        arms = []
        for arm in experiment.arms:
            days = []
            for day in arm.day_list():
                row = self.agent_row(experiment, arm, day, agent_id)
                if row is None:
                    continue
                days.append(_agent_day_view(row, experiment))
            arms.append({"arm_id": arm.arm_id, "label": arm.label, "role": arm.role, "days": days})
        return {"experiment_id": experiment_id, "agent": roster.get(agent_id) or {"id": agent_id}, "arms": arms}

    # -- 비교 -----------------------------------------------------------------

    def comparison(self, experiment_id: str) -> dict[str, Any]:
        experiment = self.experiment(experiment_id)
        if experiment.kind == "policy_paired":
            return paired_policy_comparison(self, experiment)
        if experiment.kind != "no_smoking_zone":
            return {"experiment_id": experiment_id, "available": False, "reason": "이 실험 종류의 비교 규칙이 아직 없습니다"}
        return no_smoking_comparison(self, experiment)


def _agent_day_view(row: dict[str, Any], experiment: Experiment) -> dict[str, Any]:
    receipts = [
        {
            "time": receipt.get("scheduled_time"),
            "category": receipt.get("category"),
            "amount": _int(receipt.get("amount")),
            "own_paid": _int(receipt.get("own_paid")),
            "status": receipt.get("purchase_status"),
            "poi_id": receipt.get("poi_id"),
            "event_id": receipt.get("event_id"),
        }
        for receipt in row.get("execution_receipts") or []
        if isinstance(receipt, dict) and receipt.get("kind") == "purchase_receipt"
    ]
    no_smoking = row.get("no_smoking") if isinstance(row.get("no_smoking"), dict) else {}
    view: dict[str, Any] = {
        "day": row.get("experience_day"),
        "status": row.get("status"),
        "receipts": receipts,
        "spend": _number(row.get("cm_today_total")),
        "spend_incl_online": _number(row.get("cm_today_total_incl_online")),
        "mood": _number(row.get("mood")),
        "fatigue": _number(row.get("fatigue")),
        "avg_sat": _number(row.get("avg_sat")),
        "balance": _number(row.get("balance")),
        "evidence": bool(row.get("interview_evidence")),
    }
    if row.get("status") == "skipped":
        view["skip_kind"] = row.get("skip_kind")
        view["attempts"] = row.get("attempts")
    if experiment.kind == "no_smoking_zone":
        aliases = (experiment.analysis or {}).get("facility_aliases") or {}
        view["no_smoking"] = {
            "policy_active": no_smoking.get("policy_active"),
            "smoking_status": no_smoking.get("smoking_status"),
            "payment_count": no_smoking.get("payment_count"),
            "revenue_krw": no_smoking.get("revenue_krw"),
            "by_poi": [
                {
                    "poi_id": poi.get("poi_id"),
                    "facility": aliases.get(poi.get("facility_type"), poi.get("facility_type")),
                    "district_code": poi.get("district_code"),
                    "revenue_krw": poi.get("revenue_krw"),
                    "payment_count": poi.get("payment_count"),
                }
                for poi in no_smoking.get("by_poi") or []
                if isinstance(poi, dict)
            ],
        }
    return view


def no_smoking_comparison(store: ExperimentStore, experiment: Experiment) -> dict[str, Any]:
    """`scripts/experiments/no_smoking_zone.py` 의 점수 규칙을 그대로 따른 미리보기.

    * 시설 분류: screen_golf → indoor_golf
    * 묶음: 시설, 시설:구, 시설:흡연여부 / 지표: revenue_krw, payment_count
    * 시행 전(pre)은 공통 사전 팔에서, 시행 후(post)는 두 팔 모두 ok 인 대상자-일만
    * 건너뛴 대상자-일은 결측 — 0 으로 세지 않는다
    공식 점수는 두 팔이 끝난 뒤 실험 코드의 ``score`` 로 계산한다. 이 값은 그것을 대신하지 않는다.
    """
    roles = {arm.role: arm for arm in experiment.arms}
    control, treatment, shared = roles.get("control"), roles.get("treatment"), roles.get("shared_pre")
    facilities = list(((experiment.analysis or {}).get("facilities") or {}).keys())

    def empty() -> dict[str, dict[str, int]]:
        return defaultdict(lambda: {"revenue_krw": 0, "payment_count": 0})

    def add(groups: dict[str, dict[str, int]], compact: dict[str, Any]) -> None:
        smoking = compact.get("smoking") or "unknown"
        for facility, district, revenue, count in compact.get("t") or []:
            for group in (facility, f"{facility}:{district}", f"{facility}:{smoking}"):
                groups[group]["revenue_krw"] += revenue
                groups[group]["payment_count"] += count

    pre_totals = empty()
    pre_days = 0
    if shared:
        for day in shared.day_list():
            loaded = store.cache.get(experiment, shared, day)
            if loaded is None:
                continue
            pre_days += 1
            for compact in loaded["summary"]["agents"].values():
                if compact.get("s") == "ok":
                    add(pre_totals, compact)

    post = {"off": empty(), "on": empty()}
    observed = excluded = 0
    common_days: list[str] = []
    daily: list[dict[str, Any]] = []
    if control and treatment:
        for day in control.day_list():
            off_day = store.cache.get(experiment, control, day)
            on_day = store.cache.get(experiment, treatment, day)
            if off_day is None or on_day is None:
                continue
            common_days.append(day)
            off_agents, on_agents = off_day["summary"]["agents"], on_day["summary"]["agents"]
            day_totals = {"off": {"revenue_krw": 0, "payment_count": 0}, "on": {"revenue_krw": 0, "payment_count": 0}}
            for aid in set(off_agents) | set(on_agents):
                a, b = off_agents.get(aid), on_agents.get(aid)
                if not a or not b or a.get("s") != "ok" or b.get("s") != "ok":
                    excluded += 1
                    continue
                observed += 1
                add(post["off"], a)
                add(post["on"], b)
                for arm_key, compact in (("off", a), ("on", b)):
                    for _facility, _district, revenue, count in compact.get("t") or []:
                        day_totals[arm_key]["revenue_krw"] += revenue
                        day_totals[arm_key]["payment_count"] += count
            daily.append({"day": day, **day_totals})

    comparisons = []
    for group in sorted(set(post["off"]) | set(post["on"]) | set(facilities)):
        for metric in ("revenue_krw", "payment_count"):
            off_value = post["off"][group][metric]
            on_value = post["on"][group][metric]
            comparisons.append(
                {
                    "group": group,
                    "metric": metric,
                    "off": off_value,
                    "on": on_value,
                    "on_minus_off": on_value - off_value,
                    "percent_change": round(100 * (on_value - off_value) / off_value, 3) if off_value else None,
                    "status": "estimable" if off_value else "zero_baseline_not_estimable",
                }
            )
    complete = bool(control and treatment) and len(common_days) == control.days and all(
        (store.cache.get(experiment, arm, day) or {}).get("summary", {}).get("unique_agents", 0) >= experiment.cohort_size
        for arm in (control, treatment)
        for day in arm.day_list()
    )
    return {
        "experiment_id": experiment.experiment_id,
        "available": bool(common_days),
        "status": "complete_preview" if complete else "interim_preview",
        "reason": None if common_days else "두 팔이 함께 끝낸 시행 후 날짜가 아직 없습니다",
        "rules": "no_smoking_zone.py score 와 같은 묶음·결측 규칙 (공식 점수 아님)",
        "pre": {"days": pre_days, "totals": dict(sorted(pre_totals.items()))},
        "post": {
            "days": common_days,
            "observed_paired_agent_days": observed,
            "excluded_agent_days": excluded,
            "totals": {arm: dict(sorted(groups.items())) for arm, groups in post.items()},
            "daily": daily,
        },
        "comparisons": comparisons,
        "limitations": (experiment.analysis or {}).get("limitations") or [],
    }


# --- 정책 실험: 같은 시민·같은 날의 정책 있음 − 없음 -------------------------------

PAIRED_BOOTSTRAP = 500
PAIRED_SEED = 20261022
_paired_cache: dict[str, tuple[tuple, dict[str, Any]]] = {}
_paired_lock = threading.Lock()


def _ratio(on: float, off: float) -> float | None:
    return on / off - 1 if off > 0 else None


def paired_policy_comparison(store: ExperimentStore, experiment: Experiment) -> dict[str, Any]:
    """정책 있음 팔과 없음 팔을 **같은 시민·같은 날** 로 짝지어 비교한다.

    두 팔 모두 정상(ok)으로 기록된 대상자-일만 쓴다. 한쪽이라도 건너뛰었으면 그 짝은 뺀다
    (0 으로 세지 않는다). 효과 = (정책 있음 합 ÷ 정책 없음 합) − 1. 95% 신뢰구간은 시민 단위
    퍼센타일 부트스트랩(같은 수의 시민을 복원 추출, PAIRED_BOOTSTRAP 회, 시드 고정)이다. 한 사람의
    여러 날은 함께 뽑힌다 — 같은 사람의 날들은 서로 독립이 아니다.
    """
    control = next((arm for arm in experiment.arms if arm.role == "control"), None)
    treatment = next((arm for arm in experiment.arms if arm.role == "treatment"), None)
    base = {"experiment_id": experiment.experiment_id, "kind": "policy_paired"}
    if control is None or treatment is None:
        return {**base, "available": False, "reason": "정책 있음·없음 팔이 정의되지 않았습니다"}
    treatment_days = set(treatment.day_list())
    days = [day for day in control.day_list() if day in treatment_days]

    loaded: list[tuple[str, dict, dict]] = []
    signature: list[Any] = []
    for day in days:
        off = store.cache.get(experiment, control, day)
        on = store.cache.get(experiment, treatment, day)
        if off is None or on is None:
            continue
        loaded.append((day, off["summary"]["agents"], on["summary"]["agents"]))
        signature.append((day, (off.get("source") or {}).get("sha256"), (on.get("source") or {}).get("sha256")))
    if not loaded:
        return {**base, "available": False, "reason": "두 팔이 함께 기록한 날이 아직 없습니다"}
    key = tuple(signature)
    with _paired_lock:
        hit = _paired_cache.get(experiment.experiment_id)
    if hit is not None and hit[0] == key:
        return hit[1]

    # 대상자별로 기간 합을 모은다
    people: dict[str, dict[str, Any]] = {}
    daily = []
    excluded = 0
    category_total: Counter[str] = Counter()
    for day, off_agents, on_agents in loaded:
        day_off = day_on = 0.0
        pairs = 0
        for aid in set(off_agents) | set(on_agents):
            a, b = off_agents.get(aid), on_agents.get(aid)
            if not a or not b or a.get("s") != "ok" or b.get("s") != "ok":
                excluded += 1
                continue
            if a.get("spend_all") is None or b.get("spend_all") is None:
                excluded += 1
                continue
            pairs += 1
            person = people.setdefault(
                aid, {"days": 0, "tot": [0.0, 0.0], "shop": [0.0, 0.0], "cat": defaultdict(lambda: [0.0, 0.0]), "pol": 0.0}
            )
            person["days"] += 1
            person["tot"][0] += a["spend_all"]
            person["tot"][1] += b["spend_all"]
            person["shop"][0] += a.get("spend") or 0
            person["shop"][1] += b.get("spend") or 0
            for side, compact in ((0, a), (1, b)):
                for category, amount in (compact.get("cat") or {}).items():
                    person["cat"][category][side] += amount
                    category_total[category] += amount
            person["pol"] += b.get("pol") or 0
            day_off += a["spend_all"]
            day_on += b["spend_all"]
        daily.append({"day": day, "pairs": pairs, "off": round(day_off), "on": round(day_on), "effect": _ratio(day_on, day_off)})

    ids = sorted(people)
    n = len(ids)
    if n == 0:
        return {**base, "available": False, "reason": "두 팔 모두 정상으로 기록된 시민-일이 없습니다"}
    grand = sum(category_total.values()) or 1
    # 업종은 두 팔 합계의 0.5% 이상인 것만 — 몇 건 안 되는 업종의 비율은 흔들려 읽을 수 없다
    categories = [c for c, v in category_total.most_common() if v / grand >= 0.005][:12]
    metrics = [("total", "총소비 (온라인 포함)"), ("shop", "가게 소비 (온라인 제외)")] + [(f"cat:{c}", c) for c in categories]

    def pair(person: dict[str, Any], metric: str) -> tuple[float, float]:
        if metric == "total":
            return person["tot"][0], person["tot"][1]
        if metric == "shop":
            return person["shop"][0], person["shop"][1]
        value = person["cat"].get(metric[4:])
        return (value[0], value[1]) if value else (0.0, 0.0)

    table = [[pair(people[aid], metric) for metric, _ in metrics] for aid in ids]
    width = len(metrics)
    off_sum = [sum(row[k][0] for row in table) for k in range(width)]
    on_sum = [sum(row[k][1] for row in table) for k in range(width)]

    rnd = random.Random(PAIRED_SEED)
    draws: list[list[float]] = [[] for _ in metrics]
    for _ in range(PAIRED_BOOTSTRAP):
        so = [0.0] * width
        sn = [0.0] * width
        for _i in range(n):
            row = table[rnd.randrange(n)]
            for k in range(width):
                so[k] += row[k][0]
                sn[k] += row[k][1]
        for k in range(width):
            effect = _ratio(sn[k], so[k])
            if effect is not None:
                draws[k].append(effect)

    def interval(values: list[float]) -> list[float] | None:
        if len(values) < PAIRED_BOOTSTRAP * 0.9:
            return None
        values = sorted(values)
        return [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]

    agent_days = sum(people[aid]["days"] for aid in ids)
    rows = []
    for k, (metric, label) in enumerate(metrics):
        effect = _ratio(on_sum[k], off_sum[k])
        ci = interval(draws[k])
        rows.append(
            {
                "key": metric,
                "label": label,
                "group": "category" if metric.startswith("cat:") else "total",
                "off": round(off_sum[k]),
                "on": round(on_sum[k]),
                "off_per_agent_day": round(off_sum[k] / agent_days),
                "on_per_agent_day": round(on_sum[k] / agent_days),
                "effect": effect,
                "ci95": ci,
                # 구간이 0 을 포함해도 점추정의 부호는 적는다. 유의 여부만 따로 밝힌다
                "significant": bool(ci and (ci[0] > 0 or ci[1] < 0)),
            }
        )
    wallet = sum(people[aid]["pol"] for aid in ids)
    result = {
        **base,
        "available": True,
        "days": [day for day, _, _ in loaded],
        "agents": n,
        "paired_agent_days": agent_days,
        "excluded_agent_days": excluded,
        "rows": rows,
        "wallet": {"on_total": round(wallet), "per_agent_day": round(wallet / agent_days)},
        "daily": daily,
        "method": {
            "pairing": "같은 시민·같은 날, 두 팔 모두 정상(ok)인 대상자-일",
            "effect": "(정책 있음 합 ÷ 정책 없음 합) − 1",
            "ci": f"시민 단위 퍼센타일 부트스트랩 95% 신뢰구간, {PAIRED_BOOTSTRAP}회, 시드 {PAIRED_SEED}",
        },
        "limitations": (experiment.analysis or {}).get("limitations") or [],
    }
    with _paired_lock:
        _paired_cache[experiment.experiment_id] = (key, result)
    return result
