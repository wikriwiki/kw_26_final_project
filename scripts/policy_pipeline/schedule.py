"""
schedule.py
===========
정책 적재 스케줄 — 시뮬 시작 전 일괄 주입용.

`data/policies/schedule.yaml` 한 파일에 "어떤 정책을 언제부터 언제까지 활성화할지"
를 모아 적는다. 각 항목은 **절대 날짜** (`effective_from: 2026-05-15`) 또는
**시뮬 시작 기준 상대 일수** (`effective_from_day: 5`) 둘 중 하나로 적을 수 있다.

스케줄 적재 흐름:
  1) load_schedule()       → ScheduleEntry 리스트 + sim_start 파싱
  2) resolve_dates()       → 상대 일수를 절대 날짜로 변환
  3) apply_schedule_entry  → 정책 JSON 의 effective_from/until 을 override 한 뒤
                              inject_validated_payload 호출

스케줄은 정책 본질(name/benefit/대상)을 건드리지 않는다. 단지 "언제 켜고 끌지"만 결정.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------
@dataclass
class ScheduleEntry:
    """schedule.yaml 의 policies 리스트 한 항목."""
    file: str                                  # data/neo4j_load/policies/ 기준 상대경로 또는 절대경로
    effective_from: date | None = None         # 절대 날짜 (사용 시)
    effective_until: date | None = None
    effective_from_day: int | None = None      # 상대 일수 (sim_start 기준)
    effective_until_day: int | None = None
    deactivate_others: bool = False            # 적재 시 다른 정책 일괄 비활성화 여부

    # resolve_dates() 후 채워지는 필드
    resolved_from: date | None = None
    resolved_until: date | None = None


@dataclass
class Schedule:
    sim_start: date | None
    entries: list[ScheduleEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 검증 예외
# ---------------------------------------------------------------------------
class ScheduleError(ValueError):
    """스케줄 파일 오류 — 검증 실패 시 raise."""


# ---------------------------------------------------------------------------
# 로더
# ---------------------------------------------------------------------------
def load_schedule(path: Path) -> Schedule:
    """schedule.yaml 을 읽어 Schedule 객체로. 검증 실패 시 ScheduleError."""
    import yaml

    if not path.exists():
        raise ScheduleError(f"schedule file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ScheduleError("schedule root must be a mapping")

    sim_start = _parse_date_or_none(raw.get("sim_start"), field_name="sim_start")

    policies_raw = raw.get("policies") or []
    if not isinstance(policies_raw, list):
        raise ScheduleError("`policies` must be a list")

    entries: list[ScheduleEntry] = []
    for idx, item in enumerate(policies_raw):
        entries.append(_parse_entry(item, idx))

    _validate_entries(entries, sim_start)
    return Schedule(sim_start=sim_start, entries=entries)


def _parse_entry(item: Any, idx: int) -> ScheduleEntry:
    if not isinstance(item, dict):
        raise ScheduleError(f"policies[{idx}] must be a mapping")
    if "file" not in item:
        raise ScheduleError(f"policies[{idx}].file is required")

    entry = ScheduleEntry(
        file=str(item["file"]),
        effective_from=_parse_date_or_none(item.get("effective_from"),
                                           field_name=f"policies[{idx}].effective_from"),
        effective_until=_parse_date_or_none(item.get("effective_until"),
                                            field_name=f"policies[{idx}].effective_until"),
        effective_from_day=_parse_int_or_none(item.get("effective_from_day"),
                                              field_name=f"policies[{idx}].effective_from_day"),
        effective_until_day=_parse_int_or_none(item.get("effective_until_day"),
                                                field_name=f"policies[{idx}].effective_until_day"),
        deactivate_others=bool(item.get("deactivate_others", False)),
    )
    return entry


def _validate_entries(entries: list[ScheduleEntry], sim_start: date | None) -> None:
    needs_sim_start = any(
        (e.effective_from_day is not None or e.effective_until_day is not None)
        for e in entries
    )
    if needs_sim_start and sim_start is None:
        raise ScheduleError(
            "sim_start is required when any entry uses effective_from_day / effective_until_day"
        )

    for idx, e in enumerate(entries):
        # 절대 vs 상대 동시 지정 금지 (from)
        if e.effective_from is not None and e.effective_from_day is not None:
            raise ScheduleError(
                f"policies[{idx}]: specify only one of effective_from / effective_from_day"
            )
        if e.effective_until is not None and e.effective_until_day is not None:
            raise ScheduleError(
                f"policies[{idx}]: specify only one of effective_until / effective_until_day"
            )


# ---------------------------------------------------------------------------
# 날짜 해석
# ---------------------------------------------------------------------------
def resolve_dates(schedule: Schedule) -> Schedule:
    """상대 일수를 절대 날짜로 변환해 resolved_from / resolved_until 채움.

    원본 effective_from / effective_until 가 있으면 그것 우선. 없으면 sim_start + day_offset.
    둘 다 없으면 None (기존 JSON 의 effective_from 을 그대로 쓰겠다는 의미).
    """
    for e in schedule.entries:
        if e.effective_from is not None:
            e.resolved_from = e.effective_from
        elif e.effective_from_day is not None:
            assert schedule.sim_start is not None  # validate 에서 보장
            e.resolved_from = schedule.sim_start + timedelta(days=e.effective_from_day)

        if e.effective_until is not None:
            e.resolved_until = e.effective_until
        elif e.effective_until_day is not None:
            assert schedule.sim_start is not None
            e.resolved_until = schedule.sim_start + timedelta(days=e.effective_until_day)

        if (e.resolved_from is not None and e.resolved_until is not None
                and e.resolved_from > e.resolved_until):
            raise ScheduleError(
                f"{e.file}: resolved effective_from ({e.resolved_from}) "
                f"> effective_until ({e.resolved_until})"
            )
    return schedule


# ---------------------------------------------------------------------------
# 정책 payload override
# ---------------------------------------------------------------------------
def override_payload_dates(payload: dict, entry: ScheduleEntry) -> dict:
    """정책 JSON payload 의 effective_from / effective_until 을 스케줄로 덮어쓴 dict 반환.

    원본 dict 는 건드리지 않는다 (shallow copy). entry.resolved_* 가 None 이면 해당
    필드는 원본 그대로 유지.
    """
    out = dict(payload)
    if entry.resolved_from is not None:
        out["effective_from"] = entry.resolved_from.isoformat()
    if entry.resolved_until is not None:
        out["effective_until"] = entry.resolved_until.isoformat()
    return out


# ---------------------------------------------------------------------------
# 경로 해석 — 정책 파일 base directory 와 합쳐 절대 경로
# ---------------------------------------------------------------------------
def resolve_file_path(file_str: str, base_dir: Path) -> Path:
    p = Path(file_str)
    if p.is_absolute():
        return p
    return base_dir / p


# ---------------------------------------------------------------------------
# 작은 헬퍼
# ---------------------------------------------------------------------------
def _parse_date_or_none(value: Any, *, field_name: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ScheduleError(f"{field_name}: invalid date '{value}' (use YYYY-MM-DD)") from exc
    raise ScheduleError(f"{field_name}: must be a YYYY-MM-DD string or date")


def _parse_int_or_none(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):  # bool 은 int 의 subclass 라 명시 차단
        raise ScheduleError(f"{field_name}: must be an integer, got bool")
    if isinstance(value, int):
        if value < 0:
            raise ScheduleError(f"{field_name}: must be >= 0")
        return value
    raise ScheduleError(f"{field_name}: must be a non-negative integer")
