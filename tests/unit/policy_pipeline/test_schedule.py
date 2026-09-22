"""schedule.yaml 파싱·날짜 해석·payload override 테스트."""

from datetime import date
from pathlib import Path

import pytest

from scripts.policy_pipeline.schedule import (
    ScheduleError,
    load_schedule,
    override_payload_dates,
    resolve_dates,
    resolve_file_path,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "schedule.yaml"
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 로딩 + 검증
# ---------------------------------------------------------------------------
def test_load_with_absolute_dates(tmp_path):
    p = _write(tmp_path, """
policies:
  - file: P007.json
    effective_from: 2026-05-06
    effective_until: 2026-06-30
""")
    sch = load_schedule(p)
    assert sch.sim_start is None
    assert len(sch.entries) == 1
    e = sch.entries[0]
    assert e.file == "P007.json"
    assert e.effective_from == date(2026, 5, 6)
    assert e.effective_until == date(2026, 6, 30)


def test_load_with_relative_days_requires_sim_start(tmp_path):
    p = _write(tmp_path, """
policies:
  - file: P007.json
    effective_from_day: 5
""")
    with pytest.raises(ScheduleError, match="sim_start is required"):
        load_schedule(p)


def test_load_with_sim_start_and_relative(tmp_path):
    p = _write(tmp_path, """
sim_start: 2026-05-01
policies:
  - file: P007.json
    effective_from_day: 5
    effective_until_day: 60
""")
    sch = load_schedule(p)
    assert sch.sim_start == date(2026, 5, 1)
    assert sch.entries[0].effective_from_day == 5
    assert sch.entries[0].effective_until_day == 60


def test_absolute_and_relative_conflict_rejected(tmp_path):
    p = _write(tmp_path, """
sim_start: 2026-05-01
policies:
  - file: P007.json
    effective_from: 2026-05-06
    effective_from_day: 5
""")
    with pytest.raises(ScheduleError, match="only one of"):
        load_schedule(p)


def test_invalid_date_format(tmp_path):
    p = _write(tmp_path, """
policies:
  - file: P007.json
    effective_from: 2026/05/06
""")
    with pytest.raises(ScheduleError, match="invalid date"):
        load_schedule(p)


def test_negative_day_rejected(tmp_path):
    p = _write(tmp_path, """
sim_start: 2026-05-01
policies:
  - file: P007.json
    effective_from_day: -3
""")
    with pytest.raises(ScheduleError, match=">= 0"):
        load_schedule(p)


def test_missing_file_field(tmp_path):
    p = _write(tmp_path, """
policies:
  - effective_from: 2026-05-06
""")
    with pytest.raises(ScheduleError, match="file is required"):
        load_schedule(p)


# ---------------------------------------------------------------------------
# resolve_dates
# ---------------------------------------------------------------------------
def test_resolve_absolute_passthrough(tmp_path):
    p = _write(tmp_path, """
policies:
  - file: P007.json
    effective_from: 2026-05-06
    effective_until: 2026-06-30
""")
    sch = resolve_dates(load_schedule(p))
    e = sch.entries[0]
    assert e.resolved_from == date(2026, 5, 6)
    assert e.resolved_until == date(2026, 6, 30)


def test_resolve_relative_adds_offset(tmp_path):
    p = _write(tmp_path, """
sim_start: 2026-05-01
policies:
  - file: P007.json
    effective_from_day: 5
    effective_until_day: 60
""")
    sch = resolve_dates(load_schedule(p))
    e = sch.entries[0]
    assert e.resolved_from == date(2026, 5, 6)      # 2026-05-01 + 5
    assert e.resolved_until == date(2026, 6, 30)    # 2026-05-01 + 60


def test_resolve_no_dates_leaves_unset(tmp_path):
    """from/until 어느 것도 안 적힌 경우 — 원본 JSON 의 effective_* 를 그대로 쓰겠다."""
    p = _write(tmp_path, """
policies:
  - file: P007.json
""")
    sch = resolve_dates(load_schedule(p))
    e = sch.entries[0]
    assert e.resolved_from is None
    assert e.resolved_until is None


def test_resolve_inverted_dates_rejected(tmp_path):
    p = _write(tmp_path, """
sim_start: 2026-05-01
policies:
  - file: P007.json
    effective_from_day: 60
    effective_until_day: 10
""")
    with pytest.raises(ScheduleError, match="> effective_until"):
        resolve_dates(load_schedule(p))


# ---------------------------------------------------------------------------
# override_payload_dates
# ---------------------------------------------------------------------------
def test_override_payload_replaces_dates(tmp_path):
    p = _write(tmp_path, """
sim_start: 2026-05-01
policies:
  - file: P007.json
    effective_from_day: 5
    effective_until_day: 60
""")
    sch = resolve_dates(load_schedule(p))
    payload = {"id": "P007", "effective_from": "2026-04-01", "effective_until": "2026-04-30"}
    overridden = override_payload_dates(payload, sch.entries[0])

    assert overridden["effective_from"] == "2026-05-06"
    assert overridden["effective_until"] == "2026-06-30"
    # 원본 보존
    assert payload["effective_from"] == "2026-04-01"


def test_override_payload_leaves_unset_keys_alone(tmp_path):
    """resolved_* 가 None 이면 원본 effective_* 가 유지돼야."""
    p = _write(tmp_path, """
policies:
  - file: P007.json
""")
    sch = resolve_dates(load_schedule(p))
    payload = {"id": "P007", "effective_from": "2026-04-01", "effective_until": "2026-04-30"}
    overridden = override_payload_dates(payload, sch.entries[0])
    assert overridden["effective_from"] == "2026-04-01"
    assert overridden["effective_until"] == "2026-04-30"


# ---------------------------------------------------------------------------
# resolve_file_path
# ---------------------------------------------------------------------------
def test_resolve_file_path_relative(tmp_path):
    assert resolve_file_path("P007.json", tmp_path) == tmp_path / "P007.json"


def test_resolve_file_path_absolute(tmp_path):
    abs_path = tmp_path / "x.json"
    assert resolve_file_path(str(abs_path), Path("/other")) == abs_path
