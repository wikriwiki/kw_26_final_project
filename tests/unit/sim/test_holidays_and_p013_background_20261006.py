# -*- coding: utf-8 -*-
"""2026-10-06 — 평일 공휴일(5/5·12/25)은 쉬는 날, P013 기간 배경(4/20~5/5 권고, 5/9 유흥시설 집합금지)."""
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(ROOT))

from kr_holidays import day_type_of, holiday_name, is_day_off  # noqa: E402
import stage1_intent  # noqa: E402
from scripts.sim.environments.registry import build_environment  # noqa: E402


def test_weekday_holidays_are_days_off():
    for d, name in [("2020-05-05", "어린이날"), ("2020-12-25", "성탄절"), ("2021-10-04", "개천절 대체공휴일"),
                    ("2025-06-03", "대통령 선거일")]:
        dd = date.fromisoformat(d)
        assert is_day_off(dd) and day_type_of(dd) == "weekend" and holiday_name(dd) == name


def test_ordinary_days_unchanged():
    d = date(2020, 1, 1)
    while d <= date(2021, 12, 31):
        if holiday_name(d) is None:
            assert day_type_of(d) == ("weekend" if d.weekday() >= 5 else "weekday")
            assert stage1_intent._dow_kr(d) == "월화수목금토일"[d.weekday()]
        d += timedelta(days=1)


def test_prompt_labels():
    assert stage1_intent._dow_kr(date(2020, 5, 5)) == "화, 어린이날 공휴일"
    assert stage1_intent._day_type(date(2020, 5, 5)) == "weekend"
    assert stage1_intent._day_type(date(2020, 5, 4)) == "weekday"


def test_unknown_year_stops_at_launch_not_in_engine():
    from kr_holidays import require_covered
    assert is_day_off(date(2019, 5, 4)) and not is_day_off(date(2019, 5, 6))   # 표 밖: 요일로만
    require_covered(date(2020, 5, 4), date(2020, 5, 17))
    with pytest.raises(SystemExit):
        require_covered(date(2021, 12, 25), date(2022, 1, 7))


def test_p013_background_every_day():
    for i in range(14):
        d = date(2020, 5, 4) + timedelta(days=i)
        env = build_environment("covid_2021", d)
        assert env and env["facts"], d                 # 빈 배경이 없다
        assert env["facts"][0].startswith("서울 신규 확진")
    e4 = build_environment("covid_2021", date(2020, 5, 4))
    assert "4월 20일부터" in e4["headline"] and any("운영 자제 권고" in f for f in e4["facts"])
    e6 = build_environment("covid_2021", date(2020, 5, 6))
    assert not any("권고" in f for f in e6["facts"])     # 권고는 다음 구간으로 이어지지 않는다
    for d in (date(2020, 5, 9), date(2020, 5, 17), date(2020, 6, 14)):
        assert any(f.startswith("집합금지:") and "유흥시설" in f for f in build_environment("covid_2021", d)["facts"])
    for d in (date(2020, 5, 8), date(2020, 6, 15), date(2020, 7, 30)):
        assert not any(f.startswith("집합금지:") for f in build_environment("covid_2021", d)["facts"])


def test_stage2_prompt_states_today():
    from types import SimpleNamespace as NS
    import stage2_poi
    ev = [NS(time="12:00", anchor="zone:11110", category="식사", sub_category="한식", intent="점심",
             reasoning="r", trigger="lifestyle", order=0, companions=[], with_whom=None)]
    cands = {0: [{"poi_id": "P1", "name": "가게", "distance_km": 0.1, "price_band": 1,
                  "known": False, "recent": False, "sat": None, "visit_count": 0}]}
    persona = {"daily_wd": 10000, "daily_we": 20000, "lifestyle": "x", "tendency": "t", "income": "중", "id": "A"}
    want = {date(2020, 5, 4): "오늘: 2020-05-04 (월요일, 평일)",
            date(2020, 5, 5): "오늘: 2020-05-05 (화요일, 공휴일(어린이날))",
            date(2020, 5, 9): "오늘: 2020-05-09 (토요일, 주말)"}
    for d, line in want.items():
        p = stage2_poi.build_stage2_prompt(ev, cands, persona=persona, state={}, today=d)
        assert line in p.splitlines()
    assert not any(l.startswith("오늘:") for l in
                   stage2_poi.build_stage2_prompt(ev, cands, persona=persona, state={}, today=None).splitlines())


def test_life_stage_typos_shown_as_unknown():
    import dawn_context
    for v in ("자녀 학대", "자녀 학대 중", "자녀 학대기", "자녀학대", "자녀 학대 기간"):
        assert dawn_context._life_stage_shown(v) == "미상"
    for v in ("자녀양육", "은퇴", "자녀 양육 중", "학생"):
        assert dawn_context._life_stage_shown(v) == v
