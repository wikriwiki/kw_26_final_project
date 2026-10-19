"""거리두기 대조 환경 — 11-23 의 규칙(1.5단계)을 이어 가고 확진 소식은 그날 것 (2026-10-05)."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/sim"))
from environments import build_environment  # noqa: E402
from environments.covid_2021 import disease_facts  # noqa: E402


def test_hold_keeps_previous_step_but_same_disease_news():
    day = date(2020, 11, 24)
    on = build_environment("covid_2021", day)
    hold = build_environment("covid_2020_hold_1123", day)
    assert "2단계" in on["headline"] and "1.5단계" in hold["headline"]
    assert "오늘부터" not in hold["headline"]
    assert "비교 조건" not in hold["headline"]
    news = disease_facts(day)
    assert on["facts"][:len(news)] == news and hold["facts"][:len(news)] == news


def test_hold_follows_the_day_for_disease_news():
    later = build_environment("covid_2020_hold_1123", date(2020, 11, 27))
    assert later["facts"][:1] == disease_facts(date(2020, 11, 27))[:1]
    assert "1.5단계" in later["headline"]


def test_hold_equals_actual_up_to_1123():
    for d in (date(2020, 11, 19), date(2020, 11, 23)):
        assert build_environment("covid_2020_hold_1123", d) == build_environment("covid_2021", d)


def test_takeout_only_cafe_not_listed_with_restaurant_cutoff():
    joined = "\n".join(build_environment("covid_2021", date(2020, 11, 25))["facts"])
    assert "카페는 시간과 무관하게" in joined
    assert "식당·카페 매장 취식" not in joined and "식당 매장 취식 21:00까지" in joined
