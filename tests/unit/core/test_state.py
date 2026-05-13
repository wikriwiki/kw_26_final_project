"""State 검증 + 헬퍼 함수 테스트."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core import (
    AgeBand,
    Gender,
    Lifestyle,
    Persona,
    Segment,
    State,
    carry_over,
    initial_state,
)


def _persona() -> Persona:
    return Persona(
        persona_id="p_00001",
        age_band=AgeBand.THIRTIES,
        gender=Gender.MALE,
        segment=Segment.COMMUTER,
        lifestyles=[Lifestyle.CAFE_LOVER],
        home_dong="11680510",
        income_level=3,
        monthly_budget=300_000,
    )


def test_initial_state_uses_persona_defaults():
    p = _persona()
    s = initial_state(p)
    assert s.day == 0
    assert s.agent_id == "p_00001"
    assert s.current_dong == "11680510"
    assert s.money_balance == 300_000
    assert s.today_spending == 0
    assert s.mood == 0.5
    assert s.fatigue == 0.3
    assert s.aware_events == set()
    assert s.state_id == "s_p_00001_d0"


def test_initial_state_with_custom_day():
    s = initial_state(_persona(), day=5)
    assert s.day == 5
    assert s.state_id.endswith("_d5")


def test_state_mood_range_validation():
    p = _persona()
    with pytest.raises(ValidationError):
        State(
            state_id="s_x_d0",
            agent_id="x",
            day=0,
            current_dong="11680510",
            mood=1.5,  # 1.0 초과
            fatigue=0.0,
            money_balance=0,
        )
    with pytest.raises(ValidationError):
        State(
            state_id="s_x_d0",
            agent_id="x",
            day=0,
            current_dong="11680510",
            mood=-0.1,
            fatigue=0.0,
            money_balance=0,
        )


def test_carry_over_recovers_mood_and_fatigue():
    p = _persona()
    prev = initial_state(p)
    prev.mood = 0.5
    prev.fatigue = 0.6
    prev.today_spending = 25_000

    nxt = carry_over(prev, new_day=1)
    assert nxt.day == 1
    assert nxt.mood == pytest.approx(0.7)        # 0.5 + 0.2
    assert nxt.fatigue == pytest.approx(0.3)     # 0.6 - 0.3
    assert nxt.today_spending == 0               # 리셋
    assert nxt.aware_events == set()             # 망각


def test_carry_over_clamps_mood_to_one():
    p = _persona()
    prev = initial_state(p)
    prev.mood = 0.95

    nxt = carry_over(prev, new_day=1)
    assert nxt.mood == 1.0  # 0.95 + 0.2 = 1.15 → 1.0으로 clip


def test_carry_over_clamps_fatigue_to_zero():
    p = _persona()
    prev = initial_state(p)
    prev.fatigue = 0.1

    nxt = carry_over(prev, new_day=1)
    assert nxt.fatigue == 0.0  # 0.1 - 0.3 = -0.2 → 0.0


def test_state_rejects_invalid_dong_code():
    with pytest.raises(ValidationError):
        State(
            state_id="s_x_d0",
            agent_id="x",
            day=0,
            current_dong="abc",  # 8자리 숫자 아님
            mood=0.5,
            fatigue=0.5,
            money_balance=0,
        )


def test_state_money_balance_nonnegative():
    with pytest.raises(ValidationError):
        State(
            state_id="s_x_d0",
            agent_id="x",
            day=0,
            current_dong="11680510",
            mood=0.5,
            fatigue=0.5,
            money_balance=-1,
        )
