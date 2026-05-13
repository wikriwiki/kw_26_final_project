"""Persona / Agent 검증 테스트."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core import AgeBand, Agent, Gender, Lifestyle, Persona, Segment


def _persona(**overrides: object) -> Persona:
    defaults: dict[str, object] = dict(
        persona_id="p_00001",
        age_band=AgeBand.THIRTIES,
        gender=Gender.MALE,
        occupation="사무직",
        segment=Segment.COMMUTER,
        lifestyles=[Lifestyle.CAFE_LOVER, Lifestyle.GOURMET],
        home_dong="11680510",
        work_dong="11680560",
        income_level=3,
        monthly_budget=300_000,
    )
    defaults.update(overrides)
    return Persona(**defaults)


def test_persona_basic_fields():
    p = _persona()
    assert p.persona_id == "p_00001"
    assert p.segment == Segment.COMMUTER
    assert Lifestyle.CAFE_LOVER in p.lifestyles


def test_persona_is_frozen():
    p = _persona()
    with pytest.raises(ValidationError):
        p.age_band = AgeBand.FORTIES  # type: ignore[misc]


def test_persona_rejects_extra_fields():
    # 시드 데이터에 오타 필드가 섞이면 즉시 실패해야 함
    with pytest.raises(ValidationError):
        Persona(
            persona_id="p_x",
            age_band=AgeBand.TWENTIES,
            gender=Gender.FEMALE,
            segment=Segment.RESIDENT,
            lifestyles=[Lifestyle.HOMEBODY],
            home_dong="11680510",
            income_level=2,
            monthly_budget=100_000,
            unknown_field="oops",  # type: ignore[call-arg]
        )


def test_persona_dong_code_must_be_8_digits():
    with pytest.raises(ValidationError):
        _persona(home_dong="123")        # 짧음
    with pytest.raises(ValidationError):
        _persona(home_dong="1234567a")   # 숫자 아님


def test_persona_lifestyles_min_one():
    with pytest.raises(ValidationError):
        _persona(lifestyles=[])


def test_persona_lifestyles_max_three():
    with pytest.raises(ValidationError):
        _persona(
            lifestyles=[
                Lifestyle.CAFE_LOVER,
                Lifestyle.GOURMET,
                Lifestyle.BUDGET,
                Lifestyle.HEALTH,
            ]
        )


def test_persona_income_level_range():
    with pytest.raises(ValidationError):
        _persona(income_level=0)
    with pytest.raises(ValidationError):
        _persona(income_level=6)


def test_persona_work_dong_optional():
    p = _persona(work_dong=None)
    assert p.work_dong is None


def test_agent_from_persona():
    p = _persona()
    a = Agent.from_persona(p)
    assert a.agent_id == p.persona_id
    assert a.persona is p


def test_agent_rejects_extra_fields():
    p = _persona()
    with pytest.raises(ValidationError):
        Agent(agent_id="a1", persona=p, extra="oops")  # type: ignore[call-arg]
