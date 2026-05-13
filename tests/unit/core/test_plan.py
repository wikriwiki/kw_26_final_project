"""Plan / Episode / Draft 검증 및 변환 테스트."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core import (
    ActionType,
    Episode,
    EpisodeDraft,
    EpisodeSource,
    Plan,
    PlanDraft,
    TimeSlot,
)


def _draft_episode(**overrides: object) -> EpisodeDraft:
    defaults: dict[str, object] = dict(
        time_slot=TimeSlot.LUNCH,
        hour=12,
        sequence=0,
        action=ActionType.DINING,
        industry_code="K01",
        poi_id="poi_1234",
        region_code="11680560",
        source=EpisodeSource.ROUTINE,
        motivation="점심시간 직장 근처 한식",
    )
    defaults.update(overrides)
    return EpisodeDraft(**defaults)


# ─── EpisodeDraft ─────────────────────────────────────────


def test_episode_draft_basic():
    e = _draft_episode()
    assert e.action == ActionType.DINING
    assert e.time_slot == TimeSlot.LUNCH


def test_episode_draft_hour_range():
    with pytest.raises(ValidationError):
        _draft_episode(hour=24)
    with pytest.raises(ValidationError):
        _draft_episode(hour=-1)


def test_episode_draft_motivation_max_length():
    long = "x" * 201
    with pytest.raises(ValidationError):
        _draft_episode(motivation=long)


def test_episode_draft_rejects_extra():
    with pytest.raises(ValidationError):
        EpisodeDraft(
            time_slot=TimeSlot.LUNCH,
            hour=12,
            sequence=0,
            action=ActionType.HOME,
            region_code="11680510",
            source=EpisodeSource.ROUTINE,
            motivation="x",
            ghost_field="oops",  # type: ignore[call-arg]
        )


# ─── PlanDraft ────────────────────────────────────────────


def test_plan_draft_min_one_episode():
    with pytest.raises(ValidationError):
        PlanDraft(episodes=[])


def test_plan_draft_max_12_episodes():
    eps = [_draft_episode(sequence=i, hour=10 + (i % 10)) for i in range(13)]
    with pytest.raises(ValidationError):
        PlanDraft(episodes=eps)


# ─── Episode (full) ───────────────────────────────────────


def test_episode_auto_generates_id():
    e = Episode(**_draft_episode().model_dump())
    assert e.episode_id is not None
    assert e.spending is None
    assert e.satisfaction is None


def test_episode_satisfaction_range():
    base = _draft_episode().model_dump()
    with pytest.raises(ValidationError):
        Episode(**base, satisfaction=1.5)
    with pytest.raises(ValidationError):
        Episode(**base, satisfaction=-0.1)


# ─── Plan (full) + from_draft 변환 ────────────────────────


def test_plan_from_draft_promotes_episodes():
    draft = PlanDraft(
        episodes=[
            _draft_episode(sequence=0, hour=8, time_slot=TimeSlot.MORNING),
            _draft_episode(sequence=1, hour=12, time_slot=TimeSlot.LUNCH),
            _draft_episode(sequence=2, hour=19, time_slot=TimeSlot.EVENING),
        ]
    )
    plan = Plan.from_draft(draft, agent_id="p_00001", day=3, llm_model="qwen")

    assert plan.agent_id == "p_00001"
    assert plan.day == 3
    assert plan.llm_model == "qwen"
    assert len(plan.episodes) == 3
    # 모든 episode가 ID 자동 부여됨
    assert all(e.episode_id is not None for e in plan.episodes)
    # Draft 필드는 그대로 보존됨
    assert plan.episodes[0].time_slot == TimeSlot.MORNING
    assert plan.episodes[1].time_slot == TimeSlot.LUNCH


def test_plan_recompute_aggregates():
    draft = PlanDraft(
        episodes=[
            _draft_episode(sequence=0, hour=12),
            _draft_episode(sequence=1, hour=19, time_slot=TimeSlot.EVENING),
        ]
    )
    plan = Plan.from_draft(draft, agent_id="a", day=0)

    # 사후에 rule engine이 채운 결과를 시뮬레이션
    plan.episodes[0].spending = 12_000
    plan.episodes[0].satisfaction = 0.8
    plan.episodes[1].spending = 30_000
    plan.episodes[1].satisfaction = 0.6

    plan.recompute_aggregates()
    assert plan.total_spending == 42_000
    assert plan.avg_satisfaction == pytest.approx(0.7)


def test_plan_recompute_aggregates_handles_missing_outcomes():
    draft = PlanDraft(episodes=[_draft_episode()])
    plan = Plan.from_draft(draft, agent_id="a", day=0)

    plan.recompute_aggregates()
    assert plan.total_spending == 0
    assert plan.avg_satisfaction is None
