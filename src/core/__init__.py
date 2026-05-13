"""`src.core` — Pydantic domain models.

Pure data classes only. No business logic, no I/O, no LLM/DB access.
Other modules import from here; nothing here imports from sibling packages.

[한글 요약]
도메인 모델 (Persona, Agent, State, Plan, Episode 등) 정의.
순수 데이터만 — 비즈니스 로직/I/O/외부 호출 일체 없음.
다른 모든 모듈이 여기를 import하므로 의존성 zero로 유지.
"""
from __future__ import annotations

from .agent import (
    Agent,
    AgeBand,
    Gender,
    Lifestyle,
    Persona,
    Segment,
)
from .plan import (
    ActionType,
    Episode,
    EpisodeDraft,
    EpisodeSource,
    Plan,
    PlanDraft,
    TimeSlot,
)
from .state import State, carry_over, initial_state

__all__ = [
    # agent
    "Agent",
    "AgeBand",
    "Gender",
    "Lifestyle",
    "Persona",
    "Segment",
    # state
    "State",
    "carry_over",
    "initial_state",
    # plan
    "ActionType",
    "Episode",
    "EpisodeDraft",
    "EpisodeSource",
    "Plan",
    "PlanDraft",
    "TimeSlot",
]
