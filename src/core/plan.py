"""Daily plan == daily log for one agent.

In this simulation we deliberately collapse "what was planned" and "what
actually happened" into one structure: the `Plan` LLM produces IS the day's
event log. Post-hoc fields (spending, satisfaction) are filled in as the
rule engine resolves each episode.

Split into two layers:
  - `PlanDraft` / `EpisodeDraft` : the minimal schema the LLM outputs.
  - `Plan` / `Episode`           : the full record we persist (adds IDs,
                                    metadata, post-hoc outcomes).

The LLM only ever sees the Draft schema → fewer tokens, fewer ways for it
to hallucinate IDs.

[한글 요약]
한 사람의 하루 계획 == 하루 일지.
LLM이 만든 게 그대로 발생 기록이 됨 (별도 추적 안 함).
LLM 출력용 Draft와 저장용 Full 모델을 분리해 토큰 효율 + 안전성 확보.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def _now_utc() -> datetime:
    # Python 3.13에서 datetime.utcnow() deprecated. timezone-aware UTC 사용.
    return datetime.now(UTC)


# ─── 분류 enum ──────────────────────────────────────────────────────


class TimeSlot(StrEnum):
    """하루를 7개 시간대로 나눈 단위.

    LLM이 시각을 직접 결정하는 것보다 슬롯을 고르는 게 일관성 ↑ + 캐시 친화.
    구체적인 hour는 후처리에서 슬롯 중간값으로 채울 수 있음.
    """

    EARLY_MORNING = "early_morning"  # 05-08
    MORNING = "morning"              # 08-11
    LUNCH = "lunch"                  # 11-14
    AFTERNOON = "afternoon"          # 14-17
    EVENING = "evening"              # 17-20
    NIGHT = "night"                  # 20-23
    LATE_NIGHT = "late_night"        # 23-05


class ActionType(StrEnum):
    """활동 대분류. 업종 코드보다 상위 개념."""

    DINING = "dining"      # 외식
    SHOPPING = "shopping"  # 쇼핑
    LEISURE = "leisure"    # 여가/문화
    COMMUTE = "commute"    # 출퇴근/이동
    HOME = "home"          # 재택/휴식
    MEETING = "meeting"    # 사람 만남


class EpisodeSource(StrEnum):
    """이 활동이 왜 일어났는지. Plan 분석 + 디버깅용."""

    ROUTINE = "routine"                  # 일상 패턴 (출근/식사)
    DISCRETIONARY = "discretionary"      # LLM 자유 선택
    APPOINTMENT = "appointment"          # 어제 잡힌 약속 (Night 주입)
    POLICY_DRIVEN = "policy_driven"      # 정책 유도
    PEER_RECOMMENDED = "peer_recommended"  # 동료 추천


# ─── LLM 출력 전용 (Draft) ─────────────────────────────────────────
# LLM은 ID/타임스탬프/사후 결과를 만들지 않음. 호출자가 부여.


class EpisodeDraft(BaseModel):
    """What the LLM emits per episode. Minimal schema.

    호출자가 Episode로 변환할 때 episode_id 부여 + spending/satisfaction은
    None으로 두고 후속 단계에서 채움.
    """

    # strict 스키마 — sglang JSON 강제 디코딩 친화.
    model_config = ConfigDict(extra="forbid")

    time_slot: TimeSlot
    hour: int = Field(ge=0, le=23)
    sequence: int = Field(ge=0)              # 하루 내 순서 (0,1,2,...)

    action: ActionType
    industry_code: str | None = None         # 외식/쇼핑일 때 채워짐
    poi_id: str | None = None                # 점포 ID (있으면)

    region_code: str = Field(pattern=r"^\d{8}$")  # 행정동 코드
    partner_agent_id: str | None = None      # 약속/만남 상대

    source: EpisodeSource
    motivation: str = Field(max_length=200)  # 한 줄 이유


class PlanDraft(BaseModel):
    """What the LLM emits as a full day plan. Just a list of EpisodeDraft."""

    model_config = ConfigDict(extra="forbid")

    episodes: list[EpisodeDraft] = Field(min_length=1, max_length=12)
    # ↑ 평균 5~8개. 12개 상한은 토큰 폭주 방지.


# ─── 저장용 풀모델 ──────────────────────────────────────────────────


class Episode(EpisodeDraft):
    """Full episode record — Draft + 사후 부여 필드."""

    model_config = ConfigDict(extra="forbid")

    # 호출자(orchestrator/rule engine)가 부여 ──────────
    episode_id: UUID = Field(default_factory=uuid4)
    spending: int | None = Field(default=None, ge=0)
    satisfaction: float | None = Field(default=None, ge=0.0, le=1.0)


class Plan(BaseModel):
    """Full plan record. Mutable — post-hoc aggregates 갱신됨.

    Neo4j mapping: `(Agent)-[:HAS_PLAN]->(Plan)-[:INCLUDES]->(Episode)`.
    """

    model_config = ConfigDict(extra="forbid")

    plan_id: UUID = Field(default_factory=uuid4)
    agent_id: str
    day: int = Field(ge=0)

    episodes: list[Episode] = Field(min_length=1, max_length=12)

    # 메타데이터 ────────────────────────────────────────
    generated_at: datetime = Field(default_factory=_now_utc)
    llm_model: str | None = None  # "exaone" | "qwen" | None(테스트)

    # 사후 집계 (rule engine이 채움) ────────────────────
    total_spending: int = Field(default=0, ge=0)
    avg_satisfaction: float | None = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def from_draft(
        cls,
        draft: PlanDraft,
        *,
        agent_id: str,
        day: int,
        llm_model: str | None = None,
    ) -> Plan:
        """Draft → Plan 변환. LLM 호출 직후 orchestrator가 호출.

        Episode가 EpisodeDraft를 상속하므로 필드 그대로 복사 + episode_id 자동 부여.
        """
        episodes = [Episode(**e.model_dump()) for e in draft.episodes]
        return cls(
            agent_id=agent_id,
            day=day,
            episodes=episodes,
            llm_model=llm_model,
        )

    def recompute_aggregates(self) -> None:
        """rule engine이 episode마다 spending/satisfaction을 채운 뒤 호출.

        mutable 메서드 — Plan을 그 자리에서 업데이트.
        """
        total = 0
        scored: list[float] = []
        for ep in self.episodes:
            if ep.spending is not None:
                total += ep.spending
            if ep.satisfaction is not None:
                scored.append(ep.satisfaction)
        self.total_spending = total
        self.avg_satisfaction = (sum(scored) / len(scored)) if scored else None
