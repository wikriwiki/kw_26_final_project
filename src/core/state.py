"""Per-day runtime snapshot of an agent.

A `State` represents what one agent is feeling / where they are / how much
money they have on a specific simulation day. It is created fresh each morning
from the previous day's State and updated as episodes happen.

Persona stays constant; State is the part that evolves over time.

[한글 요약]
에이전트의 "오늘 컨디션".
매일 아침 새로 만들고, 하루 동안 일부 필드(기분/피곤/지갑)가 갱신됨.
Day별로 별도 노드로 저장해 시계열 추적 가능.
"""
from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from .agent import Persona


def _now_utc() -> datetime:
    # Python 3.13에서 datetime.utcnow() deprecated. timezone-aware UTC 사용.
    return datetime.now(UTC)


class State(BaseModel):
    """Snapshot of one agent on one simulation day.

    Neo4j mapping: `(Agent)-[:HAS_STATE]->(State {day: t})`. Keeping a separate
    node per day preserves the time series, at the cost of 60K × N_days nodes.
    For 7-day simulations that's ~420K nodes — manageable.
    """

    # extra=forbid : 직렬화 / 역직렬화 시 알 수 없는 필드를 거부.
    # frozen 아님 : mood/fatigue/money_balance 등이 하루 중 업데이트됨.
    model_config = ConfigDict(extra="forbid")

    # 식별 ───────────────────────────────────────────────
    state_id: str = Field(min_length=1)        # 예: "s_p00001_d3"
    agent_id: str = Field(min_length=1)
    day: int = Field(ge=0)                     # 시뮬레이션 day (0부터)

    # 위치 ───────────────────────────────────────────────
    current_dong: str = Field(pattern=r"^\d{8}$")  # 행정동 코드
    current_poi_id: str | None = None              # 점포 안에 있는 경우

    # 감정/체력 (0.0 ~ 1.0) ─────────────────────────────
    mood: float = Field(ge=0.0, le=1.0)
    # ↑ 외출/소비 의향. 높을수록 활동적.
    fatigue: float = Field(ge=0.0, le=1.0)
    # ↑ 피로도. 높을수록 간편식/배달 선호 → 외식 줄어듦.

    # 경제 ──────────────────────────────────────────────
    money_balance: int = Field(ge=0)                # 이번 달 잔여 예산
    today_spending: int = Field(default=0, ge=0)    # 오늘 누적 지출

    # 알고 있는 이벤트 ID (뉴스/정책) ────────────────────
    aware_events: set[str] = Field(default_factory=set)
    # ↑ "AWARE" 상태인 이벤트만 들어감. HEARD/UNAWARE는 별도 처리(추후).

    # 최근 활동 참조 ─────────────────────────────────────
    last_episode_id: str | None = None
    last_updated_at: datetime = Field(default_factory=_now_utc)


# ─── 헬퍼 함수 ──────────────────────────────────────────────────────


def initial_state(persona: Persona, day: int = 0) -> State:
    """Create the Day 0 State from a Persona.

    Defaults:
      - 위치: 집(home_dong)
      - 기분: 0.5 (중간)
      - 피곤도: 0.3 (조금)
      - 지갑: monthly_budget 그대로 (월초 가정)
    """
    return State(
        state_id=f"s_{persona.persona_id}_d{day}",
        agent_id=persona.persona_id,
        day=day,
        current_dong=persona.home_dong,
        mood=0.5,
        fatigue=0.3,
        money_balance=persona.monthly_budget,
        today_spending=0,
    )


def carry_over(prev: State, new_day: int) -> State:
    """Build the next day's State from the previous one.

    Rules (단순 — 추후 룰 엔진으로 대체):
      - mood : +0.2 자연 회복 (잠자고 일어남)
      - fatigue : -0.3 회복
      - money_balance : 그대로 (월 단위 리셋은 별도 처리)
      - today_spending : 0으로 리셋
      - current_dong : 집으로 (자고 일어남 가정)
      - aware_events : 모두 망각 (TODO: 일부 보존 정책)
    """
    return State(
        state_id=f"s_{prev.agent_id}_d{new_day}",
        agent_id=prev.agent_id,
        day=new_day,
        current_dong=prev.current_dong,  # 호출자가 home_dong으로 덮어쓸 수 있음
        mood=min(1.0, prev.mood + 0.2),
        fatigue=max(0.0, prev.fatigue - 0.3),
        money_balance=prev.money_balance,
        today_spending=0,
        aware_events=set(),
        last_episode_id=prev.last_episode_id,  # 어제 마지막 활동 기억은 유지
    )
