"""Static persona attributes for a simulated Seoul citizen.

A `Persona` is the immutable identity card of an agent: demographics, segment,
lifestyle tags, home/work locations, daily routine. Created once from seed data
and never modified during simulation.

`Agent` is a thin wrapper that pairs a `Persona` with a runtime identifier. The
social graph (KNOWS, KNOWS_POI edges) is NOT inlined — it lives in Neo4j and is
queried on demand, because carrying ~30 friends per agent × 60K agents would
blow up memory.

[한글 요약]
가상 시민의 "주민등록증 + 성향"에 해당하는 정적 데이터.
한 번 만들면 시뮬레이션 내내 안 바뀜.
6만 명 분량을 메모리에 들고 있어야 하므로 가볍게 유지.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


# ─── 분류 enum ──────────────────────────────────────────────────────


class Segment(StrEnum):
    """소비 패턴 대분류. 프로토타입의 4종을 그대로 계승."""

    COMMUTER = "commuter"                # 출퇴근 직장인
    RESIDENT = "resident"                # 지역 거주민
    EVENING_VISITOR = "evening_visitor"  # 저녁/야간 방문
    WEEKEND_VISITOR = "weekend_visitor"  # 주말 방문객


class Lifestyle(StrEnum):
    """업종 선호도에 영향을 주는 성향 태그. 사람당 1~3개 보유."""

    CAFE_LOVER = "cafe_lover"        # 카페러버: 카페/디저트 가중
    GOURMET = "gourmet"              # 미식가: 일식/한식 가중
    BUDGET = "budget"                # 가성비추구: 패스트푸드/분식
    HEALTH = "health"                # 건강지향: 슈퍼/한식
    SHOPAHOLIC = "shopaholic"        # 쇼핑중독: 패션/전자
    CULTURE = "culture"              # 문화예술: 문화여가/카페
    HOMEBODY = "homebody"            # 집순이: 편의점/슈퍼
    NIGHT_EATER = "night_eater"      # 야식파: 치킨/주류


class Gender(StrEnum):
    MALE = "male"
    FEMALE = "female"


class AgeBand(StrEnum):
    """10세 단위 연령대. 5세 단위는 LLM에게 너무 세분화됨."""

    TEENS = "10s"
    TWENTIES = "20s"
    THIRTIES = "30s"
    FORTIES = "40s"
    FIFTIES = "50s"
    SIXTIES_PLUS = "60s+"


# ─── 도메인 모델 ────────────────────────────────────────────────────


class Persona(BaseModel):
    """Immutable identity of one agent.

    Neo4j mapping: stored as properties on the `(Agent)` node — no separate
    Persona node, since these fields never change after creation.
    """

    # frozen=True : 생성 후 수정 불가. 캐시 키로도 안전하게 쓸 수 있음.
    # extra=forbid : 시드 데이터에 오타 필드가 들어와도 즉시 실패 (silent drop 방지).
    model_config = ConfigDict(frozen=True, extra="forbid")

    # 식별 ───────────────────────────────────────────────
    persona_id: str = Field(min_length=1)  # 예: "p_00001"

    # 인구통계 ──────────────────────────────────────────
    age_band: AgeBand
    gender: Gender
    occupation: str | None = None  # 자유 텍스트 (사무직/자영업/학생/...)

    # 행동 분류 ─────────────────────────────────────────
    segment: Segment
    lifestyles: list[Lifestyle] = Field(min_length=1, max_length=3)
    # ↑ 여러 성향을 동시에 가질 수 있음 (예: cafe_lover + gourmet).
    #   3개 상한으로 LLM 프롬프트 토큰 폭주 방지.

    # 지리 (B079 행정동 코드 8자리) ─────────────────────
    home_dong: str = Field(pattern=r"^\d{8}$")
    work_dong: str | None = Field(default=None, pattern=r"^\d{8}$")
    # ↑ 학생/무직/은퇴자는 work_dong = None.

    # 경제 ──────────────────────────────────────────────
    income_level: int = Field(ge=1, le=5)        # 1=최하 ~ 5=최상
    monthly_budget: int = Field(ge=0)            # 외식+소비 월 예산 (원)

    # 일과표 — Plan 생성 시 컨텍스트로 사용 ──────────────
    typical_breakfast_hour: int = Field(default=7, ge=5, le=10)
    typical_lunch_hour: int = Field(default=12, ge=11, le=14)
    typical_dinner_hour: int = Field(default=19, ge=17, le=22)
    work_start_hour: int | None = Field(default=None, ge=0, le=23)
    work_end_hour: int | None = Field(default=None, ge=0, le=23)


class Agent(BaseModel):
    """Runtime agent — persona + identity. Social edges live in Neo4j.

    Use this as the in-memory representation passed around between modules.
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1)
    # 보통 agent_id == persona.persona_id (1:1 매핑). 별도로 두는 이유는
    # 미래에 한 페르소나로 여러 시나리오를 돌릴 가능성에 대비.

    persona: Persona

    @classmethod
    def from_persona(cls, persona: Persona) -> Agent:
        """Persona의 ID를 그대로 agent_id로 쓰는 표준 생성자."""
        return cls(agent_id=persona.persona_id, persona=persona)
