"""
scope.py
========
Section 6 — Scope Analysis.

검증된 정책 한 건이 시뮬레이션의 어느 단위(서울 전체 / 자치구 / 행정동 /
상권 / 업종 / POI / Agent 그룹)에 영향을 미치는지 산출.

이 모듈은 **Pure logic + 외부 그래프 조회 Protocol** 의 두 층으로 분리:

  - 정책 텍스트(`ValidatedPolicy`)에서 곧바로 알 수 있는 범위(자치구 명, 업종 명)
    는 `analyze_textual_scope()` 가 직접 계산.
  - 시뮬레이션 그래프에 의존하는 항목(영향받는 POI 목록, 영향받는 Agent id 등)
    은 `GraphReader` Protocol 을 거쳐 조회. 실 구현(Neo4j) 은 이번 PR 범위 밖.

산출물 `PolicyScope` 는 다음 단계(`invalidator.py`, `summary_jobs.py`)의 입력이 된다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

from src.policy_pipeline.models import ValidatedPolicy
from src.policy_pipeline.vocabulary import (
    SEOUL_DISTRICTS,
    is_seoul_wide_scope,
)


# ---------------------------------------------------------------------------
# Scope 단위 분류
# ---------------------------------------------------------------------------
class ScopeUnit(str, Enum):
    SEOUL_WIDE = "seoul_wide"           # 서울 전체
    DISTRICT = "district"                # 자치구 (강남구 등)
    DONG = "dong"                        # 행정동
    COMMERCIAL_AREA = "commercial_area"  # 상권
    INDUSTRY = "industry"                # 업종
    POI = "poi"                          # 개별 매장/장소
    AGENT_GROUP = "agent_group"          # 특정 조건 에이전트 집합
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# 그래프 조회 Protocol
# ---------------------------------------------------------------------------
class GraphReader(Protocol):
    """Neo4j 등 외부 그래프 store 의 읽기 인터페이스.

    실제 Neo4j 구현은 별도 PR 에서 `src/graph/queries/` 에 둔다.
    이번 PR 은 시그니처와 stub 구현만 제공.
    """

    def dongs_in_districts(self, districts: list[str]) -> list[str]:
        """자치구 리스트가 포함하는 행정동 코드 목록."""
        ...

    def pois_in_dongs_for_industries(
        self, dongs: list[str], industries: list[str],
    ) -> list[str]:
        """동 × 업종 으로 필터링된 POI id 목록."""
        ...

    def agents_in_dongs_for_groups(
        self, dongs: list[str], target_groups: list[str],
    ) -> list[str]:
        """대상 그룹(청년, 노인 등) 조건에 맞는 에이전트 id 목록."""
        ...


class NullGraphReader:
    """Neo4j 가 아직 연결 안 된 환경용 stub. 항상 빈 리스트 반환.

    파이프라인은 이걸 받아도 textual scope 부분은 정상 작동한다.
    """

    def dongs_in_districts(self, districts: list[str]) -> list[str]:
        return []

    def pois_in_dongs_for_industries(
        self, dongs: list[str], industries: list[str],
    ) -> list[str]:
        return []

    def agents_in_dongs_for_groups(
        self, dongs: list[str], target_groups: list[str],
    ) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# 결과 모델
# ---------------------------------------------------------------------------
class PolicyScope(BaseModel):
    """한 정책이 영향을 주는 시뮬레이션 단위들의 집합.

    `affected_*` 필드 = 직접/간접 영향을 받는 노드 id 들.
    `cache_keys_to_invalidate` / `summary_jobs_to_rebuild` 는 후행 모듈이 채워준다.
    """
    policy_id: str

    # 정책 텍스트만으로 결정되는 부분
    scope_units: list[ScopeUnit] = Field(default_factory=list)
    affected_regions: list[str] = Field(default_factory=list)        # ["서울"] 등
    affected_districts: list[str] = Field(default_factory=list)      # ["강남구", ...]
    affected_industries: list[str] = Field(default_factory=list)
    affected_target_groups: list[str] = Field(default_factory=list)

    # 그래프 조회를 통해 채워지는 부분 (Neo4j 미연결 시 빈 리스트)
    affected_dongs: list[str] = Field(default_factory=list)
    affected_pois: list[str] = Field(default_factory=list)
    affected_agents: list[str] = Field(default_factory=list)

    # 후행 모듈(invalidator, summary_jobs)이 채움
    cache_keys_to_invalidate: list[str] = Field(default_factory=list)
    summary_jobs_to_rebuild: list[str] = Field(default_factory=list)

    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# 메인 API
# ---------------------------------------------------------------------------
def analyze_textual_scope(policy: ValidatedPolicy) -> PolicyScope:
    """그래프 조회 없이 정책 텍스트만으로 산출 가능한 범위.

    `affected_dongs/pois/agents` 는 비어 있고, `analyze_graph_scope()` 가 이어서 채운다.
    """
    units: list[ScopeUnit] = []

    # 지역 단위 결정
    if is_seoul_wide_scope(policy.target_regions):
        units.append(ScopeUnit.SEOUL_WIDE)
        affected_regions = ["서울"]
    else:
        affected_regions = list(policy.target_regions)

    valid_districts = [d for d in policy.target_districts if d in SEOUL_DISTRICTS]
    if valid_districts:
        units.append(ScopeUnit.DISTRICT)

    if policy.target_industries:
        units.append(ScopeUnit.INDUSTRY)

    if policy.target_groups:
        units.append(ScopeUnit.AGENT_GROUP)

    if not units:
        units.append(ScopeUnit.UNKNOWN)

    return PolicyScope(
        policy_id=policy.policy_id,
        scope_units=units,
        affected_regions=affected_regions,
        affected_districts=valid_districts,
        affected_industries=list(policy.target_industries),
        affected_target_groups=list(policy.target_groups),
    )


def analyze_graph_scope(
    policy: ValidatedPolicy,
    graph_reader: GraphReader,
    *,
    base_scope: PolicyScope | None = None,
) -> PolicyScope:
    """텍스트 스코프를 그래프 조회로 확장.

    - SEOUL_WIDE 면 행정동/POI/Agent 를 전수 조회하지 않고 비워둔 채 두고,
      대신 후행 invalidator 가 'seoul-wide' 키를 통째로 무효화하게 한다.
    - 자치구 지정이면 해당 자치구의 모든 행정동 → 업종 필터로 POI/Agent 조회.
    """
    scope = base_scope or analyze_textual_scope(policy)

    if ScopeUnit.SEOUL_WIDE in scope.scope_units:
        # 전체 무효화로 갈 것이므로 노드 열거 생략
        return scope

    if not scope.affected_districts:
        return scope

    dongs = graph_reader.dongs_in_districts(scope.affected_districts)
    if dongs:
        scope.affected_dongs = dongs
        if scope.scope_units.count(ScopeUnit.DONG) == 0:
            scope.scope_units.append(ScopeUnit.DONG)

    if scope.affected_industries and dongs:
        pois = graph_reader.pois_in_dongs_for_industries(
            dongs, scope.affected_industries,
        )
        if pois:
            scope.affected_pois = pois
            scope.scope_units.append(ScopeUnit.POI)

    if scope.affected_target_groups and dongs:
        agents = graph_reader.agents_in_dongs_for_groups(
            dongs, scope.affected_target_groups,
        )
        if agents:
            scope.affected_agents = agents

    return scope


def analyze_scope(
    policy: ValidatedPolicy,
    graph_reader: GraphReader | None = None,
) -> PolicyScope:
    """텍스트 + 그래프 scope 를 한 번에. graph_reader 가 None 이면 NullGraphReader 사용."""
    reader: GraphReader = graph_reader or NullGraphReader()
    base = analyze_textual_scope(policy)
    return analyze_graph_scope(policy, reader, base_scope=base)
