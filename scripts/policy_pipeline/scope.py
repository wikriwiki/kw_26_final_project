"""
scope.py
========
Section 6 — Scope Analysis.

검증된 정책 한 건이 다음 단위에 영향을 미치는지 산출:
  서울 전체 / 자치구 (District) / 행정동 (Dong) / 카테고리 / POI / Agent

`GraphReader` Protocol 로 그래프 조회를 추상화 → 실제 구현은 neo4j_reader.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

from scripts.policy_pipeline.models import ValidatedPolicy
from scripts.policy_pipeline.vocabulary import SEOUL_DISTRICTS


class ScopeUnit(str, Enum):
    SEOUL_WIDE = "seoul_wide"
    DISTRICT = "district"
    DONG = "dong"
    CATEGORY = "category"
    POI = "poi"
    AGENT_GROUP = "agent_group"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# 그래프 조회 인터페이스
# ---------------------------------------------------------------------------
class GraphReader(Protocol):
    def dongs_in_districts(self, district_names: list[str]) -> list[str]:
        """자치구 이름 리스트 → 행정동 코드 리스트."""
        ...

    def pois_in_dongs_for_categories(
        self, dong_codes: list[str], l1_categories: list[str],
    ) -> list[str]:
        """동 코드 × L1 카테고리 → POI id 리스트."""
        ...

    def agents_in_dongs_for_groups(
        self, dong_codes: list[str], target_groups: list[str],
    ) -> list[str]:
        """동 코드 × 대상 그룹 → 에이전트 id 리스트."""
        ...


class NullGraphReader:
    """Neo4j 미연결 환경용 stub. 항상 빈 리스트."""

    def dongs_in_districts(self, district_names: list[str]) -> list[str]:
        return []

    def pois_in_dongs_for_categories(self, dong_codes, l1_categories) -> list[str]:
        return []

    def agents_in_dongs_for_groups(self, dong_codes, target_groups) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# 결과
# ---------------------------------------------------------------------------
class PolicyScope(BaseModel):
    policy_id: str
    scope_units: list[ScopeUnit] = Field(default_factory=list)
    affected_districts: list[str] = Field(default_factory=list)
    affected_dongs: list[str] = Field(default_factory=list)
    affected_categories: list[str] = Field(default_factory=list)
    affected_pois: list[str] = Field(default_factory=list)
    affected_agents: list[str] = Field(default_factory=list)
    affected_target_groups: list[str] = Field(default_factory=list)

    # 후행 모듈이 채움
    cache_keys_to_invalidate: list[str] = Field(default_factory=list)
    summary_jobs_to_rebuild: list[str] = Field(default_factory=list)

    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def analyze_textual_scope(policy: ValidatedPolicy) -> PolicyScope:
    """그래프 조회 없이 정책 텍스트만으로 산출."""
    units: list[ScopeUnit] = []

    valid_districts = [d for d in policy.target_districts if d in SEOUL_DISTRICTS]

    # 25 자치구 전부 명시 = 서울 전역으로 취급
    if len(valid_districts) >= len(SEOUL_DISTRICTS):
        units.append(ScopeUnit.SEOUL_WIDE)
    elif valid_districts:
        units.append(ScopeUnit.DISTRICT)

    if policy.benefit_categories:
        units.append(ScopeUnit.CATEGORY)
    if policy.target_groups:
        units.append(ScopeUnit.AGENT_GROUP)

    if not units:
        units.append(ScopeUnit.UNKNOWN)

    return PolicyScope(
        policy_id=policy.policy_id,
        scope_units=units,
        affected_districts=valid_districts,
        affected_categories=list(policy.benefit_categories),
        affected_target_groups=list(policy.target_groups),
    )


def analyze_graph_scope(
    policy: ValidatedPolicy,
    graph_reader: GraphReader,
    *,
    base_scope: PolicyScope | None = None,
) -> PolicyScope:
    """textual scope 를 그래프 조회로 확장.

    SEOUL_WIDE 면 노드 열거 생략 (전체 무효화로 처리).
    그 외엔 자치구 → 동, 동 × 카테고리 → POI, 동 × 대상그룹 → Agent.
    """
    scope = base_scope or analyze_textual_scope(policy)

    if ScopeUnit.SEOUL_WIDE in scope.scope_units:
        return scope

    if not scope.affected_districts:
        return scope

    dongs = graph_reader.dongs_in_districts(scope.affected_districts)
    if dongs:
        scope.affected_dongs = dongs
        if ScopeUnit.DONG not in scope.scope_units:
            scope.scope_units.append(ScopeUnit.DONG)

    if scope.affected_categories and dongs:
        pois = graph_reader.pois_in_dongs_for_categories(
            dongs, scope.affected_categories,
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
    reader = graph_reader or NullGraphReader()
    return analyze_graph_scope(policy, reader, base_scope=analyze_textual_scope(policy))
