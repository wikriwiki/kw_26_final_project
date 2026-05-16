"""scope: 텍스트 스코프 + GraphReader 통합."""

from datetime import datetime, timezone

from src.policy_pipeline.models import BenefitType, ValidatedPolicy
from src.policy_pipeline.scope import (
    NullGraphReader,
    ScopeUnit,
    analyze_graph_scope,
    analyze_scope,
    analyze_textual_scope,
)


def _make_validated(**overrides) -> ValidatedPolicy:
    base = {
        "policy_id": "policy_x",
        "title": "t",
        "summary": "s",
        "source_file": "/tmp/x.txt",
        "target_regions": [],
        "target_districts": [],
        "target_industries": [],
        "target_groups": [],
        "benefit_type": BenefitType.COUPON,
        "validated_at": datetime.now(timezone.utc),
        "validation_notes": [],
    }
    base.update(overrides)
    return ValidatedPolicy(**base)


def test_seoul_wide_textual_scope():
    p = _make_validated(target_regions=["서울시"])
    scope = analyze_textual_scope(p)
    assert ScopeUnit.SEOUL_WIDE in scope.scope_units
    assert scope.affected_regions == ["서울"]


def test_district_textual_scope():
    p = _make_validated(target_districts=["강남구", "마포구"])
    scope = analyze_textual_scope(p)
    assert ScopeUnit.DISTRICT in scope.scope_units
    assert scope.affected_districts == ["강남구", "마포구"]


def test_unknown_unit_when_empty():
    p = _make_validated()
    scope = analyze_textual_scope(p)
    assert scope.scope_units == [ScopeUnit.UNKNOWN]


class _StubReader:
    def __init__(self, dongs, pois, agents):
        self._dongs = dongs
        self._pois = pois
        self._agents = agents

    def dongs_in_districts(self, districts):
        return self._dongs

    def pois_in_dongs_for_industries(self, dongs, industries):
        return self._pois

    def agents_in_dongs_for_groups(self, dongs, target_groups):
        return self._agents


def test_graph_reader_fills_affected_nodes():
    p = _make_validated(
        target_districts=["강남구"],
        target_industries=["음식점"],
        target_groups=["청년"],
    )
    reader = _StubReader(dongs=["1168010100"], pois=["poi_1"], agents=["AGT_a"])
    scope = analyze_scope(p, reader)
    assert scope.affected_dongs == ["1168010100"]
    assert scope.affected_pois == ["poi_1"]
    assert scope.affected_agents == ["AGT_a"]


def test_seoul_wide_skips_node_enumeration():
    # SEOUL_WIDE 면 그래프 조회 생략, 전체 무효화 키로 처리됨
    p = _make_validated(target_regions=["서울시"])
    reader = _StubReader(dongs=["x"], pois=["y"], agents=["z"])
    scope = analyze_graph_scope(p, reader)
    assert scope.affected_dongs == []
    assert scope.affected_pois == []


def test_null_graph_reader_returns_empty_lists():
    p = _make_validated(target_districts=["강남구"])
    scope = analyze_scope(p, NullGraphReader())
    assert scope.affected_dongs == []
    assert scope.affected_districts == ["강남구"]
