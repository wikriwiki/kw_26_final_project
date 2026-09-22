"""scope: 텍스트 스코프 + GraphReader 통합."""

from datetime import datetime, timezone

from scripts.policy_pipeline.models import PolicyType, ValidatedPolicy
from scripts.policy_pipeline.scope import (
    NullGraphReader,
    ScopeUnit,
    analyze_graph_scope,
    analyze_scope,
    analyze_textual_scope,
)
from scripts.policy_pipeline.vocabulary import SEOUL_DISTRICTS


def _make(**overrides) -> ValidatedPolicy:
    base = {
        "policy_id": "policy_x",
        "title": "t",
        "summary": "s",
        "source_file": "/tmp/x.txt",
        "target_districts": [],
        "target_dongs": [],
        "benefit_categories": [],
        "target_groups": [],
        "policy_type": PolicyType.COUPON,
        "validated_at": datetime.now(timezone.utc),
        "validation_notes": [],
    }
    base.update(overrides)
    return ValidatedPolicy(**base)


def test_25_districts_means_seoul_wide():
    p = _make(target_districts=list(SEOUL_DISTRICTS))
    scope = analyze_textual_scope(p)
    assert ScopeUnit.SEOUL_WIDE in scope.scope_units


def test_partial_districts_means_district_unit():
    p = _make(target_districts=["강남구", "마포구"])
    scope = analyze_textual_scope(p)
    assert ScopeUnit.DISTRICT in scope.scope_units
    assert scope.affected_districts == ["강남구", "마포구"]


def test_unknown_when_empty():
    p = _make()
    scope = analyze_textual_scope(p)
    assert scope.scope_units == [ScopeUnit.UNKNOWN]


class _StubReader:
    def __init__(self, dongs, pois, agents):
        self._d, self._p, self._a = dongs, pois, agents
    def dongs_in_districts(self, districts):
        return self._d
    def pois_in_dongs_for_categories(self, dongs, l1s):
        return self._p
    def agents_in_dongs_for_groups(self, dongs, groups):
        return self._a


def test_graph_reader_expands_nodes():
    p = _make(
        target_districts=["강남구"],
        benefit_categories=["식사"],
        target_groups=["청년"],
    )
    reader = _StubReader(dongs=["1168010100"], pois=["poi_1"], agents=["AGT_a"])
    scope = analyze_scope(p, reader)
    assert scope.affected_dongs == ["1168010100"]
    assert scope.affected_pois == ["poi_1"]
    assert scope.affected_agents == ["AGT_a"]


def test_seoul_wide_skips_graph_lookup():
    p = _make(target_districts=list(SEOUL_DISTRICTS))
    reader = _StubReader(dongs=["x"], pois=["y"], agents=["z"])
    scope = analyze_graph_scope(p, reader)
    assert scope.affected_dongs == []
    assert scope.affected_pois == []


def test_null_reader_returns_empty():
    p = _make(target_districts=["강남구"])
    scope = analyze_scope(p, NullGraphReader())
    assert scope.affected_dongs == []
    assert scope.affected_districts == ["강남구"]
