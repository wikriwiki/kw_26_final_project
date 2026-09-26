"""The 2020 emergency grant uses its own declared merchant rule in runtime and scoring."""
import json
from pathlib import Path

from scripts.sim.eligibility import validated_restricted_rules


ROOT = Path(__file__).resolve().parents[3]
RELATIVE_POLICY = "data/experiments/P013_v53_policy_20260926.json"
POLICY = json.loads((ROOT / RELATIVE_POLICY).read_text(encoding="utf-8"))


def test_experimental_copy_keeps_original_policy_intact():
    original = json.loads((ROOT / "data/neo4j_load/policies/P013.json").read_text(encoding="utf-8"))
    assert original["effective_from"] == "2020-05-13"
    assert "eligibility" not in original
    for key in ("id", "type", "decile_grants", "effective_until", "poi_restricted"):
        assert POLICY[key] == original[key]
    assert POLICY["effective_from"] == "2020-05-11"
    assert POLICY["eligible_marker"] == "[사용가능]"


def test_2020_rule_excludes_large_and_vice_but_not_every_chain():
    rules = validated_restricted_rules(POLICY["eligibility"])
    assert not rules.eligible("롯데마트 서울점", "종합소매", "마트", "G99999", False)[0]
    assert not rules.eligible("롯데백화점 서울점", "종합소매", "마트", "G99999", False)[0]
    assert not rules.eligible("보통 주점", "일반주점", "주점", "I21101", True)[0]
    assert rules.eligible("스타벅스 서울점", "카페", "카페", "I21201", False)[0]
    assert rules.eligible("동네식당", "한식", "식사", "I10000", False)[0]


def test_scorer_uses_the_experimental_policy_rule_not_2025_coupon_rule():
    from scripts.sim import score_policy

    rows = [
        {"pname": "스타벅스 서울점", "sub": "카페", "l1": "카페", "upjong_l3": "I21201",
         "pdong": "11680670", "hdong": "11680790", "elig": False},
        {"pname": "롯데마트 서울점", "sub": "종합소매", "l1": "마트", "upjong_l3": "G99999",
         "pdong": "11680670", "hdong": "11680790", "elig": True},
    ]
    description = score_policy.apply_policy_eligibility(rows, RELATIVE_POLICY)
    assert "적격 규칙(exclude)" in description
    assert [row["elig"] for row in rows] == [True, False]
