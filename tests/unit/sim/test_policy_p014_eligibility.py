"""An explicit merchant-category ban must survive the presence of a code."""
import json
from pathlib import Path

from scripts.sim.eligibility import Rules, validated_restricted_rules
from scripts.sim import dawn_context


POLICY = json.loads((Path(__file__).resolve().parents[3] /
                     "data/neo4j_load/policies/P014.json").read_text(encoding="utf-8"))


def test_p014_excludes_department_stores_with_and_without_industry_code():
    rules = validated_restricted_rules(POLICY["eligibility"])
    for code in (None, "G99999"):
        eligible, _ = rules.eligible("롯데백화점 본점", "백화점", "쇼핑", code, True)
        assert not eligible
        eligible, _ = rules.eligible("면세업체", "면세점", "쇼핑", code, True)
        assert not eligible
        # The current POI taxonomy often maps a store to a broad retail sub.
        eligible, _ = rules.eligible("롯데백화점 본점", "종합소매", "마트", code, True)
        assert not eligible


def test_p014_retains_home_district_and_ordinary_shop_rules():
    rules = Rules(POLICY["eligibility"])
    assert rules.eligible("동네식당", "한식", "식사", "I10000", True)[0]
    assert not rules.eligible("동네식당", "한식", "식사", "I10000", False)[0]
    assert rules.eligible("국대떡볶이이마트목동점", "한식", "식사", "I10000", True)[0]
    assert rules.eligible("백화점약국", "약국", "건강", "G99999", True)[0]
    assert rules.eligible("이마트24 길음점", "편의점", "편의점", "G99999", True)[0]


def test_fallback_sub_rule_still_defers_to_known_code():
    rules = Rules({"mode": "exclude", "exclude": {"subs": {"other": ["백화점"]}}})
    assert rules.eligible("가게", "백화점", "쇼핑", "G99999", True)[0]
    assert not rules.eligible("가게", "백화점", "쇼핑", None, True)[0]


def test_all_candidate_queries_supply_industry_code_to_policy_rules():
    for query in (dawn_context.STAGE2_CANDIDATE_CYPHER,
                  dawn_context.STAGE2_FALLBACK_L1_DONG_CYPHER,
                  dawn_context.STAGE2_FALLBACK_L1_DISTRICT_CYPHER):
        assert "p.upjong_l3 AS upjong_l3" in query
