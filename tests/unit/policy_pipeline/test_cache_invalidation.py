"""cache_keys / version_registry / invalidator."""

import pytest

from scripts.policy_pipeline.cache_keys import (
    agent_context_key,
    community_summary_key,
    industry_summary_key,
    parse_key,
    policy_context_key,
    seoul_wide_summary_key,
)
from scripts.policy_pipeline.invalidator import invalidate_for_scope
from scripts.policy_pipeline.scope import PolicyScope, ScopeUnit
from scripts.policy_pipeline.version_registry import (
    bump_context_versions,
    get_context_version,
    load_registry,
    save_registry,
)


def test_community_summary_key():
    assert community_summary_key("1168010100", 3) == "community_summary:1168010100:3"


def test_industry_key_with_korean_l1():
    assert industry_summary_key("식사", 2) == "industry_summary:식사:2"


def test_parse_key_three_part():
    p = parse_key("community_summary:1168010100:3")
    assert p.prefix == "community_summary"
    assert p.entity_id == "1168010100"
    assert p.version == 3


def test_parse_key_two_part():
    p = parse_key("policy_context:7")
    assert p.entity_id is None
    assert p.version == 7


def test_parse_key_invalid_raises():
    with pytest.raises(ValueError):
        parse_key("not_a_key")


def test_registry_roundtrip(tmp_path):
    path = tmp_path / "v.json"
    reg = load_registry(path)
    bump_context_versions(reg, ["dong1", "식사"])
    save_registry(reg, path)
    reloaded = load_registry(path)
    assert get_context_version(reloaded, "dong1") == 2
    assert get_context_version(reloaded, "식사") == 2


def test_invalidate_district_scope_minimal(tmp_path):
    scope = PolicyScope(
        policy_id="P007_test",
        scope_units=[ScopeUnit.DISTRICT, ScopeUnit.DONG, ScopeUnit.CATEGORY],
        affected_districts=["강남구"],
        affected_dongs=["dong1"],
        affected_categories=["식사"],
    )
    result = invalidate_for_scope(
        scope,
        registry_path=tmp_path / "v.json",
        log_path=tmp_path / "inv.jsonl",
    )
    assert "dong1" in result.bumped_dong_versions
    assert "식사" in result.bumped_category_versions
    assert any(k.startswith("community_summary:dong1:") for k in result.invalidated_keys)
    assert any(k.startswith("industry_summary:식사:") for k in result.invalidated_keys)


def test_invalidate_seoul_wide(tmp_path):
    scope = PolicyScope(
        policy_id="P007_test",
        scope_units=[ScopeUnit.SEOUL_WIDE],
    )
    result = invalidate_for_scope(
        scope,
        registry_path=tmp_path / "v.json",
        log_path=tmp_path / "inv.jsonl",
    )
    assert any(k.startswith("seoul_wide_summary:") for k in result.invalidated_keys)
    assert "seoul_wide" in result.invalidated_prefix_cache_groups
