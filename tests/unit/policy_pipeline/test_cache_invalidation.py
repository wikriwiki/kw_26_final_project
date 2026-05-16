"""cache_keys / version_registry / invalidator 통합 테스트."""

import pytest

from src.policy_pipeline.cache_keys import (
    agent_context_key,
    community_summary_key,
    industry_summary_key,
    parse_key,
    policy_context_key,
    seoul_wide_summary_key,
)
from src.policy_pipeline.invalidator import invalidate_for_scope
from src.policy_pipeline.scope import PolicyScope, ScopeUnit
from src.policy_pipeline.version_registry import (
    VersionRegistry,
    bump_context_versions,
    get_context_version,
    load_registry,
    save_registry,
)


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------
def test_community_summary_key_format():
    assert community_summary_key("1168010100", 3) == "community_summary:1168010100:3"


def test_industry_key_format():
    assert industry_summary_key("음식점", 2) == "industry_summary:음식점:2"


def test_agent_key_format():
    assert agent_context_key("AGT_a", 5) == "agent_context:AGT_a:5"


def test_policy_context_key_format():
    assert policy_context_key(7) == "policy_context:7"


def test_seoul_wide_key_format():
    assert seoul_wide_summary_key(4) == "seoul_wide_summary:4"


def test_parse_key_three_part():
    p = parse_key("community_summary:1168010100:3")
    assert p.prefix == "community_summary"
    assert p.entity_id == "1168010100"
    assert p.version == 3


def test_parse_key_two_part():
    p = parse_key("policy_context:7")
    assert p.entity_id is None
    assert p.version == 7


def test_parse_key_rejects_bad_format():
    with pytest.raises(ValueError):
        parse_key("not_a_key")


# ---------------------------------------------------------------------------
# Version registry (uses tmp_path so no global pollution)
# ---------------------------------------------------------------------------
def test_get_context_version_default_one(tmp_path):
    reg = load_registry(tmp_path / "v.json")
    assert get_context_version(reg, "unknown_dong") == 1


def test_bump_context_versions_idempotent_save_roundtrip(tmp_path):
    path = tmp_path / "v.json"
    reg = load_registry(path)
    bump_context_versions(reg, ["dong1", "dong2"])
    save_registry(reg, path)

    reloaded = load_registry(path)
    assert get_context_version(reloaded, "dong1") == 2
    assert get_context_version(reloaded, "dong2") == 2


# ---------------------------------------------------------------------------
# Invalidator
# ---------------------------------------------------------------------------
def test_invalidate_district_scope_bumps_only_affected(tmp_path):
    registry_path = tmp_path / "v.json"
    log_path = tmp_path / "inv.jsonl"

    scope = PolicyScope(
        policy_id="policy_abc",
        scope_units=[ScopeUnit.DISTRICT, ScopeUnit.DONG, ScopeUnit.INDUSTRY],
        affected_districts=["강남구"],
        affected_dongs=["d1", "d2"],
        affected_industries=["음식점"],
    )
    result = invalidate_for_scope(scope, registry_path=registry_path, log_path=log_path)

    assert set(result.bumped_dong_versions.keys()) == {"d1", "d2"}
    assert "음식점" in result.bumped_industry_versions
    assert any(k.startswith("community_summary:d1:") for k in result.invalidated_keys)
    assert any(k.startswith("industry_summary:음식점:") for k in result.invalidated_keys)
    # 영향 받지 않은 동은 bump 안 됨
    reg = load_registry(registry_path)
    assert get_context_version(reg, "d_other") == 1


def test_invalidate_seoul_wide_uses_seoul_key(tmp_path):
    registry_path = tmp_path / "v.json"
    log_path = tmp_path / "inv.jsonl"

    scope = PolicyScope(
        policy_id="policy_abc",
        scope_units=[ScopeUnit.SEOUL_WIDE],
        affected_regions=["서울"],
    )
    result = invalidate_for_scope(scope, registry_path=registry_path, log_path=log_path)
    assert any(k.startswith("seoul_wide_summary:") for k in result.invalidated_keys)
    assert "seoul_wide" in result.invalidated_prefix_cache_groups


def test_invalidate_updates_scope_object(tmp_path):
    scope = PolicyScope(
        policy_id="policy_abc",
        scope_units=[ScopeUnit.DISTRICT, ScopeUnit.DONG],
        affected_dongs=["d1"],
    )
    invalidate_for_scope(
        scope,
        registry_path=tmp_path / "v.json",
        log_path=tmp_path / "inv.jsonl",
    )
    # scope.cache_keys_to_invalidate 가 채워졌는지
    assert scope.cache_keys_to_invalidate
    assert any("community_summary:d1:" in k for k in scope.cache_keys_to_invalidate)
