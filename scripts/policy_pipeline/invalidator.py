"""
invalidator.py
==============
PolicyScope → version bump → 무효화 키 리스트.

agent_persona 브랜치는 아직 L3 캐시 store 가 없으므로 이 모듈은 **scaffolding**.
키 리스트와 prefix 그룹 라벨만 산출 — 실제 store 어댑터(Redis 등)가 들어오면
산출물을 그대로 소비하면 됨.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from scripts.policy_pipeline.cache_keys import (
    agent_context_key,
    community_summary_key,
    industry_summary_key,
    policy_context_key,
    seoul_wide_summary_key,
)
from scripts.policy_pipeline.scope import PolicyScope, ScopeUnit
from scripts.policy_pipeline.version_registry import (
    DEFAULT_REGISTRY_PATH,
    bump_context_versions,
    bump_policy_version,
    load_registry,
    save_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVALIDATION_LOG_PATH = (
    PROJECT_ROOT / "output" / "policy_pipeline" / "cache_invalidation.jsonl"
)


class InvalidationResult(BaseModel):
    policy_id: str
    new_policy_version: int
    bumped_dong_versions: dict[str, int] = Field(default_factory=dict)
    bumped_category_versions: dict[str, int] = Field(default_factory=dict)
    bumped_agent_versions: dict[str, int] = Field(default_factory=dict)
    invalidated_keys: list[str] = Field(default_factory=list)
    invalidated_prefix_cache_groups: list[str] = Field(default_factory=list)
    invalidated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def invalidate_for_scope(
    scope: PolicyScope,
    *,
    registry_path: Path | None = None,
    log_path: Path | None = None,
) -> InvalidationResult:
    registry_path = registry_path or DEFAULT_REGISTRY_PATH
    log_path = log_path or DEFAULT_INVALIDATION_LOG_PATH

    registry = load_registry(registry_path)
    new_policy_version = bump_policy_version(registry)

    result = InvalidationResult(
        policy_id=scope.policy_id,
        new_policy_version=new_policy_version,
    )
    keys: list[str] = [policy_context_key(new_policy_version)]
    groups: list[str] = ["policy_context"]

    if ScopeUnit.SEOUL_WIDE in scope.scope_units:
        seoul_v = bump_context_versions(registry, ["seoul"])["seoul"]
        keys.append(seoul_wide_summary_key(seoul_v))
        groups.append("seoul_wide")
    else:
        if scope.affected_dongs:
            bumped = bump_context_versions(registry, scope.affected_dongs)
            result.bumped_dong_versions = bumped
            keys.extend(community_summary_key(d, v) for d, v in bumped.items())
            groups.append("community")

        if scope.affected_categories:
            bumped = bump_context_versions(registry, scope.affected_categories)
            result.bumped_category_versions = bumped
            keys.extend(industry_summary_key(c, v) for c, v in bumped.items())
            groups.append("industry")

        if scope.affected_agents:
            bumped = bump_context_versions(registry, scope.affected_agents)
            result.bumped_agent_versions = bumped
            keys.extend(agent_context_key(a, v) for a, v in bumped.items())
            groups.append("agent")

    result.invalidated_keys = keys
    result.invalidated_prefix_cache_groups = sorted(set(groups))

    save_registry(registry, registry_path)
    _append_log(result, log_path)

    scope.cache_keys_to_invalidate = list(keys)
    return result


def _append_log(result: InvalidationResult, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")
