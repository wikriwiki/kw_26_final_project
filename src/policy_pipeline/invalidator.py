"""
invalidator.py
==============
Section 8 — 캐시 무효화 + Prompt Prefix Cache 처리.

PolicyScope 를 받아 다음 두 가지를 한다:

  1. **버전 기반 stale 처리** — 영향 받은 행정동/업종/에이전트의 context_version 을
     +1 (`version_registry`). 전 삭제 대신 옛 버전을 자연 만료.
  2. **무효화 대상 키 열거** — 후행 시스템(prefix cache, summary worker 등)이
     쓸 키 리스트 산출. 이 리스트가 `PolicyScope.cache_keys_to_invalidate` 에 채워진다.

본 모듈은 **실제 캐시 store 와는 분리**된다. 키 리스트만 만든다. 실 store(Redis,
SGLang prefix cache 등)에 invalidation 명령을 보내는 것은 store 어댑터 책임.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from src.policy_pipeline.cache_keys import (
    agent_context_key,
    community_summary_key,
    industry_summary_key,
    policy_context_key,
    seoul_wide_summary_key,
)
from src.policy_pipeline.scope import PolicyScope, ScopeUnit
from src.policy_pipeline.version_registry import (
    DEFAULT_REGISTRY_PATH,
    VersionRegistry,
    bump_context_versions,
    bump_policy_version,
    get_context_version,
    load_registry,
    save_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVALIDATION_LOG_PATH = PROJECT_ROOT / "output" / "logs" / "cache_invalidation.jsonl"


class InvalidationResult(BaseModel):
    policy_id: str
    new_policy_version: int
    bumped_dong_versions: dict[str, int] = Field(default_factory=dict)
    bumped_industry_versions: dict[str, int] = Field(default_factory=dict)
    bumped_agent_versions: dict[str, int] = Field(default_factory=dict)
    invalidated_keys: list[str] = Field(default_factory=list)
    invalidated_prefix_cache_groups: list[str] = Field(default_factory=list)
    invalidated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# 메인 API
# ---------------------------------------------------------------------------
def invalidate_for_scope(
    scope: PolicyScope,
    *,
    registry_path: Path | None = None,
    log_path: Path | None = None,
) -> InvalidationResult:
    registry_path = registry_path or DEFAULT_REGISTRY_PATH
    log_path = log_path or DEFAULT_INVALIDATION_LOG_PATH
    """PolicyScope 에 따른 최소 무효화.

    동작:
      - SEOUL_WIDE → 모든 동·업종 키가 함께 묶이는 `seoul_wide_summary` 만 무효화
        (개별 동/업종 키는 자연 stale 됨). 이후 정책이 한 번 더 들어와도 동일.
      - 그 외 → 영향 받은 엔티티만 context_version 을 올린다.
    """
    registry = load_registry(registry_path)
    new_policy_version = bump_policy_version(registry)

    result = InvalidationResult(
        policy_id=scope.policy_id,
        new_policy_version=new_policy_version,
    )

    invalidated_keys: list[str] = [policy_context_key(new_policy_version)]
    prefix_groups: list[str] = ["policy_context"]

    if ScopeUnit.SEOUL_WIDE in scope.scope_units:
        seoul_v = bump_context_versions(registry, ["seoul"])["seoul"]
        invalidated_keys.append(seoul_wide_summary_key(seoul_v))
        prefix_groups.append("seoul_wide")
    else:
        # 동별 무효화
        if scope.affected_dongs:
            bumped_dongs = bump_context_versions(registry, scope.affected_dongs)
            result.bumped_dong_versions = bumped_dongs
            invalidated_keys.extend(
                community_summary_key(did, v) for did, v in bumped_dongs.items()
            )
            prefix_groups.append("community")

        # 업종별 무효화
        if scope.affected_industries:
            bumped_inds = bump_context_versions(registry, scope.affected_industries)
            result.bumped_industry_versions = bumped_inds
            invalidated_keys.extend(
                industry_summary_key(ind, v) for ind, v in bumped_inds.items()
            )
            prefix_groups.append("industry")

        # 에이전트 컨텍스트 무효화
        if scope.affected_agents:
            bumped_agents = bump_context_versions(registry, scope.affected_agents)
            result.bumped_agent_versions = bumped_agents
            invalidated_keys.extend(
                agent_context_key(aid, v) for aid, v in bumped_agents.items()
            )
            prefix_groups.append("agent")

    result.invalidated_keys = invalidated_keys
    result.invalidated_prefix_cache_groups = sorted(set(prefix_groups))

    save_registry(registry, registry_path)
    _append_invalidation_log(result, log_path)

    # scope 객체에도 결과를 채워준다 — pipeline 이 그대로 전달 가능하도록.
    scope.cache_keys_to_invalidate = list(invalidated_keys)

    return result


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _append_invalidation_log(result: InvalidationResult, log_path: Path) -> None:
    import json
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")
