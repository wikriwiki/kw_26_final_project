"""
cache_keys.py
=============
캐시 키 포맷.

agent_persona 브랜치의 추후 L3 community summary 캐시 / SGLang prefix cache
무효화를 위한 키 네임스페이스. 현재 시뮬은 Neo4j 직접 조회라 캐시 미사용이지만,
docs/schedule_generation_plan/runtime_ontology.md 가 향후 Redis L3 캐시를 명시.
이 모듈이 그 자리에 들어갈 키 포맷을 미리 잡아둔다.
"""

from __future__ import annotations

from dataclasses import dataclass


KEY_SEP = ":"

PREFIX_COMMUNITY = "community_summary"      # 행정동 단위 요약
PREFIX_INDUSTRY = "industry_summary"        # 업종(L1 category) 단위
PREFIX_AGENT = "agent_context"
PREFIX_POLICY = "policy_context"
PREFIX_SEOUL = "seoul_wide_summary"


def community_summary_key(dong_code: str, context_version: int) -> str:
    return f"{PREFIX_COMMUNITY}{KEY_SEP}{dong_code}{KEY_SEP}{context_version}"


def industry_summary_key(l1_category: str, context_version: int) -> str:
    return f"{PREFIX_INDUSTRY}{KEY_SEP}{l1_category}{KEY_SEP}{context_version}"


def agent_context_key(agent_id: str, context_version: int) -> str:
    return f"{PREFIX_AGENT}{KEY_SEP}{agent_id}{KEY_SEP}{context_version}"


def policy_context_key(policy_version: int) -> str:
    return f"{PREFIX_POLICY}{KEY_SEP}{policy_version}"


def seoul_wide_summary_key(context_version: int) -> str:
    return f"{PREFIX_SEOUL}{KEY_SEP}{context_version}"


@dataclass(frozen=True)
class ParsedKey:
    prefix: str
    entity_id: str | None
    version: int


def parse_key(key: str) -> ParsedKey:
    parts = key.split(KEY_SEP)
    if len(parts) == 2:
        return ParsedKey(prefix=parts[0], entity_id=None, version=int(parts[1]))
    if len(parts) == 3:
        return ParsedKey(prefix=parts[0], entity_id=parts[1], version=int(parts[2]))
    raise ValueError(f"unrecognized cache key format: {key}")
