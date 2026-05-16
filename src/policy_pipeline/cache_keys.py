"""
cache_keys.py
=============
Section 8 — 캐시 키 설계.

키 포맷:
  community_summary:{dong_id}:{context_version}
  industry_summary:{industry_id}:{context_version}
  agent_context:{agent_id}:{context_version}
  policy_context:{policy_version}
  seoul_wide_summary:{context_version}   # 서울 전체 단위 요약(전체 무효화 대용)

context_version / policy_version 은 `version_registry.py` 가 관리한다.

본 모듈은 **키 생성과 식별만** 책임진다. 무효화 동작은 `invalidator.py` 가 한다.
"""

from __future__ import annotations

from dataclasses import dataclass


KEY_SEP = ":"

PREFIX_COMMUNITY = "community_summary"
PREFIX_INDUSTRY = "industry_summary"
PREFIX_AGENT = "agent_context"
PREFIX_POLICY = "policy_context"
PREFIX_SEOUL = "seoul_wide_summary"


# ---------------------------------------------------------------------------
# 키 생성
# ---------------------------------------------------------------------------
def community_summary_key(dong_id: str, context_version: int) -> str:
    return f"{PREFIX_COMMUNITY}{KEY_SEP}{dong_id}{KEY_SEP}{context_version}"


def industry_summary_key(industry_id: str, context_version: int) -> str:
    return f"{PREFIX_INDUSTRY}{KEY_SEP}{industry_id}{KEY_SEP}{context_version}"


def agent_context_key(agent_id: str, context_version: int) -> str:
    return f"{PREFIX_AGENT}{KEY_SEP}{agent_id}{KEY_SEP}{context_version}"


def policy_context_key(policy_version: int) -> str:
    return f"{PREFIX_POLICY}{KEY_SEP}{policy_version}"


def seoul_wide_summary_key(context_version: int) -> str:
    return f"{PREFIX_SEOUL}{KEY_SEP}{context_version}"


# ---------------------------------------------------------------------------
# 키 식별 / 파싱
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedKey:
    prefix: str
    entity_id: str | None  # policy_context / seoul_wide_summary 는 None
    version: int


def parse_key(key: str) -> ParsedKey:
    parts = key.split(KEY_SEP)
    if len(parts) == 2:
        # policy_context:{policy_version} 또는 seoul_wide_summary:{context_version}
        return ParsedKey(prefix=parts[0], entity_id=None, version=int(parts[1]))
    if len(parts) == 3:
        return ParsedKey(prefix=parts[0], entity_id=parts[1], version=int(parts[2]))
    raise ValueError(f"unrecognized cache key format: {key}")
