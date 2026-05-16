"""
version_registry.py
===================
Section 8 — 버전 기반 stale 처리.

캐시를 실제로 "삭제" 하지 않고, **버전을 올려 옛 키를 자연 expiration** 시키는
전략. 인프라 의존(Redis 등) 없이 JSON 파일 한 개로 동작.

세 종류 버전을 관리:
  - policy_version:  정책 자체의 글로벌 버전. 새 정책이 적용될 때마다 +1.
  - context_version[dong_id]:  특정 행정동의 컨텍스트 버전. 그 동 관련 무효화 시 +1.
  - context_version[industry_id]:  특정 업종 컨텍스트 버전. 그 업종 무효화 시 +1.
  - summary_version[dong_id]:  요약 재생성 시 +1 (rebuild 완료 표시용).

스레드 안전성: 파일 락 미사용. 본 파이프라인은 단일 워커가 직렬 처리하므로 OK.
다중 워커 환경에선 별도 잠금 필요.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "output" / "state" / "version_registry.json"


class VersionRegistry(BaseModel):
    """버전 카운터들의 단일 객체.

    Pydantic 으로 (de)serialize. 명시되지 않은 키는 1 로 본다 — `get_*()` 헬퍼 참고.
    """

    policy_version: int = 1
    context_versions: dict[str, int] = Field(default_factory=dict)  # dong_id 또는 industry_id
    summary_versions: dict[str, int] = Field(default_factory=dict)  # dong_id


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()


def load_registry(path: Path | None = None) -> VersionRegistry:
    path = path or DEFAULT_REGISTRY_PATH
    if not path.exists():
        return VersionRegistry()
    return VersionRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def save_registry(registry: VersionRegistry, path: Path | None = None) -> None:
    path = path or DEFAULT_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(registry.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 조회
# ---------------------------------------------------------------------------
def get_context_version(registry: VersionRegistry, entity_id: str) -> int:
    return registry.context_versions.get(entity_id, 1)


def get_summary_version(registry: VersionRegistry, dong_id: str) -> int:
    return registry.summary_versions.get(dong_id, 1)


# ---------------------------------------------------------------------------
# 증분 (무효화 의미)
# ---------------------------------------------------------------------------
def bump_policy_version(registry: VersionRegistry) -> int:
    with _LOCK:
        registry.policy_version += 1
    return registry.policy_version


def bump_context_versions(registry: VersionRegistry, entity_ids: list[str]) -> dict[str, int]:
    """주어진 엔티티들의 context_version 을 +1. 새로운 버전 dict 반환."""
    updated: dict[str, int] = {}
    with _LOCK:
        for eid in entity_ids:
            new_v = registry.context_versions.get(eid, 1) + 1
            registry.context_versions[eid] = new_v
            updated[eid] = new_v
    return updated


def bump_summary_versions(registry: VersionRegistry, dong_ids: list[str]) -> dict[str, int]:
    updated: dict[str, int] = {}
    with _LOCK:
        for did in dong_ids:
            new_v = registry.summary_versions.get(did, 1) + 1
            registry.summary_versions[did] = new_v
            updated[did] = new_v
    return updated
