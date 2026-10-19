"""
version_registry.py
===================
버전 기반 stale 처리. 캐시 삭제 대신 context_version 을 +1 → 옛 키 자연 만료.

agent_persona 브랜치에서는 아직 캐시 store 가 없으므로 이 레지스트리는 향후
Redis L3 캐시·Celery 재생성 워커가 들어왔을 때 진실의 원천으로 쓰일 것.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "output" / "policy_pipeline" / "version_registry.json"


class VersionRegistry(BaseModel):
    policy_version: int = 1
    context_versions: dict[str, int] = Field(default_factory=dict)  # dong_code / L1 cat / agent_id
    summary_versions: dict[str, int] = Field(default_factory=dict)


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


def get_context_version(registry: VersionRegistry, entity_id: str) -> int:
    return registry.context_versions.get(entity_id, 1)


def bump_policy_version(registry: VersionRegistry) -> int:
    with _LOCK:
        registry.policy_version += 1
    return registry.policy_version


def bump_context_versions(registry: VersionRegistry, entity_ids: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    with _LOCK:
        for eid in entity_ids:
            nv = registry.context_versions.get(eid, 1) + 1
            registry.context_versions[eid] = nv
            out[eid] = nv
    return out
