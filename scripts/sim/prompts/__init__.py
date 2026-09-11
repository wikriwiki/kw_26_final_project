# -*- coding: utf-8 -*-
"""정책별 소비행동 프롬프트 변형.

정책마다 소비 기전이 다르다. 소비쿠폰(P010)은 정책지갑에서 꺼내 쓰는 것이고
상생소비지원금(P012)은 실적을 쌓아 다음 달에 돌려받는 것이다. 기전이 다르면
프롬프트도 달라져야 하는데, 한 파일을 공유하면 한쪽을 고칠 때 다른 쪽의
재현성이 깨진다.

그래서 정책군마다 프롬프트 모듈을 따로 둔다.

    p010  민생회복 소비쿠폰 — BOK 대조 검증본. 동결. 골든 테스트가 감시한다.
    p012  상생소비지원금 — 현재 p010 과 동일한 본문. 자유롭게 수정 가능.

선택은 환경변수 `SIM_PROMPT_VARIANT` 로 한다. 지정하지 않으면 p010 이라
기존 실행 경로는 그대로다.

    SIM_PROMPT_VARIANT=p012 python scripts/sim/run_simulation.py ...
"""
from __future__ import annotations

import os
from types import ModuleType

from . import p010, p012

_VARIANTS: dict[str, ModuleType] = {"p010": p010, "p012": p012}
DEFAULT = "p010"


def list_variants() -> list[str]:
    return sorted(_VARIANTS)


def get(name: str | None = None) -> ModuleType:
    """프롬프트 변형 모듈. name 이 없으면 환경변수, 그것도 없으면 기본값."""
    key = (name or os.environ.get("SIM_PROMPT_VARIANT") or DEFAULT).strip()
    mod = _VARIANTS.get(key)
    if mod is None:
        raise KeyError(f"알 수 없는 프롬프트 변형: {key} (가능: {list_variants()})")
    return mod


def active_name() -> str:
    return (os.environ.get("SIM_PROMPT_VARIANT") or DEFAULT).strip()
