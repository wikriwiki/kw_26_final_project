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

from . import v1, v2, v3, v4, v5, v6  # noqa: E402
from . import v7, v8, v9  # noqa: E402  (2차 후보, 사전등록 2026-09-18)
from . import v10  # noqa: E402  (validation_v3: independent candidate)
from . import v11  # noqa: E402  (validation_v4: universal grounded planning)
from . import v12  # noqa: E402  (validation_v4b: temporal construction)
from . import v14  # noqa: E402  (relative-time representation)
from . import v15  # noqa: E402  (concise neutral planning)
from . import v16  # noqa: E402  (separate observed facts from chosen activities)
from . import v17  # noqa: E402  (first-person grounded choices)
from . import v18  # noqa: E402  (verbatim evidence protocol)

# p010 은 동결된 역사 기록, p012 는 캐시백 전용 과도기판.
# v1~v6 이 1차 기전 중립 후보, v5 가 그 중 홀드아웃까지 마친 확정판이다.
# v7~v9 는 2차 후보 — 훈련 정책에서 드러난 결함 둘을 겨냥하는 문장을 더한다.
_VARIANTS: dict[str, ModuleType] = {
    "p010": p010, "p012": p012,
    "v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5, "v6": v6,
    "v7": v7, "v8": v8, "v9": v9,
    "v10": v10,
    "v11": v11,
    "v12": v12,
    "v14": v14,
    "v15": v15,
    "v16": v16,
    "v17": v17,
    "v18": v18,
}
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
