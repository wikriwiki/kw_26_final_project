"""LLM mode, task, and per-model capability registry.

Single source of truth for which model is active and what it can do.
Mode is selected once via `LLM_MODE` env var; the rest of the codebase reads
through `resolve_mode()` / `get_profile()` so callers never hardcode a model.

[한글 요약]
어떤 LLM 모델을 쓸지를 결정하는 "단일 진실 공급원". 환경변수 LLM_MODE 하나로
전체 코드베이스의 모델 선택이 바뀜. 호출자(phases/dawn 등)는 모델 이름을
직접 알 필요 없음 — resolve_mode()/get_profile()만 호출하면 됨.

왜 토글 방식인가:
1. A100 80GB 한 장에 두 모델을 동시에 띄우면 KV 캐시 풀이 분산되어 적중률 하락
2. 국내 대회는 EXAONE 단독 강제 → 코드 변경 없이 모드만 바꿔서 대응
3. 개발은 빠른 Qwen, 본번은 EXAONE — 두 환경의 모델 차이를 코드에서 신경 안 써도 됨
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

from .errors import ModeMismatchError


class LLMMode(StrEnum):
    """Active model family. Selected via `LLM_MODE` env var.

    Toggle-style: exactly one model is loaded into the sglang server at a time.
    Korean competition runs MUST use EXAONE.
    """

    EXAONE = "exaone"
    QWEN = "qwen"


class LLMTask(StrEnum):
    """Logical task category — drives prompt template selection and metrics labels.

    Tasks do NOT pick the model. The active `LLMMode` does.
    """

    PLAN_GENERATION = "plan_generation"
    INTENT_CLASSIFICATION = "intent_classification"
    INTERACTION_SUMMARY = "interaction_summary"
    POLICY_EXTRACTION = "policy_extraction"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Capabilities and runtime tuning for a single model."""

    mode: LLMMode
    display_name: str
    hf_id: str
    context_window: int
    supports_json_schema: bool
    max_running_requests: int
    typical_prefix_tokens: int


# 두 모델의 능력 카탈로그. 새 모델 추가 시 여기만 수정하면 됨.
# max_running_requests는 A100 80GB 기준 KV 캐시 풀 계산값:
#   - EXAONE 33B FP8: weights 33GB → KV 47GB → 토큰당 131KB(FP8) → 약 96 동시
#   - Qwen 4B: weights 8GB → KV 72GB → 훨씬 여유 → 256 동시 가능
PROFILES: dict[LLMMode, ModelProfile] = {
    LLMMode.EXAONE: ModelProfile(
        mode=LLMMode.EXAONE,
        display_name="EXAONE-4.5-33B-FP8",
        hf_id="LGAI-EXAONE/EXAONE-4.5-33B-FP8",
        context_window=262_144,         # 모델 자체 한도 256K, 실제 사용은 32K로 제한 (serve script)
        supports_json_schema=True,
        max_running_requests=96,        # A100 80GB 메모리 산정 기반
        typical_prefix_tokens=3_500,    # L1~L4 합산 추정치 (튜닝 참고용)
    ),
    LLMMode.QWEN: ModelProfile(
        mode=LLMMode.QWEN,
        display_name="Qwen3.5-4B",
        hf_id="Qwen/Qwen3.5-4B",
        context_window=262_144,
        supports_json_schema=True,
        max_running_requests=256,       # 4B는 weights 작아 동시성 여유
        typical_prefix_tokens=3_500,
    ),
}


def resolve_mode(mode: str | LLMMode | None = None) -> LLMMode:
    """Resolve mode from explicit arg, then env var, then default.

    Order: arg > `LLM_MODE` env > "qwen". Case-insensitive.
    Raises ModeMismatchError for unknown values.
    """
    # 우선순위: 함수 인자 > 환경변수 > 기본값(qwen)
    # 테스트에서 monkeypatch로 환경변수만 바꿔도 동작하게 함
    if isinstance(mode, LLMMode):
        return mode
    raw = (mode or os.getenv("LLM_MODE") or "qwen").lower()
    try:
        return LLMMode(raw)
    except ValueError as e:
        # 'exaone'/'qwen' 외 값이 들어오면 즉시 실패 — 오타로 인한 silent fallback 방지
        allowed = [m.value for m in LLMMode]
        raise ModeMismatchError(
            f"Invalid LLM_MODE={raw!r}. Allowed: {allowed}"
        ) from e


def get_profile(mode: str | LLMMode | None = None) -> ModelProfile:
    """Return the ModelProfile for the resolved mode."""
    # 현재 모드의 능력 카탈로그를 반환. 호출자는 이걸로 hf_id/한도 등을 알 수 있음.
    return PROFILES[resolve_mode(mode)]
