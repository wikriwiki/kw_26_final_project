"""Public API for `src.infra.llm`.

Callers should import from this module only — internal layout may change.

[한글 요약 — 호출자가 알아야 할 4가지]
1. EngineClient        — sglang 서버에 1:1 연결. 보통 시뮬레이션 시작 시 1개 만들고 끝까지 재사용.
2. PromptLayers        — 7층 프롬프트 데이터 클래스. 모든 LLM 호출은 이걸 만들어서 넘김.
3. BatchController     — 6만 명 요청을 묶어서 효율적으로 처리. group_key로 행정동 정렬.
4. generate_structured — Pydantic 스키마 기반 JSON 응답. Plan 생성 등에 사용.

나머지(errors, metrics, warmup 등)는 보조 기능. 자세한 흐름은 각 파일 docstring 참고.
"""
from .batch import BatchController
from .engine_client import EngineClient
from .errors import (
    EngineHTTPError,
    EngineTimeoutError,
    LLMError,
    ModeMismatchError,
    SchemaParseError,
)
from .metrics import EngineMetrics, diff_metrics, scrape_engine_metrics
from .models import PROFILES, LLMMode, LLMTask, ModelProfile, get_profile, resolve_mode
from .prompt_layers import PromptLayers, empty_layers
from .structured import build_response_format, generate_structured
from .warmup import prewarm

__all__ = [
    "BatchController",
    "EngineClient",
    "EngineHTTPError",
    "EngineMetrics",
    "EngineTimeoutError",
    "LLMError",
    "LLMMode",
    "LLMTask",
    "ModelProfile",
    "ModeMismatchError",
    "PROFILES",
    "PromptLayers",
    "SchemaParseError",
    "build_response_format",
    "diff_metrics",
    "empty_layers",
    "generate_structured",
    "get_profile",
    "prewarm",
    "resolve_mode",
    "scrape_engine_metrics",
]
