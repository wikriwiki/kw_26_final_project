"""Public API for `src.infra.llm`.

Callers should import from this module only — internal layout may change.
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
