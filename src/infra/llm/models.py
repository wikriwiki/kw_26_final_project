"""LLM mode, task, and per-model capability registry.

Single source of truth for which model is active and what it can do.
Mode is selected once via `LLM_MODE` env var; the rest of the codebase reads
through `resolve_mode()` / `get_profile()` so callers never hardcode a model.
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


PROFILES: dict[LLMMode, ModelProfile] = {
    LLMMode.EXAONE: ModelProfile(
        mode=LLMMode.EXAONE,
        display_name="EXAONE-4.5-33B-FP8",
        hf_id="LGAI-EXAONE/EXAONE-4.5-33B-FP8",
        context_window=262_144,
        supports_json_schema=True,
        max_running_requests=96,
        typical_prefix_tokens=3_500,
    ),
    LLMMode.QWEN: ModelProfile(
        mode=LLMMode.QWEN,
        display_name="Qwen3.5-4B",
        hf_id="Qwen/Qwen3.5-4B",
        context_window=262_144,
        supports_json_schema=True,
        max_running_requests=256,
        typical_prefix_tokens=3_500,
    ),
}


def resolve_mode(mode: str | LLMMode | None = None) -> LLMMode:
    """Resolve mode from explicit arg, then env var, then default.

    Order: arg > `LLM_MODE` env > "qwen". Case-insensitive.
    Raises ModeMismatchError for unknown values.
    """
    if isinstance(mode, LLMMode):
        return mode
    raw = (mode or os.getenv("LLM_MODE") or "qwen").lower()
    try:
        return LLMMode(raw)
    except ValueError as e:
        allowed = [m.value for m in LLMMode]
        raise ModeMismatchError(
            f"Invalid LLM_MODE={raw!r}. Allowed: {allowed}"
        ) from e


def get_profile(mode: str | LLMMode | None = None) -> ModelProfile:
    """Return the ModelProfile for the resolved mode."""
    return PROFILES[resolve_mode(mode)]
