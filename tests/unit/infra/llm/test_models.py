"""Tests for mode resolution and model profile registry."""
from __future__ import annotations

import pytest

from src.infra.llm import PROFILES, LLMMode, get_profile, resolve_mode
from src.infra.llm.errors import ModeMismatchError


def test_resolve_mode_from_string() -> None:
    assert resolve_mode("exaone") == LLMMode.EXAONE
    assert resolve_mode("qwen") == LLMMode.QWEN


def test_resolve_mode_is_case_insensitive() -> None:
    assert resolve_mode("EXAONE") == LLMMode.EXAONE
    assert resolve_mode("Qwen") == LLMMode.QWEN


def test_resolve_mode_passes_enum_through() -> None:
    assert resolve_mode(LLMMode.EXAONE) is LLMMode.EXAONE


def test_resolve_mode_invalid_raises_mode_mismatch() -> None:
    with pytest.raises(ModeMismatchError):
        resolve_mode("gpt4")


def test_resolve_mode_defaults_to_qwen_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_MODE", raising=False)
    assert resolve_mode(None) == LLMMode.QWEN


def test_resolve_mode_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODE", "exaone")
    assert resolve_mode(None) == LLMMode.EXAONE


def test_resolve_mode_arg_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODE", "qwen")
    assert resolve_mode("exaone") == LLMMode.EXAONE


def test_profiles_exist_for_both_modes() -> None:
    assert LLMMode.EXAONE in PROFILES
    assert LLMMode.QWEN in PROFILES


def test_exaone_profile_specs() -> None:
    p = get_profile(LLMMode.EXAONE)
    assert p.hf_id == "LGAI-EXAONE/EXAONE-4.5-33B-FP8"
    assert p.context_window == 262_144
    assert p.supports_json_schema is True
    assert p.max_running_requests > 0


def test_qwen_profile_specs() -> None:
    p = get_profile(LLMMode.QWEN)
    assert p.hf_id == "Qwen/Qwen3.5-4B"
    assert p.context_window == 262_144
    assert p.supports_json_schema is True
    assert p.max_running_requests > 0


def test_profile_is_frozen() -> None:
    p = get_profile(LLMMode.QWEN)
    with pytest.raises(AttributeError):
        p.context_window = 1000  # type: ignore[misc]
