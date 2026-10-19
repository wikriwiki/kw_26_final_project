from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/persona"))
import build_rank_coupling as build
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/sim"))
import llm_client


@pytest.mark.parametrize("explicit,environment,expected", [
    (None, None, "exaone_4_5"), ("exaone_fp8", None, "exaone_fp8"),
    (None, "exaone", "exaone"), ("exaone_4_5", "exaone_fp8", "exaone_4_5"),
])
def test_summary_respects_registry_priority(monkeypatch, explicit, environment, expected):
    monkeypatch.delenv("LLM_MODE", raising=False)
    if environment:
        monkeypatch.setenv("LLM_MODE", environment)
    monkeypatch.setattr(build, "_build_persona_summary_prompt", lambda agent: "fixture")
    requests = []
    def call(mode, *args, **kwargs):
        requests.append(mode)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="요약"))])
    monkeypatch.setattr(llm_client, "call_chat", call)
    assert build.summarize_persona_llm({}, explicit) == "요약"
    assert requests == [expected]


def test_removed_model_does_not_silently_return_old_lifestyle(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "qwen8b")
    with pytest.raises(ValueError, match="Unknown LLM_MODE"):
        build.summarize_persona_llm({"personality": {"lifestyle": "원문"}})
