"""LG is the active default; removed model selections cannot silently fall back."""
import asyncio
from datetime import date
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/sim"))
import llm_client as llm
from report_model import recorded_model_label


@pytest.fixture(autouse=True)
def isolated_model_selection(monkeypatch):
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.delenv("SIM_NO_SMOKING_MANIFEST", raising=False)
    monkeypatch.delenv("SIM_NO_SMOKING_ARM", raising=False)
    monkeypatch.delenv("SIM_JSON_GRAMMAR_MODE", raising=False)


def test_default_is_official_lg_awq():
    assert llm.resolve_mode() == "exaone_4_5"
    assert llm.get_spec().hf_id == "LGAI-EXAONE/EXAONE-4.5-33B-AWQ"
    assert {s.family for s in llm.MODELS.values()} == {"exaone", "midm"}


def test_recorded_request_timeout_is_used_without_sdk_retries(monkeypatch):
    observed = {}
    monkeypatch.setattr(llm, 'OpenAI', lambda **kwargs: observed.update(kwargs))
    monkeypatch.setenv('SIM_LLM_TIMEOUT_SECONDS', '600')
    llm.make_client('http://127.0.0.1:8000/v1')
    assert observed['timeout'] == 600.0
    assert observed['max_retries'] == 0
    monkeypatch.setenv('SIM_LLM_TIMEOUT_SECONDS', 'unbounded')
    with pytest.raises(ValueError, match='must be an integer'):
        llm.make_client('http://127.0.0.1:8000/v1')
    monkeypatch.setenv('SIM_LLM_TIMEOUT_SECONDS', '0')
    with pytest.raises(ValueError, match='between 30 and 1800'):
        llm.make_client('http://127.0.0.1:8000/v1')


@pytest.mark.parametrize("removed", ["qwen8b", "qwen9b", "qwen14b", "qwen32b", "qwen35_9b_awq",
                                    "qwen3_8b_awq", "qwen36_35b_a3b_awq", "qwen3_30b_a3b_awq"])
def test_removed_modes_reject_cli_and_environment_without_network(monkeypatch, removed):
    monkeypatch.setattr(llm, "get_client", lambda: pytest.fail("Rejected selection must not contact an endpoint"))
    with pytest.raises(ValueError, match="Unknown LLM_MODE"):
        llm.call_chat(removed, "system", "user")
    monkeypatch.setenv("LLM_MODE", removed)
    with pytest.raises(ValueError, match="Unknown LLM_MODE"):
        llm.call_chat(None, "system", "user")


def test_lg_requests_disable_thinking_and_keep_selected_model():
    requests = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: requests.append(kw))))
    llm.call_chat(None, "system", "user", client=client)
    assert requests[-1]["model"] == llm.get_spec().hf_id
    assert requests[-1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert llm._extra_body_for("midm") == {}


def test_recorded_json_object_mode_replaces_schema_at_dispatch(monkeypatch):
    requests = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: requests.append(kwargs))))
    schema = {'type': 'json_schema', 'json_schema': {'name': 'probe', 'schema': {'type': 'object'}}}
    monkeypatch.setenv('SIM_JSON_GRAMMAR_MODE', 'json_object')
    llm.call_chat(None, 'system', 'user', client=client, response_format=schema)
    assert requests[0]['response_format'] == {'type': 'json_object'}
    monkeypatch.setenv('SIM_JSON_GRAMMAR_MODE', 'json_schema')
    llm.call_chat(None, 'system', 'user', client=client, response_format=schema)
    assert requests[1]['response_format'] == schema


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-8B", "deepseek-ai/DeepSeek-R1", "unregistered/example"])
def test_direct_http_configs_must_use_known_model_id(model_id):
    with pytest.raises(ValueError, match="Unsupported model ID"):
        llm.require_supported_model_id(model_id)
    assert llm.require_supported_model_id(llm.get_spec().hf_id) == llm.get_spec().hf_id


def test_historical_config_cannot_dispatch_removed_model(tmp_path, monkeypatch):
    import validate_reasoning_prompt
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": "Qwen/Qwen3-8B"}))
    monkeypatch.setattr(sys, "argv", ["validate_reasoning_prompt", "--source", "unused.json",
                                     "--config", str(config), "--out", str(tmp_path / "run")])
    monkeypatch.setattr(validate_reasoning_prompt, "urlopen", lambda *a, **k: pytest.fail("network call"))
    with pytest.raises(ValueError, match="Unsupported model ID"):
        validate_reasoning_prompt.main()
    assert not (tmp_path / "run").exists()


def test_bdc_generator_uses_shared_lg_default_and_rejects_foreign_id():
    from scripts.bdc.generate_agents import call_vllm
    requests = []
    def respond(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=respond)))
    assert asyncio.run(call_vllm(client, "system", "user", 20)) == "[]"
    assert requests[-1]["model"] == llm.get_spec().hf_id
    assert requests[-1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    with pytest.raises(ValueError, match="Unregistered model"):
        asyncio.run(call_vllm(client, "system", "user", 20, model="Qwen/Qwen3-8B"))
    assert len(requests) == 1


def test_report_uses_recorded_models_instead_of_current_default(tmp_path):
    assert recorded_model_label(date(2017, 12, 3), 1, tmp_path) == "모델 실행 기록 없음"
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    records = [{"decision_provenance": {"model_id": model}} for model in ["archived/example", llm.get_spec().hf_id]]
    (metrics / "day_2017-12-03.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    result = recorded_model_label(date(2017, 12, 3), 1, tmp_path)
    assert "archived/example" in result and llm.get_spec().hf_id in result
