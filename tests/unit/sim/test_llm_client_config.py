from __future__ import annotations

from scripts.sim import llm_client as L


def test_k_exaone_mode_resolves_friendli_endpoint(monkeypatch):
    monkeypatch.setenv("LG_EXAONE_KEY", "flp_test")
    monkeypatch.setenv("K_EXAONE_ENDPOINT_ID", "endpoint-test")

    assert L.resolve_mode("exaone_api") == "k_exaone"

    spec = L.get_spec("k_exaone")
    cfg = L._client_config("k_exaone")

    assert cfg.base_url == L.FRIENDLI_DEDICATED_BASE_URL
    assert cfg.api_key == "flp_test"
    assert L._model_id_for(spec) == "endpoint-test"
    assert L._extra_body_for(spec.family) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_local_llm_mode_keeps_openai_compatible_dummy_key(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.delenv("SGLANG_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    cfg = L._client_config("qwen8b")

    assert cfg.base_url == "http://localhost:9999/v1"
    assert cfg.api_key == "EMPTY"
