import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
import prompt_budget as budget


def test_optional_guard_preserves_nonexperiment_calls(monkeypatch):
    monkeypatch.delenv('SIM_PROMPT_TOKEN_GUARD', raising=False)
    assert budget.check_request_budget({}) is None


def test_exact_context_guard_includes_output_reservation(monkeypatch):
    monkeypatch.setenv('SIM_PROMPT_TOKEN_GUARD', 'required')
    monkeypatch.setenv('SIM_TOKENIZER_PATH', 'test-fixture-only')
    monkeypatch.setenv('SIM_MODEL_CONTEXT_LENGTH', '1000')
    monkeypatch.setattr(budget, 'tokenizer_manifest', lambda: {'model': 'LG-fixture'})
    monkeypatch.setattr(budget, 'load_tokenizer', lambda path: SimpleNamespace(apply_chat_template=lambda *a, **k: [0] * 700))
    request = {'model': 'LG-fixture', 'messages': [], 'max_tokens': 100}
    assert budget.check_request_budget(request)['input_tokens'] == 700
    with pytest.raises(ValueError, match='exceeds'):
        budget.check_request_budget(dict(request, max_tokens=173))
    with pytest.raises(ValueError, match='identity'):
        budget.check_request_budget(dict(request, model='unmatched'))


def test_guard_counts_token_ids_not_batch_encoding_fields(monkeypatch):
    monkeypatch.setenv('SIM_PROMPT_TOKEN_GUARD', 'required')
    monkeypatch.setenv('SIM_TOKENIZER_PATH', 'test-fixture-only')
    monkeypatch.setenv('SIM_MODEL_CONTEXT_LENGTH', '1000')
    monkeypatch.setattr(budget, 'tokenizer_manifest', lambda: {'model': 'LG-fixture'})
    calls = []

    def encode(*args, **kwargs):
        calls.append(kwargs)
        return {'input_ids': [1] * 700, 'attention_mask': [1] * 700}

    monkeypatch.setattr(budget, 'load_tokenizer', lambda path: SimpleNamespace(apply_chat_template=encode))
    request = {'model': 'LG-fixture', 'messages': [], 'max_tokens': 100}
    assert budget.check_request_budget(request)['input_tokens'] == 700
    assert calls[0]['return_dict'] is False
    with pytest.raises(ValueError, match='exceeds'):
        budget.check_request_budget(dict(request, max_tokens=173))


def test_modified_tokenizer_is_rejected_before_loading(tmp_path, monkeypatch):
    (tmp_path / 'tokenizer.json').write_bytes(b'fixture')
    manifest = {'files': {'tokenizer.json': hashlib.sha256(b'fixture').hexdigest()}}
    monkeypatch.setattr(budget, 'tokenizer_manifest', lambda: manifest)
    assert budget.verify_tokenizer_files(tmp_path) == manifest
    (tmp_path / 'tokenizer.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs'):
        budget.verify_tokenizer_files(tmp_path)


def test_required_guard_never_accepts_missing_tokenizer(monkeypatch):
    monkeypatch.setenv('SIM_PROMPT_TOKEN_GUARD', 'required')
    monkeypatch.delenv('SIM_TOKENIZER_PATH', raising=False)
    with pytest.raises(ValueError, match='SIM_TOKENIZER_PATH'):
        budget.check_request_budget({})
