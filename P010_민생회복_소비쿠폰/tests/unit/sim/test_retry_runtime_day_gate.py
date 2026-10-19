import hashlib
import importlib.util
from datetime import date
from functools import wraps
from pathlib import Path
import sys
from types import SimpleNamespace
import json

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/sim'))
PATCH = ROOT / 'deploy/vast/runtime_hotfix_v22_retry_contract'


def test_replacement_is_day_gated_and_separate_from_frozen_provenance(monkeypatch):
    import stage1_intent as s1
    import stage2_poi as s2
    import grounded_schema as schema
    import agent_day_store as store
    spec = importlib.util.spec_from_file_location('retry_day_gate_test', PATCH / 'runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    # Register originals with monkeypatch so installation is fully undone.
    for mod, name in ((s1, 'call_stage1'), (s2, 'call_stage2'), (s2, 'call_stage1'),
                      (schema, 'rejected_response_feedback'), (store, 'save_result')):
        monkeypatch.setattr(mod, name, getattr(mod, name))
    @wraps(s1.call_stage1)
    def old_stage1(*a, **k):
        return 'old-day', {'s1_timing': {}}
    monkeypatch.setattr(s1, 'call_stage1', old_stage1)
    monkeypatch.setattr(store, 'save_result', lambda tx, result: result)
    raw = (PATCH / 'manifest.json').read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    runtime.install(PATCH, sha)
    assert s1.call_stage1('A', date(2017, 11, 23))[0] == 'old-day'
    monkeypatch.setenv('SIM_PROMPT_VARIANT', 'no_smoking_v1')
    monkeypatch.setattr(s1, '_format_dawn_blocks', lambda *a: '[E0001] 오늘 집에서 쉰다')
    content = json.dumps({'events': [{'time': '08:00', 'anchor': 'residence', 'category': '집',
        'intent': '휴식', 'reasoning': '오늘 집에서 쉰다.', 'trigger': 'none', 'evidence_ref': 'E0001'}]})
    monkeypatch.setattr(s1, '_llm_call', lambda *a, **k: SimpleNamespace(model='stub',
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5),
        choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=content))]))
    result, meta = s1.call_stage1('A', date(2017, 11, 24),
        ctx=SimpleNamespace(persona={}, prompt_timing={}), max_retry=0, log_failures=False)
    assert meta['s1_timing']['retry_contract_sha256'] == sha
    assert result.events[0].evidence_ref == 'E0001'
    old_row = {'experience_day': '2017-11-23'}
    assert store.save_result(None, old_row) == old_row
    new_row = {'experience_day': '2017-11-24'}
    sealed = store.save_result(None, new_row)
    assert sealed['retry_contract_sha256'] == sha
    assert 'retry_contract_sha256' not in new_row
