import json
from datetime import date
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
import interview_evidence as evidence
import night_recovery as recovery
from evidence_integrity import digest, seal, verify
from night_completion import validate_accounting

DAY = '2017-11-23'
DATA = {'simulation_day': DAY, 'a': {}, 'b': {}}


def response():
    return SimpleNamespace(id='stub', model='stub', choices=[SimpleNamespace(index=0,
        finish_reason='stop', message=SimpleNamespace(content='{"evidence_ref":"E0001,E0002"}'))],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5))


@pytest.fixture
def configured(tmp_path, monkeypatch):
    import no_smoking_context
    import experience_provenance
    runtime = SimpleNamespace(arm='off', agent_ids=['a', 'b'])
    monkeypatch.setattr(no_smoking_context, 'configured_context', lambda: runtime)
    monkeypatch.setattr(experience_provenance, 'source_fingerprint', lambda: 'a' * 64)
    monkeypatch.setenv('SIM_OUTPUT_DIR', str(tmp_path))
    monkeypatch.setenv('SIM_RUN_ID', 'run')
    yield tmp_path
    evidence.clear_evidence()


def journal(root, count, success=False):
    token = evidence.begin_evidence(root, 'run', 'off', DAY, ['a', 'b'], digest(['a', 'b']),
                                    'a' * 64, context=DATA)
    evidence.set_evidence_stage('night_intent')
    result = {'initiator_id': 'a', 'recipient_id': 'b', 'intent': '기타'}
    try:
        for _ in range(count):
            evidence.record_chat_call({'messages': [{'role': 'user', 'content': 'input'}]}, response)
        if success:
            result['interview_evidence'] = evidence.archive_interaction(result)
        return result
    finally:
        evidence.clear_evidence(token)


def test_success_is_reused_without_model_call(configured):
    expected = journal(configured, 1, success=True)
    actual = recovery.classify_recoverable(lambda *a, **k: pytest.fail('must reuse'), ('a', 'b'), DATA)
    assert actual == expected


def test_exhausted_pair_skips_and_preserves_invalid_output(configured):
    journal(configured, 6)
    actual = recovery.classify_recoverable(lambda *a, **k: pytest.fail('seventh call'), ('a', 'b'), DATA)
    assert actual['status'] == 'skipped' and actual['observed_interaction'] is False
    recovery.verify_skipped(configured, actual)
    assert actual['attempts'] == 6
    again = recovery.classify_recoverable(lambda *a, **k: pytest.fail('repeat exhausted'), ('a', 'b'), DATA)
    assert again == actual
    index = evidence.commit_night_evidence(configured, 'run', 'off', DAY, [], skipped=[actual])
    marker = seal({'cohort': ['a', 'b'], 'run_id': 'run', 'arm': 'off', 'day': DAY,
                   'status': 'complete', 'evidence_ref': index, 'conversation_count': 0})
    (configured / f'night2_completed_{DAY}.json').write_text(json.dumps(marker), encoding='utf-8')
    items, present = evidence._night_items(configured, 'a', DAY, ('run', 'off', digest(['a', 'b']), 'a'*64))
    assert present and [r['kind'] for r in items] == ['skipped_social_interaction']
    assert 'E0001,E0002' in (configured / actual['journals'][0]['path']).read_text(encoding='utf-8')


def test_restart_consumes_only_remaining_call(configured):
    journal(configured, 5)
    called = []
    def classifier(pair, data, max_retry):
        called.append((max_retry, recovery.ATTEMPT_OFFSET.get()))
        journal(configured, max_retry + 1)
        return {'error': 'still invalid'}
    result = recovery.classify_recoverable(classifier, ('a', 'b'), DATA)
    assert called == [(0, 5)] and result['attempts'] == 6


def test_foreign_context_and_changed_journal_are_rejected(configured):
    journal(configured, 6)
    with pytest.raises(ValueError, match='context'):
        recovery.classify_recoverable(None, ('a', 'b'), dict(DATA, changed=True))
    result = recovery.classify_recoverable(None, ('a', 'b'), DATA)
    path = configured / result['journals'][0]['path']
    path.write_text(path.read_text(encoding='utf-8').replace('input', 'altered'), encoding='utf-8')
    with pytest.raises(ValueError):
        recovery.verify_skipped(configured, result)


@pytest.mark.parametrize('ok,skip,total', [(771,1,772), (0,2,2), (2,0,2), (0,0,0)])
def test_accounting_accepts_explicit_terminal_skips(ok, skip, total):
    validate_accounting({'processed': ok, 'skipped': skip, 'matched': total}, total)


@pytest.mark.parametrize('stats', [{'processed':1,'skipped':0}, {'processed':1,'skipped':True},
    {'processed':1,'skipped':1,'errors':1}, {'processed':1,'skipped':1,'matched':3}])
def test_unaccounted_or_false_skip_is_rejected(stats):
    with pytest.raises(ValueError):
        validate_accounting(stats, 2)


def test_all_skipped_transaction_still_writes_night_outbox(monkeypatch):
    import night_intent_llm as night
    import night_store
    import no_smoking_context
    class Tx:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def commit(self): actions.append('commit')
        def begin_transaction(self): return self
    actions = []
    monkeypatch.setattr(night, 'driver_session', lambda: Tx())
    monkeypatch.setattr(no_smoking_context, 'configured_context', lambda: SimpleNamespace(arm='off'))
    monkeypatch.setattr(evidence, 'commit_night_evidence', lambda *a, **k: actions.append(k) or {'ref':1})
    monkeypatch.setattr(night_store, 'save', lambda tx, day, runtime, stats: actions.append(stats))
    result = night.write_conversations(date.fromisoformat(DAY), [], skipped=[{'status':'skipped'}])
    assert result['created'] == 0 and result['evidence_ref'] == {'ref':1}
    assert actions[1]['processed'] == 0 and actions[1]['skipped'] == 1 and actions[-1] == 'commit'


def test_night_answer_evidence_ref_is_zero_padded_like_the_live_runtime():
    import json
    import night_intent_llm
    raw = 'note {"intent": "기타", "evidence_ref": "E12", "reason": "r"} trailing'
    parsed = json.loads(night_intent_llm._extract_first_json(raw))
    assert parsed["evidence_ref"] == "E0012"
    kept = json.loads(night_intent_llm._extract_first_json('{"evidence_ref": "E0007"}'))
    assert kept["evidence_ref"] == "E0007"
