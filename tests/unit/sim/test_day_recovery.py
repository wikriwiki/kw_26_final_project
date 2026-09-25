"""Regression tests for stale checkpoints and day completion barriers."""
from datetime import date
from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
import run_simulation as runner
import night_intent_llm as night

DAY = date(2026, 9, 15)

@pytest.fixture
def output(tmp_path, monkeypatch):
    for key, folder in [('OUT_DIR', tmp_path), ('CHECK_DIR', tmp_path/'check'), ('METRICS_DIR', tmp_path/'metrics')]:
        folder.mkdir(exist_ok=True)
        monkeypatch.setattr(runner, key, folder)
    return tmp_path

def test_stale_done_checkpoint_never_skips_database_reconciliation(output, monkeypatch):
    (output/'check'/f'done_{DAY}.json').write_text('["A"]')
    evidence = output/'metrics'/f'day_{DAY}.jsonl'
    evidence.write_text('previous-corrupt-evidence\n')
    called = []
    def process(aid, *args):
        called.append(aid)
        return {'aid':aid, 'status':'error'}
    monkeypatch.setattr(runner, 'process_one', process)
    with pytest.raises(RuntimeError, match='incomplete agent day'):
        runner.run_day(['A'], DAY, 0, workers=1)
    assert called == ['A']
    assert evidence.read_text().startswith('previous-corrupt-evidence\n')
    cohort = json.loads((output/f'cohort_{DAY}.json').read_text(encoding='utf-8'))
    assert cohort['prompt_variant'] == runner._ACTIVE_PROMPT_VARIANT
    assert cohort['system_prompt_sha256'] == hashlib.sha256(
        runner.DAWN_SYSTEM_PROMPT.encode('utf-8')).hexdigest()

@pytest.mark.parametrize('agents', [[], ['A', 'A'], ['']])
def test_invalid_cohort_rejected_before_execution(output, agents):
    with pytest.raises(ValueError, match='cohort'):
        runner.run_day(agents, DAY, 0, workers=1)


def test_prompt_environment_change_after_import_is_rejected(output, monkeypatch):
    monkeypatch.setattr(runner, 'active_prompt_name', lambda: 'changed-in-process')
    with pytest.raises(ValueError, match='prompt variant changed'):
        runner.run_day(['A'], DAY, 0, workers=1)
    assert not (output/f'cohort_{DAY}.json').exists()

class Result:
    def single(self): return {'n':0}
class Session:
    def run(self, *args, **kwargs): return Result()


def test_single_pair_is_classified_when_database_empty(monkeypatch):
    @contextmanager
    def session(): yield Session()
    monkeypatch.setattr(night, 'driver_session', session)
    monkeypatch.setattr(night, 'fetch_pair_data', lambda *args: {('A','B'): {}})
    monkeypatch.setattr(night, 'classify_intent', lambda *args: {'initiator_id':'A','recipient_id':'B'})
    monkeypatch.setattr(night, 'write_conversations', lambda day, rows: {'created':len(rows)})
    result = night.run_intent_classification(DAY, [{'a':'A','b':'B'}], workers=1)
    assert result['processed'] == 1
    assert result['write']['created'] == 1


def test_partial_night_does_not_advance_day(output, monkeypatch):
    import neo4j_load._common as common
    monkeypatch.setattr(runner,'process_one',lambda aid,*args: {'aid':aid,'status':'ok'})
    monkeypatch.setattr(runner,'_write_timing_diagnostics',lambda *args: {})
    class Partial(Session):
        def run(self,*args,**kwargs):
            class Count:
                def single(self): return {'n':50}
            return Count()
    @contextmanager
    def session(): yield Partial()
    monkeypatch.setattr(common,'driver_session',session)
    with pytest.raises(RuntimeError, match='Night2 failed'):
        runner.run_day(['A'], DAY, 0, workers=1)
    assert not (output/f'night2_completed_{DAY}.json').exists()


@pytest.mark.parametrize('missing', [True, False])
def test_missing_pair_or_failed_classification_never_writes_partial_night(monkeypatch, missing):
    @contextmanager
    def session(): yield Session()
    monkeypatch.setattr(night, 'driver_session', session)
    monkeypatch.setattr(night, 'fetch_pair_data', lambda *args: {} if missing else {('A','B'): {}})
    monkeypatch.setattr(night, 'classify_intent', lambda *args: {'error':'injected'})
    monkeypatch.setattr(night, 'write_conversations', lambda *args: pytest.fail('partial night written'))
    with pytest.raises(RuntimeError):
        night.run_intent_classification(DAY, [{}], workers=1)


def test_completed_empty_night_reuses_verified_marker(output, monkeypatch):
    import neo4j_load._common as common
    import night_interaction
    @contextmanager
    def session(): yield Session()
    monkeypatch.setattr(common, 'driver_session', session)
    monkeypatch.setattr(runner, 'process_one', lambda aid,*args: {'aid':aid,'status':'ok'})
    monkeypatch.setattr(runner, '_write_timing_diagnostics', lambda *args: {})
    monkeypatch.setattr(night_interaction, 'select_interaction_pairs', lambda *args,**kwargs: [])
    assert runner.run_day(['A'],DAY,0,workers=1)['ok'] == 1
    monkeypatch.setattr(night_interaction, 'select_interaction_pairs', lambda *args,**kwargs: pytest.fail('completed night repeated'))
    assert runner.run_day(['A'],DAY,0,workers=1)['ok'] == 1
    canonical = (output/'metrics'/f'day_{DAY}.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(canonical) == 1 and json.loads(canonical[0])['aid'] == 'A'
    attempts = sorted((output/'metrics'/'attempts').glob(f'day_{DAY}_*.jsonl'))
    assert len(attempts) == 2
    assert len(attempts[-1].read_text(encoding='utf-8').splitlines()) == 2


def test_failed_day_resume_keeps_attempt_and_publishes_one_success(output, monkeypatch):
    import neo4j_load._common as common
    import night_interaction
    calls = []
    def process(aid, *args):
        calls.append(aid)
        return {'aid': aid, 'status': 'error' if len(calls) == 1 else 'ok'}
    @contextmanager
    def session(): yield Session()
    monkeypatch.setattr(common, 'driver_session', session)
    monkeypatch.setattr(runner, 'process_one', process)
    monkeypatch.setattr(runner, '_write_timing_diagnostics', lambda *args: {})
    monkeypatch.setattr(night_interaction, 'select_interaction_pairs', lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match='incomplete agent day'):
        runner.run_day(['A'], DAY, 0, workers=1)
    assert runner.run_day(['A'], DAY, 0, workers=1)['ok'] == 1
    canonical = (output/'metrics'/f'day_{DAY}.jsonl').read_text(encoding='utf-8').splitlines()
    assert [json.loads(line)['status'] for line in canonical] == ['ok']
    raw = next((output/'metrics'/'attempts').glob(f'day_{DAY}_*.jsonl'))
    assert [json.loads(line)['status'] for line in raw.read_text(encoding='utf-8').splitlines()] == ['error', 'ok']


def test_resume_rejects_changed_baseline_income_map(output, monkeypatch):
    import neo4j_load._common as common
    import night_interaction
    from income import _baseline_map
    map_path = output/'baseline.json'
    def write_map(amount):
        map_path.write_text(json.dumps({
            'schema': 'baseline_income_v1',
            'policy_free_success_rows_verified': True,
            'citizen_count': 1,
            'daily_income_by_aid': {'A': amount},
        }), encoding='utf-8')
    write_map(100)
    monkeypatch.setenv('EXP_DAILY_INCOME', 'baseline')
    monkeypatch.setenv('EXP_DAILY_INCOME_MAP', str(map_path))
    @contextmanager
    def session(): yield Session()
    monkeypatch.setattr(common, 'driver_session', session)
    monkeypatch.setattr(runner, 'process_one', lambda aid, *args: {'aid': aid, 'status': 'ok'})
    monkeypatch.setattr(runner, '_write_timing_diagnostics', lambda *args: {})
    monkeypatch.setattr(night_interaction, 'select_interaction_pairs', lambda *args, **kwargs: [])
    assert runner.run_day(['A'], DAY, 0, workers=1)['ok'] == 1
    write_map(200)
    _baseline_map.cache_clear()  # a resumed process starts with a fresh cache
    with pytest.raises(ValueError, match='cohort or execution settings changed'):
        runner.run_day(['A'], DAY, 0, workers=1)
