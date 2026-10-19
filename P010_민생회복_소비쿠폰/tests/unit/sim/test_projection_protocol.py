import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
import validate_bounded_planner as runner


def invoke(monkeypatch, tmp_path, *, wrong_endpoint=False, wrong_trigger=False):
    events = [{'time': t, 'anchor': 'residence', 'category': '집', 'intent': '휴식', 'reasoning': '오늘의 일상 선택', 'trigger': 'none'}
              for t in ['07:00','07:30','08:10','08:20','09:00','19:00']]
    if wrong_endpoint:
        events[0].update(anchor='zone:A', category='여가')
    if wrong_trigger:
        events[1]['trigger'] = 'policy'
    raw = json.dumps({'events': events, 'daily_propensity': .5}, ensure_ascii=False)
    def fake(**kwargs):
        return {'forced_reasoning_boundary': True, 'response': {'meta_info': {}}}, {'response': {'text': raw, 'meta_info': {'finish_reason': {'type': 'stop'}}}}
    monkeypatch.setattr(runner, 'run', fake)
    (tmp_path/'attempts').mkdir()
    cell = {'aid': 'A', 'case': 'empty_memory', 'arm': 'off', 'date': '2026-09-21', 'context_sha256': 'h',
            'user': '독서를 즐긴다.', 'zones': ['A'], 'has_work': False, 'allowed_triggers': ['none','lifestyle','mood'],
            'fixed_times': ['08:20','09:00']}
    config = {'schema_mode': 'endpoint_guard', 'reasoning_evidence': True, 'input_trigger_guard': True,
              'temporal_projection': {'max_shift_minutes': 10}, 'answer_tokens': 100, 'sampling': {}, 'timeout_seconds': 10}
    row = runner.invoke(({'id': 'candidate', 'thinking_tokens': 100}, 1, cell), config, 'local', {('candidate','h'): 'P'}, tmp_path)
    return row, raw


def test_raw_failure_and_adjusted_execution_plan_are_both_preserved(monkeypatch, tmp_path):
    row, raw = invoke(monkeypatch, tmp_path)
    assert row['raw'] == raw and row['raw_valid'] is False and row['raw_errors'] == ['time']
    assert row['valid'] and row['execution_plan']['events'][2]['time'] == '08:00'
    assert json.loads(raw)['events'][2]['time'] == '08:10'
    assert row['temporal_projection']['total_absolute_shift_minutes'] == 10
    assert len(list((tmp_path/'attempts').glob('*_answer.json'))) == 1


def test_projection_does_not_hide_non_time_failure(monkeypatch, tmp_path):
    row, _ = invoke(monkeypatch, tmp_path, wrong_endpoint=True)
    assert not row['valid'] and 'endpoints' in row['errors']
    assert 'execution_plan' not in row


def test_unavailable_trigger_is_rejected_even_if_server_breaks_grammar(monkeypatch, tmp_path):
    row, _ = invoke(monkeypatch, tmp_path, wrong_trigger=True)
    assert not row['valid'] and 'unavailable_trigger' in row['errors']
    assert 'execution_plan' not in row
