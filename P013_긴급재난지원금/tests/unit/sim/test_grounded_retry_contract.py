"""Regression cases from v22 failures; no GPU, network or graph writes."""
import copy
from datetime import date
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))


def reply(value):
    return SimpleNamespace(model='stub', usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
                           choices=[SimpleNamespace(finish_reason='stop',
                               message=SimpleNamespace(content=json.dumps(value, ensure_ascii=False)))])


@pytest.fixture
def s1(monkeypatch):
    import stage1_intent as module
    monkeypatch.setenv('SIM_PROMPT_VARIANT', 'no_smoking_v1')
    monkeypatch.setenv('POLICY_BACKTEST_DETERMINISTIC', '1')
    monkeypatch.setattr(module, '_format_dawn_blocks', lambda *args: '[E0001] 오늘 집에서 휴식한다')
    ctx = SimpleNamespace(persona={}, prompt_timing={}, appointment=[], zone_candidates=[])
    event = {'time': '08:00', 'anchor': 'residence', 'category': '집', 'intent': '휴식',
             'reasoning': '오늘 집에서 휴식한다.', 'trigger': 'none', 'evidence_ref': 'E0001'}
    return module, ctx, event


def test_stage1_repeated_bad_response_uses_exact_six_calls_with_changing_feedback(s1, monkeypatch):
    module, ctx, event = s1
    event['evidence_ref'] = 'E9999'
    prompts = []
    def call(_mode, system, user, **kwargs):
        prompts.append(user)
        return reply({'events': [event]})
    monkeypatch.setattr(module, '_llm_call', call)
    with pytest.raises(module.Stage1Exhausted) as failure:
        module.call_stage1('A', date(2017, 11, 24), ctx, max_retry=5, log_failures=False)
    assert len(prompts) == failure.value.attempts == 6
    assert len(set(prompts)) == 6
    for i, prompt in enumerate(prompts[1:], 1):
        assert f'재시도 {i}/5' in prompt
        assert "events[0] time='08:00'" in prompt
        assert prompt.count('[검증 오류 이유]') == 1
        assert '[직전 시도 검증 실패]' not in prompt
        assert prompt.endswith('위 실수를 반복하지 마세요.')


@pytest.mark.parametrize('value,reason', [('8시', 'time must be HH:MM'), ('25:00', 'time must be HH:MM')])
def test_stage1_bad_time_names_exact_event(s1, monkeypatch, value, reason):
    module, ctx, event = s1
    event['time'] = value
    monkeypatch.setattr(module, '_llm_call', lambda *a, **k: reply({'events': [event]}))
    with pytest.raises(module.Stage1Exhausted, match=r'events\[0\].*' + reason):
        module.call_stage1('A', date(2017, 11, 24), ctx, max_retry=0, log_failures=False)


def test_stage1_membership_is_checked_without_schema_backend(s1, monkeypatch):
    module, ctx, event = s1
    event['with_agents'] = ['invented-agent']
    monkeypatch.setattr(module, '_llm_call', lambda *a, **k: reply({'events': [event]}))
    with pytest.raises(module.Stage1Exhausted, match='with_agents'):
        module.call_stage1('A', date(2017, 11, 24), ctx, max_retry=0, log_failures=False)
    ctx.appointment = [{'with_agents': ['invented-agent']}]
    _, meta = module.call_stage1('A', date(2017, 11, 24), ctx, max_retry=0, log_failures=False)
    assert meta['temp'] == 0 and meta['temp_requested'] == .2
    assert meta['s1_timing']['n_llm_calls'] == 1


def test_feedback_keeps_middle_rejected_item():
    from grounded_schema import rejected_response_feedback
    events = [{'time': '08:00', 'reasoning': 'x' * 500} for _ in range(10)]
    events[5]['anchor'] = 'invalid-anchor-in-middle'
    prompt = rejected_response_feedback(json.dumps({'events': events}),
                                        'events[5] time=08:00: anchor invalid', 'Fix this item.')
    assert 'invalid-anchor-in-middle' in prompt and '[오류가 난 항목 발췌]' in prompt
    assert prompt.count('events[5]') == 1


@pytest.fixture
def s2(monkeypatch):
    import stage2_poi as module
    from stage1_intent import Stage1Output
    monkeypatch.setenv('POLICY_BACKTEST_DETERMINISTIC', '1')
    plan = Stage1Output.model_validate({'events': [
        {'time': '08:00', 'anchor': 'residence', 'category': '집', 'intent': '휴식'},
        {'time': '12:00', 'anchor': 'zone:1', 'category': '식사', 'intent': '점심'},
        {'time': '18:00', 'anchor': 'zone:1', 'category': '식사', 'intent': '저녁'},
    ]})
    monkeypatch.setattr(module, 'fetch_candidates_for_events', lambda *a, **k: {
        i: [{'poi_id': f'P{i}', 'name': '식당', 'unit_anchor': 12000,
             'price_band': 2, 'price_factor': 1.0}] for i in (1, 2)})
    monkeypatch.setattr(module, 'build_stage2_prompt', lambda *a, **k: '점심 후보 P1\n저녁 후보 P2')
    picks = [{'order': i, 'poi_id': f'P{i}', 'actual_spent': 12000,
              'actual_satisfaction': .6, 'pick_reason': '오늘 입력된 식사 목적에 맞는 후보를 선택한다.',
              'pick_factor': 'distance', 'evidence_ref': 'E0001'} for i in (1, 2)]
    return module, plan, picks


@pytest.mark.parametrize('repeat_exact', [False, True])
def test_stage2_remaining_scope_and_exact_replay_keep_prior_choice(s2, monkeypatch, repeat_exact):
    module, plan, picks = s2
    requests = []
    def call(_mode, system, user, **kwargs):
        requests.append((system, user))
        return reply({'picks': [picks[0]] if len(requests) == 1 else
                     picks if repeat_exact else [picks[1]]})
    monkeypatch.setattr(module, '_llm_call', call)
    result, _, meta = module.call_stage2('A', plan, {'_no_smoking_prompt': '규칙'},
                                       date(2017, 11, 24), max_retry=1)
    assert [p.poi_id for p in result.picks] == ['P1', 'P2']
    assert len(requests) == 2
    assert '각 외출 이벤트의 order에 대해' not in requests[1][0]
    assert '모든 필수 order' not in requests[1][1]
    assert requests[1][1].startswith('[이번 응답에서 요청한 order] 2\n')
    assert requests[1][1].endswith('위 실수를 반복하지 마세요.')
    assert meta['temp'] == 0 and meta['s2_timing']['n_llm_calls'] == 2
    if repeat_exact:
        assert meta['s2_timing']['attempts'][1]['already_saved_picks_ignored'] == 1


def test_stage2_conflicting_replay_cannot_overwrite_collected_pick(s2, monkeypatch):
    module, plan, picks = s2
    count = 0
    def call(*a, **k):
        nonlocal count
        count += 1
        value = copy.deepcopy(picks)
        if count > 1:
            value[0]['actual_spent'] = 50000
        return reply({'picks': [value[0]] if count == 1 else value})
    monkeypatch.setattr(module, '_llm_call', call)
    with pytest.raises(RuntimeError, match='got 1, 2; remaining 2'):
        module.call_stage2('A', plan, {'_no_smoking_prompt': '규칙'}, date(2017, 11, 24), max_retry=1)
    assert count == 2
