import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from temporal_projection import project


def plan(times):
    return {'events': [{'time': t, 'intent': 'activity '+str(i), 'anchor': 'residence', 'reasoning': 'input fact', 'trigger': 'none'} for i,t in enumerate(times)], 'daily_propensity': .42}


def test_minimum_shift_keeps_choices_and_pinned_obligations():
    raw = plan(['07:30','08:10','08:20','09:00']); before = copy.deepcopy(raw)
    out, record = project(raw, fixed_times=['08:20','09:00'])
    assert [e['time'] for e in out['events']] == ['07:30','08:00','08:20','09:00']
    assert raw == before and record['total_absolute_shift_minutes'] == 10
    assert [{k:v for k,v in e.items() if k!='time'} for e in out['events']] == [{k:v for k,v in e.items() if k!='time'} for e in raw['events']]
    assert out['daily_propensity'] == raw['daily_propensity']


def test_valid_schedule_is_unchanged():
    raw = plan(['07:00','08:00','12:00','22:00'])
    out, record = project(raw)
    assert out == raw and not record['shifts']


def test_impossible_close_fixed_times_fail_without_reordering():
    with pytest.raises(ValueError, match='No feasible'):
        project(plan(['09:00','09:10']), fixed_times=['09:00','09:10'])


def test_shift_budget_and_day_boundary_are_not_relaxed():
    with pytest.raises(ValueError, match='No feasible'):
        project(plan(['00:00','00:00']))
    with pytest.raises(ValueError, match='No feasible'):
        project(plan(['12:00','07:00']))


def test_invalid_time_is_not_silently_parsed():
    with pytest.raises(ValueError, match='clock'):
        project(plan(['25:00']))


def test_given_commute_is_preserved_without_changing_activity():
    raw = plan(['17:00', '17:30', '18:30'])
    raw['events'][0]['anchor'] = 'workplace'
    out, record = project(raw, fixed_times=['17:00'], transitions=[{'from_anchor': 'workplace', 'to_anchor': 'residence', 'minimum_minutes': 40}])
    assert out['events'][1]['time'] == '17:40'
    assert record['shifts'] == [{'event_index': 1, 'before': '17:30', 'after': '17:40', 'minutes': 10}]
    with pytest.raises(ValueError, match='No feasible'):
        project(raw, max_shift=5, fixed_times=['17:00'], transitions=[{'from_anchor': 'workplace', 'to_anchor': 'residence', 'minimum_minutes': 40}])
