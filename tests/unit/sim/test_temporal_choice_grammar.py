from pathlib import Path
import sys
import json
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from temporal_choice_grammar import build
from action_plan_contract import inspect


def cell():return {'date':'2026-09-21','has_work':True,'zones':['Z'],'user':'09:00부터 17:00까지 사무실에서 일한다.',
    'required_activities':[{'time':'09:00','activity_id':'office_work','anchor':'workplace','evidence':'09:00부터 17:00까지 사무실에서 일한다.'}],
    'required_presence_intervals':[{'start':'09:00','end':'17:00','anchor':'workplace','evidence':'09:00부터 17:00까지 사무실에서 일한다.'}],
    'minimum_transitions':[{'from_anchor':'residence','to_anchor':'workplace','minimum_minutes':40},{'from_anchor':'workplace','to_anchor':'residence','minimum_minutes':40}]}


def test_constructed_example_satisfies_independent_full_contract():
    grammar,audit=build(cell(),clock_step=60)
    assert grammar.startswith('root ::=') and audit['grammar_bytes']>0
    result=inspect(json.dumps(audit['feasible_example']),cell(),max_shift=0)
    assert result['raw_valid'] and result['valid']


def test_new_committed_time_is_preserved_even_off_grid():
    c=cell();c['required_activities'][0]['time']='09:13';c['required_presence_intervals'][0]['start']='09:13'
    c['user']=c['user'].replace('09:00','09:13')
    c['required_activities'][0]['evidence']=c['user'];c['required_presence_intervals'][0]['evidence']=c['user']
    _,a=build(c,clock_step=60)
    assert any(e['time']=='09:13' and e['activity_id']=='office_work' for e in a['feasible_example']['events'])


def test_unsupported_route_constraints_not_ignored():
    c=cell();c['minimum_transitions'].append({'from_anchor':'zone:Z','to_anchor':'workplace','minimum_minutes':70})
    with pytest.raises(ValueError,match='route'):build(c)


def test_declared_output_coverage_is_structural_not_a_policy_preference():
    _,audit=build(cell(),clock_step=60,last_start_not_before='20:00')
    assert audit['last_start_not_before']=='20:00'
    assert audit['feasible_example']['events'][-1]['time']>='20:00'
    assert inspect(json.dumps(audit['feasible_example']),cell(),max_shift=0)['raw_valid']
