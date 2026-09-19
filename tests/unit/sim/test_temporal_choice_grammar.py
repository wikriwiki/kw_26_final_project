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


def test_declared_school_zone_is_not_collapsed_with_other_outside_locations():
    c=cell();c['zones']=['school','other'];c['has_work']=False;c['user']='학교에서 09:00부터 15:00까지 수업에 참석한다.'
    c['provided_activities']=[{'id':'school_class','anchors':['zone:school'],'category':'교육','intent':'수업',
                              'purchase_channel':None,'billing_status':'no_transaction','evidence':c['user']}]
    c['required_activities']=[{'time':'09:00','activity_id':'school_class','anchor':'zone:school','evidence':c['user']}]
    c['required_presence_intervals']=[{'start':'09:00','end':'15:00','anchor':'zone:school','evidence':c['user']}]
    c['minimum_transitions']=[{'from_anchor':'residence','to_anchor':'zone:school','minimum_minutes':45},
                              {'from_anchor':'zone:school','to_anchor':'residence','minimum_minutes':45}]
    _,audit=build(c,clock_step=60,allow_zone_commitments=True,last_start_not_before='18:00')
    assert audit['distinct_committed_zones']==['zone:school']
    assert inspect(json.dumps(audit['feasible_example']),c,max_shift=0)['raw_valid']
    # Every school-zone terminal used within the commitment stays in that exact zone.
    grammar,_=build(c,clock_step=60,allow_zone_commitments=True)
    assert 'zone:other' not in next(line for line in grammar.splitlines() if line.startswith('e_10_3 ::='))
