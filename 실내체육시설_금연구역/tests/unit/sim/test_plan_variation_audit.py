import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from action_plan_contract import inspect
from plan_variation_audit import audit,distance,sequence


def fixture(root):
    events=[{'time':t,'activity_id':a,'anchor':'residence'} for t,a in [('07:00','home_prepare'),('08:00','home_meal'),('12:00','home_meal'),('15:00','home_leisure'),('19:00','home_meal'),('22:00','home_sleep')]]
    raw=json.dumps({'events':events});cells=[];rows=[]
    for aid in ['A','B']:
        for arm in ['off','on']:
            c={'aid':aid,'case':'m','arm':arm,'date':'2026-09-21','has_work':False,'zones':[],'user':'오늘 집에서 생활'}
            cells.append(c);p=inspect(raw,c,max_shift=0)['execution_plan']
            for seed in [1,2]:rows.append(dict(c,replicate=seed,variant='frozen',eligible=True,raw=raw,execution_plan=p,answer_usage={'finish_reason':{'type':'stop'}}))
    (root/'manifest.json').write_text(json.dumps({'config':{'seeds':[1,2],'candidates':[{'id':'frozen'}],'max_shift_minutes':0}}))
    (root/'frozen_inputs.json').write_text(json.dumps({'cells':cells}))
    write_rows(root,rows)
    return rows


def write_rows(root,rows):
    (root/'responses.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')


def test_distance_and_spatial_identity_do_not_invent_behavior_difference():
    assert distance([],[])==0 and distance(['a'],['b'])==1
    assert distance(['a','b'],['a'])==0.5
    a={'execution_plan':{'events':[{'time':'09:00','activity_id':'walk','anchor':'zone:1'}]}}
    b=copy.deepcopy(a);b['execution_plan']['events'][0].update(anchor='zone:2',time='10:00')
    assert sequence(a)==sequence(b) and sequence(a,include_time=True)!=sequence(b,include_time=True)


def test_repeat_and_counterfactual_denominators_are_separate(tmp_path):
    fixture(tmp_path);r=audit(tmp_path)['mechanisms']['m']['activities_only']
    assert all(v['pairs']==4 and v['identical_fraction']==1 for v in r.values())


@pytest.mark.parametrize('failure',['missing','failed','tampered','calendar'])
def test_incomplete_or_invalid_paths_are_not_silently_removed(tmp_path,failure):
    rows=fixture(tmp_path)
    if failure=='missing':rows.pop()
    elif failure=='failed':rows[0]['eligible']=False
    elif failure=='tampered':rows[0]['execution_plan']['events'][0]['time']='06:00'
    elif failure=='calendar':rows[0]['date']='2026-09-22'
    write_rows(tmp_path,rows)
    with pytest.raises(ValueError):audit(tmp_path)
