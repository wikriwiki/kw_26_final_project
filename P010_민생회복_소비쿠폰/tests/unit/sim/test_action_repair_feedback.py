import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from action_repair_feedback import feedback,append_feedback


def source():
    cell={'date':'2026-09-21','zones':['A'],'has_work':False,'user':'오늘 집에서 지내야 한다.',
          'evaluation_requirements':[{'kind':'no_outside'}]}
    events=[{'time':t,'activity_id':a,'anchor':'residence'} for t,a in [
        ('07:00','home_prepare'),('08:00','home_meal'),('10:00','home_chores'),
        ('12:00','home_meal'),('18:00','home_leisure'),('22:00','home_sleep')]]
    return cell,events


def test_valid_output_not_regenerated():
    c,e=source();assert feedback(json.dumps({'events':e}),c) is None
    with pytest.raises(ValueError):append_feedback('original',None)


def test_outside_failure_uses_supplied_fact_not_effect_target():
    c,e=source();e[2].update(activity_id='walk',anchor='zone:A');c['desired_effect']=9999
    f=feedback(json.dumps({'events':e}),c)
    assert f['violated_supplied_restrictions']==[{'kind':'no_outside'}]
    assert '9999' not in json.dumps(f) and f['original_plan']==json.dumps({'events':e})


def test_bad_order_preserved_and_no_success_plan_fabricated():
    c,e=source();e[2]['time']='06:00';raw=json.dumps({'events':e});f=feedback(raw,c)
    assert f['original_plan']==raw and 'time' in f['violations']
    assert 'execution_plan' not in f


def test_malformed_response_retained():
    c,_=source();f=feedback('{',c)
    assert f['original_plan']=='{' and f['violations']==['output_contract']
