import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from activity_purchase_bridge import prepare_case

CELL = {'date':'2026-09-21','has_work':False,'zones':['A'],'user':'오늘 개인 휴일.','fixed_times':[]}
PLAN = json.dumps({'events':[{'time':t,'activity_id':act,'anchor':a} for t,act,a in [
    ('07:00','home_prepare','residence'),('08:00','home_meal','residence'),('10:00','walk','zone:A'),
    ('12:00','home_delivery','residence'),('18:00','home_meal','residence'),('22:00','home_sleep','residence')]]})
QUOTE = {'id':'food','description':'한 끼 묶음','price_won':5000,'eligible_wallets':[],'channel':'online','price_provenance':'Explicit synthetic test quote, not observed price'}


def test_free_activity_is_never_routed_to_a_paid_poi_and_online_stays_online():
    case, audit = prepare_case(raw_plan=PLAN,cell=CELL,quotes_by_event={'3':[QUOTE]},cash=8000,wallet_lots={},offers={})
    assert len(case['events']) == 6 and case['events'][2]['candidates'] == []
    assert case['events'][3]['channel'] == 'online' and case['events'][3]['candidates'][0]['price_won'] == 5000
    assert audit['coverage'][2]['free_activity']


def test_missing_or_unlabelled_price_cannot_be_replaced_by_spend_anchor():
    with pytest.raises(ValueError,match='coverage'):
        prepare_case(raw_plan=PLAN,cell=CELL,quotes_by_event={},cash=8000,wallet_lots={},offers={})
    with pytest.raises(ValueError,match='Unlabelled'):
        prepare_case(raw_plan=PLAN,cell=CELL,quotes_by_event={'3':[dict(QUOTE,price_provenance='')]},cash=8000,wallet_lots={},offers={})


def test_commercial_substitute_for_free_walk_is_rejected():
    with pytest.raises(ValueError,match='Free activity'):
        prepare_case(raw_plan=PLAN,cell=CELL,quotes_by_event={'2':[QUOTE],'3':[QUOTE]},cash=8000,wallet_lots={},offers={})


def test_time_valid_schedule_cannot_bypass_explicit_outing_restriction():
    with pytest.raises(ValueError,match='restriction'):
        prepare_case(raw_plan=PLAN,cell=dict(CELL,evaluation_requirements=[{'kind':'no_outside'}]),quotes_by_event={'3':[QUOTE]},cash=8000,wallet_lots={},offers={})


def test_appointment_without_fee_information_is_not_silently_free():
    obj=json.loads(PLAN);obj['events'][2]['activity_id']='clinic'
    provided={'id':'clinic','anchors':['zone:A'],'category':'건강','intent':'입력 예약 참석','purchase_channel':None,'evidence':'진료 예약 있음.'}
    cell=dict(CELL,user='진료 예약 있음.',provided_activities=[provided])
    with pytest.raises(ValueError,match='unresolved billing'):
        prepare_case(raw_plan=json.dumps(obj),cell=cell,quotes_by_event={'3':[QUOTE]},cash=8000,wallet_lots={},offers={})
