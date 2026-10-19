import sys,json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from transaction_contract import inspect,schema

CASE={'cash':1000,'wallets':{'W':8000},'events':[
    {'order':0,'channel':'offline','candidates':[{'poi_id':'A','eligible_wallets':['W']}]},
    {'order':1,'channel':'online','candidates':[{'poi_id':'B','eligible_wallets':[]}]}]}

def choice(order,pid,amount=0,payment=None):
    return dict(order=order,poi_id=pid,actual_spent=amount,policy_spend=payment or {},pick_reason='입력에 따른 선택')

def test_null_visit_and_zero_spend_can_be_valid_without_missing_day():
    obj,ledger,errors=inspect(json.dumps({'picks':[choice(0,None),choice(1,None)]}),CASE)
    assert ledger['complete'] and ledger['total_including_online']==0 and not errors

def test_same_union_wrong_order_and_duplicate_orders_are_rejected():
    for picks in [[choice(0,'B'),choice(1,'A')],[choice(0,'A'),choice(0,'A')]]:
        with pytest.raises(ValueError): inspect(json.dumps({'picks':picks}),CASE)

def test_known_wallet_cannot_pay_for_ineligible_online_order():
    with pytest.raises(ValueError):
        inspect(json.dumps({'picks':[choice(0,None),choice(1,'B',5000,{'W':4000})]}),CASE)

def test_affordable_choice_is_preserved_and_explicit_task_constraint_checked():
    case=CASE|{'requirements':[{'order':0,'actual_spent':6000}]}
    picks=[choice(0,'A',5000,{'W':4000}),choice(1,None)]
    _,ledger,errors=inspect(json.dumps({'picks':picks}),case)
    assert ledger['total_including_online']==5000 and errors==['fact:0:actual_spent']

def test_prompt_has_no_named_policy_or_effect_target():
    from prompts.transaction_v2 import SYSTEM_PROMPT
    for word in ['P010','P012','P013','P014','P015','캐시백','쿠폰','상품권','%']:
        assert word not in SYSTEM_PROMPT

def test_empty_required_event_roster_can_complete_without_inventing_purchase():
    case={'cash':8000,'wallets':{},'events':[]}
    _,ledger,errors=inspect('{"picks":[]}',case)
    assert ledger['complete'] and ledger['total_including_online']==0 and not errors
    assert schema(case)['properties']['picks']['maxItems']==0
