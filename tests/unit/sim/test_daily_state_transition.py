from copy import deepcopy
import json

import pytest

from asset_day_checkpoint import commit_day
from daily_state_transition import advance


def fixture():
    need={'id':'wash','description':'Laundry backlog','desired_count':2,'mandatory':False,'fulfilled_by':['wash']}
    case={'cash':1000,'wallet_lots':{},'offers':{},'events':[{'id':'w','time':'08:00','activity_id':'wash','channel':'offline','candidates':[]}],
          'daily_conditions':{'provenance':{'kind':'synthetic_assumption','source':'unit case'},
              'resources':{'soap':{'unit':'dose','opening_quantity':2}},'activity_consumption':{'wash':{'soap':1}},
              'quote_receipts':{},'quote_receipt_delay_minutes':{},'needs':[need]}}
    raw=json.dumps({'acquisition_units':{},'purchases':[{'id':'w','candidate_id':None,'wallet_spend':{}}]})
    template=deepcopy(case['daily_conditions']);template['needs']=[]
    return case,raw,template


def test_unmet_need_carries_without_recreating_fulfilled_need_or_initial_stock():
    case,raw,template=fixture()
    state=advance(case,raw,template)
    assert state['daily_conditions']['resources']['soap']['opening_quantity']==1
    assert state['daily_conditions']['needs'][0]['desired_count']==1
    second=dict(case,**state)
    final=advance(second,raw,template)
    assert final['daily_conditions']['needs']==[]
    assert final['daily_conditions']['resources']['soap']['opening_quantity']==0
    assert template['resources']['soap']['opening_quantity']==2


def test_new_need_added_once_and_changed_identity_rejected():
    case,raw,template=fixture();template['needs']=deepcopy(case['daily_conditions']['needs'])
    assert advance(case,raw,template)['daily_conditions']['needs'][0]['desired_count']==3
    template['needs'][0]['description']='A different need'
    with pytest.raises(ValueError,match='identity'):advance(case,raw,template)


def test_units_and_pending_receipts_cannot_be_invented():
    case,raw,template=fixture();template['resources']['soap']['unit']='bottle'
    with pytest.raises(ValueError,match='convert'):advance(case,raw,template)
    template['resources']['soap']['unit']='dose'
    template['opening_pending_receipts']=[{'minute':0,'event_id':'extra','quantities':{'soap':1}}]
    with pytest.raises(ValueError,match='invent'):advance(case,raw,template)


def test_actual_purchase_cash_and_after_midnight_receipt_carry_together():
    case,_,template=fixture()
    case['events']=[{'id':'buy','time':'23:30','activity_id':'shop','channel':'online',
                    'candidates':[{'id':'soap','price_won':100,'eligible_wallets':[]}]}]
    case['daily_conditions']['quote_receipts']={'soap':{'soap':10}}
    case['daily_conditions']['quote_receipt_delay_minutes']={'soap':90}
    raw=json.dumps({'acquisition_units':{},'purchases':[{'id':'buy','candidate_id':'soap','wallet_spend':{}}]})
    state=advance(case,raw,template)
    assert state['cash']==900
    assert state['daily_conditions']['resources']['soap']['opening_quantity']==2
    assert state['daily_conditions']['opening_pending_receipts']==[{'minute':60,'quantities':{'soap':10},'event_id':'buy'}]
    assert state['daily_conditions']['needs'][0]['desired_count']==2


def test_day_barrier_rejects_dropped_backlog_and_implicit_new_needs(tmp_path):
    case,raw,template=fixture();row={'aid':'A','complete':True,'raw':raw,'transaction_protocol':'v4'}
    first=commit_day(tmp_path,day='2026-09-21',roster=['A'],cases={'A':case},rows=[row],state_protocol='carry_needs_v1')
    state=advance(case,raw,template);second=dict(case,**state)
    tampered=deepcopy(second);tampered['daily_conditions']['needs']=[]
    kwargs=dict(day='2026-09-22',roster=['A'],rows=[row],previous=first,state_protocol='carry_needs_v1',transition_templates={'A':template})
    with pytest.raises(ValueError,match='discontinuity'):commit_day(tmp_path,cases={'A':tampered},**kwargs)
    saved=commit_day(tmp_path,cases={'A':second},**kwargs)
    assert json.loads(saved.read_bytes())['ledgers']['A']['resource_ledger']['unfulfilled_counts']=={'wash':0}


def test_cannot_silently_turn_need_carry_off(tmp_path):
    case,raw,template=fixture();row={'aid':'A','complete':True,'raw':raw,'transaction_protocol':'v4'}
    first=commit_day(tmp_path,day='2026-09-21',roster=['A'],cases={'A':case},rows=[row],state_protocol='carry_needs_v1')
    with pytest.raises(ValueError,match='protocol changed'):
        commit_day(tmp_path,day='2026-09-22',roster=['A'],cases={'A':case},rows=[row],previous=first)
    with pytest.raises(ValueError,match='templates required'):
        commit_day(tmp_path,day='2026-09-22',roster=['A'],cases={'A':case},rows=[row],previous=first,state_protocol='carry_needs_v1')
