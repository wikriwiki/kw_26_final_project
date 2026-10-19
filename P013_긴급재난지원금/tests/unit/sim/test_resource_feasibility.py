from copy import deepcopy

from prepare_resource_stress import prepare
from resource_feasibility import check


def test_every_known_feasible_witness_passes_necessary_bound():
    for cell in prepare()['cells']:
        assert not check(cell['transaction_case'])['impossible_even_with_all_candidates']


def test_late_delivery_proves_schedule_impossible_without_calling_model():
    case=deepcopy(prepare()['cells'][0]['transaction_case'])
    case['events'][1]['time']='10:00'
    r=check(case)
    assert r['impossible_even_with_all_candidates']
    assert r['shortfalls']==[{'event_id':'event:1','time':'10:00','resource':'soap','maximum_available':0,'required':1}]


def test_bound_includes_pending_receipts_and_does_not_invent_missing_goods():
    case=deepcopy(prepare()['cells'][0]['transaction_case']);case['events'][0]['candidates']=[]
    assert check(case)['impossible_even_with_all_candidates']
    case['daily_conditions']['opening_pending_receipts']=[{'minute':630,'event_id':'yesterday','quantities':{'soap':1}}]
    assert not check(case)['impossible_even_with_all_candidates']


def test_optimistic_bound_is_not_an_affordability_certificate():
    case=deepcopy(prepare()['cells'][0]['transaction_case']);case['cash']=0
    assert not check(case)['impossible_even_with_all_candidates']
