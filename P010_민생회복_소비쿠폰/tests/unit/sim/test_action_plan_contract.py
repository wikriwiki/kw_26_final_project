import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from action_plan_contract import catalog, inspect

CELL = {'date': '2026-09-21', 'has_work': True, 'zones': ['A'], 'user': '외출 금지. 카페 휴업. 확정 출근 09:00.', 'fixed_times': ['09:00'],
        'minimum_transitions': [{'from_anchor': 'residence', 'to_anchor': 'workplace', 'minimum_minutes': 40}]}


def plan():
    return {'events': [{'time': t, 'activity_id': activity, 'anchor': anchor} for t, activity, anchor in
                      [('07:00','home_prepare','residence'),('08:00','home_meal','residence'),('09:00','office_work','workplace'),
                       ('12:00','office_meal','workplace'),('18:00','home_meal','residence'),('21:00','home_sleep','residence')]]}


def test_typed_activity_cannot_fabricate_medication_or_online_wallet_use():
    obj = plan(); obj['events'][1]['intent'] = '정기약 복용'
    with pytest.raises(ValueError, match='field'): inspect(json.dumps(obj), CELL)
    obj = plan(); obj['events'][1]['activity_id'] = 'invented_medication'
    with pytest.raises(ValueError, match='Unavailable'): inspect(json.dumps(obj), CELL)


def test_work_and_home_activities_cannot_exchange_places():
    obj = plan(); obj['events'][2]['activity_id'] = 'remote_work'
    with pytest.raises(ValueError, match='location'): inspect(json.dumps(obj), CELL)


def test_supplied_closure_rules_filter_actions_without_forcing_purchases():
    cell = dict(CELL, action_rules=[{'kind': 'forbid_activity', 'activity_ids': ['cafe_dine_in','cafe_takeaway'], 'evidence': '카페 휴업.'}])
    options = catalog(cell)
    assert 'cafe_dine_in' not in options and 'groceries' in options and 'walk' in options


def test_free_activity_has_no_commerce_channel_and_given_commitments_are_checked():
    assert catalog(CELL)['walk']['purchase_channel'] is None
    cell = dict(CELL, required_activities=[{'time': '09:00', 'activity_id': 'office_work', 'anchor': 'workplace', 'evidence': '확정 출근 09:00.'}])
    assert inspect(json.dumps(plan()), cell)['valid']
    obj = plan(); obj['events'][2]['time'] = '10:00'
    assert 'missing_commitment' in inspect(json.dumps(obj), cell)['errors']


def test_rule_or_extra_activity_cannot_launder_absent_input_facts():
    with pytest.raises(ValueError, match='evidence'):
        catalog(dict(CELL, action_rules=[{'kind':'forbid_activity','activity_ids':['shopping'],'evidence':'물건 고장'}]))
