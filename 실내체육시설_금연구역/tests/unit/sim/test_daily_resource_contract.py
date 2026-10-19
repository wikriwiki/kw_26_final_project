import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from daily_resource_contract import settle, validate


def case(stock=0):
    return {'daily_conditions': {
        'provenance': {'kind': 'synthetic_assumption', 'source': 'test scenario'},
        'resources': {'soap': {'unit': 'dose', 'opening_quantity': stock}},
        'activity_consumption': {'wash': {'soap': 1}}, 'quote_receipts': {'bottle': {'soap': 10}},
        'quote_receipt_delay_minutes': {'bottle': 60},
        'needs': [{'id': 'clothes', 'description': 'Want clean clothes', 'fulfilled_by': ['wash'],
                   'desired_count': 1, 'mandatory': False}]},
        'events': [{'id': 'e1', 'activity_id': 'shop', 'time': '09:00', 'candidates': [{'id': 'bottle'}]},
                   {'id': 'e2', 'activity_id': 'wash', 'time': '11:00', 'candidates': []}]}


def choices(buy=True):
    return json.dumps({'purchases': [{'id': 'e1', 'candidate_id': 'bottle' if buy else None},
                                     {'id': 'e2', 'candidate_id': None}]})


def test_purchase_then_use_conserves_units_without_mutating_input():
    source = case()
    result = settle(choices(), source)
    assert result['closing_resources'] == {'soap': 9}
    assert result['fulfilled_counts'] == {'clothes': 1}
    assert source['daily_conditions']['resources']['soap']['opening_quantity'] == 0


def test_skipped_purchase_cannot_pay_for_physical_use():
    with pytest.raises(ValueError, match='shortage'):
        settle(choices(False), case())
    assert settle(choices(False), case(2))['closing_resources'] == {'soap': 1}


def test_future_receipt_cannot_fund_past_use():
    source = case()
    source['events'][0]['activity_id'] = 'wash'
    source['events'][1].update(activity_id='shop', candidates=[{'id': 'bottle'}])
    raw = json.dumps({'purchases': [{'id': 'e1', 'candidate_id': None}, {'id': 'e2', 'candidate_id': 'bottle'}]})
    with pytest.raises(ValueError, match='shortage'):
        settle(raw, source)


def test_optional_need_is_not_hidden_required_spending():
    source = case()
    source['events'][1]['activity_id'] = 'rest'
    result = settle(choices(False), source)
    assert result['closing_resources'] == {'soap': 0}
    assert result['unfulfilled_counts'] == {'clothes': 1}


def test_delivery_arrival_is_required_before_use_and_late_receipts_are_preserved():
    source = case()
    source['events'][1]['time'] = '09:30'
    with pytest.raises(ValueError, match='shortage'):
        settle(choices(), source)
    source['events'][1]['activity_id'] = 'rest'
    result = settle(choices(), source)
    assert result['closing_resources'] == {'soap': 10}
    assert result['end_of_day_receipts'][0]['minute'] == 600
    source['daily_conditions']['quote_receipt_delay_minutes']['bottle'] = 1440
    result = settle(choices(), source)
    assert result['closing_resources'] == {'soap': 0}
    assert result['pending_receipts'][0]['minute'] == 1980


@pytest.mark.parametrize('value', [True, -1, 1.5, float('nan')])
def test_invalid_quantity_refused(value):
    source = case(value)
    with pytest.raises(ValueError):
        validate(source['daily_conditions'])


def test_unknown_units_or_out_of_order_cannot_be_silently_interpreted():
    source = case()
    source['daily_conditions']['quote_receipts']['bottle'] = {'different_resource': 1}
    with pytest.raises(ValueError, match='Unknown'):
        validate(source['daily_conditions'])
    source = case()
    source['events'][1]['time'] = '08:00'
    with pytest.raises(ValueError, match='chronological'):
        settle(choices(), source)
