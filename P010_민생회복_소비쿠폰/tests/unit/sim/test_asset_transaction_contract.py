import copy
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from asset_transaction_contract import inspect

CASE = {'cash': 9000, 'wallet_lots': {},
        'offers': {'quote': {'wallet_id': 'V', 'unit_face': 10000, 'unit_cash_cost': 9000, 'max_units': 1}},
        'events': [{'id': 'food', 'channel': 'offline', 'candidates': [{'id': 'bundle', 'price_won': 10000, 'eligible_wallets': ['V']}]},
                   {'id': 'walk', 'channel': 'offline', 'candidates': []}]}
ACQUIRE = {'kind': 'acquire_wallet', 'id': 'acquire:quote', 'offer_id': 'quote', 'units': 1, 'reason': 'Available quote'}
BUY = {'kind': 'consume', 'id': 'food', 'candidate_id': 'bundle', 'cash_payment': 0, 'wallet_spend': {'V': 10000}, 'reason': 'Planned food'}
WALK = {'kind': 'consume', 'id': 'walk', 'candidate_id': None, 'cash_payment': 0, 'wallet_spend': {}, 'reason': 'Free walk'}


def test_quote_is_paid_once_and_free_activity_stays_zero():
    _, out = inspect(json.dumps({'actions': [ACQUIRE, BUY, WALK]}), CASE)
    assert out['total_consumption'] == 10000 and out['cash_outflow_total'] == 9000
    assert out['own_funded_consumption'] == 9000 and out['concession_funded_consumption'] == 1000


@pytest.mark.parametrize('actions', [[BUY, ACQUIRE, WALK], [ACQUIRE, WALK, BUY], [ACQUIRE, BUY], [ACQUIRE, BUY, BUY, WALK]])
def test_unfunded_reordered_missing_duplicate_actions_fail(actions):
    with pytest.raises(ValueError): inspect(json.dumps({'actions': actions}), CASE)


def test_cannot_invent_lower_price_or_switch_event_candidate():
    buy = copy.deepcopy(BUY); buy['wallet_spend']['V'] = 8000
    with pytest.raises(ValueError, match='Funding'): inspect(json.dumps({'actions': [ACQUIRE, buy, WALK]}), CASE)
    walk = dict(WALK, candidate_id='bundle')
    with pytest.raises(ValueError, match='candidate'): inspect(json.dumps({'actions': [ACQUIRE, BUY, walk]}), CASE)


def test_explicit_no_purchase_is_valid_without_offer_acquisition():
    decline = dict(BUY, candidate_id=None, cash_payment=0, wallet_spend={})
    _, out = inspect(json.dumps({'actions': [decline, WALK]}), CASE)
    assert out['total_consumption'] == 0 and out['closing_cash'] == 9000
