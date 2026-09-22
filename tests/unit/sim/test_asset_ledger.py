import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from asset_ledger import settle_asset_actions

OFFERS = {'offer': {'wallet_id': 'V', 'unit_face': 10000, 'unit_cash_cost': 9000, 'max_units': 2}}


def acquire(): return {'id': 'buy', 'kind': 'acquire_wallet', 'offer_id': 'offer', 'units': 1}
def consume(pid, amount, payment):
    return {'id': pid, 'kind': 'consume', 'channel': 'offline', 'amount': amount, 'cash_payment': amount-payment, 'wallet_spend': {'V': payment} if payment else {}}


def test_acquisition_is_not_goods_consumption_or_free_cash():
    out = settle_asset_actions([acquire()], cash=12000, wallet_lots={}, offers=OFFERS, eligible_wallets_by_purchase={})
    assert out['total_consumption'] == 0 and out['cash_outflow_total'] == 9000
    assert out['closing_cash'] == 3000 and out['closing_wallet_lots']['V'] == [{'face': 10000, 'own_basis': 9000}]


def test_partial_and_full_redemption_preserve_private_basis_without_double_count():
    actions = [acquire(), consume('A', 4000, 4000), consume('B', 6000, 6000)]
    before = copy.deepcopy(actions)
    out = settle_asset_actions(actions, cash=12000, wallet_lots={}, offers=OFFERS, eligible_wallets_by_purchase={'A': {'V'}, 'B': {'V'}})
    assert out['total_consumption'] == 10000 and out['cash_outflow_total'] == 9000
    assert out['own_funded_consumption'] == 9000 and out['concession_funded_consumption'] == 1000
    assert out['closing_wallet_lots']['V'] == [{'face': 0, 'own_basis': 0}] and actions == before


def test_explicit_unused_wallet_and_zero_activity_are_preserved():
    out = settle_asset_actions([consume('A', 5000, 0), consume('B', 0, 0)], cash=10000,
                               wallet_lots={'V': [{'face': 10000, 'own_basis': 0}]}, offers={}, eligible_wallets_by_purchase={'A': {'V'}, 'B': set()})
    assert out['closing_cash'] == 5000 and out['closing_wallet_lots']['V'][0]['face'] == 10000
    assert out['concession_funded_consumption'] == 0


def test_funds_must_be_available_before_the_purchase():
    with pytest.raises(ValueError, match='Unavailable'):
        settle_asset_actions([consume('A', 10000, 10000), acquire()], cash=12000, wallet_lots={}, offers=OFFERS, eligible_wallets_by_purchase={'A': {'V'}})


def test_ineligible_redemption_rejected():
    with pytest.raises(ValueError, match='ineligible'):
        settle_asset_actions([acquire(), consume('A', 10000, 10000)], cash=12000, wallet_lots={}, offers=OFFERS, eligible_wallets_by_purchase={'A': set()})


def test_partial_rounding_releases_all_basis_at_full_redemption():
    out = settle_asset_actions([consume('A', 1, 1), consume('B', 2, 2)], cash=0,
                               wallet_lots={'V': [{'face': 3, 'own_basis': 2}]}, offers={}, eligible_wallets_by_purchase={'A': {'V'}, 'B': {'V'}})
    assert out['own_funded_consumption'] == 2 and out['concession_funded_consumption'] == 1
