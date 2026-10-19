import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from transaction_ledger import settle_choices


def test_zero_purchase_and_unused_wallet_stay_zero():
    tx = [{'id':'walk','channel':'offline','amount':0,'policy_spend':{}}]
    before = copy.deepcopy(tx)
    out = settle_choices(tx,cash=10000,wallets={'W':20000},eligible_wallets_by_transaction={'walk':{'W'}})
    assert out['total_including_online']==0 and out['closing_wallets']=={'W':20000}
    assert out['closing_cash']==10000 and tx==before


def test_online_offline_and_funding_reconcile_without_rescaling():
    tx = [{'id':'shop','channel':'offline','amount':8000,'policy_spend':{'W':5000}},
          {'id':'delivery','channel':'online','amount':7000,'policy_spend':{}}]
    out = settle_choices(tx,cash=12000,wallets={'W':6000},eligible_wallets_by_transaction={'shop':{'W'},'delivery':set()})
    assert out['offline_total']==8000 and out['online_total']==7000
    assert out['total_including_online']==15000 and out['own_payment_total']==10000
    assert out['closing_cash']==2000 and out['closing_wallets']=={'W':1000}


@pytest.mark.parametrize('amount,payment,eligible,cash,wallet',[
    (5000,6000,{'W'},10000,10000),
    (5000,1000,set(),10000,10000),
    (5000,1000,{'W'},3000,10000),
    (5000,4000,{'W'},10000,3000),
    (float('nan'),0,{'W'},10000,10000),
    (float('inf'),0,{'W'},10000,10000),
    (True,0,{'W'},10000,10000),
    (5000.5,0,{'W'},10000,10000),
])
def test_invalid_accounting_is_rejected_not_repaired(amount,payment,eligible,cash,wallet):
    with pytest.raises(ValueError):
        settle_choices([{'id':'x','channel':'offline','amount':amount,'policy_spend':{'W':payment}}],
                       cash=cash,wallets={'W':wallet},eligible_wallets_by_transaction={'x':eligible})


def test_duplicate_transactions_and_missing_eligibility_rejected():
    tx = {'id':'x','channel':'offline','amount':0,'policy_spend':{}}
    with pytest.raises(ValueError):
        settle_choices([tx,tx],cash=0,wallets={},eligible_wallets_by_transaction={'x':set()})
    with pytest.raises(ValueError):
        settle_choices([tx],cash=0,wallets={},eligible_wallets_by_transaction={})


def test_completed_no_transaction_day_is_valid_zero():
    out = settle_choices([],cash=8000,wallets={},eligible_wallets_by_transaction={})
    assert out['complete'] and out['total_including_online']==0 and out['closing_cash']==8000
