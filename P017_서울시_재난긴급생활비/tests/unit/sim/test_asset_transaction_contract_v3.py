import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from asset_transaction_contract_v3 import inspect


def case():return {'cash':9000,'wallet_lots':{},'offers':{'O':{'wallet_id':'V','unit_face':10000,'unit_cash_cost':9000,'max_units':1}},
    'execution_assumptions':{'offers_available_before_any_purchase':True,'intraday_income':0},
    'events':[{'id':'A','channel':'offline','candidates':[{'id':'Q','price_won':2500,'eligible_wallets':['V']}]}]}


def choice():return {'acquisition_units':{'O':1},'purchases':[{'id':'A','candidate_id':'Q','wallet_spend':{'V':2500}}]}


def test_price_paid_once_and_chosen_asset_acquired_before_use():
    _,ledger=inspect(json.dumps(choice()),case())
    assert ledger['total_consumption']==2500 and ledger['cash_outflow_total']==9000
    assert ledger['actions'][0]['kind']=='acquire_wallet' and ledger['actions'][1]['cash_payment']==0
    assert ledger['closing_wallet_lots']['V']==[{'face':7500,'own_basis':6750}]


def test_no_implicit_extra_acquisition_or_payment_adjustment():
    obj=choice();obj['acquisition_units']['O']=0
    with pytest.raises(ValueError,match='Unavailable'):inspect(json.dumps(obj),case())
    obj=choice();obj['purchases'][0]['wallet_spend']['V']=2501
    with pytest.raises(ValueError,match='exceeds'):inspect(json.dumps(obj),case())


def test_assumptions_required_and_unspent_assets_stay_assets():
    c=case();c.pop('execution_assumptions')
    with pytest.raises(ValueError,match='explicit'):inspect(json.dumps(choice()),c)
    c=case();obj=choice();obj['purchases'][0].update(candidate_id=None,wallet_spend={})
    _,ledger=inspect(json.dumps(obj),c)
    assert ledger['total_consumption']==0 and ledger['asset_acquisition_cash_outflow']==9000


def test_cash_residual_preserves_optional_split():
    c=case();c['cash']=10000;obj=choice();obj['purchases'][0]['wallet_spend']['V']=1500
    _,ledger=inspect(json.dumps(obj),c)
    assert ledger['actions'][1]['cash_payment']==1000 and ledger['total_consumption']==2500
