from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from forced_no_purchase import resolve


def base():return {'cash':9000,'wallet_lots':{'V':[{'face':10000,'own_basis':9000}]},'offers':{},
                   'events':[{'id':'walk','channel':'offline','candidates':[]}]}


def test_unique_zero_preserves_cash_and_prepaid_assets():
    result=resolve(base());ledger=result['ledger']
    assert result['model_calls']==0 and ledger['total_consumption']==0
    assert ledger['closing_cash']==9000 and ledger['closing_wallet_lots']==base()['wallet_lots']


def test_asset_only_choice_must_not_be_skipped():
    case=base();case['offers']={'O':{'wallet_id':'V','unit_face':10000,'unit_cash_cost':9000,'max_units':1}}
    assert resolve(case) is None


def test_optional_goods_never_forced_to_zero():
    case=base();case['events'][0]['candidates']=[{'id':'q','price_won':4000,'eligible_wallets':[]}]
    assert resolve(case) is None
