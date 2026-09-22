from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from purchase_decision_view import build


def test_full_spendable_balance_not_replaced_by_cost_basis():
    case={'cash':0,'offers':{},'events':[],'wallet_lots':{'V':[{'face':4000,'own_basis':2000},{'face':1000,'own_basis':0}]}}
    result=build(case)
    assert result['wallet_balances_won']=={'V':5000} and 'wallet_lots' not in result
    assert case['wallet_lots']['V'][0]['own_basis']==2000


def test_invalid_ledger_lot_not_hidden_by_presentation():
    with pytest.raises(ValueError,match='Cost basis'):
        build({'wallet_lots':{'V':[{'face':1000,'own_basis':2000}]}})
