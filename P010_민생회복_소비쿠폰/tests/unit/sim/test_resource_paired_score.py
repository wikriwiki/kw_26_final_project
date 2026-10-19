import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from paired_asset_score import score


def rows(stock):
    case={'cash':1000,'wallet_lots':{},'offers':{},'events':[{'id':'use','time':'10:00','activity_id':'wash','channel':'offline','candidates':[]}],
          'daily_conditions':{'provenance':{'kind':'synthetic_assumption','source':'test'},
                              'resources':{'soap':{'unit':'dose','opening_quantity':stock}},
                              'activity_consumption':{'wash':{'soap':1}},'quote_receipts':{},'quote_receipt_delay_minutes':{},'needs':[]}}
    raw=json.dumps({'acquisition_units':{},'purchases':[{'id':'use','candidate_id':None,'wallet_spend':{}}]})
    return [{'aid':'A','date':'2026-09-21','replicate':1,'arm':arm,'valid':True,'transaction_protocol':'v4','raw':raw,'transaction_case':case} for arm in ['off','on']]


def test_green_financial_rows_cannot_hide_physical_shortage():
    with pytest.raises(ValueError,match='shortage'):
        score(rows(0),roster=['A'],days=['2026-09-21'],seeds=[1])
    result=score(rows(1),roster=['A'],days=['2026-09-21'],seeds=[1])
    assert result['complete_matrix']


def test_incorrect_recorded_inventory_is_not_trusted():
    r=rows(1);r[0]['resource_ledger']={'closing_resources':{'soap':99}}
    with pytest.raises(ValueError,match='physical state'):
        score(r,roster=['A'],days=['2026-09-21'],seeds=[1])
