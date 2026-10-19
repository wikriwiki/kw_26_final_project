import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from asset_transaction_contract_v2 import inspect,schema


def case():return {'cash':9000,'wallet_lots':{},'offers':{'O':{'wallet_id':'V','unit_face':10000,'unit_cash_cost':9000,'max_units':1}},
    'events':[{'id':'A','channel':'offline','candidates':[{'id':'Q','price_won':5000,'eligible_wallets':['V']}]}]}


def choice(position='A'):return {'acquisitions':[{'offer_id':'O','units':1,'before_event_id':position,'reason':'취득 선택'}],
    'purchases':[{'id':'A','candidate_id':'Q','cash_payment':0,'wallet_spend':{'V':5000},'reason':'후보 선택'}]}


def test_actual_acquisition_position_controls_funding_not_json_field_order():
    _,ledger=inspect(json.dumps(choice()),case())
    assert ledger['total_consumption']==5000 and ledger['asset_acquisition_cash_outflow']==9000
    with pytest.raises(ValueError,match='Unavailable'):inspect(json.dumps(choice(None)),case())


def test_roster_missing_duplicate_and_duplicate_offer_rejected():
    for change in ['missing','duplicate_purchase','duplicate_offer']:
        obj=choice()
        if change=='missing':obj['purchases']=[]
        elif change=='duplicate_purchase':obj['purchases']*=2
        else:obj['acquisitions']*=2
        with pytest.raises(ValueError):inspect(json.dumps(obj),case())


def test_asset_only_and_empty_day_contracts():
    c=case();c['events']=[];obj=choice(None);obj['purchases']=[]
    _,ledger=inspect(json.dumps(obj),c)
    assert ledger['total_consumption']==0 and ledger['closing_wallet_lots']['V'][0]['face']==10000
    assert schema(c)['properties']['purchases']['maxItems']==0


def test_opt_in_forced_route_does_not_call_model(tmp_path,monkeypatch):
    import validate_purchase_probe as runner
    monkeypatch.setattr(runner,'run',lambda **kw:pytest.fail('No model choice exists'))
    c=case();c['offers']={};c['events'][0]['candidates']=[]
    cell={'aid':'A','case':'test','arm':'off','date':'2026-09-21','transaction_case':c}
    (tmp_path/'attempts').mkdir()
    row=runner.invoke((1,cell),{'transaction_protocol':'v2','skip_forced_no_purchase':True},None,{},tmp_path)
    assert row['valid'] and row['model_calls']==0 and row['transaction_protocol']=='v2'
    assert json.loads(row['raw'])['purchases'][0]['candidate_id'] is None
