import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from paired_asset_score import score


def rows():
    result=[]
    for arm in ['off','on']:
        case={'cash':9000,'wallet_lots':{},'offers':{},'events':[{'id':'a','channel':'offline','candidates':[]}]}
        actions=[{'kind':'consume','id':'a','candidate_id':None,'cash_payment':0,'wallet_spend':{},'reason':'구매 없음'}]
        if arm=='on':
            case['offers']={'offer':{'wallet_id':'V','unit_face':10000,'unit_cash_cost':9000,'max_units':1}}
            actions.insert(0,{'kind':'acquire_wallet','id':'acquire:offer','offer_id':'offer','units':1,'reason':'나중에 사용'})
        result.append({'aid':'A','date':'2026-09-20','replicate':1,'arm':arm,'valid':True,'transaction_case':case,'raw':json.dumps({'actions':actions})})
    return result


def evaluate(rr):return score(rr,roster=['A'],days=['2026-09-20'],seeds=[1])


def test_assets_not_consumption_and_zero_denominator_not_infinite_effect():
    s=evaluate(rows())['by_seed'][0]['metrics']
    assert s['total_consumption']['difference']==0 and s['total_consumption']['relative_change'] is None
    assert s['asset_acquisition_cash_outflow']['difference']==9000


@pytest.mark.parametrize('kind',['missing','duplicate','failure','tampered'])
def test_no_silent_selection_and_raw_choices_are_revalidated(kind):
    rr=rows()
    if kind=='missing':rr.pop()
    if kind=='duplicate':rr.append(copy.deepcopy(rr[0]))
    if kind=='failure':rr[0]['valid']=False
    if kind=='tampered':rr[1]['transaction_case']['cash']=0
    with pytest.raises(ValueError):evaluate(rr)
