import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from prepare_purchase_probe import build_case


def source(arm='on',mechanism='local_voucher'):
    events=[{'time':t,'activity_id':a,'anchor':z} for t,a,z in [
        ('07:00','home_prepare','residence'),('08:00','home_meal','residence'),
        ('12:00','meal_dine_in','zone:11680510'),('14:00','groceries','zone:11650510'),
        ('18:00','home_online_goods','residence'),('22:00','home_sleep','residence')]]
    row={'eligible':True,'execution_plan':{'events':events},'raw':json.dumps({'events':events}),'attempt_key':'K'}
    cell={'has_work':False,'zones':['11680510','11650510'],'date':'2026-09-21','user':'조건\n\n## 오늘\n계획 작성',
        'case':mechanism,'arm':arm,'synthetic_state':{'balance':100000,'grant_remaining':{'P013':280000}}}
    return row,cell,{'home_dong_code':'11680521'}


def test_quotes_shared_across_arms_but_wallet_eligibility_is_explicit():
    r,c,p=source();on,_=build_case(r,c,p);off,_=build_case(r,dict(c,arm='off'),p)
    for a,b in zip(on['events'],off['events']):
        assert [q['price_won'] for q in a['candidates']]==[q['price_won'] for q in b['candidates']]
    assert on['events'][2]['candidates'][0]['eligible_wallets']==['P014']
    assert on['events'][3]['candidates'][0]['eligible_wallets']==[]
    assert on['events'][4]['candidates'][0]['eligible_wallets']==[]
    assert off['offers']=={} and on['wallet_lots']=={}
    assert '계획 작성' not in on['context']


def test_grant_is_separate_asset_and_cashback_never_becomes_current_cash():
    r,c,p=source(mechanism='grant');grant,_=build_case(r,c,p)
    assert grant['cash']==100000 and grant['wallet_lots']=={'P013':[{'face':280000,'own_basis':0}]}
    c['case']='cashback';cb,_=build_case(r,c,p)
    assert cb['cash']==100000 and cb['wallet_lots']=={} and cb['offers']=={}


def test_closed_shop_has_no_quote_not_substitute_purchase():
    r,c,p=source();c['user']='집합금지: 유흥주점';r=copy.deepcopy(r)
    r['execution_plan']['events'][2]['activity_id']='bar';r['raw']=json.dumps(r['execution_plan'])
    case,_=build_case(r,c,p)
    assert case['events'][2]['candidates']==[]
