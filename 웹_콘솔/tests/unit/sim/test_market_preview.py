import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from action_plan_contract import inspect,catalog
from prepare_market_preview import prepare
from prepare_purchase_probe import preview_for,base_quote,build_case,render_preview


def source():
    p={'id':'A','home_dong_code':'11680521','work_poi_id':None}
    c={'aid':'A','case':'local_voucher','arm':'on','date':'2026-09-21','has_work':False,
       'zones':['11680510','11650510'],'user':'기존 시민·환경 정보\n\n## 오늘\n일정 작성',
       'synthetic_state':{'balance':100000,'grant_remaining':{'P013':280000}}}
    return {'personas':[p],'cells':[c]}


def test_preview_addition_preserves_original_facts_and_is_not_reapplied():
    s=source();old=copy.deepcopy(s);new=prepare(s);c=new['cells'][0]
    assert s==old and c['user'].replace(render_preview(c['purchase_preview']),'')==s['cells'][0]['user']
    assert 'own_basis' not in json.dumps(c['purchase_preview']) and 'wallet_lots' not in json.dumps(c['purchase_preview'])
    with pytest.raises(ValueError,match='already supplied'):prepare(new)


def test_policy_only_changes_explicit_financing_not_menu_prices_or_goods():
    s=source();c=s['cells'][0];p=s['personas'][0]
    on=preview_for(c,p);off=preview_for(dict(c,arm='off'),p)
    assert on['candidates']==off['candidates'] and not off['offers'] and not off['wallet_acceptance']
    group=on['wallet_acceptance'][0]
    assert group['anchors']==['zone:11680510'] and 'groceries' in group['activity_ids']
    assert 'home_online_goods' not in group['activity_ids']


def test_each_preview_quote_uses_the_purchase_provider():
    s=source();c=s['cells'][0];p=s['personas'][0]
    preview=preview_for(c,p)
    for candidate in preview['candidates']:
        q=base_quote(candidate['activity_id'],catalog(c)[candidate['activity_id']]['anchors'][0],c,p)
        assert candidate['price_won']==q['price_won'] and candidate['description']==q['description']
        assert candidate['candidate_id']==q['id'] and candidate['channel']==q['channel']


def test_purchase_cannot_change_price_after_plan_or_hide_preview_from_request():
    s=prepare(source());c=s['cells'][0];p=s['personas'][0]
    events=[{'time':t,'activity_id':a,'anchor':z} for t,a,z in [
        ('07:00','home_prepare','residence'),('08:00','home_meal','residence'),
        ('12:00','groceries','zone:11680510'),('15:00','home_leisure','residence'),
        ('19:00','home_meal','residence'),('22:00','home_sleep','residence')]]
    raw=json.dumps({'events':events});row={'eligible':True,'raw':raw,'attempt_key':'K','execution_plan':inspect(raw,c,max_shift=0)['execution_plan']}
    case,_=build_case(row,c,p,max_shift_minutes=0,static_offers=True)
    assert case['events'][2]['candidates'][0]['price_won']==5000
    with pytest.raises(ValueError,match='differ from preview'):build_case(row,c,p,price_factor=2)
    bad=copy.deepcopy(c);bad['purchase_preview']['candidates'][0]['price_won']+=1
    with pytest.raises(ValueError,match='differ from preview'):build_case(row,bad,p)
    bad=copy.deepcopy(c);bad['user']=source()['cells'][0]['user']
    with pytest.raises(ValueError,match='differ from preview'):build_case(row,bad,p)
