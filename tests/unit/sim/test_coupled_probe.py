import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from action_plan_contract import inspect
from report_coupled_probe import report


def fixture(root):
    plans=root/'plans';purchases=root/'purchases';plans.mkdir();purchases.mkdir()
    cells=[];rows=[];quotes=[];transactions=[]
    events=[{'time':t,'activity_id':a,'anchor':'residence'} for t,a in [('07:00','home_prepare'),('08:00','home_meal'),('12:00','home_meal'),('15:00','home_leisure'),('19:00','home_meal'),('22:00','home_sleep')]]
    raw=json.dumps({'events':events})
    for arm in ['off','on']:
        cell={'aid':'A','case':'mechanism','arm':arm,'date':'2026-09-21','has_work':False,'zones':[],'user':'명시된 개인 휴일'}
        executed=inspect(raw,cell,max_shift=0)['execution_plan']
        row={k:cell[k] for k in ['aid','case','arm','date']}|{'attempt_key':arm,'eligible':True,'raw':raw,'execution_plan':executed,'answer_usage':{'finish_reason':{'type':'stop'}}}
        case={'id':arm,'cash':1000,'wallet_lots':{},'offers':{},'events':[
            {k:e[k] for k in ['time','anchor','activity_id','intent']}|{'id':'event:'+str(i),'channel':'offline','candidates':[]}
            for i,e in enumerate(executed['events'])]}
        quotes.append({k:row[k] for k in ['aid','case','arm','date','attempt_key']}|{'transaction_case':case,'bridge_audit':{'schedule_report':{'execution_plan':executed}}})
        transactions.append({k:cell[k] for k in ['aid','case','arm','date']}|{'replicate':1,'valid':True,'transaction_protocol':'v4','raw':json.dumps({'acquisition_units':{},'purchases':[{'id':'event:'+str(i),'candidate_id':None,'wallet_spend':{}} for i in range(len(events))]})})
        cells.append(cell);rows.append(row)
    (plans/'manifest.json').write_text(json.dumps({'config':{'max_shift_minutes':0,'last_start_not_before':'18:00'}}))
    (plans/'frozen_inputs.json').write_text(json.dumps({'cells':cells}))
    (plans/'responses.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    hashes={name:hashlib.sha256((plans/name).read_bytes()).hexdigest() for name in ['manifest.json','frozen_inputs.json','responses.jsonl']}
    (purchases/'frozen_inputs.json').write_text(json.dumps({'source_sha256':hashes,'cells':quotes}))
    (purchases/'manifest.json').write_text(json.dumps({'config':{'seeds':[1]}}))
    (purchases/'responses.jsonl').write_text('\n'.join(json.dumps(r) for r in transactions)+'\n')
    return plans,purchases


def test_matched_valid_zero_remains_in_complete_conditional_matrix(tmp_path):
    a,b=fixture(tmp_path);r=report(a,b)
    assert r['all_matrices_complete'] and not r['macro_validated']
    metric=r['mechanisms']['mechanism']['matched_contrasts']['by_seed'][0]['metrics']['total_consumption']
    assert metric['difference']==0 and metric['relative_change'] is None


def test_changed_upstream_file_or_wrong_decision_link_rejected(tmp_path):
    a,b=fixture(tmp_path);p=b/'frozen_inputs.json';s=json.loads(p.read_bytes());s['cells'][0]['transaction_case']['id']='wrong';p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='wrong upstream'):report(a,b)
    (a/'responses.jsonl').write_text('')
    with pytest.raises(ValueError,match='does not match'):report(a,b)


def test_failed_purchase_not_zero_filled_or_removed(tmp_path):
    a,b=fixture(tmp_path);p=b/'responses.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[0]['valid']=False
    p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    r=report(a,b)
    assert not r['all_matrices_complete'] and 'not_scored' in r['mechanisms']['mechanism']


def test_quote_source_cannot_drop_a_planned_activity(tmp_path):
    a,b=fixture(tmp_path);p=b/'frozen_inputs.json';s=json.loads(p.read_bytes());s['cells'][0]['transaction_case']['events'].pop();p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='dropped planned'):report(a,b)
