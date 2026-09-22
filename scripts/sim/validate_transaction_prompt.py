"""Synthetic accounting/choice development cases; independent of empirical effects."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from urllib.request import Request, urlopen
from prompts.transaction_v2 import SYSTEM_PROMPT
from transaction_contract import schema, inspect
from validate_prompt_v3 import atomic, digest


def make_cases():
    def candidate(pid, price, eligible=()):
        return {'poi_id':pid,'listed_price_won':price,'eligible_wallets':list(eligible)}
    def event(order,intent,candidates,channel='offline'):
        return dict(order=order,intent=intent,candidates=candidates,channel=channel)
    return [
        {'id':'free_walk','cash':8000,'wallets':{},'context':'오늘은 집 앞 공원을 산책만 하고 돌아오기로 했다. 음식·음료 구매 계획이나 필요는 없다.',
         'events':[event(0,'공원 산책',[candidate('CAFE',4000)])],
         'requirements':[{'order':0,'poi_id':None,'actual_spent':0,'policy_spend':{}}]},
        {'id':'future_funds','cash':0,'wallets':{},'context':'현재 쓸 수 있는 돈은 없다. 별도 지원 재원은 내일 입금 예정이고 지금 사용하거나 차입할 수 없다. 구매 희망은 내일로 미룰 수 있다.',
         'events':[event(0,'음료 구매를 검토',[candidate('CAFE',4000)])],
         'requirements':[{'order':0,'actual_spent':0,'policy_spend':{}}]},
        {'id':'fixed_necessary_purchase','cash':4000,'wallets':{},'context':'오늘 아침에 먹을 빵 한 개가 필요하고, 아래 가게에서 표시 가격 2000원인 빵 한 개를 사기로 확정했다. 추가 구매는 없다.',
         'events':[event(0,'빵 한 개 구매',[candidate('BAKERY',2000)])],
         'requirements':[{'order':0,'poi_id':'BAKERY','actual_spent':2000,'policy_spend':{}}]},
        {'id':'ineligible_wallet','cash':0,'wallets':{'W':6000},'context':'W는 현재 사용 가능한 별도 지갑이나 eligible_wallets에 W가 있는 거래에만 쓸 수 있다. 외상이나 차입은 없다. 아래 구매는 희망 사항이며 미룰 수 있다.',
         'events':[event(0,'물품 구매 검토',[candidate('SHOP',3000)])],
         'requirements':[{'order':0,'actual_spent':0,'policy_spend':{}}]},
        {'id':'mixed_funding','cash':1500,'wallets':{'W':3500},'context':'필요한 식료품 한 묶음의 가격은 5000원이며 오늘 한 묶음만 사기로 확정했다. W가 적격인 거래에서는 개인 돈과 W를 함께 결제할 수 있고 부분 결제가 가능하다. 차입과 외상은 없다.',
         'events':[event(0,'식료품 한 묶음 구매',[candidate('MARKET',5000,['W'])])],
         'requirements':[{'order':0,'poi_id':'MARKET','actual_spent':5000,'policy_spend':{'W':3500}}]},
        {'id':'shared_wallet','cash':0,'wallets':{'W':4000},'context':'W는 현재 사용 가능한 지갑이며 적격 거래에 쓸 수 있다. 오늘 필요한 두 품목을 아래 서로 다른 가게에서 한 개씩 사기로 확정했다. 각각 2000원이며 추가 구매는 없다.',
         'events':[event(0,'품목 A 한 개',[candidate('SHOP_A',2000,['W'])]),event(1,'품목 B 한 개',[candidate('SHOP_B',2000,['W'])])],
         'requirements':[{'order':0,'poi_id':'SHOP_A','actual_spent':2000,'policy_spend':{'W':2000}}, {'order':1,'poi_id':'SHOP_B','actual_spent':2000,'policy_spend':{'W':2000}}]},
        {'id':'online_offline','cash':2000,'wallets':{'W':2000},'context':'W는 적격 거래에서만 사용 가능하며 개인 돈과 함께 결제할 수 있다. 오늘 필요한 물품 A와 B를 각각 한 개씩 아래 가게와 온라인 상점에서 사기로 확정했다. 각 가격은 2000원이고 추가 구매는 없다. 외상과 차입은 없다.',
         'events':[event(0,'물품 A 한 개',[candidate('LOCAL',2000,['W'])]),event(1,'물품 B 한 개',[candidate('WEB',2000)],'online')],
         'requirements':[{'order':0,'poi_id':'LOCAL','actual_spent':2000,'policy_spend':{'W':2000}}, {'order':1,'poi_id':'WEB','actual_spent':2000,'policy_spend':{}}]}
    ]


def invoke(job,config,base):
    thinking,rep,case=job
    user={k:v for k,v in case.items() if k!='requirements'}
    payload={'model':config['model'],'messages':[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':json.dumps(user,ensure_ascii=False)}],
             'temperature':config['temperature'],'top_p':config['top_p'],'presence_penalty':config['presence_penalty'],
             'max_tokens':config['max_tokens'],'seed':int(digest([rep,case['id']])[:8],16)%2147483647,
             'chat_template_kwargs':{'enable_thinking':thinking},
             'response_format':{'type':'json_schema','json_schema':{'name':'transactions','strict':True,'schema':schema(case)}}}
    out={'case':case['id'],'thinking':thinking,'replicate':rep,'request_sha256':digest(payload)}
    started=time.monotonic()
    try:
        req=Request(base+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=config['timeout_seconds']) as response: response=json.load(response)
        out['response']=response
        choice=response['choices'][0]; raw=choice['message'].get('content') or ''
        out['raw']=raw
        obj,ledger,errors=inspect(raw,case)
        if choice.get('finish_reason')!='stop': errors.append('incomplete_generation')
        out.update(valid=not errors,errors=errors,ledger=ledger,finish_reason=choice.get('finish_reason'),usage=response.get('usage'))
    except Exception as exc:
        out.update(valid=False,errors=[str(exc)])
    out['elapsed_seconds']=round(time.monotonic()-started,3)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--out',required=True); ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args(); config=json.loads(Path(args.config).read_text(encoding='utf-8')); cases=make_cases()
    folder=Path(args.out)
    if (folder/'responses.jsonl').exists(): raise ValueError('Refusing overwrite/retry')
    current={'config':config,'config_sha256':digest(config),'cases_sha256':digest(cases),'system_sha256':digest(SYSTEM_PROMPT),
             'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'contract_sha256':hashlib.sha256(Path(__file__).with_name('transaction_contract.py').read_bytes()).hexdigest(),
             'ledger_sha256':hashlib.sha256(Path(__file__).with_name('transaction_ledger.py').read_bytes()).hexdigest()}
    if folder.exists():
        saved=json.loads((folder/'manifest.json').read_text())
        assert all(saved[k]==v for k,v in current.items())
    else:
        folder.mkdir(parents=True)
        atomic(folder/'manifest.json',current|{'registered_at':datetime.now(timezone.utc).isoformat()})
        atomic(folder/'frozen_cases.json',cases)
        (folder/'system.txt').write_text(SYSTEM_PROMPT,encoding='utf-8')
    if args.prepare_only: print('Frozen seven development cases; no model calls.'); return
    base=os.environ.get('LLM_BASE_URL','http://localhost:8000/v1').rstrip('/')
    with urlopen(base+'/models') as response: assert config['model'] in [m['id'] for m in json.load(response)['data']]
    jobs=[(t,rep,c) for t in config['thinking_modes'] for rep in config['seeds'] for c in cases]
    random.Random(config['order_seed']).shuffle(jobs); rows=[]
    with (folder/'responses.jsonl').open('x',encoding='utf-8') as fp,ThreadPoolExecutor(max_workers=config['workers']) as pool:
        futures=[pool.submit(invoke,j,config,base) for j in jobs]
        for future in as_completed(futures):
            row=future.result(); rows.append(row); fp.write(json.dumps(row,ensure_ascii=False)+'\n'); fp.flush(); os.fsync(fp.fileno())
            print(f"completed {len(rows)}/{len(jobs)} thinking={row['thinking']} case={row['case']} valid={row['valid']} errors={row['errors']}",flush=True)
    summary={'macro_claim':False,'scope':'Synthetic factual/financial feasibility checks, not policy effect targets','modes':{}}
    for mode in config['thinking_modes']:
        rr=[r for r in rows if r['thinking']==mode]
        expected={(c['id'],rep) for c in cases for rep in config['seeds']}
        complete=len(rr)==len(expected) and {(r['case'],r['replicate']) for r in rr}==expected
        summary['modes'][str(mode)]={'complete':complete,'responses':len(rr),'valid':sum(r['valid'] for r in rr),'all_pass':complete and all(r['valid'] for r in rr)}
    atomic(folder/'summary.json',summary); print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
