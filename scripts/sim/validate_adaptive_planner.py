"""Registered low-budget planning with at most one explicit constraint repair.

First-pass and repaired outcomes stay separate. No success filtering, outcome
targets, or silent retries; a remaining error blocks downstream execution.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
from urllib.request import urlopen
from action_plan_contract import catalog
from action_repair_feedback import feedback,append_feedback
from prompts.v23 import SYSTEM_PROMPT
from validate_action_planner import invoke
from validate_prompt_v3 import atomic,digest


def prefix_for(cell,tokenizer):
    user=cell['user'].replace('/no_think','').replace('/think','')
    user+='\n\n## 선택 가능한 활동 사전\n'+json.dumps(list(catalog(cell).values()),ensure_ascii=False)
    if cell.get('required_activities'):
        user+='\n\n## 입력에 명시된 일정의 실행 표기\n'+json.dumps(cell['required_activities'],ensure_ascii=False)
    if cell.get('required_presence_intervals'):
        user+='\n\n## 입력에 명시된 장소 유지 구간\n'+json.dumps(cell['required_presence_intervals'],ensure_ascii=False)
    return tokenizer.apply_chat_template([{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':user}],
        tokenize=False,add_generation_prompt=True,enable_thinking=True)


def run_cell(job,config,base,tokenizer,prefixes,folder):
    seed,cell=job
    first_candidate={'id':'initial','thinking_tokens':config['initial_thinking_tokens']}
    first=invoke((first_candidate,seed,cell),config,base,prefixes,folder)
    stages=[first];packet=None
    if not first['eligible'] and first.get('raw'):
        packet=feedback(first['raw'],cell,max_shift=config['max_shift_minutes'])
    if packet is not None:
        repaired=dict(cell,user=append_feedback(cell['user'],packet))
        candidate={'id':'repair','thinking_tokens':config['repair_thinking_tokens']}
        repair_key=digest(['repair',seed,cell['aid'],cell['case'],cell['arm']])
        atomic(folder/'attempts'/f'{repair_key}_feedback.json',{'parent_attempt_key':first['attempt_key'],'packet':packet,'submitted_cell':repaired})
        p={('repair',cell['aid'],cell['case'],cell['arm']):prefix_for(repaired,tokenizer)}
        stages.append(invoke((candidate,seed,repaired),config,base,p,folder))
    final=stages[-1]
    return {k:cell[k] for k in ['aid','case','arm','date']} | {'replicate':seed,'stages':stages,
        'first_eligible':first['eligible'],'first_raw_valid':first.get('raw_valid',False),
        'eligible':final['eligible'],'valid':final['valid'],'raw':final.get('raw'),
        'execution_plan':final.get('execution_plan'),'repair_attempted':len(stages)==2,
        'final_errors':final['errors'],'factual_errors':final.get('factual_errors',[]),
        'generated_tokens':sum(r.get('thinking_usage',{}).get('completion_tokens',0)+r.get('answer_usage',{}).get('completion_tokens',0) for r in stages)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--tokenizer',required=True);args=ap.parse_args()
    config=json.loads(args.config.read_bytes());raw=args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=config['source_sha256']:raise ValueError('Source hash mismatch')
    source=json.loads(raw);people={p['id']:p for p in source['personas']}
    for c in source['cells']:c['has_work']=bool(people[c['aid']].get('work_poi_id'))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=True)
    prefixes={('initial',c['aid'],c['case'],c['arm']):prefix_for(c,tokenizer) for c in source['cells']}
    folder=args.out;folder.mkdir(parents=True,exist_ok=False);(folder/'attempts').mkdir();(folder/'code').mkdir();hashes={}
    for name in ['validate_adaptive_planner.py','validate_action_planner.py','action_repair_feedback.py','action_plan_contract.py','presence_contract.py','temporal_projection.py','bounded_reasoning.py']:
        data=Path(__file__).with_name(name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();(folder/'code'/name).write_bytes(data)
    atomic(folder/'manifest.json',{'config':config,'input_sha256':hashlib.sha256(raw).hexdigest(),'code_sha256':hashes,'system_sha256':digest(SYSTEM_PROMPT),
        'prefix_sha256':{'|'.join(k):digest(v) for k,v in prefixes.items()},'template_sha256':digest(tokenizer.chat_template),'registered_at':datetime.now(timezone.utc).isoformat()})
    atomic(folder/'frozen_inputs.json',source);(folder/'system.txt').write_text(SYSTEM_PROMPT,encoding='utf-8')
    base=os.environ.get('LLM_BASE_URL','http://localhost:8000/v1').rstrip('/').removesuffix('/v1')
    with urlopen(base+'/v1/models',timeout=10) as r:
        if config['model'] not in [m['id'] for m in json.load(r)['data']]:raise ValueError('Model mismatch')
    jobs=[(s,c) for s in config['seeds'] for c in source['cells']];random.Random(config['order_seed']).shuffle(jobs);rows=[]
    with (folder/'responses.jsonl').open('x',encoding='utf-8') as fp,ThreadPoolExecutor(max_workers=config['workers']) as pool:
        pending=[pool.submit(run_cell,j,config,base,tokenizer,prefixes,folder) for j in jobs]
        for future in as_completed(pending):
            r=future.result();rows.append(r);fp.write(json.dumps(r,ensure_ascii=False)+'\n');fp.flush();os.fsync(fp.fileno())
            print(f"completed {len(rows)}/{len(jobs)} {r['case']} first={r['first_eligible']} final={r['eligible']} repair={r['repair_attempted']}",flush=True)
    expected={(s,c['aid'],c['case'],c['arm']) for s in config['seeds'] for c in source['cells']}
    complete=len(rows)==len(expected) and {(r['replicate'],r['aid'],r['case'],r['arm']) for r in rows}==expected
    summary={'scope':'Adaptive execution protocol on known development cases. First failures preserved; at most one factual repair. No policy effect or untouched confirmation claim.',
        'macro_claim':False,'variants':{'v23_adaptive':{'responses':len(rows),'complete':complete,'first_raw_valid':sum(r['first_raw_valid'] for r in rows),
        'first_eligible':sum(r['first_eligible'] for r in rows),'repair_attempted':sum(r['repair_attempted'] for r in rows),'eligible':sum(r['eligible'] for r in rows),
        'all_pass':complete and all(r['eligible'] for r in rows),'mean_generated_tokens':sum(r['generated_tokens'] for r in rows)/len(rows)}}}
    atomic(folder/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
