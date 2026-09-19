"""Frozen paired purchase experiment; bounded reasoning, raw inputs and failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from urllib.request import urlopen
from asset_transaction_contract import schema, inspect
from bounded_reasoning import run
from paired_asset_score import score
from prompts.asset_transaction_v1 import SYSTEM_PROMPT
from validate_prompt_v3 import atomic, digest


def invoke(job, config, base, prefixes, folder):
    seed, cell = job; case = cell['transaction_case']; key = digest([seed,cell['aid'],cell['case'],cell['arm']])
    row = {k:cell[k] for k in ['aid','case','arm','date']}; row.update(replicate=seed,attempt_key=key)
    started = time.monotonic()
    try:
        sample_seed = int(digest([seed,cell['aid'],cell['case']])[:8],16) % 2147483647
        first, second = run(prefix=prefixes[case['id']],schema=schema(case),base=base,seed=sample_seed,
            thinking_tokens=config['thinking_tokens'],answer_tokens=config['answer_tokens'],sampling=config['sampling'],
            timeout=config['timeout_seconds'],whitespace_limit=2,
            on_deliberation=lambda value: atomic(folder/'attempts'/f'{key}_deliberation.json',value))
        atomic(folder/'attempts'/f'{key}_answer.json',second)
        row.update(raw=second['response']['text'],thinking_usage=first['response']['meta_info'],
                   answer_usage=second['response']['meta_info'],forced_reasoning_boundary=first['forced_reasoning_boundary'])
        _, ledger = inspect(row['raw'],case)
        errors = [] if row['answer_usage']['finish_reason']['type']=='stop' else ['incomplete_generation']
        row.update(valid=not errors,errors=errors,ledger=ledger)
    except Exception as exc: row.update(valid=False,errors=[str(exc)])
    row['elapsed_seconds'] = round(time.monotonic()-started,3)
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--tokenizer',required=True);args=ap.parse_args()
    config=json.loads(args.config.read_bytes()); raw=args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=config['source_sha256']:raise ValueError('Input hash mismatch')
    source=json.loads(raw);cells=source['cells']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=True)
    prefixes={c['transaction_case']['id']:tokenizer.apply_chat_template(
        [{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':json.dumps(c['transaction_case'],ensure_ascii=False)}],
        tokenize=False,add_generation_prompt=True,enable_thinking=True) for c in cells}
    if len(prefixes)!=len(cells):raise ValueError('Duplicate source id')
    folder=args.out;folder.mkdir(parents=True,exist_ok=False);(folder/'attempts').mkdir();(folder/'code').mkdir()
    hashes={}
    for name in ['validate_purchase_probe.py','asset_transaction_contract.py','asset_ledger.py','transaction_ledger.py','bounded_reasoning.py','paired_asset_score.py']:
        data=Path(__file__).with_name(name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();(folder/'code'/name).write_bytes(data)
    atomic(folder/'manifest.json',{'config':config,'input_sha256':hashlib.sha256(raw).hexdigest(),'code_sha256':hashes,
        'system_sha256':digest(SYSTEM_PROMPT),'prefix_sha256':{k:digest(v) for k,v in prefixes.items()},
        'template_sha256':digest(tokenizer.chat_template),'registered_at':datetime.now(timezone.utc).isoformat()})
    atomic(folder/'frozen_inputs.json',source);(folder/'system.txt').write_text(SYSTEM_PROMPT,encoding='utf-8')
    base=os.environ.get('LLM_BASE_URL','http://localhost:8000/v1').rstrip('/').removesuffix('/v1')
    with urlopen(base+'/v1/models',timeout=10) as response:
        if config['model'] not in [m['id'] for m in json.load(response)['data']]:raise ValueError('Model mismatch')
    jobs=[(s,c) for s in config['seeds'] for c in cells];random.Random(config['order_seed']).shuffle(jobs);rows=[]
    with (folder/'responses.jsonl').open('x',encoding='utf-8') as fp,ThreadPoolExecutor(max_workers=config['workers']) as pool:
        pending=[pool.submit(invoke,j,config,base,prefixes,folder) for j in jobs]
        for future in as_completed(pending):
            row=future.result();rows.append(row);fp.write(json.dumps(row,ensure_ascii=False)+'\n');fp.flush();os.fsync(fp.fileno())
            print(f"completed {len(rows)}/{len(jobs)} {row['case']} valid={row['valid']} errors={row['errors']}",flush=True)
    expected={(s,c['aid'],c['case'],c['arm']) for s in config['seeds'] for c in cells}
    complete=len(rows)==len(expected) and {(r['replicate'],r['aid'],r['case'],r['arm']) for r in rows}==expected
    summary={'scope':'Hypothetical item/eligibility development probe; fixed upstream plans. No empirical policy-effect claim.',
        'macro_claim':False,'variants':{'asset_transaction_v1_bounded2048':{'responses':len(rows),'complete':complete,
        'valid':sum(r['valid'] for r in rows),'all_pass':complete and all(r['valid'] for r in rows)}},'contrasts':{}}
    lookup={(c['aid'],c['case'],c['arm']):c['transaction_case'] for c in cells}
    for mechanism in sorted({c['case'] for c in cells}):
        selected=[dict(r,transaction_case=lookup[(r['aid'],r['case'],r['arm'])]) for r in rows if r['case']==mechanism]
        intended=[c for c in cells if c['case']==mechanism]
        try:summary['contrasts'][mechanism]=score(selected,roster=sorted({c['aid'] for c in intended}),days=sorted({c['date'] for c in intended}),seeds=config['seeds'])
        except ValueError as exc:summary['contrasts'][mechanism]={'not_scored':str(exc)}
    atomic(folder/'summary.json',summary);print(json.dumps(summary['variants']),flush=True)


if __name__=='__main__':main()
