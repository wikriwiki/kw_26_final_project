"""Registered reasoning ablation. Frozen inputs, raw responses, no repairs/retries."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
import argparse
import hashlib
import json
import os
import random
import time

from validate_prompt_v3 import ROOT, atomic, digest
from validate_prompt_v4 import prepare, summarize
from planning_contract import inspect_schedule, schedule_schema


def invoke(job, config, systems, base):
    candidate, rep, cell = job
    seed = int(digest([rep, cell['aid'], cell['case']])[:8], 16) % 2147483647
    out = {k: cell[k] for k in ['aid', 'case', 'arm', 'date', 'context_sha256']}
    out.update(variant=candidate['id'], replicate=rep, seed=seed,
               structured=True, thinking=candidate['thinking'])
    schema = schedule_schema(cell['zones'], date.fromisoformat(cell['date']).weekday() >= 5, cell['has_work'])
    payload = dict(model=config['model'], messages=[
        {'role': 'system', 'content': systems[candidate['id']]},
        {'role': 'user', 'content': cell['user']}],
        temperature=config['temperature'], top_p=config['top_p'],
        max_tokens=config['max_tokens'], seed=seed,
        chat_template_kwargs={'enable_thinking': candidate['thinking']},
        response_format={'type':'json_schema', 'json_schema':{'name':'citizen_day','strict':True,'schema':schema}})
    if 'presence_penalty' in config:
        payload['presence_penalty'] = config['presence_penalty']
    out.update(schema_sha256=digest(schema), request_sha256=digest(payload))
    start = time.monotonic()
    try:
        req = Request(base + '/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        with urlopen(req, timeout=config['timeout_seconds']) as response:
            response = json.load(response)
        choice = response['choices'][0]
        raw = choice['message'].get('content') or ''
        obj, errors, flags = inspect_schedule(raw, cell)
        if choice.get('finish_reason') != 'stop':
            errors.append('incomplete_generation')
        out.update(raw=raw, errors=errors, valid=not errors, semantic_flags=flags,
                   reasoning_content=choice['message'].get('reasoning_content'),
                   finish_reason=choice.get('finish_reason'), usage=response.get('usage'),
                   response=response, propensity=(obj or {}).get('daily_propensity'))
    except Exception as exc:
        out.update(valid=False, errors=['request_or_response'], error=str(exc), semantic_flags=[])
    out['elapsed_seconds'] = round(time.monotonic()-start, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=False)
    inputs = prepare(args.source, config)
    for cell in inputs['cells']:
        cell['user'] = cell['user'].replace('/no_think', '').replace('/think', '')
        cell['context_sha256'] = digest(cell['user'])
    base = os.environ.get('LLM_BASE_URL', 'http://localhost:8000/v1').rstrip('/')
    with urlopen(base + '/models', timeout=10) as response:
        assert config['model'] in [m['id'] for m in json.load(response)['data']]
    with urlopen(base.removesuffix('/v1') + '/server_info', timeout=10) as response:
        server = json.load(response)
    assert server.get('reasoning_parser') == 'qwen3'
    files = [Path(__file__), ROOT/'scripts/sim/planning_contract.py', ROOT/'scripts/sim/validate_prompt_v3.py', ROOT/'scripts/sim/validate_prompt_v4.py']
    manifest = {'config':config, 'config_sha256':digest(config), 'inputs_sha256':digest(inputs),
                'registered_at':datetime.now(timezone.utc).isoformat(),
                'source_sha256':hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
                'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'system_hashes':{k:digest(v) for k,v in inputs['systems'].items()},
                'server_info_sha256':digest(server)}
    atomic(folder/'server_info.json', server)
    atomic(folder/'frozen_inputs.json', inputs)
    atomic(folder/'manifest.json', manifest)
    jobs = [(c,rep,cell) for c in config['candidates'] for rep in config['replicate_seeds'] for cell in inputs['cells']]
    random.Random(config['job_order_seed']).shuffle(jobs)
    rows = []
    with (folder/'responses.jsonl').open('x', encoding='utf-8') as fp, ThreadPoolExecutor(max_workers=config['workers']) as pool:
        pending = [pool.submit(invoke, job, config, inputs['systems'], base) for job in jobs]
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            fp.write(json.dumps(row, ensure_ascii=False)+'\n'); fp.flush(); os.fsync(fp.fileno())
            print(f"completed {len(rows)}/{len(jobs)} {row['variant']} valid={row['valid']} errors={row['errors']}", flush=True)
    summary = summarize(rows, config, inputs['cells'])
    for candidate in config['candidates']:
        rr = [r for r in rows if r['variant']==candidate['id']]
        summary['variants'][candidate['id']]['reasoning_responses'] = sum(bool(r.get('reasoning_content')) for r in rr)
        summary['variants'][candidate['id']]['mean_elapsed_seconds'] = sum(r['elapsed_seconds'] for r in rr)/len(rr)
    atomic(folder/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
