"""Registered bounded-deliberation planning experiment; file-only frozen contexts."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import date,datetime,timezone
import hashlib,json,os,random,time
from pathlib import Path
from urllib.request import urlopen
from bounded_reasoning import run
from planning_contract import inspect_schedule,schedule_schema,endpoint_schedule_schema
from validate_prompt_v3 import ROOT,atomic,digest
from validate_prompt_v4 import prepare
from evidence_contract import evidence_atoms,constrain_evidence,constrain_field,constrain_trigger_evidence
from temporal_projection import project,transition_violations


def invoke(job,config,base,prefixes,folder):
    c,rep,cell=job
    key=digest([c['id'],rep,cell['aid'],cell['case'],cell['arm']])
    row={k:cell[k] for k in ['aid','case','arm','date','context_sha256']}
    row.update(variant=c['id'],replicate=rep,protocol='bounded_deliberation_then_json')
    seed=int(digest([rep,cell['aid'],cell['case']])[:8],16)%2147483647
    start=time.monotonic()
    try:
        builder=endpoint_schedule_schema if config.get('schema_mode')=='endpoint_guard' else schedule_schema
        s=builder(cell['zones'],date.fromisoformat(cell['date']).weekday()>=5,cell['has_work'])
        atoms=evidence_atoms(cell['user']) if config.get('reasoning_evidence',False) else None
        if atoms is not None: s=constrain_evidence(s,atoms)
        if config.get('input_trigger_guard'): s=constrain_field(s,'trigger',cell['allowed_triggers'])
        if config.get('trigger_evidence_guard'): s=constrain_trigger_evidence(s,cell['trigger_evidence'])
        row.update(schema_mode=config.get('schema_mode','standard'),schema_sha256=digest(s))
        first,second=run(prefix=prefixes[(c['id'],cell['context_sha256'])],schema=s,base=base,seed=seed,
                         thinking_tokens=c['thinking_tokens'],answer_tokens=config['answer_tokens'],
                         sampling=config['sampling'],timeout=config['timeout_seconds'],
                         whitespace_limit=config.get('whitespace_limit'),
                         on_deliberation=lambda value:atomic(folder/'attempts'/f'{key}_deliberation.json',value))
        atomic(folder/'attempts'/f'{key}_answer.json',second)
        raw=second['response']['text']; obj,errors,flags=inspect_schedule(raw,cell)
        if atoms is not None and obj is not None and any(e.get('reasoning') not in atoms for e in obj.get('events',[])):
            errors.append('unsupported_evidence_span')
        if config.get('input_trigger_guard') and obj is not None and any(e.get('trigger') not in cell['allowed_triggers'] for e in obj.get('events',[])):
            errors.append('unavailable_trigger')
        if config.get('trigger_evidence_guard') and obj is not None and any(e.get('trigger') in {'policy','appointment','rumor'} and e.get('reasoning') not in cell['trigger_evidence'].get(e['trigger'],[]) for e in obj.get('events',[])):
            errors.append('wrong_evidence_channel')
        reason=second['response']['meta_info'].get('finish_reason',{})
        if reason.get('type')!='stop': errors.append('incomplete_generation')
        if obj is not None:
            travel_errors=transition_violations(obj,cell.get('minimum_transitions',[]))
            row['raw_transition_violations']=travel_errors
            if travel_errors: errors.append('transition_time')
        row.update(raw_valid=not errors,raw_errors=list(errors))
        if config.get('temporal_projection') and obj is not None and set(errors)<= {'time','transition_time'}:
            try:
                projected,projection=project(obj,max_shift=config['temporal_projection']['max_shift_minutes'],
                                             gap=20,fixed_times=cell['fixed_times'],transitions=cell.get('minimum_transitions',[]))
                obj,errors,flags=inspect_schedule(json.dumps(projected,ensure_ascii=False),cell)
                if transition_violations(projected,cell.get('minimum_transitions',[])): errors.append('transition_time')
                row.update(execution_plan=projected,temporal_projection=projection)
            except ValueError as exc:
                errors=list(errors)+['temporal_projection_infeasible']
                row['projection_error']=str(exc)
        row.update(raw=raw,valid=not errors,errors=errors,semantic_flags=flags,
                   forced_reasoning_boundary=first['forced_reasoning_boundary'],
                   thinking_usage=first['response']['meta_info'],answer_usage=second['response']['meta_info'],
                   attempt_key=key,propensity=(obj or {}).get('daily_propensity'))
    except Exception as exc:
        row.update(valid=False,errors=['request_or_response'],error=str(exc),semantic_flags=[])
    row['elapsed_seconds']=round(time.monotonic()-start,3)
    return row


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--source',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--tokenizer',required=True)
    args=ap.parse_args();config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config.get('schema_mode','standard') not in {'standard','endpoint_guard'}: raise ValueError('Unknown schema mode')
    if config.get('trigger_evidence_guard') and not (config.get('reasoning_evidence') and config.get('input_trigger_guard')):
        raise ValueError('Causal evidence requires explicit input/evidence guards')
    folder=Path(args.out);folder.mkdir(parents=True,exist_ok=False);(folder/'attempts').mkdir()
    inputs=prepare(args.source,config)
    for cell in inputs['cells']:
        if config.get('input_trigger_guard'):
            if not cell.get('allowed_triggers') or not set(cell['allowed_triggers'])<={'appointment','rumor','policy','lifestyle','mood','none'}:
                raise ValueError('Missing/invalid typed trigger availability')
        if config.get('temporal_projection') and 'fixed_times' not in cell: raise ValueError('Missing fixed-time metadata')
    for cell in inputs['cells']:
        cell['user']=cell['user'].replace('/no_think','').replace('/think','')
        cell['context_sha256']=digest(cell['user'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=True)
    prefixes={}
    for c in config['candidates']:
        for cell in inputs['cells']:
            prefixes[(c['id'],cell['context_sha256'])]=tokenizer.apply_chat_template(
                [{'role':'system','content':inputs['systems'][c['id']]},{'role':'user','content':cell['user']}],
                tokenize=False,add_generation_prompt=True,enable_thinking=True)
    base=os.environ.get('LLM_BASE_URL','http://localhost:8000/v1').rstrip('/').removesuffix('/v1')
    with urlopen(base+'/v1/models',timeout=10) as response: assert config['model'] in [m['id'] for m in json.load(response)['data']]
    code={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
          ['validate_bounded_planner.py','bounded_reasoning.py','planning_contract.py','validate_prompt_v3.py','validate_prompt_v4.py','evidence_contract.py','temporal_projection.py']}
    manifest={'config':config,'config_sha256':digest(config),'inputs_sha256':digest(inputs),'code_sha256':code,
              'registered_at':datetime.now(timezone.utc).isoformat(),'tokenizer_path':args.tokenizer,
              'template_sha256':digest(tokenizer.chat_template),'prefix_hashes':{a+'|'+b:digest(v) for (a,b),v in prefixes.items()}}
    atomic(folder/'manifest.json',manifest);atomic(folder/'frozen_inputs.json',inputs)
    (folder/'code').mkdir()
    for name in code:
        (folder/'code'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    jobs=[(c,rep,cell) for c in config['candidates'] for rep in config['seeds'] for cell in inputs['cells']]
    random.Random(config['order_seed']).shuffle(jobs);rows=[]
    with (folder/'responses.jsonl').open('x',encoding='utf-8') as fp,ThreadPoolExecutor(max_workers=config['workers']) as pool:
        pending=[pool.submit(invoke,j,config,base,prefixes,folder) for j in jobs]
        for future in as_completed(pending):
            row=future.result();rows.append(row);fp.write(json.dumps(row,ensure_ascii=False)+'\n');fp.flush();os.fsync(fp.fileno())
            print(f"completed {len(rows)}/{len(jobs)} {row['variant']} valid={row['valid']} errors={row['errors']} error={row.get('error','')}",flush=True)
    summary={'macro_claim':False,'first_response_only_protocol':False,'variants':{}}
    for c in config['candidates']:
        rr=[r for r in rows if r['variant']==c['id']]
        expected={(rep,cell['aid'],cell['case'],cell['arm']) for rep in config['seeds'] for cell in inputs['cells']}
        actual=Counter((r['replicate'],r['aid'],r['case'],r['arm']) for r in rr)
        complete=set(actual)==expected and all(n==1 for n in actual.values())
        rate=sum(r['valid'] for r in rr)/len(rr) if rr else 0
        summary['variants'][c['id']]={'responses':len(rr),'valid':sum(r['valid'] for r in rr),
            'complete_unique_matrix':complete,'valid_rate':rate,
            'protocol_format_gate':complete and rate>=.95 and not any('error' in r for r in rr),
            'all_pass':complete and all(r['valid'] for r in rr),
            'raw_valid':sum(r.get('raw_valid',False) for r in rr),
            'adjusted_responses':sum(bool(r.get('temporal_projection',{}).get('shifts')) for r in rr),
            'semantic_flags':dict(Counter(f['kind'] for r in rr for f in r['semantic_flags'])),
            'mean_generated_tokens':sum(r.get('thinking_usage',{}).get('completion_tokens',0)+r.get('answer_usage',{}).get('completion_tokens',0) for r in rr)/len(rr) if rr else None,
            'errors':dict(Counter(e for r in rr for e in r['errors']))}
    atomic(folder/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
