"""Read-only original-matrix/request/raw-plan audit for daily-state experiments."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from action_plan_contract import inspect
from action_repair_feedback import feedback
from validate_prompt_v3 import digest, atomic


def audit(folder, source_path):
    folder=Path(folder);source_path=Path(source_path)
    raw=source_path.read_bytes();source=json.loads(raw)
    manifest=json.loads((folder/'manifest.json').read_bytes());config=manifest['config']
    if hashlib.sha256(raw).hexdigest()!=manifest['inputs_sha256'] or manifest['inputs_sha256']!=config['source_inputs_sha256']:
        raise ValueError('Original source hash differs from registered input')
    frozen=json.loads((folder/'frozen_inputs.json').read_bytes())
    if frozen['personas']!=source['personas']:raise ValueError('Persona snapshot changed')
    key=lambda c:(c['aid'],c['case'],c['arm'])
    cells={key(c):c for c in frozen['cells']};original={key(c):c for c in source['cells']}
    if len(cells)!=len(frozen['cells']) or len(original)!=len(source['cells']) or set(cells)!=set(original):
        raise ValueError('Frozen condition roster differs')
    for k,c in cells.items():
        if any(c.get(field)!=value for field,value in original[k].items()):
            raise ValueError('Original condition changed inside runner')
        if c['context_sha256']!=digest(c['user']):raise ValueError('Context hash mismatch')
    rows=[json.loads(line) for line in (folder/'responses.jsonl').read_bytes().splitlines()]
    expected={(v['id'],s,*k) for v in config['candidates'] for s in config['seeds'] for k in cells}
    actual=[(r['variant'],r['replicate'],*key(r)) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Incomplete/duplicate original matrix')
    errors=[];counts=Counter();repeated=[];late_preparation=[];duplicate_commitment=[];request_files=0
    for row in rows:
        c=cells[key(row)];attempt=row['attempt_key']
        try:
            checked=inspect(row['raw'],c,max_shift=config['max_shift_minutes'])
            if feedback(row['raw'],c,max_shift=config['max_shift_minutes']) is not None:
                raise ValueError('Raw fact/execution violation')
            if checked['execution_plan']!=row['execution_plan'] or not row['eligible']:
                raise ValueError('Stored execution/eligibility differs')
            if row['answer_usage']['finish_reason']['type']!='stop':raise ValueError('Incomplete answer')
            events=checked['execution_plan']['events']
            if events[-1]['time']<config['last_start_not_before']:raise ValueError('Declared coverage')
            counts.update(e['activity_id'] for e in events)
            if any((a['activity_id'],a['anchor'])==(b['activity_id'],b['anchor']) for a,b in zip(events,events[1:])):repeated.append(attempt)
            duties=c.get('required_activities',[])
            if any(d['anchor']=='workplace' and any(e['activity_id']=='office_prepare' and e['time']>d['time'] for e in events) for d in duties):late_preparation.append(attempt)
            if any(sum(e['activity_id']==d['activity_id'] for e in events)>1 for d in duties):duplicate_commitment.append(attempt)
            first=folder/'attempts'/(attempt+'_deliberation_request.json')
            if not first.exists():first=folder/'attempts'/(attempt+'_request.json')
            prefix=json.loads(first.read_bytes())['text']
            prefix_key='|'.join([row['variant'],*key(row)])
            if digest(prefix)!=manifest['prefix_sha256'][prefix_key] or c['submitted_user'] not in prefix:
                raise ValueError('Submitted context/prefix hash mismatch')
            requests=list((folder/'attempts').glob(attempt+'*_request.json'))
            request_files+=len(requests)
            for path in requests:
                text=json.loads(path.read_bytes())['text']
                if not text.startswith(prefix) or any(word in text for word in ['own_basis','wallet_lots','evaluation_requirements','quarantined']):
                    raise ValueError('Request contains mismatched prefix or hidden ledger/evaluation field')
        except (ValueError,KeyError,TypeError,OSError) as exc:
            errors.append({'attempt_key':attempt,'error':str(exc)})
    return {'rows':len(rows),'original_matrix_complete':True,'independent_errors':errors,'request_files_audited':request_files,
            'activity_counts':dict(counts),'adjacent_duplicate_keys':repeated,'office_prepare_after_committed_start_keys':late_preparation,
            'repeated_commitment_activity_keys':duplicate_commitment,
            'scope':'Raw execution, supplied facts and request fidelity. Behavioral flags are descriptive and not post-hoc acceptance gates; optional need completion is not required spending.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ['run','source','out']:ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Refusing overwrite')
    result=audit(args.run,args.source);atomic(args.out,result)
    print(json.dumps({'rows':result['rows'],'independent_errors':len(result['independent_errors']),
                      'request_files':result['request_files_audited'],'laundry_activities':result['activity_counts'].get('provided_laundry',0)}))
