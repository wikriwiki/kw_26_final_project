"""Prepare explicit retrospective repair cases from a completed registered run."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from action_repair_feedback import feedback, append_feedback
from validate_prompt_v3 import atomic, digest


def prepare(run, *, audit_presence=False, resource_verdicts=None):
    run=Path(run)
    summary=json.loads((run/'summary.json').read_bytes())
    if not all(v['complete'] for v in summary['variants'].values()):raise ValueError('Source run incomplete')
    source=json.loads((run/'frozen_inputs.json').read_bytes())
    cells={(c['aid'],c['case'],c['arm']):c for c in source['cells']}
    rows=[json.loads(line) for line in (run/'responses.jsonl').read_bytes().splitlines()]
    prepared=[];parents=[];unrepairable=[]
    # A gate-blocked cell is contract-valid, so selecting on `eligible` alone skipped exactly
    # the failures the resource gate exists to catch.
    verdicts = resource_verdicts or {}
    for row in rows:
        verdict = verdicts.get((row['aid'], row['case'], row['arm']))
        blocked = bool((verdict or {}).get('shortfalls'))
        if row['eligible'] and not audit_presence and not blocked:continue
        if not row.get('raw'):
            unrepairable.append(row['attempt_key']);continue
        cell=copy.deepcopy(cells[(row['aid'],row['case'],row['arm'])])
        if audit_presence and row['case'].startswith('work_conflict'):
            from grounding_stress import EXTRA
            evidence=EXTRA['work_conflict']
            if evidence not in cell['user']:raise ValueError('Not the registered synthetic work interval source')
            cell['required_presence_intervals']=[{'start':'09:00','end':'17:00','anchor':'workplace','evidence':evidence}]
        packet=feedback(row['raw'],cell,resource=verdict)
        if packet is None:
            if row['eligible'] and not blocked:continue
            raise ValueError('Reported failure not reproduced by current execution checks')
        cell.pop('submitted_user',None)
        cell['case']=cell['case']+'__repair_'+row['attempt_key'][:12]
        cell['user']=append_feedback(cell['user'],packet);cell['context_sha256']=digest(cell['user'])
        prepared.append(cell);parents.append({'aid':cell['aid'],'case':cell['case'],'arm':cell['arm'],
            'original_attempt_key':row['attempt_key'],'original_variant':row['variant'],'original_seed':row['replicate'],
            'original_raw_sha256':hashlib.sha256(row['raw'].encode()).hexdigest(),'feedback':packet})
    return {'personas':source['personas'],'cells':prepared,'repair_parents':parents,
            'unrepairable_attempts':unrepairable,'presence_audit_added':audit_presence,'resource_blocked_included':sum(1 for p in parents if p['feedback'].get('resource_shortfalls')),'source_responses_sha256':hashlib.sha256((run/'responses.jsonl').read_bytes()).hexdigest(),
            'scope':'Retrospective development repair of every reproducible failed output. Not a new first-pass or untouched confirmation result. No successful output regenerated.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--audit-presence',action='store_true')
    ap.add_argument('--resource-source',help='purchase_source json carrying the gate verdicts')
    args=ap.parse_args()
    verdicts=None
    if args.resource_source:
        src=json.loads(Path(args.resource_source).read_bytes())
        verdicts={(c['aid'],c['case'],c['arm']):c.get('resource_feasibility') for c in src['cells']}
    if args.out.exists():raise ValueError('Refusing overwrite')
    result=prepare(args.run,audit_presence=args.audit_presence,resource_verdicts=verdicts);atomic(args.out,result)
    print(json.dumps({'cells':len(result['cells']),'unrepairable':len(result['unrepairable_attempts']),
                      'sha256':hashlib.sha256(args.out.read_bytes()).hexdigest()}))
