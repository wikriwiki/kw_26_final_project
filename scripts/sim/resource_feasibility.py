"""Necessary physical feasibility bound before committing to fixed purchases.

Buying every offered candidate (even mutually exclusive/unaffordable ones) gives
an optimistic supply bound. A deficit under that bound proves impossibility;
passing it does NOT establish affordability or a feasible joint purchase choice.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from daily_resource_contract import validate
from validate_prompt_v3 import atomic


def check(case):
    c=case['daily_conditions'];validate(c)
    stock={k:v['opening_quantity'] for k,v in c['resources'].items()}
    pending=deepcopy(c.get('opening_pending_receipts',[]));shortfalls=[];previous=-1
    for event in case['events']:
        if not re.fullmatch(r'([01][0-9]|2[0-3]):[0-5][0-9]',event['time']):raise ValueError('Event time')
        current=int(event['time'][:2])*60+int(event['time'][3:])
        if current<=previous:raise ValueError('Chronological events required')
        previous=current
        due=[q for q in pending if q['minute']<=current];pending=[q for q in pending if q['minute']>current]
        for receipt in due:
            for rid,value in receipt['quantities'].items():stock[rid]+=value
        for quote in event['candidates']:
            cid=quote['id'];amounts=c['quote_receipts'].get(cid,{})
            if not amounts:continue
            delay=c['quote_receipt_delay_minutes'][cid]
            if delay:pending.append({'minute':current+delay,'quantities':deepcopy(amounts)})
            else:
                for rid,value in amounts.items():stock[rid]+=value
        for rid,value in c['activity_consumption'].get(event['activity_id'],{}).items():
            if stock[rid]<value:
                shortfalls.append({'event_id':event['id'],'time':event['time'],'resource':rid,
                    'maximum_available':stock[rid],'required':value})
            stock[rid]-=value
    return {'impossible_even_with_all_candidates':bool(shortfalls),'shortfalls':shortfalls,
            'scope':'Necessary physical bound only; no money or exclusive-choice feasibility guarantee. Fixed original schedule, no repair.'}


def gate(cells, *, allow_exclusion=False):
    """Split prepared purchase cells before any model call.

    A plan that cannot be supplied even by buying every offered candidate is
    physically impossible. Spending a purchase call on it records an engine fact
    as a model failure, so the caller excludes it here. This never repairs a plan,
    shifts a time, or deletes the row: the blocked cell is returned with its
    shortfall evidence and the caller must report the exclusion.
    """
    for cell in cells:
        cell.setdefault('resource_feasibility', check(cell['transaction_case']))
    blocked = [c for c in cells if c['resource_feasibility']['impossible_even_with_all_candidates']]
    if blocked and not allow_exclusion:
        raise ValueError('Resource gate blocked %d cell(s); opt in explicitly to run the remainder as an incomplete matrix' % len(blocked))
    runnable = [c for c in cells if not c['resource_feasibility']['impossible_even_with_all_candidates']]
    if not runnable:
        raise ValueError('Resource gate excluded every cell')
    return runnable, blocked


def gate_report(runnable, blocked):
    """What the run must disclose about the gate. An empty exclusion is still reported."""
    return {'checked': len(runnable) + len(blocked), 'excluded': len(blocked),
            'cells': [{k: c[k] for k in ['aid', 'case', 'arm', 'date'] if k in c}
                      | {'shortfalls': c['resource_feasibility']['shortfalls']} for c in blocked],
            'whole_matrix_eligible': not blocked,
            'effect': 'Excluded before any model call. A non-empty exclusion means the matrix is '
                      'incomplete by construction: the remaining cells are not a complete comparison, '
                      'and the exclusion is not a repaired plan.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Refusing overwrite')
    raw=args.source.read_bytes();source=json.loads(raw);results=[]
    for cell in source['cells']:
        results.append({k:cell[k] for k in ['aid','case','arm']}|{'result':check(cell['transaction_case'])})
    result={'source_sha256':hashlib.sha256(raw).hexdigest(),'rows':len(results),
            'proven_impossible':sum(r['result']['impossible_even_with_all_candidates'] for r in results),'results':results}
    atomic(args.out,result);print(json.dumps({k:result[k] for k in ['rows','proven_impossible']}))
