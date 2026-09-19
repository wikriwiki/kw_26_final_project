"""Commit real completed day-one choices and preview carried day-two openings.

No day-two schedule or LLM is run. Zero new needs/income are explicit dry-run
assumptions, never claims about actual citizens or future weekday obligations.
"""
import argparse
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

from asset_day_checkpoint import commit_day
from audit_purchase_probe import audit
from daily_state_transition import advance
from validate_prompt_v3 import atomic


def run(folder, out):
    folder=Path(folder);out=Path(out)
    if out.exists():raise ValueError('Refusing to overwrite state paths')
    checked=audit(folder)
    if checked['independent_errors']:raise ValueError('Failed source audit')
    inputs=json.loads((folder/'frozen_inputs.json').read_bytes())
    by_cell={(c['aid'],c['case'],c['arm']):c for c in inputs['cells']}
    rows=[json.loads(line) for line in (folder/'responses.jsonl').read_bytes().splitlines()]
    out.mkdir(parents=True)
    previews=[]
    for row in rows:
        cell=by_cell[(row['aid'],row['case'],row['arm'])];case=cell['transaction_case']
        root=out/row['attempt_key']
        path=commit_day(root,day=row['date'],roster=[row['aid']],cases={row['aid']:case},
            rows=[dict(row,complete=True)],state_protocol='carry_needs_v1')
        original=case['daily_conditions']
        template={k:deepcopy(original[k]) for k in ['resources','activity_consumption','quote_receipts','quote_receipt_delay_minutes']}
        template.update(needs=[],provenance={'kind':'synthetic_assumption',
            'source':'State continuity dry run: zero exogenous new needs and income, same resource definitions. No next-day attendance or policy assumptions.'})
        next_state=advance(case,row['raw'],template)
        preview={'next_day':(date.fromisoformat(row['date'])+timedelta(days=1)).isoformat(),
                 'previous_checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'transition_template':template,'opening_state':next_state,
                 'day_two_executed':False}
        atomic(root/'next_opening_preview.json',preview)
        previews.append({k:row[k] for k in ['attempt_key','aid','case','arm','replicate']} | {
            'cash':next_state['cash'],
            'resources':{k:v['opening_quantity'] for k,v in next_state['daily_conditions']['resources'].items()},
            'unfulfilled_needs':{n['id']:n['desired_count'] for n in next_state['daily_conditions']['needs']},
            'pending_receipts':next_state['daily_conditions']['opening_pending_receipts']})
    result={'source_hashes':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
        for name in ['manifest.json','frozen_inputs.json','responses.jsonl']},
        'day_one_committed':len(rows),'day_two_openings':len(previews),'day_two_executed':False,
        'new_needs_and_income_assumption':0,'states':previews,
        'scope':'Actual completed choices feed isolated cash/wallet/inventory/delivery/unmet-need continuity. No next-day model choice or policy-effect validation.'}
    atomic(out/'summary.json',result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();result=run(args.run,args.out)
    print(json.dumps({k:result[k] for k in ['day_one_committed','day_two_openings','day_two_executed']}))
