"""Expose the unchanged hypothetical purchase menu before activity decisions."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from prepare_purchase_probe import preview_for,render_preview
from validate_prompt_v3 import atomic,digest


def prepare(source):
    result=copy.deepcopy(source);people={p['id']:p for p in result['personas']}
    for c in result['cells']:
        if 'purchase_preview' in c or '## 오늘의 구매 후보 개요' in c['user']:raise ValueError('Preview already supplied')
        c['has_work']=bool(people[c['aid']].get('work_poi_id'))
        c['purchase_preview']=preview_for(c,people[c['aid']])
        marker='\n\n## 오늘\n'
        if c['user'].count(marker)!=1:raise ValueError('Unrecognized planner context boundary')
        c['user']=c['user'].replace(marker,render_preview(c['purchase_preview'])+marker)
        c['context_sha256']=digest(c['user'])
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Refusing overwrite')
    raw=args.source.read_bytes();result=prepare(json.loads(raw))
    result['market_preview_provenance']={'source_sha256':hashlib.sha256(raw).hexdigest(),
        'provider_sha256':hashlib.sha256(Path(__file__).with_name('prepare_purchase_probe.py').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Input visibility ablation only. Same narrow hypothetical menu, quantities, prices and eligibility. No new observed prices or realistic-demand claim.'}
    atomic(args.out,result)
    print(json.dumps({'cells':len(result['cells']),'sha256':hashlib.sha256(args.out.read_bytes()).hexdigest()}))
