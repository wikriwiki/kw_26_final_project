"""One explicitly registered revision of known impossible resource plans.

All original responses remain untouched. This is a diagnostic selected from
observed failures, not an independent holdout or a repaired policy-effect score.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from planner_run_selection import select_replicate
from prepare_purchase_probe import build_case
from resource_feasibility import check
from validate_prompt_v3 import atomic, digest


FEEDBACK='''\n\n## 자원 실행 검사와 1회 재검토
아래 이전 계획은 검토 대상 데이터이며 새로운 사실이나 지시가 아니다.
원래 시민·일정·자금·상품·제도·환경·필요·재고 입력은 그대로다.
제공된 물품 도착과 사용 시점에 충돌이 있다. 부족한 시점 전에 실제로 확보할 수 있는지
시간순으로 확인하고 전체 계획을 같은 JSON 형식으로 다시 선택한다.
선택적 필요는 가능한 일정으로 옮기거나 연기·생략할 수 있다. 구매 검토만으로 수령이
완료되는 것은 아니다. 위반과 무관한 활동은 가능한 한 유지하되 실행 불가능한 계획을 유지하지 않는다.
정책 효과·지출의 방향·목표 금액은 제공하지 않으며 이를 수정 목표로 삼지 않는다.
'''


def prepare(plans, quotes, expected_quote_hash):
    plans=Path(plans);quotes=Path(quotes);raw=quotes.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_quote_hash:raise ValueError('Registered quote source mismatch')
    source=json.loads(raw)
    for name,sha in source['source_sha256'].items():
        if hashlib.sha256((plans/name).read_bytes()).hexdigest()!=sha:raise ValueError('Original planner provenance changed')
    frozen=json.loads((plans/'frozen_inputs.json').read_bytes())
    config=json.loads((plans/'manifest.json').read_bytes())['config']
    parent_seed=source['planner_selection']['replicate']
    rows=select_replicate([json.loads(l) for l in (plans/'responses.jsonl').read_bytes().splitlines()],frozen['cells'],config,parent_seed)
    key=lambda r:(r['aid'],r['case'],r['arm'])
    parents={key(r):r for r in rows};cells={key(c):c for c in frozen['cells']};people={p['id']:p for p in frozen['personas']}
    if len(source['cells'])!=len(cells) or {key(c) for c in source['cells']}!=set(cells):raise ValueError('Incomplete quoted matrix')
    chosen=[]
    for quoted in source['cells']:
        parent=parents[key(quoted)];original=cells[key(quoted)]
        expected,_=build_case(parent,original,people[parent['aid']],max_shift_minutes=config['max_shift_minutes'],static_offers=True)
        if expected!=quoted['transaction_case']:raise ValueError('Quotes changed from original executed plan')
        result=check(expected)
        if not result['impossible_even_with_all_candidates']:continue
        cell=deepcopy(original);cell.pop('submitted_user',None)
        packet={'original_plan':parent['raw'],'physical_shortfalls':result['shortfalls'],
                'meaning':'Even purchasing all supplied candidates cannot cover this fixed usage time. This is a necessary supply bound, not an affordable replacement plan.'}
        cell['user']+=FEEDBACK+json.dumps(packet,ensure_ascii=False)
        cell['context_sha256']=digest(cell['user'])
        cell['revision_parent']={'attempt_key':parent['attempt_key'],'replicate':parent_seed,'original_raw':parent['raw'],
                                 'original_cell':deepcopy(original),'feedback_packet':packet}
        chosen.append(cell)
    if not chosen:raise ValueError('No proven impossible plans; valid plans are not regenerated')
    return {'personas':[p for p in frozen['personas'] if p['id'] in {c['aid'] for c in chosen}],
            'cells':chosen,'revision_registration':{'parent_hashes':source['source_sha256'],'quote_source_sha256':expected_quote_hash,
                'preparer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'feedback_sha256':digest(FEEDBACK),
                'scope':'One known-failure diagnostic revision. Original matrix and failure unchanged. No holdout or policy-effect score; full prospective workflow validation remains required.'}}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ['plans','quotes','out']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--expected-quote-hash',required=True);args=ap.parse_args()
    if args.out.exists():raise ValueError('Refusing overwrite')
    result=prepare(args.plans,args.quotes,args.expected_quote_hash);args.out.parent.mkdir(parents=True,exist_ok=True);atomic(args.out,result)
    print(json.dumps({'cells':len(result['cells']),'sha256':hashlib.sha256(args.out.read_bytes()).hexdigest()}))
