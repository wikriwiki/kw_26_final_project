"""Revalidate a linked, complete one-day planner/purchase experiment.

Catalog consumption is not total household consumption. Do not use this report
to score historical policy-effect magnitude or select a desired effect sign.
"""
import argparse
import hashlib
import json
from pathlib import Path
from paired_asset_score import score
from validate_prompt_v3 import atomic
from action_plan_contract import inspect
from action_repair_feedback import feedback


def linked_rows(plan_folder,purchase_folder):
    plan_folder=Path(plan_folder);purchase_folder=Path(purchase_folder)
    plans=[json.loads(x) for x in (plan_folder/'responses.jsonl').read_bytes().splitlines()]
    source_raw=(purchase_folder/'frozen_inputs.json').read_bytes()
    source=json.loads(source_raw)
    manifest=json.loads((purchase_folder/'manifest.json').read_bytes())
    if hashlib.sha256(source_raw).hexdigest()!=manifest['input_sha256'] or manifest['input_sha256']!=manifest['config']['source_sha256']:
        raise ValueError('Purchase input snapshot changed from registered source')
    provenance=source['source_sha256']
    for name in ['manifest.json','responses.jsonl','frozen_inputs.json']:
        if hashlib.sha256((plan_folder/name).read_bytes()).hexdigest()!=provenance[name]:
            raise ValueError('Purchase source does not match frozen planner '+name)
    frozen=json.loads((plan_folder/'frozen_inputs.json').read_bytes())
    plan_config=json.loads((plan_folder/'manifest.json').read_bytes())['config']
    selection=source.get('planner_selection')
    if selection is not None:
        from planner_run_selection import select_replicate
        if selection['variant']!=plan_config['candidates'][0]['id'] or selection['whole_original_matrix_required'] is not True:
            raise ValueError('Planner selection does not match frozen registration')
        plans=select_replicate(plans,frozen['cells'],plan_config,selection['replicate'])
    key=lambda r:(r['aid'],r['case'],r['arm'])
    by_plan={key(r):r for r in plans}
    if len(by_plan)!=len(plans):raise ValueError('Select one planner replicate and variant explicitly')
    if not all(r['eligible'] for r in plans):raise ValueError('Failed upstream plans cannot form an effect matrix')
    expected={(c['aid'],c['case'],c['arm']) for c in frozen['cells']}
    if set(by_plan)!=expected:raise ValueError('Incomplete upstream planner matrix')
    for c in frozen['cells']:
        p=by_plan[key(c)]
        if p.get('answer_usage',{}).get('finish_reason',{}).get('type')!='stop':raise ValueError('Incomplete upstream generation')
        if feedback(p['raw'],c,max_shift=plan_config['max_shift_minutes']) is not None:raise ValueError('Upstream raw plan fails independent facts or execution')
        checked=inspect(p['raw'],c,max_shift=plan_config['max_shift_minutes'])
        if checked['execution_plan']!=p['execution_plan']:raise ValueError('Recorded execution differs from raw plan validation')
        if plan_config.get('last_start_not_before') and checked['execution_plan']['events'][-1]['time']<plan_config['last_start_not_before']:
            raise ValueError('Upstream output coverage violated')
    cases={key(c):c for c in source['cells']}
    if set(cases)!=expected or len(cases)!=len(source['cells']):raise ValueError('Quote source dropped or duplicated citizen conditions')
    for k,c in cases.items():
        p=by_plan[k]
        if c['date']!=p['date'] or c['attempt_key']!=p['attempt_key'] or c['transaction_case']['id']!=p['attempt_key']:
            raise ValueError('Purchase linked to wrong upstream decision')
        if c['bridge_audit']['schedule_report']['execution_plan']!=p['execution_plan']:
            raise ValueError('Quoted activities differ from frozen executed plan')
        events=p['execution_plan']['events'];quoted=c['transaction_case']['events']
        if len(events)!=len(quoted):raise ValueError('Purchase events dropped planned activities')
        for i,(event,q) in enumerate(zip(events,quoted)):
            if q['id']!='event:'+str(i) or any(q[k]!=event[k] for k in ['time','anchor','activity_id','intent']):
                raise ValueError('Purchase event differs from planned activity')
            if q['channel']!=(event['purchase_channel'] or 'offline') or (event['purchase_channel'] is None and q['candidates']):
                raise ValueError('Purchase channel or free activity changed')
    rows=[json.loads(x) for x in (purchase_folder/'responses.jsonl').read_bytes().splitlines()]
    enriched=[]
    for r in rows:
        if key(r) not in cases or r['date']!=cases[key(r)]['date']:raise ValueError('Purchase row outside intended conditions')
        enriched.append(dict(r,transaction_case=cases[key(r)]['transaction_case']))
    return frozen,source,manifest,enriched


def report(plan_folder,purchase_folder):
    frozen,source,manifest,rows=linked_rows(plan_folder,purchase_folder)
    result={'scope':'One-day matched conditional catalog-consumption probe. Same people and calendar on/off. No population significance, observed historical magnitude or total household consumption claim.',
        'macro_validated':False,'source_plan_hashes':source['source_sha256'],'purchase_config':manifest['config'],
        'planner_selection':source.get('planner_selection'),'source_cells':len(source['cells']),'purchase_rows':len(rows),'mechanisms':{}}
    for mechanism in sorted({c['case'] for c in source['cells']}):
        cells=[c for c in source['cells'] if c['case']==mechanism]
        selected=[r for r in rows if r['case']==mechanism]
        item={'intended_conditions':len(cells),'financially_valid_rows':sum(r['valid'] for r in selected),
              'paid_quote_opportunities':sum(bool(e['candidates']) for c in cells for e in c['transaction_case']['events']),
              'scope':'Consumption covers only the fixed candidate catalog. Asset acquisition outflow is separate.'}
        try:
            item['matched_contrasts']=score(selected,roster=sorted({c['aid'] for c in cells}),days=sorted({c['date'] for c in cells}),seeds=manifest['config']['seeds'])
        except ValueError as exc:item['not_scored']=str(exc)
        result['mechanisms'][mechanism]=item
    result['all_matrices_complete']=all('matched_contrasts' in v for v in result['mechanisms'].values())
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plans',type=Path,required=True);ap.add_argument('--purchases',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise ValueError('Refusing to overwrite a coupled report')
    result=report(args.plans,args.purchases);atomic(args.out,result)
    print(json.dumps({'all_matrices_complete':result['all_matrices_complete'],'conditions':result['source_cells'],'macro_validated':False}))


if __name__=='__main__':main()
