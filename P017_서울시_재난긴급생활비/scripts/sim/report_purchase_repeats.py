"""Describe complete same-input repeat contrasts; never count repeats as people."""
import argparse
import json
from pathlib import Path
import statistics

from report_coupled_probe import linked_rows, report
from validate_prompt_v3 import atomic


PLAN_FIELDS=['model','prompt_module','candidates','answer_tokens','max_shift_minutes',
             'temporal_clock_step','sampling','last_start_not_before','allow_zone_commitments']
PAYMENT_FIELDS=['model','seeds','thinking_tokens','answer_tokens','sampling','transaction_protocol',
                'purchase_prompt_module','decision_view','skip_forced_no_purchase']


def compare(pairs):
    if len(pairs)<2:raise ValueError('At least two complete distinct planner repeats required')
    reference=None;repeats=[];seen=set()
    for plans,purchases in pairs:
        frozen,source,pm,_=linked_rows(plans,purchases)
        am=json.loads((Path(plans)/'manifest.json').read_bytes())
        config=am['config'];selection=source.get('planner_selection')
        if selection:seed=selection['replicate']
        elif len(config['seeds'])==1:seed=config['seeds'][0]
        else:raise ValueError('Explicit planner repeat required')
        if seed in seen:raise ValueError('Duplicate planner repeat is not new evidence')
        seen.add(seed)
        identity={'frozen_inputs':frozen,
                  'plan_settings':{k:config.get(k) for k in PLAN_FIELDS},
                  'purchase_settings':{k:pm['config'].get(k) for k in PAYMENT_FIELDS},
                  'plan_code':am['code_sha256'],'purchase_code':pm['code_sha256'],
                  'plan_system':am['system_sha256'],'purchase_system':pm['system_sha256']}
        if reference is None:reference=identity
        elif identity!=reference:raise ValueError('Repeated source, prompt, code or sampling settings differ')
        result=report(plans,purchases)
        if not result['all_matrices_complete']:raise ValueError('Failed or incomplete replicate cannot be omitted or pooled')
        repeats.append({'planner_seed':seed,'plans':str(plans),'purchases':str(purchases),
                        'plan_source_hashes':source['source_sha256'],'report':result})
    cases=sorted(repeats[0]['report']['mechanisms'])
    summaries={}
    for case in cases:
        metrics={}
        first=repeats[0]['report']['mechanisms'][case]['matched_contrasts']
        if first['replicates']!=1:raise ValueError('This report fixes one payment seed across planner repeats')
        for metric in first['by_seed'][0]['metrics']:
            values=[r['report']['mechanisms'][case]['matched_contrasts']['by_seed'][0]['metrics'][metric] for r in repeats]
            delta=[v['difference'] for v in values]
            signs=[1 if v>0 else -1 if v<0 else 0 for v in delta]
            off=statistics.mean(v['off_mean'] for v in values);on=statistics.mean(v['on_mean'] for v in values)
            metrics[metric]={'off_mean_across_repeats':off,'on_mean_across_repeats':on,
                'mean_difference':on-off,'relative_change_of_means':(on-off)/off if off else None,
                'repeat_differences':dict(zip([str(r['planner_seed']) for r in repeats],delta)),
                'difference_range':[min(delta),max(delta)],'all_repeat_signs_equal':len(set(signs))==1,
                'range_is_confidence_interval':False}
        summaries[case]={'citizens':first['citizens'],'days':first['days'],
            'planner_repeats':len(repeats),'independent_citizen_count':first['citizens'],'metrics':metrics}
    return {'scope':'Descriptive same-input conditional catalog contrasts across frozen planner repeats, fixed payment seed. No population inference, confidence interval, empirical effect-size accuracy or optimal-prompt verdict.',
            'macro_validated':False,'all_original_matrices_complete':True,
            'planner_seeds':[r['planner_seed'] for r in repeats],
            'source_runs':[{k:v for k,v in r.items() if k!='report'} for r in repeats],'mechanisms':summaries}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pair',nargs=2,action='append',required=True,metavar=('PLANS','PURCHASES'))
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise ValueError('Refusing overwrite')
    result=compare(args.pair);atomic(args.out,result)
    print(json.dumps({'planner_seeds':result['planner_seeds'],'macro_validated':False}))
