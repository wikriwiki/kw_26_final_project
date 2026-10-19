"""Descriptive activity variation, never a policy-effect accuracy score.

Separates matched on/off differences from within-person generation variation.
Calendar, full matrix and independent eligibility are required; no failed paths
are dropped. Different clock representations are not silently pooled.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import statistics
from action_plan_contract import inspect
from action_repair_feedback import feedback
from validate_prompt_v3 import atomic


def sequence(row, *, include_time=False):
    # Zone identity is omitted so two different home districts do not by
    # themselves count as different behavior. Residence/workplace remain distinct.
    return [(e['activity_id'], 'zone' if e['anchor'].startswith('zone:') else e['anchor'])
            + ((e['time'],) if include_time else ()) for e in row['execution_plan']['events']]


def distance(a, b):
    if not a and not b:return 0.0
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]/max(len(a),len(b))


def describe(pairs):
    values=[distance(a,b) for a,b in pairs]
    return {'pairs':len(values),'mean_normalized_edit_distance':statistics.mean(values) if values else None,
            'identical_pairs':sum(v==0 for v in values),
            'identical_fraction':sum(v==0 for v in values)/len(values) if values else None}


def audit(folder):
    folder=Path(folder)
    config=json.loads((folder/'manifest.json').read_bytes())['config']
    cells=json.loads((folder/'frozen_inputs.json').read_bytes())['cells']
    rows=[json.loads(x) for x in (folder/'responses.jsonl').read_bytes().splitlines()]
    if len(config['candidates'])!=1:raise ValueError('Explicitly select one frozen candidate')
    seeds=config['seeds'];variant=config['candidates'][0]['id']
    if len(seeds)<2 or len(set(seeds))!=len(seeds):raise ValueError('At least two distinct repeats required')
    key=lambda r:(r['aid'],r['case'],r['arm'])
    sources={key(c):c for c in cells}
    expected={(s,*key(c)) for s in seeds for c in cells}
    by_row={(r['replicate'],*key(r)):r for r in rows}
    if len(sources)!=len(cells) or len(by_row)!=len(rows) or set(by_row)!=expected:
        raise ValueError('Incomplete or duplicate repeat matrix')
    for r in rows:
        c=sources[key(r)]
        if r['variant']!=variant or r['date']!=c['date']:raise ValueError('Mixed candidate or calendar')
        if not r['eligible'] or r.get('answer_usage',{}).get('finish_reason',{}).get('type')!='stop':
            raise ValueError('Failed or incomplete plan cannot be dropped from variation audit')
        checked=inspect(r['raw'],c,max_shift=config['max_shift_minutes'])
        if feedback(r['raw'],c,max_shift=config['max_shift_minutes']) is not None or checked['execution_plan']!=r['execution_plan']:
            raise ValueError('Raw plan does not match supplied execution and facts')
        if config.get('last_start_not_before') and checked['execution_plan']['events'][-1]['time']<config['last_start_not_before']:
            raise ValueError('Registered output coverage violated')
    result={'scope':'Descriptive conditional plan variation. No direction, effect magnitude, causal significance, realistic demand or optimal-prompt verdict.',
            'calendar_rule':'Each on/off pair uses the same person and date; repeat comparisons also fix the condition.',
            'rows':len(rows),'seeds':seeds,'mechanisms':{}}
    for case in sorted({c['case'] for c in cells}):
        selected=[c for c in cells if c['case']==case]
        if len({c['date'] for c in selected})!=1:raise ValueError('Different people must share the comparison calendar')
        roster=sorted({c['aid'] for c in selected})
        if {key(c) for c in selected}!={(a,case,arm) for a in roster for arm in ['off','on']}:
            raise ValueError('Incomplete on/off source matrix')
        for aid in roster:
            if sources[(aid,case,'on')]['date']!=sources[(aid,case,'off')]['date']:
                raise ValueError('On/off calendar mismatch')
        groups={}
        for clock in [False,True]:
            seq=lambda seed,aid,arm:sequence(by_row[(seed,aid,case,arm)],include_time=clock)
            groups['activities_and_times' if clock else 'activities_only']={
                'same_person_same_condition_across_seeds':describe((seq(s,a,arm),seq(t,a,arm)) for a in roster for arm in ['off','on'] for s,t in combinations(seeds,2)),
                'same_person_on_off_same_seed':describe((seq(s,a,'off'),seq(s,a,'on')) for a in roster for s in seeds),
                'different_people_same_condition_seed':describe((seq(s,a,arm),seq(s,b,arm)) for s in seeds for arm in ['off','on'] for a,b in combinations(roster,2))}
        result['mechanisms'][case]=groups
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Refusing overwrite')
    atomic(args.out,audit(args.run))
