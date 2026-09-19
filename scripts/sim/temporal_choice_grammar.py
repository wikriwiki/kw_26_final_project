"""Finite-clock grammar: choose activities only among executable time sequences.

This is a NEW representational constraint, not an invisible correction of a raw
schedule. Default clock has 30-minute resolution plus explicit commitment times.
Only supplied adjacent residence/workplace commute bounds are supported here;
no full routing, economic preference, or policy-effect target is inferred.
"""
from datetime import date
from functools import lru_cache
import json
from action_plan_contract import catalog
from presence_contract import minute


def build(cell, *, clock_step=30,last_start_not_before=None):
    if clock_step not in {20,30,60}:raise ValueError('Unregistered clock resolution')
    specs=catalog(cell);required={};presence=[];travel={}
    for row in cell.get('required_activities',[]):
        if not row.get('evidence') or row['evidence'] not in cell['user']:raise ValueError('Commitment lacks evidence')
        t=minute(row['time'])
        if t in required:raise ValueError('Simultaneous committed starts unsupported')
        if row['activity_id'] not in specs or row['anchor'] not in specs[row['activity_id']]['anchors']:raise ValueError('Invalid committed activity')
        required[t]=row
    for row in cell.get('required_presence_intervals',[]):
        if not row.get('evidence') or row['evidence'] not in cell['user']:raise ValueError('Presence lacks evidence')
        a,b=minute(row['start']),minute(row['end'])
        if a>=b or row['anchor'] not in {'residence','workplace'}:raise ValueError('Unsupported presence interval')
        presence.append((a,b,row['anchor']))
    for row in cell.get('minimum_transitions',[]):
        a,b=row['from_anchor'],row['to_anchor']
        if a not in {'residence','workplace'} or b not in {'residence','workplace'}:raise ValueError('Full route bounds require another grammar')
        value=row['minimum_minutes']
        if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError('Invalid commute')
        travel[(a,b)]=max(travel.get((a,b),0),value)
    times=sorted(set(range(0,1440,clock_step))|set(required)|{minute(t) for t in cell.get('fixed_times',[])})
    groups=['residence','workplace','zone'];weekend=date.fromisoformat(cell['date']).weekday()>=5
    low,high=(4,8) if weekend else (6,10)
    ending_floor=minute(last_start_not_before) if last_start_not_before is not None else 0
    productions={};examples={};leaves={}
    def group(anchor):return 'zone' if anchor.startswith('zone:') else anchor
    def literal(text):return json.dumps(text,ensure_ascii=False)
    for ti,t in enumerate(times):
        for gi,g in enumerate(groups):
            choices=[]
            for aid,spec in specs.items():
                for anchor in spec['anchors']:
                    if group(anchor)!=g:continue
                    if any(a<=t<b and anchor!=where for a,b,where in presence):continue
                    if t in required and (aid,anchor)!=(required[t]['activity_id'],required[t]['anchor']):continue
                    event={'time':f'{t//60:02d}:{t%60:02d}','activity_id':aid,'anchor':anchor}
                    choices.append(event)
            if choices:
                # Factor the common JSON prefix and Cartesian activity/anchor
                # choices. Repeating complete object literals bloats compilation.
                by_activity={}
                for e in choices:by_activity.setdefault(e['activity_id'],[]).append(e['anchor'])
                by_anchors={}
                for aid,anchors in by_activity.items():by_anchors.setdefault(tuple(anchors),[]).append(aid)
                branches=[]
                for anchors,aids in by_anchors.items():
                    head=literal('{"time":'+json.dumps(f'{t//60:02d}:{t%60:02d}')+',"activity_id":')
                    activities='('+' | '.join(literal(json.dumps(a,ensure_ascii=False)) for a in aids)+')'
                    locations='('+' | '.join(literal(json.dumps(a,ensure_ascii=False)) for a in anchors)+')'
                    branches.append(head+' '+activities+' '+literal(',"anchor":')+' '+locations+' '+literal('}'))
                name=f'e_{ti}_{gi}';productions[name]=' | '.join(branches)
                leaves[(ti,gi)]=name;examples[name]=choices[0]

    @lru_cache(None)
    def state(count,pi,pg):
        previous=times[pi] if pi>=0 else -100000
        prior=groups[pg] if pg>=0 else None
        next_required=min([t for t in required if t>previous] or [1440])
        alternatives=[];example=None
        for ti,t in enumerate(times):
            if t<=previous or t>next_required:continue
            for gi,g in enumerate(groups):
                if count==0 and g!='residence':continue
                leaf=leaves.get((ti,gi))
                if leaf is None:continue
                if count and t-previous<max(20,travel.get((prior,g),0)):continue
                if any((count==0 and t>a) or (previous<a<t and prior!=where)
                       for a,b,where in presence):continue
                if any(prior==where and g!=where and previous<b<=t and t-travel.get((prior,g),0)<b
                       for a,b,where in presence):continue
                new_count=count+1
                end_ok=new_count>=low and g=='residence' and t>=ending_floor and not any(rt>t for rt in required)
                end_ok=end_ok and not any(t<b and g!=where for a,b,where in presence)
                following=None
                if new_count<high:
                    following=state(new_count,ti,gi)
                if end_ok:
                    alternatives.append(leaf+(' ('+literal(',')+' '+following+')?' if following else ''))
                    if example is None:example=[examples[leaf]]
                elif following is not None:
                    alternatives.append(leaf+' '+literal(',')+' '+following)
                    if example is None:example=[examples[leaf]]+examples[following]
        if not alternatives:return None
        name=f's_{count}_{pi+1}_{pg+1}';productions[name]=' | '.join(alternatives);examples[name]=example
        return name
    start=state(0,-1,-1)
    if start is None:raise ValueError('No executable schedule on this registered clock grid')
    root='root ::= '+literal('{"events":[')+' '+start+' '+literal(']}')
    grammar=root+'\n'+'\n'.join(name+' ::= '+body for name,body in productions.items())+'\n'
    return grammar,{'clock_step_minutes':clock_step,'last_start_not_before':last_start_not_before,'clock_values':len(times),'productions':len(productions),
        'grammar_bytes':len(grammar.encode()),'feasible_example':{'events':examples[start]},
        'scope':'Temporal feasibility by construction on finite clock. Does not enforce evaluation-only closure checks, full routing, whole-day consumption or preference validity.'}
