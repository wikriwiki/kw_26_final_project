"""Check explicitly supplied place commitments over an interval, not just arrival."""
import re


def minute(value):
    if not isinstance(value,str) or not re.fullmatch(r'([01][0-9]|2[0-3]):[0-5][0-9]',value):raise ValueError('Presence time')
    return int(value[:2])*60+int(value[3:])


def violations(events,cell):
    problems=[]
    for rule in cell.get('required_presence_intervals',[]):
        if rule.get('evidence') not in cell['user'] or not rule.get('evidence'):raise ValueError('Presence lacks provider evidence')
        start,end=minute(rule['start']),minute(rule['end']);anchor=rule['anchor']
        if start>=end:raise ValueError('Presence interval must be nonempty within one day')
        timed=[(minute(e['time']),e['anchor']) for e in events]
        if any(a[0]>=b[0] for a,b in zip(timed,timed[1:])):
            problems.append('presence_unordered');continue
        before=[a for t,a in timed if t<=start]
        if not before or before[-1]!=anchor:problems.append('presence_start')
        if any(a!=anchor for t,a in timed if start<=t<end):problems.append('presence_departure')
        # Known travel to the next location cannot consume committed time.
        for (t,a),(nt,na) in zip(timed,timed[1:]):
            if a!=anchor or na==anchor or t>=end or nt<end:continue
            required=max([r['minimum_minutes'] for r in cell.get('minimum_transitions',[])
                          if r['from_anchor']==a and r['to_anchor']==na] or [0])
            if nt-required<end and nt>start:problems.append('presence_travel_overlap')
    return sorted(set(problems))
