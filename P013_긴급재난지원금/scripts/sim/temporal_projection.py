"""Explicit bounded schedule projection, preserving every non-time choice.

No hidden repair: callers must save raw outputs, raw errors, adjusted output and
all shifts separately. This does not validate semantic timing or policy hours.
"""
from copy import deepcopy
import re


def minute(value):
    if not isinstance(value, str) or not re.fullmatch(r'(?:[01][0-9]|2[0-3]):[0-5][0-9]', value):
        raise ValueError('Invalid clock value')
    h, m = map(int, value.split(':'))
    return h * 60 + m


def transition_violations(obj, rules):
    limits = {(r['from_anchor'], r['to_anchor']): r['minimum_minutes'] for r in rules}
    failures = []
    events = obj.get('events', [])
    for index, (previous, current) in enumerate(zip(events, events[1:]), 1):
        minimum = limits.get((previous.get('anchor'), current.get('anchor')))
        if minimum is None: continue
        try:
            elapsed = minute(current.get('time')) - minute(previous.get('time'))
        except ValueError:
            continue  # The clock contract independently rejects invalid times.
        if elapsed < minimum:
            failures.append({'event_index': index, 'elapsed_minutes': elapsed, 'minimum_minutes': minimum})
    return failures


def project(obj, *, max_shift=10, gap=20, fixed_times=(), transitions=()):
    if isinstance(max_shift, bool) or not isinstance(max_shift, int) or max_shift < 0:
        raise ValueError('Invalid max shift')
    if isinstance(gap, bool) or not isinstance(gap, int) or gap <= 0:
        raise ValueError('Invalid minimum gap')
    times = [minute(e['time']) for e in obj['events']]
    if not times:
        raise ValueError('Empty schedule')
    fixed = {minute(t) for t in fixed_times}
    travel = {}
    for rule in transitions:
        duration = rule['minimum_minutes']
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
            raise ValueError('Invalid transition duration')
        key = (rule['from_anchor'], rule['to_anchor'])
        travel[key] = max(travel.get(key, 0), duration)
    states = {}
    for index, original in enumerate(times):
        allowed = [original] if original in fixed else range(max(0, original-max_shift), min(1439, original+max_shift)+1)
        following = {}
        for current in allowed:
            if index == 0:
                following[current] = (abs(current-original), (current,))
            else:
                key = (obj['events'][index-1]['anchor'], obj['events'][index]['anchor'])
                minimum = max(gap, travel.get(key, 0))
                previous = [v for t, v in states.items() if current-t >= minimum]
                if previous:
                    cost, path = min(previous)
                    following[current] = (cost+abs(current-original), path+(current,))
        states = following
        if not states:
            raise ValueError('No feasible schedule within registered shift budget')
    cost, path = min(states.values())
    output = deepcopy(obj)
    shifts = []
    for index, (old, new) in enumerate(zip(times, path)):
        text = f'{new//60:02d}:{new%60:02d}'
        if old != new:
            shifts.append({'event_index': index, 'before': obj['events'][index]['time'], 'after': text, 'minutes': new-old})
        output['events'][index]['time'] = text
    return output, {'total_absolute_shift_minutes': cost, 'shifts': shifts,
                    'fixed_times': sorted(fixed_times), 'maximum_shift_per_event': max_shift,
                    'transitions': list(transitions),
                    'scope': 'Clock spacing and supplied adjacent anchor transitions only; not a complete route or factual/policy validator.'}
