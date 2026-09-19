"""Experimental typed activity decisions; prose is not executable simulator state.

Finite action vocabulary is disclosed. Policy facts do not set desired effects.
Extra concrete commitments must be supplied explicitly by the context provider.
"""
from copy import deepcopy
from datetime import date
import json
import re
from temporal_projection import project, transition_violations

# id, location type, display category, literal activity, potential purchase channel
ACTIVITIES = [
    ('home_prepare', 'home', '집', '집에서 하루 준비', None),
    ('home_meal', 'home', '집', '집에서 식사', None),
    ('home_chores', 'home', '집', '집안 정리', None),
    ('home_leisure', 'home', '집', '집에서 휴식·취미 활동', None),
    ('home_sleep', 'home', '집', '취침 준비', None),
    ('remote_work', 'home', '집', '집에서 업무', None),
    ('home_delivery', 'home', '집', '집으로 식사 배달 주문 검토', 'online'),
    ('home_online_goods', 'home', '집', '집에서 온라인 물품 구매 검토', 'online'),
    ('office_work', 'work', '직장', '사무실에서 업무', None),
    ('office_prepare', 'work', '직장', '사무실 도착 후 업무 준비', None),
    ('office_finish', 'work', '직장', '업무 종료 후 퇴근 준비', None),
    ('office_meal', 'work', '직장', '직장 안에서 식사', None),
    ('office_break', 'work', '직장', '직장 안에서 휴식', None),
    ('office_delivery', 'work', '직장', '직장으로 식사 배달 주문 검토', 'online'),
    ('walk', 'zone', '여가', '무료 산책', None),
    ('outdoor_leisure', 'zone', '여가', '무료 야외 여가', None),
    ('library', 'zone', '교육', '도서관 이용', None),
    ('meal_dine_in', 'zone', '식사', '음식점 매장 식사 검토', 'offline'),
    ('meal_takeaway', 'zone', '식사', '음식 포장 구매 검토', 'offline'),
    ('cafe_dine_in', 'zone', '카페', '카페 매장 이용 검토', 'offline'),
    ('cafe_takeaway', 'zone', '카페', '카페 음료 포장 구매 검토', 'offline'),
    ('dessert', 'zone', '디저트', '디저트 구매 검토', 'offline'),
    ('groceries', 'zone', '마트', '식료품 구매 검토', 'offline'),
    ('convenience', 'zone', '편의점', '편의점 물품 구매 검토', 'offline'),
    ('shopping', 'zone', '쇼핑', '물품 구매 검토', 'offline'),
    ('hair', 'zone', '미용', '미용 서비스 이용 검토', 'offline'),
    ('health_goods', 'zone', '건강', '건강 관련 물품 구매 검토', 'offline'),
    ('leisure_service', 'zone', '여가', '유료 여가 서비스 이용 검토', 'offline'),
    ('education_service', 'zone', '교육', '교육 서비스 이용 검토', 'offline'),
    ('bar', 'zone', '주점', '주점 이용 검토', 'offline'),
    ('other_service', 'zone', '기타', '기타 서비스 이용 검토', 'offline'),
]


def catalog(cell):
    result = {}
    for aid, kind, category, intent, channel in ACTIVITIES:
        anchors = ['residence'] if kind == 'home' else ['workplace'] if kind == 'work' else ['zone:' + str(z) for z in cell['zones']]
        if kind == 'work' and not cell['has_work']: continue
        if aid == 'remote_work' and not cell['has_work']: continue
        if not anchors: continue
        result[aid] = {'id': aid, 'anchors': anchors, 'category': category, 'intent': intent, 'purchase_channel': channel}
    for entry in cell.get('provided_activities', []):
        if entry['id'] in result or not entry.get('evidence') or entry['evidence'] not in cell['user']:
            raise ValueError('Additional activity must have unique ID and verbatim provider evidence')
        if not set(entry['anchors']) <= {'residence', 'workplace', *['zone:' + str(z) for z in cell['zones']]}:
            raise ValueError('Provided activity has invalid anchor')
        result[entry['id']] = deepcopy(entry)
    for rule in cell.get('action_rules', []):
        if rule['evidence'] not in cell['user']: raise ValueError('Rule lacks exact input evidence')
        if rule['kind'] == 'forbid_activity':
            for aid in rule['activity_ids']: result.pop(aid, None)
        elif rule['kind'] == 'allowed_anchors':
            for value in result.values(): value['anchors'] = [a for a in value['anchors'] if a in rule['anchors']]
        else: raise ValueError('Unknown activity rule')
    return {k: v for k, v in result.items() if v['anchors']}


def schema(cell):
    items = catalog(cell); kinds = []
    for aid, spec in items.items():
        props = {'time': {'type': 'string', 'pattern': '^([01][0-9]|2[0-3]):[0-5][0-9]$'},
                 'activity_id': {'const': aid}, 'anchor': {'type': 'string', 'enum': spec['anchors']}}
        kinds.append({'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False})
    home = [branch for branch in kinds if branch['properties']['anchor']['enum'] == ['residence']]
    weekend = date.fromisoformat(cell['date']).weekday() >= 5
    return {'type': 'object', '$defs': {'activity': {'anyOf': kinds}, 'home': {'anyOf': home}},
            'properties': {'events': {'anyOf': [{'type': 'array', 'minItems': n, 'maxItems': n,
              'prefixItems': [{'$ref': '#/$defs/home'}] + [{'$ref': '#/$defs/activity'}] * (n - 2) + [{'$ref': '#/$defs/home'}], 'items': False}
              for n in range(4 if weekend else 6, 9 if weekend else 11)]}}, 'required': ['events'], 'additionalProperties': False}


def inspect(raw, cell, max_shift=10):
    obj = json.loads(raw); specs = catalog(cell)
    if not isinstance(obj, dict) or set(obj) != {'events'} or not isinstance(obj['events'], list): raise ValueError('Output shape')
    events = obj['events']; weekend = date.fromisoformat(cell['date']).weekday() >= 5
    if not (4 if weekend else 6) <= len(events) <= (8 if weekend else 10): raise ValueError('Activity count')
    rendered = []
    for event in events:
        if set(event) != {'time', 'activity_id', 'anchor'}: raise ValueError('Unexpected generated field')
        if event['activity_id'] not in specs: raise ValueError('Unavailable activity')
        spec = specs[event['activity_id']]
        if event['anchor'] not in spec['anchors']: raise ValueError('Wrong activity location')
        if not re.fullmatch(r'([01][0-9]|2[0-3]):[0-5][0-9]', event['time']): raise ValueError('Time representation')
        rendered.append(dict(event, category=spec['category'], intent=spec['intent'], purchase_channel=spec['purchase_channel']))
    if any(events[i]['anchor'] != 'residence' for i in [0, -1]): raise ValueError('Home endpoints')
    raw_errors = []
    times = [int(e['time'][:2])*60 + int(e['time'][3:]) for e in events]
    if any(b-a < 20 for a, b in zip(times, times[1:])): raw_errors.append('time')
    transitions = transition_violations(obj, cell.get('minimum_transitions', []))
    if transitions: raw_errors.append('transition_time')
    for obligation in cell.get('required_activities', []):
        if not any(all(e.get(k) == v for k, v in obligation.items() if k != 'evidence') for e in rendered):
            raw_errors.append('missing_commitment')
    errors = list(raw_errors); projected = None; projection = None
    try:
        projected, projection = project({'events': rendered}, max_shift=max_shift, gap=20,
                                         fixed_times=cell.get('fixed_times', []), transitions=cell.get('minimum_transitions', []))
        errors = []
    except ValueError: errors.append('temporal_projection_infeasible')
    check = projected['events'] if projected else rendered
    for obligation in cell.get('required_activities', []):
        if obligation['evidence'] not in cell['user']: raise ValueError('Obligation lacks provider evidence')
        if not any(all(e.get(k) == v for k, v in obligation.items() if k != 'evidence') for e in check): errors.append('missing_commitment')
    return {'raw_valid': not raw_errors, 'raw_errors': sorted(set(raw_errors)),
            'valid': not errors, 'errors': errors, 'execution_plan': projected, 'temporal_projection': projection,
            'raw_transition_violations': transitions, 'raw_plan': obj}
