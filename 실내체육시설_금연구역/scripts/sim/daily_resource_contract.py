"""Conserve declared physical inventory across actual purchases and activities.

Resource units and quote yields are provider facts/explicit scenario assumptions.
No inferred package conversions, desired spending, or automatically repaired use.
"""
from copy import deepcopy
import json
import re


def quantity(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Resource quantities must be nonnegative integers in the declared unit')
    return value


def validate(conditions):
    if conditions.get('provenance', {}).get('kind') not in {'observed', 'synthetic_assumption'}:
        raise ValueError('Daily conditions need explicit provenance')
    if not conditions['provenance'].get('source'):
        raise ValueError('Daily conditions need source')
    resources = conditions['resources']
    if not isinstance(resources, dict):
        raise ValueError('Resource map required')
    for rid, spec in resources.items():
        if not isinstance(rid, str) or not rid or not spec.get('unit'):
            raise ValueError('Explicit resource ID and unit required')
        quantity(spec['opening_quantity'])
    for field in ['activity_consumption', 'quote_receipts']:
        for key, effects in conditions[field].items():
            if not isinstance(key, str) or not key or not isinstance(effects, dict) or not effects:
                raise ValueError('Resource effect map required')
            for rid, value in effects.items():
                if rid not in resources or quantity(value) == 0:
                    raise ValueError('Unknown resource or nonpositive effect')
    if set(conditions['quote_receipt_delay_minutes']) != set(conditions['quote_receipts']):
        raise ValueError('Every physical quote needs an explicit delivery delay')
    for value in conditions['quote_receipt_delay_minutes'].values():
        quantity(value)
    for receipt in conditions.get('opening_pending_receipts', []):
        if set(receipt) != {'minute', 'quantities', 'event_id'} or not isinstance(receipt['event_id'], str) or not receipt['event_id']:
            raise ValueError('Pending receipt identity and availability required')
        quantity(receipt['minute'])
        if not isinstance(receipt['quantities'], dict) or not receipt['quantities']:
            raise ValueError('Pending receipt quantities required')
        for rid, value in receipt['quantities'].items():
            if rid not in resources or quantity(value) == 0:
                raise ValueError('Pending receipt resource or quantity')
    needs = conditions['needs']
    if len({n['id'] for n in needs}) != len(needs):
        raise ValueError('Duplicate need ID')
    for need in needs:
        if not need.get('description') or not need.get('fulfilled_by') or quantity(need['desired_count']) == 0:
            raise ValueError('Explicit need and activity required')
        if need.get('mandatory') is not False:
            raise ValueError('This contract tracks optional needs, not unimplemented mandatory demands')


def settle(raw, case):
    conditions = case['daily_conditions']
    validate(conditions)
    choice = json.loads(raw)
    purchases = choice['purchases']
    events = case['events']
    if len(purchases) != len(events) or len({e['id'] for e in events}) != len(events):
        raise ValueError('Resource event roster')
    previous = -1
    pending = deepcopy(conditions.get('opening_pending_receipts', []))
    stocks = {rid: spec['opening_quantity'] for rid, spec in conditions['resources'].items()}
    fulfilled = {n['id']: 0 for n in conditions['needs']}
    journal = []
    for event, purchase in zip(events, purchases):
        if not re.fullmatch(r'([01][0-9]|2[0-3]):[0-5][0-9]', event['time']):
            raise ValueError('Resource event time')
        current = int(event['time'][:2]) * 60 + int(event['time'][3:])
        if purchase['id'] != event['id'] or current <= previous:
            raise ValueError('Resources require the complete chronological event order')
        previous = current
        due = [q for q in pending if q['minute'] <= current]
        pending = [q for q in pending if q['minute'] > current]
        for receipt in due:
            for rid, value in receipt['quantities'].items():
                stocks[rid] += value
        before = deepcopy(stocks)
        cid = purchase['candidate_id']
        if cid is not None:
            if cid not in {q['id'] for q in event['candidates']}:
                raise ValueError('Unknown purchased resource quote')
            quantities = conditions['quote_receipts'].get(cid, {})
            if quantities:
                delay = conditions['quote_receipt_delay_minutes'][cid]
                if delay:
                    pending.append({'minute': current + delay, 'quantities': deepcopy(quantities), 'event_id': event['id']})
                else:
                    for rid, value in quantities.items():
                        stocks[rid] += value
        consumption = conditions['activity_consumption'].get(event['activity_id'], {})
        for rid, value in consumption.items():
            if stocks[rid] < value:
                raise ValueError(f'Physical inventory shortage: {event["id"]}/{rid}')
            stocks[rid] -= value
        for need in conditions['needs']:
            if event['activity_id'] in need['fulfilled_by']:
                fulfilled[need['id']] += 1
        journal.append({'event_id': event['id'], 'opening': before, 'closing': deepcopy(stocks)})
    end_of_day_receipts = [q for q in pending if q['minute'] < 1440]
    for receipt in end_of_day_receipts:
        for rid, value in receipt['quantities'].items():
            stocks[rid] += value
    return {'opening_resources': {rid: spec['opening_quantity'] for rid, spec in conditions['resources'].items()},
            'closing_resources': stocks, 'resource_units': {rid: spec['unit'] for rid, spec in conditions['resources'].items()},
            'fulfilled_counts': fulfilled,
            'unfulfilled_counts': {n['id']: max(0, n['desired_count'] - fulfilled[n['id']]) for n in conditions['needs']},
            'journal': journal, 'end_of_day_receipts': end_of_day_receipts,
            'pending_receipts': [q for q in pending if q['minute'] >= 1440], 'complete': True,
            'scope': 'Only declared physical resources. Unfulfilled optional needs are reported, not converted to required purchases.'}
