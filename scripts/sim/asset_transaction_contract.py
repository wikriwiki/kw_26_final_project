"""Explicit quoted purchases and prepaid acquisitions, separate from the legacy engine.

One consumption choice per supplied activity, in its supplied chronological order.
Free/declined activities remain zero. Offers are supplied facts, never inferred here.
"""
import json
from asset_ledger import settle_asset_actions
from transaction_ledger import won


def schema(case):
    wallet_ids = sorted(set(case['wallet_lots']) | {q['wallet_id'] for q in case['offers'].values()})
    branches = []
    for event in case['events']:
        props = {'kind': {'const': 'consume'}, 'id': {'const': event['id']},
                 'candidate_id': {'enum': [None] + [c['id'] for c in event['candidates']]},
                 'cash_payment': {'type': 'integer', 'minimum': 0},
                 'wallet_spend': {'type': 'object', 'properties': {w: {'type': 'integer', 'minimum': 0} for w in wallet_ids}, 'additionalProperties': False},
                 'reason': {'type': 'string', 'minLength': 1}}
        branches.append({'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False})
    for oid, quote in case['offers'].items():
        props = {'kind': {'const': 'acquire_wallet'}, 'id': {'const': 'acquire:' + oid},
                 'offer_id': {'const': oid}, 'units': {'type': 'integer', 'minimum': 1, 'maximum': quote['max_units']},
                 'reason': {'type': 'string', 'minLength': 1}}
        branches.append({'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False})
    return {'type': 'object', 'properties': {'actions': {'type': 'array', 'minItems': len(case['events']),
             'maxItems': len(case['events']) + len(case['offers']), 'items': {'anyOf': branches} if branches else False}},
            'required': ['actions'], 'additionalProperties': False}


def inspect(raw, case):
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) != {'actions'} or not isinstance(obj['actions'], list):
        raise ValueError('Output shape')
    expected = {e['id']: e for e in case['events']}
    if len(expected) != len(case['events']) or any(not isinstance(k, str) or k.startswith('acquire:') for k in expected):
        raise ValueError('Input event IDs')
    order = []; actions = []; eligibility = {}
    for action in obj['actions']:
        if not isinstance(action, dict) or not isinstance(action.get('reason'), str) or not action['reason'].strip():
            raise ValueError('Missing choice reason')
        if action.get('kind') == 'acquire_wallet':
            if set(action) != {'kind', 'id', 'offer_id', 'units', 'reason'} or action['id'] != 'acquire:' + action['offer_id']:
                raise ValueError('Acquisition fields')
            actions.append(action)
        elif action.get('kind') == 'consume':
            if set(action) != {'kind', 'id', 'candidate_id', 'cash_payment', 'wallet_spend', 'reason'} or action['id'] not in expected:
                raise ValueError('Purchase fields/roster')
            event = expected[action['id']]; order.append(action['id'])
            candidates = {c['id']: c for c in event['candidates']}
            if len(candidates) != len(event['candidates']): raise ValueError('Duplicate input candidate')
            candidate = action['candidate_id']
            if candidate is None:
                if action['cash_payment'] != 0 or action['wallet_spend'] != {}:
                    raise ValueError('No purchase cannot create a payment')
                amount = 0; allowed = []
            else:
                if candidate not in candidates: raise ValueError('Wrong event candidate')
                amount = won(candidates[candidate]['price_won'], 'quoted price')
                allowed = candidates[candidate]['eligible_wallets']
            eligibility[action['id']] = allowed
            actions.append(dict(action, amount=amount, channel=event['channel']))
        else: raise ValueError('Unknown action kind')
    if order != [e['id'] for e in case['events']]: raise ValueError('Chronological event roster')
    ledger = settle_asset_actions(actions, cash=case['cash'], wallet_lots=case['wallet_lots'],
                                  offers=case['offers'], eligible_wallets_by_purchase=eligibility)
    return obj, ledger
