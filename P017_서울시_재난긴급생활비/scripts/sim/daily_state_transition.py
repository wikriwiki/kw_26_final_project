"""Explicit next-day opening state from actual choices, never initial-state reset.

The next template supplies exogenous assumptions (duty, menu, new optional needs).
Its resource opening quantities are placeholders; actual closing stocks and late
receipts determine the opening state. No income, expiry or new needs are inferred.
"""
from copy import deepcopy

from asset_transaction_contract_v4 import inspect
from daily_resource_contract import settle, validate


def advance(previous_case, previous_raw, next_template):
    validate(next_template)
    _, money = inspect(previous_raw, previous_case)
    physical = settle(previous_raw, previous_case)
    conditions = deepcopy(next_template)
    units = {rid: value['unit'] for rid, value in conditions['resources'].items()}
    if units != physical['resource_units']:
        raise ValueError('Cannot drop or silently convert carried resources')
    if conditions.get('opening_pending_receipts'):
        raise ValueError('New template must not invent or duplicate pending deliveries')
    for rid, value in conditions['resources'].items():
        value['opening_quantity'] = physical['closing_resources'][rid]
    conditions['opening_pending_receipts'] = [dict(q, minute=q['minute']-1440) for q in physical['pending_receipts']]
    needs = {}
    for old in previous_case['daily_conditions']['needs']:
        remaining = physical['unfulfilled_counts'][old['id']]
        if remaining:
            needs[old['id']] = dict(deepcopy(old), desired_count=remaining)
    for added in next_template['needs']:
        nid = added['id']
        if nid in needs:
            old = {k:v for k,v in needs[nid].items() if k!='desired_count'}
            new = {k:v for k,v in added.items() if k!='desired_count'}
            if old != new:
                raise ValueError('Reused need identity has incompatible meaning')
            needs[nid]['desired_count'] += added['desired_count']
        else:
            needs[nid] = deepcopy(added)
    conditions['needs'] = list(needs.values())
    for need in needs.values():
        # Fulfillment may legitimately be a non-consuming activity. Its identity
        # is retained as supplied, not replaced by an inferred substitute.
        if not isinstance(need['fulfilled_by'], list) or not all(isinstance(x,str) and x for x in need['fulfilled_by']):
            raise ValueError('Explicit fulfillment identities required')
    validate(conditions)
    return {'cash': money['closing_cash'], 'wallet_lots': deepcopy(money['closing_wallet_lots']),
            'daily_conditions': conditions}
