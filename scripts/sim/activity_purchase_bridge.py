"""Join typed activities to explicitly supplied quotes; never manufacture spend.

Quote construction and policy eligibility are separate data-provider duties.
Missing prices are an error, not an invitation to use a historical spend anchor.
"""
from copy import deepcopy
from action_plan_contract import catalog, inspect
from transaction_ledger import won


def prepare_case(*, raw_plan, cell, quotes_by_event, cash, wallet_lots, offers,max_shift_minutes=10):
    report = inspect(raw_plan, cell,max_shift=max_shift_minutes)
    if not report['valid']: raise ValueError('Planner contract incomplete')
    activities = catalog(cell); events = []; audit = []
    provided_ids = {a['id'] for a in cell.get('provided_activities', [])}
    plan = report['execution_plan']['events']
    for rule in cell.get('evaluation_requirements', []):
        if rule['kind'] == 'no_outside': violation = any(e['anchor'] != 'residence' for e in plan)
        elif rule['kind'] == 'forbid_activity': violation = any(e['activity_id'] in rule['ids'] for e in plan)
        elif rule['kind'] == 'forbid_after': violation = any(e['activity_id'] in rule['ids'] and e['time'] >= rule['time'] for e in plan)
        else: raise ValueError('Unknown supplied activity restriction')
        if violation: raise ValueError('Plan violates a supplied activity restriction')
    if not set(quotes_by_event) <= {str(i) for i in range(len(plan))}: raise ValueError('Unknown quote event')
    for i, event in enumerate(plan):
        aid = event['activity_id']; spec = activities[aid]; channel = spec['purchase_channel']; key = str(i)
        if aid in provided_ids and channel is None and spec.get('billing_status') != 'no_transaction':
            raise ValueError('Provided appointment has unresolved billing; cannot assume free')
        if channel is None:
            if quotes_by_event.get(key): raise ValueError('Free activity cannot receive commercial substitutes')
            candidates = []
        else:
            if channel not in {'online','offline'}: raise ValueError('Unsupported transaction channel')
            if key not in quotes_by_event: raise ValueError('Missing explicit quote coverage')
            candidates = deepcopy(quotes_by_event[key])
            if not isinstance(candidates, list): raise ValueError('Quote list required')
            seen = set()
            for quote in candidates:
                if not isinstance(quote.get('id'), str) or not quote['id'] or quote['id'] in seen: raise ValueError('Quote IDs')
                seen.add(quote['id']); won(quote.get('price_won'), 'quoted price')
                if not quote.get('description') or not quote.get('price_provenance'): raise ValueError('Unlabelled price assumption')
                if not isinstance(quote.get('eligible_wallets'), list): raise ValueError('Missing eligibility decision')
                if quote.get('channel') != channel: raise ValueError('Quote/choice channel mismatch')
        events.append({'id': 'event:' + key, 'intent': spec['intent'], 'time': event['time'],
                       'anchor': event['anchor'], 'activity_id': aid, 'channel': channel or 'offline', 'candidates': candidates})
        audit.append({'event_id': 'event:' + key, 'free_activity': channel is None, 'quoted_candidates': len(candidates)})
    return {'context': cell['user'], 'cash': won(cash, 'cash'), 'wallet_lots': deepcopy(wallet_lots),
            'offers': deepcopy(offers), 'events': events}, {'schedule_report': report, 'coverage': audit,
            'max_shift_minutes':max_shift_minutes,
            'scope': 'Explicit candidate purchases only. Home-food inventory, missing obligations and overall daily demand coverage are not inferred.'}
