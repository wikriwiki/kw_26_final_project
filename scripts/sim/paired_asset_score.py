"""Descriptive matched contrasts, revalidated from raw asset purchase choices."""
from collections import Counter
from datetime import date
import statistics
from asset_transaction_contract import inspect

METRICS = ['total_consumption','offline_consumption','online_consumption',
           'own_funded_consumption','concession_funded_consumption',
           'asset_acquisition_cash_outflow','cash_outflow_total']


def score(rows, *, roster, days, seeds):
    for label, values in [('roster',roster),('days',days),('seeds',seeds)]:
        if not values or len(set(values)) != len(values): raise ValueError('Invalid '+label)
    for d in days:
        if date.fromisoformat(d).isoformat() != d: raise ValueError('Invalid calendar date')
    expected = {(a,d,s,arm) for a in roster for d in days for s in seeds for arm in ['off','on']}
    counts = Counter((r['aid'],r['date'],r['replicate'],r['arm']) for r in rows)
    if set(counts) != expected or any(n != 1 for n in counts.values()): raise ValueError('Incomplete/duplicate matched matrix')
    ledgers = {}
    for row in rows:
        if row.get('valid') is not True: raise ValueError('Failed response cannot be omitted or zero-filled')
        protocol=row.get('transaction_protocol','v1')
        if protocol=='v2':
            from asset_transaction_contract_v2 import inspect as check
        elif protocol in {'v3','v4'}:
            from asset_transaction_contract_v3 import inspect as check
        elif protocol=='v1':check=inspect
        else:raise ValueError('Unknown transaction protocol')
        _, ledger = check(row['raw'], row['transaction_case'])
        if 'daily_conditions' in row['transaction_case']:
            from daily_resource_contract import settle
            resources = settle(row['raw'], row['transaction_case'])
            if 'resource_ledger' in row and row['resource_ledger'] != resources:
                raise ValueError('Recorded physical state differs from raw choices')
        ledgers[(row['aid'],row['date'],row['replicate'],row['arm'])] = ledger
    by_seed = []
    for seed in seeds:
        result = {'replicate': seed, 'metrics': {}}
        for metric in METRICS:
            per_person = {a:{arm: statistics.mean(ledgers[(a,d,seed,arm)][metric] for d in days) for arm in ['off','on']} for a in roster}
            means = {arm:statistics.mean(v[arm] for v in per_person.values()) for arm in ['off','on']}
            delta = means['on']-means['off']
            result['metrics'][metric] = {'off_mean':means['off'], 'on_mean':means['on'], 'difference':delta,
                'relative_change':delta/means['off'] if means['off'] else None,
                'paired_differences':{a:v['on']-v['off'] for a,v in per_person.items()}}
        by_seed.append(result)
    return {'complete_matrix':True,'citizens':len(roster),'days':len(days),'replicates':len(seeds),
            'unit':'won per person per day','by_seed':by_seed,
            'scope':'Equal-weight descriptive conditional-scenario contrasts. No population inference, significance claim, target fitting or empirical magnitude validation. Asset acquisition is separate from consumption.'}
