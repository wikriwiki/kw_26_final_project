"""Matched-calendar experimental accounting outcomes, without a policy target.

This file does not read the legacy graph or infer missing citizen-days as zero.
Intervals require a separately registered sampling design; this module reports
finite-sample descriptive contrasts for each independent run seed.
"""
from collections import Counter
from datetime import date
import math
import statistics
from transaction_ledger import won


def ledger_totals(ledger):
    if ledger.get('complete') is not True:
        raise ValueError('Incomplete ledger')
    cash = won(ledger['opening_cash'], 'opening_cash')
    wallets = {k: won(v, k) for k, v in ledger['opening_wallets'].items()}
    used = {k: 0 for k in wallets}
    channels = {'offline': 0, 'online': 0}
    own = 0
    ids = set()
    for tx in ledger['transactions']:
        if not isinstance(tx['id'], str) or not tx['id'] or tx['id'] in ids:
            raise ValueError('Transaction ID roster')
        ids.add(tx['id'])
        if tx['channel'] not in channels:
            raise ValueError('Unknown channel')
        amount = won(tx['amount'], 'amount')
        funding = 0
        for pid, raw in tx['policy_spend'].items():
            if pid not in used:
                raise ValueError('Unknown funding source')
            payment = won(raw, pid)
            funding += payment
            used[pid] += payment
        if funding > amount or won(tx['own_payment'], 'own_payment') != amount - funding:
            raise ValueError('Transaction funding does not reconcile')
        own += amount - funding
        channels[tx['channel']] += amount
    if own > cash or any(used[k] > wallets[k] for k in wallets):
        raise ValueError('Overspent funds')
    if won(ledger['closing_cash'], 'closing_cash') != cash - own:
        raise ValueError('Cash does not reconcile')
    if set(ledger['closing_wallets']) != set(wallets) or any(
        won(ledger['closing_wallets'][k], k) != wallets[k] - used[k] for k in wallets
    ):
        raise ValueError('Wallets do not reconcile')
    totals = {'total_including_online': sum(channels.values()), 'offline_total': channels['offline'],
              'online_total': channels['online'], 'own_payment_total': own, 'policy_payment_total': sum(used.values())}
    if any(won(ledger[k], k) != value for k, value in totals.items()):
        raise ValueError('Recorded totals do not reconcile')
    return totals


def score(rows, *, roster, days, seeds, weights=None):
    """Require the complete aid x day x seed x on/off matrix before scoring.

    A row contains aid, day, replicate, arm, and a validated complete ledger.
    Days are the SAME calendar dates on both arms. Weights are fixed beforehand,
    not inferred from expenditure. No filtering to citizens with positive spend.
    """
    for name, values in [('roster', roster), ('days', days), ('seeds', seeds)]:
        if not values or len(set(values)) != len(values):
            raise ValueError(f'Empty or duplicate {name}')
    for day in days:
        if date.fromisoformat(day).isoformat() != day:
            raise ValueError('Non-canonical date')
    weights = dict(weights) if weights is not None else {aid: 1.0 for aid in roster}
    if set(weights) != set(roster) or any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError('Invalid fixed weights')
    expected = {(aid, day, seed, arm) for aid in roster for day in days for seed in seeds for arm in ['off', 'on']}
    counts = Counter((r['aid'], r['day'], r['replicate'], r['arm']) for r in rows)
    if set(counts) != expected or any(n != 1 for n in counts.values()):
        raise ValueError('Incomplete, extra, or duplicate matched matrix')
    values = {(r['aid'], r['day'], r['replicate'], r['arm']): ledger_totals(r['ledger']) for r in rows}
    metrics = list(next(iter(values.values())))
    total_weight = sum(weights.values())
    by_seed = []
    for seed in seeds:
        entry = {'replicate': seed, 'metrics': {}}
        for metric in metrics:
            individuals = {aid: {arm: sum(values[(aid, day, seed, arm)][metric] for day in days) / len(days)
                                 for arm in ['off', 'on']} for aid in roster}
            means = {arm: sum(weights[aid] * individuals[aid][arm] for aid in roster) / total_weight for arm in ['off', 'on']}
            difference = means['on'] - means['off']
            entry['metrics'][metric] = {'off_won_per_person_day': means['off'], 'on_won_per_person_day': means['on'],
                'difference_won_per_person_day': difference,
                'relative_change': difference / means['off'] if means['off'] else None,
                'relative_change_definition': '(weighted mean on - weighted mean off) / weighted mean off',
                'individual_paired_differences': {aid: v['on'] - v['off'] for aid, v in individuals.items()}}
        by_seed.append(entry)
    across = {}
    for metric in metrics:
        differences = [entry['metrics'][metric]['difference_won_per_person_day'] for entry in by_seed]
        across[metric] = {'mean_difference': statistics.mean(differences),
                         'run_standard_deviation': statistics.stdev(differences) if len(differences) > 1 else None,
                         'min_difference': min(differences), 'max_difference': max(differences)}
    return {'complete_matrix': True, 'citizens': len(roster), 'days': len(days), 'replicates': len(seeds),
            'unit': 'won per person per day', 'by_seed': by_seed, 'across_seed_descriptive': across,
            'scope': 'Descriptive matched-calendar contrasts. Not a significance test, empirical magnitude validation, or proof of population generalization. Eligibility must be validated before this accounting audit.'}
