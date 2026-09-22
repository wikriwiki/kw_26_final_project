"""Strict experimental accounting: validate chosen transactions, never invent spend.

This is separate from the historical calibrated consumption engine. It is not a
price model or a policy-effect estimator. Wallet eligibility must be resolved from
the actual policy rules by the caller; missing eligibility is not unrestricted.
"""
from __future__ import annotations
from copy import deepcopy


def won(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field}: expected nonnegative integer won')
    if not value >= 0 or value == float('inf') or int(value) != value:
        raise ValueError(f'{field}: expected nonnegative integer won')
    return int(value)


def settle_choices(transactions, *, cash, wallets, eligible_wallets_by_transaction):
    """Return a complete immutable ledger, or reject the entire proposed day.

    transactions: [{id, channel: offline|online, amount, policy_spend:{pid:won}}]
    wallets contains only balances currently spendable on this date. An explicit
    set of eligible wallet IDs is required for every transaction (including zero).
    No automatic wallet substitution, online-share formula, scaling or minimum.
    """
    opening_cash = won(cash, 'cash')
    opening_wallets = {str(k):won(v, f'wallet:{k}') for k,v in wallets.items()}
    if len(opening_wallets) != len(wallets):
        raise ValueError('duplicate normalized wallet IDs')
    result = []
    seen = set()
    used = {pid:0 for pid in opening_wallets}
    own_total = 0
    for original in transactions:
        tx = deepcopy(original)
        key = tx.get('id')
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError('missing or duplicate transaction ID')
        seen.add(key)
        if tx.get('channel') not in {'offline','online'}:
            raise ValueError(f'{key}: unsupported channel')
        if key not in eligible_wallets_by_transaction:
            raise ValueError(f'{key}: missing eligibility evaluation')
        allowed = set(eligible_wallets_by_transaction[key])
        amount = won(tx.get('amount'), f'{key}:amount')
        requested = tx.get('policy_spend')
        if not isinstance(requested, dict):
            raise ValueError(f'{key}: explicit policy_spend required')
        settled = {}
        for pid, raw in requested.items():
            payment = won(raw, f'{key}:{pid}')
            if pid not in opening_wallets or pid not in allowed:
                raise ValueError(f'{key}: unavailable or ineligible wallet {pid}')
            settled[pid] = payment
            used[pid] += payment
        policy_total = sum(settled.values())
        if policy_total > amount:
            raise ValueError(f'{key}: policy payment exceeds purchase')
        tx.update(amount=amount, policy_spend=settled, own_payment=amount-policy_total)
        own_total += tx['own_payment']
        result.append(tx)
    if set(eligible_wallets_by_transaction) != seen:
        raise ValueError('eligibility roster differs from transaction roster')
    if own_total > opening_cash:
        raise ValueError('personal funds exceeded')
    if any(used[pid] > opening_wallets[pid] for pid in used):
        raise ValueError('wallet funds exceeded')
    totals = {ch:sum(t['amount'] for t in result if t['channel']==ch) for ch in ['offline','online']}
    total = sum(totals.values())
    assert total == own_total + sum(used.values())
    return {'transactions':result, 'opening_cash':opening_cash,
            'closing_cash':opening_cash-own_total, 'opening_wallets':opening_wallets,
            'closing_wallets':{pid:opening_wallets[pid]-used[pid] for pid in used},
            'own_payment_total':own_total, 'policy_payment_total':sum(used.values()),
            'offline_total':totals['offline'], 'online_total':totals['online'],
            'total_including_online':total, 'complete':True}
