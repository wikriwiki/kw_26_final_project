"""Read-only P010 treatment accounting diagnostic before graph restore."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.neo4j_load._common import driver_session
from scripts.report.export_policy_daily_ledger import SPEND_QUERY, _policy_amount
from scripts.sim.score_policy import apply_policy_eligibility


ROOT = Path('/data/multipolicy_v53_20260928/p010')
ARM = ROOT / 'on'
roster = json.loads((ROOT / 'roster.json').read_text(encoding='utf-8'))
assert len(roster) == 80
days = ('2025-07-21', '2025-07-22', '2025-07-23')
policy_file = 'data/experiments/P010_v53_policy_20260927.json'

transactions = []
with driver_session() as session:
    for day in days:
        rows = [dict(record) for record in session.run(SPEND_QUERY, day=day, aids=roster)]
        apply_policy_eligibility(rows, policy_file)
        for row in rows:
            row['day'] = day
            transactions.append(row)

positive = [r for r in transactions if int(r.get('amt') or 0) > 0]
eligible = [r for r in positive if r['elig']]
funded = [r for r in positive if _policy_amount(r['spent_from_policy'], 'P010') > 0]
assert all(r['elig'] for r in funded)
by_day = {}
for day in days:
    subset = [r for r in positive if r['day'] == day]
    by_day[day] = {
        'positive_purchase_events': len(subset),
        'eligible_purchase_events': sum(bool(r['elig']) for r in subset),
        'funded_purchase_events': sum(_policy_amount(r['spent_from_policy'], 'P010') > 0 for r in subset),
        'eligible_purchase_won': sum(int(r['amt']) for r in subset if r['elig']),
        'funded_won': sum(_policy_amount(r['spent_from_policy'], 'P010') for r in subset),
    }

metrics = []
for day in days:
    path = ARM / 'metrics' / f'day_{day}.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 80 and len({row['aid'] for row in rows}) == 80
    metrics.extend(rows)

result = {
    'source': 'read-only current P010 ON graph and canonical 240 citizen-day metrics',
    'policy_file': policy_file,
    'days': list(days),
    'citizens': len(roster),
    'positive_purchase_events': len(positive),
    'eligible_purchase_events': len(eligible),
    'ineligible_purchase_events': len(positive) - len(eligible),
    'eligible_purchase_without_policy_payment_events': sum(
        _policy_amount(r['spent_from_policy'], 'P010') == 0 for r in eligible
    ),
    'funded_purchase_events': len(funded),
    'positive_purchase_won': sum(int(r['amt']) for r in positive),
    'eligible_purchase_won': sum(int(r['amt']) for r in eligible),
    'funded_won': sum(_policy_amount(r['spent_from_policy'], 'P010') for r in funded),
    'citizen_days_with_eligible_purchase': len({(r['day'], r['aid']) for r in eligible}),
    'citizen_days_with_policy_payment': len({(r['day'], r['aid']) for r in funded}),
    'policy_request_positive_citizen_days': sum(
        int(r.get('cm_policy_requested_total') or 0) > 0 for r in metrics
    ),
    'policy_requested_won': sum(int(r.get('cm_policy_requested_total') or 0) for r in metrics),
    'policy_allocated_positive_citizen_days': sum(
        int(r.get('cm_policy_allocated_total') or 0) > 0 for r in metrics
    ),
    'policy_allocated_won': sum(int(r.get('cm_policy_allocated_total') or 0) for r in metrics),
    'policy_hits_positive_citizen_days': sum(int(r.get('policy_hits') or 0) > 0 for r in metrics),
    'stage1_grant_style_present_citizen_days': sum(r.get('s1_grant_style') is not None for r in metrics),
    'stage1_grant_use_present_citizen_days': sum(r.get('s1_grant_use') is not None for r in metrics),
    'choice_mode_citizen_days': sum(r.get('cm_grant_choice_mode') is True for r in metrics),
    'choice_share_positive_citizen_days': sum(
        float(r.get('cm_grant_choice_share_mean') or 0) > 0 for r in metrics
    ),
    'by_day': by_day,
    'eligibility_by_category': dict(sorted(Counter(
        f"{r.get('l1')} / {r.get('sub')} / {'eligible' if r['elig'] else 'ineligible'}"
        for r in positive
    ).items())),
}
out = ARM / 'p010_wallet_diagnosis.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: value for key, value in result.items()
                  if key not in ('eligibility_by_category',)}, ensure_ascii=False, indent=2))
