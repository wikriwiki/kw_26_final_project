"""Compare preserved P010 ON execution receipts to the frozen graph ledger."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / 'output/recovery_20260928/multipolicy_v53/p010/on/on'
days = ('2025-07-21', '2025-07-22', '2025-07-23')
sources = [ARM / 'policy.ledger.jsonl',
           ARM / 'policy.ledger.jsonl.manifest.json'] + [
    ARM / 'metrics' / f'day_{day}.jsonl' for day in days]
source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sources}
ledger = {(r['aid'], r['day']): r for r in (
    json.loads(line) for line in sources[0].read_text(encoding='utf-8').splitlines()
    if line.strip()) if r['day'] in days}
assert len(ledger) == 240

metric_rows = []
for p in sources[2:]:
    rows = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()
            if line.strip()]
    assert len(rows) == 80 and all(r['status'] == 'ok' for r in rows)
    for row in rows:
        row['day'] = p.stem[4:]
    metric_rows.extend(rows)
assert len({(r['aid'], r['day']) for r in metric_rows}) == 240

receipts = []
row_comparisons = []
for row in metric_rows:
    key = (row['aid'], row['day'])
    raw = row.get('execution_receipts') or []
    purchases = [r for r in raw if r.get('kind') == 'purchase_receipt'
                 and int(r.get('amount') or 0) > 0]
    receipts.extend(purchases)
    observed = sum(int(r['amount']) for r in purchases)
    recorded = int(ledger[key]['offline_spent'])
    if observed != recorded:
        row_comparisons.append({'aid': key[0], 'day': key[1],
                                'receipt_offline_won': observed,
                                'graph_offline_won': recorded,
                                'difference_won': recorded - observed,
                                'positive_receipt_events': len(purchases),
                                'receipt_statuses': dict(Counter(r.get('purchase_status')
                                                                 for r in purchases))})

eligible = [r for r in receipts
            if (r.get('policy_facts') or {}).get('P010', {}).get('eligible_under_modeled_rules') is True]
requested = [r for r in eligible
             if ((r.get('decision') or {}).get('requested_payments') or {}).get('P010', 0) > 0]
funded = [r for r in eligible
          if (r.get('policy_facts') or {}).get('P010', {}).get('paid', 0) > 0]
result = {
    'purpose': 'read-only audit of preserved canonical response receipts and graph-derived daily ledger',
    'source_sha256': source_hashes,
    'citizen_days': len(metric_rows),
    'positive_receipt_events': len(receipts),
    'positive_receipt_statuses': dict(Counter(r.get('purchase_status') for r in receipts)),
    'receipt_eligible_events': len(eligible),
    'receipt_requested_policy_payment_events': len(requested),
    'receipt_requested_policy_won': sum(
        int((r['decision']['requested_payments'])['P010']) for r in requested),
    'receipt_funded_events': len(funded),
    'receipt_funded_won': sum(int(r['policy_facts']['P010']['paid']) for r in funded),
    'graph_eligible_offline_won': sum(int(r['eligible_offline_spent']) for r in ledger.values()),
    'graph_offline_won': sum(int(r['offline_spent']) for r in ledger.values()),
    'receipt_offline_won': sum(int(r['amount']) for r in receipts),
    'citizen_day_amount_mismatches': row_comparisons,
}
out = ARM / 'p010_receipt_ledger_audit.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
               encoding='utf-8')
print(json.dumps({key: value for key, value in result.items()
                  if key not in ('source_sha256', 'citizen_day_amount_mismatches')},
                 ensure_ascii=False, indent=2))
print(f'mismatch_rows={len(row_comparisons)} output_sha256={hashlib.sha256(out.read_bytes()).hexdigest()}')
