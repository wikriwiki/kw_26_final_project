"""Score every answer-key indicator from the recorded purchase probe, one by one.

The point is to show the comparison rather than summarise it. For each indicator the
report carries three things: what the published study measured, what our runs produce,
and - when the two are not the same kind of number - an empty cell with the reason,
never a number that would look like agreement.

Sources joined per cell:
    purchase_source_*.json      cells[].transaction_case.events[]
                                anchor, activity_id, channel, candidates[{id, price_won}]
    purchase_*/responses.jsonl  raw.purchases[{id, candidate_id, wallet_spend}]

The sector mapping is stated here rather than buried, because it is an assumption. The
catalog sells unit items (a 1kg bag of rice, one bottle of water) and the published
studies measure card sales by merchant category. Only ratios survive that difference,
never absolute won.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path

SECTOR = {
    'meal_dine_in': '식사', 'meal_takeaway': '식사',
    'home_delivery': '식사', 'office_delivery': '식사',
    'cafe_dine_in': '카페', 'cafe_takeaway': '카페', 'dessert': '카페',
    'groceries': '쇼핑+마트', 'convenience': '쇼핑+마트', 'shopping': '쇼핑+마트',
    'home_online_goods': '쇼핑+마트',
    'hair': '서비스', 'health_goods': '서비스', 'leisure_service': '서비스',
    'education_service': '서비스', 'other_service': '서비스', 'bar': '유흥',
}


def is_excluded(ev):
    """P012 excludes large marts, department stores, online malls and bars.

    The catalog has no mart or department tier, so the proxy is the online channel
    or a bar. Stating it here keeps it auditable.
    """
    return ev.get('channel') == 'online' or ev.get('activity_id') == 'bar'


def purchases(source, responses):
    """Yield one record per cell with its purchased items, joined across the two files."""
    src = json.loads(Path(source).read_text(encoding='utf-8'))
    cells = {(c['aid'], c['case'], c['arm']): c for c in src['cells']}
    with io.open(responses, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('valid') is not True:
                continue
            cell = cells.get((r['aid'], r['case'], r['arm']))
            if cell is None:
                continue
            events = {e['id']: e for e in cell['transaction_case']['events']}
            try:
                raw = json.loads(r['raw'])
            except (TypeError, ValueError):
                continue
            bought = []
            for p in raw.get('purchases', []):
                if not p.get('candidate_id'):
                    continue
                ev = events.get(p['id'])
                if ev is None:
                    continue
                quote = None
                for q in ev.get('candidates', []):
                    if q.get('id') == p['candidate_id']:
                        quote = q
                        break
                if quote is None:
                    continue
                bought.append({
                    'activity_id': ev.get('activity_id'),
                    'anchor': ev.get('anchor'),
                    'channel': ev.get('channel'),
                    'price': int(quote['price_won']),
                    'wallet': bool(p.get('wallet_spend')),
                    'excluded': is_excluded(ev),
                    'sector': SECTOR.get(ev.get('activity_id'), '기타'),
                })
            yield {'aid': r['aid'], 'case': r['case'], 'arm': r['arm'], 'items': bought}


def totals(recs, homes):
    """Per (case, arm): the aggregates every indicator is built from."""
    out = defaultdict(lambda: defaultdict(float))
    for rec in recs:
        key = (rec['case'], rec['arm'])
        home = str(homes.get(rec['aid']) or '')
        out[key]['cells'] += 1
        for it in rec['items']:
            p = it['price']
            out[key]['total'] += p
            out[key]['sector:' + it['sector']] += p
            out[key]['excluded' if it['excluded'] else 'eligible'] += p
            anchor = str(it['anchor'] or '')
            if anchor.startswith('zone:'):
                z = anchor.removeprefix('zone:')
                if z == home:
                    out[key]['home_dong'] += p
                if z[:5] != home[:5]:
                    out[key]['out_district'] += p
    return out


def pct(on, off):
    """Percent change. None when the baseline is zero - a ratio on nothing is not a number."""
    if not off:
        return None
    return 100.0 * (on - off) / off


def share_shift(t, mech, key):
    """Percentage-point change in a share of that mechanism's own spend."""
    on_t = t.get((mech, 'on'), {}).get('total', 0.0)
    off_t = t.get((mech, 'off'), {}).get('total', 0.0)
    if not on_t or not off_t:
        return None
    return 100.0 * (t[(mech, 'on')].get(key, 0.0) / on_t
                    - t[(mech, 'off')].get(key, 0.0) / off_t)


def indicators(t):
    """Every indicator this probe can build. Ones it cannot are simply absent."""
    def g(case, arm, k):
        return t.get((case, arm), {}).get(k, 0.0)

    v = {}
    v['P012-1'] = pct(g('cashback', 'on', 'eligible'), g('cashback', 'off', 'eligible'))
    v['P012-2'] = pct(g('cashback', 'on', 'excluded'), g('cashback', 'off', 'excluded'))
    v['EM-3'] = pct(g('grant', 'on', 'total'), g('grant', 'off', 'total'))
    v['LV-1'] = pct(g('local_voucher', 'on', 'total'), g('local_voucher', 'off', 'total'))
    v['LV-2'] = share_shift(t, 'local_voucher', 'home_dong')
    v['LV-3'] = share_shift(t, 'local_voucher', 'out_district')
    for name, sec in (('DS-1', '식사'), ('DS-2', '쇼핑+마트'), ('DS-4', '카페')):
        v[name] = pct(g('distancing', 'on', 'sector:' + sec),
                      g('distancing', 'off', 'sector:' + sec))
    if v.get('DS-1') is not None and v.get('DS-2') is not None:
        # Rank indicator: retail growth minus restaurant growth. Positive = order holds.
        v['DS-3'] = v['DS-2'] - v['DS-1']
    return {k: val for k, val in v.items() if val is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True,
                    metavar='label=source.json:responses.jsonl')
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    personas = json.loads(Path(args.frozen).read_text(encoding='utf-8'))['personas']
    homes = {p['id']: p.get('home_dong_code') for p in personas}

    per_run = {}
    for item in args.run:
        label, paths = item.split('=', 1)
        src, resp = paths.split(':', 1)
        t = totals(purchases(src, resp), homes)
        per_run[label] = {
            'indicators': indicators(t),
            'totals': {case + '|' + arm: dict(d) for (case, arm), d in t.items()},
        }

    ids = sorted({k for r in per_run.values() for k in r['indicators']})
    pooled = {}
    for k in ids:
        vals = [r['indicators'][k] for r in per_run.values() if k in r['indicators']]
        pooled[k] = {
            'mean': statistics.mean(vals), 'n_runs': len(vals),
            'min': min(vals), 'max': max(vals),
            'sd': statistics.stdev(vals) if len(vals) > 1 else None,
        }

    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(json.dumps({
        'sector_map': SECTOR,
        'excluded_proxy': 'online channel or bar',
        'unit': 'percent change on vs off, or percentage-point change in a share',
        'per_run': per_run, 'pooled': pooled,
    }, ensure_ascii=False, indent=1))

    print('wrote', args.out)
    for k in ids:
        p = pooled[k]
        sd = ('%.1f' % p['sd']) if p['sd'] is not None else '-'
        print('  %-8s %+8.1f  (런 %d · %+.1f~%+.1f · SD %s)'
              % (k, p['mean'], p['n_runs'], p['min'], p['max'], sd))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
