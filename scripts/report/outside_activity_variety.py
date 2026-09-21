"""How many different out-of-home purchase activities the plans actually use.

Seventeen activities in the catalog can lead to a purchase. In 480 plans, eleven of them
never appear once, and the eleven are exactly the ones that require leaving the house.
Within one category the split is clean: `home_online_goods` is used 95 times and
`shopping` - the same category, at a shop - zero.

That matters upstream of everything else. Purchase candidates are built from the
activities a plan contains, so an activity the plan never picks is a category the probe
never offers, which is a share of the citizen's own card mix that cannot be bought at any
price. Seven of eleven categories were never offered in v18.

So composition distance has a floor that no prompt can cross, and this counts the thing
that is actually reachable instead: the variety of out-of-home purchase activities the
plan commits to. It uses no answer-key value, only the citizen's own catalog.

    python scripts/report/outside_activity_variety.py \\
        --run v25=plans/responses.jsonl@70001 [...] --out variety.json
"""
from __future__ import annotations

import argparse
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sim'))

# Purchase-capable activities, split by whether finishing them requires leaving home.
# Stated here rather than inferred, because the split is the claim being measured.
AT_HOME = {'home_delivery', 'office_delivery', 'home_online_goods'}
OUTSIDE = {
    'meal_dine_in': '식사(매장)', 'meal_takeaway': '식사(포장)',
    'cafe_dine_in': '카페(매장)', 'cafe_takeaway': '카페(포장)', 'dessert': '디저트',
    'groceries': '마트', 'convenience': '편의점', 'shopping': '쇼핑(매장)',
    'hair': '미용', 'health_goods': '건강', 'leisure_service': '여가',
    'education_service': '교육', 'other_service': '기타', 'bar': '주점',
}


def load_rows(spec):
    """plan_responsiveness.load, but importable from a workstation without the server."""
    path, _, replicate = spec.partition('@')
    out = {}
    for line in io.open(path, encoding='utf-8'):
        if not line.strip():
            continue
        r = json.loads(line)
        if replicate and str(r.get('replicate')) != replicate:
            continue
        if r.get('valid') is not True:
            continue
        # 복제(seed)를 키에 넣지 않으면 네 seed 가 서로를 덮어써 384칸이 96칸이 된다.
        out[(r.get('replicate'), r['aid'], r['date'], r['case'], r['arm'])] = r
    return out


def measure(rows):
    used = Counter()
    per_cell, home_only = [], 0
    cells = 0
    for _, r in rows.items():
        events = (r.get('execution_plan') or {}).get('events', [])
        if not events:
            continue
        cells += 1
        kinds = {e['activity_id'] for e in events if e['activity_id'] in OUTSIDE}
        at_home = {e['activity_id'] for e in events if e['activity_id'] in AT_HOME}
        for a in kinds:
            used[a] += 1
        per_cell.append(len(kinds))
        if at_home and not kinds:
            home_only += 1
    return {'cells': cells, 'used': used, 'per_cell': per_cell, 'home_only_cells': home_only}


def paired_bootstrap(a, b, draws=4000, seed=20260921):
    """Difference in per-cell variety, resampling cells. Positive means b uses more kinds."""
    n = min(len(a), len(b))
    if n < 20:
        return None
    diffs = [b[i] - a[i] for i in range(n)]
    rng = random.Random(seed)
    point = sum(diffs) / n
    draw = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    lo, hi = draw[int(0.025 * draws)], draw[int(0.975 * draws) - 1]
    return {'mean_difference': point, 'ci95': [lo, hi],
            'resolved': (lo > 0) == (hi > 0), 'cells_compared': n,
            'note': ('칸을 재추출한 차이다. 같은 칸끼리 짝지은 것이 아니라 분포끼리 견준 것이므로 '
                     '후보가 같은 코호트·같은 seed 일 때만 뜻이 있다.')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, metavar='label=responses.jsonl@rep')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    results, order = {}, []
    for item in args.run:
        label, spec = item.split('=', 1)
        results[label] = measure(load_rows(spec))
        order.append(label)

    doc = {'activities_outside': OUTSIDE, 'activities_at_home': sorted(AT_HOME), 'runs': {}}
    for label in order:
        m = results[label]
        per = m['per_cell']
        doc['runs'][label] = {
            'cells': m['cells'],
            'distinct_kinds_used': len(m['used']),
            'kinds_available': len(OUTSIDE),
            'never_used': sorted(set(OUTSIDE) - set(m['used'])),
            'uses_by_activity': dict(sorted(m['used'].items(), key=lambda kv: -kv[1])),
            'mean_kinds_per_cell': (sum(per) / len(per)) if per else 0.0,
            'cells_with_any_outside': sum(1 for v in per if v),
            'cells_buying_only_from_home': m['home_only_cells'],
        }
    if len(order) == 2:
        doc['paired_vs'] = {'a': order[0], 'b': order[1],
                            'result': paired_bootstrap(results[order[0]]['per_cell'],
                                                       results[order[1]]['per_cell'])}

    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(doc, ensure_ascii=False, indent=1))
    print('wrote', args.out)
    for label in order:
        d = doc['runs'][label]
        print('  %-12s 종류 %2d/%d · 칸당 %.3f · 집밖 있는 칸 %d/%d'
              % (label, d['distinct_kinds_used'], d['kinds_available'],
                 d['mean_kinds_per_cell'], d['cells_with_any_outside'], d['cells']))
        print('               한 번도 안 쓴 것: %s' % ', '.join(d['never_used']))
    pv = (doc.get('paired_vs') or {}).get('result')
    if pv:
        print('  차이 (%s − %s)  %+.4f  [%+.4f, %+.4f]%s'
              % (order[1], order[0], pv['mean_difference'], pv['ci95'][0], pv['ci95'][1],
                 '' if pv['resolved'] else '   ← 구간이 0을 지난다'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
