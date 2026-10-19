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
    by_cell = {}
    cells = 0
    for key, r in rows.items():
        events = (r.get('execution_plan') or {}).get('events', [])
        if not events:
            continue
        cells += 1
        kinds = {e['activity_id'] for e in events if e['activity_id'] in OUTSIDE}
        at_home = {e['activity_id'] for e in events if e['activity_id'] in AT_HOME}
        for a in kinds:
            used[a] += 1
        per_cell.append(len(kinds))
        # 같은 칸(사람·기전·팔)을 두 후보가 어떻게 달리 계획했는지 짝지으려면
        # 칸 이름으로 찾을 수 있어야 한다. 복제는 키에서 뺀다 — 후보끼리 seed 가 같다.
        by_cell[key[1:]] = len(kinds)
        if at_home and not kinds:
            home_only += 1
    return {'cells': cells, 'used': used, 'per_cell': per_cell,
            'by_cell': by_cell, 'home_only_cells': home_only}


def paired_bootstrap(a, b, draws=4000, seed=20260921, a_cells=None, b_cells=None):
    """Difference in per-cell variety. Matched by cell when both sides expose their cells.

    Two candidates plan the same 480 cells, so the same citizen on the same day appears on
    both sides. Matching them removes the between-citizen variance, which is most of the
    variance here: v18 ran 0.03 per cell on sixty citizens and 0.16 on twelve. An unmatched
    comparison carries that spread into the interval for no reason.

    Falls back to comparing the two distributions when the cells cannot be matched.
    """
    if a_cells and b_cells:
        shared = sorted(set(a_cells) & set(b_cells))
        diffs = [b_cells[k] - a_cells[k] for k in shared]
        matched = True
    else:
        n = min(len(a), len(b))
        diffs = [b[i] - a[i] for i in range(n)]
        matched = False
    n = len(diffs)
    if n < 20:
        return None
    rng = random.Random(seed)
    point = sum(diffs) / n
    draw = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    lo, hi = draw[int(0.025 * draws)], draw[int(0.975 * draws) - 1]
    return {'mean_difference': point, 'ci95': [lo, hi],
            'resolved': (lo > 0) == (hi > 0), 'cells_compared': n, 'cell_matched': matched,
            'note': ('같은 칸끼리 짝지어 뺀 차이다. 사람 간 분산이 빠져 구간이 좁다.'
                     if matched else
                     '칸을 짝짓지 못해 분포끼리 견줬다. 사람 간 분산이 구간에 그대로 남는다.')}


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
    # 첫 번째 런을 대조군으로 보고 나머지를 하나씩 견준다. v21 은 후보가 셋이라
    # 둘씩만 비교하면 한 번에 판정할 수 없다.
    if len(order) >= 2:
        base = order[0]
        doc['vs_baseline'] = {'baseline': base, 'comparisons': {
            label: paired_bootstrap(results[base]['per_cell'], results[label]['per_cell'],
                                    a_cells=results[base].get('by_cell'),
                                    b_cells=results[label].get('by_cell'))
            for label in order[1:]}}

    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(doc, ensure_ascii=False, indent=1))
    print('wrote', args.out)
    for label in order:
        d = doc['runs'][label]
        print('  %-12s 종류 %2d/%d · 칸당 %.3f · 집밖 있는 칸 %d/%d'
              % (label, d['distinct_kinds_used'], d['kinds_available'],
                 d['mean_kinds_per_cell'], d['cells_with_any_outside'], d['cells']))
        print('               한 번도 안 쓴 것: %s' % ', '.join(d['never_used']))
    vb = doc.get('vs_baseline')
    if vb:
        print()
        print('  대조군 %s 대비' % vb['baseline'])
        for label, r in vb['comparisons'].items():
            if r is None:
                print('    %-6s (칸이 모자라 견줄 수 없다)' % label)
                continue
            print('    %-6s %+.4f  [%+.4f, %+.4f]  칸 %d %s %s'
                  % (label, r['mean_difference'], r['ci95'][0], r['ci95'][1],
                     r['cells_compared'], '짝지음' if r.get('cell_matched') else '분포비교',
                     '· 갈림' if r['resolved'] else '· 0을 지난다'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
