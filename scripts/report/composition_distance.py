"""How far is what the citizens bought from what their own profile says they buy?

The target is the citizen's own assigned category mix, which is written into the model's
input. It has nothing to do with any measured policy effect, so aiming at it is not
circular - it is internal consistency.

The distance is total variation over category shares, and shares are the point: a
candidate cannot win by buying more of the same thing. v7 showed why that matters - v30
raised the purchase rate by 2.2 SE and every extra purchase went to delivery, which made
two answer-key indicators significantly wrong.

    python scripts/report/composition_distance.py --run a=src.json:resp.jsonl [...] \
        --source action_source.json --arm off --out distance.json
"""
from __future__ import annotations

import argparse
import io
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_indicators import purchases

# Catalog activity -> the category name the citizen's own profile uses.
# Stated here rather than inferred, because it is an assumption that can be argued with.
CATEGORY = {
    'meal_dine_in': '식사', 'meal_takeaway': '식사',
    'home_delivery': '식사', 'office_delivery': '식사',
    'cafe_dine_in': '디저트', 'cafe_takeaway': '디저트', 'dessert': '디저트',
    'groceries': '마트',
    'convenience': '편의점',
    'shopping': '쇼핑', 'home_online_goods': '쇼핑',
    'hair': '미용',
    'health_goods': '건강',
    'leisure_service': '여가',
    'education_service': '교육',
    'other_service': '기타',
    'bar': '주점',
}

# Two renderer generations are in use: the older cells say '…부여한 업종별 지출 구성:'
# and the newer ones '평소 업종별 지출 구성(카드 실측):'. Requiring the colon to follow
# the word directly silently matched only the first, which made every distance on the
# newer cohort vacuously zero.
PROFILE_RE = re.compile(r'업종별 지출 구성[^:\n]*:\s*(.+)')
PART_RE = re.compile(r'([가-힣·]+)\s*(\d+)%')


def assigned_mix(source_path):
    """{aid: {category: share}} read from each citizen's own input text."""
    src = json.loads(Path(source_path).read_text(encoding='utf-8'))
    out = {}
    for cell in src['cells']:
        aid = cell['aid']
        if aid in out:
            continue
        m = PROFILE_RE.search(cell['user'])
        if not m:
            continue
        parts = {}
        for name, pct in PART_RE.findall(m.group(1).split('/')[0]):
            parts[name] = parts.get(name, 0) + int(pct)
        total = sum(parts.values())
        if total:
            out[aid] = {k: v / total for k, v in parts.items()}
    return out


def bought_mix(runs, arm):
    """{aid: {category: share}} plus the raw counts, from what was actually purchased."""
    won = defaultdict(lambda: defaultdict(float))
    items = defaultdict(int)
    cells = defaultdict(int)
    for src, resp in runs:
        for rec in purchases(src, resp):
            if rec['arm'] != arm:
                continue
            cells[rec['aid']] += 1
            for it in rec['items']:
                cat = CATEGORY.get(it['activity_id'])
                if cat is None:
                    continue
                won[rec['aid']][cat] += it['price']
                items[rec['aid']] += 1
    mix = {}
    for aid, d in won.items():
        total = sum(d.values())
        if total:
            mix[aid] = {k: v / total for k, v in d.items()}
    return mix, dict(items), dict(cells)


def total_variation(a, b):
    """0 = identical mixes, 1 = no overlap. Shares only, so volume cannot win."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def pooled(mixes, weights=None):
    """Aggregate several per-citizen mixes into one, weighting by spend if given."""
    acc = defaultdict(float)
    for aid, mix in mixes.items():
        w = 1.0 if weights is None else weights.get(aid, 0.0)
        for k, v in mix.items():
            acc[k] += v * w
    total = sum(acc.values())
    return {k: v / total for k, v in acc.items()} if total else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True,
                    metavar='label=source.json:responses.jsonl')
    ap.add_argument('--source', required=True, help='action_source.json with the profiles')
    ap.add_argument('--arm', default='off')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    runs = []
    for item in args.run:
        _label, paths = item.split('=', 1)
        src, resp = paths.split(':', 1)
        runs.append((src, resp))

    want = assigned_mix(args.source)
    got, items, cells = bought_mix(runs, args.arm)

    per_agent = {}
    for aid in sorted(set(want) & set(got)):
        per_agent[aid] = {'distance': total_variation(got[aid], want[aid]),
                          'items': items.get(aid, 0), 'cells': cells.get(aid, 0)}

    want_pool, got_pool = pooled(want), pooled(got)
    result = {
        'arm': args.arm,
        'category_map': CATEGORY,
        'assigned_pooled': want_pool,
        'bought_pooled': got_pool,
        'distance_pooled': total_variation(got_pool, want_pool),
        'distance_per_agent_mean': (sum(v['distance'] for v in per_agent.values()) / len(per_agent)
                                    if per_agent else None),
        'agents_with_any_purchase': len(per_agent),
        'items_total': sum(items.values()),
        'cells_total': sum(cells.values()),
        'items_per_cell': (sum(items.values()) / sum(cells.values())) if cells else None,
        'never_bought': sorted(set(want_pool) - set(got_pool)),
        'per_agent': per_agent,
        'note': 'Shares only. Buying more of the same category does not reduce the distance.',
    }
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(result, ensure_ascii=False, indent=1))
    print('wrote', args.out)
    print('  구성 거리 (합산)        %.4f' % result['distance_pooled'])
    print('  구성 거리 (사람별 평균)  %s'
          % ('%.4f' % result['distance_per_agent_mean'] if per_agent else '—'))
    print('  칸당 구매 건수          %.3f  (%d건 / %d칸)'
          % (result['items_per_cell'], result['items_total'], result['cells_total']))
    print('  한 번도 안 산 업종       %s' % (', '.join(result['never_bought']) or '없음'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
