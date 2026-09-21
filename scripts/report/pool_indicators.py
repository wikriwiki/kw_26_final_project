"""Compute each indicator once from all runs pooled, with a bootstrap interval.

The per-run scorer averaged a ratio across four runs, and the runs disagreed enough to
flip signs: P012-1 ranged from -64.3 to +280.0 for one unchanged prompt. That is what a
ratio does when its denominator is a handful of purchases - a run where the off arm
bought two items instead of five moves the percentage by hundreds of points.

Pooling first fixes the arithmetic: sum the won across every run, then divide once. The
denominator is four times larger and the estimate stops being dominated by whichever run
had the thinnest baseline.

The interval is a cell bootstrap. Cells are resampled within (case, arm) because that is
the unit the design randomises; a percentile interval then says how much of the spread
survives when the same cells could have come out differently.

    python scripts/report/pool_indicators.py --run a=src.json:resp.jsonl [...] \
        --frozen frozen_inputs.json --draws 2000 --out pooled.json
"""
from __future__ import annotations

import argparse
import io
import json
import random
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_indicators import SECTOR, indicators, is_excluded, purchases


def cell_totals(rec, home):
    """One cell's contribution to the aggregates, kept separate so it can be resampled."""
    t = defaultdict(float)
    t['cells'] = 1.0
    for it in rec['items']:
        p = it['price']
        t['total'] += p
        t['sector:' + it['sector']] += p
        t['excluded' if it['excluded'] else 'eligible'] += p
        anchor = str(it['anchor'] or '')
        if anchor.startswith('zone:'):
            z = anchor.removeprefix('zone:')
            if z == home:
                t['home_dong'] += p
            if z[:5] != home[:5]:
                t['out_district'] += p
    return t


def collect(runs, homes):
    """{(case, arm): [per-cell totals]} across every run."""
    bucket = defaultdict(list)
    for src, resp in runs:
        for rec in purchases(src, resp):
            bucket[(rec['case'], rec['arm'])].append(
                (rec['aid'], cell_totals(rec, str(homes.get(rec['aid']) or ''))))
    return bucket


def collect_blocks(runs, homes):
    """One bucket per run. A run is a seed block, and blocks vary more than cells do.

    Resampling cells alone gave the same prompt "undecided" in one block and
    "confidently wrong" in the next (v25's DS-1 across seeds 63001-4 and 64001-4).
    That spread is real and a cell-only bootstrap cannot see it.
    """
    return [collect([r], homes) for r in runs]


def summed(bucket, pick=None):
    """Aggregate to the shape `indicators` expects. `pick` selects indices per key."""
    out = {}
    for key, cells in bucket.items():
        idx = range(len(cells)) if pick is None else pick[key]
        acc = defaultdict(float)
        for i in idx:
            for k, v in cells[i][1].items():
                acc[k] += v
        out[key] = acc
    return out


def merge(blocks, chosen):
    """Concatenate the chosen blocks into one bucket."""
    out = defaultdict(list)
    for i in chosen:
        for k, v in blocks[i].items():
            out[k].extend(v)
    return out


def percentiles(keep):
    out = {}
    for name, vals in keep.items():
        vals.sort()
        n = len(vals)
        if n < 20:
            continue
        out[name] = {'lo': vals[int(0.025 * n)], 'hi': vals[int(0.975 * n) - 1],
                     'draws': n,
                     'sign_stable': (vals[int(0.025 * n)] > 0) == (vals[int(0.975 * n) - 1] > 0)}
    return out


def cluster_bootstrap(bucket, draws, seed=20260921):
    """Resample citizens, keeping every cell each drawn citizen contributed.

    v18 ran a single replicate, so the two-stage bootstrap degenerates to the cell
    bootstrap whose intervals v7 showed were too narrow. The citizen is the next
    honest unit: the cohort is drawn by person, one person's cells move together,
    and resampling them as a bundle keeps that dependence instead of assuming it away.
    """
    rng = random.Random(seed)
    by_aid = defaultdict(lambda: defaultdict(list))
    for key, cells in bucket.items():
        for aid, totals in cells:
            by_aid[aid][key].append(totals)
    aids = sorted(by_aid)
    keep = defaultdict(list)
    for _ in range(draws):
        drawn = [aids[rng.randrange(len(aids))] for _ in aids]
        acc = {}
        for aid in drawn:
            for key, rows in by_aid[aid].items():
                bag = acc.setdefault(key, defaultdict(float))
                for totals in rows:
                    for k, v in totals.items():
                        bag[k] += v
        for name, val in indicators(acc).items():
            keep[name].append(val)
    return percentiles(keep), len(aids)


def bootstrap(bucket, draws, seed=20260920, blocks=None):
    """Two stages when blocks are given: resample seed blocks, then cells inside them.

    With one block this degenerates to the cell bootstrap, which is what the earlier
    runs used and why their intervals were too narrow.
    """
    rng = random.Random(seed)
    keep = defaultdict(list)
    for _ in range(draws):
        if blocks and len(blocks) > 1:
            chosen = [rng.randrange(len(blocks)) for _ in blocks]
            bucket = merge(blocks, chosen)
        pick = {k: [rng.randrange(len(v)) for _ in v] for k, v in bucket.items()}
        for name, val in indicators(summed(bucket, pick)).items():
            keep[name].append(val)
    return percentiles(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True,
                    metavar='label=source.json:responses.jsonl')
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--draws', type=int, default=2000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cluster', action='store_true',
                    help='resample citizens as bundles; use when there is one replicate')
    args = ap.parse_args()

    personas = json.loads(Path(args.frozen).read_text(encoding='utf-8'))['personas']
    homes = {p['id']: p.get('home_dong_code') for p in personas}

    runs = []
    for item in args.run:
        _label, paths = item.split('=', 1)
        src, resp = paths.split(':', 1)
        runs.append((src, resp))

    bucket = collect(runs, homes)
    blocks = collect_blocks(runs, homes) if len(runs) > 1 else None
    point = indicators(summed(bucket))
    if args.cluster:
        ci, n_clusters = cluster_bootstrap(bucket, args.draws)
        method = ('Sum every run, then take the ratio once. The interval resamples '
                  'citizens as bundles, because one replicate leaves the seed-block '
                  'bootstrap nothing to resample and a cell-only interval is too narrow.')
    else:
        ci, n_clusters = bootstrap(bucket, args.draws, blocks=blocks), None
        method = ('Sum every run, then take the ratio once. Interval resamples seed blocks '
                  'first, then cells inside them, because blocks vary more than cells.')
    cells = {f'{c}|{a}': len(v) for (c, a), v in bucket.items()}

    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(json.dumps({
        'method': method,
        'resample_unit': 'citizen' if args.cluster else 'seed block then cell',
        'clusters': n_clusters,
        'blocks': len(runs),
        'runs': len(runs), 'cells_per_case_arm': cells, 'draws': args.draws,
        'pooled': {k: {'mean': v,
                       'lo': ci.get(k, {}).get('lo'), 'hi': ci.get(k, {}).get('hi'),
                       'sign_stable': ci.get(k, {}).get('sign_stable'),
                       'n_runs': len(runs), 'min': ci.get(k, {}).get('lo'),
                       'max': ci.get(k, {}).get('hi'), 'sd': None}
                   for k, v in point.items()},
    }, ensure_ascii=False, indent=1))

    unit = f'시민 {n_clusters}명 군집' if args.cluster else f'런 {len(runs)} 묶음'
    print('wrote', args.out, f'({unit} · 붓스트랩 {args.draws})')
    for k in sorted(point):
        c = ci.get(k, {})
        mark = '' if c.get('sign_stable') else '   ← 구간이 0을 지난다'
        print('  %-8s %+8.1f   [%+.1f, %+.1f]%s'
              % (k, point[k], c.get('lo', float('nan')), c.get('hi', float('nan')), mark))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
