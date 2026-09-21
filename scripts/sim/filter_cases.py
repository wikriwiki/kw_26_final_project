"""Keep only the scenarios a round actually measures, so the cells go where the question is.

A round that asks about one indicator still pays for four scenarios. v27 spent 480 cells
and left DS-1 standing on twenty-one citizens per arm, because the dine-in errand went to
whoever happened to spend most on food. Dropping the scenarios the round does not ask about
buys the same compute back as denominator.

This only drops cells. It does not touch a cell's text, conditions or hash, and it records
what was dropped so the round cannot later be read as a complete on/off matrix.

    python scripts/sim/filter_cases.py --source source_meal.json --keep distancing \
        --out source_distancing.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
from pathlib import Path


def filter_cases(source, keep):
    keep = set(keep)
    present = {c['case'] for c in source['cells']}
    missing = keep - present
    if missing:
        raise ValueError('Source has no cells for: %s (has %s)'
                         % (sorted(missing), sorted(present)))
    result = dict(source)
    result['cells'] = [c for c in source['cells'] if c['case'] in keep]
    result['case_filter'] = {
        'kept': sorted(keep),
        'dropped': sorted(present - keep),
        'cells_before': len(source['cells']),
        'cells_after': len(result['cells']),
        'why': 'This round measures only the kept scenarios. The dropped cells were never '
               'planned, so no indicator outside the kept set may be read from this run.',
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--keep', required=True, nargs='+')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    source = json.loads(Path(args.source).read_text(encoding='utf-8'))
    result = filter_cases(source, args.keep)
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(result, ensure_ascii=False, indent=1))
    f = result['case_filter']
    print('wrote', args.out)
    print('  sha256', hashlib.sha256(Path(args.out).read_bytes()).hexdigest())
    print('  칸 %d -> %d · 남긴 시나리오 %s · 버린 것 %s'
          % (f['cells_before'], f['cells_after'], f['kept'], f['dropped']))
    print('  시민 %d · 팔별 칸 %s'
          % (len({c['aid'] for c in result['cells']}),
             dict(collections.Counter(c['arm'] for c in result['cells']))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
