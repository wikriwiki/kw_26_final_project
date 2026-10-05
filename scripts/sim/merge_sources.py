"""Join two cohorts built by the same pipeline, so a thin indicator gets more people.

The sixty- and hundred-twenty-person cohorts share exactly one citizen, so together they
are a hundred and seventy-nine. For an indicator whose interval is set by how many people
contributed - and dine-in has been contributed by two, three and five - more citizens is
worth more than more samples of the same citizens.

They may only be joined if the rules are the same. Every non-cell key is compared: the ones
that agree are carried, and the ones that differ may differ only in what describes the
particular input - the hashes of the files that went in and the counts of what came out -
which is what two runs of one pipeline over different persona slices produce. Anything else
differing stops the merge, because that would mean the cells were built under different
rules and the pooled cohort would not be one cohort.

    python scripts/sim/merge_sources.py --source a.json --source b.json --out merged.json
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import io
import json
from pathlib import Path

# Entries that describe the particular input rather than the rule applied to it: hashes of
# the files that went in, and counts of what came out. Two cohorts from one pipeline differ
# here and must not differ anywhere else.
PER_SOURCE = {'source_sha256', 'personas_source_sha256', 'profiles_sha256',
              'cells_before', 'cells_after'}


def _rules_only(value):
    """Strip per-input entries so two metadata blocks can be compared on their rules."""
    if isinstance(value, dict):
        return {k: _rules_only(v) for k, v in sorted(value.items()) if k not in PER_SOURCE}
    if isinstance(value, list):
        return [_rules_only(v) for v in value]
    return value


def merge(sources, names):
    if len(sources) < 2:
        raise ValueError('Merging needs at least two sources')
    head = sources[0]
    meta_keys = {k for s in sources for k in s if k not in ('cells', 'personas')}
    for k in sorted(meta_keys):
        shapes = {json.dumps(_rules_only(s.get(k)), ensure_ascii=False, sort_keys=True)
                  for s in sources}
        if len(shapes) > 1:
            raise ValueError('Sources disagree on %r beyond per-input hashes and counts; '
                             'they were not built under the same rules' % k)

    result = {k: copy.deepcopy(head[k]) for k in head if k not in ('cells', 'personas')}
    cells, seen, dropped = [], set(), []
    for name, s in zip(names, sources):
        for c in s['cells']:
            key = (c['aid'], c['case'], c['arm'])
            if key in seen:
                dropped.append({'aid': c['aid'], 'case': c['case'], 'arm': c['arm'],
                                'from': name})
                continue
            seen.add(key)
            cells.append(copy.deepcopy(c))
    result['cells'] = cells

    people, pseen = [], set()
    for s in sources:
        for p in s.get('personas', []):
            if p['id'] in pseen:
                continue
            pseen.add(p['id'])
            people.append(copy.deepcopy(p))
    result['personas'] = people

    result['cohort_merge'] = {
        'kind': 'pooled_cohorts',
        'sources': [{'name': n,
                     'sha256': hashlib.sha256(
                         json.dumps(s, ensure_ascii=False, sort_keys=True).encode('utf-8')
                     ).hexdigest(),
                     'cells': len(s['cells']),
                     'citizens': len({c['aid'] for c in s['cells']})}
                    for n, s in zip(names, sources)],
        'cells': len(cells),
        'citizens': len({c['aid'] for c in cells}),
        'duplicate_cells_dropped': dropped,
        'checked': 'Every non-cell key matched once per-input hashes and counts were set '
                   'aside. Rules were identical.',
        'why': 'Dine-in has been contributed by two to five citizens per round. The interval '
               'is set by how many people contributed, so citizens were pooled rather than '
               'samples of the same citizens multiplied.',
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', action='append', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    sources = [json.loads(Path(p).read_text(encoding='utf-8')) for p in args.source]
    result = merge(sources, args.source)
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(result, ensure_ascii=False, indent=1))
    m = result['cohort_merge']
    print('wrote', args.out)
    print('  sha256', hashlib.sha256(Path(args.out).read_bytes()).hexdigest())
    print('  칸 %d · 시민 %d · 겹쳐서 버린 칸 %d'
          % (m['cells'], m['citizens'], len(m['duplicate_cells_dropped'])))
    print('  팔별', dict(collections.Counter(c['arm'] for c in result['cells'])))
    for s in m['sources']:
        print('   - %s: 칸 %d · 시민 %d' % (s['name'], s['cells'], s['citizens']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
