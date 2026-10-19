"""Code-keyed commerce centroids from a read-only Dong export, never name joins."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from mobility import haversine_km


def build(rows, source_sha256):
    centroids = {}; skipped = []; seen = set(); groups = defaultdict(list)
    for row in rows:
        code = row['code']
        if not isinstance(code, str) or not re.fullmatch(r'11\d{6}', code) or code in seen:
            raise ValueError('Invalid or duplicate Seoul administrative code')
        seen.add(code)
        if row['lon'] is None or row['lat'] is None:
            skipped.append(code); continue
        lon, lat = row['lon'], row['lat']
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in [lon, lat]):
            raise ValueError('Invalid coordinate')
        if not 125 < lon < 129 or not 36 < lat < 39: raise ValueError('Coordinate outside plausible Seoul region')
        centroids[code] = [lon, lat]; groups[code[:5]].append([lon, lat])
    if not centroids: raise ValueError('No code-keyed centroids')
    fallback = {gu: [sum(p[i] for p in coords)/len(coords) for i in range(2)] for gu, coords in groups.items()}
    return {'_meta': {'source': 'Read-only Dong.code/Dong.lon/Dong.lat export. Original loader computes code-keyed commerce-coordinate means; not administrative boundary centroids.',
                      'source_sha256': source_sha256, 'n_dong_centroids': len(centroids), 'n_gu_fallback': len(fallback),
                      'missing_coordinate_codes': skipped, 'name_based_join': False},
            'centroids': centroids, 'gu_fallback': fallback}


def compare(old, new):
    changes = [{'code': code, 'old': coords, 'new': new['centroids'][code],
                'shift_km': haversine_km(*coords, *new['centroids'][code])}
               for code, coords in old['centroids'].items() if code in new['centroids']]
    changes.sort(key=lambda r: (-r['shift_km'], r['code']))
    return {'common_codes': len(changes), 'over_1km': sum(r['shift_km'] > 1 for r in changes),
            'over_3km': sum(r['shift_km'] > 3 for r in changes), 'changes': changes,
            'interpretation': 'Not every shift is a mapping error; sampling different POIs also shifts means. Bare-name collisions in the old builder are independently confirmed in source code.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--old'); ap.add_argument('--audit')
    args = ap.parse_args(); raw = Path(args.source).read_bytes(); result = build(json.loads(raw), hashlib.sha256(raw).hexdigest())
    target = Path(args.out)
    if target.exists(): raise ValueError('Refusing overwrite')
    target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if args.old:
        out = Path(args.audit)
        if out.exists(): raise ValueError('Refusing audit overwrite')
        out.write_text(json.dumps(compare(json.loads(Path(args.old).read_bytes()), result), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['_meta'], ensure_ascii=False))
