"""Rebuild the existing descriptive hub signature from code-keyed POI counts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from build_hub_signature import UBIQUITOUS, MIN_POI, MIN_DEVIATION


def build(rows, hubs):
    by_code = defaultdict(Counter); city = Counter(); seen = set()
    for row in rows:
        code, l1, count = row['code'], row['l1'], row['n']
        if not isinstance(code, str) or len(code) != 8 or not code.isdigit() or not isinstance(l1, str) or not l1:
            raise ValueError('Invalid group')
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or (code, l1) in seen:
            raise ValueError('Invalid/duplicate group count')
        seen.add((code, l1)); by_code[code][l1] = count; city[l1] += count
    if not city: raise ValueError('No category counts')
    shares = {k: v / sum(city.values()) for k, v in city.items()}; output = {}
    for hub in hubs:
        if not hub.get('is_top_hub'): continue
        code = hub['code']; counts = by_code[code]; n = sum(counts.values())
        share = {k: v/n for k, v in counts.items()} if n else {}
        best, deviation = 'general', MIN_DEVIATION
        if n >= MIN_POI:
            for l1 in sorted(share):
                if l1 not in UBIQUITOUS and share[l1] - shares[l1] > deviation:
                    best, deviation = l1, share[l1] - shares[l1]
        output[code] = {'name': hub['name'], 'signature': best, 'n_poi': n,
                        'l1_share': {k: round(v, 3) for k, v in sorted(share.items(), key=lambda pair: (-pair[1], pair[0]))[:5]}}
    return {'_meta': {'source': 'Read-only code-keyed POI category counts, full current database. Existing descriptive threshold retained.',
                     'name_based_join': False, 'n_hubs': len(output), 'city_share': shares,
                     'limitation': 'Commerce category presence is not a person-specific preference or historical shop inventory.'}, 'hubs': output}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--catalog', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); raw = Path(args.source).read_bytes(); catalog = Path(args.catalog).read_bytes()
    result = build(json.loads(raw), json.loads(catalog)['hubs'])
    result['_meta'].update(source_sha256=hashlib.sha256(raw).hexdigest(), catalog_sha256=hashlib.sha256(catalog).hexdigest())
    out = Path(args.out)
    if out.exists(): raise ValueError('Refusing overwrite')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
