"""Freeze a shared Day0 merchant catalogue without reading historical memories."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from planning_contract import CATEGORIES
from reference_preflight import inspect_references
from validate_prompt_v3 import ROOT, atomic

QUERY="""
MATCH (p:POI {type:'commerce'})-[:IN_DONG]->(d:Dong {code:$dong})
MATCH (p)-[:IN_CATEGORY]->(c:Category)
WHERE c.name=$category OR c.parent=$category
WITH p,d,collect(DISTINCT c.name) AS sub_categories
OPTIONAL MATCH (anchor:POI) WHERE anchor.id IN $anchors
WITH p,d,sub_categories,
 min(CASE WHEN p.lon IS NOT NULL AND p.lat IS NOT NULL
                AND anchor.lon IS NOT NULL AND anchor.lat IS NOT NULL
          THEN point.distance(point({longitude:p.lon,latitude:p.lat}),
                              point({longitude:anchor.lon,latitude:anchor.lat}))/1000.0
          ELSE NULL END) AS km
RETURN p.id AS poi_id,p.name AS name,d.code AS dong_code,sub_categories,
       p.upjong_l3 AS upjong_l3,km
ORDER BY km ASC,poi_id ASC LIMIT $limit
"""


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inputs',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--limit',type=int,default=8)
    ap.add_argument('--reference-dir',type=Path,default=ROOT/'output/stats')
    args=ap.parse_args(); out=Path(args.out)
    if out.exists(): raise ValueError('Refusing overwrite')
    if not 1 <= args.limit <= 50: raise ValueError('Invalid candidate limit')
    raw=Path(args.inputs).read_bytes(); inputs=json.loads(raw)
    references=inspect_references(args.reference_dir,require_code_geography=True)
    people={p['id']:p for p in inputs['personas']}
    tasks=sorted({(c['aid'],str(z),cat) for c in inputs['cells'] for z in c['zones'] for cat in CATEGORIES})
    result=[]; stats=Counter()
    from neo4j_load._common import driver_session
    with driver_session() as session:
        for aid,dong,cat in tasks:
            p=people[aid]
            anchors=sorted({str(p[k]) for k in ['home_poi_id','work_poi_id'] if p.get(k)})
            rows=[dict(row) for row in session.run(QUERY,dong=dong,category=cat,anchors=anchors,limit=args.limit)]
            assert len(rows)==len({r['poi_id'] for r in rows})
            for r in rows: r['sub_categories']=sorted(r['sub_categories'])
            stats['groups']+=1; stats['empty_groups']+=not bool(rows); stats['candidates']+=len(rows)
            stats['missing_distance']+=sum(r['km'] is None for r in rows)
            stats['missing_industry_code']+=sum(not r.get('upjong_l3') for r in rows)
            result.append({'aid':aid,'dong':dong,'category':cat,'candidates':rows})
    manifest={'prepared_at':datetime.now(timezone.utc).isoformat(),'source_inputs_sha256':hashlib.sha256(raw).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'query_sha256':hashlib.sha256(QUERY.encode()).hexdigest(),
              'limit_per_group':args.limit,'reference_inputs':references,'counts':dict(stats),
              'scope':'Shared read-only Day0 market. No historical Agent memory/KNOWS_POI/Plan/State reads. No policy markers or price outcomes. Same candidate catalogue for on/off.',
              'limitations':['POI availability is from current database, not a reconstructed historical market.',
                            'Empty categories are explicit; do not replace them with an arbitrary category or charge for a free activity.',
                            'No individual shop menu prices supplied. Separate receipt references are not per-item prices.']}
    atomic(out,{'manifest':manifest,'groups':result})
    print(json.dumps(manifest['counts'])+' sha256='+hashlib.sha256(out.read_bytes()).hexdigest())


if __name__=='__main__': main()
