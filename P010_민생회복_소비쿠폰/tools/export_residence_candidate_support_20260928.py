"""Read-only count of actual residence POI and resident-anchor support."""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(os.environ.get('PILOT_REPO_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'scripts'))
from neo4j_load._common import driver_session

QUERY="""
MATCH(d:Dong)
OPTIONAL MATCH(p:POI {type:'residence'})-[:IN_DONG]->(d)
WITH d,count(DISTINCT p)AS residence_pois,
     sum(CASE WHEN p IS NOT NULL AND toString(p.dong_code)<>toString(d.code)THEN 1 ELSE 0 END)AS poi_property_link_mismatch
OPTIONAL MATCH(a:Agent)-[:LIVES_AT]->(h:POI)-[:IN_DONG]->(d)
RETURN toString(d.code)AS code,d.name AS name,d.sigungu_name AS gu,
       residence_pois,poi_property_link_mismatch,count(DISTINCT a)AS anchored_agents,
       count(DISTINCT CASE WHEN a.personal_age>=20 THEN a END)AS adult_anchored_agents
ORDER BY code
"""


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    with driver_session()as session:
        if session.run('MATCH(p:Policy) RETURN count(p)AS n').single()['n']:
            raise ValueError('Expected completed OFF graph; query gate failed')
        rows=[dict(r)for r in session.run(QUERY)]
    assert len(rows)==427
    obj={'schema':'residence_candidate_graph_support_v1','read_only':True,'graph_modified':False,'model_calls':0,
         'captured_at':datetime.now(timezone.utc).isoformat(),'source_tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'query':QUERY,'source_scope':'Current completed P014 OFF graph and2026POIs; not historic boundary evidence','rows':rows}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'dongs':len(rows),'sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),
                      'no_residence_poi_dongs':sum(r['residence_pois']==0 for r in rows),
                      'no_adult_anchored_agent_dongs':sum(r['adult_anchored_agents']==0 for r in rows)}))


if __name__=='__main__':main()
