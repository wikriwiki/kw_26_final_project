"""Read only code/name/anchor projection; no model calls or graph mutations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get('PILOT_REPO_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'scripts'))
from neo4j_load._common import driver_session

DONG_QUERY = """
MATCH (d:Dong)
RETURN toString(d.code) AS code, d.name AS name, d.sigungu_name AS gu
ORDER BY code
"""
AGENT_QUERY = """
MATCH (a:Agent)
OPTIONAL MATCH (a)-[:LIVES_AT]->(h:POI)
OPTIONAL MATCH (h)-[:IN_DONG]->(d:Dong)
RETURN a.id AS aid,
       a.p_gender AS sex,
       a.p_age_group AS age_band,
       a.personal_age AS age,
       a.p_income_level AS unverified_generated_income_tier,
       toString(a.residence_dong_code_raw) AS raw_residence_code,
       collect(DISTINCT h.id) AS anchor_ids,
       collect(DISTINCT toString(h.dong_code)) AS anchor_codes,
       collect(DISTINCT toString(d.code)) AS linked_codes
ORDER BY aid
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with driver_session() as session:
        policies = [dict(r) for r in session.run('MATCH(p:Policy) RETURN p.id AS id')]
        if policies:
            raise ValueError('Expected completed policy-OFF graph')
        dongs = [dict(r) for r in session.run(DONG_QUERY)]
        agents = [dict(r) for r in session.run(AGENT_QUERY)]
    assert len({r['aid'] for r in agents}) == len(agents)
    obj = {'schema': 'admin_code_graph_projection_v1', 'read_only': True,
           'graph_modified': False, 'model_calls': 0,
           'income_verified': False,
           'source_scope': 'Current completed P014 OFF graph with 2026 native POI catalog; not a historical population census',
           'captured_at': datetime.now(timezone.utc).isoformat(),
           'source_tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'queries': {'dong': DONG_QUERY, 'agent': AGENT_QUERY},
           'policy_nodes': policies, 'dongs': dongs, 'agents': agents}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(obj, ensure_ascii=False, separators=(',', ':'))+'\n', encoding='utf-8')
    print(json.dumps({'dongs': len(dongs), 'agents': len(agents),
                      'bytes': args.out.stat().st_size,
                      'sha256': hashlib.sha256(args.out.read_bytes()).hexdigest()}))


if __name__ == '__main__': main()
