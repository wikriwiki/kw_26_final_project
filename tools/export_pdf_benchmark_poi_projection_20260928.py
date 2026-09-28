"""Read-only projection of existing POI taxonomy/geography for receipt joins.

No policy, state, plan, relationship or serving process is modified.  No
credentials, citizen profiles, shop names or addresses are emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("PILOT_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))
from neo4j_load._common import driver_session

QUERY = """
MATCH (p:POI)
OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
WITH p, collect(DISTINCT {sub:c.name, l1:c.parent}) AS categories
OPTIONAL MATCH (p)-[:IN_DONG]->(d:Dong)
RETURN p.id AS id, categories,
       toString(coalesce(p.dong_code, d.code)) AS dong_code,
       d.name AS dong_name, d.sigungu_name AS district_name,
       p.lon AS lon, p.lat AS lat
ORDER BY id
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with driver_session() as session:
        policies = [dict(r) for r in session.run("MATCH(p:Policy) RETURN p.id AS id")]
        if policies:
            raise ValueError("Expected completed P014 OFF graph without Policy nodes")
        rows = {}
        ambiguous = []
        missing_taxonomy = 0
        missing_geo = 0
        for record in session.run(QUERY):
            r = dict(record)
            id_ = r.pop("id")
            if not id_ or id_ in rows:
                raise ValueError("POI IDs or IN_DONG mapping ambiguous")
            cats = [c for c in r.pop("categories") if c.get("sub") is not None]
            if len(cats) > 1:
                ambiguous.append(id_)
            r["categories"] = cats
            r["sub"] = cats[0]["sub"] if len(cats) == 1 else None
            r["l1"] = cats[0]["l1"] if len(cats) == 1 else None
            missing_taxonomy += len(cats) != 1
            missing_geo += not (r["dong_code"] and r["dong_name"] and r["district_name"])
            rows[id_] = r
    result = {"schema": "pdf_benchmark_poi_projection_v1",
              "read_only": True, "graph_arm": "P014_OFF_completed",
              "captured_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "policy_nodes": policies, "poi_count": len(rows),
              "ambiguous_taxonomy_poi_ids": ambiguous,
              "missing_unique_taxonomy_pois": missing_taxonomy,
              "missing_geography_pois": missing_geo,
              "interpretation": "Existing native model POI projection, not historical2020 merchant census; no data correction or graph write",
              "pois": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "bytes": args.out.stat().st_size,
                      "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
