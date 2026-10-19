"""Read-only P014 proxy POI catalog and floor-area support audit.

The current simulation labels are not KIPF's 165 m² supermarket strata.
This audit diagnoses proxy support only; it cannot establish source mapping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from neo4j_load._common import driver_session


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    assert isinstance(roster, list) and len(roster) == len(set(roster)) == 40
    subs = ["\uc2dd\ub8cc\ud488", "\uc288\ud37c\ub9c8\ucf13"]
    with driver_session() as session:
        policies = [dict(row) for row in session.run(
            "MATCH (p:Policy) RETURN p.id AS id")]
        assert not policies, "Expected preserved P014 OFF graph with no Policy"
        homes = [dict(row) for row in session.run("""
            MATCH (a:Agent)-[:LIVES_AT]->(h:POI)
            WHERE a.id IN $aids
            OPTIONAL MATCH (h)-[:IN_DONG]->(d:Dong)
            RETURN a.id AS aid, a.residence_gu AS district_name,
                   toString(coalesce(h.dong_code, d.code)) AS home_dong_code
            ORDER BY aid
        """, aids=roster)]
        assert len(homes) == 40 and {x["aid"] for x in homes} == set(roster)
        catalog = [dict(row) for row in session.run("""
            MATCH (p:POI)-[:IN_CATEGORY]->(c:Category)
            WHERE c.name IN $subs
            WITH DISTINCT p, c
            RETURN c.name AS sub, c.parent AS l1, p.type AS poi_type,
                   substring(toString(p.dong_code), 0, 5) AS district_code,
                   p.upjong_l3 AS upjong_l3, count(DISTINCT p) AS poi_count
            ORDER BY sub, district_code, upjong_l3, poi_type
        """, subs=subs)]
        properties = [dict(row) for row in session.run("""
            MATCH (p:POI {type:'commerce'})
            UNWIND keys(p) AS property
            RETURN property, count(*) AS poi_count
            ORDER BY property
        """)]
        graph_counts = dict(session.run("""
            MATCH (p:POI)
            RETURN count(p) AS all_poi_count,
                   sum(CASE WHEN p.type='commerce' THEN 1 ELSE 0 END) AS commerce_poi_count
        """).single())
    district_roster = Counter((x.get("home_dong_code") or "")[:5] for x in homes)
    names = defaultdict(set)
    for row in homes:
        names[(row.get("home_dong_code") or "")[:5]].add(row.get("district_name"))
    city = {sub: sum(r["poi_count"] for r in catalog if r["sub"] == sub) for sub in subs}
    coverage = []
    for district, n in sorted(district_roster.items()):
        counts = {sub: sum(r["poi_count"] for r in catalog
                          if r["sub"] == sub and r["district_code"] == district)
                  for sub in subs}
        coverage.append({"district_code": district,
                         "district_names": sorted(v for v in names[district] if v),
                         "roster_agents": n, "poi_count_by_proxy_sub": counts})
    patterns = ("area", "floor", "size", "sqm", "sqft", "\uba74\uc801", "\ud3c9\uc218")
    area_properties = [r for r in properties
                       if any(p in str(r["property"]).lower() for p in patterns)]
    support = {sub: {"citywide_proxy_pois": city[sub],
                     "roster_home_district_proxy_pois": sum(r["poi_count_by_proxy_sub"][sub]
                                                            for r in coverage),
                     "roster_agents_with_home_district_proxy_support": sum(
                         r["roster_agents"] for r in coverage
                         if r["poi_count_by_proxy_sub"][sub] > 0)}
               for sub in subs}
    out = {"audit_id": "p014_proxy_catalog_support_20260928",
           "captured_at_utc": datetime.now(timezone.utc).isoformat(),
           "read_only": True, "graph_arm": "P014_OFF_completed",
           "policy_nodes": policies, "roster_n": len(roster),
           "roster_sha256": sha(args.roster), "graph_counts": graph_counts,
           "proxy_subcategory_support": support,
           "roster_home_district_coverage": coverage,
           "catalog_aggregated_rows": catalog,
           "commerce_poi_property_counts": properties,
           "floor_area_like_properties": area_properties,
           "interpretation": [
               "Counts describe existing simulation proxy labels, not KIPF industry strata.",
               "KIPF 47121/47129 split supermarkets at 165 square metres; current 식료품 label is not the under-165-square-metre group.",
               "Absence of floor-area-like properties prevents verifying that split from this POI graph alone.",
               "No model calls, graph writes, reset, or active source edits were performed."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assert not args.out.exists(), "Refusing to overwrite audit evidence"
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "sha256": sha(args.out),
                      "support": support, "area_like_properties": area_properties},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
