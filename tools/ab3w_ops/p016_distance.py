# 장보기 가게까지의 거리(읽기 전용): 그 장보기 장면의 동(집·직장이면 그 자리, 아니면 동 중심)에서 고른 가게까지.
#   python p016_distance.py <bolt> <첫날> <끝날>
import os, sys, collections, statistics as st
from neo4j import GraphDatabase
bolt, d0, d1 = sys.argv[1:4]
Q = """
MATCH (a:Agent)-[h:HAS_PLAN]->(pl:Plan)-[i:INCLUDES]->(poi:POI)-[:IN_CATEGORY]->(c:Category)
WHERE h.day >= date($d0) AND h.day <= date($d1) AND c.name IN ['슈퍼마켓','식료품','청과','정육'] AND coalesce(i.actual_spent,0) > 0
  AND i.anchor STARTS WITH 'zone:'
WITH a, i, poi, substring(i.anchor, 5) AS zone
MATCH (dg:Dong {code: zone})
OPTIONAL MATCH (a)-[:LIVES_AT|WORKS_AT]->(anc:POI) WHERE anc.dong_code = zone
WITH i, poi, zone, dg, head(collect(anc)) AS anc
RETURN poi.name AS name, poi.mart_format AS fmt, poi.dong_code = zone AS same_dong,
       point.distance(point({longitude: poi.lon, latitude: poi.lat}),
                      point({longitude: coalesce(anc.lon, dg.lon), latitude: coalesce(anc.lat, dg.lat)})) / 1000.0 AS km
"""
d = GraphDatabase.driver(f"bolt://localhost:{bolt}", auth=("neo4j", os.environ["NEO4J_PASSWORD_ON"]))
with d.session() as s:
    rows = s.run(Q, d0=d0, d1=d1).data()
d.close()
km = sorted(r["km"] for r in rows if r["km"] is not None)
far = [r for r in rows if (r["km"] or 0) > 3.0]
print(f"{bolt} {d0}~{d1}: 장보기 {len(rows)}건 · 같은 동 가게 {sum(r['same_dong'] for r in rows)} · 대형 형태(대형마트·SSM) {sum(1 for r in rows if r['fmt'])}")
if km:
    print(f"  장면 동에서 거리 km: 중앙 {st.median(km):.2f} · 90% {km[int(.9*len(km))-1]:.2f} · 최대 {km[-1]:.2f} · 3km 넘음 {len(far)}")
for r in far[:10]:
    print("   3km 넘음:", r)
