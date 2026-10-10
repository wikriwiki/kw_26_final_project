# 브랜드 중립 '가까운 대형 형태 매장 N곳' 규칙별 참여 체인 노출률(읽기 전용)
import json, os, sys, collections
from neo4j import GraphDatabase
bolt, roster = sys.argv[1], sys.argv[2]
ids = json.load(open(roster))
CH = r"^(이마트(?!24| ?에브리데이)|롯데마트|롯데쇼핑롯데마트|.*하나로(마트|클럽)|GS ?더 ?프레시|지에스더프레시|GS ?수퍼|지에스리테일 ?GS ?수퍼)"
drv = GraphDatabase.driver(f"bolt://localhost:{bolt}", auth=("neo4j", os.environ["NEO4J_PASSWORD_ON"]))
with drv.session() as s:
    fm = s.run("MATCH (p:POI) WHERE p.mart_format IS NOT NULL RETURN p.mart_format AS f, p.chain AS c, count(*) AS n ORDER BY f, n DESC").data()
    for r in fm: print("  형태", r["f"], r["c"], r["n"])
    rows = s.run("""
    UNWIND $ids AS aid
    MATCH (a:Agent {id: aid})-[:LIVES_AT|WORKS_AT]->(h:POI) WHERE h.lon IS NOT NULL
    MATCH (m:POI) WHERE m.mart_format IN ['hypermarket','ssm'] AND m.type = 'commerce'
    WITH aid, m, min(point.distance(point({longitude:m.lon,latitude:m.lat}), point({longitude:h.lon,latitude:h.lat}))) AS d
    WHERE d <= 3000
    WITH aid, m, d ORDER BY d
    WITH aid, collect({name: m.name, fmt: m.mart_format, d: d})[0..5] AS near
    RETURN aid, near
    """, ids=ids).data()
drv.close()
import re
ch = re.compile(CH)
n = len(ids)
print(f"명부 {n} · 3km 안 대형 형태 매장이 하나라도 있는 사람 {len(rows)}")
for N in (2, 3, 4, 5):
    k = sum(1 for r in rows if any(ch.match(x["name"] or "") for x in r["near"][:N]))
    hyper_only = sum(1 for r in rows if any(ch.match(x["name"] or "") for x in [y for y in r["near"] if y["fmt"] == "hypermarket"][:N]))
    print(f"N={N}: 참여 체인이 후보에 하나라도 {k}/{n} ({100*k/n:.1f}%) · 대형마트만 N곳이면 {hyper_only}/{n} ({100*hyper_only/n:.1f}%)")
dist = [r["near"][0]["d"] for r in rows]
dist.sort(); print("가장 가까운 대형 형태 매장 거리 중앙값(m)", round(dist[len(dist)//2]))
