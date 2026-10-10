# P016 명부 2,000명의 참여 체인 접근성(읽기 전용). python p016_reach.py <bolt> <roster>
import json, os, re, sys, collections
from neo4j import GraphDatabase
bolt, roster = sys.argv[1], sys.argv[2]  # Cypher =~ 는 이름 전체가 맞아야 한다 — 패턴 끝에 .* 를 붙인다
ids = json.load(open(roster))
CH = r"^(이마트(?!24| ?에브리데이)|롯데마트|롯데쇼핑롯데마트|.*하나로(마트|클럽)|GS ?더 ?프레시|지에스더프레시|GS ?수퍼|지에스리테일 ?GS ?수퍼).*"
Q = """
UNWIND $ids AS aid
MATCH (a:Agent {id: aid})-[:LIVES_AT]->(h:POI)
OPTIONAL MATCH (p:POI)-[:IN_CATEGORY]->(c:Category) WHERE c.name IN ['슈퍼마켓','식료품','청과','정육'] AND p.name =~ $ch
  AND point.distance(point({longitude:p.lon,latitude:p.lat}), point({longitude:h.lon,latitude:h.lat})) <= 3000
WITH a, h, collect(DISTINCT p) AS ps
OPTIONAL MATCH (m:POI {mart_format:'hypermarket'})
  WHERE point.distance(point({longitude:m.lon,latitude:m.lat}), point({longitude:h.lon,latitude:h.lat})) <= 3000
WITH a, h, ps, collect(DISTINCT m) AS ms
RETURN a.id AS aid, h.dong_code AS dong, size(ps) AS chain3,
       size([x IN ps WHERE x.dong_code = h.dong_code]) AS chain_dong,
       size(ms) AS hyper3, size([x IN ms WHERE x.name =~ $ch]) AS hyper_chain3
"""
drv = GraphDatabase.driver(f"bolt://localhost:{bolt}", auth=("neo4j", os.environ["NEO4J_PASSWORD_ON"]))
with drv.session() as s:
    rows = s.run(Q, ids=ids, ch=CH).data()
drv.close()
n = len(rows)
def pct(k, f): 
    c = sum(1 for r in rows if f(r[k])); return f"{c}/{n} ({100*c/n:.1f}%)"
print(f"명부 {len(ids)} · 집 있는 사람 {n}")
print("집 3km 안 참여 체인 매장(모든 형태) ≥1:", pct("chain3", lambda v: v >= 1))
print("집 동 안 참여 체인 매장 ≥1:", pct("chain_dong", lambda v: v >= 1))
print("집 3km 안 대형마트(참여·비참여) ≥1:", pct("hyper3", lambda v: v >= 1))
print("집 3km 안 참여 대형마트 ≥1 (가까운 대형마트 후보 2곳에 들 수 있는):", pct("hyper_chain3", lambda v: v >= 1))
print("3km 안 참여 체인 매장 수 분포:", sorted(collections.Counter(min(r['chain3'], 10) for r in rows).items()))
