"""3주 A/B 실행기: 복제한 두 그래프가 같은지 보는 지문 — 라벨·관계 수와 명부 사람들의 상태 요약.

    NEO4J_URI=... python tools/ab3w_graph_fingerprint.py <roster.json>
"""
import hashlib
import json
import sys

from scripts.neo4j_load._common import driver_session

ids = json.load(open(sys.argv[1], encoding="utf-8"))
with driver_session() as s:
    labels = {r["l"]: r["n"] for r in s.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS n")}
    rels = {r["t"]: r["n"] for r in s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n")}
    agents = [dict(r) for r in s.run(
        "MATCH (a:Agent) WHERE a.id IN $ids "
        "OPTIONAL MATCH (a)-[h:HAS_STATE]->(st:State) "
        "WITH a, count(st) AS states, max(h.day) AS last_day "
        "OPTIONAL MATCH (a)-[h2:HAS_STATE]->(cur:State) WHERE h2.day = last_day "
        "RETURN a.id AS id, states, toString(last_day) AS last_day, cur.balance AS balance, "
        "cur.month_spent AS month_spent ORDER BY id", ids=ids)]
digest = hashlib.sha256(json.dumps(agents, default=str, sort_keys=True).encode()).hexdigest()
print(json.dumps({"labels": labels, "rels": rels, "agents": len(agents), "agents_sha256": digest}, sort_keys=True))
