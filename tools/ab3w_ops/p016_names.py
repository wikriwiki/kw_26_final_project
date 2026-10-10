# 참여 매장 이름 규칙 점검(읽기 전용): 장보기 업종 가게 이름 중 규칙에 걸리는 것 / 체인 이름 비슷한데 안 걸리는 것.
import os, re, json, collections
from neo4j import GraphDatabase
pol = json.load(open("/data/repo_ab3w_20261011/data/experiments/P016_ab3w_policy_20261011.json"))
rx = re.compile(pol["eligibility"]["include"]["name_regex"]); subs = set(pol["eligibility"]["include"]["subs"])
chains = {k: re.compile(v) for k, v in pol["sectors"]["농축산물"]["chains"].items()}
near = re.compile(r"이마트|emart|롯데|lotte|하나로|농협|GS|지에스|더프레시|후레쉬|수퍼|홈플러스|트레이더스|노브랜드|에브리데이", re.I)
d = GraphDatabase.driver("bolt://localhost:7694", auth=("neo4j", os.environ["NEO4J_PASSWORD_ON"]))
with d.session() as s:
    rows = s.run("MATCH (p:POI {type:'commerce'})-[:IN_CATEGORY]->(c:Category) WHERE c.name IN $subs RETURN p.name AS n, c.name AS c, p.mart_format AS f, p.chain AS ch", subs=list(subs)).data()
    other = s.run("MATCH (p:POI {type:'commerce'})-[:IN_CATEGORY]->(c:Category) WHERE NOT c.name IN $subs AND p.name =~ '.*(이마트|롯데마트|하나로|더프레시|GS수퍼).*' RETURN c.name AS c, count(*) AS n ORDER BY n DESC LIMIT 15", subs=list(subs)).data()
d.close()
inc = collections.Counter(); exc = collections.Counter(); multi = []
for r in rows:
    n = r["n"] or ""
    if rx.match(n):
        hit = [k for k, c in chains.items() if c.search(n) or c.match(n)]
        m = [k for k, c in chains.items() if (c.match(n) if c.pattern.startswith("^") else c.search(n))]
        if len(m) != 1: multi.append((n, m))
        inc[(m[0] if m else "체인 없음", n)] += 1
    elif near.search(n):
        exc[n] += 1
print("규칙에 걸린 가게(체인별 이름 표본):")
by = collections.defaultdict(list)
for (ch, n), k in inc.items(): by[ch].append(n)
for ch, ns in by.items(): print(f"  {ch}: {len(ns)}곳 — {sorted(ns)[:40]}")
print("체인 판정이 0개 또는 2개 이상:", multi[:20])
print(f"체인 비슷한데 안 걸린 이름 {len(exc)}종:", sorted(exc)[:120])
print("장보기 업종이 아닌데 체인 이름인 가게(할인 대상 아님):", other)
