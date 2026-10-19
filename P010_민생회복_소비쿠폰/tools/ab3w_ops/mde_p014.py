# P014 지표의 검출력: 같은 사람 정책 있음−없음 차이의 분산(거리두기 11/24~26, P013 5/11~15)으로 7일 최소 검출 차이를 잰다.
import os, json, glob, collections, statistics as st, math, sys
from neo4j import GraphDatabase
PW = os.environ["NEO4J_PASSWORD_ON"]
G = {"슈퍼마켓": {"슈퍼마켓"}, "종합소매": {"종합소매"}, "식품전문": {"식료품","정육","청과","수산","음료소매"}, "식사": None, "총액": "ALL"}
def pull(bolt, days):
    d = GraphDatabase.driver(f"bolt://localhost:{bolt}", auth=("neo4j", PW))
    with d.session() as s:
        rows = s.run("""MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan)-[r:INCLUDES]->(x:POI) WHERE toString(p.day) IN $days AND r.actual_spent > 0
            RETURN a.id AS aid, toString(p.day) AS d, r.category AS cat, coalesce(head([(x)-[:IN_CATEGORY]->(oc:Category) | oc.name]), r.sub_category) AS sub, r.actual_spent AS amt""", days=days).data()
    d.close(); return rows
for run, bon, boff, days in [("distancing_main",7691,7692,["2020-11-24","2020-11-25","2020-11-26"]), ("p013_main",7689,7690,["2020-05-11","2020-05-12","2020-05-13","2020-05-14","2020-05-15"])]:
    ids = sorted({json.loads(l)["aid"] for l in open(glob.glob(f"/data/ab3w/{run}/on/metrics/day_{days[0]}.jsonl")[0])})
    tot = {}
    for arm, b in (("on",bon),("off",boff)):
        t = collections.defaultdict(float)
        for r in pull(b, days):
            for g, subs in G.items():
                if subs == "ALL" or (subs is None and r["cat"] == "식사") or (isinstance(subs,set) and r["sub"] in subs):
                    t[(g, r["aid"])] += r["amt"]
        tot[arm] = t
    print(f"== {run} {len(days)}일 · {len(ids)}명")
    for g in G:
        diff = [tot["on"][(g,a)] - tot["off"][(g,a)] for a in ids]
        base = st.mean([tot["off"][(g,a)] for a in ids])
        sd7 = st.pstdev(diff) * math.sqrt(7/len(days))   # 날끼리 독립 가정
        base7 = base * 7/len(days)
        mde = 2.8 * sd7 / math.sqrt(len(ids))
        print(f"  {g}: 7일 1인 기준 {base7:,.0f}원 · 차이 표준편차 {sd7:,.0f} · 80% 검출 최소 차이 {100*mde/max(base7,1):.1f}% · 지금 있음−없음 {100*st.mean(diff)/max(base,1):+.1f}%")
