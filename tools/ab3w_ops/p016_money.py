# 한 사람의 하루 회계 추적(읽기 전용): 결제·할인·지갑 차감·상태 잔액·오늘 지출. python p016_money.py <bolt> <aid 끝부분> <날 ...>
import os, sys, json
from neo4j import GraphDatabase
bolt, tail, days = sys.argv[1], sys.argv[2], sys.argv[3:]
d = GraphDatabase.driver(f"bolt://localhost:{bolt}", auth=("neo4j", os.environ["NEO4J_PASSWORD_ON"]))
with d.session() as s:
    aid = s.run("MATCH (a:Agent) WHERE a.id ENDS WITH $t RETURN a.id AS id", t=tail).single()["id"]
    for day in days:
        st = s.run("MATCH (a:Agent {id:$a})-[h:HAS_STATE]->(x:State) WHERE toString(h.day)=$d RETURN properties(x) AS p", a=aid, d=day).single()
        print("== 상태", day, {k: v for k, v in (st["p"] if st else {}).items() if k in ("balance", "today_spent", "month_spent", "instant_discount_today", "policy_used", "policy_balance", "grant_balance", "daily_income", "income_today")})
        for r in s.run("""MATCH (a:Agent {id:$a})-[h:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(poi:POI) WHERE toString(h.day)=$d AND coalesce(i.actual_spent,0)>0
                          RETURN poi.name AS n, i.actual_spent AS sp, i.discount_total AS dc, i.instant_discount AS idc, i.spent_from_policy AS sfp, i.policy_rebate AS rb, i.produce_spent AS pr ORDER BY i.order""", a=aid, d=day):
            print("   ", dict(r))
    print("상태 키:", sorted((s.run("MATCH (a:Agent {id:$a})-[:HAS_STATE]->(x:State) RETURN properties(x) AS p LIMIT 1", a=aid).single()["p"]).keys()))
d.close()
