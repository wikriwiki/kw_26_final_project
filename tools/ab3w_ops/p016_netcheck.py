# 모델이 actual_spent 에 할인을 미리 뺐는지: 정책 있는 쪽 장보기 결제의 이유·메뉴 원문을 본다(읽기 전용).
import os, sys
from neo4j import GraphDatabase
d = GraphDatabase.driver(f"bolt://localhost:{sys.argv[1]}", auth=("neo4j", os.environ["NEO4J_PASSWORD_ON"]))
with d.session() as s:
    for r in s.run("""MATCH (a:Agent)-[h:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(poi:POI)
      WHERE h.day >= date('2020-07-30') AND i.produce_spent IS NOT NULL
      RETURN a.id AS aid, toString(h.day) AS d, poi.name AS n, i.actual_spent AS sp, i.produce_spent AS pr, i.discount_total AS dc,
             i.unit_price AS up, i.pay_count AS pc, i.menu AS menu, i.pick_reason AS why, i.extra_spent AS ex, i.would_buy_anyway AS wba ORDER BY d""").data():
        print(f"{r['d']} {r['aid'][-12:]} {r['n']} 결제{r['sp']} 단가{r['up']}×{r['pc']} 농{r['pr']} 할인{r['dc']} extra{r['ex']} wba{r['wba']}\n    메뉴: {r['menu']}\n    이유: {r['why']}")
d.close()
