#!/usr/bin/env python3
"""에이전트별 BDC 업종 구성(weekday_top_categories)을 Agent 노드에 적재.

근거: 페르소나 생성 시 BDC 동별 industry_ratio를 기반으로 만든 값인데 Agent 노드에 실리지
않아 Stage1이 보지 못했다. 그 결과 이벤트 구성이 기준선에서 벗어났다(측정: 여가가 BDC
기준선 0.8%의 8배, 교육 2배). 우리 데이터를 우리 시뮬에 되돌리는 것이며 BOK와 무관하다.
"""
import sys, json
sys.path.insert(0, "/data/exp001_repo/scripts/neo4j_load")
from _common import driver_session

rows = json.load(open("/data/exp001/agent_cats.json", encoding="utf-8"))
with driver_session() as s:
    for i in range(0, len(rows), 1000):
        s.run("UNWIND $rows AS r MATCH (a:Agent {id: r.id}) "
              "SET a.cat_ratio_wd = r.wd, a.cat_ratio_we = r.we", rows=rows[i:i+1000])
    r = s.run("MATCH (a:Agent) WHERE a.cat_ratio_wd IS NOT NULL RETURN count(a) AS n").single()
    print(f"  적재 {r['n']:,}명 / 입력 {len(rows):,}")
