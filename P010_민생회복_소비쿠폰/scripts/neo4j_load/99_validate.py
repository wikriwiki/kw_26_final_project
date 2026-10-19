"""Day 0 적재 무결성 검증.

체크:
1. 노드 카운트 (District/Dong/Category/POI/Agent/Memory)
2. 엣지 카운트 (HAS_DONG/IN_DONG/IN_CATEGORY/ADJACENT_TO/LIVES_AT/WORKS_AT/KNOWS/KNOWS_POI/REMEMBERS/ABOUT_POI)
3. LIVES_AT 누락 agent
4. WORKS_AT 오생성 (life_stage in {학생,은퇴,주부})
5. KNOWS_POI 무게 분포 (per agent average)
6. POI 동 매칭 정합성 (POI.dong_code = (POI)-[:IN_DONG]->(Dong).code)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session


def main():
    with driver_session() as s:
        out = {}
        # 1. 노드
        out["node_counts"] = {}
        for label in ["District", "Dong", "Category", "POI", "Agent", "Memory"]:
            n = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            out["node_counts"][label] = n

        # POI by type
        out["poi_by_type"] = {}
        for r in s.run("MATCH (p:POI) RETURN p.type AS t, count(*) AS c"):
            out["poi_by_type"][r["t"]] = r["c"]

        # 2. 엣지
        out["edge_counts"] = {}
        for rel in ["HAS_DONG", "IN_DONG", "IN_CATEGORY", "ADJACENT_TO",
                    "LIVES_AT", "WORKS_AT", "KNOWS", "KNOWS_POI",
                    "REMEMBERS", "ABOUT_POI"]:
            n = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
            out["edge_counts"][rel] = n

        # 3. LIVES_AT 누락
        out["lives_at_missing"] = s.run(
            "MATCH (a:Agent) WHERE NOT (a)-[:LIVES_AT]->() RETURN count(a) AS c"
        ).single()["c"]

        # 4. WORKS_AT 오생성 (학생·은퇴·주부)
        out["works_at_misassigned"] = s.run("""
            MATCH (a:Agent)-[:WORKS_AT]->()
            WHERE a.p_life_stage CONTAINS '학생' OR a.p_life_stage CONTAINS '은퇴' OR a.p_life_stage CONTAINS '주부'
            RETURN count(a) AS c
        """).single()["c"]

        # 5. KNOWS_POI per agent 평균
        avg = s.run("""
            MATCH (a:Agent)-[:KNOWS_POI]->(:POI)
            WITH a, count(*) AS n
            RETURN avg(n) AS avg_per_agent,
                   min(n) AS min, max(n) AS max,
                   count(a) AS agents_with_known
        """).single()
        out["knows_poi_per_agent"] = dict(avg)

        # 6. POI dong_code 정합성
        out["poi_dong_mismatch"] = s.run("""
            MATCH (p:POI)-[:IN_DONG]->(d:Dong)
            WHERE p.dong_code <> d.code
            RETURN count(p) AS c
        """).single()["c"]

        # 7. Agent 카운트 by life_stage
        out["agent_by_life_stage_top5"] = {}
        for r in s.run("""
            MATCH (a:Agent) RETURN a.p_life_stage AS s, count(*) AS c
            ORDER BY c DESC LIMIT 5
        """):
            out["agent_by_life_stage_top5"][r["s"] or ""] = r["c"]

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
