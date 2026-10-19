"""KNOWS (Agent ↔ Agent) 소셜 그래프 생성.

알고리즘 (POC baseline):
- 같은 work_dong 동료: strength=0.6, agent당 최대 ~5명
- 같은 home_dong 이웃: strength=0.4, agent당 최대 ~3명
- 무방향 의미지만 Neo4j 방향 강제 → 양방향 MERGE

LIVES_AT/WORKS_AT 적재 후 실행해야 dong_code 그룹 가능.
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session, bulk_run

SEED = 42
N_COLLEAGUE = 5
N_NEIGHBOR = 3


def main():
    rng = random.Random(SEED)
    with driver_session() as s:
        # Agent → home_dong, work_dong 추출
        print("[fetch] agent dong groups ...")
        home_group = defaultdict(list)
        work_group = defaultdict(list)
        agent_ids = []
        for r in s.run("""
            MATCH (a:Agent)
            OPTIONAL MATCH (a)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(hd:Dong)
            OPTIONAL MATCH (a)-[:WORKS_AT]->(:POI)-[:IN_DONG]->(wd:Dong)
            RETURN a.id AS id, hd.code AS home, wd.code AS work
        """):
            agent_ids.append(r["id"])
            if r["home"]:
                home_group[r["home"]].append(r["id"])
            if r["work"]:
                work_group[r["work"]].append(r["id"])
        print(f"  agents: {len(agent_ids)}, home_dongs: {len(home_group)}, work_dongs: {len(work_group)}")

        # 매핑 생성
        pairs = set()  # (a, b) with a < b
        # 동료 (work_dong)
        for cd, members in work_group.items():
            if len(members) < 2:
                continue
            for a in members:
                # 같은 동 멤버 중 N_COLLEAGUE 랜덤
                others = [m for m in members if m != a]
                k = min(N_COLLEAGUE, len(others))
                if k == 0:
                    continue
                for b in rng.sample(others, k):
                    key = (a, b) if a < b else (b, a)
                    pairs.add((key, 0.6, "colleague"))
        # 이웃 (home_dong)
        for cd, members in home_group.items():
            if len(members) < 2:
                continue
            for a in members:
                others = [m for m in members if m != a]
                k = min(N_NEIGHBOR, len(others))
                if k == 0:
                    continue
                for b in rng.sample(others, k):
                    key = (a, b) if a < b else (b, a)
                    # 이미 colleague면 skip (강한 게 우선)
                    if any(p[0] == key for p in pairs):
                        continue
                    pairs.add((key, 0.4, "neighbor"))

        rows = []
        for (a, b), strength, rel in pairs:
            rows.append({"a": a, "b": b, "strength": strength, "rel": rel})
        print(f"  KNOWS pairs: {len(rows)}")

        # 양방향 MERGE
        bulk_run(s, """
            UNWIND $batch AS p
            MATCH (a:Agent {id: p.a}), (b:Agent {id: p.b})
            MERGE (a)-[k1:KNOWS]->(b) SET k1.strength = p.strength, k1.relation = p.rel
            MERGE (b)-[k2:KNOWS]->(a) SET k2.strength = p.strength, k2.relation = p.rel
        """, batch=rows, batch_size=5000)
        print(f"  + KNOWS x {len(rows) * 2} (양방향)")

    print("[done] social graph built.")


if __name__ == "__main__":
    main()
