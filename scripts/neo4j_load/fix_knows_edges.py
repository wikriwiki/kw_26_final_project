"""KNOWS 엣지 적재 — agent 노드 속성 직접 사용 (06_social.py 수정 버전).

- 같은 work_dong (workplace_dong_code_raw) 동료: strength=0.6, agent당 최대 5명
- 같은 home_dong (residence_dong_code_raw) 이웃: strength=0.4, agent당 최대 3명
- 양방향 MERGE
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session  # noqa: E402

SEED = 42
N_COLLEAGUE = 5
N_NEIGHBOR = 3


def main():
    rng = random.Random(SEED)
    with driver_session() as s:
        print("[1] Agent fetch")
        home_group = defaultdict(list)
        work_group = defaultdict(list)
        agents = []
        for r in s.run("""
            MATCH (a:Agent)
            RETURN a.id AS id,
                   a.residence_dong_code_raw AS home,
                   a.workplace_dong_code_raw AS work
        """):
            aid = r["id"]
            agents.append(aid)
            if r["home"]:
                home_group[r["home"]].append(aid)
            if r["work"]:
                work_group[r["work"]].append(aid)
        print(f"  agents: {len(agents):,}")
        print(f"  home_dongs: {len(home_group):,} (avg {sum(len(v) for v in home_group.values())/max(len(home_group),1):.1f} agents/dong)")
        print(f"  work_dongs: {len(work_group):,} (avg {sum(len(v) for v in work_group.values())/max(len(work_group),1):.1f} agents/dong)")
        print()

        print("[2] pair 생성")
        pairs: dict[tuple[str, str], tuple[float, str]] = {}

        # colleague (work_dong, strength 0.6)
        for dong, members in work_group.items():
            if len(members) < 2:
                continue
            for a in members:
                others = [m for m in members if m != a]
                k = min(N_COLLEAGUE, len(others))
                if k == 0:
                    continue
                for b in rng.sample(others, k):
                    key = (a, b) if a < b else (b, a)
                    # 더 강한 관계 우선
                    if key not in pairs or pairs[key][0] < 0.6:
                        pairs[key] = (0.6, "colleague")

        # neighbor (home_dong, strength 0.4)
        for dong, members in home_group.items():
            if len(members) < 2:
                continue
            for a in members:
                others = [m for m in members if m != a]
                k = min(N_NEIGHBOR, len(others))
                if k == 0:
                    continue
                for b in rng.sample(others, k):
                    key = (a, b) if a < b else (b, a)
                    if key not in pairs:
                        pairs[key] = (0.4, "neighbor")
                    # 이미 colleague면 skip (강한 게 우선)

        n_col = sum(1 for _, (s_, rel) in pairs.items() if rel == "colleague")
        n_neigh = sum(1 for _, (s_, rel) in pairs.items() if rel == "neighbor")
        print(f"  colleague pairs: {n_col:,}")
        print(f"  neighbor  pairs: {n_neigh:,}")
        print(f"  총 pairs: {len(pairs):,}")
        print()

        print("[3] KNOWS 엣지 적재 (양방향 MERGE)")
        rows = []
        for (a, b), (strength, rel) in pairs.items():
            rows.append({"a": a, "b": b, "strength": strength, "rel": rel})

        BATCH = 5000
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i+BATCH]
            s.run("""
                UNWIND $rows AS row
                MATCH (a:Agent {id: row.a}), (b:Agent {id: row.b})
                MERGE (a)-[r1:KNOWS]->(b)
                SET r1.type = row.rel, r1.strength = row.strength
                MERGE (b)-[r2:KNOWS]->(a)
                SET r2.type = row.rel, r2.strength = row.strength
            """, rows=batch)
            print(f"  ... {min(i+BATCH, len(rows)):,}/{len(rows):,}")

        print()
        print("[4] 검증")
        r = s.run("MATCH ()-[r:KNOWS]->() RETURN count(r) AS n").single()
        print(f"  KNOWS 엣지 총: {r['n']:,}")

        rows = s.run("MATCH ()-[r:KNOWS]->() RETURN r.type AS type, count(*) AS n ORDER BY n DESC").data()
        for r in rows:
            print(f"    {r['type']}: {r['n']:,}")

        # agent당 평균 KNOWS
        r = s.run("""
            MATCH (a:Agent)
            OPTIONAL MATCH (a)-[r:KNOWS]->()
            WITH a, count(r) AS k
            RETURN avg(k) AS avg_k, max(k) AS max_k, min(k) AS min_k
        """).single()
        print(f"  agent당 KNOWS: avg={r['avg_k']:.1f}, max={r['max_k']}, min={r['min_k']}")


if __name__ == "__main__":
    main()
