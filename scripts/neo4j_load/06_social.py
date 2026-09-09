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
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session, bulk_run

SEED = 42
N_COLLEAGUE = 5
N_NEIGHBOR = 3


class _OtherMembers(Sequence):
    """Ordered view of members excluding every occurrence of one agent.

    random.sample draws the same indices as it does for the filtered list.
    Large groups no longer need a new O(group size) list for each agent.
    Shifted exclusion positions also preserve duplicate input rows.
    """

    def __init__(self, members, excluded_positions):
        self.members = members
        self.offsets = [position - i for i, position in enumerate(excluded_positions)]

    def __len__(self):
        return len(self.members) - len(self.offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.members[index + bisect_right(self.offsets, index)]


def build_social_pairs(work_group, home_group, rng):
    """Build the original ordered sampling process with indexed membership."""
    pairs = set()
    pair_keys = set()
    for groups, limit, strength, relation in (
        (work_group, N_COLLEAGUE, 0.6, "colleague"),
        (home_group, N_NEIGHBOR, 0.4, "neighbor"),
    ):
        for members in groups.values():
            if len(members) < 2:
                continue
            positions = defaultdict(list)
            for i, member in enumerate(members):
                positions[member].append(i)
            # Build each view once, including for repeated agent IDs.
            others_by_agent = {a: _OtherMembers(members, pos) for a, pos in positions.items()}
            for a in members:
                others = others_by_agent[a]
                k = min(limit, len(others))
                if k == 0:
                    continue
                for b in rng.sample(others, k):
                    key = (a, b) if a < b else (b, a)
                    if relation == "neighbor" and key in pair_keys:
                        continue
                    pairs.add((key, strength, relation))
                    pair_keys.add(key)
    return pairs


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
        pairs = build_social_pairs(work_group, home_group, rng)

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
