"""Neo4j Agent 노드에 workplace_dong_code_raw 충원.

source: output/stats/workplace_population.json (동별 성별·연령 직장인구)
대상: 학생·은퇴 제외 모든 agent (12,593 / 15,000 추정)
방식: 각 agent의 성별·연령에 맞는 동별 직장인구 가중치로 random assign.

직장 dong이 거주 dong과 같을 수 있음 (자영업·재택 등). 실제 출퇴근 인구 통계 반영.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session  # noqa: E402

WORKPLACE_POP_JSON = Path("output/stats/workplace_population.json")
SEED = 42
SKIP_LIFE_STAGES = {"은퇴", "학생"}  # 직장 없는 그룹


def load_workplace_distribution():
    """동별 성별·연령 가중치 → {(gender, age_group): [(dong_code, weight), ...]}"""
    with open(WORKPLACE_POP_JSON, encoding="utf-8") as f:
        d = json.load(f)
    # 가능한 키: F_20대, F_30대, F_40대, F_50대, F_60대, M_*
    dist: dict[str, list[tuple[str, int]]] = {}
    for dong_code, info in d.items():
        for ga_key, n in (info.get("by_gender_age") or {}).items():
            if n <= 0:
                continue
            dist.setdefault(ga_key, []).append((dong_code, n))
    print(f"  로드된 (gender, age) 키: {len(dist)}")
    for k, v in dist.items():
        print(f"    {k}: {len(v):,} 동 (합 {sum(w for _, w in v):,}명)")
    return dist


def weighted_choice(items: list[tuple[str, int]], rng: random.Random) -> str:
    """[(dong, weight), ...] 에서 weight 비례 random pick."""
    total = sum(w for _, w in items)
    r = rng.uniform(0, total)
    cum = 0
    for dong, w in items:
        cum += w
        if r <= cum:
            return dong
    return items[-1][0]


def main():
    rng = random.Random(SEED)
    print("[1] workplace 인구 분포 로드")
    dist = load_workplace_distribution()
    print()

    print("[2] Neo4j Agent fetch (직장 가능한 agent만)")
    with driver_session() as s:
        # 학생·은퇴 제외, residence_dong_code_raw 있는 agent
        rows = list(s.run("""
            MATCH (a:Agent)
            WHERE NOT a.p_life_stage IN $skip
              AND a.residence_dong_code_raw IS NOT NULL
              AND a.residence_dong_code_raw <> ''
            RETURN a.id AS aid, a.p_gender AS gender, a.p_age_group AS age, a.residence_dong_code_raw AS home
        """, skip=list(SKIP_LIFE_STAGES)))
        print(f"  대상 agent: {len(rows):,}")
        print()

        print("[3] workplace dong assign")
        updates = []
        no_dist = 0
        for r in rows:
            aid = r["aid"]
            gender = r["gender"]  # F / M
            age = r["age"]        # 20대 / 30대 ...
            if not gender or not age:
                continue
            key = f"{gender}_{age}"
            choices = dist.get(key)
            if not choices:
                no_dist += 1
                continue
            wd_code = weighted_choice(choices, rng)
            updates.append({"aid": aid, "wd": wd_code})

        print(f"  assigned: {len(updates):,}")
        print(f"  분포 없는 (gender,age) skip: {no_dist:,}")
        print()

        print("[4] Neo4j UPDATE (workplace_dong_code_raw)")
        # workplace_dong_name도 같이 — workplace_population.json에 dong_name 있음
        with open(WORKPLACE_POP_JSON, encoding="utf-8") as f:
            wp_meta = json.load(f)
        dong_name_map = {code: info.get("dong_name") for code, info in wp_meta.items()}

        BATCH = 1000
        for i in range(0, len(updates), BATCH):
            batch = [{**u, "wd_name": dong_name_map.get(u["wd"], "")} for u in updates[i:i+BATCH]]
            s.run("""
                UNWIND $rows AS row
                MATCH (a:Agent {id: row.aid})
                SET a.workplace_dong_code_raw = row.wd,
                    a.workplace_dong_name = row.wd_name
            """, rows=batch)
            print(f"  ... {min(i + BATCH, len(updates)):,}/{len(updates):,}")

        print()
        print("[5] 검증")
        r = s.run("""
            MATCH (a:Agent)
            RETURN count(a) AS total,
                   sum(CASE WHEN a.workplace_dong_code_raw IS NOT NULL AND a.workplace_dong_code_raw <> '' THEN 1 ELSE 0 END) AS with_work
        """).single()
        print(f"  Agent 전체: {r['total']:,}")
        print(f"  workplace_dong_code_raw 채움: {r['with_work']:,} ({r['with_work']*100/r['total']:.1f}%)")

        # 일부 sample
        print()
        print("  Sample assigned agent:")
        rs = s.run("""
            MATCH (a:Agent)
            WHERE a.workplace_dong_code_raw IS NOT NULL AND a.workplace_dong_code_raw <> ''
            RETURN a.id AS aid, a.p_gender AS g, a.p_age_group AS age,
                   a.residence_dong_code_raw AS home, a.workplace_dong_code_raw AS work,
                   a.workplace_dong_name AS work_name
            LIMIT 5
        """).data()
        for row in rs:
            print(f"    {row['aid']} ({row['g']},{row['age']}) home={row['home']} → work={row['work']} ({row['work_name']})")


if __name__ == "__main__":
    main()
