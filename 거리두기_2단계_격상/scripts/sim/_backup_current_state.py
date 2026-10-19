"""현재 Neo4j 정상 데이터 전체 백업 (rollback 완료 시점, 6/1 Night2 redo 전).

대상:
  1. Plan 5/25~6/1 (52,499 nodes)
  2. State 5/25~6/1 (52,499 nodes)
  3. INCLUDES 5/25~6/1 (~450K edges)
  4. Conversation 5/25~5/31 (~38K nodes)
  5. Memory 5/25~6/1 (visited + rumor)
  6. KNOWS_POI 전체 (515K — 5/25~6/1 시점 상태)
  7. Memory{visited day=6/1} (25K — 6/2 night1이 만든 보존 분)

복원 시: JSONL → MERGE cypher 재적재 가능.
"""
import sys
import json
import os
from datetime import datetime

sys.path.insert(0, "scripts/neo4j_load")
from _common import driver_session

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_DIR = f"output/neo4j_backup_{TIMESTAMP}"
os.makedirs(BACKUP_DIR, exist_ok=True)

BATCH_SIZE = 10000


def dump_query(s, query, out_file, batch_label="rows"):
    cnt = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for r in s.run(query):
            f.write(json.dumps(dict(r), ensure_ascii=False, default=str) + "\n")
            cnt += 1
            if cnt % BATCH_SIZE == 0:
                print(f"    ... {cnt:,} {batch_label}", flush=True)
    return cnt


print("=" * 60)
print(f"Backup current normal state — {datetime.now().isoformat()}")
print(f"Output: {BACKUP_DIR}")
print("=" * 60)

with driver_session() as s:
    # 1. Plan 5/25 ~ 6/1
    print("\n[1/7] Plan 5/25 ~ 6/1 ...")
    n = dump_query(s, """
        MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan)
        WHERE p.day >= date('2026-05-25') AND p.day <= date('2026-06-07')
        RETURN p.id AS pid, toString(p.day) AS day, a.id AS aid
        ORDER BY p.day, a.id
    """, f"{BACKUP_DIR}/plan.jsonl", "plans")
    print(f"  Plan: {n:,}")

    # 2. State 5/25 ~ 6/1
    print("\n[2/7] State 5/25 ~ 6/1 ...")
    n = dump_query(s, """
        MATCH (a:Agent)-[:HAS_STATE]->(st:State)
        WHERE st.day >= date('2026-05-25') AND st.day <= date('2026-06-07')
        RETURN a.id AS aid, toString(st.day) AS day,
               st.balance AS balance, st.energy AS energy, st.mood AS mood,
               st.fatigue AS fatigue, st.month_spent AS month_spent,
               st.grant_received AS grant_received,
               st.grant_remaining AS grant_remaining,
               st.policy_lifecycle AS policy_lifecycle
        ORDER BY st.day, a.id
    """, f"{BACKUP_DIR}/state.jsonl", "states")
    print(f"  State: {n:,}")

    # 3. INCLUDES 5/25 ~ 6/1
    print("\n[3/7] INCLUDES 5/25 ~ 6/1 ...")
    n = dump_query(s, """
        MATCH (p:Plan)-[i:INCLUDES]->(poi:POI)
        WHERE p.day >= date('2026-05-25') AND p.day <= date('2026-06-07')
        RETURN p.id AS pid, poi.id AS poi_id,
               i.order AS ord, toString(i.time) AS tm,
               i.anchor AS anchor, i.category AS category, i.sub_category AS sub_category,
               i.intent AS intent, i.actual_spent AS actual_spent,
               i.actual_satisfaction AS actual_satisfaction,
               i.spent_from_policy AS spent_from_policy
        ORDER BY p.id, i.order
    """, f"{BACKUP_DIR}/includes.jsonl", "includes")
    print(f"  INCLUDES: {n:,}")

    # 4. Conversation 5/25 ~ 6/7 (sim 진행에 따라 확장)
    print("\n[4/7] Conversation 5/25 ~ 6/7 ...")
    n = dump_query(s, """
        MATCH (c:Conversation)
        WHERE c.day >= date('2026-05-25') AND c.day <= date('2026-06-07')
        RETURN c.id AS cid, toString(c.day) AS day, c.intent AS intent,
               c.initiator_id AS init_id, c.recipient_id AS recip_id,
               c.topic_type AS topic_type, c.topic_value AS topic_value,
               c.should_inject AS should_inject,
               c.target_day_offset AS target_day_offset,
               c.target_time AS target_time,
               c.meeting_location_hint AS meeting_location_hint,
               c.reasoning AS reasoning
        ORDER BY c.day
    """, f"{BACKUP_DIR}/conv.jsonl", "convs")
    print(f"  Conv: {n:,}")

    # 5. Memory 5/25 ~ 6/1 (rumor + visited)
    print("\n[5/7] Memory 5/25 ~ 6/1 (rumor + visited) ...")
    n = dump_query(s, """
        MATCH (m:Memory)
        WHERE m.day >= date('2026-05-25') AND m.day <= date('2026-06-07')
        RETURN m.id AS mid, m.type AS type, toString(m.day) AS day,
               m.source AS source, m.importance AS importance,
               m.topic_type AS topic_type, m.topic_value AS topic_value,
               m.summary AS summary, m.satisfaction AS satisfaction
        ORDER BY m.day
    """, f"{BACKUP_DIR}/memory.jsonl", "memories")
    print(f"  Memory: {n:,}")

    # 6. KNOWS_POI 전체 (5/25~6/1 시점 정상 상태)
    print("\n[6/7] KNOWS_POI 전체 ...")
    n = dump_query(s, """
        MATCH (a:Agent)-[kp:KNOWS_POI]->(poi:POI)
        RETURN a.id AS aid, poi.id AS poi_id,
               kp.source AS source, toString(kp.since) AS since,
               kp.visit_count AS visit_count,
               kp.avg_satisfaction AS avg_satisfaction,
               kp.affinity AS affinity,
               toString(kp.last_visit) AS last_visit,
               [d IN coalesce(kp.recent_visit_dates,[]) | toString(d)] AS recent_visit_dates
    """, f"{BACKUP_DIR}/knows_poi.jsonl", "knows_poi")
    print(f"  KNOWS_POI: {n:,}")

    # 7. REMEMBERS, PARTICIPATES_IN, ABOUT_* 관계 별도 백업 (Conv/Memory 재적재 시 필요)
    print("\n[7/7] 관계 백업 (REMEMBERS, PARTICIPATES_IN, ABOUT_POI, ABOUT_POLICY, MENTIONS_POI) ...")
    n = dump_query(s, """
        MATCH (a:Agent)-[r:REMEMBERS]->(m:Memory)
        WHERE m.day >= date('2026-05-25') AND m.day <= date('2026-06-07')
        RETURN a.id AS aid, m.id AS mid, toString(r.day) AS day
    """, f"{BACKUP_DIR}/remembers.jsonl", "remembers")
    print(f"  REMEMBERS: {n:,}")

    n = dump_query(s, """
        MATCH (a:Agent)-[r:PARTICIPATES_IN]->(c:Conversation)
        WHERE c.day >= date('2026-05-25') AND c.day <= date('2026-06-07')
        RETURN a.id AS aid, c.id AS cid, r.role AS role
    """, f"{BACKUP_DIR}/participates_in.jsonl", "PI")
    print(f"  PARTICIPATES_IN: {n:,}")

    # ABOUT_POI from Memory
    n = dump_query(s, """
        MATCH (m:Memory)-[:ABOUT_POI]->(poi:POI)
        WHERE m.day >= date('2026-05-25') AND m.day <= date('2026-06-07')
        RETURN m.id AS mid, poi.id AS poi_id
    """, f"{BACKUP_DIR}/about_poi.jsonl", "about_poi")
    print(f"  ABOUT_POI: {n:,}")

print("\n" + "=" * 60)
print("BACKUP DONE")
print("=" * 60)
print(f"Files in {BACKUP_DIR}:")
total_bytes = 0
for f in sorted(os.listdir(BACKUP_DIR)):
    sz = os.path.getsize(f"{BACKUP_DIR}/{f}")
    total_bytes += sz
    print(f"  {f}: {sz:,} bytes")
print(f"  TOTAL: {total_bytes / 1024 / 1024:.1f} MB")
