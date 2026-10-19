"""6/1 Night2 단독 redo (fixed APPLIED_TO → applied_to bug).

전제: Neo4j에서 6/1 Conv + Memory{rumor, day=6/1} 미리 삭제 완료.
멱등성: Conv.id가 UUID이므로 사전 삭제 필수 (이미 됨).

실행:
  cd /g/내 드라이브/Kw/final_project
  python3 scripts/sim/_redo_6_1_night2.py
"""
import sys
from datetime import date

sys.path.insert(0, 'scripts/sim')
sys.path.insert(0, 'scripts/neo4j_load')

from night_interaction import select_interaction_pairs
from night_intent_llm import run_intent_classification
from _common import driver_session

DAY = date(2026, 6, 1)

print("=" * 60)
print(f"6/1 Night2 단독 redo (fixed FETCH_POLICY_CYPHER bug)")
print("=" * 60)

# 사전 검증: Conv 6/1 = 0 (이미 삭제됨), Memory rumor 6/1 = 0
with driver_session() as s:
    conv_pre = s.run("MATCH (c:Conversation) WHERE c.day=date('2026-06-01') RETURN count(c) AS n").single()["n"]
    mem_pre = s.run("MATCH (m:Memory {type:'rumor'}) WHERE m.day=date('2026-06-01') RETURN count(m) AS n").single()["n"]
    print(f"\n[Pre] Conv 6/1: {conv_pre} (=0이어야 안전), Mem_rumor 6/1: {mem_pre}")
    if conv_pre > 0 or mem_pre > 0:
        print("⚠️ 사전 삭제 안 됐음 — 중복 적재 위험. 종료.")
        sys.exit(1)

# 1. 쌍 선택
print(f"\n[1/3] select_interaction_pairs({DAY}) ...")
pairs = select_interaction_pairs(DAY, verbose=False)
print(f"  Selected: {len(pairs)} pairs")

if not pairs:
    print("  No pairs — exit.")
    sys.exit(0)

# 2. 의도 분류 + 적재
print(f"\n[2/3] run_intent_classification (workers=32) ...")
stats = run_intent_classification(DAY, pairs, workers=32, verbose=False)
print(f"  Stats: {stats}")

# 3. 검증
print(f"\n[3/3] 검증 ...")
with driver_session() as s:
    conv = s.run("MATCH (c:Conversation) WHERE c.day=date('2026-06-01') RETURN count(c) AS n").single()["n"]
    mem_r = s.run("MATCH (m:Memory {type:'rumor'}) WHERE m.day=date('2026-06-01') RETURN count(m) AS n").single()["n"]
    pol_conv = s.run("MATCH (c:Conversation)-[:ABOUT_POLICY]->(:Policy {id:'P009'}) WHERE c.day=date('2026-06-01') RETURN count(c) AS n").single()["n"]
    by_intent = s.run("MATCH (c:Conversation) WHERE c.day=date('2026-06-01') RETURN c.intent AS i, count(c) AS n ORDER BY n DESC").data()
    print(f"  Conv 6/1: {conv}")
    print(f"  Memory{{rumor, day=6/1}}: {mem_r}")
    print(f"  Conv ABOUT_POLICY P009: {pol_conv} (정책 화제 — fixed bug 효과)")
    print(f"  By intent: {by_intent}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
