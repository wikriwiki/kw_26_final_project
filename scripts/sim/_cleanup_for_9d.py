"""9일 baseline 준비 cleanup.

1) Policy 노드 + APPLIED_TO/applied_to 삭제 (무정책 baseline)
2) 시뮬 잔재 삭제: Plan, State, Conversation, Memory (+ 모든 부속 엣지)
3) 비표본 agent(7500 외 7381명) DETACH DELETE

정적 구조(POI, Category, District, Dong, KNOWS, KNOWS_POI, LIVES_AT, WORKS_AT)는 보존.
백업: data/neo4j_load/dumps/neo4j_3day_p009_20260601_1515.dump
--apply 없으면 dry-run.
"""
import sys
sys.path.insert(0, 'scripts/neo4j_load')
from _common import driver_session

APPLY = '--apply' in sys.argv

with open('output/sim_9d_baseline/agent_ids_7500.txt', encoding='utf-8') as f:
    ids7500 = [l.strip() for l in f if l.strip()]


def batch_delete_label(s, label, batch=5000):
    total = 0
    while True:
        n = s.run(f"MATCH (n:`{label}`) WITH n LIMIT {batch} DETACH DELETE n RETURN count(n) AS c").single()['c']
        total += n
        if n == 0:
            break
    return total


with driver_session() as s:
    print("=== BEFORE ===")
    for lab in ['Agent', 'Plan', 'State', 'Conversation', 'Memory', 'Policy', 'POI']:
        n = s.run(f"MATCH (n:`{lab}`) RETURN count(n) AS c").single()['c']
        print(f"  {lab}: {n:,}")

    if not APPLY:
        # dry-run: 삭제 대상 카운트만
        nonsample = s.run("MATCH (a:Agent) WHERE NOT a.id IN $ids RETURN count(a) AS c", ids=ids7500).single()['c']
        print(f"\n[DRY-RUN] 삭제 예정: Policy 전체, Plan/State/Conversation/Memory 전체, 비표본 agent {nonsample}")
        print("--apply 로 실행")
        sys.exit(0)

    print("\n=== 1) Policy 삭제 ===")
    d = batch_delete_label(s, 'Policy')
    print(f"  Policy 삭제: {d}")

    print("=== 2) 시뮬 잔재 삭제 ===")
    for lab in ['Plan', 'State', 'Conversation', 'Memory']:
        d = batch_delete_label(s, lab)
        print(f"  {lab} 삭제: {d:,}")

    print("=== 3) 비표본 agent 삭제 ===")
    # 7500 keep agent에 임시 라벨
    s.run("MATCH (a:Agent) WHERE a.id IN $ids SET a:Keep", ids=ids7500)
    kept = s.run("MATCH (a:Keep) RETURN count(a) AS c").single()['c']
    print(f"  Keep 마킹: {kept}")
    total = 0
    while True:
        n = s.run("MATCH (a:Agent) WHERE NOT a:Keep WITH a LIMIT 500 DETACH DELETE a RETURN count(a) AS c").single()['c']
        total += n
        if n == 0:
            break
    print(f"  비표본 agent 삭제: {total:,}")
    s.run("MATCH (a:Keep) REMOVE a:Keep")

    print("\n=== AFTER ===")
    for lab in ['Agent', 'Plan', 'State', 'Conversation', 'Memory', 'Policy', 'POI']:
        n = s.run(f"MATCH (n:`{lab}`) RETURN count(n) AS c").single()['c']
        print(f"  {lab}: {n:,}")
