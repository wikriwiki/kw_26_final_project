"""6/1 + 6/2 INCLUDES 후처리 복구.

LLM이 pick_reason에 "P009"/"지원금" 언급했으나 spent_from_policy를 빈 dict로 둔 케이스를
agent별 grant 한도 내에서 actual_spent 전액을 policy_spend로 복구.

처리:
- 각 agent의 6/1+6/2 거래 시간순
- reasoning에 P009/지원금 + spent_from_policy 빈 케이스 식별
- grant_remaining 추적하며 actual_spent 적재 (한도 내)
- State 6/2 grant_remaining 재계산
"""
import sys, json
sys.path.insert(0, 'scripts/neo4j_load')
from _common import driver_session

INCOME_GRANT = {"중상": 100000, "중": 250000, "중하": 450000, "하": 600000}

print("=" * 60)
print("Reasoning-JSON 불일치 후처리 복구 (6/1 + 6/2)")
print("=" * 60)

with driver_session() as s:
    # 1. 복구 대상 agent
    agents = s.run("""
        MATCH (a:Agent)-[:HAS_STATE]->(st:State {day:date('2026-06-01')})
        WHERE st.grant_received IS NOT NULL AND st.grant_received <> '{}'
        RETURN a.id AS aid, a.p_income_level AS inc
    """).data()
    print(f"\n[1] 복구 대상 agent: {len(agents)}")

    # 2. agent별 처리
    total_fixed = 0
    total_recovered = 0
    by_inc = {"중상": 0, "중": 0, "중하": 0, "하": 0}

    for ai, ag in enumerate(agents):
        aid = ag['aid']
        inc = ag['inc']
        received = INCOME_GRANT.get(inc, 0)
        if received == 0:
            continue

        # 이 agent의 6/1+6/2 reasoning-JSON 불일치 거래 시간순
        txns = s.run("""
            MATCH (a:Agent {id:$aid})-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->()
            WHERE p.day IN [date('2026-06-01'), date('2026-06-02')]
                AND i.pick_reason IS NOT NULL
                AND (i.pick_reason CONTAINS 'P009' OR i.pick_reason CONTAINS '지원금')
                AND (i.spent_from_policy IS NULL OR i.spent_from_policy = '{}' OR i.spent_from_policy = 'null')
                AND i.actual_spent > 0
            RETURN p.day AS day, i.order AS ord, p.id AS pid, i.actual_spent AS sp
            ORDER BY p.day, i.order
        """, aid=aid).data()

        if not txns:
            continue

        # 이미 사용된 분 (reasoning-JSON 일치된 부분) 차감
        already_used = s.run("""
            MATCH (a:Agent {id:$aid})-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->()
            WHERE p.day IN [date('2026-06-01'), date('2026-06-02')]
                AND i.spent_from_policy IS NOT NULL
                AND i.spent_from_policy <> '{}'
                AND i.spent_from_policy <> 'null'
            RETURN sum(i.actual_spent) AS s
        """, aid=aid).single()['s'] or 0

        remaining = received - int(already_used)
        if remaining <= 0:
            continue

        # 각 거래에 적재
        for txn in txns:
            if remaining <= 0:
                break
            amt = min(int(txn['sp']), remaining)
            sp_json = json.dumps({"P009": amt})
            s.run("""
                MATCH (p:Plan {id:$pid})-[i:INCLUDES]->()
                WHERE i.order = $ord
                SET i.spent_from_policy = $sp
            """, pid=txn['pid'], ord=int(txn['ord']), sp=sp_json)
            remaining -= amt
            total_fixed += 1
            total_recovered += amt
            by_inc[inc] = by_inc.get(inc, 0) + 1

        # State 6/2 grant_remaining 갱신
        s.run("""
            MATCH (a:Agent {id:$aid})-[:HAS_STATE]->(st:State {day:date('2026-06-02')})
            SET st.grant_remaining = $rem
        """, aid=aid, rem=json.dumps({"P009": remaining}))

        if (ai + 1) % 1000 == 0:
            print(f"  {ai+1}/{len(agents)} agents, {total_fixed} INCLUDES 복구, {total_recovered:,}원")

    print(f"\n[2] 복구 완료:")
    print(f"  총 INCLUDES 복구: {total_fixed}")
    print(f"  총 정책 사용액: {total_recovered:,}원")
    print(f"  분위별: {by_inc}")

    # 3. 검증
    print(f"\n[3] 검증:")
    for d in ['2026-06-01', '2026-06-02']:
        rm = s.run(f"""
            MATCH (:Plan {{day:date('{d}')}})-[i:INCLUDES]->()
            WHERE i.pick_reason IS NOT NULL
                AND (i.pick_reason CONTAINS 'P009' OR i.pick_reason CONTAINS '지원금')
                AND (i.spent_from_policy IS NULL OR i.spent_from_policy = '{{}}')
            RETURN count(i) AS n
        """).single()['n']
        u = s.run(f"""
            MATCH (:Plan {{day:date('{d}')}})-[i:INCLUDES]->()
            WHERE i.spent_from_policy IS NOT NULL
                AND i.spent_from_policy <> '{{}}'
                AND i.spent_from_policy <> 'null'
            RETURN count(i) AS n, sum(i.actual_spent) AS sp
        """).single()
        uu = s.run(f"""
            MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan {{day:date('{d}')}})-[i:INCLUDES]->()
            WHERE i.spent_from_policy IS NOT NULL
                AND i.spent_from_policy <> '{{}}'
                AND i.spent_from_policy <> 'null'
            RETURN count(DISTINCT a) AS n
        """).single()['n']
        print(f"  {d}: 정책 거래 {u['n']}건 / {(u['sp'] or 0):,.0f}원, unique {uu}명, 잔여 누락 {rm}건")

print("=" * 60)
print("DONE")
print("=" * 60)
