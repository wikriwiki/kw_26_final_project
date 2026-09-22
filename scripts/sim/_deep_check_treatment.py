"""Day 8 (treatment 1일차) 정밀 점검 — 9개 axis.

1. 정책 노드·관계 무결성
2. State JSON 적재 정확성 (grant_received, grant_remaining, policy_lifecycle)
3. INCLUDES.spent_from_policy JSON 적재 정확성
4. 분위별 grant 차등 적용 (10만/25만/45만/60만)
5. balance vs grant 독립 지갑 (우리 수정 코드 작동 확인)
6. Day별 분포 일관성 (baseline vs treatment)
7. 데이터 손실·중복
8. jsonl vs Neo4j 일치
9. 만족도·소비액 정상 범위
"""
import json, sys
from collections import Counter, defaultdict
sys.path.insert(0, 'scripts/neo4j_load')
from _common import driver_session

print("=" * 70)
print("Day 8 (treatment D1) 정밀 점검")
print("=" * 70)

with driver_session() as s:
    # 1. 정책 무결성
    print("\n[1] 정책 노드·관계 무결성")
    pol = s.run("MATCH (p:Policy {id:'P009'}) RETURN p").single()
    if pol:
        p = pol['p']
        print(f"  P009 노드 ✓: from={p.get('effective_from')}, until={p.get('effective_until')}")
        print(f"  type={p.get('type')}, target_all_seoul={p.get('target_all_seoul')}")
        print(f"  income_grants={p.get('income_grants')}")
        print(f"  excluded_income={p.get('excluded_income')}")
    nap = s.run("MATCH (:Policy {id:'P009'})-[r:applied_to]->(:District) RETURN count(r) AS n").single()['n']
    print(f"  applied_to → District: {nap} (=25 OK)" if nap == 25 else f"  ⚠️ applied_to {nap} ≠ 25")

    # 2. State JSON 적재
    print("\n[2] Day 8 State JSON 적재 정확성")
    sample = s.run("""
        MATCH (st:State) WHERE st.day = date('2026-06-01')
        RETURN st.agent_id AS aid, st.balance AS bal, st.grant_received AS gr,
               st.grant_remaining AS rem, st.policy_lifecycle AS lc LIMIT 3
    """).data()
    for r in sample:
        print(f"  {r['aid']}: balance={r['bal']:,}원")
        print(f"    grant_received={r['gr']}")
        print(f"    grant_remaining={r['rem']}")
        print(f"    policy_lifecycle={r['lc']}")

    # 3. INCLUDES.spent_from_policy JSON 적재
    print("\n[3] INCLUDES.spent_from_policy JSON 적재")
    spt = s.run("""
        MATCH (:Plan {day: date('2026-06-01')})-[i:INCLUDES]->()
        WHERE i.spent_from_policy IS NOT NULL AND i.spent_from_policy <> '{}' AND i.spent_from_policy <> 'null'
        RETURN i.actual_spent AS sp, i.spent_from_policy AS pol_sp LIMIT 5
    """).data()
    for r in spt:
        print(f"  actual_spent={r['sp']}원, spent_from_policy={r['pol_sp']}")

    # 4. 분위별 grant 차등 적용 검증
    print("\n[4] 분위별 grant 차등 적용 (jsonl + State)")
    EXPECTED = {"중상": 100000, "중": 250000, "중하": 450000, "하": 600000}
    inc_grants = defaultdict(list)
    rows = s.run("""
        MATCH (a:Agent)-[:HAS_STATE]->(st:State {day: date('2026-06-01')})
        WHERE st.grant_received IS NOT NULL AND st.grant_received <> '{}'
        RETURN a.p_income_level AS inc, st.grant_received AS gr LIMIT 1000
    """).data()
    for r in rows:
        try:
            gr_json = json.loads(r['gr'])
            p009_amt = gr_json.get('P009', 0)
            if p009_amt > 0:
                inc_grants[r['inc']].append(p009_amt)
        except: pass
    for inc, expected in EXPECTED.items():
        amts = inc_grants.get(inc, [])
        if amts:
            unique = set(amts)
            ok = len(unique) == 1 and list(unique)[0] == expected
            print(f"  {inc}: 예상 {expected:,}원 / 실제 {list(unique)} ({len(amts)}명) {'✓' if ok else '⚠️'}")
        else:
            print(f"  {inc}: 데이터 없음")

    # 5. 독립 지갑 — balance에 grant 안 더해졌는지
    print("\n[5] 독립 지갑 검증 — balance에 grant 안 들어가야")
    # Day 7 (5/31) State balance vs Day 8 (6/1) State balance 비교
    # grant 받은 agent의 balance가 Day 7 → Day 8 사이 grant만큼 증가했으면 버그
    bal_check = s.run("""
        MATCH (a:Agent)-[:HAS_STATE]->(st_pre:State {day: date('2026-05-31')})
        MATCH (a)-[:HAS_STATE]->(st_post:State {day: date('2026-06-01')})
        WHERE st_post.grant_received IS NOT NULL AND st_post.grant_received <> '{}'
        RETURN a.id AS aid, a.p_income_level AS inc,
               st_pre.balance AS pre_bal, st_post.balance AS post_bal,
               st_post.grant_received AS gr LIMIT 5
    """).data()
    if bal_check:
        print(f"  Day 7 → Day 8 balance 변화 + grant 비교 (grant가 balance에 안 더해져야):")
        for r in bal_check:
            try:
                gr = json.loads(r['gr']).get('P009', 0)
            except: gr = 0
            diff = r['post_bal'] - r['pre_bal']
            # diff < 0 (소비로 감소) 또는 작은 양수면 OK. diff ≈ gr이면 버그
            anomaly = "⚠️ 버그 가능" if diff > gr * 0.5 and gr > 0 else "✓"
            print(f"    {r['aid']} ({r['inc']}): Day7 bal {r['pre_bal']:,} → Day8 bal {r['post_bal']:,} (diff {diff:+,}), grant {gr:,} {anomaly}")

    # 6. Day별 분포 일관성 (baseline 평균 vs Day 8)
    print("\n[6] Day별 분포 일관성 — baseline 평균 vs Day 8")
    for d in ['2026-05-25', '2026-05-28', '2026-05-29', '2026-06-01']:
        r = s.run(f"""
            MATCH (:Plan {{day: date('{d}')}})-[i:INCLUDES]->(p:POI {{type:'commerce'}})
            WHERE i.actual_spent > 0
            RETURN count(i) AS n, avg(i.actual_spent) AS avg_sp,
                   avg(i.actual_satisfaction) AS avg_sat, sum(i.actual_spent) AS tot
        """).data()[0]
        print(f"  {d}: commerce {r['n']}건, avg {r['avg_sp']:.0f}원, sat {r['avg_sat']:.3f}, 총 {int(r['tot']):,}원")

    # 7. 데이터 손실·중복
    print("\n[7] Day 8 데이터 손실·중복")
    plan = s.run("MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan {day: date('2026-06-01')}) RETURN count(DISTINCT a) AS agents, count(p) AS plans").single()
    state = s.run("MATCH (a:Agent)-[:HAS_STATE]->(st:State {day: date('2026-06-01')}) RETURN count(DISTINCT a) AS agents, count(st) AS states").single()
    print(f"  Plan: {plan['plans']} / unique agent: {plan['agents']} (=7500 OK)" if plan['agents'] == 7500 else f"  ⚠️ Plan agents {plan['agents']} ≠ 7500")
    print(f"  State: {state['states']} / unique agent: {state['agents']} (=7500 OK)" if state['agents'] == 7500 else f"  ⚠️ State agents {state['agents']} ≠ 7500")

# 8. jsonl vs Neo4j 일치
print("\n[8] jsonl vs Neo4j 일치 검증")
fp = r'C:\Users\Administrator\sim_output_9d\metrics\day_2026-06-01.jsonl'
jsonl_aids = set()
jsonl_grant_total = 0
with open(fp, encoding='utf-8') as f:
    for line in f:
        try:
            r = json.loads(line)
            if r.get('status') == 'ok':
                jsonl_aids.add(r['aid'])
                jsonl_grant_total += r.get('grant_applied_today', 0)
        except: pass
with driver_session() as s:
    neo4j_state_aids = set(r['aid'] for r in s.run("MATCH (a:Agent)-[:HAS_STATE]->(:State {day: date('2026-06-01')}) RETURN a.id AS aid"))
    # State에 grant_received 합계
    grant_state_total = 0
    for r in s.run("MATCH (:State {day: date('2026-06-01')}) RETURN grant_received AS gr").data():
        try:
            grant_state_total += json.loads(r['gr']).get('P009', 0)
        except: pass
print(f"  jsonl ok agent: {len(jsonl_aids)}")
print(f"  Neo4j State agent: {len(neo4j_state_aids)}")
print(f"  jsonl만: {len(jsonl_aids - neo4j_state_aids)} / Neo4j만: {len(neo4j_state_aids - jsonl_aids)}")
print(f"  jsonl grant_applied_today 합: {jsonl_grant_total:,}원")

# 9. 만족도·소비액 정상 범위
print("\n[9] 만족도·소비액 정상 범위")
with driver_session() as s:
    r = s.run("""
        MATCH (:Plan {day: date('2026-06-01')})-[i:INCLUDES]->(:POI {type:'commerce'})
        WHERE i.actual_spent > 0
        RETURN min(i.actual_spent) AS min_sp, max(i.actual_spent) AS max_sp, avg(i.actual_spent) AS avg_sp,
               min(i.actual_satisfaction) AS min_sat, max(i.actual_satisfaction) AS max_sat,
               sum(CASE WHEN i.actual_satisfaction < 0 OR i.actual_satisfaction > 1 THEN 1 ELSE 0 END) AS sat_out,
               sum(CASE WHEN i.actual_spent < 0 OR i.actual_spent > 1000000 THEN 1 ELSE 0 END) AS sp_out
    """).single()
    print(f"  소비액: min={r['min_sp']}원, max={r['max_sp']:,}원, avg={r['avg_sp']:.0f}원, 범위 외 {r['sp_out']}건")
    print(f"  만족도: min={r['min_sat']}, max={r['max_sat']}, 0~1 범위 외 {r['sat_out']}건")
