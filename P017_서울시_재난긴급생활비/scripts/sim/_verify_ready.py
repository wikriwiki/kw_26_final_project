"""9일 baseline 시작 전 최종 검증 — 사용자 3대 진단 + 시드."""
import sys, json
from collections import Counter
sys.path.insert(0, 'scripts/neo4j_load')
from _common import driver_session

with open('output/agents/agents_7500.json', encoding='utf-8') as f:
    agents = json.load(f)
N = len(agents)
o_gu = Counter(a['residence']['gu'] for a in agents)
o_inc = Counter(a['personal'].get('income_level') for a in agents)

with driver_session() as s:
    print("=== 진단 1: 에이전트 7500 + 비율 ===")
    nagent = s.run("MATCH (a:Agent) RETURN count(a) AS c").single()['c']
    nlives = s.run("MATCH (a:Agent) WHERE (a)-[:LIVES_AT]->() RETURN count(a) AS c").single()['c']
    print(f"  Agent: {nagent} (LIVES_AT 보유: {nlives})")
    # gu 비율 대조
    db_gu = Counter()
    for r in s.run("MATCH (a:Agent) RETURN a.residence_gu AS gu"):
        db_gu[r['gu']] += 1
    maxd = max(abs(db_gu[g]*100/nagent - o_gu[g]*100/N) for g in o_gu)
    print(f"  자치구 비율 max diff vs 표본파일: {maxd:.2f}%p")
    # income 비율
    db_inc = Counter()
    for r in s.run("MATCH (a:Agent) RETURN a.p_income_level AS inc"):
        db_inc[r['inc']] += 1
    print("  소득 비율(DB):", {k: f"{db_inc[k]*100/nagent:.1f}%" for k in ['상','중상','중','중하','하']})

    print("\n=== 진단 2: 정책 미주입 ===")
    npol = s.run("MATCH (p:Policy) RETURN count(p) AS c").single()['c']
    nap = s.run("MATCH ()-[r:APPLIED_TO]->() RETURN count(r) AS c").single()['c']
    nap2 = s.run("MATCH ()-[r:applied_to]->() RETURN count(r) AS c").single()['c']
    print(f"  Policy 노드: {npol}, APPLIED_TO: {nap}, applied_to: {nap2}  {'✓ 무정책' if npol==0 else '⚠️ 정책 존재'}")

    print("\n=== 진단 3: POI 무결성 ===")
    npoi = s.run("MATCH (p:POI) RETURN count(p) AS c").single()['c']
    nres = s.run("MATCH (p:POI {type:'residence'}) RETURN count(p) AS c").single()['c']
    ncom = s.run("MATCH (p:POI {type:'commerce'}) RETURN count(p) AS c").single()['c']
    print(f"  POI 총: {npoi:,} (residence {nres}, commerce {ncom:,})")

    print("\n=== 시드 검증: Day-0 State (2026-05-24) ===")
    nst = s.run("MATCH (st:State) WHERE st.day = date('2026-05-24') RETURN count(st) AS c").single()['c']
    bal = s.run("MATCH (st:State) WHERE st.day = date('2026-05-24') RETURN min(st.balance) AS mn, max(st.balance) AS mx, avg(st.balance) AS av").single()
    print(f"  State: {nst} (balance min={bal['mn']:,} max={bal['mx']:,} avg={bal['av']:,.0f})")
    flat = "⚠️ 평준화(시드 실패)" if bal['mn'] == bal['mx'] else "✓ 차등 시드"
    print(f"  {flat}")

    print("\n=== 사회 그래프 (induced subgraph) ===")
    nknows = s.run("MATCH ()-[r:KNOWS]->() RETURN count(r) AS c").single()['c']
    print(f"  KNOWS: {nknows:,} (평균 차수 ~{nknows/nagent:.1f})")
