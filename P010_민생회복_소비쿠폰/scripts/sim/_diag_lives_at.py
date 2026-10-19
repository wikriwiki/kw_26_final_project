"""LIVES_AT 155명 누락 근본 원인 진단."""
import sys
sys.path.insert(0, 'scripts/neo4j_load')
from _common import driver_session

with open('output/sim_9d_baseline/agent_ids_7500.txt', encoding='utf-8') as f:
    ids7500 = [l.strip() for l in f if l.strip()]

with driver_session() as s:
    # 1. 전체에서 LIVES_AT 없는 agent
    tot_no = s.run("MATCH (a:Agent) WHERE NOT (a)-[:LIVES_AT]->() RETURN count(a) AS n").single()['n']
    print(f"전체 14,881 중 LIVES_AT 없음: {tot_no}")

    # 2. 7500 중 LIVES_AT 없는 agent 상세
    no_lives = s.run(
        "MATCH (a:Agent) WHERE a.id IN $ids AND NOT (a)-[:LIVES_AT]->() "
        "RETURN a.id AS id, a.residence_dong_name AS dong, a.residence_dong_code_raw AS code, a.residence_gu AS gu",
        ids=ids7500).data()
    print(f"7500 중 LIVES_AT 없음: {len(no_lives)}")
    print("샘플 5:")
    for r in no_lives[:5]:
        print(f"  {r['id']} | dong={r['dong']} code={r['code']} gu={r['gu']}")

    # 3. residence POI가 어떤 라벨/속성으로 들어있나 — 먼저 POI 타입 키 확인
    pk = s.run("MATCH (p:POI) RETURN keys(p) AS k LIMIT 1").single()
    print(f"\nPOI keys: {pk['k']}")

    # 4. 누락 agent들의 dong code 분포 + 해당 dong residence POI 유무
    dong_dist = {}
    for r in no_lives:
        dong_dist[r['code']] = dong_dist.get(r['code'], 0) + 1
    print(f"\n누락 agent dong code 분포 (상위 10):")
    for code, n in sorted(dong_dist.items(), key=lambda x: -x[1])[:10]:
        # residence POI를 가진 dong인지 — LIVES_AT 대상 POI 확인
        # 같은 dong에 사는 다른 agent는 LIVES_AT 있나?
        other = s.run(
            "MATCH (a:Agent {residence_dong_code_raw: $c})-[:LIVES_AT]->() RETURN count(a) AS n",
            c=code).single()['n']
        print(f"  code={code}: 누락 {n}명 / 같은 dong에 LIVES_AT 보유 agent {other}명")
