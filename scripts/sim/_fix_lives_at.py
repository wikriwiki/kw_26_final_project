"""LIVES_AT 없는 agent(321명)를 같은 gu의 residence POI에 연결.

근본 원인: 특정 dong(강북 수유/번 일부, 강동 상일동, 강남 일원2동 등 9개)이
소상공인 POI 데이터에 residence POI가 없음 + dong 좌표도 없어 05_anchors의
nearest fallback이 실패 → LIVES_AT 누락.

fix: agent의 residence_gu(gu-prefix 앞5자리) 내 residence POI 중 seeded-random 1개 배정.
gu 비율 보존. additive MERGE라 기존 LIVES_AT는 안 건드림.
"""
import sys, random
from collections import defaultdict
sys.path.insert(0, 'scripts/neo4j_load')
from _common import driver_session

rng = random.Random(42)
APPLY = '--apply' in sys.argv

with driver_session() as s:
    no_lives = s.run(
        "MATCH (a:Agent) WHERE NOT (a)-[:LIVES_AT]->() "
        "RETURN a.id AS id, a.residence_dong_code_raw AS code, a.residence_gu AS gu").data()
    print(f"LIVES_AT 없는 agent: {len(no_lives)}")

    res_by_gu = defaultdict(list)
    for r in s.run("MATCH (p:POI {type:'residence'}) RETURN p.id AS id, p.dong_code AS code"):
        if r['code']:
            res_by_gu[str(r['code'])[:5]].append(r['id'])
    print(f"gu-prefix 풀: {len(res_by_gu)}개 gu, residence POI {sum(len(v) for v in res_by_gu.values())}")

    assignments = []
    fail = []
    for a in no_lives:
        gu5 = (a['code'] or '')[:5]
        pool = res_by_gu.get(gu5)
        if pool:
            assignments.append({"aid": a['id'], "poi_id": rng.choice(pool)})
        else:
            fail.append(a['id'])
    print(f"배정 성공: {len(assignments)}, 실패(gu에 residence POI 없음): {len(fail)}")
    if fail:
        print(f"  실패 샘플: {fail[:5]}")

    by_gu = defaultdict(int)
    for a in no_lives:
        by_gu[a['gu']] += 1
    print("gu별 누락 분포:", dict(by_gu))

    if not APPLY:
        print("\n[DRY-RUN] --apply 없으면 적용 안 함")
    else:
        BATCH = 1000
        for i in range(0, len(assignments), BATCH):
            chunk = assignments[i:i+BATCH]
            s.run("""
                UNWIND $batch AS p
                MATCH (a:Agent {id: p.aid}), (poi:POI {id: p.poi_id})
                MERGE (a)-[:LIVES_AT]->(poi)
            """, batch=chunk)
        print(f"\n[APPLIED] LIVES_AT +{len(assignments)}")
        still = s.run("MATCH (a:Agent) WHERE NOT (a)-[:LIVES_AT]->() RETURN count(a) AS n").single()['n']
        print(f"적용 후 LIVES_AT 없는 agent: {still}")
