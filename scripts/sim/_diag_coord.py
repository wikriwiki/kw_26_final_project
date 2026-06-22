"""누락 dong들의 좌표 확보 가능성 진단."""
import sys
sys.path.insert(0, 'scripts/neo4j_load')
from _common import driver_session

with open('output/sim_9d_baseline/agent_ids_7500.txt', encoding='utf-8') as f:
    ids7500 = [l.strip() for l in f if l.strip()]

with driver_session() as s:
    # 누락 agent들의 dong code 집합
    no_lives = s.run(
        "MATCH (a:Agent) WHERE a.id IN $ids AND NOT (a)-[:LIVES_AT]->() "
        "RETURN DISTINCT a.residence_dong_code_raw AS code",
        ids=ids7500).data()
    dongs = [r['code'] for r in no_lives]
    print(f"누락 dong 수: {len(dongs)}")
    print(f"dong codes: {dongs}")

    print("\n=== 각 dong별 좌표 소스 ===")
    for code in dongs:
        # 1. 이 dong에 ANY POI (any type)
        any_poi = s.run(
            "MATCH (p:POI) WHERE p.dong_code = $c RETURN count(p) AS n, avg(p.lon) AS lon, avg(p.lat) AS lat",
            c=code).single()
        # 2. residence POI
        res_poi = s.run(
            "MATCH (p:POI {type:'residence'}) WHERE p.dong_code = $c RETURN count(p) AS n",
            c=code).single()['n']
        # 3. Dong 노드 좌표
        dong_node = s.run(
            "MATCH (d:Dong {code:$c}) RETURN d.lat AS lat, d.lon AS lon, keys(d) AS k",
            c=code).single()
        dn = f"lat={dong_node['lat']},lon={dong_node['lon']}" if dong_node else "Dong노드없음"
        print(f"  {code}: POI(any)={any_poi['n']} (centroid lon={any_poi['lon']},lat={any_poi['lat']}) / res_poi={res_poi} / Dong노드: {dn}")

    # 전체 residence POI 좌표 범위 (nearest 대상 존재 확인)
    rng = s.run("MATCH (p:POI {type:'residence'}) RETURN count(p) AS n, min(p.lon) AS minlon, max(p.lon) AS maxlon").single()
    print(f"\n전체 residence POI: {rng['n']} (lon {rng['minlon']}~{rng['maxlon']})")
