"""누락 dong의 gu-prefix 커버리지 — 코드 불일치 vs 특정 dong 누락 판별."""
import sys
sys.path.insert(0, 'scripts/neo4j_load')
from _common import driver_session

missing_dongs = ['11740520', '11305590', '11305630', '11305620', '11230536',
                 '11305610', '11305606', '11305600', '11680740']

with driver_session() as s:
    print("=== gu-prefix(앞5자리)별 residence POI 커버리지 ===")
    seen = set()
    for code in missing_dongs:
        gu5 = code[:5]
        if gu5 in seen:
            continue
        seen.add(gu5)
        # 이 gu prefix로 시작하는 dong_code의 residence POI
        res = s.run(
            "MATCH (p:POI {type:'residence'}) WHERE p.dong_code STARTS WITH $g "
            "RETURN count(p) AS n, count(DISTINCT p.dong_code) AS dongs, avg(p.lon) AS lon, avg(p.lat) AS lat",
            g=gu5).single()
        # 이 gu 이름
        gu_name = s.run(
            "MATCH (a:Agent) WHERE a.residence_dong_code_raw STARTS WITH $g "
            "RETURN a.residence_gu AS gu LIMIT 1", g=gu5).single()
        gn = gu_name['gu'] if gu_name else '?'
        print(f"  prefix {gu5} ({gn}): residence POI {res['n']} across {res['dongs']} dongs, centroid lon={res['lon']},lat={res['lat']}")

    # 전체 POI dong_code 중 11305로 시작하는 것 (강북구 추정) — 다른 코드로 들어있나?
    print("\n=== 강북구 POI는 어떤 dong_code로? (gu name=강북구 agent의 코드 vs POI 코드) ===")
    # 강북구 agent들이 LIVES_AT 보유한 경우 그 POI의 dong_code 확인
    gb = s.run(
        "MATCH (a:Agent {residence_gu:'강북구'})-[:LIVES_AT]->(p:POI) "
        "RETURN DISTINCT p.dong_code AS code LIMIT 10").data()
    print(f"  강북구 agent의 LIVES_AT POI dong_code 샘플: {[r['code'] for r in gb]}")
    # 강북구 residence_dong_code_raw 샘플
    gb2 = s.run(
        "MATCH (a:Agent {residence_gu:'강북구'}) RETURN DISTINCT a.residence_dong_code_raw AS code LIMIT 12").data()
    print(f"  강북구 agent residence_dong_code_raw 샘플: {sorted(r['code'] for r in gb2)}")
