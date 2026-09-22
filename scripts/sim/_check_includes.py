import sys
sys.path.insert(0, 'scripts/neo4j_load')
from _common import driver_session

with driver_session() as s:
    rows = s.run(
        "MATCH (a:Agent)-[:HAS_PLAN {day: date('2026-05-25')}]->(p:Plan)-[i:INCLUDES]->(poi:POI) "
        "RETURN a.id AS aid, poi.name AS poi, poi.type AS ptype, "
        "i.actual_spent AS spent, i.actual_satisfaction AS sat, keys(i) AS ikeys LIMIT 8"
    ).data()
    print(f"INCLUDES 샘플 {len(rows)}건:")
    for r in rows:
        print(f"  {r['aid']} -> {r['poi']}({r['ptype']}) spent={r['spent']} sat={r['sat']}")
    if rows:
        print(f"INCLUDES keys: {rows[0]['ikeys']}")

    stat = s.run(
        "MATCH (:Plan {day: date('2026-05-25')})-[i:INCLUDES]->() "
        "RETURN count(i) AS total, "
        "sum(CASE WHEN i.actual_satisfaction IS NOT NULL THEN 1 ELSE 0 END) AS with_sat, "
        "sum(CASE WHEN i.actual_spent IS NOT NULL THEN 1 ELSE 0 END) AS with_spent, "
        "avg(i.actual_satisfaction) AS avg_sat, avg(i.actual_spent) AS avg_spent"
    ).single()
    print(f"전체 INCLUDES: {stat['total']}")
    print(f"  actual_satisfaction 있음: {stat['with_sat']} (avg={stat['avg_sat']})")
    print(f"  actual_spent 있음: {stat['with_spent']} (avg={stat['avg_spent']})")
