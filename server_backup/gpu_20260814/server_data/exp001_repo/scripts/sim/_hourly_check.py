import sys, json
from collections import Counter
sys.path.insert(0, 'scripts/neo4j_load')
from _common import driver_session

DAYS = ['2026-05-25','2026-05-26','2026-05-27','2026-05-28','2026-05-29',
        '2026-05-30','2026-05-31','2026-06-01','2026-06-02']

with driver_session() as s:
    npol = s.run("MATCH (p:Policy) RETURN count(p) AS c").single()['c']
    print(f"(5) Policy 노드: {npol} {'✓PASS' if npol==0 else '⚠️정책 존재'}")
    print()
    print("(4) Neo4j 적재 일별:")
    for d in DAYS:
        r = s.run("""
            MATCH (:Plan {day: date($d)})-[i:INCLUDES]->()
            WITH count(i) AS tot,
                 sum(CASE WHEN i.actual_satisfaction IS NOT NULL THEN 1 ELSE 0 END) AS sat_n,
                 avg(i.actual_satisfaction) AS avg_s,
                 sum(coalesce(i.actual_spent,0)) AS spent
            RETURN tot, sat_n, avg_s, spent
        """, d=d).single()
        if r['tot'] > 0:
            sat_pct = r['sat_n'] * 100 / r['tot']
            avg_s = r['avg_s'] if r['avg_s'] else 0
            print(f"  {d}: INCLUDES={r['tot']}, sat 적재={r['sat_n']}({sat_pct:.1f}%), avg_sat={avg_s:.3f}, total_spent={r['spent']:,}")
    # KNOWS_POI · Memory 누적
    kp = s.run("MATCH ()-[r:KNOWS_POI]->() RETURN count(r) AS c").single()['c']
    mem = s.run("MATCH (m:Memory) RETURN count(m) AS c").single()['c']
    conv = s.run("MATCH (c:Conversation) RETURN count(c) AS c").single()['c']
    print(f"  KNOWS_POI 총: {kp:,} / Memory: {mem:,} / Conversation: {conv:,}")

# (6) jsonl 환각
import glob
print("\n(6) 환각 지표 (jsonl):")
for fp in sorted(glob.glob(r'C:\Users\Administrator\sim_output_9d\metrics\day_*.jsonl')):
    rows = []
    with open(fp, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    st = Counter(r.get('status') for r in rows)
    ok = [r for r in rows if r.get('status') == 'ok']
    psc = sum(r.get('policy_spend_corrected', 0) or 0 for r in ok)
    s2fb = sum(r.get('s2_fallback', 0) or 0 for r in ok)
    halluc = sum(r.get('hallucinations', 0) or 0 for r in ok)
    day = fp.split('day_')[-1].replace('.jsonl','')
    print(f"  {day}: total={len(rows)} status={dict(st)} policy_spend_corrected={psc} s2_fallback={s2fb} hallucinations={halluc}")
