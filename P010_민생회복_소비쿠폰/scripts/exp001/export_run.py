#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""런의 결제 이벤트를 Neo4j에서 jsonl로 내보낸다.

exp_run.sh 는 시작 시 덤프를 재적재하므로, 다음 런을 띄우기 전에 반드시 실행해야
현재 런의 업종·품목 지표(B·D1·D2·E)를 나중에 다시 계산할 수 있다.

사용:  python3 export_run.py <RUN_NAME>
출력:  /data/exp001/out_<RUN>/events.jsonl  (+ poi_summary.json)
"""
import sys, json, glob, os

sys.path.insert(0, '/data/exp001_repo/scripts/neo4j_load')
from _common import driver_session   # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else 'FINAL'
OUT = f'/data/exp001/out_{RUN}'

days = [f.split('day_')[1][:-6]
        for f in sorted(glob.glob(f'{OUT}/metrics/day_*.jsonl'))]
if not days:
    sys.exit(f"{RUN}: metrics 없음")

path = f'{OUT}/events.jsonl'
n = 0
with driver_session() as s, open(path, 'w', encoding='utf-8') as fh:
    for dy in days:
        rows = s.run("""
            MATCH (pl:Plan {day: date($dy)})-[i:INCLUDES]->(p:POI)
            WHERE coalesce(i.actual_spent,0) > 0
            RETURN pl.day_type AS day_type, i.category AS l1,
                   i.sub_category AS sub, i.actual_spent AS amt,
                   i.spent_from_policy AS sp, i.extra_spent AS ex,
                   i.would_buy_anyway AS wba, i.coupon_eligible AS elig,
                   p.adm_cd AS dong""", dy=dy)
        for r in rows:
            d = dict(r)
            d['day'] = dy
            fh.write(json.dumps(d, ensure_ascii=False) + '\n')
            n += 1
    poi = s.run("""MATCH (p:POI) RETURN count(p) AS n,
                   sum(CASE WHEN p.coupon_eligible THEN 1 ELSE 0 END) AS elig
                """).single()
    json.dump({'poi_total': poi['n'], 'poi_eligible': poi['elig']},
              open(f'{OUT}/poi_summary.json', 'w'), ensure_ascii=False)

print(f"{RUN}: {len(days)}일 · 결제 {n:,}건 → {path} "
      f"({os.path.getsize(path)/1e6:.1f} MB)")
