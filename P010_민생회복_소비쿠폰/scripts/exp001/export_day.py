#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""완료된 하루치 결제원장을 내보낸다 — 시뮬을 멈추지 않는 읽기 전용 작업.

그래프 덤프는 Neo4j를 정지해야 하므로 매일 뜰 수 없다. 대신 결제원장을 매일 빼두면
그래프에 무슨 일이 생겨도 32개 지표를 전부 재현할 수 있다
(1단계에서 FINAL 그래프를 잃고도 원장으로 살아남은 경로).

사용:  python3 export_day.py <RUN> <YYYY-MM-DD>
출력:  /data/exp001/out_<RUN>/events_<DAY>.jsonl
"""
import sys, json, os

sys.path.insert(0, '/data/exp001_repo/scripts/neo4j_load')
from _common import driver_session   # noqa: E402

RUN, DAY = sys.argv[1], sys.argv[2]
OUT = f'/data/exp001/out_{RUN}'
path = f'{OUT}/events_{DAY}.jsonl'
tmp = path + '.part'

n = 0
with driver_session() as s, open(tmp, 'w', encoding='utf-8') as fh:
    rows = s.run("""
        MATCH (pl:Plan {day: date($dy)})-[i:INCLUDES]->(p:POI)
        WHERE coalesce(i.actual_spent,0) > 0
        RETURN pl.day_type AS day_type, i.category AS l1, i.sub_category AS sub,
               i.actual_spent AS amt, i.spent_from_policy AS sp, i.extra_spent AS ex,
               i.would_buy_anyway AS wba, i.coupon_eligible AS elig,
               i.trigger AS trigger, i.pick_factor AS pick_factor""", dy=DAY)
    for r in rows:
        d = dict(r); d['day'] = DAY
        fh.write(json.dumps(d, ensure_ascii=False) + '\n')
        n += 1
os.replace(tmp, path)          # 부분 파일이 완성본으로 오인되지 않게 원자적 교체
print(f"{RUN} {DAY}: 결제 {n:,}건 → {path} ({os.path.getsize(path)/1e6:.1f} MB)")
