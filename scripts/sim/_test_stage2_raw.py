"""Stage2 raw 응답 1건 캡처 — actual_satisfaction을 모델이 실제 내놓는지 확정."""
import sys
from datetime import date
sys.path.insert(0, 'scripts/sim')
sys.path.insert(0, 'scripts/neo4j_load')
from _common import driver_session
from dawn_context import build_dawn_context
from stage1_intent import call_stage1
import stage2_poi

# LIVES_AT 보유 agent 1명 선택
with driver_session() as s:
    aid = s.run("MATCH (a:Agent)-[:LIVES_AT]->() RETURN a.id AS id ORDER BY a.id LIMIT 1").single()['id']
print(f"테스트 agent: {aid}")

today = date(2026, 5, 25)
ctx = build_dawn_context(aid, today)
s1, m1 = call_stage1(aid, today, ctx=ctx)
print(f"Stage1 events: {len(s1.events)}")

# Stage2 verbose=True로 raw 출력
s2, cands, m2 = stage2_poi.call_stage2(aid, s1, ctx.persona, today, verbose=True)
print(f"\n=== Stage2 picks ({len(s2.picks)}) — actual_satisfaction 확인 ===")
for p in s2.picks[:10]:
    print(f"  order={p.order} poi={p.poi_id} spent={p.actual_spent} sat={p.actual_satisfaction}")
