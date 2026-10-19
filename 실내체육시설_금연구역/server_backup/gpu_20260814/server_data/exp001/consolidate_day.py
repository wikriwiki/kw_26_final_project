#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""구간 경계일 야간 정산 보정 — visited Memory + KNOWS_POI 를 함께 복구한다.

왜 필요한가:
  run_simulation 은 각 런의 첫날(day_idx==0)에 '전날 야간 정산'을 건너뛴다(의도된 동작).
  순차 전후 설계에서는 1구간 마지막 날이 그 대상이 되어, 하루치 학습이 통째로 사라진다.
  2단계에서 실제로 2025-07-20 이 그렇게 날아갔다 — 기억 0건, KNOWS_POI 갱신 0건,
  그날 처음 간 가게 21.7% 는 엣지 자체가 생성되지 않았다.

기존 recover_visited_memory.py 와의 차이:
  - KNOWS_POI 도 복구한다(그쪽이 단골·후보 점수에 직접 쓰이는 더 큰 손실이다)
  - 기억에 인터뷰용 필드(why/pick_why/trigger/spent/paid_policy/extra_spent)를 넣는다
  - last_visit 을 뒤로 되돌리지 않는다(이미 더 최근 방문이 있으면 유지)
  - 이미 정산된 날에 다시 돌려도 값이 어긋나지 않는다(멱등)

사용:  python3 consolidate_day.py 2025-07-20
"""
import sys
from pathlib import Path

sys.path.insert(0, '/data/exp001_repo/scripts/neo4j_load')
from _common import driver_session   # noqa: E402

DAY = sys.argv[1]

# plan_writer.py 의 야간 정산과 같은 로직. 멱등성을 위해 MERGE 와 조건부 SET 만 쓴다.
CYPHER = """
MATCH (a:Agent)-[:HAS_PLAN {day: date($day)}]->(:Plan)-[i:INCLUDES]->(poi:POI)
WHERE i.actual_satisfaction IS NOT NULL
  AND (i.anchor STARTS WITH 'zone:'
       OR (i.category IS NOT NULL AND NOT i.category IN ['집','직장']))
WITH a, i, poi,
     0.5 + 1.5 * i.actual_satisfaction AS importance,
     toString(coalesce(i.actual_spent, 0)) + '원 · [' + coalesce(i.trigger, '-') + '] ' +
       left(coalesce(i.reasoning, ''), 90) AS summary,
     'mem_vis_' + a.id + '_' + poi.id + '_' + $day + '_' + toString(i.order) AS mem_id
MERGE (m:Memory {id: mem_id})
  ON CREATE SET
    m.type = 'visited', m.day = date($day), m.importance = importance,
    m.summary = summary, m.satisfaction = i.actual_satisfaction,
    m.why = i.reasoning, m.pick_why = i.pick_reason, m.trigger = i.trigger,
    m.spent = coalesce(i.actual_spent, 0),
    m.paid_policy = (i.spent_from_policy IS NOT NULL
                     AND i.spent_from_policy <> '{}' AND i.spent_from_policy <> 'null'),
    m.extra_spent = i.extra_spent,
    m.category = coalesce(i.sub_category, i.category),
    m.recovered = true
MERGE (a)-[:REMEMBERS {day: date($day)}]->(m)
MERGE (m)-[:ABOUT_POI]->(poi)

MERGE (a)-[kp:KNOWS_POI]->(poi)
ON CREATE SET
  kp.since = date($day), kp.source = 'visited',
  kp.visit_count = 1, kp.avg_satisfaction = i.actual_satisfaction,
  kp.last_visit = date($day), kp.recent_visit_dates = [date($day)]
ON MATCH SET
  kp.visit_count = coalesce(kp.visit_count, 0) + 1,
  kp.avg_satisfaction = (coalesce(kp.avg_satisfaction, 0.5) * coalesce(kp.visit_count, 0)
                         + i.actual_satisfaction) / (coalesce(kp.visit_count, 0) + 1),
  // 이미 더 최근 방문이 기록돼 있으면 되돌리지 않는다
  kp.last_visit = CASE WHEN kp.last_visit IS NULL OR kp.last_visit < date($day)
                       THEN date($day) ELSE kp.last_visit END,
  kp.recent_visit_dates =
    [d IN coalesce(kp.recent_visit_dates, []) WHERE d <> date($day)
       AND duration.inDays(d, date($day)).days < 30] + [date($day)]
RETURN count(m) AS n
"""


def main():
    with driver_session() as s:
        before = s.run("""MATCH (m:Memory {type:'visited'}) WHERE m.day=date($d)
                          RETURN count(m) AS n""", d=DAY).single()['n']
        kp_before = s.run("""MATCH ()-[kp:KNOWS_POI]->() WHERE kp.last_visit=date($d)
                             RETURN count(kp) AS n""", d=DAY).single()['n']
        ev = s.run("""MATCH (:Plan {day: date($d)})-[i:INCLUDES]->()
                      WHERE i.actual_satisfaction IS NOT NULL
                      RETURN count(i) AS n""", d=DAY).single()['n']
        print(f"{DAY} 복구 전 — 이벤트 {ev:,} · 기억 {before:,} · KNOWS_POI(last_visit) {kp_before:,}")
        if ev == 0:
            print("  대상 없음"); return
        if before >= ev * 0.9 and kp_before > 0:
            print("  이미 정산됨 — 건너뜀"); return

        s.run(CYPHER, day=DAY).consume()

        after = s.run("""MATCH (m:Memory {type:'visited'}) WHERE m.day=date($d)
                         RETURN count(m) AS n""", d=DAY).single()['n']
        kp_after = s.run("""MATCH ()-[kp:KNOWS_POI]->() WHERE kp.last_visit=date($d)
                            RETURN count(kp) AS n""", d=DAY).single()['n']
        print(f"{DAY} 복구 후 — 기억 {after:,} (+{after-before:,}) · "
              f"KNOWS_POI {kp_after:,} (+{kp_after-kp_before:,})")


if __name__ == '__main__':
    main()
