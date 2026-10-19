// 5/20 KNOWS_POI 보정 — visited Memory 누락분의 단골화 집계를 시간순으로 끼워넣기.
// visited Memory (anchor zone:) 와 동일 조건. 시뮬 kill 후 1회만 실행 (멱등 아님 — 재실행 금지).
// last_visit / since 는 시간 비교로 정확히, avg_satisfaction 은 산술누적, affinity 는 EMA 근사.
MATCH (a:Agent)-[:HAS_PLAN {day: date('2026-05-20')}]->(:Plan)-[i:INCLUDES]->(poi:POI)
WHERE i.actual_satisfaction IS NOT NULL AND i.anchor STARTS WITH 'zone:'
MERGE (a)-[kp:KNOWS_POI]->(poi)
ON CREATE SET
  kp.since = date('2026-05-20'), kp.source = 'visited',
  kp.visit_count = 1, kp.avg_satisfaction = i.actual_satisfaction,
  kp.last_visit = date('2026-05-20'),
  kp.affinity = 0.3 + 0.4 * i.actual_satisfaction,
  kp.recent_visit_dates = [date('2026-05-20')]
ON MATCH SET
  kp.visit_count = coalesce(kp.visit_count, 0) + 1,
  kp.avg_satisfaction = (coalesce(kp.avg_satisfaction, 0.5) * coalesce(kp.visit_count, 0) + i.actual_satisfaction)
                         / (coalesce(kp.visit_count, 0) + 1),
  kp.last_visit = CASE WHEN kp.last_visit >= date('2026-05-20') THEN kp.last_visit ELSE date('2026-05-20') END,
  kp.since = CASE WHEN kp.since <= date('2026-05-20') THEN kp.since ELSE date('2026-05-20') END,
  kp.affinity = coalesce(kp.affinity, 0.5) * 0.7 + i.actual_satisfaction * 0.3,
  kp.recent_visit_dates =
    CASE WHEN date('2026-05-20') IN coalesce(kp.recent_visit_dates, [])
         THEN kp.recent_visit_dates
         ELSE coalesce(kp.recent_visit_dates, []) + [date('2026-05-20')] END
RETURN count(*) AS updated;
