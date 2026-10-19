"""시뮬 끝난 후 visited Memory 누락 일괄 복구.

원인:
  - day_idx==0에서 night_finalize_yesterday를 skip하는 코드 패턴 (의도된 동작이지만)
  - silent kill 후 resume 시 어제 visited Memory 생성 단계 누락 가능

이 스크립트는 각 시뮬일별로 Plan/INCLUDES는 있는데 Memory{type:'visited'}가
누락된 케이스를 anchor STARTS WITH 'zone:' 조건으로 MERGE 복구한다.

CLI:
  python scripts/sim/recover_visited_memory.py --start 2026-05-25 --days 4
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))

from _common import driver_session


# plan_writer.NIGHT_VISITED_CYPHER와 동일 로직 + 명시적 day 인자.
# anchor zone:으로 시작하는 외출만 visited Memory 생성.
RECOVER_CYPHER = """
MATCH (a:Agent)-[:HAS_PLAN {day: date($day)}]->(:Plan)-[i:INCLUDES]->(poi:POI)
WHERE i.actual_satisfaction IS NOT NULL
  AND i.anchor STARTS WITH 'zone:'
WITH a, i, poi,
     0.5 + 1.5 * i.actual_satisfaction AS importance,
     poi.name + '(' + coalesce(i.sub_category, i.category) + ') 방문, 만족도 ' +
       toString(round(i.actual_satisfaction * 100) / 100.0) AS summary,
     'mem_vis_' + a.id + '_' + poi.id + '_' + $day + '_' + toString(i.order) AS mem_id
MERGE (m:Memory {id: mem_id})
  ON CREATE SET
    m.type = 'visited',
    m.day = date($day),
    m.importance = importance,
    m.summary = summary,
    m.satisfaction = i.actual_satisfaction
MERGE (a)-[:REMEMBERS {day: date($day)}]->(m)
MERGE (m)-[:ABOUT_POI]->(poi)
RETURN count(DISTINCT m) AS n_memories
"""

COUNT_CYPHER = """
MATCH (m:Memory {type:'visited'}) WHERE m.day = date($day)
RETURN count(m) AS n
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--days", type=int, default=4)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    days = [(start + timedelta(days=i)).isoformat() for i in range(args.days)]

    with driver_session() as s:
        for d in days:
            before = s.run(COUNT_CYPHER, day=d).single()["n"]
            r = s.run(RECOVER_CYPHER, day=d).single()
            after = s.run(COUNT_CYPHER, day=d).single()["n"]
            print(f"[{d}] before={before:,} after={after:,} delta={after-before:+,}")


if __name__ == "__main__":
    main()
