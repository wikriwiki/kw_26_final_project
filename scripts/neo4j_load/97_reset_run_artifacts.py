# -*- coding: utf-8 -*-
"""런 산출물 초기화 — 포스트런 덤프를 'clean Day0' 상태로 되돌린다.

용도: 새 실험(EXP-xxx) 전, 이전 런이 담긴 덤프(예: neo4j_3day_p009_*.dump)를 로드한 뒤
      시뮬이 만든 노드/관계만 지워 Day0 베이스(Agent·POI·Category·Dong·KNOWS 등)만 남긴다.

지우는 것:  Plan(+HAS_PLAN/INCLUDES) · Memory(+REMEMBERS/ABOUT_POI) ·
            Conversation(+PARTICIPATES_IN/MENTIONS_POI) · State(+HAS_STATE) · Policy(+엣지)
리셋하는 것: KNOWS_POI 방문 카운터(visit_count·avg_satisfaction·last_visit·recent_visit_dates)
            — 관계 자체(초기 인지)는 보존
남기는 것:  Agent·POI·Category·District/Dong·KNOWS(지인)·LIVES_AT/WORKS_AT/IN_*

이후 순서: 08_initial_state.py(DAY_ZERO 환경변수로 시드) → 10_load_grant_policy.py(실험 정책)

사용: python scripts/neo4j_load/97_reset_run_artifacts.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

COUNTS = """
OPTIONAL MATCH (p:Plan)  WITH count(p) AS plans
OPTIONAL MATCH (m:Memory) WITH plans, count(m) AS memories
OPTIONAL MATCH (c:Conversation) WITH plans, memories, count(c) AS convs
OPTIONAL MATCH (s:State) WITH plans, memories, convs, count(s) AS states
OPTIONAL MATCH (pol:Policy) WITH plans, memories, convs, states, count(pol) AS policies
OPTIONAL MATCH ()-[kp:KNOWS_POI]->() WHERE coalesce(kp.visit_count,0) > 0
RETURN plans, memories, convs, states, policies, count(kp) AS visited_kp
"""

# 대량 삭제는 배치 트랜잭션 (Neo4j 5.x)
DELETES = [
    ("Plan(+INCLUDES)",   "MATCH (p:Plan) CALL { WITH p DETACH DELETE p } IN TRANSACTIONS OF 10000 ROWS"),
    ("Memory",            "MATCH (m:Memory) CALL { WITH m DETACH DELETE m } IN TRANSACTIONS OF 10000 ROWS"),
    ("Conversation",      "MATCH (c:Conversation) CALL { WITH c DETACH DELETE c } IN TRANSACTIONS OF 10000 ROWS"),
    ("State",             "MATCH (s:State) CALL { WITH s DETACH DELETE s } IN TRANSACTIONS OF 10000 ROWS"),
    ("Policy",            "MATCH (pol:Policy) CALL { WITH pol DETACH DELETE pol } IN TRANSACTIONS OF 10000 ROWS"),
]
RESET_KP = """
MATCH ()-[kp:KNOWS_POI]->()
CALL { WITH kp
  SET kp.visit_count = 0,
      kp.avg_satisfaction = null,
      kp.last_visit = null,
      kp.recent_visit_dates = null
} IN TRANSACTIONS OF 50000 ROWS
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="삭제 없이 대상 카운트만")
    args = ap.parse_args()

    with driver_session() as s:
        r = s.run(COUNTS).single()
        print("대상 현황: Plan=%s Memory=%s Conversation=%s State=%s Policy=%s "
              "방문이력 KNOWS_POI=%s" % tuple(r))
        if args.dry_run:
            print("[dry-run] 변경 없음")
            return
        for name, q in DELETES:
            s.run(q)
            print(f"  삭제 완료: {name}")
        s.run(RESET_KP)
        print("  리셋 완료: KNOWS_POI 방문 카운터 (관계 보존)")
        r2 = s.run(COUNTS).single()
        assert all(int(v or 0) == 0 for v in r2), f"잔존 산출물: {tuple(r2)}"
        print("✔ clean Day0 베이스 확인 (런 산출물 0)")
        print("다음: DAY_ZERO=<시작-1일> python scripts/neo4j_load/08_initial_state.py")


if __name__ == "__main__":
    main()
