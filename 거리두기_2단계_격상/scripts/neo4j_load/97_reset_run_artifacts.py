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

## !! Policy 를 지운다 — 순서를 뒤집으면 정책 없는 런이 조용히 돈다 !!

**정책을 먼저 적재하고 이 스크립트를 뒤에 돌리면 그 정책이 지워진다.** 오류도 안 나고
적재 로그에는 "적재 완료" 가 찍혀 있어서, 런이 끝날 때까지 모른다. 그렇게 세 런을
잃었다 (2026-09-25 확인):

    p013_ruler        12일 × 700명 — 지원금 0원, "정책 반응 +6.36%" 는 예열 드리프트
    p012_28d 첫 시도   18일 × 500명 — 성향에 정책 단계가 없고 policy_hits 모든 날 0
    scope_fact_round  P090 위약 — 같은 순서 결함

위 "이후 순서" 가 처음부터 맞게 적혀 있었다. 러너 셋이 그것을 어겼다.

`--keep-policy` 를 쓰면 Policy 를 남긴다 — 정책을 이미 적재한 뒤에 부를 때 안전하다.

사용: python scripts/neo4j_load/97_reset_run_artifacts.py [--dry-run] [--keep-policy]
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
    ap.add_argument("--keep-policy", action="store_true",
                    help="Policy 를 남긴다 — 정책을 **이미 적재한 뒤** 부를 때 쓴다")
    args = ap.parse_args()

    # Policy 를 지우는 것이 이 스크립트의 함정이다. 무엇을 할지 **먼저 말한다.**
    todo = [(n, q) for n, q in DELETES
            if not (args.keep_policy and n == "Policy")]
    with driver_session() as s:
        r = s.run(COUNTS).single()
        print("대상 현황: Plan=%s Memory=%s Conversation=%s State=%s Policy=%s "
              "방문이력 KNOWS_POI=%s" % tuple(r))
        if args.keep_policy:
            print("  --keep-policy: **Policy 는 남긴다** (%s개)" % r[4])
        else:
            print("  ** Policy 도 지운다 (%s개). 정책 적재는 이 다음에 한다 —"
                  " 먼저 적재했다면 --keep-policy 를 쓰라" % r[4])
        if args.dry_run:
            print("[dry-run] 변경 없음")
            return
        for name, q in todo:
            s.run(q)
            print(f"  삭제 완료: {name}")
        s.run(RESET_KP)
        print("  리셋 완료: KNOWS_POI 방문 카운터 (관계 보존)")
        r2 = s.run(COUNTS).single()
        # Policy 를 남기기로 했으면 그 칸은 0 이 아니어도 된다.
        expect_zero = [v for i, v in enumerate(r2)
                       if not (args.keep_policy and i == 4)]
        assert all(int(v or 0) == 0 for v in expect_zero), f"잔존 산출물: {tuple(r2)}"
        print("✔ clean Day0 베이스 확인 (런 산출물 0%s)"
              % (" · Policy %s개 보존" % r2[4] if args.keep_policy else ""))
        print("다음: DAY_ZERO=<시작-1일> python scripts/neo4j_load/08_initial_state.py")


if __name__ == "__main__":
    main()
