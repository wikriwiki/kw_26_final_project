"""Day 0 초기 State 시드.

이유:
- Night Phase 2 Urgency 계산은 어제 State의 mood·fatigue를 읽음
- 시뮬 첫날엔 어제 State가 없으므로 시드 필요
- 또한 Stage 1 Dawn ② 어제 State 쿼리도 첫날 빈 결과 회피

생성:
  (:State {id, agent_id, day, balance, energy, yesterday_satisfaction,
           mood, fatigue, month_spent, policy_lifecycle})
  (:Agent)-[:HAS_STATE {day}]->(:State)
  모든 Agent에 대해 1개씩.

balance·energy는 페르소나 spending_level에 따라 차등 (선택).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session, bulk_run

# 시뮬 시작일 = Day 0의 다음날(t=1)부터 시뮬 시작.
# 시드는 t=0 = SIM_START - 1 일자로 생성.
DAY_ZERO = "2026-04-30"   # POC 시뮬 시작 전날 (다음 날부터 Dawn 진행)

# spending_level (1~10) → 초기 잔액·에너지 매핑
def initial_balance(level: int | None) -> int:
    if not level:
        return 1500000
    # 분위 1=50만, 10=500만 선형
    return int(500000 + (level - 1) * 500000)


def main():
    with driver_session() as s:
        # 모든 Agent의 id + spending_level 추출
        print("[fetch] Agent ids + spending_level ...")
        rows = []
        for r in s.run("""
            MATCH (a:Agent)
            RETURN a.id AS id, a.spending_level_wd AS lvl
        """):
            aid = r["id"]
            lvl = r["lvl"]
            rows.append({
                "id": f"{aid}_{DAY_ZERO}",
                "agent_id": aid,
                "day": DAY_ZERO,
                "balance": initial_balance(lvl),
                "energy": 0.8,
                "yesterday_satisfaction": 0.5,
                "mood": 0.5,         # 중립
                "fatigue": 0.3,      # 낮은 피로
                "month_spent": 0,
                "policy_lifecycle": "{}",   # JSON string
            })
        print(f"  agents: {len(rows)}")

        bulk_run(s, """
            UNWIND $batch AS x
            MATCH (a:Agent {id: x.agent_id})
            MERGE (s:State {id: x.id})
            ON CREATE SET
              s.agent_id = x.agent_id,
              s.day = date(x.day),
              s.balance = x.balance,
              s.energy = x.energy,
              s.yesterday_satisfaction = x.yesterday_satisfaction,
              s.mood = x.mood,
              s.fatigue = x.fatigue,
              s.month_spent = x.month_spent,
              s.policy_lifecycle = x.policy_lifecycle
            MERGE (a)-[:HAS_STATE {day: date(x.day)}]->(s)
        """, batch=rows, batch_size=2000)
        print(f"  + State (initial) x {len(rows)}")

    print("[done] initial State seeded.")


if __name__ == "__main__":
    main()
