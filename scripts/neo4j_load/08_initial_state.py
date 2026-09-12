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

import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session, bulk_run

# 시뮬 시작일 = Day 0의 다음날(t=1)부터 시뮬 시작.
# 시드는 t=0 = SIM_START - 1 일자로 생성.
# 실험별 시작일 지정: DAY_ZERO=2026-05-19 python 08_initial_state.py (기본 2026-05-24)
DAY_ZERO = os.environ.get("DAY_ZERO", "2026-05-24")
date.fromisoformat(DAY_ZERO)  # 형식 검증 (잘못된 값 조기 실패)

# 초기 잔액은 본인 소비 앵커 × BALANCE_DAYS 로 잡는다.
#
# [교정 이력] 이전 공식 `500000 + (level-1)*500000` 은 앵커와 무관한 임의 선형이라
# "버틸 일수"가 분위마다 15.5~33.9일로 벌어졌다. 1분위와 10분위가 먼저 말라붙는
# U자 패턴이 생겨 분위별 이질성 분석이 잔액 시드에 오염됐다(P012 200×28 런 실측).
# 앵커 비례로 바꾸면 모든 분위의 runway 가 같아진다.
#
# BALANCE_DAYS 기본 39 = 한 달(30일) × 여유 1.3. 관측 기간과 무관하게 고정해
# 런 길이가 잔액 수준을 바꾸지 않도록 한다(잔액은 프롬프트에 노출되므로
# 길이에 따라 달라지면 비교가 깨진다).
# 적립업종이 총지출에서 차지하는 몫 (dawn_context.SANGSAENG_BASE_RATIO 와 같은 값)
SANGSAENG_RATIO = float(os.environ.get("EXP_SANGSAENG_BASE_RATIO", "0.268"))
SEED_SANGSAENG = os.environ.get("EXP_SEED_SANGSAENG", "1") == "1"
BALANCE_DAYS = float(os.environ.get("EXP_BALANCE_DAYS", "39"))
LEGACY_BALANCE = os.environ.get("EXP_LEGACY_BALANCE", "0") == "1"


def anchor_daily(wd, we) -> float:
    """평일 5 · 주말 2 가중 일평균 소비 앵커."""
    try:
        w = float(wd or 0); e = float(we or 0)
    except (TypeError, ValueError):
        return 0.0
    if w <= 0 and e <= 0:
        return 0.0
    if w <= 0:
        w = e
    if e <= 0:
        e = w
    return (w * 5 + e * 2) / 7


def initial_balance(level: int | None, wd=None, we=None) -> int:
    # P010 재현용 옛 공식 (EXP_LEGACY_BALANCE=1)
    if LEGACY_BALANCE:
        if not level:
            return 1500000
        return int(500000 + (level - 1) * 500000)
    a = anchor_daily(wd, we)
    if a <= 0:
        return 1500000
    return int(round(a * BALANCE_DAYS))


def main():
    with driver_session() as s:
        # 모든 Agent의 id + spending_level 추출
        print("[fetch] Agent ids + spending_level ...")
        rows = []
        for r in s.run("""
            MATCH (a:Agent)
            RETURN a.id AS id, a.spending_level_wd AS lvl,
                   a.s_daily_wd AS wd, a.s_daily_we AS we
        """):
            aid = r["id"]
            lvl = r["lvl"]
            rows.append({
                "id": f"{aid}_{DAY_ZERO}",
                "agent_id": aid,
                "day": DAY_ZERO,
                "balance": initial_balance(lvl, r["wd"], r["we"]),
                "energy": 0.8,
                "yesterday_satisfaction": 0.5,
                "mood": 0.5,         # 중립
                "fatigue": 0.3,      # 낮은 피로
                "month_spent": 0,
                # 적립 누적을 월중 위치로 시드한다. Day0 이 월초가 아니면 그 사람은
                # 이미 그 달에 소비를 했고, 0 으로 시작하면 문턱까지 남은 금액이
                # 실제보다 커져 도달 불가로 읽힌다. 본인 앵커에서 환산하며
                # 무정책·정책 양쪽에 동일하게 적용된다.
                "sangsaeng_month_spent": int(round(
                    anchor_daily(r["wd"], r["we"]) * SANGSAENG_RATIO
                    * (date.fromisoformat(DAY_ZERO).day)
                )) if SEED_SANGSAENG else 0,
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
              s.sangsaeng_month_spent = x.sangsaeng_month_spent,
              s.policy_lifecycle = x.policy_lifecycle
            MERGE (a)-[:HAS_STATE {day: date(x.day)}]->(s)
        """, batch=rows, batch_size=2000)
        print(f"  + State (initial) x {len(rows)}")

    print("[done] initial State seeded.")


if __name__ == "__main__":
    main()
