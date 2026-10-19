"""정책 원장이 업종별 총액과 그 정책 사용처 몫을 함께 적는다 — P013 업종 묶음 효과용."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
import export_policy_daily_ledger as L  # noqa: E402

POLICY = "data/experiments/P013_v53_policy_20260926.json"


def test_big_mart_is_in_gross_but_not_in_eligible():
    states = [{"aid": "a", "online_spent": 0, "self_month_cumulative": 0,
               "grant_received": {"P013": 280000}, "grant_remaining": {"P013": 270000}}]
    spends = [
        {"aid": "a", "amt": 30000, "spent_from_policy": {}, "pname": "이마트 성수점",
         "sub": "슈퍼마켓", "l1": "마트", "upjong_l3": None, "pdong": "1120011400", "hdong": "1120011400"},
        {"aid": "a", "amt": 10000, "spent_from_policy": {"P013": 10000}, "pname": "행복슈퍼",
         "sub": "슈퍼마켓", "l1": "마트", "upjong_l3": None, "pdong": "1120011400", "hdong": "1120011400"},
        {"aid": "a", "amt": 8000, "spent_from_policy": {}, "pname": "김밥천국",
         "sub": "한식", "l1": "식사", "upjong_l3": None, "pdong": "1120011400", "hdong": "1120011400"},
    ]
    rows = L.aggregate_day(states, spends, roster=["a"], day="2020-05-11", arm="on",
                           policy_id="P013", policy_file=POLICY)
    r = rows[0]
    assert r["by_sub"] == {"슈퍼마켓": 40000, "한식": 8000}
    assert r["eligible_by_sub"] == {"슈퍼마켓": 10000, "한식": 8000}   # 대형마트 3만원은 빠진다
    assert r["eligible_offline_spent"] == 18000
    assert r["offline_spent"] == 48000
    assert sum(r["eligible_by_sub"].values()) == r["eligible_offline_spent"]
