"""MPC 자기보고 비율의 분모는 최종 정책결제 원장이어야 한다."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/sim"))
from consumption import settled_mpc_measure  # noqa: E402
from plan_writer import validate_policy_spend  # noqa: E402


def test_최종_정책결제액으로_건별_신규분을_가중한다():
    events = [
        {"actual_spent": 100, "desired_spent": 100,
         "policy_spend": {"P010": 20}, "extra_spent": 50},
        {"actual_spent": 100, "desired_spent": 100,
         "policy_spend": {"P010": 80}, "extra_spent": 0},
    ]
    result = settled_mpc_measure(events, [100, 100])
    assert result["share"] == pytest.approx(0.1)
    assert result["paid_won"] == 100
    assert result["coverage"] == 1


def test_줄어든_결제나_응답누락은_임의의_신규소비로_채우지_않는다():
    events = [
        {"actual_spent": 100, "desired_spent": 100,
         "policy_spend": {"P010": 20}, "would_buy_anyway": False},
        {"actual_spent": 50, "desired_spent": 100,
         "policy_spend": {"P010": 50}, "extra_spent": 100},
        {"actual_spent": 30, "desired_spent": 30,
         "policy_spend": {"P010": 30}, "extra_spent": None,
         "would_buy_anyway": None},
    ]
    result = settled_mpc_measure(events, [100, 100, 30])
    assert result["share"] is None
    assert result["lower"] == pytest.approx(0.2)
    assert result["upper"] == pytest.approx(1.0)
    assert result["unresolved_won"] == 80
    assert result["coverage"] == pytest.approx(0.2)


def test_정책결제가_없으면_비율을_정의하지_않는다():
    result = settled_mpc_measure(
        [{"actual_spent": 100, "policy_spend": {}}], [100])
    assert result["share"] is None
    assert result["paid_won"] == 0


def test_원금_행수가_어긋나면_일부_거래만으로_완전한_비율을_만들지_않는다():
    events = [{"actual_spent": 100, "desired_spent": 100,
               "policy_spend": {"P010": 60}, "extra_spent": 40}]
    result = settled_mpc_measure(events, [])
    assert result["share"] is None
    assert result["unresolved_won"] == 60
    assert (result["lower"], result["upper"]) == (0, 1)


def test_사용처_잔액_검증으로_결제액이_줄면_그_금액으로_가중한다():
    events = [{"category": "쇼핑", "poi_id": "C_1", "actual_spent": 100,
               "desired_spent": 100, "policy_spend": {"P010": 100},
               "coupon_eligible": True, "extra_spent": 25}]
    assert validate_policy_spend(
        events, policy_remaining={"P010": 40}, restricted_pids={"P010"}) == 1
    result = settled_mpc_measure(events, [100])
    assert result["paid_won"] == 40
    assert result["share"] == pytest.approx(0.25)


def test_실행기에서_최종_검증_후에_mpc를_계산한다():
    source = (ROOT / "scripts/sim/run_simulation.py").read_text(encoding="utf-8")
    assert source.index("policy_spend_corrected = validate_policy_spend(") \
        < source.index("mpc = settled_mpc_measure(") \
        < source.index("today_policy_spend = aggregate_policy_spend(")
