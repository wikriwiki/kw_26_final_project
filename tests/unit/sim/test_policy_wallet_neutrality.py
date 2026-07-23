"""정책지갑이 소비를 강제하지 않고 결제수단으로만 작동하는지 검증."""
from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SIM_DIR = ROOT / "scripts" / "sim"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from consumption import (  # noqa: E402
    ANCHOR_PROPENSITY,
    apply_consumption_model,
    available_today,
    spend_today,
)
from plan_writer import aggregate_policy_spend, validate_policy_spend  # noqa: E402


def _events(policy_spend: dict | None = None, *, eligible: bool = True) -> list[dict]:
    return [
        {
            "category": "식사",
            "poi_id": "C_MEAL",
            "actual_spent": 18_000,
            "price_factor": 1.0,
            "coupon_eligible": eligible,
            "policy_spend": copy.deepcopy(policy_spend or {}),
            "pick_reason": "오늘 일정에 맞는 평소 식사",
        },
        {
            "category": "카페",
            "poi_id": "C_CAFE",
            "actual_spent": 7_000,
            "price_factor": 1.0,
            "coupon_eligible": True,
            "policy_spend": {},
            "pick_reason": "오후 휴식",
        },
    ]


def _apply(
    events: list[dict],
    *,
    with_policy: bool,
    propensity: float | None = None,
    balance: int = 800_000,
) -> dict:
    return apply_consumption_model(
        events,
        daily=35_000,
        income_tier="중",
        tendency="표준형",
        balance=balance,
        grant_avail=None,
        llm_propensity=propensity,
        restricted_envelopes=(
            [{
                "pid": "P010",
                "amount": 150_000,
                "require_poi_eligible": True,
            }]
            if with_policy
            else None
        ),
    )


def test_policy_balance_does_not_expand_daily_consumption_budget():
    assert available_today(35_000, 0) == available_today(35_000, 150_000)
    assert spend_today(ANCHOR_PROPENSITY, 35_000, 0)["total"] == 35_000
    with_grant = spend_today(ANCHOR_PROPENSITY, 35_000, 150_000)
    assert with_grant["total"] == 35_000
    assert with_grant["grant_part"] == 0


def test_same_plan_policy_on_off_has_same_total_and_only_payment_differs():
    off_events = _events()
    on_events = _events({"P010": 10_000})

    off_meta = _apply(off_events, with_policy=False)
    on_meta = _apply(on_events, with_policy=True)

    assert off_meta["today_total"] == on_meta["today_total"]
    assert sum(e["actual_spent"] for e in off_events) == sum(
        e["actual_spent"] for e in on_events
    )
    assert aggregate_policy_spend(off_events) == {}
    assert aggregate_policy_spend(on_events) == {"P010": 10_000}
    assert on_meta["mechanical_policy_uplift"] == 0


def test_policy_can_increase_consumption_when_agent_changes_plan():
    off_events = _events()
    on_events = _events({"P010": 10_000})
    on_events[0]["actual_spent"] = 30_000
    on_events[1]["actual_spent"] = 12_000

    off_meta = _apply(off_events, with_policy=False)
    on_meta = _apply(on_events, with_policy=True)

    assert on_meta["planned_total"] > off_meta["planned_total"]
    assert on_meta["today_total"] > off_meta["today_total"]


def test_policy_can_increase_consumption_through_agent_propensity_choice():
    off_events = _events()
    on_events = _events({"P010": 5_000})

    off_meta = _apply(off_events, with_policy=False, propensity=0.62)
    on_meta = _apply(on_events, with_policy=True, propensity=0.86)

    assert on_meta["day_multiplier"] > off_meta["day_multiplier"]
    assert on_meta["today_total"] > off_meta["today_total"]


def test_selected_policy_payment_can_relax_liquidity_constraint():
    off_events = _events()
    on_events = _events({"P010": 10_000})

    off_meta = _apply(off_events, with_policy=False, balance=10_000)
    on_meta = _apply(on_events, with_policy=True, balance=10_000)

    assert off_meta["today_total"] == 10_000
    assert on_meta["selected_policy_liquidity"] == 10_000
    assert on_meta["today_total"] == 20_000


def test_eligible_store_does_not_force_policy_payment():
    events = _events()
    meta = _apply(events, with_policy=True)

    assert all(e["policy_spend"] == {} for e in events)
    assert meta["envelope_requested"]["P010"] == 0
    assert meta["envelope_eligible_events"]["P010"] == 2


def test_validator_never_autofills_payment_from_reason_text():
    events = _events()
    events[0]["pick_reason"] = "지원금 사용 가능 매장이지만 오늘은 개인 결제를 선택"

    corrected = validate_policy_spend(
        events,
        policy_remaining={"P010": 150_000},
        restricted_pids={"P010"},
    )

    assert corrected == 0
    assert aggregate_policy_spend(events) == {}


def test_validator_only_clamps_eligibility_transaction_and_wallet_limits():
    events = [
        {
            "category": "식사",
            "poi_id": "C_OK",
            "actual_spent": 12_000,
            "coupon_eligible": True,
            "policy_spend": {"P010": 20_000},
        },
        {
            "category": "쇼핑",
            "poi_id": "C_NO",
            "actual_spent": 30_000,
            "coupon_eligible": False,
            "policy_spend": {"P010": 30_000},
        },
    ]

    corrected = validate_policy_spend(
        events,
        policy_remaining={"P010": 15_000},
        restricted_pids={"P010"},
    )

    assert corrected == 2
    assert events[0]["policy_spend"] == {"P010": 12_000}
    assert events[1]["policy_spend"] == {}
    assert aggregate_policy_spend(events) == {"P010": 12_000}


def test_zero_payment_choice_does_not_geometrically_exhaust_wallet_in_seven_days():
    remaining = 150_000
    for _ in range(7):
        events = _events()
        _apply(events, with_policy=True)
        validate_policy_spend(
            events,
            policy_remaining={"P010": remaining},
            restricted_pids={"P010"},
        )
        remaining -= aggregate_policy_spend(events).get("P010", 0)

    assert remaining == 150_000
