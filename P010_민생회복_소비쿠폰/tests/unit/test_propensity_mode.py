"""Opt-in removes only propensity prior clipping; payment constraints remain."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/sim"))
import consumption  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    monkeypatch.delenv("EXP_PROPENSITY_MODE", raising=False)
    monkeypatch.setenv("EXP_PAYMENT_CHOICE", "1")
    monkeypatch.setattr(consumption, "EXP_PLAN_DRIVES_TOTAL", False)
    monkeypatch.setattr(consumption, "EXP_SPLIT_ANCHOR", False)
    monkeypatch.setattr(consumption, "ELIGIBLE_SHARE_SEOUL", 0.5)


def events():
    return [
        {"poi_id": "ELIGIBLE", "category": "식사", "actual_spent": 12_000,
         "coupon_eligible": True, "policy_spend": {"TEST": 12_000}},
        {"poi_id": "INELIGIBLE", "category": "마트", "actual_spent": 8_000,
         "coupon_eligible": False, "policy_spend": {}},
    ]


def apply(ev, *, p=0.9, income="상", cashback=False, balance=100_000,
          wallet=5_000, discounts=None):
    return consumption.apply_consumption_model(
        ev, daily=35_000, income_tier=income, tendency="standard",
        balance=balance, llm_propensity=p, cashback_active=cashback,
        restricted_envelopes=[{"pid": "TEST", "amount": wallet,
                               "require_poi_eligible": True}],
        instant_discount_specs=discounts,
    )


@pytest.mark.parametrize("p", [None, float("nan"), float("inf"), -float("inf"),
                              "not-a-number", True])
def test_invalid_passed_scalar_blocks_before_event_mutation(monkeypatch, p):
    monkeypatch.setenv("EXP_PROPENSITY_MODE", "llm_budget_only")
    ev = events()
    original = copy.deepcopy(ev)
    with pytest.raises(ValueError, match="finite daily_propensity"):
        apply(ev, p=p)
    assert ev == original
    with pytest.raises(ValueError, match="finite daily_propensity"):
        apply([], p=p)


def test_unknown_mode_fails_before_mutation(monkeypatch):
    monkeypatch.setenv("EXP_PROPENSITY_MODE", "llm_budgte_only")
    ev = events()
    original = copy.deepcopy(ev)
    with pytest.raises(ValueError, match="unknown EXP_PROPENSITY_MODE"):
        apply(ev)
    assert ev == original


@pytest.mark.parametrize("p,expected", [(-2, 0.0), (0, 0.0), (0.23456789, 0.23456789),
                                      (1, 1.0), (7, 1.0)])
def test_contract_boundaries_do_not_use_income_or_radius(monkeypatch, p, expected):
    monkeypatch.setenv("EXP_PROPENSITY_MODE", "llm_budget_only")
    for income in ("하", "중하", "중", "중상", "상", None):
        for band in (0.0, 0.12, 0.3):
            assert consumption.clamp_propensity(
                p, income, balance=10_000_000, daily_wd=30_000,
                tendency="과소비", band=band) == expected


def test_legacy_default_and_explicit_mode_have_identical_ledgers(monkeypatch):
    default_events = events()
    default = apply(default_events, p=None)
    monkeypatch.setenv("EXP_PROPENSITY_MODE", "legacy")
    explicit_events = events()
    explicit = apply(explicit_events, p=None)
    assert default_events == explicit_events
    assert default == explicit
    assert "propensity_mode" not in default
    assert consumption.clamp_propensity(None, "하", balance=0, daily_wd=35_000) == 0.9
    assert consumption.clamp_propensity(0.1, "하", balance=0, daily_wd=35_000) == 0.78
    assert consumption.clamp_propensity(0.95, "상", balance=0, daily_wd=35_000) == 0.7


def test_cashback_special_radius_removed_only_in_opt_in(monkeypatch):
    legacy_off = apply(events(), p=0.95, balance=0, wallet=0)
    legacy_on = apply(events(), p=0.95, cashback=True, balance=0, wallet=0)
    assert legacy_off["propensity"] == 0.7
    assert legacy_on["propensity"] == 0.88
    monkeypatch.setenv("EXP_PROPENSITY_MODE", "llm_budget_only")
    for cashback in (False, True):
        for income in ("하", "상"):
            meta = apply(events(), p=0.95, income=income, cashback=cashback)
            assert meta["propensity"] == 0.95
            assert meta["propensity_center"] == consumption.ANCHOR_PROPENSITY
            assert meta["propensity_mode"] == "llm_budget_only"
            assert meta["propensity_input_value"] == 0.95


@pytest.mark.parametrize("balance,wallet", [(0, 0), (0, 5_000), (1, 1),
                                           (4_000, 5_000), (100_000, 5_000)])
def test_cash_policy_and_transaction_ledgers_conserve_budget(monkeypatch, balance, wallet):
    monkeypatch.setenv("EXP_PROPENSITY_MODE", "llm_budget_only")
    ev = events()
    meta = apply(ev, balance=balance, wallet=wallet)
    offline = sum(e["actual_spent"] for e in ev)
    funded = sum(sum(e.get("policy_spend", {}).values()) for e in ev)
    assert offline == meta["today_total"]
    assert funded == meta["policy_spend_allocated_total"]
    assert 0 <= funded <= wallet
    assert ev[1]["policy_spend"] == {}
    assert all(sum(e.get("policy_spend", {}).values()) <= e["actual_spent"] for e in ev)
    own_paid = offline + meta["online_total"] - funded - meta["instant_discount_total"]
    assert 0 <= own_paid <= balance
    assert meta["today_total_incl_online"] == offline + meta["online_total"]
    assert meta["mechanical_policy_uplift"] == 0


def test_low_propensity_does_not_claim_to_disable_existing_plan_floor(monkeypatch):
    monkeypatch.setenv("EXP_PROPENSITY_MODE", "llm_budget_only")
    meta = apply(events(), p=0.0, wallet=0)
    assert meta["propensity"] == 0.0
    assert meta["anchor_total"] == 0
    assert meta["today_total_incl_online"] == meta["planned_total"] == 20_000


def test_instant_discount_and_chosen_wallet_keep_cash_ledger_conservation(monkeypatch):
    from instant_discount import active_rate_discounts

    monkeypatch.setenv("EXP_PROPENSITY_MODE", "llm_budget_only")
    monkeypatch.setattr(consumption, "ELIGIBLE_SHARE_SEOUL", 1.0)
    ev = events()
    ev[0]["sub_category"] = "한식"
    specs = active_rate_discounts([{
        "id": "DISCOUNT", "type": "sector_voucher",
        "sectors": {"food": {"mode": "rate", "rate": 0.2, "cap": 2_000}},
        "eligibility": {"mode": "include", "include": {"subs": ["한식"]}},
    }])
    meta = apply(ev, balance=8_000, wallet=5_000, discounts=specs)
    gross = sum(e["actual_spent"] for e in ev)
    funded = sum(sum(e.get("policy_spend", {}).values()) for e in ev)
    discount = sum(sum(e.get("instant_discount", {}).values()) for e in ev)
    assert 0 < discount <= 2_000
    assert discount == meta["instant_discount_total"]
    assert funded <= 5_000
    assert 0 <= gross + meta["online_total"] - funded - discount <= 8_000
    assert ev[1]["policy_spend"] == ev[1]["instant_discount"] == {}
    assert all(sum(e.get("policy_spend", {}).values())
               + sum(e.get("instant_discount", {}).values()) <= e["actual_spent"]
               for e in ev)


def test_no_commerce_records_zero_after_valid_contract(monkeypatch):
    monkeypatch.setenv("EXP_PROPENSITY_MODE", "llm_budget_only")
    meta = apply([], p=0.3)
    assert meta["today_total_incl_online"] == 0
    assert meta["propensity"] == 0.3
    assert meta["reason"] == "no_commerce"


def test_upstream_nonfinite_coercion_is_not_claimed_as_raw_validation():
    # Stage1Intent._clip_propensity currently uses this expression. The new
    # downstream guard cannot tell this coerced number from an actual 1.0.
    assert max(0.0, min(1.0, float("nan"))) == 1.0
    assert not math.isfinite(float("nan"))
