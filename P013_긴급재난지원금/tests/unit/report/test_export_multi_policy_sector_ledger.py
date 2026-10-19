from __future__ import annotations

import pytest

from scripts.report.export_multi_policy_sector_ledger import aggregate_day


def test_category_export_preserves_zero_spender_and_reconciles_policy_payment():
    states = [{"aid": "a", "online_spent": 30},
              {"aid": "b", "online_spent": 0}]
    spends = [{"aid": "a", "amt": 70, "sub": "한식", "l1": "식사",
               "spent_from_policy": '{"P010": 20}', "sangsaeng_eligible": True},
              {"aid": "a", "amt": 5, "sub": None, "l1": None,
               "spent_from_policy": None, "sangsaeng_eligible": False}]
    got = aggregate_day(states, spends, roster=["a", "b"], day="2025-07-21",
                        arm="on", policy_id="P010")
    assert len(got) == 2
    assert got[0]["total_spent"] == 105
    assert got[0]["by_sub"] == {"한식": 70}
    assert got[0]["funded_by_sub"] == {"한식": 20}
    assert got[0]["policy_funded_won"] == 20
    assert got[0]["unclassified_won"] == 5
    assert got[1]["total_spent"] == 0
    assert got[1]["by_sub"] == {}


def test_category_export_splits_each_sector_into_eligible_and_excluded():
    """'제외업종 중 유통'(KDI) 은 업종별 적립/제외 분해 없이는 셀 수 없다."""
    spends = [{"aid": "a", "amt": 40, "sub": "편의점", "l1": "유통",
               "spent_from_policy": None, "sangsaeng_eligible": True},
              {"aid": "a", "amt": 60, "sub": "할인점", "l1": "유통",
               "spent_from_policy": None, "sangsaeng_eligible": False}]
    got = aggregate_day([{"aid": "a", "online_spent": 0}], spends,
                        roster=["a"], day="2021-10-01", arm="off", policy_id=None)
    assert got[0]["by_l1"] == {"유통": 100}
    assert got[0]["eligible_by_l1"] == {"유통": 40}
    assert got[0]["eligible_by_sub"] == {"편의점": 40}
    # 제외분은 차로 나온다 — 따로 적지 않는다.
    assert got[0]["by_sub"]["할인점"] - got[0]["eligible_by_sub"].get("할인점", 0) == 60


def test_category_export_rejects_policy_payment_in_control():
    with pytest.raises(ValueError, match="control transaction"):
        aggregate_day([{"aid": "a", "online_spent": 0}],
                      [{"aid": "a", "amt": 10, "sub": "마트", "l1": "마트",
                        "spent_from_policy": '{"P010": 1}'}],
                      roster=["a"], day="2025-07-19", arm="off", policy_id=None)
