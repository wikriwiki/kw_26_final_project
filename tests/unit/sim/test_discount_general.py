"""할인·환급 회계 일반화 (2026-10-06) — P014 할인 구매 상품권, P015 업종별 할인·환급."""
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/sim"))
from instant_discount import active_rate_discounts, settle_instant_discounts  # noqa: E402

P014 = {"id": "P014", "type": "price_discount", "discount_rate": 0.1, "purchase_cap_monthly": 500000,
        "eligibility": {"mode": "exclude", "exclude": {"subs_always": {"other": ["백화점"]}},
                        "require_same_district": True}}


def _ev(poi, sub, l1, same=None, time="12:00"):
    return {"poi_id": poi, "sub_category": sub, "category": l1, "poi_same_gu": same, "time": time}


def test_voucher_discount_only_in_home_district_and_monthly_cap():
    specs = active_rate_discounts([P014], date(2020, 9, 21))
    assert specs[0]["key"] == "P014@2020-09" and specs[0]["cap"] == 50000
    events = [_ev("A", "한식", "식사", same=True), _ev("B", "한식", "식사", same=False),
              _ev("C", "한식", "식사", same=None), _ev("D", "백화점", "쇼핑", same=True)]
    r = settle_instant_discounts(events, [30000, 30000, 30000, 30000], specs)
    assert r["by_event"] == [{"P014": 3000}, {}, {}, {}]
    # 월 상한(50만원어치 x 10% = 5만원)을 넘지 않는다
    r2 = settle_instant_discounts([_ev("A", "한식", "식사", same=True)], [600000], specs,
                                  {"P014@2020-09": 45000})
    assert r2["by_pid"]["P014"] == 5000
    # 다음 달에는 새 한도
    assert active_rate_discounts([P014], date(2020, 10, 1))[0]["key"] == "P014@2020-10"


P015 = {"id": "P015", "type": "sector_voucher", "sectors": {
    "외식": {"mode": "count_rebate", "min_amount": 20000, "count": 3, "rebate": 10000,
             "window": {"from_weekday": 4, "from_time": "16:00", "to_weekday": 6, "to_time": "24:00"},
             "eligibility": {"mode": "include", "include": {"l1s": ["식사"]}}, "from": "2020-10-30"},
    "숙박": {"mode": "flat", "amount": 30000, "tiers": [{"max": 70000, "amount": 30000}, {"amount": 40000}],
             "eligibility": {"mode": "include", "include": {"subs": ["숙박"]}}, "from": "2020-11-04"},
    "체육": {"mode": "rebate", "min_amount": 80000, "amount": 30000, "max_uses": 1,
             "eligibility": {"mode": "include", "include": {"subs": ["헬스장"]}}, "from": "2020-11-02"},
}}


def test_sector_start_dates():
    assert {s["sector"] for s in active_rate_discounts([P015], date(2020, 10, 31))} == {"외식"}
    assert {s["sector"] for s in active_rate_discounts([P015], date(2020, 11, 4))} == {"외식", "숙박", "체육"}


def test_count_rebate_counts_weekend_payments_and_pays_on_the_third():
    specs = active_rate_discounts([P015], date(2020, 11, 6))      # 금요일
    friday = 4
    evs = [_ev("R1", "한식", "식사", time="15:00"),   # 16시 전 — 세지 않는다
           _ev("R2", "한식", "식사", time="18:00"),
           _ev("R3", "한식", "식사", time="19:00")]
    r = settle_instant_discounts(evs, [25000, 25000, 15000], specs, weekday=friday)
    assert r["used_after"]["P015:외식#n"] == 1 and r["rebate_total"] == 0   # 15,000원은 조건 미달
    r2 = settle_instant_discounts([_ev("R4", "한식", "식사", time="12:00"), _ev("R5", "한식", "식사", time="13:00")],
                                  [30000, 30000], specs, r["used_after"], weekday=5)
    assert r2["rebate_by_event"] == [{}, {"P015": 10000}] and r2["used_after"]["P015:외식#n"] == 0
    assert r2["by_pid"]["P015"] == 0          # 환급은 오늘 자기부담을 줄이지 않는다
    # 평일에는 세지 않는다
    r3 = settle_instant_discounts([_ev("R6", "한식", "식사", time="12:00")], [30000], specs, weekday=1)
    assert r3["used_after"]["P015:외식#n"] == 0


def test_flat_tiers_and_rebate_threshold():
    specs = active_rate_discounts([P015], date(2020, 11, 5))
    r = settle_instant_discounts([_ev("H1", "숙박", "숙박"), _ev("H2", "숙박", "숙박")],
                                 [60000, 90000], specs, weekday=3)
    assert r["by_event"] == [{"P015": 30000}, {"P015": 40000}]
    g = settle_instant_discounts([_ev("G1", "헬스장", "운동"), _ev("G2", "헬스장", "운동")],
                                 [50000, 90000], specs, weekday=3)
    assert g["rebate_by_event"] == [{}, {"P015": 30000}]
    again = settle_instant_discounts([_ev("G3", "헬스장", "운동")], [90000], specs, g["used_after"], weekday=3)
    assert again["rebate_total"] == 0          # 1인 1회


def test_window_needs_known_weekday():
    specs = active_rate_discounts([P015], date(2020, 11, 6))
    r = settle_instant_discounts([_ev("R", "한식", "식사", time="18:00")], [30000], specs, weekday=None)
    assert r["used_after"]["P015:외식#n"] == 0


def test_two_discount_policies_at_once_are_refused():
    with pytest.raises(ValueError, match="중복"):
        active_rate_discounts([P014, P015], date(2020, 11, 6))
