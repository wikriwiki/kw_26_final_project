"""원문 대조로 확인한 2020 규칙 (2026-10-06).

P014: 서울시 2020-07-08 — 기본 7%, 구별 10%, 구별 월 70만원. 서울시 2019-12-19 — 발행 구 안에서만, 누구나 어느 구든 구매.
P015 외식: 농식품부 2020-10-28 — 금 16시~일 24시 2만원 이상 3회 → 4번째 1만원, 카드사별 1일 2회, 같은 업소 1일 1회.
P015 체육: 이데일리 2020-10-18 — 11/2~11/30 에 8만원 이상 사용하면 3만원 환급(기간 누적).
"""
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/sim"))
from instant_discount import active_rate_discounts, settle_instant_discounts  # noqa: E402

P014 = {"id": "P014", "type": "price_discount", "discount_rate": 0.07, "purchase_cap_monthly": 700000,
        "district_rates": {"11620": 0.10}, "use_scope": "issuing_district",
        "eligibility": {"mode": "exclude", "exclude": {"subs_always": {"other": ["백화점"]}},
                        "require_same_district": True}}


def _ev(poi, sub, l1, gu=None, time="12:00"):
    return {"poi_id": poi, "sub_category": sub, "category": l1, "poi_gu": gu, "time": time}


def test_issuing_district_rates_home_and_work():
    specs = active_rate_discounts([P014], date(2020, 9, 21), gus=["11620", "11140"])   # 관악 거주, 중구 직장
    by_gu = {s["poi_gu"]: s for s in specs}
    assert by_gu["11620"]["rate"] == 0.10 and by_gu["11620"]["cap"] == 70000
    assert by_gu["11140"]["rate"] == 0.07 and by_gu["11140"]["cap"] == 49000
    evs = [_ev("A", "한식", "식사", "11620"), _ev("B", "한식", "식사", "11140"), _ev("C", "한식", "식사", "11680"),
           _ev("D", "한식", "식사", None)]
    r = settle_instant_discounts(evs, [10000, 10000, 10000, 10000], specs)
    assert r["by_event"] == [{"P014": 1000}, {"P014": 700}, {}, {}]
    assert r["used_after"]["P014:11620@2020-09"] == 1000


def test_issuing_district_needs_people_gus():
    with pytest.raises(ValueError):
        active_rate_discounts([P014], date(2020, 9, 21))


EAT = {"id": "P015", "type": "sector_voucher", "sectors": {"외식": {
    "mode": "count_rebate", "min_amount": 20000, "count": 3, "on_next": True, "rebate": 10000,
    "max_per_day": 2, "once_per_poi_per_day": True,
    "window": {"from_weekday": 4, "from_time": "16:00", "to_weekday": 6, "to_time": "24:00"},
    "eligibility": {"mode": "include", "include": {"subs": ["한식"]}}}}}


def test_dining_fourth_payment_daily_limits():
    specs = active_rate_discounts([EAT], date(2020, 10, 31))
    # 토요일: 같은 가게 두 번 → 한 번만, 하루 세 번 → 두 번만
    sat = [_ev("R1", "한식", "식사", time="12:00"), _ev("R1", "한식", "식사", time="13:00"),
           _ev("R2", "한식", "식사", time="18:00"), _ev("R3", "한식", "식사", time="19:00")]
    r = settle_instant_discounts(sat, [20000] * 4, specs, weekday=5)
    assert r["used_after"]["P015:외식#n"] == 2 and r["rebate_total"] == 0
    # 일요일: 3번째 결제는 세기만, 4번째 결제에서 환급
    sun = [_ev("R4", "한식", "식사", time="12:00"), _ev("R5", "한식", "식사", time="18:00")]
    r2 = settle_instant_discounts(sun, [20000, 20000], specs, r["used_after"], weekday=6)
    assert r2["rebate_by_event"] == [{}, {"P015": 10000}]


GYM = {"id": "P015", "type": "sector_voucher", "sectors": {"체육": {
    "mode": "rebate", "min_amount": 80000, "amount": 30000, "cumulative": True, "max_uses": 1,
    "from": "2020-11-02", "until": "2020-11-30",
    "eligibility": {"mode": "include", "include": {"subs": ["헬스장"]}}}}}


def test_gym_cumulative_threshold_once():
    specs = active_rate_discounts([GYM], date(2020, 11, 3))
    r = settle_instant_discounts([_ev("G", "헬스장", "건강")], [50000], specs, weekday=1)
    assert r["rebate_total"] == 0
    r2 = settle_instant_discounts([_ev("G", "헬스장", "건강")], [40000], specs, r["used_after"], weekday=2)
    assert r2["rebate_total"] == 30000
    r3 = settle_instant_discounts([_ev("G", "헬스장", "건강")], [90000], specs, r2["used_after"], weekday=3)
    assert r3["rebate_total"] == 0
    assert active_rate_discounts([GYM], date(2020, 11, 1)) == []
