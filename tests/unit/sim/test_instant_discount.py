"""정책 JSON의 즉시 할인율·적격·누적한도를 실제 결제 회계에 적용한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/sim"))
import consumption  # noqa: E402
from instant_discount import active_rate_discounts, settle_instant_discounts  # noqa: E402
from plan_writer import NIGHT_STATE_CYPHER, night_create_state, write_plan  # noqa: E402
from mechanisms import sector_voucher  # noqa: E402


def _policy():
    return {"id": "TEST", "type": "sector_voucher",
            "sectors": {"식재료": {"mode": "rate", "rate": 0.2, "cap": 10_000}},
            "eligibility": {"mode": "include", "include": {"subs": ["청과", "정육"]}}}


def test_적격_매출은_할인전_금액이고_시민부담만_할인된다():
    specs = active_rate_discounts([_policy()])
    events = [
        {"poi_id": "C1", "category": "마트", "sub_category": "청과"},
        {"poi_id": "C2", "category": "마트", "sub_category": "수산"},
    ]
    result = settle_instant_discounts(events, [20_000, 30_000], specs)
    assert result["eligible_gross"] == 20_000
    assert result["total"] == 4_000
    assert result["by_event"] == [{"TEST": 4_000}, {}]
    assert sum([20_000, 30_000]) - result["total"] == 46_000


def test_개인_누적_상한을_날짜를_넘어_보존한다():
    specs = active_rate_discounts([_policy()])
    event = [{"poi_id": "C1", "category": "마트", "sub_category": "정육"}]
    first = settle_instant_discounts(event, [45_000], specs)
    second = settle_instant_discounts(event, [20_000], specs, first["used_after"])
    assert first["total"] == 9_000
    assert second["total"] == 1_000
    assert second["used_after"]["TEST"] == 10_000


def test_지원되지_않는_할인방식은_조용히_정책효과로_채우지_않는다():
    policy = _policy()
    policy["sectors"]["식재료"]["mode"] = "count_rebate"
    with pytest.raises(ValueError, match="지원되지 않는"):
        active_rate_discounts([policy])


def test_p016_정책_원본의_적격과_상한을_그대로_읽는다():
    policy = json.loads((ROOT / "data/neo4j_load/policies/P016.json").read_text(
        encoding="utf-8"))
    specs = active_rate_discounts([policy])
    events = [{"poi_id": "C1", "category": "마트", "sub_category": "청과"},
              {"poi_id": "C2", "category": "마트", "sub_category": "수산"}]
    result = settle_instant_discounts(events, [60_000, 60_000], specs)
    assert result["by_event"] == [{"P016": 10_000}, {}]
    assert result["used_after"]["P016"] == 10_000


def test_neo4j의_기전_json에서도_같은_할인규칙을_읽는다():
    policy = _policy()
    row = {"id": policy["id"], "type": policy["type"],
           "mech_params": json.dumps({"sectors": policy["sectors"],
                                      "eligibility": policy["eligibility"]})}
    specs = active_rate_discounts([row])
    result = settle_instant_discounts(
        [{"poi_id": "C1", "sub_category": "청과", "category": "마트"}],
        [10_000], specs)
    assert result["total"] == 2_000


def test_다음날_프롬프트에는_남은_할인한도가_표시된다():
    line = sector_voucher.status(
        "TEST", _policy(), {}, {"policy_used": '{"TEST": 9000}'})
    assert "남은 할인 1,000원" in line


def test_환급_방식의_누적한도를_즉시할인_잔액으로_보여주지_않는다():
    policy = _policy()
    policy["sectors"]["식재료"]["mode"] = "rate_rebate"
    line = sector_voucher.status("TEST", policy, {}, {"policy_used": '{"TEST": 9000}'})
    assert "남은 할인" not in line
    assert "1인 누적 10,000원 한도" in line


def test_즉시할인이_구매가능액을_늘리되_시민잔액을_초과해_쓰지_않는다(monkeypatch):
    monkeypatch.setattr(consumption, "ELIGIBLE_SHARE_SEOUL", 1.0)
    policy = json.loads((ROOT / "data/neo4j_load/policies/P016.json").read_text(
        encoding="utf-8"))
    base = {"category": "마트", "sub_category": "청과", "poi_id": "C1",
            "actual_spent": 10_000, "policy_spend": {}, "coupon_eligible": True}
    discounted, ordinary = [dict(base)], [dict(base)]
    args = {"daily": 10_000, "income_tier": "중", "tendency": "standard",
            "balance": 8_000, "grant_avail": {}, "llm_propensity": 0.74,
            "online_share": 0}
    changed = consumption.apply_consumption_model(
        discounted, **args,
        instant_discount_specs=active_rate_discounts([policy]),
        discount_used_before={})
    consumption.apply_consumption_model(ordinary, **args)
    gross = discounted[0]["actual_spent"]
    assert gross > ordinary[0]["actual_spent"]
    assert gross - changed["instant_discount_total"] <= 8_000
    assert changed["instant_discount_total"] == discounted[0]["instant_discount"]["P016"]


def test_할인_영수증과_잔액_차감_인자가_저장된다():
    class FakeTx:
        def __init__(self):
            self.calls = []

        def run(self, query, **params):
            self.calls.append((query, params))
            return self

        def single(self):
            return {"written": 1, "balance": 8_000}

    tx = FakeTx()
    event = {"poi_id": "C1", "order": 1, "time": "12:00:00",
             "category": "마트", "sub_category": "청과", "actual_spent": 10_000,
             "instant_discount": {"P016": 2_000}}
    from datetime import date
    day = date(2020, 8, 4)
    write_plan("a", day, [event], "weekday", transaction=tx)
    assert any(call[1].get("events", [{}])[0].get("instant_discount_json")
               == '{"P016": 2000}' for call in tx.calls if "events" in call[1])
    night_create_state("a", day, today_instant_discount=2_000, transaction=tx)
    assert tx.calls[-1][0] == NIGHT_STATE_CYPHER
    assert tx.calls[-1][1]["today_instant_discount"] == 2_000
    assert "today_spent - $today_policy_spent - $today_instant_discount" in NIGHT_STATE_CYPHER
