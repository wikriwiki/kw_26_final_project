"""[2026-10-11] P016 농할: 참여 체인 한정, 농축산물 몫만 할인, 체인별 한도, 기록 필드."""
import importlib
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(ROOT / "scripts"))

from eligibility import Rules  # noqa: E402
from instant_discount import active_rate_discounts, settle_instant_discounts  # noqa: E402

CHAIN_RE = r"^(이마트(?!24| ?에브리데이)|롯데마트|.*하나로(마트|클럽)|GS ?더 ?프레시|지에스더프레시|GS ?수퍼)"
POLICY = {
    "id": "P016", "type": "sector_voucher",
    "sectors": {"농축산물": {
        "mode": "rate", "rate": 0.2, "cap": 10000, "base": "produce_share",
        "chains": {"이마트": r"^이마트", "롯데마트": r"^롯데마트", "하나로": r"하나로(마트|클럽)", "GS더프레시": r"GS|지에스"},
        "eligibility": {"mode": "include", "include": {"subs": ["슈퍼마켓", "식료품", "청과", "정육"], "name_regex": CHAIN_RE}},
    }},
}


def _ev(name, amount, share, sub="슈퍼마켓"):
    return {"poi_id": "x", "poi_name": name, "poi_sub_category": sub, "category": "마트",
            "actual_spent": amount, "produce_share": share}


def test_include_name_regex_limits_to_participating_chains():
    r = Rules(POLICY["sectors"]["농축산물"]["eligibility"])
    assert r.eligible("이마트 양재점", "슈퍼마켓", "마트", None, None)[0]
    assert r.eligible("강동농협하나로마트", "슈퍼마켓", "마트", None, None)[0]
    assert not r.eligible("이마트24 잠실점", "슈퍼마켓", "마트", None, None)[0]
    assert not r.eligible("이마트에브리데이", "슈퍼마켓", "마트", None, None)[0]
    assert not r.eligible("홈플러스 영등포점", "슈퍼마켓", "마트", None, None)[0]
    assert not r.eligible("동네청과", "청과", "마트", None, None)[0]


def test_discount_uses_produce_share_only():
    specs = active_rate_discounts([POLICY], date(2020, 7, 31))
    out = settle_instant_discounts([_ev("이마트 양재점", 50000, 0.4)], [50000], specs, {})
    assert out["by_event"][0] == {"P016": 4000}          # 50,000 × 0.4 × 20%
    assert out["eligible_gross"] == 20000
    assert out["eligible_gross_basis"] == "model_reported_produce"
    assert out["product_lines_observed"] is True


def test_missing_produce_answer_gets_no_discount():
    specs = active_rate_discounts([POLICY], date(2020, 7, 31))
    out = settle_instant_discounts([_ev("이마트 양재점", 50000, None)], [50000], specs, {})
    assert out["by_event"][0] == {} and out["total"] == 0


def test_cap_counts_per_chain_and_carries_over():
    specs = active_rate_discounts([POLICY], date(2020, 7, 31))
    day1 = settle_instant_discounts([_ev("이마트 양재점", 100000, 0.5)], [100000], specs, {})
    assert day1["by_event"][0] == {"P016": 10000}
    assert day1["used_after"]["P016:이마트"] == 10000
    # 다음 날 같은 체인은 한도 끝, 다른 체인은 새 한도
    day2 = settle_instant_discounts([_ev("이마트 성수점", 50000, 1.0), _ev("롯데마트 송파점", 20000, 1.0)],
                                    [50000, 20000], specs, day1["used_after"])
    assert day2["by_event"][0] == {} and day2["by_event"][1] == {"P016": 4000}


def test_non_participating_store_not_discounted():
    specs = active_rate_discounts([POLICY], date(2020, 7, 31))
    out = settle_instant_discounts([_ev("홈플러스 영등포점", 50000, 1.0)], [50000], specs, {})
    assert out["total"] == 0


def test_stage2_produce_field_only_when_flag_on(monkeypatch):
    monkeypatch.setenv("EXP_PRODUCE_FIELD", "1")
    import stage2_poi
    importlib.reload(stage2_poi)
    assert stage2_poi._PRODUCE_FIELD is True
    pick = stage2_poi.Stage2Pick(order=0, poi_id="x", actual_spent=30000, produce_spent=12000)
    assert pick.model_dump()["produce_spent"] == 12000
    monkeypatch.setenv("EXP_PRODUCE_FIELD", "0")
    importlib.reload(stage2_poi)
    assert stage2_poi._PRODUCE_FIELD is False
    assert "produce_spent" not in stage2_poi.Stage2Pick(order=0, poi_id="x", actual_spent=1).model_dump()


def test_status_shows_received_discount_not_balance():
    # 결제할 때 깎이는 할인을 '남은 잔액'처럼 보이게 하지 않는다 — 받은 할인과 한도 규칙만 적는다.
    import json as _json
    from mechanisms import sector_voucher as sv
    before = sv.status("P016", POLICY, {}, {"policy_used": "{}"}, today=date(2020, 7, 30))
    assert "지금까지 받은 할인 — 없음" in before and "유통업체마다 1인 최대 10,000원" in before
    assert "남은" not in before and "10,000원, " not in before
    after = sv.status("P016", POLICY, {}, {"policy_used": _json.dumps({"P016:이마트": 6400})}, today=date(2020, 7, 31))
    assert "지금까지 받은 할인 — 이마트 6,400원" in after and "롯데마트" not in after
