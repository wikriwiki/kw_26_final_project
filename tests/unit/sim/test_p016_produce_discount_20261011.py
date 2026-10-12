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

# 아래 손 사본(CHAIN_RE·POLICY)은 할인 기전 시험용 가짜다 — 실제 참여 업체 규칙은 아래 REAL(정책 파일) 시험이 지킨다.
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


# ---- 실제 정책 파일로 시험한다(손으로 적은 사본은 실제 규칙을 지키지 못한다) ----
import json as _json  # noqa: E402
REAL = _json.loads((ROOT / "data" / "experiments" / "P016_ab3w_policy_20261011.json").read_text(encoding="utf-8"))


def _real_rules():
    return Rules(REAL["sectors"]["농축산물"]["eligibility"]), Rules(REAL["eligibility"])


def test_real_names_included():
    # 그래프에 실제로 있는 참여 매장 이름(2026-10-11 p016_names.py)
    for n in ["이마트 왕십리점", "이마트메트로", "이마트월계점", "롯데마트 천호점", "롯데마트", "강동농협하나로마트",
              "농협유통하나로마트 외대역점", "서울축협 하나로마트 증미역점", "GS더프레시 관악점", "지에스더프레시역삼",
              "GS THE FRESH 구의역점", "GSTHEFRESH광진", "GS슈퍼광진화양점-제휴",
              # 정책브리핑 2020-08-06: 롯데마트(슈퍼)·홈플러스(슈퍼) 참여
              "홈플러스 영등포점", "홈플러스익스프레스 길동점", "홈플러스 메가푸드마켓 2.0 강동점", "롯데슈퍼프레시 가락점",
              "롯데쇼핑롯데슈퍼마포주", "롯데쇼핑(롯데Market999묵동점", "롯데FRESH", "롯데쇼핑강변점", "롯데슈퍼 강동둔촌점"]:
        for r in _real_rules():
            assert r.eligible(n, "슈퍼마켓", "마트", None, None)[0], n


def test_real_names_excluded():
    # 참여하지 않았거나 마트가 아닌 것: 편의점·이마트에브리데이·노브랜드·2020 뒤 매장·문화센터 결제 항목·서점·본사·롯데리아
    for n in ["이마트24 잠실점", "이마트에브리데이", "이마트애브리데이새생활공판장", "레뱅드매일이마트에브리데이서초동점",
              "이마트노브랜드영등포", "이마트/노브랜드영등포", "노브랜드강남", "이마트푸드마켓", "이마트도서/공연문화비",
              "롯데마트문화비", "롯데마트서울문화비", "GS25 신림서부", "홈플러스도서문화비", "홈플러스도서강서점",
              "홈플러스강서사옥", "홈플러스상품", "홈플러스일반", "롯데리아길동", "롯데쇼핑제타플렉스", "롯데쇼핑토이저러스제타플렉스",
              "롯데밀크마트면목２호점", "롯데지리산축산마트", "늘봄하나로축산", "코사할인마트", "동네청과"]:
        for r in _real_rules():
            assert not r.eligible(n, "슈퍼마켓", "마트", None, None)[0], n


def test_real_chain_assignment_unique():
    specs = active_rate_discounts([REAL], date(2020, 7, 31))
    chains = specs[0]["chains"]
    for n, want in [("이마트 왕십리점", "이마트"), ("롯데마트 천호점", "롯데(롯데마트·롯데슈퍼)"),
                    ("롯데슈퍼프레시 가락점", "롯데(롯데마트·롯데슈퍼)"), ("홈플러스익스프레스 길동점", "홈플러스"), ("강동농협하나로마트", "농협하나로마트"),
                    ("GS THE FRESH 구의역점", "GS더프레시"), ("지에스더프레시역삼", "GS더프레시"), ("GS슈퍼광진화양점-제휴", "GS더프레시")]:
        hit = [c for c, pat in chains if pat.search(n)]
        assert hit == [want], (n, hit)


def test_real_window_only_policy_days():
    # 기간은 할인 계산이 아니라 그날 정책을 불러오는 쿼리가 거른다(정책 밖 날에는 ctx.policy 에 P016 이 없다).
    import dawn_context
    assert "$today >= pol.effective_from AND $today <= pol.effective_until" in dawn_context.POLICY_CYPHER
    assert (REAL["effective_from"], REAL["effective_until"]) == ("2020-07-30", "2020-08-09")


def test_real_same_day_two_payments_same_chain_hit_cap():
    specs = active_rate_discounts([REAL], date(2020, 7, 31))
    out = settle_instant_discounts([_ev("이마트 왕십리점", 40000, 1.0), _ev("이마트 용산점", 40000, 1.0)],
                                   [40000, 40000], specs, {})
    assert out["by_event"] == [{"P016": 8000}, {"P016": 2000}]      # 같은 체인 다른 지점도 한 한도
    assert out["used_after"]["P016:이마트"] == 10000 and out["total"] == 10000


def test_real_two_chains_same_day_separate_caps():
    specs = active_rate_discounts([REAL], date(2020, 7, 31))
    out = settle_instant_discounts([_ev("이마트 왕십리점", 60000, 1.0), _ev("GS더프레시 관악점", 60000, 1.0),
                                    _ev("강동농협하나로마트", 10000, 0.5)], [60000, 60000, 10000], specs, {})
    assert [e.get("P016") for e in out["by_event"]] == [10000, 10000, 1000]
    assert out["total"] == 21000


def test_real_non_grocery_category_or_zero_amount():
    specs = active_rate_discounts([REAL], date(2020, 7, 31))
    out = settle_instant_discounts([_ev("이마트 왕십리점", 30000, 1.0, sub="화장품"), _ev("이마트 왕십리점", 0, 1.0),
                                    _ev("이마트 왕십리점", 30000, 0.0)], [30000, 0, 30000], specs, {})
    assert out["total"] == 0


def test_real_status_and_card_render():
    from mechanisms import sector_voucher as sv
    facts = sv.facts(REAL)
    assert any("이마트·롯데(롯데마트·롯데슈퍼)·홈플러스·농협하나로마트·GS더프레시" in f and "1인 최대 10,000원" in f for f in facts)
    line = sv.status("P016", REAL, {}, {"policy_used": _json.dumps({"P016:이마트": 10000, "P016:GS더프레시": 2400})},
                     today=date(2020, 8, 1))
    assert "이마트 10,000원" in line and "GS더프레시 2,400원" in line and "남은" not in line
