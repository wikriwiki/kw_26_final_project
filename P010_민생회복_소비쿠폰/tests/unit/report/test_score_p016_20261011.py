"""[2026-10-11] P016 채점기 — 답을 아는 가짜 기억 모음으로 지표·판정·무결성 검사를 확인한다."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import score_p016 as sp  # noqa: E402

POL = json.loads((ROOT / "data" / "experiments" / "P016_ab3w_policy_20261011.json").read_text(encoding="utf-8"))
DAYS = ["2020-07-30", "2020-07-31"]


def it(poi, sub, spent, prod=None, disc=0):
    return {"poi": poi, "poi_sub": sub, "actual_spent": spent, "produce_spent": prod, "discount_total": disc}


def rec(aid, items_by_day):
    return {"aid": aid, "plans": [{"day": d, "items": items_by_day.get(d, [])} for d in DAYS]}


def build():
    on = {
        "A": rec("A", {"2020-07-30": [it("이마트 왕십리점", "슈퍼마켓", 50000, 30000, 6000), it("김밥천국", "분식", 5000)]}),
        "B": rec("B", {"2020-07-31": [it("GS THE FRESH 구의역점", "슈퍼마켓", 20000, 10000, 2000),
                                       it("동네청과", "청과", 10000, 10000)]}),
    }
    off = {
        "A": rec("A", {"2020-07-30": [it("이마트 왕십리점", "슈퍼마켓", 40000, 20000), it("김밥천국", "분식", 5000)]}),
        "B": rec("B", {"2020-07-31": [it("GS THE FRESH 구의역점", "슈퍼마켓", 20000, 10000),
                                       it("동네청과", "청과", 20000, 15000)]}),
    }
    return on, off


def test_metrics_match_hand_calculation():
    on, off = build()
    r = sp.score(on, off, POL, DAYS, boot=200)
    m = r["metrics"]
    assert (m["C1"]["on_sum"], m["C1"]["off_sum"]) == (40000, 30000)        # 30000+10000 / 20000+10000
    assert m["C1"]["pct"] == round(100 * 10000 / 30000, 2)
    assert (m["C2"]["on_sum"], m["C2"]["off_sum"]) == (70000, 60000)
    assert (m["M1_nonpart_total"]["on_sum"], m["M1_nonpart_total"]["off_sum"]) == (10000, 20000)   # 동네청과 = 대체
    assert m["C3"]["pp"] == round(100 * 40000 / 70000 - 100 * 30000 / 60000, 2)
    assert m["E1"]["value"] == round(10000 / 8000, 2) and m["E1"]["discount_sum"] == 8000
    assert m["M2"]["people_with_discount"] == 2
    assert r["judgement"]["C2_order"].startswith("맞음")
    assert r["hard_failures"] == [] and r["problems"] == []
    assert r["integrity"]["농축산물_답_비율"] == 1.0


def test_plan_subcategory_is_not_used_only_poi_category():
    on, off = build()
    # 계획 업종이 '장보기'여도 가게 업종이 분식이면 장보기가 아니다
    on["A"]["plans"][0]["items"][1]["sub_category"] = "장보기"
    r = sp.score(on, off, POL, DAYS, boot=50)
    assert r["metrics"]["gro_total"]["on_sum"] == 80000


def test_integrity_catches_engine_errors():
    on, off = build()
    on["A"]["plans"][0]["items"].append(it("코사할인마트", "슈퍼마켓", 30000, 30000, 6000))   # 비참여 매장 할인
    on["B"]["plans"][1]["items"].append(it("GS더프레시 관악점", "슈퍼마켓", 90000, 90000, 9000))    # 같은 체인 누적 11000
    off["A"]["plans"][0]["items"][0]["discount_total"] = 1000                                      # 정책 없는 쪽 할인
    on["A"]["plans"][0]["items"].append(it("이마트 용산점", "슈퍼마켓", 10000, 0, 500))              # 농축산물 0 인데 할인
    r = sp.score(on, off, POL, DAYS, boot=50)
    hf = " ".join(r["hard_failures"])
    for w in ("참여매장_아닌데_할인", "체인한도_초과", "농축산물_0인데_할인", "정책 없는 쪽에 할인"):
        assert w in hf, w


def test_missing_produce_answer_and_missing_day_are_reported():
    on, off = build()
    off["B"]["plans"][1]["items"][0]["produce_spent"] = None
    del on["A"]["plans"][1]
    r = sp.score(on, off, POL, DAYS, boot=50)
    assert r["integrity"]["off_장보기_농축산물_답없음"] == 1
    assert any("날 빠짐" in p for p in r["problems"])


def test_c1_undefined_when_off_arm_has_no_participating_produce():
    on, off = build()
    for rr in off.values():
        for p in rr["plans"]:
            p["items"] = [x for x in p["items"] if not sp.re.match(POL["eligibility"]["include"]["name_regex"], x["poi"])]
    r = sp.score(on, off, POL, DAYS, boot=50)
    assert r["metrics"]["C1"]["pct"] is None and r["judgement"]["C1"].startswith("판정 불가")


def test_m3_trips_and_basket():
    on, off = build()
    on["A"]["plans"][1]["items"].append(it("이마트 용산점", "슈퍼마켓", 30000, 20000, 0))   # 정책 있는 쪽 방문 1회 더
    r = sp.score(on, off, POL, DAYS, boot=50)
    m3 = r["metrics"]["M3"]
    assert r["metrics"]["part_trips"]["on_sum"] == 3 and r["metrics"]["part_trips"]["off_sum"] == 2
    assert m3["trips_pct"] == 50.0
    assert m3["basket_pct"] == round(100 * ((100000 / 3) - 30000) / 30000, 2)
    assert r["judgement"]["M3_trips"].startswith("정답지와 반대")
