"""모든 지표 대조표 — scripts/report/all_indicators_table.py

이 시험들은 **오늘 실제로 저지른 오독 네 가지**를 막는다. 넷 다 "그럴듯한 수가
나오는" 종류라 눈으로는 안 잡힌다.

  ① HO-1 의 desc "20% 할인(최대 1만원)" 에서 20% 를 실측으로 읽었다
  ② P012-1 에서 기각된 후보(v45, n=499)가 v5(n=497)를 표본 크기로 이겼다
  ③ HO-* 에서 **무정책 기준선 팔의 수**(+16.1%)를 정책 결과로 읽었다.
     진짜 값은 note 의 "차 +21.5%p" 다
  ④ DS-6 을 "관측부족" 으로 읽었다. 지표 정의는 "그래프에 hub_type 0건" 이라고
     말한다 — 표본을 늘려 되는 것과 자료를 구해야 되는 것은 다르다
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _mod():
    p = ROOT / "scripts" / "report" / "all_indicators_table.py"
    spec = importlib.util.spec_from_file_location("all_indicators_table", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = _mod()


# ---------------------------------------------------------------- ① 실측 읽기

def test_할인율을_실측으로_읽지_않는다():
    assert A.truth_of("마트 지출 몫 증가 — 20% 할인(최대 1만원) 대상") == (None, None)


def test_괄호_안의_실측만_읽는다():
    assert A.truth_of("적립업종 (실측 +20.82%)") == (20.82, "%")
    assert A.truth_of("1인 캐시백 (실측 47,880원)") == (47880.0, "원")
    assert A.truth_of("지급 전후 (실측 +11.1%p)") == (11.1, "%p")


def test_무차원_실측도_읽는다():
    v, u = A.truth_of("한계소비성향 (실측 0.21)")
    assert v == pytest.approx(0.21) and u == ""


# ---------------------------------------------------------------- ② 후보 가로지르기

def test_기각된_후보_블록을_읽지_않는다():
    pol = {"indicators": [],
           "result_r2_v5": {"X": {"pct": 1.9, "n": 497}},
           "result_r2_v45": {"X": {"pct": 0.6, "n": 499}}}
    assert A.best_block(pol, "X")[0] == "result_r2_v5"


# ---------------------------------------------------------------- ③ 기준선 팔

def test_기준선_블록을_정책_결과로_읽지_않는다():
    """**오독 ③.** 무정책 팔의 수를 정책 효과로 보고하면 안 된다."""
    pol = {"indicators": [],
           "result_baseline_x": {"X": {"pct": 16.1, "n": 199}},
           "result_policy_x_hits": {"X": {"note": "기준선 +16.1% → 정책 +37.6% · 차 +21.5%p"}}}
    bk, e = A.best_block(pol, "X")
    assert "policy" in bk, "note 가 있는 정책 블록을 읽어야 한다"
    kind, v, u = A.from_note(e)
    assert (kind, v, u) == ("값", 21.5, "%p")


def test_기준선만_있으면_그것도_읽지_않는다():
    pol = {"indicators": [], "result_baseline_x": {"X": {"pct": 16.1, "n": 199}}}
    assert A.best_block(pol, "X") is None


def test_note_의_상태를_숫자로_억지로_바꾸지_않는다():
    for note, want in (("**무효** — 적격 판정이 상생 기준이었다", "무효"),
                       ("p=0.786 · 몫 0.7~1.7% 라 관측이 너무 적다", "관측부족"),
                       ("총소비 — 밴드 안", "밴드안")):
        kind = A.from_note({"note": note})[0]
        assert kind == want, "%s -> %s 여야 한다" % (note[:20], want)


def test_note_에서_원_단위_차이도_읽는다():
    kind, v, u = A.from_note({"note": "차 +526 · CI[-867,+1919] · p=0.459"})
    assert (kind, v, u) == ("값", 526.0, "원")


# ---------------------------------------------------------------- ④ 측정 불가

def test_지표가_측정_불가라고_하면_자료없음이다():
    """**오독 ④.** 표본을 늘려 되는 것과 자료를 구해야 되는 것은 다르다."""
    ind = {"id": "DS-6", "expect": "info",
           "desc": "관광특구 > 발달상권 — **측정 불가: 그래프에 상권 유형(hub_type) 속성이 0건이다**"}
    st, err, shown = A.classify(ind, {"note": "관측부족"}, -8.7, "%")
    assert st == "자료없음" and err is None


def test_관측부족은_그대로_관측부족이다():
    ind = {"id": "EM-4", "expect": "rank", "desc": "준내구재 > 대면서비스 (실측 +10.8%p vs +3.6%p)"}
    st, _e, _s = A.classify(ind, {"note": "관측부족(scale=main)"}, None, None)
    assert st == "관측부족"


# ---------------------------------------------------------------- 오차 계산

def test_같은_단위일_때만_오차를_낸다():
    ind = {"id": "X", "expect": "+", "desc": "(실측 +7.3%)"}
    st, err, _ = A.classify(ind, {"pct": 12.6, "n": 200}, 7.3, "%")
    assert st == "대조가능" and err == pytest.approx(5.3)

    st2, err2, _ = A.classify(ind, {"pct": 13.1, "n": 200}, 11.1, "%p")
    assert st2 == "단위다름" and err2 is None


def test_다른자로_잰_것은_오차를_내지_않는다():
    ind = {"id": "X", "expect": "0", "desc": "(실측 +2.85%)"}
    st, err, _ = A.classify(ind, {"pct": -13.9, "n": 23}, 2.85, "%", suspect=True)
    assert st == "다른자" and err is None


def test_순위지표는_간격으로_맞댄다():
    ind = {"id": "X", "expect": "rank", "desc": "(실측 +10.8%p vs +3.6%p)"}
    st, err, shown = A.classify(ind, {"got": "A +45.8% vs B -22.5%"}, None, None)
    assert st == "대조가능"
    assert err == pytest.approx(abs((45.8 - (-22.5)) - (10.8 - 3.6)), abs=0.01)
    assert "간격" in shown


def test_시뮬_값이_없으면_없음():
    ind = {"id": "X", "expect": "+", "desc": "(실측 +1%)"}
    assert A.classify(ind, None, 1.0, "%")[0] == "없음"


def test_무차원_비율만으로_추정량_일치를_단정하지_않는다():
    """근접한 숫자도 조사 정의·표본·기간 확인을 대체하지 못한다."""
    ind = {"id": "P010-1", "expect": "+", "desc": "한계소비성향 (실측 0.21)"}
    st, err, shown = A.classify(ind, {"mean": 0.216, "n": 1971}, 0.21, "")
    assert st == "정의확인"
    assert err is None
    assert shown == "0.216"


def test_무차원_오차는_퍼센트포인트_합에_안_들어간다():
    """단위가 섞이면 총합이 뜻을 잃는다. %p 만 더한다."""
    rows = [{"err": 5.3, "truth_unit": "%"}, {"err": 4.4, "truth_unit": "%"},
            {"err": 0.006, "truth_unit": ""}]
    pp = [r["err"] for r in rows if r["truth_unit"] in ("%", "%p")]
    assert len(pp) == 2 and sum(pp) == pytest.approx(9.7)


def test_실측_단위가_퍼센트면_절대값과_맞대지_않는다():
    """표시 단위를 환산해도 월말 범위·분모 확인은 남는다."""
    ind = {"id": "P012-6", "expect": "+", "desc": "한도 도달 (실측 21.0%)"}
    st, err, shown = A.classify(ind, {"mean": 0.0, "n": 500}, 21.0, "%")
    assert st == "정의확인" and err is None
    assert shown == "0.00%"


def test_기준선_note도_정책_결과가_아니다():
    pol = {"result_baseline_x": {"X": {"note": "차 +21.5%p"}},
           "result_policy_x": {"X": {"pct": 2.0, "n": 10}}}
    assert A.best_block(pol, "X")[0] == "result_policy_x"


def test_순위_간격_0도_관측값이다():
    pol = {"result_x": {"X": {"got": "A +3.1% vs B +3.1%"}}}
    assert A.best_block(pol, "X") is not None


def test_실측_크기_부재는_단위불일치가_아니다():
    ind = {"id": "P012-3", "expect": "+"}
    assert A.classify(ind, {"mean": 0.168}, None, None) == ("방향만", None, "16.80%")
    for iid in ("PT-1", "PT-2"):
        st, err, _ = A.classify({"id": iid, "expect": "0"},
                                 {"mean": 2317, "hit": False}, None, None)
        assert st == "동등성검증" and err is None


def test_누적_캐시백은_일평균으로_설명하지_않는다():
    ind = {"id": "P012-4", "expect": "+"}
    st, err, shown = A.classify(ind, {"mean": 1311}, 47880, "원")
    assert (st, err, shown) == ("정의확인", None, "1311원")
    assert "10~11월 수령자 평균" in A.UNIT_NOTE["P012-4"]


def test_삼중차분_로그계수를_짧은_전후_증가율과_비교하지_않는다():
    ind = {"id": "P012-1", "expect": "+", "empirical_audit": {
        "comparison": "different_estimand"}}
    st, err, shown = A.classify(ind, {"pct": 1.9}, 0.2082, "log-point")
    assert (st, err, shown) == ("다른자", None, "+1.90%")


def test_P012_실측_원문_계수를_퍼센트_오차에_더하지_않는다():
    table = json.loads((ROOT / 'data/experiments/scoring_table.json').read_text(encoding='utf-8'))
    ind = next(i for i in table['P012']['indicators'] if i['id'] == 'P012-1')
    audit = ind['empirical_audit']
    assert audit['reported_value'] == pytest.approx(0.2082)
    assert audit['reported_unit'] == 'log-point'
    entry = table['P012']['result_r2_v5']['P012-1']
    st, err, _ = A.classify(ind, entry, audit['reported_value'], audit['reported_unit'])
    assert st == '다른자' and err is None


def test_KDI_공식_수치여도_추정량이_다르면_크기_적중을_주지_않는다():
    table = json.loads((ROOT / 'data/experiments/scoring_table.json').read_text(encoding='utf-8'))
    for iid, observed, reported, unit in [('EM-2', 13.1, 11.1, '%p'),
                                           ('EM-3', 12.6, 7.3, '%')]:
        ind = next(i for i in table['EMERGENCY_2020']['indicators'] if i['id'] == iid)
        audit = ind['empirical_audit']
        assert audit['source'] == 'https://www.kdi.re.kr/share/pressView?bd_no=4018'
        assert audit['comparison'] == 'different_estimand'
        assert (audit['reported_value'], audit['reported_unit']) == (reported, unit)
        st, err, shown = A.classify(ind, {'pct': observed, 'n': 200}, reported, unit)
        assert (st, err, shown) == ('다른자', None, f'{observed:+.2f}%')


def test_전년비와_이틀_전후비를_맞대지_않는다():
    table = json.loads((ROOT / 'data/experiments/scoring_table.json').read_text(encoding='utf-8'))
    pol = table['DISTANCING_2020']
    for iid, ref, observed in [('DS-1', -14.1, -9.7), ('DS-2', 4.2, 5.1)]:
        ind = next(i for i in pol['indicators'] if i['id'] == iid)
        st, err, shown = A.classify(ind, {'pct': observed, 'n': 500}, ref, '%')
        assert st == '다른자' and err is None
        assert shown == f'{observed:+.2f}%'
