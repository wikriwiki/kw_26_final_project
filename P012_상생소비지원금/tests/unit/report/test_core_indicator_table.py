"""정책별 핵심지표 대조표 — scripts/report/core_indicator_table.py

이 시험들은 **오늘 실제로 저지른 두 오독**을 막는다.

  ① HO-1 의 desc "20% 할인(최대 1만원) 대상" 에서 20% 를 실측으로 집어
     `실측 +20.00%` 라고 적었다. **할인율은 정답지가 잰 수가 아니다.**
  ② P012-1 에서 `result_r2_v45`(n=499) 가 `result_r2_v5`(n=497) 를 이겨
     **기각된 후보의 수**를 정답지와 맞댔다.

둘 다 "그럴듯한 수가 나오는" 오독이라 눈으로는 안 잡힌다.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _mod():
    p = ROOT / "scripts" / "report" / "core_indicator_table.py"
    spec = importlib.util.spec_from_file_location("core_indicator_table", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod()


# ---------------------------------------------------------------- 실측 읽기

def test_정책_파라미터를_실측으로_읽지_않는다():
    """**오늘의 오독 ①.** 할인율·한도는 결과가 아니다."""
    desc = "농수산물(마트) 지출 몫 증가 — 20% 할인(최대 1만원) 대상"
    assert C.truth_of(desc) == (None, None)


def test_괄호_안의_실측만_읽는다():
    v, u = C.truth_of("적립업종 지출 개인 내 쌍체차 (실측 +20.82%)")
    assert (v, u) == (20.82, "%")


def test_블록의_실측_필드를_먼저_쓴다():
    v, u = C.truth_of("아무 말", explicit="-14.1%")
    assert (v, u) == (-14.1, "%")


def test_단위_없는_무차원_지표도_읽는다():
    """MPC 0.21 처럼 단위가 없는 실측이 있다."""
    v, u = C.truth_of("한계소비성향 (실측 0.21)")
    assert v == pytest.approx(0.21) and u == ""


def test_퍼센트포인트를_구분한다():
    v, u = C.truth_of("지급 전후 차 (실측 +11.1%p)")
    assert (v, u) == (11.1, "%p")


def test_실측이_없으면_없다고_한다():
    assert C.truth_of("총소비 무반응 — 방어선 지표") == (None, None)
    assert C.truth_of(None) == (None, None)


# ---------------------------------------------------------------- 블록 고르기

def _pol(blocks):
    return {"indicators": [{"id": "X-1", "desc": ""}], **blocks}


def test_기각된_후보_블록을_읽지_않는다():
    """**오늘의 오독 ②.** n 이 크다고 기각된 후보를 정답지와 맞대면 안 된다."""
    pol = _pol({
        "result_r2_v5": {"X-1": {"pct": 1.90, "n": 497}},
        "result_r2_v45": {"X-1": {"pct": 0.60, "n": 499}},
    })
    bk, e = C.best_block(pol, "X-1")
    assert bk == "result_r2_v5", "선택된 프롬프트(v5)의 블록을 읽어야 한다"
    assert e["pct"] == pytest.approx(1.90)


def test_같은_프롬프트_안에서는_표본이_큰_쪽():
    pol = _pol({
        "result_a": {"X-1": {"pct": 1.0, "n": 200}},
        "result_b": {"X-1": {"pct": 2.0, "n": 500}},
    })
    assert C.best_block(pol, "X-1")[0] == "result_b"


def test_값이_없는_블록은_건너뛴다():
    pol = _pol({
        "result_empty": {"X-1": {"n": 900}},
        "result_real": {"X-1": {"mean": 5.0, "n": 100}},
    })
    assert C.best_block(pol, "X-1")[0] == "result_real"


def test_아무_블록도_없으면_None():
    assert C.best_block(_pol({}), "X-1") is None


# ---------------------------------------------------------------- 핵심지표 정의

def test_모든_정책에_핵심지표와_근거가_있다():
    for pk, (iid, why) in C.CORE.items():
        assert iid and len(why) > 15, "%s 의 핵심지표 근거가 부실하다" % pk


def test_핵심지표가_채점표에_등록된_지표다():
    import json
    sc = json.loads((ROOT / "data/experiments/scoring_table.json").read_text(encoding="utf-8"))
    for pk, (iid, _why) in C.CORE.items():
        pol = sc.get(pk)
        if not isinstance(pol, dict):
            pytest.fail("채점표에 %s 블록이 없다" % pk)
        ids = [i["id"] for i in (pol.get("indicators") or [])]
        assert iid in ids, "%s 의 핵심지표 %s 가 등록돼 있지 않다" % (pk, iid)
