"""후보 6 탐침의 산술 — scripts/sim/s1_propensity_probe.py

가장 중요한 둘:
  · `test_events_안의_값을_집지_않는다` — 최상위 daily_propensity 를 재야 한다
  · `test_동점을_숨기지_않는다`       — 최빈 비중이 곧 유효표본의 크기다.
                                        후보 2 에서 동점 31/38 을 못 보고 6:1 을
                                        신호로 착각했다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))


def _mod():
    p = ROOT / "scripts" / "sim" / "s1_propensity_probe.py"
    spec = importlib.util.spec_from_file_location("s1_propensity_probe", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _mod()


# ---------------------------------------------------------------- 파싱

def test_최상위_값을_집는다():
    assert P.parse_propensity('{"daily_propensity": 0.72, "events": []}') == 0.72


def test_events_안의_값을_집지_않는다():
    """최상위가 먼저 나오면 그것을 집는다 — 총액에 실리는 것은 최상위다."""
    raw = '{"daily_propensity": 0.61, "events": [{"daily_propensity": 0.99}]}'
    assert P.parse_propensity(raw) == 0.61


def test_범위를_벗어나면_버린다():
    assert P.parse_propensity('{"daily_propensity": 1.4}') is None
    assert P.parse_propensity('{"daily_propensity": -0.2}') is None


def test_없으면_None():
    assert P.parse_propensity('{"events": []}') is None
    assert P.parse_propensity("") is None


# ---------------------------------------------------------------- 모양

def test_동점을_숨기지_않는다():
    """최빈 비중이 곧 유효표본이다. 후보 2 가 여기서 걸렸어야 했다."""
    s = P.shape([0.68] * 48 + [0.65] * 2)
    assert s["uniq"] == 2
    assert s["modal"] == 0.68
    assert s["modal_share"] == pytest.approx(0.96)


def test_흩어지면_표준편차가_는다():
    a = P.shape([0.68] * 50)
    b = P.shape([0.5 + 0.01 * i for i in range(50)])
    assert a["sd"] == 0 and b["sd"] > 0.1
    assert b["uniq"] > a["uniq"]


# ---------------------------------------------------------------- 상관

def test_완전히_따라가면_r이_1():
    xs = [0.5, 0.6, 0.7, 0.8]
    assert P.pearson(xs, xs) == pytest.approx(1.0)


def test_상수_출력이면_r이_0():
    """**겨냥한 결함이 이것이다.** 누구에게나 같은 값을 적으면 상관이 정의되지 않는다."""
    assert P.pearson([0.68] * 5, [0.5, 0.6, 0.7, 0.8, 0.9]) == 0.0


def test_반대로_가면_음수():
    assert P.pearson([0.9, 0.8, 0.7, 0.6], [0.5, 0.6, 0.7, 0.8]) == pytest.approx(-1.0)


# ---------------------------------------------------------------- 부호검정

def test_부호검정은_동점을_분모에서_뺀다():
    assert P.sign_p(6, 1) == pytest.approx(0.125, abs=1e-3)
    assert P.sign_p(22, 22) == pytest.approx(1.0)
    assert P.sign_p(0, 0) == 1.0


def test_유효표본이_커지면_같은_비율도_유의해진다():
    """후보 2 의 교훈 — 6:1(p=0.125)과 60:10 은 다른 증거다."""
    assert P.sign_p(6, 1) > 0.05
    assert P.sign_p(60, 10) < 0.05


# ---------------------------------------------------------------- 쌍 맞추기

def _rows(key="arm"):
    out = []
    for case, a, b in (("c1", 0.60, 0.70), ("c2", 0.50, 0.55)):
        for name, v in (("v5", a), ("v5self", b)):
            out.append({"aid": "x", "case": case, "date": "d", "seed": 1,
                        key: name, "raw": '{"daily_propensity": %s}' % v})
    return out


def test_쌍은_사람_상황_날짜_시드로_맞춘다(capsys):
    """aid 만으로 맞추면 같은 사람의 다른 칸이 뭉개진다 — 전에 24칸이 5쌍이 됐다."""
    P.report(_rows())
    assert "쌍 2" in capsys.readouterr().out


def test_러너가_쓰는_arm_키를_읽는다(capsys):
    """기존 러너는 `arm` 으로 적는다. 여기서 어긋나면 쌍이 0이 되고 탐침이 헛돈다."""
    P.report(_rows("arm"))
    assert "쌍 2" in capsys.readouterr().out


def test_variant_키도_받는다(capsys):
    P.report(_rows("variant"))
    assert "쌍 2" in capsys.readouterr().out
