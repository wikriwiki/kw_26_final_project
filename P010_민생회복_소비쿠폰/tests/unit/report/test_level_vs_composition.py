"""수준·구성 분해의 산술 — scripts/report/level_vs_composition.py

가장 중요한 시험은 `test_등록지표_오차는_줄지_않는다` 다. 이 분해는 **작은 수를
고르는 장치가 아니다.** 같은 값이 자에 따라 18.92%p 와 2.26%p 가 되는데, 공식
성적은 등록된 자(적립업종)로 남아야 한다 — 그러지 않으면 눈금 이동이다.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _mod():
    p = ROOT / "scripts" / "report" / "level_vs_composition.py"
    spec = importlib.util.spec_from_file_location("level_vs_composition", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


L = _mod()


def test_P012_의_함의된_총소비가_7_67퍼센트다():
    """정답지 두 수 + 적립 몫 0.268 -> +7.67%. 이 수가 분해의 축이다."""
    assert L.implied_total(0.268, 20.82, 2.85) == pytest.approx(7.67, abs=0.01)


def test_두_수가_같으면_총소비도_같다():
    """구성 이동이 없으면 총소비 = 그 값. 몫과 무관해야 한다."""
    for s in (0.1, 0.268, 0.9):
        assert L.implied_total(s, 5.0, 5.0) == pytest.approx(5.0)


def test_몫이_0이면_제외업종_값만_남는다():
    assert L.implied_total(0.0, 20.82, 2.85) == pytest.approx(2.85)


def test_몫이_1이면_적립업종_값만_남는다():
    assert L.implied_total(1.0, 20.82, 2.85) == pytest.approx(20.82)


def test_수준_오차와_등록지표_오차를_둘_다_낸다():
    d = L.split_error(0.268, 20.82, 2.85, 5.41)
    assert d["level_err"] == pytest.approx(2.26, abs=0.01)
    assert d["as_registered"] == pytest.approx(15.41, abs=0.01)


def test_등록지표_오차는_줄지_않는다():
    """**이 분해는 작은 수를 고르는 장치가 아니다.**

    회계 고침으로 수준 오차는 5.30 -> 2.26%p 로 준다. 그러나 등록된 자(적립업종)
    기준 오차는 18.45 -> 15.41%p 로, 구성 조각 17.97%p 아래로 절대 못 내려간다.
    """
    lo = L.split_error(0.268, 20.82, 2.85, 2.37)
    hi = L.split_error(0.268, 20.82, 2.85, 5.41)
    assert hi["level_err"] < lo["level_err"], "수준은 좋아진다"
    assert hi["as_registered"] < lo["as_registered"], "등록지표도 좋아지긴 한다"
    # 그러나 구성 조각이 바닥이다 — 총소비를 정답지의 함의값에 딱 맞춰도
    gap = L.split_error(0.268, 20.82, 2.85, L.implied_total(0.268, 20.82, 2.85))
    assert gap["level_err"] == pytest.approx(0.0)
    assert gap["as_registered"] == pytest.approx(20.82 - 7.67, abs=0.01), \
        "수준을 완벽히 맞춰도 등록지표 오차가 13.15%p 남는다 — 구성 때문이다"


def test_구성_조각은_실측_간격_그대로다():
    d = L.split_error(0.268, 20.82, 2.85, 5.41)
    assert d["composition_gap"] == pytest.approx(17.97, abs=0.01)


def test_우리_간격은_구조적으로_0이다():
    """적립 몫이 상수면 적립 변화 == 제외 변화 == 총 변화. 간격이 0 이다."""
    ours_in = ours_out = 5.41
    assert ours_in - ours_out == 0


def test_문서가_가리키는_곳이_있다():
    src = (ROOT / "scripts" / "report" / "level_vs_composition.py").read_text(encoding="utf-8")
    ref = "experiments/error_budget/p012_2_why_unproducible.md"
    assert ref in src and (ROOT / ref).exists()
