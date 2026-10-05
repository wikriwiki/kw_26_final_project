"""**기준값에서 가른 식과 현행 식은 항등이어야 한다.** — diagnosis_04

한 상수(`ELIGIBLE_SHARE_SEOUL`)가 '앵커 과대 보정'과 '비사용처 분리' 두 가지를
함께 하고 있어서, 정책이 움직여야 할 쪽을 상수가 붙잡고 있었다. 그것을 가른다.

    현행   offline = anchor x 0.2535
    분리   offline = (anchor / 2.40) x share      share0 = 0.2535 x 2.40

**기준값에서 두 식은 같은 수를 낸다.** 그러므로 `EXP_SPLIT_ANCHOR=1` 만 켜고
에이전트가 배송 몫을 말하지 않으면(= online_share 없음) 결과가 현행과
구분되지 않아야 한다. 여기서 갈리면 수리가 수준을 건드린 것이고, 그러면
정책 효과와 뒤섞여 무엇이 무엇 때문인지 못 가른다.

같은 이유로 **모르면 안 움직인다** — online_share 가 None 이면 배수는 1.0 이다.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))


def _load(**env):
    """환경변수를 세우고 모듈을 다시 읽는다 — 상수가 임포트 때 굳기 때문."""
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        import consumption
        return importlib.reload(consumption)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _events(each: int = 24000):
    """계획액은 events 의 actual_spent 합에서 나온다(consumption.py:699).

    0 으로 두면 계획이 항상 져서 `max(앵커, 계획)` 의 어느 항이 이기는지를
    못 본다 — 그러면 항등 시험이 실제로는 아무것도 안 지킨다.
    """
    return [
        dict(category="식사", poi_id="A", actual_spent=each, policy_spend={},
             coupon_eligible=True, actual_satisfaction=0.9, price_factor=1),
        dict(category="마트", poi_id="B", actual_spent=each, policy_spend={},
             coupon_eligible=True, actual_satisfaction=0.9, price_factor=1),
    ]


def _run(mod, **kw):
    return mod.apply_consumption_model(
        _events(), daily=40000, income_tier="중", tendency="보통",
        balance=3_000_000, **kw)


@pytest.mark.parametrize("daily", [15000, 40000, 90000])
@pytest.mark.parametrize("each", [0, 24000, 60000])
def test_기준값에서_현행과_항등(daily, each):
    """켜기만 하고 배송 몫을 말하지 않으면 오프라인 금액이 같아야 한다.

    `each` 를 갈아 가며 **계획이 지는 구간과 이기는 구간을 모두** 지난다 —
    max 뒤에서 나누므로 어느 항이 이기든 항등이어야 한다.
    """
    base = _load(EXP_SPLIT_ANCHOR=None)
    a = base.apply_consumption_model(_events(each), daily=daily, income_tier="중",
                                     tendency="보통", balance=3_000_000)
    split = _load(EXP_SPLIT_ANCHOR="1")
    b = split.apply_consumption_model(_events(each), daily=daily, income_tier="중",
                                      tendency="보통", balance=3_000_000)
    _load(EXP_SPLIT_ANCHOR=None)
    # 오프라인(POI 원장으로 가는 돈) — 반올림 두 번 말고는 차이가 없어야 한다.
    assert abs(a["personal_total"] - b["personal_total"]) <= 3, (
        f"기준값에서 갈렸다: 현행 {a['personal_total']} vs 분리 {b['personal_total']}")


def test_배송몫을_말하지_않으면_안_움직인다():
    split = _load(EXP_SPLIT_ANCHOR="1")
    m = _run(split, online_share=None)
    _load(EXP_SPLIT_ANCHOR=None)
    assert m["online_share_source"] == "split_base"


def test_배송몫이_내려가면_오프라인이_커진다():
    """정책이 움직여야 할 방향 — 배송을 덜 하면 가게 돈이 는다."""
    split = _load(EXP_SPLIT_ANCHOR="1")
    hi = _run(split, online_share=0.40)   # 배송 많이
    lo = _run(split, online_share=0.05)   # 배송 적게
    _load(EXP_SPLIT_ANCHOR=None)
    assert lo["personal_total"] > hi["personal_total"], (
        f"배송 0.05 {lo['personal_total']} 가 0.40 {hi['personal_total']} 보다 커야 한다")
    assert hi["online_share_source"] == "split_behavioral"


def test_기준평균이_편차를_지운다():
    """online_share 가 기준 평균과 같으면 기준값 그대로여야 한다."""
    split = _load(EXP_SPLIT_ANCHOR="1", EXP_KEEP_MEAN="0.80")
    at_mean = _run(split, online_share=0.20)      # 1 - 0.20 = 0.80 = KEEP_MEAN
    none_ = _run(split, online_share=None)
    _load(EXP_SPLIT_ANCHOR=None, EXP_KEEP_MEAN=None)
    assert abs(at_mean["personal_total"] - none_["personal_total"]) <= 3


def test_꺼져_있으면_건드리지_않는다():
    """기본 경로는 한 글자도 안 달라져야 한다 — online_share 를 줘도 무시한다."""
    base = _load(EXP_SPLIT_ANCHOR=None)
    a = _run(base, online_share=None)
    b = _run(base, online_share=0.9)
    assert a["personal_total"] == b["personal_total"]
    assert a["online_share_source"] == "seoul_smallbiz"


def test_앞에서_나누면_계획이_이기기_시작한다():
    """BEFORE_MAX 는 **다른 실험**이다 — 수준이 오르는 것을 시험이 기억한다.

    라이브 3,090건에서 계획이 앵커를 이기는 비율이 0.7% → 63.9% 로 뒤집히고
    수준이 +40.7% 오른다. 그래서 기본은 꺼 두고, 켰을 때 정말 커지는지만 못 박는다.
    """
    after = _load(EXP_SPLIT_ANCHOR="1", EXP_ANCHOR_BEFORE_MAX=None)
    a = after.apply_consumption_model(
        _events(60000), daily=40000, income_tier="중", tendency="보통",
        balance=3_000_000)
    before = _load(EXP_SPLIT_ANCHOR="1", EXP_ANCHOR_BEFORE_MAX="1")
    b = before.apply_consumption_model(
        _events(60000), daily=40000, income_tier="중", tendency="보통",
        balance=3_000_000)
    _load(EXP_SPLIT_ANCHOR=None, EXP_ANCHOR_BEFORE_MAX=None)
    assert b["personal_total"] >= a["personal_total"]


def test_기본은_꺼져_있다():
    mod = _load(EXP_SPLIT_ANCHOR="1", EXP_ANCHOR_BEFORE_MAX=None)
    assert mod.EXP_ANCHOR_BEFORE_MAX is False
    _load(EXP_SPLIT_ANCHOR=None)
