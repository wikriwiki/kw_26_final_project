"""계획의 반응이 총액에 **통과해야 한다** — diagnosis_05

정책은 계획 금액으로 닿고 있었다(라운드2 런에서 +9.76%, p=0.0015). 그런데
`max(앵커, 계획)` 에서 계획이 이기는 경우가 23.7% 뿐이라 반응의 4분의 3 이
앵커에 먹혔다(총액 +2.37%).

**여기서 가장 중요한 시험은 `test_앵커가_약분되지_않는다` 다.** 처음에 기준선을
`REF x anchor` 로 잡았더니 `anchor x plan/(REF x anchor) = plan/REF` 로 앵커가
**약분돼 사라졌다.** 수치 추정은 맞는데 해석이 틀렸던 것이고, 그대로 돌렸으면
BDC 계층 근거를 버린 줄 모르고 결과를 읽었을 것이다. 시험이 잡았다.
"""
from __future__ import annotations

import importlib
import io
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))


def _load(**env):
    old = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        import consumption
        return importlib.reload(consumption)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _ev(each):
    return [dict(category="식사", poi_id="A", actual_spent=each, policy_spend={},
                 coupon_eligible=True, actual_satisfaction=0.9, price_factor=1),
            dict(category="마트", poi_id="B", actual_spent=each, policy_spend={},
                 coupon_eligible=True, actual_satisfaction=0.9, price_factor=1)]


def _run(mod, each, daily=40000, aid="A1"):
    return mod.apply_consumption_model(_ev(each), daily=daily, income_tier="중",
                                       tendency="보통", balance=5_000_000, aid=aid)


@pytest.fixture
def basefile(tmp_path):
    p = tmp_path / "plan_baseline.json"
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps({"A1": 20000.0, "A2": 20000.0}))
    return str(p)


def test_꺼져_있으면_현행_그대로(basefile):
    base = _load(EXP_PLAN_DRIVES_TOTAL=None, EXP_PLAN_BASELINE_FILE=basefile)
    a = _run(base, 20000)
    # meta 의 personal_total 은 온라인 몫이 빠진 뒤다. 비율로 되돌려 맞댄다.
    pre = a["personal_total"] + a["online_total"]
    assert abs(pre - max(a["anchor_total"], int(round(a["planned_total"])))) <= 2


def test_앵커가_약분되지_않는다(basefile):
    """**같은 계획, 다른 앵커 → 총액이 달라야 한다.** 이 시험이 설계 오류를 잡았다."""
    on = _load(EXP_PLAN_DRIVES_TOTAL="1", EXP_PLAN_BASELINE_FILE=basefile)
    lo = _run(on, 10000, daily=20000)
    hi = _run(on, 10000, daily=80000)
    _load(EXP_PLAN_DRIVES_TOTAL=None, EXP_PLAN_BASELINE_FILE=None)
    assert hi["anchor_total"] > lo["anchor_total"]
    assert hi["personal_total"] > lo["personal_total"], \
        "앵커가 약분됐다 — BDC 계층 근거가 날아간다"


def test_계획이_커지면_총액이_따라_커진다(basefile):
    off = _load(EXP_PLAN_DRIVES_TOTAL=None, EXP_PLAN_BASELINE_FILE=basefile)
    a_lo, a_hi = _run(off, 8000), _run(off, 12000)
    on = _load(EXP_PLAN_DRIVES_TOTAL="1", EXP_PLAN_BASELINE_FILE=basefile)
    b_lo, b_hi = _run(on, 8000), _run(on, 12000)
    _load(EXP_PLAN_DRIVES_TOTAL=None, EXP_PLAN_BASELINE_FILE=None)
    assert a_lo["personal_total"] == a_hi["personal_total"], "현행이 이미 반응한다"
    assert b_hi["personal_total"] > b_lo["personal_total"], "반응이 여전히 먹힌다"


def test_기준선이_없는_사람은_현행_그대로(basefile):
    """기준선 파일에 없는 에이전트는 건드리지 않는다 — 조용히 바뀌면 안 된다."""
    on = _load(EXP_PLAN_DRIVES_TOTAL="1", EXP_PLAN_BASELINE_FILE=basefile)
    known = _run(on, 8000, aid="A1")
    unknown = _run(on, 8000, aid="없는사람")
    off = _load(EXP_PLAN_DRIVES_TOTAL=None, EXP_PLAN_BASELINE_FILE=None)
    plain = _run(off, 8000, aid="없는사람")
    assert unknown["personal_total"] == plain["personal_total"]
    assert known["personal_total"] != plain["personal_total"]


def test_파일이_없으면_현행_그대로(tmp_path):
    on = _load(EXP_PLAN_DRIVES_TOTAL="1",
               EXP_PLAN_BASELINE_FILE=str(tmp_path / "없는파일.json"))
    a = _run(on, 8000)
    off = _load(EXP_PLAN_DRIVES_TOTAL=None, EXP_PLAN_BASELINE_FILE=None)
    b = _run(off, 8000)
    assert a["personal_total"] == b["personal_total"]


def test_클램프가_극단을_막는다(basefile):
    on = _load(EXP_PLAN_DRIVES_TOTAL="1", EXP_PLAN_BASELINE_FILE=basefile)
    huge = _run(on, 900000)
    a, sc = huge["anchor_total"], on.PLAN_SCALE
    hi = on.PLAN_CLAMP_HI
    _load(EXP_PLAN_DRIVES_TOTAL=None, EXP_PLAN_BASELINE_FILE=None)
    assert huge["personal_total"] <= a * hi * sc * 1.01


def test_클램프는_정답지를_안_보고_정했다():
    """앵커 순위상관을 현행의 90% 이상 지키는 가장 넓은 구간이 [0.5, 2.0] 이었다."""
    m = _load(EXP_PLAN_DRIVES_TOTAL="1")
    assert (m.PLAN_CLAMP_LO, m.PLAN_CLAMP_HI) == (0.5, 2.0)
    _load(EXP_PLAN_DRIVES_TOTAL=None)
