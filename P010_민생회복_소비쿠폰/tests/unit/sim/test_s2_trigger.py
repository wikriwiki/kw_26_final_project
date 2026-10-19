"""Stage1 이 붙인 `trigger` 를 Stage2 로 넘긴다 — 후보 3.

Stage2 의 이벤트 줄에는 시각·앵커·업종·의도·가격참고값이 있고 **trigger 가
빠져 있었다.** 그런데 Stage1 은 제도 때문에 만든 이벤트에 trigger:"policy" 를
붙인다. 그 표시가 도착하지 않아 Stage2 는 평범한 구매로 값을 매긴다.

지키는 것:
    1  꺼져 있으면 줄이 한 글자도 안 달라진다
    2  켜도 **reasoning 은 절대 안 넘어간다** — 거기엔 정책 문장이 들어 있다
    3  trigger 가 없거나 "none" 이면 아무것도 안 붙는다
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))


def _load(flag):
    old = os.environ.get("EXP_S2_TRIGGER")
    if flag is None:
        os.environ.pop("EXP_S2_TRIGGER", None)
    else:
        os.environ["EXP_S2_TRIGGER"] = flag
    try:
        import stage2_poi
        return importlib.reload(stage2_poi)
    finally:
        if old is None:
            os.environ.pop("EXP_S2_TRIGGER", None)
        else:
            os.environ["EXP_S2_TRIGGER"] = old


class _Ev:
    def __init__(self, trigger=None, reasoning=None):
        self.time = "18:40"
        self.anchor = "zone:11680103"
        self.category = "쇼핑"
        self.sub_category = "가전"
        self.intent = "드라이기 바꾸기"
        self.trigger = trigger
        self.reasoning = reasoning


CANDS = [{"poi_id": "P1", "name": "전자마트", "km": 0.4, "known": False,
          "price_band": "₩₩", "avg_satisfaction": 0.8, "visit_count": 0,
          "unit_anchor": 50000, "durable_anchor": True}]

REASON = "문턱까지 얼마 안 남아 이번 달에 사면 캐시백까지 받겠다 싶어 들름"


def _line(mod, ev):
    return mod._format_event_with_candidates(1, ev, CANDS, set())


def test_꺼져_있으면_안_붙는다():
    m = _load(None)
    line = _line(m, _Ev(trigger="policy", reasoning=REASON))
    assert "계기:" not in line


def test_켜면_한_낱말만_붙는다():
    m = _load("1")
    line = _line(m, _Ev(trigger="policy", reasoning=REASON))
    _load(None)
    assert "계기:policy" in line


def test_reasoning_은_절대_안_넘어간다():
    """정책 문장이 넘어가면 그건 방향 지시다 — 켜든 끄든 넘어가면 안 된다."""
    for flag in (None, "1"):
        m = _load(flag)
        line = _line(m, _Ev(trigger="policy", reasoning=REASON))
        assert "캐시백" not in line, "reasoning 이 새어 나갔다"
        assert "문턱" not in line
    _load(None)


def test_trigger_가_없으면_안_붙는다():
    m = _load("1")
    assert "계기:" not in _line(m, _Ev(trigger=None))
    assert "계기:" not in _line(m, _Ev(trigger="none"))
    _load(None)


def test_켜고_끈_차이가_그_낱말뿐이다():
    off = _line(_load(None), _Ev(trigger="policy"))
    on = _line(_load("1"), _Ev(trigger="policy"))
    _load(None)
    assert on.replace(" | 계기:policy", "", 1) == off


def test_탐침과_코드가_같은_낱말을_넣는다():
    """탐침은 돌고 있는 코드를 못 고쳐 문자열을 끼운다 — 문구가 같아야 한다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s2tp", Path(__file__).resolve().parents[3] / "scripts/sim/s2_trigger_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    m = _load("1")
    line = _line(m, _Ev(trigger="policy"))
    _load(None)
    assert probe.TAG in line, f"탐침이 넣는 {probe.TAG!r} 를 코드가 안 만든다"
    assert probe.TAG == " | 계기:policy"


def test_대조군_이벤트에는_안_붙는다():
    """lifestyle 이벤트에도 계기가 붙으면 대조군이 무너진다."""
    m = _load("1")
    line = _line(m, _Ev(trigger="lifestyle"))
    _load(None)
    assert "계기:lifestyle" in line, "코드는 lifestyle 도 붙인다 — 탐침 대조군 설계와 맞춰야 한다"
