"""v5when — 적립형에 **빠져 있던 채널**을 적는다. 후보 4.

이벤트를 만드는 대목이 통째로 "오늘 돈이 들어오는 정책" 을 전제한다.
마지막 문장이 특히 문제다 — "목돈이 없으면 그런 일은 다음으로 미루고 소액
지출로 버틴다." 적립형에는 오늘 들어오는 돈이 **없으므로**, 그대로 읽으면
제도의 기전과 **반대 방향**으로 간다.

지키는 것:
    1  v5 는 한 글자도 안 달라진다
    2  덧댄 줄만 떼면 v5 와 바이트가 같다
    3  **감쇠 두 줄이 그대로 있다**
    4  방향 지시·수치가 없고, **반응하지 않는 경우를 명시**한다
    5  덧대는 자리가 그 "목돈이 없으면…" 문장 **바로 뒤**다
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))

from prompts import get, list_variants  # noqa: E402
from prompts.v5when import WHEN_LINE, _ANCHOR  # noqa: E402

V5 = get("v5").SYSTEM_PROMPT
W = get("v5when").SYSTEM_PROMPT


def test_등록돼_있다():
    assert "v5when" in list_variants()


def test_v5_는_오염되지_않았다():
    assert "때가 달라진 것" not in V5


def test_덧댄_줄만_떼면_v5():
    assert W.replace(WHEN_LINE, "", 1) == V5


def test_감쇠를_빼지_않았다():
    assert W.count("기계적으로") == V5.count("기계적으로") == 2
    flat = " ".join(W.split())
    for frag in ("기계적으로 더하거나 빼지 않는다", "기계적으로 올리거나 내리지는 않는다"):
        assert frag in flat


def test_방향_지시가_없다():
    for w in ("더 써", "늘리", "많이 쓰", "권장", "바람직", "해야 한다", "적극"):
        assert w not in WHEN_LINE, f"방향을 지시한다: {w}"


def test_수치가_없다():
    assert not re.findall(r"\d", WHEN_LINE), "수치가 들어갔다"


def test_반응하지_않는_경우를_명시한다():
    """이것이 없으면 '무조건 당겨라' 가 된다."""
    assert "그대로다" in WHEN_LINE
    assert "형편이 안 되면" in WHEN_LINE


def test_문제의_문장_바로_뒤에_붙는다():
    """'목돈이 없으면 미루고 버틴다' 를 그대로 두고 그 뒤에 예외를 단다.

    그 문장을 **지우지 않는다** — 지갑형에는 맞는 말이고, 지우면 지갑형
    정책(P010)의 행동이 함께 달라진다.
    """
    assert _ANCHOR in V5 and _ANCHOR in W
    i = W.index(_ANCHOR) + len(_ANCHOR)
    assert W[i:i + len(WHEN_LINE)] == WHEN_LINE


def test_돈이_생긴다고_말하지_않는다():
    """캐시백은 오늘 돈을 주지 않는다. 준다고 적으면 사실이 아니다."""
    assert "돈이 늘어난 것이 아니라" in WHEN_LINE
