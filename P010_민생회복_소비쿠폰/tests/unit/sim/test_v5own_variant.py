"""v5own 은 **한 줄만 덧댄 것**이고 감쇠를 빼지 않았다 — 후보 2.

덧대는 줄은 "적립형에는 고를 결제수단이 없다" 는 **사실**과 제도가 닿는 자리가
어디인지라는 **구조**만 적는다. 여기서 지키는 것:

    1  v5 는 한 글자도 안 달라진다 (정답지 라인의 선택이 v5 다)
    2  덧댄 줄을 도로 떼면 v5 와 바이트가 같다
    3  **감쇠 두 줄이 그대로 있다** — 빼면 "정책 있으면 무조건 더 쓴다" 는
       환각이 돌아온다. 그건 맞힌 게 아니라 망가뜨린 것이다
    4  방향·수치가 없다
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))

from prompts import get, list_variants  # noqa: E402
from prompts.v5own import OWN_LINE, _ANCHOR  # noqa: E402

V5 = get("v5").SYSTEM_PROMPT
OWN = get("v5own").SYSTEM_PROMPT


def test_등록돼_있다():
    assert "v5own" in list_variants()


def test_v5_는_오염되지_않았다():
    assert "고를 결제수단이 없다" not in V5


def test_덧댄_줄을_떼면_v5_와_같다():
    assert OWN.replace(OWN_LINE, "", 1) == V5, "한 줄 덧대기가 아니라 고쳐 쓴 것이다"


def test_감쇠를_빼지_않았다():
    """감쇠 문장의 **개수까지** 같아야 한다 — 슬쩍 지우면 잡는다."""
    assert OWN.count("기계적으로") == V5.count("기계적으로")
    assert V5.count("기계적으로") >= 2, "애초에 감쇠가 둘이라는 전제가 깨졌다"
    # 실제 본문 그대로다. **둘 다 대칭**이다 — 올리는 쪽만 막는 것이 아니라
    # 내리는 쪽도 막는다. (한쪽으로만 걸렸다고 적었던 것은 내 착각이었다.)
    # 본문에서 두 문장 모두 "기계적으로" 뒤에 줄바꿈이 온다. 공백을 눌러 맞댄다.
    flat = " ".join(OWN.split())
    for frag in ("기계적으로 더하거나 빼지 않는다", "기계적으로 올리거나 내리지는 않는다"):
        assert frag in flat, f"감쇠 '{frag}' 가 사라졌다"


def test_덧댄_줄에_방향이_없다():
    for w in ("더 써", "늘리", "많이 쓰", "권장", "해야 한다", "바람직", "적극"):
        assert w not in OWN_LINE, f"방향을 지시한다: {w}"


def test_덧댄_줄에_수치가_없다():
    """단계 이름(Stage2)의 숫자는 수치가 아니다."""
    assert not re.findall(r"\d", OWN_LINE.replace("Stage2", "")), "수치가 들어갔다"


def test_판단_기준을_기존_문장으로_되돌린다():
    """새 기준을 만들지 않는다 — 이 문장이 없으면 그 줄만 따로 노는 지시가 된다."""
    assert "위의 기준을 그대로 따른다" in OWN_LINE


def test_닻이_본문에_그대로_있다():
    assert _ANCHOR in V5 and _ANCHOR in OWN
