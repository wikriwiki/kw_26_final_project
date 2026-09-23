"""v5online 은 **v5 에 항목 하나만 덧댄 것**이어야 한다 — diagnosis_04.

잠겨 있던 `online_share` 를 되살리는 변형이다. 두 가지를 못 박는다.

    1  v5 는 한 글자도 안 달라진다 — 정답지 라인의 선택이 v5 이므로,
       여기서 v5 가 오염되면 지금까지의 판정이 전부 무효가 된다.
    2  덧댄 블록에 **정책 낱말도 방향도 수치도 없다.** 묻는 것은
       "오늘 산 것을 어디서 샀나" 하나다. 정책 사실은 이미 맥락에 있고,
       거기서 무엇을 추론할지가 재려는 것이다 — 답을 적어 주면 잴 것이 없다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))

from prompts import get, list_variants  # noqa: E402

V5 = get("v5").SYSTEM_PROMPT
ON = get("v5online").SYSTEM_PROMPT
BLOCK = ON.split("[배송으로 받는 몫")[1].split("[출력 형식]")[0]


def test_등록돼_있다():
    assert "v5online" in list_variants()


def test_v5_는_오염되지_않았다():
    assert "online_share" not in V5


def test_덧댄_것은_블록_하나와_예시_한_항목뿐이다():
    """v5online 에서 덧댄 두 곳을 도로 떼면 v5 와 바이트가 같아야 한다."""
    back = ON.replace(
        '"daily_propensity": 0.72, "grant_kept_share": 0.8, "online_share": 0.25}',
        '"daily_propensity": 0.72, "grant_kept_share": 0.8}', 1)
    back = back.replace(
        "( daily_propensity·grant_kept_share·online_share에 적힌 숫자도",
        "( daily_propensity·grant_kept_share에 적힌 숫자도", 1)
    back = back.replace("[배송으로 받는 몫" + BLOCK, "", 1)
    assert back == V5, "v5 본문이 함께 달라졌다 — 덧대기가 아니라 고쳐 쓴 것이다"


def test_블록에_정책_낱말이_없다():
    for w in ("정책", "쿠폰", "캐시백", "지원금", "상생", "적립", "환급", "혜택"):
        assert w not in BLOCK, f"블록에 정책 낱말 '{w}' 가 들어갔다 — 유도가 된다"


def test_블록이_방향을_지시하지_않는다():
    for w in ("늘리", "줄이", "더 많이", "덜 ", "바람직", "권장", "해야 한다"):
        assert w not in BLOCK, f"블록이 방향을 지시한다: '{w}'"


def test_블록이_묻는_것은_행동_하나다():
    assert "`online_share` (0~1)" in BLOCK
    assert "배송" in BLOCK
    # 0 과 1 의 뜻은 스키마 설명이므로 허용하되, 그 밖의 수치는 없어야 한다.
    import re
    nums = set(re.findall(r"\d+(?:\.\d+)?", BLOCK)) - {"0", "1"}
    assert not nums, f"블록에 수치가 들어갔다: {sorted(nums)}"
