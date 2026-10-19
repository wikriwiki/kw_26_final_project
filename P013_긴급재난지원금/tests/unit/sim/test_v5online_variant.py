"""v5 파생 변형들은 **v5 에 항목 하나만 덧댄 것**이어야 한다 — diagnosis_04.

잠겨 있던 `online_share` 를 되살리는 변형 둘을 함께 지킨다.

    v5online   "배송으로 받는 몫"          — 교정 탐침 평균 0.14
    v5offsite  "동네 가게가 아닌 곳의 몫"   — 회계가 걷어내는 0.7465 와 범위를 맞춘 판

못 박는 것은 둘 다 같다.

    1  v5 는 한 글자도 안 달라진다 — 정답지 라인의 선택이 v5 이므로,
       여기서 v5 가 오염되면 지금까지의 판정이 전부 무효가 된다.
    2  덧댄 블록에 **정책 낱말도 방향도 수치도 없다.** 묻는 것은 "오늘 산 것을
       어디서 샀나" 하나다. 정책 사실은 이미 맥락에 있고, 거기서 무엇을
       추론할지가 재려는 것이다 — 답을 적어 주면 잴 것이 없다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))

from prompts import get, list_variants  # noqa: E402

V5 = get("v5").SYSTEM_PROMPT

# (변형 이름, 블록 머리말, 예시에 박은 값)
VARIANTS = [
    ("v5online", "[배송으로 받는 몫", "0.25"),
    ("v5offsite", "[동네 가게가 아닌 곳의 몫", "0.4"),
]


def _block(name: str, head: str) -> str:
    s = get(name).SYSTEM_PROMPT
    assert head in s, f"{name} 에 블록 머리말이 없다"
    return s.split(head)[1].split("[출력 형식]")[0]


@pytest.mark.parametrize("name,head,ex", VARIANTS)
def test_등록돼_있다(name, head, ex):
    assert name in list_variants()


def test_v5_는_오염되지_않았다():
    assert "online_share" not in V5


@pytest.mark.parametrize("name,head,ex", VARIANTS)
def test_덧댄_것은_블록_하나와_예시_한_항목뿐이다(name, head, ex):
    """덧댄 두 곳을 도로 떼면 v5 와 바이트가 같아야 한다."""
    s = get(name).SYSTEM_PROMPT
    back = s.replace(
        f'"daily_propensity": 0.72, "grant_kept_share": 0.8, "online_share": {ex}}}',
        '"daily_propensity": 0.72, "grant_kept_share": 0.8}', 1)
    back = back.replace(
        "( daily_propensity·grant_kept_share·online_share에 적힌 숫자도",
        "( daily_propensity·grant_kept_share에 적힌 숫자도", 1)
    back = back.replace(head + _block(name, head), "", 1)
    assert back == V5, f"{name}: v5 본문이 함께 달라졌다 — 덧대기가 아니라 고쳐 쓴 것이다"


@pytest.mark.parametrize("name,head,ex", VARIANTS)
def test_블록에_정책_낱말이_없다(name, head, ex):
    blk = _block(name, head)
    for w in ("정책", "쿠폰", "캐시백", "지원금", "상생", "적립", "제외업종", "환급", "혜택"):
        assert w not in blk, f"{name} 블록에 정책 낱말 '{w}' — 유도가 된다"


@pytest.mark.parametrize("name,head,ex", VARIANTS)
def test_블록이_방향을_지시하지_않는다(name, head, ex):
    blk = _block(name, head)
    for w in ("늘리", "줄이", "더 많이", "바람직", "권장", "해야 한다"):
        assert w not in blk, f"{name} 블록이 방향을 지시한다: '{w}'"


@pytest.mark.parametrize("name,head,ex", VARIANTS)
def test_블록이_묻는_것은_행동_하나다(name, head, ex):
    blk = _block(name, head)
    assert "`online_share` (0~1)" in blk
    # 0 과 1 의 뜻은 스키마 설명이므로 허용하되, 그 밖의 수치는 없어야 한다.
    nums = set(re.findall(r"\d+(?:\.\d+)?", blk)) - {"0", "1"}
    assert not nums, f"{name} 블록에 수치가 들어갔다: {sorted(nums)}"


def test_두_변형은_묻는_범위가_다르다():
    """같은 필드를 묻지만 **범위가 다르다** — 그것이 이 둘을 가르는 이유다."""
    a = _block("v5online", "[배송으로 받는 몫")
    b = _block("v5offsite", "[동네 가게가 아닌 곳의 몫")
    assert "대형마트" not in a and "백화점" not in a, "좁은 판이 넓어졌다"
    assert "대형마트" in b and "백화점" in b, "넓은 판이 매장 종류를 안 말한다"
