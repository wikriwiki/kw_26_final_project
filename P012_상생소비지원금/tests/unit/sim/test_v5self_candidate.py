"""후보 6 (v5self) 이 **v5 를 건드리지 않고 덧대기만** 하는지.

사전등록: experiments/plan_channel/prereg_v5self.md

가장 중요한 시험은 `test_v5_가_한_바이트도_안_바뀐다` 와
`test_정책_방향이나_수치를_말하지_않는다` 다. 앞의 것은 후보 비교가 성립하는
조건이고, 뒤의 것은 사용자가 못 박은 금지 규칙이다 — v7·v9 가 그 규칙으로
제외됐다.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))


@pytest.fixture(scope="module")
def mods():
    v5 = importlib.import_module("prompts.v5")
    v5self = importlib.import_module("prompts.v5self")
    return v5, v5self


def test_v5_가_한_바이트도_안_바뀐다(mods):
    """v5 는 홀드아웃까지 끝난 확정판이다. 후보가 원본을 바꾸면 비교가 깨진다."""
    v5, _ = mods
    got = hashlib.sha256(v5.SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
    assert got == "250311b63adbc4f6", "v5 의 sha256 앞 16자리가 골든값과 달라졌다"


def test_덧댄_것이_한_문장뿐이다(mods):
    v5, v5self = mods
    assert len(v5self.SYSTEM_PROMPT) > len(v5.SYSTEM_PROMPT)
    added = len(v5self.SYSTEM_PROMPT) - len(v5.SYSTEM_PROMPT)
    assert added < 120, "한 문장보다 커지면 무엇이 들었는지 다시 봐야 한다 (+%d자)" % added


def test_v5의_모든_줄이_그대로_남는다(mods):
    """덧대기다. 지우거나 바꾼 줄이 있으면 안 된다."""
    v5, v5self = mods
    for line in v5.SYSTEM_PROMPT.splitlines():
        s = line.strip()
        if len(s) < 12:
            continue
        assert s in v5self.SYSTEM_PROMPT, "v5 의 줄이 사라졌다: %s" % s[:40]


def test_정책_방향이나_수치를_말하지_않는다(mods):
    """**사용자가 못 박은 규칙.** v7·v9 가 이 규칙으로 제외됐다."""
    v5, v5self = mods
    added = v5self.SYSTEM_PROMPT.replace(v5.SYSTEM_PROMPT[:200], "")
    new = v5self._SELF_SCALE
    for word in ("정책", "지원금", "캐시백", "적립", "환급", "문턱", "쿠폰"):
        assert word not in new, "덧댄 문장이 정책을 언급한다: %s" % word
    for word in ("늘린다", "줄인다", "더 쓴다", "덜 쓴다", "올린다", "내린다"):
        assert word not in new, "덧댄 문장이 방향을 지시한다: %s" % word
    assert not any(c.isdigit() for c in new), "덧댄 문장에 수치가 있다"


def test_양쪽으로_열려_있다(mods):
    """'벗어난다' 는 위아래 모두다. 한쪽만 열면 방향 지시가 된다."""
    _, v5self = mods
    assert "벗어나는지" in v5self._SELF_SCALE
    assert "높" not in v5self._SELF_SCALE and "낮" not in v5self._SELF_SCALE


def test_기준을_사람에게_묶는다(mods):
    """겨냥한 결함은 r=-0.155 다 — 척도가 사람에게 안 묶여 있다."""
    _, v5self = mods
    assert "소비수준" in v5self._SELF_SCALE and "소득" in v5self._SELF_SCALE


def test_덧댄_자리가_소비성향_블록_안이다(mods):
    """총액으로 가는 통로는 daily_propensity 다. 딴 데 붙으면 겨냥이 빗나간다."""
    _, v5self = mods
    i = v5self.SYSTEM_PROMPT.index(v5self._SELF_SCALE)
    head = v5self.SYSTEM_PROMPT[:i]
    assert "daily_propensity" in head
    assert head.rindex("daily_propensity") > head.rindex("[") - 200
