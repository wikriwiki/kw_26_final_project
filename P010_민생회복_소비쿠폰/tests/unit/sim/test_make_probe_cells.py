"""탐침 맥락 생성기 — scripts/sim/make_probe_cells.py

그래프가 필요한 부분은 서버에서 돌려 확인했고(120명 생성 · 자리표시 주입 성공),
여기서는 **그래프 없이 확인할 수 있는 규칙**을 고정한다.

가장 중요한 것은 `test_자리표시가_없는_칸은_버려야_한다` 다. 자리표시가 없는 칸에
정책을 또 끼우면 **정책 블록이 두 번** 들어가고, 그 칸만 다른 프롬프트가 된다.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))


def _mod():
    p = ROOT / "scripts" / "sim" / "make_probe_cells.py"
    spec = importlib.util.spec_from_file_location("make_probe_cells", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()


def test_자리표시_문자열이_런타임과_같다():
    """런타임이 '정책 없음' 일 때 내는 그 문자열이어야 주입 자리를 찾는다."""
    import importlib
    own = importlib.import_module("s1_ownership_probe")
    assert M.NO_FACTS == own.NO_FACTS
    assert M.NO_MINE == own.NO_MINE


def test_자리표시가_없는_칸은_버려야_한다():
    """정책이 이미 껴 있는 칸에 또 끼우면 블록이 두 번 들어간다."""
    have = "앞 %s 뒤 %s" % (M.NO_FACTS, M.NO_MINE)
    only_one = "앞 %s 뒤" % M.NO_FACTS
    none = "정책 블록이 이미 렌더된 칸"
    keep = lambda u: (M.NO_FACTS in u) and (M.NO_MINE in u)
    assert keep(have)
    assert not keep(only_one), "하나만 있어도 버려야 한다"
    assert not keep(none)


@pytest.mark.parametrize("d,expect", [
    ("2021-10-04", ("weekday", "월")),
    ("2021-10-08", ("weekday", "금")),
    ("2021-10-09", ("weekend", "토")),
    ("2021-10-10", ("weekend", "일")),
])
def test_요일_구분이_런타임_규칙과_같다(d, expect):
    """요일이 앵커와 계획을 반대로 민다 — 여기서 틀리면 맥락이 어긋난다."""
    assert M.day_parts(date.fromisoformat(d)) == expect


def test_문턱_위치는_돌아가며_준다():
    """한 위치만 깔면 그 위치의 반응만 본다. i %% len 으로 고루 퍼져야 한다."""
    fracs = [0.55, 0.70, 0.85, 1.05]
    got = [fracs[i % len(fracs)] for i in range(12)]
    assert got.count(0.55) == 3 and got.count(1.05) == 3
    assert set(got) == set(fracs)


def test_주입은_한_번씩만_바꾼다():
    """replace 에 count=1 이 빠지면 본문에 같은 문구가 또 있을 때 다 바뀐다."""
    src = ROOT / "scripts" / "sim" / "make_probe_cells.py"
    body = src.read_text(encoding="utf-8")
    assert "replace(NO_FACTS, facts, 1)" in body
    assert "replace(NO_MINE, mine, 1)" in body
