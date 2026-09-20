"""v25/v26/v27 differ on one axis only, and none of them points at an answer.

The axis is how the prompt treats institutions. v26 adds an explicit read; v27
removes the clause entirely as the lower bound. Everything else must be identical,
otherwise a difference between them cannot be attributed to the axis.
"""
import importlib
import re

import pytest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))

NAMES = ('v25', 'v26', 'v27')
CLAUSE = '제도와 사회 배경을 함께 고려한다.'

# A candidate may describe a mechanism, never the outcome we measure.
FORBIDDEN = ('캐시백', '상생', '소비쿠폰', '지역화폐', '거리두기', '긴급재난', '적립',
             '실측', '늘려', '줄여', '증가시', '감소시', '%')


def prompt(name):
    return importlib.import_module('prompts.' + name).SYSTEM_PROMPT


@pytest.mark.parametrize('name', NAMES)
def test_candidate_names_no_policy_and_no_target(name):
    text = prompt(name)
    assert not [w for w in FORBIDDEN if w in text]
    assert not re.search(r'\d+\s*(원|퍼센트)', text)


def test_v26_adds_the_read_and_v27_removes_the_clause():
    v25, v26, v27 = (prompt(n) for n in NAMES)
    assert CLAUSE in v25 and CLAUSE in v26 and CLAUSE not in v27
    assert '무엇이 달라졌는지' in v26 and '무엇이 달라졌는지' not in v25
    assert '닿지 않으면 평소와 같다' in v26


def test_the_three_directions_are_written_symmetrically():
    """None of the three may be phrased as an increase or a decrease."""
    v26 = prompt('v26')
    line = next(l for l in v26.splitlines() if '오늘의 조건을 바꾼다' in l)
    # 하고 · 하고 · 한다 로 어미만 다르다. 어간이 세 번인지를 본다.
    assert line.count('달라지기도') == 3
    assert '늘기도' not in line and '줄기도' not in line


def test_only_the_institution_axis_differs():
    """Strip the axis from each candidate; the remainder must be identical."""
    def stripped(name):
        text = prompt(name)
        keep = [l for l in text.splitlines()
                if '제도' not in l and '무엇이 달라졌는지' not in l and '장소와 상품' not in l]
        return '\n'.join(keep)

    base = stripped('v25')
    assert stripped('v26') == base
    assert stripped('v27') == base


def test_planner_accepts_every_registered_candidate():
    source = (importlib.import_module('validate_action_planner').__file__)
    text = open(source, encoding='utf-8').read()
    allow = re.search(r"prompt_module','v22'\) not in \{([^}]*)\}", text).group(1)
    for name in NAMES:
        assert f"'{name}'" in allow
