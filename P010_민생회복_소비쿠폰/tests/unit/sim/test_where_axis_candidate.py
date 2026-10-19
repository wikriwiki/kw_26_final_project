"""v28 은 '있는 필요를 어디서 채울까' 한 축만 더한다 — 지출을 부추기지 않는다.

계획의 앵커가 residence 590 · workplace 183 · zone 93 이라, 동네에 걸린 조건은
하루에 닿지 못한다. v28 은 장소 선택을 열되, 없는 필요를 만들지 말라는 빗장을
그 자리에서 다시 건다. 둘 중 하나라도 빠지면 이 후보는 지출 유도가 된다.
"""
import importlib
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))

# 정책 이름·원하는 방향·수치 목표는 물론, 지출을 부추기는 낱말도 막는다.
FORBIDDEN = ('캐시백', '상생', '소비쿠폰', '지역화폐', '거리두기', '긴급재난', '적립', '실측',
             '지갑', '정책', '늘려', '줄여', '증가시', '감소시', '더 쓰', '많이 쓰', '외출을 늘', '%')


def prompt(name):
    return importlib.import_module('prompts.' + name).SYSTEM_PROMPT


def test_v28_names_no_policy_and_pushes_no_direction():
    text = prompt('v28')
    assert not [w for w in FORBIDDEN if w in text]
    assert not re.search(r'\d+\s*(원|퍼센트)', text)


def test_v28_opens_the_choice_of_place():
    v25, v28 = prompt('v25'), prompt('v28')
    assert '어디서 채울지도 고른다' in v28 and '어디서' not in v25
    assert '집·직장·동네' in v28


def test_the_guard_against_invented_needs_is_restated_beside_it():
    """장소를 열면서 빗장을 같이 걸지 않으면 없는 필요를 만들라는 말이 된다."""
    v28 = prompt('v28')
    assert '없는 필요를 만드는 것이 아니다' in v28
    assert '후보의 존재는 구매 의무가 아니다' in v28
    assert '활동 수를 채우려고 구매·외출·반복을 추가하지 않는다' in v28


def test_conditions_are_written_without_pointing_anywhere():
    """'일부에만 걸리기도 한다' — 어느 장소가 유리하다고 말하지 않는다."""
    line = next(l for l in prompt('v28').splitlines() if '일부에만' in l)
    for word in ('유리', '좋다', '권장', '우선'):
        assert word not in line


def test_only_the_place_axis_differs_from_v25():
    def stripped(name):
        return '\n'.join(l for l in prompt(name).splitlines()
                         if '어디서' not in l and '없는 필요를 만드는' not in l
                         and '필요를 지금 충족하거나' not in l)
    assert stripped('v28') == stripped('v25')


def test_planner_accepts_v28():
    src = importlib.import_module('validate_action_planner').__file__
    allow = re.search(r"prompt_module','v22'\) not in \{([^}]*)\}", open(src, encoding='utf-8').read()).group(1)
    assert "'v28'" in allow
