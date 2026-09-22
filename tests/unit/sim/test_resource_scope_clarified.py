"""v32 는 이미 있는 자원 검사 지시의 적용 범위만 분명히 한다 — 해법을 지정하지 않는다.

재고 0인데 집밥 두 끼를 계획해 게이트가 96칸 중 46칸을 막았다. 프롬프트는 이미
자원을 확인하라고 말하고 있고, 모델은 주입된 활동(세탁)에는 지키고 자기가 고른
활동(끼니)에는 안 지킨다.
"""
import importlib
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))

OLD = 'daily_conditions가 있으면 활동에 필요한 자원을 실제 사용 전 확보할 수 있는지 확인한다.'
NEW = ('daily_conditions가 있으면 배치하는 모든 활동에 대해 '
       '필요한 자원을 실제 사용 전 확보할 수 있는지 확인한다.')


def prompt(name):
    return importlib.import_module('prompts.' + name).SYSTEM_PROMPT


def test_only_the_scope_phrase_is_added():
    assert prompt('v25').replace(OLD, NEW) == prompt('v32')


def test_it_does_not_say_how_to_solve_it():
    """채우든, 나가든, 시키든, 거르든 다 되어야 한다. 하나를 지목하면 유도다."""
    v32 = prompt('v32')
    for word in ('장을 보', '나가서', '외식', '배달을 시', '먼저 사'):
        assert word not in v32, word


def test_the_existing_instruction_is_not_weakened():
    v32 = prompt('v32')
    assert '실제 사용 전 확보할 수 있는지 확인한다' in v32
    assert '물품을 주문해도 명시된 도착 전에는 사용할 수 없다' in v32
    assert '같은 재고를 여러 번 사용하거나 미래 입금을 당겨 쓰지 않는다' in v32


def test_no_policy_or_sector_or_number_leaks_in():
    v32 = prompt('v32')
    for word in ('캐시백', '지역화폐', '거리두기', '지갑', '정책', '실측', '음식점', '카페'):
        assert word not in v32
    assert not re.search(r'\d+\s*(원|퍼센트|%)', v32)


def test_planner_accepts_v32():
    src = importlib.import_module('validate_action_planner').__file__
    allow = re.search(r"prompt_module','v22'\) not in \{([^}]*)\}", open(src, encoding='utf-8').read()).group(1)
    assert "'v32'" in allow
