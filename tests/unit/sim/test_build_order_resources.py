"""v33 은 계획을 짜는 순서에 자원 확보를 넣는다 — 해법을 지정하지 않는다.

모델은 이미 안다. 재고 0인 48칸 중 22칸의 사고에 groceries 가, 26칸에 delivery 가
나온다. 그러고도 계획에는 공짜 집밥 두 끼가 들어간다. 어구(v10)도 자리(v11)도
아니었고, 남은 것은 구성 순서를 정하는 규칙에 자원이 빠져 있다는 것이다.
"""
import importlib
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))

OLD = '먼저 확정된 일정과 이동 시간을 배치하고, 그 전후에 필요한 일상과 저녁 마무리를 정한다.'
NEW = ('먼저 확정된 일정과 이동 시간을 배치하고, 자원이 있어야 하는 일은 그 자원을 확보하는 일과 '
       '함께 배치하거나 자원이 필요 없는 다른 방법을 고르고, 그 전후에 필요한 일상과 저녁 마무리를 정한다.')


def prompt(name):
    return importlib.import_module('prompts.' + name).SYSTEM_PROMPT


def test_only_the_ordering_sentence_changes():
    assert prompt('v25').replace(OLD, NEW) == prompt('v33')


def test_both_ways_out_stay_open():
    """확보하는 것도, 자원이 필요 없는 다른 방법도 허용해야 한다."""
    v33 = prompt('v33')
    assert '자원을 확보하는 일과 함께 배치하거나' in v33
    assert '자원이 필요 없는 다른 방법을 고르고' in v33


def test_it_names_no_purchase_and_no_place():
    v33 = prompt('v33')
    for word in ('장보', '마트', '배달', '음식점', '카페', '외식', '사라', '나가'):
        assert word not in v33, word


def test_the_existing_resource_rules_are_untouched():
    v33 = prompt('v33')
    for clause in ('daily_conditions가 있으면 활동에 필요한 자원을 실제 사용 전 확보할 수 있는지 확인한다.',
                   '물품을 주문해도 명시된 도착 전에는 사용할 수 없다.',
                   '후보의 존재는 구매 의무가 아니다.'):
        assert clause in v33, clause


def test_no_policy_or_number_leaks_in():
    v33 = prompt('v33')
    for word in ('캐시백', '지역화폐', '거리두기', '지갑', '정책', '실측'):
        assert word not in v33
    assert not re.search(r'\d+\s*(원|퍼센트|%)', v33)


def test_planner_accepts_v33():
    src = importlib.import_module('validate_action_planner').__file__
    allow = re.search(r"prompt_module','v22'\) not in \{([^}]*)\}", open(src, encoding='utf-8').read()).group(1)
    assert "'v33'" in allow
