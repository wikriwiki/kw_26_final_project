"""v30 은 빗장을 유지하되 겨냥만 고친다 — 명사 하나가 바뀐다.

v6 에서 빗장을 통째로 빼니 외출은 그대로인데 구매만 늘었다. 빗장이 막고 있던
것이 칸 채우기가 아니라 구매였다는 뜻이다. v30 은 빗장을 남기고 '구매'라는
낱말만 뺀다. 빼는 것도, 부추기는 것도 아니다.
"""
import importlib
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))

OLD = '활동 수를 채우려고 구매·외출·반복을 추가하지 않는다.'
NEW = '활동 수를 채우려고 활동을 추가하지 않는다.'


def prompt(name):
    return importlib.import_module('prompts.' + name).SYSTEM_PROMPT


def test_the_guard_survives_with_a_new_aim():
    v25, v30 = prompt('v25'), prompt('v30')
    assert OLD in v25 and OLD not in v30
    assert NEW in v30


def test_exactly_one_clause_differs_from_v25():
    assert prompt('v25').replace(OLD, NEW) == prompt('v30')


def test_it_neither_removes_the_guard_nor_encourages_buying():
    v29, v30 = prompt('v29'), prompt('v30')
    assert '활동 수를 채우려고' in v30, '빗장이 남아야 한다'
    assert '활동 수를 채우려고' not in v29, 'v29 는 통째로 뺀 쪽이다'
    for word in ('사라', '나가라', '더 쓰', '많이 쓰', '늘려'):
        assert word not in v30


def test_the_other_guards_are_untouched():
    v30 = prompt('v30')
    for clause in ('후보의 존재는 구매 의무가 아니다.',
                   '집 체류 시간은 재택근무 가능 여부가 아니다.',
                   '평균 지출과 과거 비중은 오늘 채울 할당량이 아니다.',
                   '없는 의무·휴무·재택 허가·부족·질병·과거 경험을 만들지 않는다.'):
        assert clause in v30, clause


def test_no_policy_or_number_leaks_in():
    v30 = prompt('v30')
    for word in ('캐시백', '지역화폐', '거리두기', '지갑', '정책', '실측'):
        assert word not in v30
    assert not re.search(r'\d+\s*(원|퍼센트)', v30)


def test_planner_accepts_v30():
    src = importlib.import_module('validate_action_planner').__file__
    allow = re.search(r"prompt_module','v22'\) not in \{([^}]*)\}", open(src, encoding='utf-8').read()).group(1)
    assert "'v30'" in allow
