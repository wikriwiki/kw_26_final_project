"""v31 은 v30 과 같은 수법을 지출 구성 빗장에 쓴다 — 빼지 않고 겨냥만 고친다.

무정책 192칸에서 산 71건 중 46건이 같은 온라인 세제였고, 그 사람에게 부여된
11개 업종 중 6개는 한 번도 안 팔렸다. 빗장은 "할당량이 아니다"까지만 말하고
그 구성이 무엇인지를 말하지 않아서 모델이 무시한다.
"""
import importlib
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))

OLD = '평균 지출과 과거 비중은 오늘 채울 할당량이 아니다.'
NEW = ('평균 지출과 과거 비중은 오늘 채울 할당량이 아니라 '
       '이 사람이 평소 어디에 돈을 쓰는지를 말해 준다.')


def prompt(name):
    return importlib.import_module('prompts.' + name).SYSTEM_PROMPT


def test_the_guard_survives_with_a_new_aim():
    v25, v31 = prompt('v25'), prompt('v31')
    assert OLD in v25 and OLD not in v31
    assert NEW in v31
    assert '할당량이 아니라' in v31, '빗장의 부정문이 남아야 한다'


def test_exactly_one_clause_differs_from_v25():
    assert prompt('v25').replace(OLD, NEW) == prompt('v31')


def test_it_names_no_sector_the_answer_keys_measure():
    """정답지가 재는 업종을 지목하면 그건 유도다."""
    v31 = prompt('v31')
    for word in ('음식점', '카페', '외식', '미용', '여가', '가전', '준내구재'):
        assert word not in v31, word


def test_no_policy_direction_or_number_leaks_in():
    v31 = prompt('v31')
    for word in ('캐시백', '지역화폐', '거리두기', '지갑', '정책', '실측',
                 '늘려', '줄여', '더 쓰', '많이 쓰'):
        assert word not in v31
    assert not re.search(r'\d+\s*(원|퍼센트|%)', v31)


def test_the_other_guards_are_untouched():
    v31 = prompt('v31')
    for clause in ('후보의 존재는 구매 의무가 아니다.',
                   '활동 수를 채우려고 구매·외출·반복을 추가하지 않는다.',
                   '집 체류 시간은 재택근무 가능 여부가 아니다.',
                   '사회 전체의 소비나 제도의 성과를 목표로 삼지 않는다.'):
        assert clause in v31, clause


def test_planner_accepts_v31():
    src = importlib.import_module('validate_action_planner').__file__
    allow = re.search(r"prompt_module','v22'\) not in \{([^}]*)\}", open(src, encoding='utf-8').read()).group(1)
    assert "'v31'" in allow
