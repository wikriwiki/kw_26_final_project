"""v35 는 부여된 집 체류 시간을 부정문에서 사실로 바꾼다 — 양방향이어야 한다.

에이전트는 자기 평일 집 체류 시간을 입력에서 듣는다(60명 평균 12.1h). 계획은
17.7h 를 집에 둔다. 480칸 중 427칸이 자기 숫자보다 두 시간 넘게 더 집에 있다.
프롬프트는 그 숫자를 "재택근무 근거가 아니다"라고 부정할 때만 언급한다.
"""
import importlib
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))

OLD = '집 체류 시간은 재택근무 가능 여부가 아니다.'
NEW = ('집 체류 시간은 재택근무 가능 여부가 아니라 이 사람이 보통 하루를 어디서 보내는지를 '
       '말해 준다. 오늘 채울 할당량은 아니지만 하루를 배치할 때 참고한다.')


def prompt(name):
    return importlib.import_module('prompts.' + name).SYSTEM_PROMPT


def test_only_that_one_sentence_changes():
    assert prompt('v25').replace(OLD, NEW) == prompt('v35')


def test_the_original_denial_survives():
    """재택근무를 추론하지 말라는 빗장은 그대로 있어야 한다."""
    assert '재택근무 가능 여부가 아니라' in prompt('v35')


def test_it_is_two_sided_not_a_nudge_to_go_out():
    """집에 더 있어야 하는 사람도 있다. 한쪽으로만 밀면 유도다."""
    v35 = prompt('v35')
    for word in ('나가', '외출을 늘', '줄여', '더 많이', '적게'):
        assert word not in v35, word
    assert '할당량은 아니지만' in v35, '할당량이 아니라는 빗장을 유지한다'


def test_it_names_no_place_sector_or_policy():
    v35 = prompt('v35')
    for word in ('캐시백', '지역화폐', '거리두기', '지갑', '정책', '실측',
                 '음식점', '카페', '마트', '장보', '배달'):
        assert word not in v35, word
    assert not re.search(r'\d+\s*(원|퍼센트|시간|h)', v35)


def test_the_other_guards_survive():
    v35 = prompt('v35')
    for clause in ('후보의 존재는 구매 의무가 아니다.',
                   '활동 수를 채우려고 구매·외출·반복을 추가하지 않는다.',
                   '평균 지출과 과거 비중은 오늘 채울 할당량이 아니다.',
                   '없는 의무·휴무·재택 허가·부족·질병·과거 경험을 만들지 않는다.'):
        assert clause in v35, clause


def test_planner_accepts_v35():
    src = importlib.import_module('validate_action_planner').__file__
    allow = re.search(r"prompt_module','v22'\) not in \{([^}]*)\}", open(src, encoding='utf-8').read()).group(1)
    assert "'v35'" in allow
