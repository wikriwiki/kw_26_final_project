"""v29 는 외출 억제 문장 하나만 뺀다 — 대신 다른 빗장은 전부 남는다.

절제(ablation)이지 제안이 아니다. 빗장을 빼면 지출이 부풀 수 있고, 그 값을
재는 것이 이 실험이다. 다만 **없는 것을 만들지 말라는 빗장까지 같이 빠지면**
무엇 때문에 움직였는지 알 수 없게 되므로 검사로 막는다.
"""
import importlib
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))

REMOVED = '활동 수를 채우려고 구매·외출·반복을 추가하지 않는다.'
KEPT = (
    '후보의 존재는 구매 의무가 아니다.',
    '집 체류 시간은 재택근무 가능 여부가 아니다.',
    '평균 지출과 과거 비중은 오늘 채울 할당량이 아니다.',
    '없는 의무·휴무·재택 허가·부족·질병·과거 경험을 만들지 않는다.',
)


def prompt(name):
    return importlib.import_module('prompts.' + name).SYSTEM_PROMPT


def test_only_the_outing_guard_is_gone():
    v25, v29 = prompt('v25'), prompt('v29')
    assert REMOVED in v25 and REMOVED not in v29


def test_every_other_guard_survives():
    v29 = prompt('v29')
    for clause in KEPT:
        assert clause in v29, clause


def test_nothing_is_added_in_its_place():
    """빼기만 한다. 대신 '나가라'를 넣으면 그건 유도다."""
    v25, v29 = prompt('v25'), prompt('v29')
    assert len(v29) < len(v25)
    assert v25.replace(REMOVED + ' ', '') == v29
    for word in ('외출', '나가', '동네', '바깥'):
        assert word not in v29.replace('구매·외출·반복', '')


def test_no_policy_or_direction_leaks_in():
    v29 = prompt('v29')
    for word in ('캐시백', '지역화폐', '거리두기', '지갑', '정책', '늘려', '더 쓰', '%'):
        assert word not in v29
    assert not re.search(r'\d+\s*(원|퍼센트)', v29)


def test_planner_accepts_v29():
    src = importlib.import_module('validate_action_planner').__file__
    allow = re.search(r"prompt_module','v22'\) not in \{([^}]*)\}", open(src, encoding='utf-8').read()).group(1)
    assert "'v29'" in allow
