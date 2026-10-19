"""v34 는 근거가 남은 수정 둘을 합친 것이고, 그 둘 말고는 v25 와 같아야 한다."""
import importlib
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))


def prompt(name):
    return importlib.import_module('prompts.' + name).SYSTEM_PROMPT


def test_it_is_exactly_v30_and_v33_stacked():
    v25, v30, v33, v34 = (prompt(n) for n in ('v25', 'v30', 'v33', 'v34'))
    guard = [l for l in v30.splitlines() if '활동 수를 채우려고' in l][0]
    order = [l for l in v33.splitlines() if '자원이 있어야 하는 일은' in l][0]
    assert guard in v34 and order in v34
    # v25 에 두 수정을 적용하면 정확히 v34 가 되어야 한다
    old_guard = [l for l in v25.splitlines() if '활동 수를 채우려고' in l][0]
    old_order = [l for l in v25.splitlines() if '먼저 확정된 일정과' in l][0]
    assert v25.replace(old_guard, guard).replace(old_order, order) == v34


def test_no_third_change_sneaked_in():
    v25, v34 = prompt('v25'), prompt('v34')
    a = [l for l in v25.splitlines() if l.strip()]
    b = [l for l in v34.splitlines() if l.strip()]
    assert len(a) == len(b), '줄 수가 같아야 한다 — 문장을 더하거나 빼지 않았다'
    assert sum(1 for x, y in zip(a, b) if x != y) == 2, '정확히 두 줄만 다르다'


def test_every_guard_survives():
    v34 = prompt('v34')
    for clause in ('후보의 존재는 구매 의무가 아니다.',
                   '집 체류 시간은 재택근무 가능 여부가 아니다.',
                   '평균 지출과 과거 비중은 오늘 채울 할당량이 아니다.',
                   '사회 전체의 소비나 제도의 성과를 목표로 삼지 않는다.',
                   '없는 의무·휴무·재택 허가·부족·질병·과거 경험을 만들지 않는다.'):
        assert clause in v34, clause


def test_no_policy_sector_direction_or_number():
    v34 = prompt('v34')
    for word in ('캐시백', '지역화폐', '거리두기', '지갑', '정책', '실측',
                 '음식점', '카페', '외식', '장보', '마트', '배달', '늘려', '더 쓰'):
        assert word not in v34, word
    assert not re.search(r'\d+\s*(원|퍼센트|%)', v34)


def test_planner_accepts_v34():
    src = importlib.import_module('validate_action_planner').__file__
    allow = re.search(r"prompt_module','v22'\) not in \{([^}]*)\}", open(src, encoding='utf-8').read()).group(1)
    assert "'v34'" in allow
