"""v36·v37 은 문장을 더하지 않는다 — 지우기만 한다.

지난 네 라운드가 같은 모양으로 실패했다. v31 은 3행의 부정문을 사실로 바꿨고,
v35 는 18행에 같은 수법을 썼다. 둘 다 아무것도 움직이지 않았다. 움직인 것은
v30 하나인데, 그것은 **금지 대상에서 낱말을 지운** 판이었다.

그래서 이번 후보 둘은 지우기만 한다. 그 성질을 여기서 기계로 고정한다 —
지우기라고 적어 놓고 슬쩍 덧붙이면 다음 라운드의 비교가 무너진다.
"""
import importlib
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))

# 정책 이름·원하는 방향·수치 목표는 물론, 지출이나 외출을 부추기는 낱말도 막는다.
FORBIDDEN = ('캐시백', '상생', '소비쿠폰', '지역화폐', '거리두기', '긴급재난', '적립', '실측',
             '지갑', '정책', '늘려', '줄여', '증가시', '감소시', '더 쓰', '많이 쓰',
             '외출을 늘', '나가서', '매장', '%')


def prompt(name):
    return importlib.import_module('prompts.' + name).SYSTEM_PROMPT


def lines(name):
    return [l for l in prompt(name).splitlines() if l.strip()]


def test_candidates_name_no_policy_and_push_no_direction():
    for name in ('v36', 'v37'):
        text = prompt(name)
        assert not [w for w in FORBIDDEN if w in text], name
        assert not re.search(r'\d+\s*(원|퍼센트)', text), name


def test_each_candidate_changes_exactly_one_line_of_v25():
    base = lines('v25')
    for name in ('v36', 'v37'):
        got = lines(name)
        assert len(got) == len(base), name
        differing = [i for i, (a, b) in enumerate(zip(base, got)) if a != b]
        assert len(differing) == 1, (name, differing)


def test_nothing_is_added_only_removed():
    """지운 판이라면 새 낱말이 하나도 없어야 한다."""
    base = set(re.findall(r'[가-힣]+', prompt('v25')))
    for name in ('v36', 'v37'):
        added = set(re.findall(r'[가-힣]+', prompt(name))) - base
        assert not added, (name, added)
        assert len(prompt(name)) < len(prompt('v25')), name


def test_v36_drops_the_sentence_that_discounts_the_citizens_own_mix():
    v25, v36 = prompt('v25'), prompt('v36')
    assert '평균 지출과 과거 비중은 오늘 채울 할당량이 아니다' in v25
    assert '평균 지출과 과거 비중은' not in v36


def test_v36_keeps_the_guard_against_aiming_at_an_aggregate():
    """집계를 겨누지 말라는 빗장까지 빼면 정책 성과를 최적화하라는 말이 된다."""
    assert '사회 전체의 소비나 제도의 성과를 목표로 삼지 않는다' in prompt('v36')


def test_v37_drops_only_the_substitution_licence():
    v25, v37 = prompt('v25'), prompt('v37')
    assert '대체·연기·생략할 수 있다' in v25
    assert '연기·생략할 수 있다' in v37 and '대체' not in v37


def test_v37_keeps_the_rest_of_the_no_obligation_line():
    """연기·생략이 남아 있어야 '오늘 반드시 채워라'가 되지 않는다."""
    v37 = prompt('v37')
    assert '보유품으로 해결하거나' in v37
    assert '후보의 존재는 구매 의무가 아니다' in v37


def test_both_keep_the_padding_guard():
    """빗장을 두 개 동시에 풀면 무엇이 움직였는지 가릴 수 없다."""
    for name in ('v36', 'v37'):
        assert '활동 수를 채우려고 구매·외출·반복을 추가하지 않는다' in prompt(name), name
