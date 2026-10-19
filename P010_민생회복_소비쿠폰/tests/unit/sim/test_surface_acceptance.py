"""자격 규칙을 조건 블록으로 옮기되, 입력에 없던 것을 만들지 않는다."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from surface_acceptance import acceptance_lines, surface

USER = ('## 현재 활성 정책의 조건\n'
        '- P013 [정책지갑 지급] | 서울특별시 · [쿠폰] 표시 POI에서만 사용\n'
        '  배경: 어떤 설명.\n\n'
        '## 사회 배경\n거리두기 2단계\n')
PREVIEW = {'wallet_acceptance': [
    {'wallet_id': 'P013', 'anchors': ['zone:11290685', 'zone:11290660'],
     'activity_ids': ['groceries', 'convenience']}]}


def test_the_rule_lands_in_the_conditions_block():
    got = surface(USER, PREVIEW)
    head = got[got.find('## 현재 활성 정책의 조건'):got.find('## 사회 배경')]
    assert 'zone:11290685' in head and 'groceries' in head


def test_the_dangling_marker_is_removed():
    """존재하지 않는 표시를 찾으라고 남겨 두면 규칙이 둘이 된다."""
    got = surface(USER, PREVIEW)
    assert '[쿠폰] 표시 POI에서만 사용' not in got


def test_the_social_background_is_untouched():
    got = surface(USER, PREVIEW)
    assert got[got.find('## 사회 배경'):] == USER[USER.find('## 사회 배경'):]


def test_nothing_is_invented_beyond_the_preview():
    line = acceptance_lines(PREVIEW)[0]
    for token in ('zone:11290685', 'zone:11290660', 'groceries', 'convenience', 'P013'):
        assert token in line
    for word in ('늘', '줄', '권장', '유리', '먼저'):
        assert word not in line


def test_a_cell_without_a_wallet_is_returned_unchanged():
    """대조군과 처치군이 지갑 있는 칸에서만 갈라지게 한다."""
    assert surface(USER, {'wallet_acceptance': []}) == USER
    assert surface(USER, None) == USER


def test_an_incomplete_acceptance_entry_is_skipped():
    assert acceptance_lines({'wallet_acceptance': [{'wallet_id': 'P0', 'anchors': [], 'activity_ids': ['x']}]}) == []


def test_a_text_without_the_block_is_refused():
    with pytest.raises(ValueError):
        surface('설명만 있는 입력', PREVIEW)
