"""W 격자는 숫자 하나만 바꿔야 한다 — 앞뒤가 어긋나면 W 가 아니라 불일치를 재게 된다."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

from threshold_response_probe import (GRID, build, monotone, read_state,  # noqa: E402
                                      set_fraction, spread)

USER = (
    '오늘의 조건\n'
    '- P012: 적립업종 이번달 누적 410,858원 / 2분기 월평균 약 513,573원 / '
    '3% 문턱 528,980원 | 초과분의 10% 다음 달 환급, 월 최대 100,000원 | '
    '문턱까지 118,122원 남음 — 적립업종에서 이만큼 더 쓰면 캐시백 자격 시작 '
    '(이번 달 7일 남음 · 하루 평균 16,875원 페이스) | 못 넘기면 이번 달 혜택은 사라짐\n'
    '- 판단 원칙: 평소 습관에 따라 판단한다.\n'
)


def test_it_reads_the_four_numbers():
    st = read_state(USER)
    assert st == {'pid': 'P012', 'spent': 410858, 'anchor': 513573,
                  'threshold': 528980, 'remaining': 118122, 'days': 7}


def test_a_context_without_the_line_is_skipped_not_guessed():
    assert read_state('오늘의 조건\n- 아무 정책도 없다\n') is None


def test_changing_w_moves_the_accumulated_amount_not_the_threshold():
    """Z 를 건드리면 정책이 달라진다. 바뀌는 것은 '지금까지 얼마 썼는가' 여야 한다."""
    out = set_fraction(USER, 0.10)
    st = read_state(out)
    assert st['threshold'] == 528980          # 문턱은 그대로
    assert st['remaining'] == round(528980 * 0.10)
    assert st['spent'] == 528980 - st['remaining']   # 누적이 옮겨졌다
    assert st['anchor'] == 513573             # 앵커도 그대로


def test_the_numbers_stay_consistent_at_every_grid_point():
    """누적 + 남은 거리 = 문턱. 이것이 깨지면 앞뒤가 안 맞는 맥락을 주는 것이다.

    절대 금액 격자를 쓰다가 W=900,000 에서 누적 0 · 남음 900,000 · 문턱 528,980 이
    되는 모순을 만들었다. 비율 격자로 바꾼 이유가 이것이다.
    """
    for f in GRID:
        st = read_state(set_fraction(USER, f))
        assert st['spent'] + st['remaining'] == st['threshold'], f
        assert st['spent'] >= 0 and st['remaining'] >= 0


def test_a_fraction_above_one_is_refused():
    with pytest.raises(ValueError, match='비율은 0~1'):
        set_fraction(USER, 1.5)


def test_the_pace_is_recomputed_to_match():
    """페이스가 옛 W 로 남아 있으면 앞뒤가 어긋난 맥락이 된다."""
    st = read_state(USER)
    out = set_fraction(USER, 0.10)
    w = round(st['threshold'] * 0.10)
    assert '하루 평균 %s원 페이스' % format(round(w / 7), ',d') in out
    assert '16,875' not in out


def test_the_days_left_are_untouched():
    out = set_fraction(USER, 0.60)
    assert '이번 달 7일 남음' in out


def test_everything_outside_the_line_is_byte_identical():
    out = set_fraction(USER, 0.10)
    assert '- 판단 원칙: 평소 습관에 따라 판단한다.' in out
    assert out.count('\n') == USER.count('\n')
    assert '못 넘기면 이번 달 혜택은 사라짐' in out


def test_w_of_zero_is_allowed_and_means_already_cleared():
    st = read_state(set_fraction(USER, 0.0))
    assert st['remaining'] == 0 and st['spent'] == 528980


def test_a_negative_fraction_is_refused():
    with pytest.raises(ValueError, match='비율은 0~1'):
        set_fraction(USER, -0.1)


def test_it_refuses_a_context_it_cannot_rewrite():
    with pytest.raises(ValueError, match='문턱 줄'):
        set_fraction('아무 문턱도 없는 글', 0.10)


def test_build_makes_one_cell_per_citizen_per_grid_point():
    frozen = {'cells': [
        {'aid': 'A', 'case': 'cashback', 'arm': 'on', 'user': USER, 'zones': [1]},
        {'aid': 'B', 'case': 'cashback', 'arm': 'on', 'user': USER, 'zones': [2]},
        {'aid': 'C', 'case': 'cashback', 'arm': 'off', 'user': USER, 'zones': [3]},
        {'aid': 'D', 'case': 'grant', 'arm': 'on', 'user': USER, 'zones': [4]},
    ]}
    cells = build(frozen, grid=(0.02, 0.60))
    assert len(cells) == 4                      # 시민 둘 x 격자 둘
    assert {c['aid'] for c in cells} == {'A', 'B'}
    assert sorted({c['fraction'] for c in cells}) == [0.02, 0.60]


def test_build_skips_a_cell_with_no_threshold_line():
    frozen = {'cells': [{'aid': 'A', 'case': 'cashback', 'arm': 'on',
                         'user': '문턱 줄 없음', 'zones': []}]}
    assert build(frozen, grid=(0.10,)) == []


def test_the_grid_spans_from_nearly_there_to_out_of_reach():
    assert GRID[0] <= 0.05 and GRID[-1] == 1.00
    assert list(GRID) == sorted(GRID)


def test_monotone_reads_decreasing_as_responding():
    assert monotone([(0.02, 0.8), (0.30, 0.6), (1.00, 0.4)])
    assert monotone([(0.02, 0.6), (0.30, 0.6), (1.00, 0.6)])   # 평평도 단조로 본다
    assert not monotone([(0.02, 0.4), (0.30, 0.7), (1.00, 0.5)])


def test_spread_is_zero_when_the_model_ignores_w():
    assert spread([(0.02, 0.6), (1.00, 0.6)]) == 0
    assert spread([(0.02, 0.8), (1.00, 0.5)]) == pytest.approx(0.3)


def test_no_directional_wording_is_introduced():
    """바뀌는 것은 숫자뿐이다. 문장을 더하지 않는다."""
    out = set_fraction(USER, 0.02)
    for word in ('더 쓰라', '늘려', '서둘러', '기회', '지금이'):
        assert word not in out
