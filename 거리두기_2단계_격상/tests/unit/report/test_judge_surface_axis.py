"""위약칸이 같이 움직이면 주 지표만 보고 성공이라 말하지 않는다."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/report'))
from judge_surface_axis import compare, decide

FLAT = [0.10, 0.11, 0.10, 0.09]
UP = [0.40, 0.42, 0.38, 0.41]


def test_only_the_changed_cells_moving_is_success():
    assert decide(compare(FLAT, UP), compare(FLAT, FLAT)).startswith('성공')


def test_both_halves_moving_is_void_not_success():
    """글자 하나 다르지 않은 칸이 움직였다면 무엇을 쟀는지 알 수 없다."""
    got = decide(compare(FLAT, UP), compare(FLAT, UP))
    assert got.startswith('무효')
    assert '잡음이거나' in got


def test_neither_moving_is_failure():
    assert decide(compare(FLAT, FLAT), compare(FLAT, FLAT)).startswith('실패')


def test_only_the_placebo_moving_is_failure_and_says_so():
    got = decide(compare(FLAT, FLAT), compare(FLAT, UP))
    assert got.startswith('실패') and '안 바뀐 칸이 움직였다' in got


def test_the_spread_comes_from_the_noisier_arm():
    noisy = [0.05, 0.45, 0.10, 0.40]
    assert compare(FLAT, noisy)['seed_spread'] > 0.1
    assert compare(FLAT, noisy)['moved'] is False


def test_a_gap_exactly_at_the_spread_does_not_count_as_moved():
    got = compare([0.0, 0.0], [0.0, 0.0])
    assert got['seed_spread'] == 0.0 and got['gap'] == 0.0 and got['moved'] is False
