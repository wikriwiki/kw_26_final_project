"""사전등록한 판정을 코드가 내린다 — 특히 '밖으로 내보낸 것'을 성공으로 세지 않는다."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/report'))
from judge_place_axis import arm_share, verdict


def test_on_only_rise_beyond_the_seed_spread_is_success():
    got = verdict([0.10, 0.12, 0.11, 0.09], [0.40, 0.42, 0.38, 0.41],
                  [0.10, 0.11, 0.10, 0.12], [0.11, 0.10, 0.12, 0.11])
    assert got['verdict'].startswith('성공')


def test_both_arms_rising_is_not_a_win():
    """정책과 무관하게 외출이 늘면 그건 지출 유도지 축이 작동한 것이 아니다."""
    got = verdict([0.10, 0.12, 0.11, 0.09], [0.40, 0.42, 0.38, 0.41],
                  [0.10, 0.11, 0.10, 0.12], [0.39, 0.41, 0.40, 0.38])
    assert got['verdict'].startswith('되레 나쁨')
    assert got['gap_off'] > got['seed_spread']


def test_a_gap_inside_the_seed_spread_is_failure():
    got = verdict([0.10, 0.30, 0.05, 0.25], [0.12, 0.31, 0.07, 0.26],
                  [0.10, 0.10, 0.10, 0.10], [0.10, 0.10, 0.10, 0.10])
    assert got['verdict'].startswith('실패')


def test_the_spread_comes_from_whichever_run_is_noisier():
    """조용한 쪽만 보면 시끄러운 후보의 우연을 실력으로 읽는다."""
    quiet, noisy = [0.10] * 4, [0.10, 0.40, 0.05, 0.35]
    assert verdict(quiet, noisy, quiet, quiet)['seed_spread'] > 0.1


def test_arm_share_pools_mechanisms_and_ignores_the_total_row():
    per_seed = {1: {'grant': {'on': {'reached': 1, 'cells': 4}, 'off': {'reached': 0, 'cells': 4}},
                    'cashback': {'on': {'reached': 3, 'cells': 4}, 'off': {'reached': 0, 'cells': 4}},
                    '_all': {'reached': 999, 'cells': 999}}}
    assert arm_share(per_seed, 'on') == [0.5]
    assert arm_share(per_seed, 'off') == [0.0]


def test_no_cells_does_not_divide_by_zero():
    per_seed = {1: {'grant': {'on': {'reached': 0, 'cells': 0}, 'off': {'reached': 0, 'cells': 0}}}}
    assert arm_share(per_seed, 'on') == [0.0]
