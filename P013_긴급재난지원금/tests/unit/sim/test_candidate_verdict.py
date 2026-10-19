"""후보 판정은 계산된다 — 결과를 본 뒤 기준을 고쳐 쓰는 일을 코드로 막는다.

v2에서 잰 seed 변동 폭이 잣대다. 후보 간 폭이 그 이하이면 '구별 불가'여야 하고,
넘더라도 후보당 seed 하나로는 후보 탓이라 단정하지 않아야 한다.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/report'))
from build_experiment_graphs import candidate_panels, mechanism_differences


def report_with(**mech_to_diff):
    return {'mechanisms': {m: {'matched_contrasts': {'by_seed': [
        {'replicate': 59001, 'metrics': {'total_consumption': {
            'difference': d, 'off_mean': 10000.0, 'on_mean': 10000.0 + d}}}]}}
        for m, d in mech_to_diff.items()}}


BAND = {'mean_width': 2937.5, 'max_width': 4000.0,
        'by_mechanism': {'캐시백': {'width': 2666.0}, '지원금': {'width': 2166.0}}}


def test_difference_within_the_band_is_not_a_candidate_difference():
    reports = {'v25': report_with(cashback=0.0),
               'v26': report_with(cashback=1000.0),
               'v27': report_with(cashback=-1000.0)}
    text = '\n'.join(candidate_panels(reports, BAND))
    assert '구별 불가' in text and '폭 초과' not in text


def test_difference_beyond_the_band_still_refuses_to_blame_the_candidate():
    reports = {'v25': report_with(cashback=0.0),
               'v26': report_with(cashback=5000.0),
               'v27': report_with(cashback=-5000.0)}
    text = '\n'.join(candidate_panels(reports, BAND))
    assert '폭 초과' in text
    assert '후보 탓이라고 단정하지 않는다' in text


def test_a_mechanism_without_a_recorded_band_is_not_judged():
    reports = {'v25': report_with(distancing=0.0), 'v26': report_with(distancing=9000.0)}
    text = '\n'.join(candidate_panels(reports, BAND))
    assert '잣대 없음' in text and '판정하지 않는다' in text


def test_a_refused_mechanism_is_dropped_not_zero_filled():
    """채점 거부를 0으로 채우면 없는 차이가 생긴다."""
    reports = {'v25': report_with(cashback=500.0),
               'v26': {'mechanisms': {'cashback': {'not_scored': '행렬 불완전'}}}}
    table = mechanism_differences(reports)
    assert table['캐시백'] == {'v25': 500.0}


def test_all_indistinguishable_says_the_second_criterion_cannot_choose():
    reports = {'v25': report_with(cashback=0.0, grant=0.0),
               'v26': report_with(cashback=100.0, grant=100.0)}
    text = '\n'.join(candidate_panels(reports, BAND))
    assert '2차 판정으로는 후보를 고를 수 없다' in text


def test_single_candidate_draws_no_comparison():
    assert candidate_panels({'v25': report_with(cashback=500.0)}, BAND)
