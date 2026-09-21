"""DS-1 must never quietly pick one reading of the published measure over the other."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/report'))
from ds1_two_definitions import (DEFINITIONS, collect, paired_bootstrap,  # noqa: E402
                                 pct, report, split)


class FakeRuns:
    """Stands in for purchases(); each record is (aid, arm, [(activity, price)])."""

    def __init__(self, recs, case='distancing'):
        self.recs = [{'aid': a, 'case': case, 'arm': arm,
                      'items': [{'activity_id': x, 'price': p} for x, p in items]}
                     for a, arm, items in recs]


@pytest.fixture
def patched(monkeypatch):
    def use(runs):
        import ds1_two_definitions as m
        monkeypatch.setattr(m, 'purchases', lambda src, resp: runs.recs)
        return [('src', 'resp')]
    return use


def test_the_two_definitions_differ_only_by_delivery():
    assert DEFINITIONS['식사 전체'] - DEFINITIONS['매장·포장'] == {'home_delivery', 'office_delivery'}


def test_delivery_lands_only_in_the_wider_definition(patched):
    runs = patched(FakeRuns([('a', 'off', [('meal_dine_in', 100.0)]),
                             ('a', 'on', [('home_delivery', 900.0)])]))
    data = collect(runs)
    assert data['매장·포장']['a']['on'] == [0.0, 0]
    assert data['식사 전체']['a']['on'] == [900.0, 1]


def test_the_same_run_can_give_opposite_signs_under_the_two_readings(patched):
    """This is the whole reason the script exists - one number was hiding the other."""
    runs = patched(FakeRuns([('a', 'off', [('meal_dine_in', 100.0)]),
                             ('a', 'on', [('meal_dine_in', 20.0), ('home_delivery', 500.0)])]))
    _, res = report(runs, min_events=1, draws=200)
    assert res['by_definition']['매장·포장']['ds1'] < 0
    assert res['by_definition']['식사 전체']['ds1'] > 0


def test_the_gate_blocks_a_verdict_on_too_few_events(patched):
    runs = patched(FakeRuns([('a', 'off', [('meal_dine_in', 100.0)]),
                             ('a', 'on', [('meal_dine_in', 50.0)])]))
    lines, res = report(runs, min_events=20, draws=200)
    assert res['gate_ok'] is False
    assert res['gate_events'] == 1
    assert any('판정을 적지 않는다' in ln for ln in lines)


def test_the_gate_counts_only_the_no_policy_arm(patched):
    """A policy arm full of events does not make the baseline a denominator."""
    runs = patched(FakeRuns(
        [('a', 'off', [('meal_dine_in', 100.0)])] +
        [(str(i), 'on', [('meal_dine_in', 10.0)]) for i in range(30)]))
    _, res = report(runs, min_events=20, draws=200)
    assert res['gate_events'] == 1 and res['gate_ok'] is False


def test_the_gate_passes_once_the_baseline_is_wide_enough(patched):
    recs = [(str(i), 'off', [('meal_dine_in', 100.0)]) for i in range(20)]
    recs += [(str(i), 'on', [('meal_dine_in', 80.0)]) for i in range(20)]
    runs = patched(FakeRuns(recs))
    _, res = report(runs, min_events=20, draws=200)
    assert res['gate_ok'] is True
    assert res['by_definition']['매장·포장']['ds1'] == pytest.approx(-20.0)


def test_only_the_named_scenario_is_read(patched):
    runs = patched(FakeRuns([('a', 'off', [('meal_dine_in', 100.0)])], case='grant'))
    _, res = report(runs, min_events=1, draws=200)
    assert res['gate_events'] == 0


def test_a_citizen_is_one_bootstrap_unit_not_one_cell(patched):
    """Ten cells from one citizen must not look like ten independent citizens."""
    one = {'x': {'off': [100.0, 1], 'on': [50.0, 1]}}
    assert paired_bootstrap(one, draws=500) == {'lo': -50.0, 'hi': -50.0}


def test_the_interval_is_paired_so_a_common_shift_cancels(patched):
    per = {c: {'off': [100.0 + i * 50, 1], 'on': [50.0 + i * 25, 1]}
           for i, c in enumerate('abcdefgh')}
    ci = paired_bootstrap(per, draws=2000)
    assert ci['lo'] == pytest.approx(-50.0) and ci['hi'] == pytest.approx(-50.0)


def test_an_empty_denominator_is_undefined_not_zero():
    assert pct(5.0, 0.0) is None
    assert paired_bootstrap({'a': {'off': [0.0, 0], 'on': [10.0, 1]}}, draws=200) is None


def test_both_readings_always_appear_in_the_output(patched):
    recs = [(str(i), 'off', [('meal_dine_in', 100.0)]) for i in range(20)]
    recs += [(str(i), 'on', [('home_delivery', 300.0)]) for i in range(20)]
    lines, _ = report(patched(FakeRuns(recs)), min_events=20, draws=200)
    body = '\n'.join(lines)
    for name in DEFINITIONS:
        assert name in body


def test_run_spec_requires_both_halves():
    assert split('a.json@b.jsonl') == ('a.json', 'b.jsonl')
    with pytest.raises(ValueError, match='source@responses'):
        split('a.json')
