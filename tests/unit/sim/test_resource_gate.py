"""The plan->purchase gate must block impossible schedules without repairing them.

The observed failure was a real one: detergent arrives after the wash it is for.
A purchase call on that plan scores an engine fact as a model failure, so the gate
removes it from the call list. It must not remove the evidence with it.
"""
from copy import deepcopy

import pytest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))

from prepare_resource_stress import prepare
from resource_feasibility import check, gate, gate_report


def cells(n=2):
    out = []
    for i, cell in enumerate(prepare()['cells'][:n]):
        out.append({'aid': f'A{i}', 'case': 'distancing', 'arm': 'on', 'date': '2020-11-24',
                    'transaction_case': deepcopy(cell['transaction_case'])})
    return out


def make_impossible(cell):
    """Move the use of a purchased resource ahead of any possible delivery."""
    cell['transaction_case']['events'][1]['time'] = '10:00'
    assert check(cell['transaction_case'])['impossible_even_with_all_candidates']
    return cell


def test_feasible_matrix_passes_and_still_reports_the_gate():
    runnable, blocked = gate(cells())
    assert len(runnable) == 2 and blocked == []
    report = gate_report(runnable, blocked)
    assert report['whole_matrix_eligible'] and report['excluded'] == 0 and report['checked'] == 2


def test_impossible_cell_is_refused_unless_the_caller_opts_in():
    rows = cells()
    make_impossible(rows[0])
    with pytest.raises(ValueError, match='Resource gate blocked'):
        gate(rows)


def test_opted_in_exclusion_keeps_the_row_and_its_shortfall_evidence():
    rows = cells()
    make_impossible(rows[0])
    runnable, blocked = gate(rows, allow_exclusion=True)
    assert [c['aid'] for c in runnable] == ['A1']
    assert [c['aid'] for c in blocked] == ['A0']
    # the blocked plan is untouched: the gate does not shift the offending event
    assert blocked[0]['transaction_case']['events'][1]['time'] == '10:00'
    assert blocked[0]['resource_feasibility']['shortfalls']


def test_exclusion_marks_the_matrix_incomplete_rather_than_passing_the_remainder():
    rows = cells()
    make_impossible(rows[0])
    report = gate_report(*gate(rows, allow_exclusion=True))
    assert report['whole_matrix_eligible'] is False
    assert report['excluded'] == 1 and report['checked'] == 2
    assert report['cells'][0]['aid'] == 'A0' and report['cells'][0]['shortfalls']


def test_gate_refuses_when_nothing_survives():
    rows = cells(1)
    make_impossible(rows[0])
    with pytest.raises(ValueError, match='excluded every cell'):
        gate(rows, allow_exclusion=True)


def test_gate_reuses_an_attached_verdict_instead_of_recomputing():
    rows = cells(1)
    rows[0]['resource_feasibility'] = {'impossible_even_with_all_candidates': True, 'shortfalls': [{'seam': 'upstream'}]}
    runnable, blocked = gate(rows + cells(1), allow_exclusion=True)
    assert blocked[0]['resource_feasibility']['shortfalls'] == [{'seam': 'upstream'}]
    assert len(runnable) == 1
