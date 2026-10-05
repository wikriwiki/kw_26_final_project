"""The two levers v29 pulls: the errand's category, and which scenarios get planned.

Both only ever widen a denominator. Neither may touch a cell's text in a way that differs
between arms - that is the line between a scenario assumption and a nudge.
"""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

from add_outside_need import add, chosen_errand          # noqa: E402
from filter_cases import filter_cases                    # noqa: E402


def cell(aid, case, arm, mix='식사 40% / 마트 30% / 건강 10%'):
    return {
        'aid': aid, 'case': case, 'arm': arm,
        'user': ('평소 업종별 지출 구성(카드 실측): %s\n\n'
                 '## 정책 적용 전 고정한 오늘의 조건\n{}\n\n## 오늘\n' % mix),
        'daily_conditions': {'needs': [], 'assumptions': [], 'resources': {},
                             'activity_consumption': {}, 'quote_receipts': {},
                             'quote_receipt_delay_minutes': {},
                             'provenance': {'kind': 'synthetic_assumption',
                                            'source': 'unit test fixture'}},
    }


def source(cells):
    return {'cells': cells}


def test_forced_category_overrides_the_own_largest_rule():
    c = cell('a', 'distancing', 'off', mix='마트 80% / 식사 5%')
    assert chosen_errand(c)[0] == 'groceries'
    assert chosen_errand(c, 'meal_dine_in')[0] == 'meal_dine_in'


def test_forced_category_reports_the_citizens_own_share_even_when_zero():
    c = cell('a', 'distancing', 'off', mix='마트 80%')
    activity, share = chosen_errand(c, 'meal_dine_in')
    assert (activity, share) == ('meal_dine_in', 0)


def test_forced_category_gives_every_citizen_an_errand():
    """The own-largest rule drops citizens whose mix has no share. Forcing must not."""
    cells = [cell('a', 'distancing', 'off', mix='마트 80%'),
             cell('b', 'distancing', 'off', mix='')]
    out = add(source(cells), forced='meal_dine_in')
    for c in out['cells']:
        ids = [n['id'] for n in c['daily_conditions']['needs']]
        assert 'errand' in ids, c['aid']


def test_forced_errand_is_optional_not_grammar_enforced():
    """v28 put the errand in required_activities and the grammar decided it. Not here."""
    out = add(source([cell('a', 'distancing', 'off')]), forced='meal_dine_in')
    c = out['cells'][0]
    need = [n for n in c['daily_conditions']['needs'] if n['id'] == 'errand'][0]
    assert need['mandatory'] is False
    assert not c.get('required_activities')


def test_both_arms_get_the_same_errand_text():
    cells = [cell('a', 'distancing', 'off'), cell('a', 'distancing', 'on')]
    out = add(source(cells), forced='meal_dine_in')
    off, on = out['cells']
    off_need = [n for n in off['daily_conditions']['needs'] if n['id'] == 'errand'][0]
    on_need = [n for n in on['daily_conditions']['needs'] if n['id'] == 'errand'][0]
    assert off_need == on_need


def test_provenance_says_it_was_assigned_not_observed():
    out = add(source([cell('a', 'distancing', 'off')]), forced='meal_dine_in')
    p = out['errand_provenance']
    assert p['forced_activity'] == 'meal_dine_in'
    assert 'Assigned uniformly' in p['rule']
    assert p['both_arms_identical'] is True


def test_unforced_provenance_still_says_the_rule_read_the_citizen():
    out = add(source([cell('a', 'distancing', 'off')]))
    p = out['errand_provenance']
    assert p['forced_activity'] is None
    assert 'largest share' in p['rule']


def test_filter_keeps_only_the_named_cases():
    cells = [cell('a', 'distancing', 'off'), cell('a', 'grant', 'off'),
             cell('b', 'distancing', 'on')]
    out = filter_cases(source(cells), ['distancing'])
    assert {c['case'] for c in out['cells']} == {'distancing'}
    assert out['case_filter']['dropped'] == ['grant']
    assert out['case_filter']['cells_after'] == 2


def test_filter_does_not_alter_a_surviving_cell():
    cells = [cell('a', 'distancing', 'off'), cell('a', 'grant', 'off')]
    before = copy.deepcopy(cells[0])
    out = filter_cases(source(cells), ['distancing'])
    assert out['cells'][0] == before


def test_filter_refuses_a_case_that_is_not_there():
    with pytest.raises(ValueError, match='no cells for'):
        filter_cases(source([cell('a', 'grant', 'off')]), ['distancing'])


def test_filter_records_that_the_run_is_not_a_full_matrix():
    out = filter_cases(source([cell('a', 'distancing', 'off'), cell('a', 'grant', 'off')]),
                       ['distancing'])
    assert 'no indicator outside the kept set' in out['case_filter']['why']


def test_filter_then_force_leaves_both_arms_balanced():
    cells = [cell(a, c, arm) for a in 'ab' for c in ('distancing', 'grant')
             for arm in ('off', 'on')]
    out = add(filter_cases(source(cells), ['distancing']), forced='meal_dine_in')
    arms = [c['arm'] for c in out['cells']]
    assert arms.count('off') == arms.count('on') == 2
