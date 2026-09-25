"""Income is off unless asked for, and a typo must never read as 'off'."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from income import daily_income, describe, parse, preflight_baseline_income  # noqa: E402


@pytest.mark.parametrize('spec', [None, '', '0', 'off', 'none', 'NO', 'False', '  '])
def test_absent_or_off_means_no_income(spec):
    assert daily_income(100000, spec) == 0


def test_every_earlier_run_reproduces_because_the_default_is_off():
    """The env var being unset is the only state earlier runs were in."""
    assert daily_income(123456, None) == 0
    assert parse(None) == ('off', 0.0)


def test_anchor_pays_the_citizens_own_daily_anchor():
    assert daily_income(117000, 'anchor') == 117000


def test_anchor_takes_a_factor():
    assert daily_income(100000, 'anchor:1.2') == 120000
    assert daily_income(100000, 'ANCHOR:0.5') == 50000


def test_a_flat_amount_ignores_the_anchor():
    assert daily_income(999999, '45000') == 45000


def test_income_scales_with_the_person_not_with_a_constant():
    """A rich and a poor citizen must not be levelled by the apparatus."""
    assert daily_income(50000, 'anchor') != daily_income(200000, 'anchor')


def test_a_missing_anchor_is_not_a_crash():
    assert daily_income(None, 'anchor') == 0
    assert daily_income(0, 'anchor') == 0


def test_a_typo_raises_rather_than_silently_paying_nothing():
    """Silently running without income would be blamed on the window. That is the trap."""
    for bad in ('ancor', 'anchor*1.2', 'yes', 'anchor:x', '1,000'):
        with pytest.raises(ValueError):
            daily_income(100000, bad)


def test_negative_income_is_refused():
    for bad in ('-1', 'anchor:-1'):
        with pytest.raises(ValueError):
            daily_income(100000, bad)


def test_the_run_log_says_whether_income_was_on():
    assert '없음' in describe(None)
    assert '앵커' in describe('anchor')
    assert 'x 1.2' in describe('anchor:1.2')
    assert '45,000' in describe('45000')


def test_rounding_is_to_whole_won():
    v = daily_income(33333, 'anchor:1.5')
    assert isinstance(v, int) and v == 50000


def test_policy_free_baseline_income_is_fixed_across_policy_changed_anchors(tmp_path):
    import json
    source = tmp_path / 'baseline.json'
    source.write_text(json.dumps({
        'schema': 'baseline_income_v1',
        'policy_free_success_rows_verified': True,
        'citizen_count': 2,
        'daily_income_by_aid': {'a': 90000, 'b': 120000},
    }), encoding='utf-8')
    assert parse('baseline') == ('baseline', 1.0)
    assert '고정' in describe('baseline')
    info = preflight_baseline_income('baseline', source, ['a', 'b'])
    assert info['citizens'] == 2 and len(info['map_sha256']) == 64
    assert preflight_baseline_income('baseline', source, ['a'])['citizens'] == 1
    assert daily_income(70000, 'baseline', aid='a', baseline_map_path=source) == 90000
    assert daily_income(150000, 'baseline', aid='a', baseline_map_path=source) == 90000
    with pytest.raises(ValueError, match='missing=1'):
        preflight_baseline_income('baseline', source, ['a', 'c'])


def test_baseline_income_requires_map_before_simulation():
    with pytest.raises(ValueError, match='EXP_DAILY_INCOME_MAP'):
        preflight_baseline_income('baseline', None, ['a'])
