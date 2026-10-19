import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from transaction_ledger import settle_choices
from paired_ledger_score import ledger_totals, score


def make_row(aid, arm, amount, *, channel='offline', policy=0, seed=1):
    ledger = settle_choices([{'id': '0', 'channel': channel, 'amount': amount, 'policy_spend': {'W': policy}}],
                            cash=1000, wallets={'W': 1000}, eligible_wallets_by_transaction={'0': {'W'}})
    return {'aid': aid, 'arm': arm, 'day': '2026-09-21', 'replicate': seed, 'ledger': ledger}


def test_complete_zero_citizen_and_online_are_in_denominator():
    rows = [make_row('A', 'off', 100), make_row('A', 'on', 300, channel='online'),
            make_row('B', 'off', 0), make_row('B', 'on', 0)]
    result = score(rows, roster=['A', 'B'], days=['2026-09-21'], seeds=[1])
    total = result['by_seed'][0]['metrics']['total_including_online']
    assert total['off_won_per_person_day'] == 50
    assert total['difference_won_per_person_day'] == 100
    assert total['relative_change'] == 2
    assert total['individual_paired_differences'] == {'A': 200, 'B': 0}


def test_funding_substitution_is_not_consumption_growth():
    rows = [make_row('A', 'off', 500), make_row('A', 'on', 500, policy=400)]
    metrics = score(rows, roster=['A'], days=['2026-09-21'], seeds=[1])['by_seed'][0]['metrics']
    assert metrics['total_including_online']['difference_won_per_person_day'] == 0
    assert metrics['own_payment_total']['difference_won_per_person_day'] == -400
    assert metrics['policy_payment_total']['difference_won_per_person_day'] == 400
    assert metrics['policy_payment_total']['relative_change'] is None


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'incomplete', 'different_calendar'])
def test_bad_matrix_never_becomes_complete_zero(kind):
    rows = [make_row('A', 'off', 0), make_row('A', 'on', 0)]
    if kind == 'missing': rows.pop()
    if kind == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    if kind == 'incomplete': rows[1]['ledger']['complete'] = False
    if kind == 'different_calendar': rows[1]['day'] = '2026-09-22'
    with pytest.raises(ValueError):
        score(rows, roster=['A'], days=['2026-09-21'], seeds=[1])


def test_tampered_totals_rejected():
    ledger = make_row('A', 'off', 100)['ledger']
    ledger['total_including_online'] = 0
    with pytest.raises(ValueError): ledger_totals(ledger)


def test_fixed_weights_and_seed_variation_are_separate():
    rows = [make_row(aid, arm, (100 if aid == 'A' else 300) * seed if arm == 'on' else 0, seed=seed)
            for seed in [1, 2] for aid in ['A', 'B'] for arm in ['off', 'on']]
    result = score(rows, roster=['A', 'B'], days=['2026-09-21'], seeds=[1, 2], weights={'A': 3, 'B': 1})
    assert [r['metrics']['total_including_online']['difference_won_per_person_day'] for r in result['by_seed']] == [150, 300]
    assert result['across_seed_descriptive']['total_including_online']['mean_difference'] == 225
