"""Regression cases for the fe039529 execution audit; no DB or LLM calls."""
import sys
import json
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts' / 'sim'))
from consumption import apply_consumption_model
from score_policy import metric_values, paired, daterange


@pytest.mark.parametrize('balance', [0, 1, 100, 10000, 100000])
@pytest.mark.parametrize('share', [0, .25, 1])
def test_final_payment_choice_never_overdraws_cash(balance, share, monkeypatch):
    monkeypatch.setenv('EXP_PAYMENT_CHOICE', '1')
    events = [dict(category='식사', poi_id='A', actual_spent=12000,
                   policy_spend={'P': int(12000 * share)}, coupon_eligible=True,
                   actual_satisfaction=.9, price_factor=1),
              dict(category='카페', poi_id='B', actual_spent=6000,
                   policy_spend={}, coupon_eligible=False,
                   actual_satisfaction=.9, price_factor=1)]
    meta = apply_consumption_model(events, daily=30000, income_tier='중',
        tendency='', balance=balance, restricted_envelopes=[
            dict(pid='P', amount=5000, require_poi_eligible=True)])
    policy_paid = sum(sum(e['policy_spend'].values()) for e in events)
    cash = meta['online_total'] + sum(e['actual_spent'] for e in events) - policy_paid
    assert 0 <= cash <= balance
    assert policy_paid <= 5000
    assert meta['grant_carry_out'] == 0
    assert events[1]['policy_spend'] == {}
    for e in events:
        assert sum(e['policy_spend'].values()) <= e['actual_spent']
        if e['actual_spent'] < e['desired_spent']:
            assert e['actual_satisfaction'] is None
            assert e['expected_satisfaction'] == .9


def test_daily_posture_does_not_create_unselected_payment(monkeypatch):
    monkeypatch.setenv('EXP_PAYMENT_CHOICE', '1')
    events = [dict(category='식사', poi_id='A', actual_spent=12000,
                   policy_spend={}, coupon_eligible=True, price_factor=1)]
    result = apply_consumption_model(events, daily=30000, income_tier='중',
        tendency='', balance=100000, grant_avail={'P': 50000}, grant_use=.8)
    assert events[0]['policy_spend'] == {}
    assert result['grant_carry_out'] == 0


def test_no_activity_has_no_choice_mode_carry(monkeypatch):
    monkeypatch.setenv('EXP_PAYMENT_CHOICE', '1')
    result = apply_consumption_model([], daily=30000, income_tier='중',
        tendency='', balance=0, grant_avail={'P': 50000})
    assert result['grant_carry_out'] == 0


def test_category_entry_and_exit_remain_in_paired_sample():
    off = [dict(aid='A', d='d1', amt=100, l1='식사'),
           dict(aid='B', d='d1', amt=100, l1='카페'),
           dict(aid='C', d='d1', amt=0, l1='집')]
    on = [dict(aid='A', d='d2', amt=100, l1='카페'),
          dict(aid='B', d='d2', amt=50, l1='식사'),
          dict(aid='C', d='d2', amt=0, l1='집')]
    for row in off + on:
        row['kdi'] = None
    before, after = metric_values('sector_spend:카페', off, on, ['d1'], ['d2'])
    assert before == {'A': 0, 'B': 100, 'C': 0}
    assert after == {'A': 100, 'B': 0, 'C': 0}
    assert paired(before, after) == [100, -100, 0]


def test_invalid_date_window_rejected():
    with pytest.raises(ValueError):
        daterange('2021-10-02:2021-10-01')


def test_disjoint_policy_results_cannot_rank_candidates(tmp_path):
    root = Path(__file__).resolve().parents[3]
    table = json.loads((root / 'data/experiments/scoring_table.json').read_text(encoding='utf-8'))
    policies = [(name, spec) for name, spec in table.items()
                if not name.startswith('_') and spec.get('indicators')][:2]
    for index, (name, spec) in enumerate(policies):
        indicator = spec['indicators'][0]
        data = dict(label=f'candidate{index}', policy=name,
                    results=[dict(indicator, hit=True, got=indicator['expect'], mean=1)])
        (tmp_path / f'{index}.json').write_text(json.dumps(data), encoding='utf-8')
    result = subprocess.run([sys.executable, str(root / 'scripts/sim/rank_candidates.py'),
                             '--dir', str(tmp_path), '--stage', '3'],
                            capture_output=True)
    assert result.returncode == 2


def test_cash_conservation_across_mixed_wallet_baskets(monkeypatch):
    import random
    monkeypatch.setenv('EXP_PAYMENT_CHOICE', '1')
    rng = random.Random(729)
    for _ in range(60):
        balance = rng.randrange(50000)
        wallet = rng.randrange(50000)
        events = []
        for index in range(rng.randrange(1, 6)):
            amount = rng.randrange(1, 50000)
            events.append(dict(category='식사', poi_id=str(index),
                actual_spent=amount, policy_spend={'P': rng.randrange(amount + 1)},
                coupon_eligible=rng.choice([True, False]), price_factor=1))
        result = apply_consumption_model(events, daily=30000, income_tier='중',
            tendency='', balance=balance, restricted_envelopes=[
                dict(pid='P', amount=wallet, require_poi_eligible=True)])
        payment = sum(sum(e['policy_spend'].values()) for e in events)
        cash = result['online_total'] + sum(e['actual_spent'] for e in events) - payment
        assert 0 <= cash <= balance
        assert payment <= wallet
