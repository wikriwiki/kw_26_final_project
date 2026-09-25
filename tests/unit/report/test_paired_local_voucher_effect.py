"""P014 location proxies require complete, geographic ON/OFF ledgers."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'report'))
from export_spatial_daily_ledger import aggregate_day, verify_graph_policy  # noqa: E402
from paired_local_voucher_effect import score  # noqa: E402


def test_spatial_export_accounts_for_zero_spender_and_every_paid_place():
    states = [{'aid': 'a', 'online_spent': 10},
              {'aid': 'b', 'online_spent': 0}]
    spends = [
        {'aid': 'a', 'amt': 100, 'hdong': '11110101', 'pdong': '11110101'},
        {'aid': 'a', 'amt': 50, 'hdong': '11110101', 'pdong': '11110202'},
        {'aid': 'a', 'amt': 20, 'hdong': '11110101', 'pdong': '22220101'},
    ]
    rows = aggregate_day(states, spends, roster=['a', 'b'],
                         day='2020-09-22', arm='on', policy_id='P014')
    assert rows[0]['offline_spent'] == 170
    assert rows[0]['home_dong_spent'] == 100
    assert rows[0]['home_district_spent'] == 150
    assert rows[0]['out_district_spent'] == 20
    assert rows[0]['online_spent'] == 10
    assert rows[1]['offline_spent'] == 0
    assert rows[0]['prepaid_voucher_settlement_verified'] is False


def test_spatial_export_rejects_unlocated_positive_transaction():
    with pytest.raises(ValueError, match='geography'):
        aggregate_day([{'aid': 'a', 'online_spent': 0}],
                      [{'aid': 'a', 'amt': 100, 'hdong': '11110101', 'pdong': None}],
                      roster=['a'], day='2020-09-22', arm='on', policy_id='P014')


def test_spatial_export_rejects_unreconciled_policy_payment():
    with pytest.raises(ValueError, match='policy-wallet payment'):
        aggregate_day([{'aid': 'a', 'online_spent': 0}],
                      [{'aid': 'a', 'amt': 100, 'hdong': '11110101',
                        'pdong': '11110101', 'spent_from_policy': '{"P014": 10}'}],
                      roster=['a'], day='2020-09-22', arm='on', policy_id='P014')


def test_policy_graph_must_match_discount_terms():
    policy = {'id': 'P014', 'type': 'price_discount',
              'effective_from': '2020-09-21', 'effective_until': '2020-10-11',
              'discount_rate': 0.1, 'purchase_cap_monthly': 500000,
              'use_scope': 'home_district', 'poi_restricted': True,
              'eligibility': {'require_same_district': True}}
    observed = {key: policy[key] for key in
                ('id', 'type', 'effective_from', 'effective_until', 'poi_restricted')}
    observed['effective_from'] = date(2020, 9, 21)
    observed['effective_until'] = date(2020, 10, 11)
    observed['mech_params'] = ('{"discount_rate": 0.1, '
                               '"purchase_cap_monthly": 500000, '
                               '"use_scope": "home_district", '
                               '"eligibility": {"require_same_district": true}}')
    verify_graph_policy([{'policy': observed}], 'on', policy)
    with pytest.raises(ValueError, match='discount_rate'):
        verify_graph_policy([{'policy': dict(observed, mech_params=(
            '{"discount_rate": 0.2, "purchase_cap_monthly": 500000, '
            '"use_scope": "home_district", '
            '"eligibility": {"require_same_district": true}}'))}],
                            'on', policy)
    with pytest.raises(ValueError, match='control graph'):
        verify_graph_policy([{'policy': observed}], 'off', policy)


def _row(aid, day, arm, gross, home, district, away, online=0):
    return {'aid': aid, 'day': day, 'arm': arm, 'policy_id': 'P014',
            'offline_spent': gross, 'online_spent': online,
            'home_dong_spent': home, 'home_district_spent': district,
            'out_district_spent': away,
            'merchant_gross_basis': 'all_modeled_offline_poi_transactions',
            'prepaid_voucher_settlement_verified': False,
            's2_choice_status': 'unrepaired'}


def test_paired_location_score_uses_all_citizens_and_offline_share_denominator():
    day = '2020-09-22'
    off = [_row('a', day, 'off', 100, 20, 60, 40, online=100),
           _row('b', day, 'off', 0, 0, 0, 0)]
    on = [_row('a', day, 'on', 100, 40, 80, 20, online=100),
          _row('b', day, 'on', 0, 0, 0, 0)]
    result = score(on, off, roster=['a', 'b'], days=[day],
                   effect_days=[day], draws=50)
    assert result['citizens'] == 2
    assert result['indicators']['LV-1']['relative_change'] == 0
    assert result['indicators']['LV-1']['difference_won_per_citizen_day'] == 0
    assert result['indicators']['LV-2']['difference_percentage_points'] == pytest.approx(20)
    assert result['indicators']['LV-3']['difference_percentage_points'] == pytest.approx(-20)
    assert result['indicators']['home_district_spend_share']['difference_percentage_points'] == pytest.approx(20)
    assert result['external_magnitude_comparable'] is False
    assert result['indicators']['LV-2']['bootstrap_valid_draws'] <= 50


def test_paired_location_score_rejects_incomplete_or_unverified_rows():
    day = '2020-09-22'
    off = [_row('a', day, 'off', 100, 20, 60, 40)]
    on = [_row('a', day, 'on', 100, 40, 80, 20)]
    with pytest.raises(ValueError, match='matrix'):
        score(on, off, roster=['a', 'b'], days=[day], effect_days=[day], draws=0)
    on[0]['prepaid_voucher_settlement_verified'] = True
    with pytest.raises(ValueError, match='unsupported voucher'):
        score(on, off, roster=['a'], days=[day], effect_days=[day], draws=0)
