"""Score P014 merchant-location proxies from complete same-calendar ON/OFF ledgers.

The policy is a prepaid discount purchase. Until purchase, redemption and subsidy
funding are reconciled, location and gross-spend contrasts are mechanism diagnostics,
not an empirical magnitude accuracy score.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

from paired_grant_effect import dates, read_jsonl, roster_file, verify_manifests

FIELDS = ('offline_spent', 'online_spent', 'home_dong_spent',
          'home_district_spent', 'out_district_spent')
CHOICES = frozenset(('unrepaired', 'not_applicable', 'partial_repair'))


def _index(rows: list[dict], roster: list[str], days: list[str], arm: str,
           policy_id: str) -> dict[tuple[str, str], dict]:
    expected = {(aid, day) for aid in roster for day in days}
    found = {}
    for row in rows:
        if row.get('arm') != arm or row.get('policy_id') != policy_id:
            raise ValueError('wrong arm or policy in spatial ledger')
        key = (row.get('aid'), row.get('day'))
        if key in found:
            raise ValueError(f'duplicate citizen-day: {arm} {key}')
        found[key] = row
    if set(found) != expected:
        raise ValueError(f'incomplete, extra or mismatched spatial matrix: {arm}')
    for row in found.values():
        if row.get('s2_choice_status') not in CHOICES:
            raise ValueError('missing or invalid Stage2 choice provenance')
        if row.get('merchant_gross_basis') != 'all_modeled_offline_poi_transactions':
            raise ValueError('unknown merchant gross basis')
        if row.get('prepaid_voucher_settlement_verified') is not False:
            raise ValueError('unsupported voucher settlement claim')
        for field in FIELDS:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'invalid {field}')
        if (row['home_district_spent'] + row['out_district_spent']
                != row['offline_spent']
                or row['home_dong_spent'] > row['home_district_spent']):
            raise ValueError('location amounts do not reconcile')
    return found


def _estimate(citizens: list[dict], day_count: int) -> dict:
    totals = {arm: {field: sum(person[arm][field] for person in citizens)
                    for field in FIELDS} for arm in ('off', 'on')}
    for arm in ('off', 'on'):
        totals[arm]['total_spent'] = (totals[arm]['offline_spent']
                                      + totals[arm]['online_spent'])
        if totals[arm]['offline_spent'] == 0:
            raise ValueError('location share undefined: zero offline spending')
    off, on = totals['off'], totals['on']
    def share(field: str) -> dict:
        a = off[field] / off['offline_spent']
        b = on[field] / on['offline_spent']
        return {'off': a, 'on': b, 'difference_percentage_points': 100 * (b - a),
                'denominator': 'all modeled offline POI merchant gross'}
    return {
        'LV-1': {'off_won': off['total_spent'], 'on_won': on['total_spent'],
                 'difference_won': on['total_spent'] - off['total_spent'],
                 'difference_won_per_citizen_day': (
                     (on['total_spent'] - off['total_spent'])
                     / (len(citizens) * day_count)),
                 'relative_change': ((on['total_spent'] - off['total_spent'])
                                     / off['total_spent'] if off['total_spent'] else None),
                 'denominator': 'all modeled merchant gross plus online spend'},
        'LV-2': share('home_dong_spent'),
        'LV-3': share('out_district_spent'),
        'home_district_spend_share': share('home_district_spent'),
    }


def score(on_rows: list[dict], off_rows: list[dict], *, roster: list[str],
          days: list[str], effect_days: list[str], policy_id: str = 'P014',
          draws: int = 2000, seed: int = 20260926) -> dict:
    if (not roster or len(set(roster)) != len(roster) or not days
            or dates(days[0], days[-1]) != days or not effect_days
            or dates(effect_days[0], effect_days[-1]) != effect_days
            or not set(effect_days).issubset(days) or draws < 0):
        raise ValueError('invalid roster, calendar, effect window or bootstrap draws')
    on = _index(on_rows, roster, days, 'on', policy_id)
    off = _index(off_rows, roster, days, 'off', policy_id)
    citizens = []
    for aid in roster:
        person = {}
        for arm, index in (('on', on), ('off', off)):
            person[arm] = {field: sum(index[(aid, day)][field] for day in effect_days)
                           for field in FIELDS}
        citizens.append(person)
    observed = _estimate(citizens, len(effect_days))
    rng = random.Random(seed)
    boot = {'LV-1': [], 'LV-2': [], 'LV-3': [],
            'home_district_spend_share': []}
    boot_won = []
    for _ in range(draws):
        try:
            trial = _estimate(rng.choices(citizens, k=len(citizens)), len(effect_days))
        except ValueError:  # Every sampled citizen had zero offline spend.
            continue
        boot_won.append(trial['LV-1']['difference_won_per_citizen_day'])
        lv1 = trial['LV-1']['relative_change']
        if lv1 is not None and math.isfinite(lv1):
            boot['LV-1'].append(lv1)
        for key in ('LV-2', 'LV-3', 'home_district_spend_share'):
            boot[key].append(trial[key]['difference_percentage_points'])
    for key, values in boot.items():
        values.sort()
        observed[key]['citizen_bootstrap_95_interval'] = (
            [values[int(0.025 * (len(values) - 1))],
             values[int(0.975 * (len(values) - 1))]] if values else None)
        observed[key]['bootstrap_valid_draws'] = len(values)
    boot_won.sort()
    observed['LV-1']['difference_won_per_citizen_day_bootstrap_95_interval'] = (
        [boot_won[int(0.025 * (len(boot_won) - 1))],
         boot_won[int(0.975 * (len(boot_won) - 1))]] if boot_won else None)
    return {
        'policy_id': policy_id, 'start': days[0], 'end': days[-1],
        'effect_start': effect_days[0], 'effect_end': effect_days[-1],
        'citizens': len(roster), 'effect_days': len(effect_days),
        'complete_matrix': True, 'spatial_ledger_reconciled': True,
        'indicators': observed,
        'comparison': 'indirect_proxy',
        'prepaid_voucher_settlement_verified': False,
        'external_magnitude_comparable': False,
        'scope': ('Paired synthetic merchant-gross and location-share contrasts. '
                  'Voucher purchase, redemption, subsidy and cash balances are not '
                  'reconciled; no empirical magnitude or no-effect pass claim.'),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--on', type=Path, required=True)
    parser.add_argument('--off', type=Path, required=True)
    parser.add_argument('--roster', type=Path, required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--effect-start', required=True)
    parser.add_argument('--effect-end', required=True)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--json-out', type=Path, required=True)
    args = parser.parse_args()
    roster, days = roster_file(args.roster), dates(args.start, args.end)
    provenance = verify_manifests(args.on, args.off, roster=roster,
                                  days=days, policy_id='P014')
    manifests = [json.loads(path.with_name(path.name + '.manifest.json').read_text(
        encoding='utf-8')) for path in (args.on, args.off)]
    for field in ('paired_environment_fingerprint', 'source_fingerprint', 'model_id'):
        if not manifests[0].get(field) or manifests[0][field] != manifests[1].get(field):
            raise ValueError(f'paired spatial arms differ in {field}')
    if any(manifest.get('prepaid_voucher_settlement_verified') is not False
           for manifest in manifests):
        raise ValueError('unsupported voucher settlement claim in manifest')
    result = score(read_jsonl(args.on), read_jsonl(args.off),
                   roster=roster, days=days,
                   effect_days=dates(args.effect_start, args.effect_end),
                   draws=args.draws)
    result['provenance'] = provenance
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.json_out.with_name(args.json_out.name + f'.tmp.{os.getpid()}')
    try:
        partial.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')
        partial.replace(args.json_out)
    finally:
        partial.unlink(missing_ok=True)
    print(args.json_out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
