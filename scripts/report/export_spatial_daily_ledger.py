"""Export complete citizen-day geography for a matched non-grant policy run.

The output measures merchant gross sales by location. It does not certify prepaid
voucher purchases, household cash outflow, or an empirical treatment effect.
Run before resetting each arm's graph, with its own metrics directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path

from export_policy_daily_ledger import (POLICY_QUERY, SPEND_QUERY, STATE_QUERY,
                                        driver_session, verify_metrics)
from export_cashback_month import verify_cohorts, verify_metric_provenance
from paired_grant_effect import dates, roster_file
from report.audit_stage2_generation import choice_status, inspect

ROOT = Path(__file__).resolve().parents[2]


def _canonical(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if hasattr(value, 'iso_format'):
        return value.iso_format()
    if isinstance(value, date):
        return value.isoformat()
    return value


def verify_graph_policy(rows: list[dict], arm: str, policy: dict) -> None:
    if arm == 'off':
        if rows:
            raise ValueError('control graph contains a Policy node')
        return
    if len(rows) != 1:
        raise ValueError('treatment graph must contain exactly one Policy')
    raw = rows[0]['policy']
    mechanism = _canonical(raw.get('mech_params')) or {}
    if not isinstance(mechanism, dict):
        raise ValueError('graph Policy has invalid mechanism parameters')
    observed = {**mechanism, **{key: value for key, value in raw.items()
                              if value is not None and key != 'mech_params'}}
    for field in ('id', 'type', 'effective_from', 'effective_until',
                  'discount_rate', 'purchase_cap_monthly', 'use_scope',
                  'poi_restricted', 'eligibility'):
        if _canonical(observed.get(field)) != _canonical(policy.get(field)):
            raise ValueError(f'graph Policy disagrees with frozen file: {field}')


def _code(value: object, label: str) -> str:
    code = str(value or '').strip()
    if len(code) < 5 or not code.isdigit():
        raise ValueError(f'missing or invalid {label} geography')
    return code


def aggregate_day(states: list[dict], spends: list[dict], *, roster: list[str],
                  day: str, arm: str, policy_id: str) -> list[dict]:
    by_aid = {}
    for row in states:
        aid = row.get('aid')
        if aid in by_aid:
            raise ValueError(f'duplicate State for {aid} {day}')
        by_aid[aid] = row
    if set(by_aid) != set(roster):
        raise ValueError(f'incomplete State roster on {day}')
    amounts = {aid: [0, 0, 0, 0] for aid in roster}
    for row in spends:
        aid = row.get('aid')
        if aid not in amounts:
            raise ValueError(f'transaction for unregistered citizen: {aid} {day}')
        amount = row.get('amt')
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError(f'invalid transaction amount: {aid} {day}')
        if amount == 0:
            continue
        funding = _canonical(row.get('spent_from_policy')) or {}
        if not isinstance(funding, dict) or any(value for value in funding.values()):
            raise ValueError(f'unexpected policy-wallet payment: {aid} {day}')
        home = _code(row.get('hdong'), 'home')
        place = _code(row.get('pdong'), 'merchant')
        if len(home) != len(place):
            raise ValueError(f'incompatible dong code formats: {aid} {day}')
        totals = amounts[aid]
        totals[0] += amount
        if home == place:
            totals[1] += amount
        if home[:5] == place[:5]:
            totals[2] += amount
        else:
            totals[3] += amount
    out = []
    for aid in roster:
        state = by_aid[aid]
        online = state.get('online_spent')
        if isinstance(online, bool) or not isinstance(online, int) or online < 0:
            raise ValueError(f'missing or invalid State.online_spent: {aid} {day}')
        gross, home_dong, home_district, out_district = amounts[aid]
        if gross != home_district + out_district or home_dong > home_district:
            raise ValueError(f'geography ledger does not reconcile: {aid} {day}')
        out.append({'aid': aid, 'day': day, 'arm': arm, 'policy_id': policy_id,
                    'offline_spent': gross, 'online_spent': online,
                    'home_dong_spent': home_dong,
                    'home_district_spent': home_district,
                    'out_district_spent': out_district,
                    'merchant_gross_basis': 'all_modeled_offline_poi_transactions',
                    'prepaid_voucher_settlement_verified': False})
    return out


def export(*, roster: list[str], days: list[str], arm: str, policy_file: str,
           metrics_dir: Path, out: Path) -> int:
    if (not roster or len(roster) != len(set(roster)) or not days
            or dates(days[0], days[-1]) != days or arm not in ('on', 'off')):
        raise ValueError('nonempty unique roster, contiguous days and arm required')
    policy_path = ROOT / policy_file
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    if policy.get('type') != 'price_discount' or not policy.get('id'):
        raise ValueError('spatial export requires a price_discount policy')
    pid = policy['id']
    metrics = {day: verify_metrics(metrics_dir / f'day_{day}.jsonl', roster,
                                   arm, pid, policy['effective_from'],
                                   policy['effective_until']) for day in days}
    quality = inspect({day: list(metrics[day].values()) for day in days},
                      expected_per_day=len(roster))
    if not quality['quality_gate_pass']:
        raise ValueError(f'Stage2 generation quality gate failed: {quality["totals"]}')
    cohort = verify_cohorts(metrics_dir, days, roster)
    verify_metric_provenance({day: list(rows.values()) for day, rows in metrics.items()},
                             cohort)
    environment_fingerprints = set()
    source_fingerprints = set()
    model_ids = set()
    for rows in metrics.values():
        for row in rows.values():
            environment_fingerprints.add(row.get('paired_environment_fingerprint'))
            source_fingerprints.add(row.get('source_fingerprint'))
            decision = row.get('decision_provenance') or {}
            prompt_hash = decision.get('prompt_sha256')
            if (not isinstance(prompt_hash, str) or len(prompt_hash) != 64
                    or any(char not in '0123456789abcdef' for char in prompt_hash)):
                raise ValueError('missing or invalid Stage1 full-context prompt hash')
            model_ids.add(decision.get('model_id'))
            if row.get('policy_spend_today') or row.get('instant_discount_today'):
                raise ValueError('unreconciled non-grant payment in spatial run')
    if (len(environment_fingerprints) != 1 or not next(iter(environment_fingerprints))
            or len(source_fingerprints) != 1 or not next(iter(source_fingerprints))
            or len(model_ids) != 1 or not next(iter(model_ids))):
        raise ValueError('source, model or environment changed within spatial run')
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f'.tmp.{os.getpid()}')
    try:
        with driver_session() as session, temporary.open('w', encoding='utf-8') as stream:
            verify_graph_policy([dict(row) for row in session.run(POLICY_QUERY)], arm, policy)
            for day in days:
                states = [dict(row) for row in session.run(STATE_QUERY, day=day, aids=roster)]
                spends = [dict(row) for row in session.run(SPEND_QUERY, day=day, aids=roster)]
                rows = aggregate_day(states, spends, roster=roster, day=day,
                                     arm=arm, policy_id=pid)
                for row in rows:
                    row['s2_choice_status'] = choice_status(metrics[day][row['aid']])
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    with out.open('rb') as stream:
        output_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = {'arm': arm, 'policy_id': pid, 'start': days[0], 'end': days[-1],
                'citizens': len(roster), 'days': len(days),
                'rows': len(roster) * len(days), 'output_sha256': output_sha,
                'policy_file_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                'roster_sha256': hashlib.sha256(json.dumps(
                    sorted(roster), ensure_ascii=False).encode('utf-8')).hexdigest(),
                'quality_gate_pass': True, 'generation_totals': quality['totals'],
                'unrepaired_choice_trace_pass': quality['unrepaired_choice_trace_pass'],
                'paired_environment_fingerprint': next(iter(environment_fingerprints)),
                'source_fingerprint': next(iter(source_fingerprints)),
                'model_id': next(iter(model_ids)),
                'prepaid_voucher_settlement_verified': False, **cohort}
    manifest_path = out.with_name(out.name + '.manifest.json')
    partial = manifest_path.with_name(manifest_path.name + f'.tmp.{os.getpid()}')
    try:
        partial.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')
        partial.replace(manifest_path)
    finally:
        partial.unlink(missing_ok=True)
    return len(roster) * len(days)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--roster', type=Path, required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--arm', choices=('on', 'off'), required=True)
    parser.add_argument('--policy-file', required=True)
    parser.add_argument('--metrics-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(f'{args.out}: {export(roster=roster_file(args.roster), days=dates(args.start, args.end), arm=args.arm, policy_file=args.policy_file, metrics_dir=args.metrics_dir, out=args.out)} complete citizen-days')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
