"""Fail-closed export gate. Verified simulation evidence is not verified public opinion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evidence_integrity import EvidenceError, verify, verify_observation, iso_day, checked_claims, digest, seal
from experience import aggregate
from experience_provenance import atomic_json


def validate_snapshot(row, day):
    verify(row)
    if row.get('experience_version') != 2 or row.get('experience_day') != day:
        raise EvidenceError('unsupported snapshot version or day')
    aid, run_id = row.get('aid'), row.get('experience_run_id')
    if not isinstance(aid, str) or not aid or not isinstance(run_id, str) or not run_id:
        raise EvidenceError('missing snapshot identity')
    provenance = row.get('decision_provenance') or {}
    if not row.get('source_fingerprint') or not row.get('execution_fingerprint') or not provenance.get('prompt_sha256') or not provenance.get('model_id'):
        raise EvidenceError('incomplete source/prompt/model provenance')
    for receipt in row.get('execution_receipts', []):
        verify_observation(receipt)
        if receipt.get('agent_id') != aid or receipt.get('run_id') != run_id or receipt.get('observed_at') != day:
            raise EvidenceError('receipt identity or day mismatch')
    for pid, appraisal in (row.get('policy_appraisals') or {}).items():
        verify(appraisal)
        as_of = iso_day(appraisal.get('as_of'))
        if appraisal.get('agent_id') != aid or appraisal.get('policy_id') != pid or as_of > day:
            raise EvidenceError('invalid appraisal identity or date')
        evidence = {}
        for observation in appraisal.get('evidence_snapshot', []):
            verify_observation(observation)
            if observation['agent_id'] != aid or observation['run_id'] != run_id or observation['observed_at'] >= as_of:
                raise EvidenceError('foreign or future evidence')
            eid = observation['event_id']
            if eid in evidence and evidence[eid] != observation:
                raise EvidenceError('conflicting evidence snapshot ID')
            evidence[eid] = observation
        ids = appraisal.get('evidence_ids')
        if not isinstance(ids, list) or not ids or any(not isinstance(i,str) or i not in evidence or pid not in evidence[i]['policy_facts'] for i in ids):
            raise EvidenceError('incomplete evidence')
        checked_claims(appraisal.get('claims'), ids, pid, evidence)
        if appraisal.get('stance') not in {'support','oppose','mixed','uncertain'}:
            raise EvidenceError('unknown stance')
        if appraisal.get('reason_status') != 'subjective_unverified':
            raise EvidenceError('subjective reasoning cannot be labelled as a verified fact')


def build_report(run_dir, day, group_by='income', *, cohort=None, min_group_size=5, strict=True):
    iso_day(day)
    if group_by not in {'income','job','life_stage'} or min_group_size < 1:
        raise EvidenceError('invalid aggregation settings')
    path = Path(run_dir) / 'metrics' / f'day_{day}.jsonl'
    raw = path.read_bytes()
    rows, malformed = [], 0
    for line in raw.decode('utf-8').splitlines():
        try:
            row = json.loads(line)
            if not isinstance(row,dict) or not isinstance(row.get('aid'),str) or not row['aid']:
                raise ValueError('not an identified record')
            rows.append(row)
        except ValueError:
            malformed += 1
    successful = {r['aid'] for r in rows if r.get('status') == 'ok'}
    failed = {r['aid'] for r in rows if r.get('status') != 'ok'} - successful
    relevant, rejected = [], 0
    for row in rows:
        if row.get('status') != 'ok':
            continue
        try:
            validate_snapshot(row, day)
            relevant.append(row)
        except (ValueError, TypeError, KeyError):
            rejected += 1
    namespaces = {r['experience_run_id'] for r in relevant}
    source_versions = {r['source_fingerprint'] for r in relevant}
    execution_versions = {r['execution_fingerprint'] for r in relevant}
    if len(namespaces) > 1 or len(source_versions) > 1 or len(execution_versions) > 1:
        raise EvidenceError('mixed runs or source versions')
    unique, duplicates = {}, 0
    for row in relevant:
        aid = row['aid']
        if aid in unique:
            if unique[aid] != row:
                raise EvidenceError('conflicting duplicate completed snapshot')
            duplicates += 1
        unique[aid] = row
    reasons = []
    if malformed: reasons.append('malformed_rows')
    if rejected: reasons.append('invalid_or_unverified_snapshots')
    if failed: reasons.append('failed_agents')
    if not unique: reasons.append('no_valid_completed_agents')
    expected, missing, unexpected = None, [], []
    if cohort is None:
        reasons.append('missing_cohort_manifest')
    else:
        ids = cohort.get('agent_ids') if isinstance(cohort,dict) else None
        if not isinstance(ids,list) or not ids or any(not isinstance(i,str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise EvidenceError('cohort requires distinct nonempty agent IDs')
        if namespaces and namespaces != {cohort.get('run_id')}:
            raise EvidenceError('cohort run differs from observations')
        if execution_versions and execution_versions != {cohort.get('execution_fingerprint')}:
            raise EvidenceError('cohort execution settings differ from observations')
        expected = len(ids)
        missing, unexpected = sorted(set(ids)-set(unique)), sorted(set(unique)-set(ids))
        if missing: reasons.append('missing_cohort_agents')
        if unexpected: reasons.append('unexpected_agents')
    groups = aggregate(unique.values(), group_by)
    for group in groups:
        group['suppressed'] = group['agents'] < min_group_size
        if group['suppressed']:
            # Reduces small-cell disclosure; not an anonymity guarantee.
            group['group'] = 'suppressed'
            group['stances'] = None
            group['appraisal_dates'] = None
        else:
            n = group['measured']
            group['stance_share_among_measured'] = {k:v/n for k,v in group['stances'].items()} if n else {}
    report = {
        'schema_version':2, 'day':day, 'group_by':group_by,
        'measure':'simulated_expressed_policy_stance',
        'release_status':'blocked' if reasons else 'validated_simulation_export',
        'release_blockers':reasons,
        'source':str(path.resolve()), 'source_sha256':hashlib.sha256(raw).hexdigest(),
        'cohort_sha256':digest(cohort) if cohort else None,
        'quality':{'expected_agents':expected,'completed_agents':len(successful),'valid_agents':len(unique),
                   'failed_agents':len(failed),'missing_agents':len(missing),'unexpected_agents':len(unexpected),
                   'malformed_rows':malformed,'rejected_rows':rejected,'duplicate_rows':duplicates},
        'groups':groups,
        'notes':['Validated factual fields do not verify subjective prose or human predictive accuracy.',
                 'Unmeasured is not neutral; appraisal dates show carried-forward measurements.',
                 'SHA-256 checks are not signatures and do not prevent an authorized operator from replacing data.'],
    }
    if strict and reasons:
        raise EvidenceError('export blocked: '+', '.join(reasons))
    return seal(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--day', required=True)
    parser.add_argument('--group-by', choices=['income','job','life_stage'], default='income')
    parser.add_argument('--cohort', help='JSON containing run_id and agent_ids')
    parser.add_argument('--min-group-size', type=int, default=5)
    parser.add_argument('--allow-degraded', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        cohort = json.loads(Path(args.cohort).read_text(encoding='utf-8')) if args.cohort else None
        report = build_report(args.run_dir,args.day,args.group_by,cohort=cohort,
                              min_group_size=args.min_group_size,strict=not args.allow_degraded)
        source = Path(args.run_dir)/'metrics'/f'day_{args.day}.jsonl'
        if Path(args.out).resolve() == source.resolve() or (args.cohort and Path(args.out).resolve() == Path(args.cohort).resolve()):
            raise EvidenceError('output must not overwrite evidence or cohort manifest')
        atomic_json(args.out,report)
    except (EvidenceError,OSError,ValueError) as exc:
        parser.exit(2,str(exc)+'\n')
    print(Path(args.out).resolve())


if __name__ == '__main__':
    main()
