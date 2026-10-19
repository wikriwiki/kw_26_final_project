"""Reuse durable night answers and account for exhausted pairs without fake events.

The six-call budget spans process restarts. A skipped pair is missing simulated
interaction data, not a model classification and not a zero-impact observation.
"""
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from evidence_integrity import EvidenceError, digest, seal, verify

ATTEMPT_OFFSET = ContextVar('night_attempt_offset', default=0)
MAX_ATTEMPTS = 6


def inspect_journal(root, path, identity, context_sha256):
    from day_resume import read_metric_rows
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root / 'evidence') or path.is_symlink():
        raise EvidenceError('Night journal escapes evidence root')
    rows = read_metric_rows(path)
    requests, responses, archived = {}, {}, []
    ids = set()
    for row in rows:
        verify(row)
        if row['record_id'] in ids or any(row.get(k) != v for k, v in identity.items()):
            raise EvidenceError('Duplicate or foreign night journal record')
        ids.add(row['record_id'])
        if row['kind'] == 'llm_request':
            if (row.get('stage') != 'night_intent' or digest(row['request']) != row['request_sha256']
                    or digest(row['context']) != row['context_sha256']
                    or row['context_sha256'] != context_sha256):
                raise EvidenceError('Night journal context or request differs')
            requests[row['record_id']] = row
        elif row['kind'] == 'llm_response':
            req = requests.get(row['request_id'])
            if (req is None or req['record_id'] in responses
                    or row['request_record_sha256'] != req['integrity_sha256']
                    or digest(row['response']) != row['response_sha256']):
                raise EvidenceError('Night response ledger link differs')
            responses[req['record_id']] = row
        elif row['kind'] == 'interaction':
            archived.append(row)
        else:
            raise EvidenceError('Unexpected record kind in night journal')
    if len(archived) > 1:
        raise EvidenceError('Multiple archived interactions in one night journal')
    reference = {'path': path.relative_to(root).as_posix(),
                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'attempts': len(requests), 'unanswered': len(requests) - len(responses)}
    return reference, archived


def history(root, pair, identity, context_sha256):
    from interview_evidence import _load_journal
    references, successes = [], []
    prefix = digest(sorted(pair))[:16]
    for path in sorted((root / 'evidence/v1' / identity['day']).glob(prefix + '_*.jsonl')):
        reference, archives = inspect_journal(root, path, identity, context_sha256)
        references.append(reference)
        for archived in archives:
            link = {'schema_version': 1, 'path': reference['path'],
                    'record_id': archived['record_id'], 'integrity_sha256': archived['integrity_sha256']}
            verified, _ = _load_journal(root, link, expected_kind='interaction')
            result = deepcopy(verified['interaction'])
            if (result.get('initiator_id'), result.get('recipient_id')) != tuple(pair):
                raise EvidenceError('Archived night answer changed participants')
            result['interview_evidence'] = link
            successes.append(result)
    if len(successes) > 1:
        raise EvidenceError('Multiple valid night answers for the same pair')
    return references, successes[0] if successes else None


def verify_skipped(root, receipt):
    verify(receipt)
    if (receipt.get('status') != 'skipped' or receipt.get('skip_kind') != 'failed_after_retries'
            or receipt.get('attempts') != MAX_ATTEMPTS or receipt.get('observed_interaction') is not False
            or len(receipt.get('pair', [])) != 2 or len(set(receipt['pair'])) != 2):
        raise EvidenceError('Invalid terminal night skip receipt')
    identity = {k: receipt[k] for k in ('run_id', 'arm', 'day', 'cohort_sha256', 'source_sha256')}
    identity['agent_ids'] = sorted(receipt['pair'])
    total = 0
    seen = set()
    for reference in receipt['journals']:
        if reference['path'] in seen:
            raise EvidenceError('Duplicate skip journal')
        seen.add(reference['path'])
        actual, archived = inspect_journal(root, Path(root) / reference['path'], identity,
                                          receipt['context_sha256'])
        if actual != reference or archived:
            raise EvidenceError('Changed or successful journal in skipped pair')
        total += actual['attempts']
    if total != MAX_ATTEMPTS:
        raise EvidenceError('Skipped night pair has not exhausted six calls')
    return receipt


def classify_recoverable(classifier, pair, data):
    from no_smoking_context import configured_context
    runtime = configured_context()
    if runtime is None:
        return classifier(pair, data, max_retry=5)
    from experience_provenance import source_fingerprint, atomic_json
    root = Path(os.environ['SIM_OUTPUT_DIR']).resolve()
    identity = {'run_id': os.environ.get('SIM_RUN_ID') or str(root), 'arm': runtime.arm,
                'day': str(data['simulation_day']), 'agent_ids': sorted(pair),
                'cohort_sha256': digest(runtime.agent_ids), 'source_sha256': source_fingerprint()}
    context_sha = digest(json.loads(json.dumps(data, default=str)))
    journals, cached = history(root, pair, identity, context_sha)
    if cached:
        return cached
    used = sum(ref['attempts'] for ref in journals)
    if used > MAX_ATTEMPTS:
        raise EvidenceError('Unclassified night pair already exceeds call budget')
    result = {'error': 'Six archived calls produced no validated interaction'}
    if used < MAX_ATTEMPTS:
        token = ATTEMPT_OFFSET.set(used)
        try:
            result = classifier(pair, data, max_retry=MAX_ATTEMPTS - used - 1)
        finally:
            ATTEMPT_OFFSET.reset(token)
        if result is not None and 'error' not in result:
            return result
        journals, cached = history(root, pair, identity, context_sha)
        if cached:
            return cached
    receipt = seal({**{k: v for k, v in identity.items() if k != 'agent_ids'},
                    'pair': list(pair), 'context_sha256': context_sha,
                    'status': 'skipped', 'skip_kind': 'failed_after_retries',
                    'attempts': MAX_ATTEMPTS, 'observed_interaction': False,
                    'reason': str((result or {}).get('error', 'classification failed'))[:400],
                    'journals': journals,
                    'night_progress_sha256': os.environ.get('NO_SMOKING_NIGHT_PROGRESS_SHA256')})
    verify_skipped(root, receipt)
    target = root / 'evidence/night_skips' / f"{identity['day']}_{digest(list(pair))[:20]}.json"
    if target.exists():
        prior = verify_skipped(root, json.loads(target.read_text(encoding='utf-8')))
        if prior['journals'] != receipt['journals'] or prior['context_sha256'] != context_sha:
            raise EvidenceError('Existing terminal night skip changed')
        return prior
    atomic_json(target, receipt)
    return receipt
