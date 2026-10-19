"""Strict local evidence contracts. Hashes detect inconsistency, not hostile forgery."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date


class EvidenceError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def seal(value):
    result = dict(value)
    result.pop('integrity_sha256', None)
    result['integrity_sha256'] = digest(result)
    return result


def verify(value):
    if not isinstance(value, dict):
        raise EvidenceError('record must be an object')
    original = dict(value)
    checksum = original.pop('integrity_sha256', None)
    if not checksum or checksum != digest(original):
        raise EvidenceError('evidence integrity mismatch or legacy evidence without checksum')
    return value


def iso_day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise EvidenceError('date must be canonical YYYY-MM-DD')
    return value


def money(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            (isinstance(value,float) and not math.isfinite(value)) or
            value < 0 or value > 2**63-1 or value != int(value)):
        raise EvidenceError('money must be a finite nonnegative integer')
    return int(value)


def collection(value, expected_type):
    if value is None or value == '':
        return expected_type()
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except ValueError as exc:
        raise EvidenceError('corrupt stored JSON') from exc
    if not isinstance(parsed, expected_type):
        raise EvidenceError('stored JSON has the wrong type')
    return parsed


def verify_observation(row):
    verify(row)
    if row.get('kind') != 'purchase_receipt' or row.get('version') != 2:
        raise EvidenceError('unsupported observation schema')
    iso_day(row.get('observed_at'))
    if not all(isinstance(row.get(k), str) and row[k] for k in ('event_id','agent_id','run_id')):
        raise EvidenceError('missing observation identity')
    amount, own = money(row.get('amount')), money(row.get('own_paid'))
    facts = collection(row.get('policy_facts'), dict)
    total = 0
    for pid, fact in facts.items():
        if not isinstance(pid, str) or not isinstance(fact, dict) or type(fact.get('eligible_under_modeled_rules')) is not bool:
            raise EvidenceError('invalid policy fact')
        total += money(fact.get('paid'))
    if own > amount or total > amount - own:
        raise EvidenceError('observation violates accounting')
    if row.get('purchase_status') not in {'purchased', 'reduced', 'not_purchased'}:
        raise EvidenceError('invalid purchase status')
    return row


CLAIM_FIELDS = {'amount', 'own_paid', 'purchase_status', 'policy_paid', 'policy_eligible'}


def fact_value(observation, policy_id, field):
    if field not in CLAIM_FIELDS:
        raise EvidenceError('unsupported factual claim')
    if field == 'policy_paid':
        return observation['policy_facts'][policy_id]['paid']
    if field == 'policy_eligible':
        return observation['policy_facts'][policy_id]['eligible_under_modeled_rules']
    return observation[field]


def checked_claims(claims, ids, policy_id, evidence):
    if not isinstance(claims, list) or not 1 <= len(claims) <= 3:
        raise EvidenceError('one to three factual claims required')
    result = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {'event_id','field','value'}:
            raise EvidenceError('invalid factual claim shape')
        event_id, field = claim['event_id'], claim['field']
        if not isinstance(event_id, str) or event_id not in ids or not isinstance(field, str):
            raise EvidenceError('claim cites unexposed evidence')
        actual = fact_value(evidence[event_id], policy_id, field)
        # bool and 1 are equal in Python but are not the same factual statement.
        if type(claim['value']) is not type(actual) or claim['value'] != actual:
            raise EvidenceError('claim does not match executed fact')
        result.append(dict(claim))
    return result


def checked_evidence_citations(citations, packet):
    """Validate exposed exact quotations or typed JSON-pointer values.

    This verifies attribution only. A quotation of a model's public rationale
    remains an unverified subjective statement, never an executed experience.
    """
    verify(packet)
    if packet.get('kind') != 'grounded_interview_packet' or packet.get('schema_version') != 1:
        raise EvidenceError('unsupported interview packet')
    through = iso_day(packet.get('through_day'))
    evidence = {}
    for item in packet.get('evidence_items', []):
        eid = item.get('evidence_id')
        if (not isinstance(eid, str) or not eid or eid in evidence
                or iso_day(item.get('day')) > through or item.get('agent_id') != packet.get('agent_id')):
            raise EvidenceError('invalid or future exposed interview evidence')
        evidence[eid] = item
    if not isinstance(citations, list) or len(citations) > 20:
        raise EvidenceError('citations must be a bounded list')
    result = []
    for citation in citations:
        if not isinstance(citation, dict) or citation.get('evidence_id') not in evidence:
            raise EvidenceError('citation names unexposed evidence')
        item = evidence[citation['evidence_id']]
        if set(citation) == {'evidence_id', 'quote', 'start', 'end'}:
            text = item.get('text')
            start, end, quote = citation['start'], citation['end'], citation['quote']
            if (not isinstance(text, str) or not isinstance(quote, str) or not quote
                    or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(text) or text[start:end] != quote):
                raise EvidenceError('citation quote is not an exact exposed text span')
        elif set(citation) == {'evidence_id', 'pointer', 'value'}:
            pointer = citation['pointer']
            if not isinstance(pointer, str) or (pointer and not pointer.startswith('/')):
                raise EvidenceError('invalid citation JSON pointer')
            actual = item['value']
            try:
                for segment in pointer.split('/')[1:] if pointer else []:
                    segment = segment.replace('~1', '/').replace('~0', '~')
                    if isinstance(actual, list):
                        if not segment.isdigit() or str(int(segment)) != segment:
                            raise ValueError('invalid list index')
                        actual = actual[int(segment)]
                    elif isinstance(actual, dict):
                        actual = actual[segment]
                    else:
                        raise ValueError('not a container')
            except (IndexError, KeyError, ValueError, TypeError) as exc:
                raise EvidenceError('citation pointer is not exposed') from exc
            if type(citation['value']) is not type(actual) or canonical(citation['value']) != canonical(actual):
                raise EvidenceError('citation value does not match exposed evidence')
        else:
            raise EvidenceError('unsupported citation shape')
        result.append({**citation, 'source_kind': item['kind'],
                       'validation': 'exact_source_match_only'})
    return result
