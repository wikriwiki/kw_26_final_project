"""Executed purchase facts -> bounded observations -> grounded policy appraisals.

No LLM or database calls. Decision text is never treated as an executed fact.
The State writer archives full receipts; only a bounded window enters the next Dawn.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

from evidence_contract import (EvidenceError, canonical, seal, verify, verify_observation,
                               collection, money, iso_day, checked_claims)

VERSION = 2
MAX_OBSERVATIONS = 8
STANCES = {'support', 'oppose', 'mixed', 'uncertain'}
PERSONA_FIELDS = {'income', 'job', 'life_stage', 'lifestyle', 'tendency',
                  'daily_wd', 'daily_we', 'home_dong', 'work_dong'}


def decode(value, default):
    return collection(value, type(default))


def visible_observations(state, today):
    """Exactly the evidence visible to the next existing decision call."""
    today = iso_day(str(today))
    rows = decode((state or {}).get('observations_json'), [])
    valid = {}
    for row in rows:
        verify_observation(row)
        if ((state or {}).get('_experience_agent_id') and row['agent_id'] != state['_experience_agent_id'] or
                (state or {}).get('experience_run_id') and row['run_id'] != state['experience_run_id']):
            raise EvidenceError('foreign evidence in personal memory')
        if row['observed_at'] < today:
            if row['event_id'] in valid and valid[row['event_id']] != row:
                raise EvidenceError('conflicting observation ID')
            valid[row['event_id']] = row
    selected, size = [], 0
    for row in reversed(sorted(valid.values(), key=lambda r: (r['observed_at'], r['event_id']))):
        cost = len(json.dumps(row, ensure_ascii=False))
        if size + cost > 5000:
            continue
        selected.append(row)
        size += cost
        if len(selected) == MAX_OBSERVATIONS:
            break
    return list(reversed(selected))


def receipts(aid, day, decisions, events, policies, namespace):
    """Archive intent separately from final accounting and expose only known facts.

    Invalid model payment requests are diagnostics, not experienced rejections.
    No visit, queue, travel, or causal additional-spending claims are manufactured.
    """
    iso_day(str(day))
    if not isinstance(namespace, str) or not namespace or not isinstance(aid, str) or not aid:
        raise EvidenceError('run and agent identity are required')
    if len(decisions) != len(events):
        raise EvidenceError('decision/execution event alignment changed')
    definitions = {str(p['id']): p for p in policies if p.get('id')}
    result = []
    for index, (decision, event) in enumerate(zip(decisions, events)):
        if not event.get('poi_id') or event.get('category') in {'집', '직장'}:
            continue
        requested = decode(decision.get('policy_spend'), {})
        paid = decode(event.get('policy_spend'), {})
        paid = {str(k): money(v) for k, v in paid.items()}
        amount = money(event.get('actual_spent', 0))
        if sum(paid.values()) > amount:
            raise ValueError('policy payments exceed executed purchase')
        policy_facts = {}
        diagnostics = []
        for pid, policy in definitions.items():
            # Only wallet grants have an executed receipt contract at present.
            if policy.get('type') != 'grant':
                continue
            eligible = ((not policy.get('poi_restricted') or event.get('coupon_eligible') is True)
                        and (not policy.get('target_l1s') or event.get('category') in policy['target_l1s']))
            req = requested.get(pid, 0)
            if isinstance(req, (int, float)) and req > 0 and not eligible:
                diagnostics.append({'code': 'invalid_payment_request', 'policy_id': pid})
            policy_facts[pid] = {'eligible_under_modeled_rules': eligible,
                                 'paid': paid.get(pid, 0)}
        if not policy_facts:
            continue
        raw_id = f'{namespace}|{aid}|{day}|{index}'
        record = {
            'event_id': 'EX_' + hashlib.sha256(raw_id.encode()).hexdigest()[:24],
            'version': VERSION, 'run_id': namespace, 'agent_id': aid, 'occurred_at': str(day),
            'observed_at': str(day), 'kind': 'purchase_receipt',
            'event_order': event.get('order', index), 'scheduled_time': event.get('time'),
            'poi_id': event['poi_id'], 'category': event.get('category'),
            'amount': amount, 'own_paid': amount - sum(paid.values()),
            'purchase_status': event.get('purchase_status') or ('purchased' if amount else 'not_purchased'),
            'policy_facts': policy_facts,
            'decision': {'planned_amount': decision.get('actual_spent'),
                         'requested_payments': requested},
            'diagnostics': diagnostics,
        }
        event['execution_event_id'] = record['event_id']
        result.append(seal(record))
    if len(decisions) != len(events):
        raise ValueError('decision/execution event alignment changed')
    return result


def observation_window(previous, new):
    # The archived receipt retains decisions and diagnostics. The citizen sees
    # only the final modeled receipt, never internal error corrections or prose.
    keep = ('version', 'run_id', 'event_id', 'agent_id', 'observed_at', 'kind', 'poi_id', 'category',
            'amount', 'own_paid', 'purchase_status', 'policy_facts')
    rows = decode(previous, [])
    for receipt in new:
        verify(receipt)
        rows.append(seal({k: receipt[k] for k in keep}))
    unique = {}
    for row in rows:
        verify_observation(row)
        old = unique.get(row['event_id'])
        if old is not None and old != row:
            raise EvidenceError('conflicting replay of executed event')
        unique[row['event_id']] = row
    identities = {(r['agent_id'],r['run_id']) for r in unique.values()}
    if len(identities) > 1:
        raise EvidenceError('cannot mix agents or runs in observation memory')
    return sorted(unique.values(), key=lambda r: (r['observed_at'], r['event_id']))[-MAX_OBSERVATIONS:]


def prompt_block(state, today):
    observations = visible_observations(state, today)
    if not observations:
        return ''
    relevant = {pid for row in observations for pid in row['policy_facts']}
    prior = {pid: {'stance': value['stance'], 'as_of': value['as_of']}
             for pid, value in decode((state or {}).get('policy_appraisals_json'), {}).items()
             if pid in relevant}
    return '\n\n[실행된 거래 관측과 기존 정책 입장]\n' + json.dumps({
        'observations': observations, 'prior_appraisals': prior,
    }, ensure_ascii=False) + '''
위 자료는 환경 엔진에서 정산된 거래 기록이다. decision 이유나 실제 인간의 정답이 아니다.
현재 페르소나와 이 관측에 근거해 정책 입장이 새로 형성되거나 바뀐 경우에만
기존 응답 JSON 최상위에 policy_appraisals 배열(최대 3개)을 추가한다. 없으면 []다.
각 항목: {"policy_id":"...", "stance":"support|oppose|mixed|uncertain",
"reason":"외부에 표현할 짧은 이유(300자 이내)", "evidence_ids":["EX_..."],
"persona_refs":["income|job|life_stage|lifestyle|tendency|daily_wd|daily_we|home_dong|work_dong"],
"claims":[{"event_id":"EX_...", "field":"policy_paid", "value":12000}]}.
claims는 1~3개이며 관측에 있는 정확한 값만 복사한다. 허용 field는 amount, own_paid,
purchase_status, policy_paid, policy_eligible이다. bool은 true/false, 금액은 정수다.
reason은 주관적 해석이며 검증된 사실과 구분된다. claims가 사실과 다르면 갱신은 기각된다.
실제 존재하는 관측 ID와 페르소나 필드만 인용한다. 상세 사고과정은 출력하지 않는다.
거래 사용 여부와 정책 찬반은 다르다. 혜택을 사용하면서 반대하거나 불편을 감수하며
찬성할 수도 있다. 입장을 강제로 바꾸지 않는다. 지원하지 않는 이동·대기·대화·거절 사건,
정책이 없었을 때의 지출을 만들어내지 않는다. 오늘 계획은 아직 실행된 경험이 아니다.
'''


def update_appraisals(aid, today, state, persona, proposals):
    today = iso_day(str(today))
    prior = dict(decode((state or {}).get('policy_appraisals_json'), {}))
    for item in prior.values():
        verify(item)
        if item.get('agent_id') != aid or item.get('as_of', '') > today:
            raise EvidenceError('invalid prior appraisal identity or time')
    evidence = {r['event_id']: r for r in visible_observations(state, today)
                if r.get('agent_id') == aid}
    accepted, rejected, seen = [], [], set()
    if not isinstance(proposals, list):
        rejected.append({'code': 'invalid_appraisal_payload'})
    for proposal in (proposals if isinstance(proposals, list) else [])[:3]:
        if not isinstance(proposal, dict):
            rejected.append({'code': 'invalid_appraisal'})
            continue
        pid = proposal.get('policy_id')
        ids, refs = proposal.get('evidence_ids'), proposal.get('persona_refs')
        reason = proposal.get('reason')
        valid = (isinstance(pid, str) and pid not in seen and
                 isinstance(proposal.get('stance'), str) and proposal['stance'] in STANCES and isinstance(reason, str) and
                 0 < len(reason.strip()) <= 300 and isinstance(ids, list) and 1 <= len(ids) <= 3 and
                 all(isinstance(i, str) and i in evidence and pid in evidence[i]['policy_facts'] for i in ids) and
                 isinstance(refs, list) and 1 <= len(refs) <= 3 and
                 all(isinstance(r, str) and r in PERSONA_FIELDS and persona.get(r) is not None for r in refs))
        if not valid:
            rejected.append({'code': 'ungrounded_appraisal', 'policy_id': pid})
            continue
        try:
            claims = checked_claims(proposal.get('claims'), ids, pid, evidence)
        except (EvidenceError, KeyError) as exc:
            rejected.append({'code': 'fact_claim_mismatch', 'policy_id': pid, 'detail': str(exc)})
            continue
        seen.add(pid)
        item = {'policy_id': pid, 'stance': proposal['stance'], 'reason': reason.strip(),
                'evidence_ids': list(dict.fromkeys(ids)),
                'persona_basis': {r: persona[r] for r in refs},
                'as_of': str(today), 'agent_id': aid, 'source': 'stage1_expressed_appraisal',
                'claims': claims, 'reason_status': 'subjective_unverified',
                'evidence_snapshot': [evidence[i] for i in dict.fromkeys(ids)]}
        item = seal(item)
        accepted.append({'previous_stance': prior.get(pid, {}).get('stance'), **item})
        prior[pid] = item
    return prior, accepted, rejected


def aggregate(rows, group_key='income'):
    """Latest completed snapshot per agent, one denominator per policy/group.

    Missing responses remain unmeasured, not neutral. This is simulated expressed
    stance, not an estimate of real public opinion or a calibrated confidence.
    """
    latest = {}
    namespace = None
    for row in rows:
        if row.get('experience_run_id'):
            if namespace is not None and namespace != row['experience_run_id']:
                raise ValueError('cannot mix experience runs')
            namespace = row['experience_run_id']
        if row.get('status') != 'ok' or not row.get('aid') or not row.get('experience_day'):
            continue
        aid = row['aid']
        old = latest.get(aid)
        if old and row['experience_day'] == old['experience_day'] and any(
                row.get(k) != old.get(k) for k in ('policy_appraisals','experience_group','experience_policy_ids')):
            raise EvidenceError('conflicting completed agent/day snapshots')
        if row['experience_day'] >= latest.get(aid, {}).get('experience_day', ''):
            latest[aid] = row
    buckets = defaultdict(lambda: {'agents': 0, 'measured': 0, 'stances': Counter(), 'as_of': Counter()})
    for row in latest.values():
        appraisals = row.get('policy_appraisals') or {}
        group = str((row.get('experience_group') or {}).get(group_key) or 'unknown')
        for pid in set(row.get('experience_policy_ids') or []) | set(appraisals):
            bucket = buckets[(pid, group)]
            bucket['agents'] += 1
            appraisal = appraisals.get(pid)
            if appraisal:
                bucket['measured'] += 1
                bucket['stances'][appraisal['stance']] += 1
                bucket['as_of'][appraisal['as_of']] += 1
    return [dict(policy_id=pid, group=group, agents=v['agents'], measured=v['measured'],
                 unmeasured=v['agents']-v['measured'], stances=dict(v['stances']),
                 appraisal_dates=dict(v['as_of'])) for (pid, group), v in sorted(buckets.items())]
