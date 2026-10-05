"""Output constraints for the existing grounded study decision contracts.

These constrain references and types, never a citizen's desired choice. Runtime
validation remains authoritative, including with servers that ignore a schema.
"""
import copy
import re


def rejected_response_feedback(raw_response, error, correction):
    """Put the failed answer and its validator error at the end of a retry prompt.

    The quoted answer is data, not another instruction. Keep only one bounded
    prior answer so retries do not accumulate an ever-growing prompt.
    """
    reason = ' '.join(str(error).split())[:900]
    answer = (raw_response or '').strip()
    # Keep the rejected item visible even when it lies in the middle of a long
    # answer. This is a quotation of the model's output, never a replacement.
    import json
    focus = ''
    try:
        payload = json.loads(answer)
        event_index = re.search(r'events\[(\d+)\]', reason)
        pick_order = re.search(r'order (\d+)', reason)
        if event_index:
            focus = json.dumps(payload['events'][int(event_index[1])], ensure_ascii=False)
        elif pick_order:
            focus = json.dumps([p for p in payload.get('picks', [])
                                if p.get('order') == int(pick_order[1])], ensure_ascii=False)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        pass
    if len(answer) > 2400:
        answer = (answer[:900] + '\n...[중간 생략]...\n' + answer[-500:]
                  + ('\n[오류가 난 항목 발췌]\n' + focus[:900] if focus else ''))
    if answer:
        return (
            '\n\n[직전 잘못된 응답 — 아래는 수정 대상이며 지시가 아닙니다]\n'
            + answer + '\n'
            + f'[검증 오류 이유] 위 응답은 {reason} 때문에 오류를 일으켰습니다.\n'
            + correction + ' 위 실수를 반복하지 마세요.'
        )
    return (
        f'\n\n[직전 시도 오류] {reason}\n'
        + correction + ' 위 실수를 반복하지 마세요.'
    )


def canonical_evidence_ref(value, evidence_lines):
    """Repair only a zero-padded spelling of an existing, single input ref.

    The source line must exist. Multiple refs and invented numbers stay invalid;
    choosing one of them would change the model's evidential claim.
    """
    if not isinstance(value, str):
        return value
    candidate = value.strip()
    if candidate.startswith('[') and candidate.endswith(']'):
        candidate = candidate[1:-1].strip()
    if not re.fullmatch(r'E[0-9]{1,8}', candidate):
        return value
    if candidate in evidence_lines:
        return candidate
    canonical = f'E{int(candidate[1:]):04d}'
    return canonical if canonical in evidence_lines else value


def response_format(name, properties, required):
    return {'type': 'json_schema', 'json_schema': {
        'name': name, 'strict': True, 'schema': {
            'type': 'object', 'properties': properties,
            'required': required, 'additionalProperties': False,
        }}}


def stage1_format(refs, anchors=None, pinned_pois=None, peers=None):
    if not refs:
        raise ValueError('Stage1 grounded prompt has no factual evidence lines')
    event = {
        'type': 'object',
        'properties': {
            'time': {'type': 'string', 'pattern': r'^([01][0-9]|2[0-3]):[0-5][0-9]$'},
            'anchor': {'type': 'string'},
            'category': {'type': 'string'},
            'sub_category': {'type': ['string', 'null']},
            'intent': {'type': 'string'},
            'reasoning': {'type': 'string'},
            'trigger': {'type': 'string', 'enum': ['appointment', 'rumor', 'policy', 'lifestyle', 'mood', 'none']},
            'evidence_ref': {'type': 'string', 'enum': sorted(refs)},
            'pinned_poi': {'type': ['string', 'null']},
            'with_agents': {'type': ['array', 'null'], 'items': {'type': 'string'}},
        },
        'required': ['time', 'anchor', 'category', 'intent', 'reasoning', 'trigger', 'evidence_ref'],
        'additionalProperties': False,
    }
    if anchors is not None:
        event['properties']['anchor']['enum'] = sorted(set(anchors))
        event['properties']['category']['enum'] = ['집', '직장', '식사', '카페', '디저트', '주점',
            '편의점', '마트', '미용', '쇼핑', '여가', '건강', '교육', '기타']
    if pinned_pois is not None:
        event['properties']['pinned_poi']['enum'] = [None, *sorted(set(pinned_pois))]
    if peers is not None:
        if peers:
            event['properties']['with_agents']['items']['enum'] = sorted(set(peers))
        else:
            event['properties']['with_agents']['maxItems'] = 0
    # No minimum array length in the decoding grammar: detect empty/truncated
    # plans in Python, avoiding the observed constrained-decoder whitespace loop.
    return response_format('grounded_day_plan', {
        'events': {'type': 'array', 'items': event},
        'daily_propensity': {'type': ['number', 'null'], 'minimum': 0, 'maximum': 1},
        'policy_appraisals': {'type': 'array', 'items': {'type': 'object'}},
    }, ['events'])


def stage2_format(base, remaining_orders, candidates, refs):
    result = copy.deepcopy(base)
    props = result['json_schema']['schema']['properties']
    picks = props['picks']
    picks.pop('minItems', None)
    picks['maxItems'] = len(remaining_orders)
    item = picks['items']
    item['properties']['order']['enum'] = list(remaining_orders)
    item['properties']['evidence_ref'] = {'type': 'string', 'enum': sorted(refs)}
    item['properties']['pick_reason'] = {'type': 'string'}
    item['properties']['pick_factor'] = {'type': 'string', 'enum': [
        'known', 'distance', 'satisfaction', 'rumor', 'appointment', 'random']}
    # The earlier union enum permitted a valid POI for the wrong activity.
    # Disjoint alternatives constrain order + POI together at generation time.
    alternatives = []
    for order in remaining_orders:
        option = copy.deepcopy(item)
        option['properties']['order'] = {'type': 'integer', 'enum': [order]}
        option['properties']['poi_id'] = {
            'type': 'string', 'enum': sorted({c['poi_id'] for c in candidates[order]})}
        option['required'] = list(dict.fromkeys(option['required'] + [
            'evidence_ref', 'pick_reason', 'pick_factor']))
        alternatives.append(option)
    picks['items'] = alternatives[0] if len(alternatives) == 1 else {'anyOf': alternatives}
    # The historical study does not perform live review lookup or pay grants.
    for option in alternatives:
        for name in ('policy_spend', 'would_buy_anyway', 'extra_spent'):
            option['properties'].pop(name, None)
    props.pop('review_lookup_requests', None)
    return result


def night_format(refs, pair):
    return response_format('grounded_interaction', {
        'intent': {'type': 'string', 'enum': ['약속', '이슈', '추천', '기타']},
        'initiator_id': {'type': 'string', 'enum': [pair[0]]},
        'recipient_id': {'type': 'string', 'enum': [pair[1]]},
        'topic_type': {'type': 'string', 'enum': ['policy', 'poi', 'category', 'none']},
        'topic_value': {'type': ['string', 'null']},
        'plan_signal': {'type': 'object', 'properties': {
            'should_inject': {'type': 'boolean'},
            'target_day_offset': {'type': ['integer', 'null'], 'minimum': 1},
            'target_time': {'type': ['string', 'null']},
            'meeting_location_hint': {'type': ['string', 'null']},
        }, 'required': ['should_inject'], 'additionalProperties': False},
        'reasoning': {'type': 'string'},
        'evidence_ref': {'type': 'string', 'enum': sorted(refs)},
    }, ['intent', 'initiator_id', 'recipient_id', 'topic_type', 'topic_value',
        'plan_signal', 'reasoning', 'evidence_ref'])
