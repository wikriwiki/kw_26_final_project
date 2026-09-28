"""Offline model stubs verify prompt/citation contracts, not LG answer quality."""
import json
from datetime import date
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from prompt_grounding import validate_stated_reason
from evidence_integrity import seal, verify
from scripts.experiments import collect_policy_stances as collector


def test_quotes_verify_input_span_without_claiming_semantic_truth():
    result = validate_stated_reason({'reasoning': '오늘 입력을 참고했다.', 'evidence_quote': 'mood=0.50'},
                                    '어제 상태 mood=0.50; 확인된 경험 없음')
    assert result['quote_span_verified'] and not result['semantic_support_verified']
    with pytest.raises(ValueError, match='does not occur'):
        validate_stated_reason({'reasoning': '어제 당구장에 갔다.', 'evidence_quote': '당구장 방문'}, '방문 기록 없음')


def test_no_smoking_stage1_allows_staying_home_and_rejects_invented_quote(monkeypatch):
    import stage1_intent as stage1
    monkeypatch.setenv('SIM_PROMPT_VARIANT', 'no_smoking_v1')
    text = '[E0001] 오늘 기분 mood=0.50; 피로 fatigue=0.30; 과거 방문 기록 없음'
    monkeypatch.setattr(stage1, '_format_dawn_blocks', lambda *args: text)
    seen = []
    response = {'events': [{'time': '08:00', 'anchor': 'residence', 'category': '집',
                            'intent': '휴식', 'reasoning': '주어진 컨디션을 보고 집에서 쉬기로 했다.',
                            'trigger': 'mood', 'evidence_ref': 'E0001'}]}
    def call(_mode, system, user, **kwargs):
        seen.append(system)
        return SimpleNamespace(model='LG-stub', usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
                               choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=json.dumps(response)))])
    monkeypatch.setattr(stage1, '_llm_call', call)
    ctx = SimpleNamespace(persona={}, prompt_timing={})
    result, meta = stage1.call_stage1('A', date(2017, 11, 20), ctx=ctx, max_retry=0, log_failures=False)
    assert all(ev.anchor == 'residence' for ev in result.events)
    assert result.events[0].evidence_quote == text.removeprefix('[E0001] ')
    assert 'evidence_quote' in seen[0]
    response['events'][0]['evidence_ref'] = 'E1'
    repaired, repaired_meta = stage1.call_stage1('A', date(2017, 11, 20), ctx=ctx,
                                                max_retry=0, log_failures=False)
    assert repaired.events[0].evidence_ref == 'E0001'
    assert repaired_meta['s1_timing']['attempts'][0]['evidence_ref_format_repairs'] == 1
    response['events'][0]['evidence_ref'] = 'E9999'
    with pytest.raises(RuntimeError, match='evidence_ref'):
        stage1.call_stage1('A', date(2017, 11, 20), ctx=ctx, max_retry=0, log_failures=False)


def test_stage1_retry_prompt_ends_with_failed_answer_and_reason(monkeypatch):
    import stage1_intent as stage1
    monkeypatch.setenv('SIM_PROMPT_VARIANT', 'no_smoking_v1')
    monkeypatch.setattr(stage1, '_format_dawn_blocks', lambda *args:
                        '[E0001] 오늘 집에서 쉬기로 했다')
    seen = []

    def call(_mode, system, user, **kwargs):
        seen.append(user)
        ref = 'E9999' if len(seen) == 1 else 'E0001'
        answer = {'events': [{'time': '08:00', 'anchor': 'residence', 'category': '집',
                              'intent': '휴식', 'reasoning': '오늘 집에서 쉬기로 했다.',
                              'trigger': 'none', 'evidence_ref': ref}]}
        return SimpleNamespace(model='LG-stub',
                               usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
                               choices=[SimpleNamespace(finish_reason='stop',
                                                        message=SimpleNamespace(content=json.dumps(answer)))])

    monkeypatch.setattr(stage1, '_llm_call', call)
    result, _ = stage1.call_stage1('A', date(2017, 11, 21),
                                   ctx=SimpleNamespace(persona={}, prompt_timing={}, appointment=[],
                                                       zone_candidates=[{'code': '11650101'}]),
                                   max_retry=1, log_failures=False)
    assert result.events[0].evidence_ref == 'E0001'
    assert len(seen) == 2 and seen[0] != seen[1]
    assert '"evidence_ref": "E9999"' in seen[1]
    assert "evidence_ref 'E9999' must identify exactly one numbered factual input line" in seen[1]
    assert '허용된 anchor는 residence, zone:11650101' in seen[1]
    assert seen[1].endswith('위 실수를 반복하지 마세요.')


def test_no_smoking_stage1_accepts_fifteen_minute_gap_but_not_duplicate_time(monkeypatch):
    import stage1_intent as stage1
    monkeypatch.setenv('SIM_PROMPT_VARIANT', 'no_smoking_v1')
    monkeypatch.setattr(stage1, '_format_dawn_blocks', lambda *args:
                        '[E0001] 오전에 집에서 쉬고 이동할 수 있다')
    events = [
        {'time': '07:30', 'anchor': 'residence', 'category': '집',
         'intent': '준비', 'reasoning': '아침에 집에서 준비한다.',
         'trigger': 'none', 'evidence_ref': 'E0001'},
        {'time': '07:45', 'anchor': 'residence', 'category': '집',
         'intent': '휴식', 'reasoning': '집에서 조금 더 쉰다.',
         'trigger': 'none', 'evidence_ref': 'E0001'},
    ]
    def call(_mode, system, user, **kwargs):
        return SimpleNamespace(model='LG-stub', usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
            choices=[SimpleNamespace(finish_reason='stop',
                                     message=SimpleNamespace(content=json.dumps({'events': events})))])
    monkeypatch.setattr(stage1, '_llm_call', call)
    ctx = SimpleNamespace(persona={}, prompt_timing={})
    result, _ = stage1.call_stage1('A', date(2017, 11, 20), ctx=ctx,
                                  max_retry=0, log_failures=False)
    assert [event.time for event in result.events] == ['07:30', '07:45']
    events[1]['time'] = '07:30'
    with pytest.raises(RuntimeError, match='strictly increase'):
        stage1.call_stage1('A', date(2017, 11, 20), ctx=ctx,
                           max_retry=0, log_failures=False)


def test_stage1_resolves_selected_source_line_and_rejects_unknown_reference(monkeypatch):
    import stage1_intent as stage1
    monkeypatch.setenv('SIM_PROMPT_VARIANT', 'no_smoking_v1')
    source = stage1._number_evidence_lines('## 페르소나\n오늘 확정 약속 없음\n현재 자금 30,000원')
    assert '[E0001] 오늘 확정 약속 없음' in source
    assert '[E0002] 현재 자금 30,000원' in source
    monkeypatch.setattr(stage1, '_format_dawn_blocks', lambda *args: source)
    event = {'time': '08:00', 'anchor': 'residence', 'category': '집',
             'intent': '휴식', 'reasoning': '확정 약속이 없어 집에서 쉬기로 했다.',
             'trigger': 'none', 'evidence_ref': 'E0001'}

    def call(_mode, system, user, **kwargs):
        return SimpleNamespace(model='LG-stub', usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
                               choices=[SimpleNamespace(finish_reason='stop',
                                                       message=SimpleNamespace(content=json.dumps({'events': [event]})))])

    monkeypatch.setattr(stage1, '_llm_call', call)
    ctx = SimpleNamespace(persona={}, prompt_timing={})
    result, _ = stage1.call_stage1('A', date(2017, 11, 20), ctx=ctx, max_retry=0, log_failures=False)
    assert result.events[0].evidence_ref == 'E0001'
    assert result.events[0].evidence_quote == '오늘 확정 약속 없음'
    event['evidence_ref'] = 'E9999'
    with pytest.raises(RuntimeError, match='evidence_ref'):
        stage1.call_stage1('A', date(2017, 11, 20), ctx=ctx, max_retry=0, log_failures=False)


def test_stage1_extracts_first_complete_json_with_nested_braces():
    from stage1_intent import _extract_json
    first = '{"events":[{"intent":"문자열 안의 }와 \\u007b"}],"daily_propensity":0.3}'
    assert _extract_json('```json\n' + first + '\n```\n{"extra":true}') == first


def test_stage2_render_uses_modeled_prices_without_repeat_directive():
    from stage2_poi import build_stage2_prompt
    ev = SimpleNamespace(time='18:00', anchor='zone:11650101', category='여가', sub_category='당구장', intent='여가')
    candidates = {0: [{'poi_id': 'P1', 'name': '가상 당구장', 'known': True, 'unit_anchor': 15000}]}
    text = build_stage2_prompt([ev], candidates, persona={'_no_smoking_prompt': '오늘 이용 규칙'}, recent_poi_ids={'P1'})
    assert '모형 참고단가' in text
    assert '단순 반복 자제' not in text
    assert 'evidence_ref' in text
    assert '/no_think' not in text


def test_stage2_schema_requires_source_ref_and_never_silently_falls_back(monkeypatch):
    import stage2_poi as stage2
    from stage1_intent import Stage1Output

    events = Stage1Output.model_validate({'events': [
        {'time': '08:00', 'anchor': 'residence', 'category': '집', 'intent': '기상'},
        {'time': '12:00', 'anchor': 'zone:11650101', 'category': '식사', 'intent': '점심'},
        {'time': '20:00', 'anchor': 'residence', 'category': '집', 'intent': '귀가'},
    ]})
    monkeypatch.setattr(stage2, 'fetch_candidates_for_events', lambda *args, **kwargs: {
        1: [{'poi_id': 'P1', 'name': '가상 식당', 'known': True,
             'unit_anchor': 12000, 'price_band': 2, 'price_factor': 1.0}],
    })
    monkeypatch.setattr(stage2, 'build_stage2_prompt', lambda *args, **kwargs: '## 후보\n- P1 가상 식당 점심 가능')
    response = {'picks': [{'order': 1, 'poi_id': 'P1', 'actual_spent': 12000,
                           'actual_satisfaction': 0.6, 'pick_reason': '점심 장소로 골랐다.',
                           'evidence_ref': 'E0001'}]}
    seen = []

    def call(_mode, system, user, **kwargs):
        seen.append((user, kwargs['response_format']))
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=12),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))])

    monkeypatch.setattr(stage2, '_llm_call', call)
    picks, _, meta = stage2.call_stage2('A', events, {'_no_smoking_prompt': '규칙'},
                                        date(2017, 11, 19), max_retry=0)
    assert picks.picks[0].evidence_ref == 'E0001'
    assert picks.picks[0].evidence_quote == '- P1 가상 식당 점심 가능'
    schema = seen[0][1]['json_schema']['schema']['properties']['picks']['items']
    assert schema['properties']['evidence_ref']['enum'] == ['E0001']
    assert 'evidence_ref' in schema['required']
    assert 'evidence_quote' not in schema['properties']
    assert meta.get('fallback_only') is not True
    response['picks'][0]['evidence_ref'] = 'E1'
    repaired, _, repaired_meta = stage2.call_stage2('A', events, {'_no_smoking_prompt': '규칙'},
                                                    date(2017, 11, 19), max_retry=0)
    assert repaired.picks[0].evidence_ref == 'E0001'
    assert repaired_meta['s2_timing']['attempts'][0]['evidence_ref_format_repairs'] == 1
    response['picks'][0]['evidence_ref'] = 'E9999'
    with pytest.raises(RuntimeError, match='grounded decision failed'):
        stage2.call_stage2('A', events, {'_no_smoking_prompt': '규칙'},
                           date(2017, 11, 19), max_retry=0)
    response['picks'][0]['evidence_ref'] = 'E0001'
    response['picks'][0]['poi_id'] = 'P2'
    with pytest.raises(RuntimeError, match='order 1'):
        stage2.call_stage2('A', events, {'_no_smoking_prompt': '규칙'},
                           date(2017, 11, 19), max_retry=0)


def test_stage2_retry_prompt_ends_with_failed_pick_and_reason(monkeypatch):
    import stage2_poi as stage2
    from stage1_intent import Stage1Output

    events = Stage1Output.model_validate({'events': [
        {'time': '12:00', 'anchor': 'zone:11650101', 'category': '식사', 'intent': '점심'},
    ]})
    monkeypatch.setattr(stage2, 'fetch_candidates_for_events', lambda *args, **kwargs: {
        1: [{'poi_id': 'P1', 'name': '가상 식당', 'unit_anchor': 12000,
             'price_band': 2, 'price_factor': 1.0}],
    })
    monkeypatch.setattr(stage2, 'build_stage2_prompt', lambda *args, **kwargs:
                        '## 후보\n- P1 가상 식당 점심 가능')
    seen = []

    def call(_mode, system, user, **kwargs):
        seen.append(user)
        answer = {'picks': [{'order': 1, 'poi_id': 'P2' if len(seen) == 1 else 'P1',
                             'actual_spent': 12000, 'actual_satisfaction': 0.6,
                             'pick_reason': '입력된 후보와 점심 목적에 맞는다.',
                             'pick_factor': 'distance', 'evidence_ref': 'E0001'}]}
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=12),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))])

    monkeypatch.setattr(stage2, '_llm_call', call)
    picks, _, _ = stage2.call_stage2('A', events, {'_no_smoking_prompt': '규칙'},
                                    date(2017, 11, 21), max_retry=1)
    assert picks.picks[0].poi_id == 'P1'
    assert len(seen) == 2 and seen[0] != seen[1]
    assert '"poi_id": "P2"' in seen[1]
    assert 'poi_id must be from that order candidates' in seen[1]
    assert seen[1].endswith('위 실수를 반복하지 마세요.')


def test_stage2_recovers_only_missing_grounded_order(monkeypatch):
    import stage2_poi as stage2
    from stage1_intent import Stage1Output

    events = Stage1Output.model_validate({'events': [
        {'time': '12:00', 'anchor': 'zone:11650101', 'category': '식사', 'intent': '점심'},
        {'time': '18:00', 'anchor': 'zone:11650101', 'category': '식사', 'intent': '저녁'},
    ]})
    candidates = {
        1: [{'poi_id': 'P1', 'name': '점심 식당', 'unit_anchor': 12000,
             'price_band': 2, 'price_factor': 1.0}],
        2: [{'poi_id': 'P2', 'name': '저녁 식당', 'unit_anchor': 13000,
             'price_band': 2, 'price_factor': 1.0}],
    }
    monkeypatch.setattr(stage2, 'fetch_candidates_for_events', lambda *args, **kwargs: candidates)
    monkeypatch.setattr(stage2, 'build_stage2_prompt', lambda *args, **kwargs: '점심 후보 P1\n저녁 후보 P2')
    seen = []

    def call(_mode, system, user, **kwargs):
        schema = kwargs['response_format']['json_schema']['schema']['properties']['picks']
        seen.append((user, schema))
        order = len(seen)
        reply = {'picks': [{'order': order, 'poi_id': f'P{order}',
                           'actual_spent': 12000, 'actual_satisfaction': 0.6,
                           'pick_reason': '입력된 후보와 오늘 식사 목적에 맞는다.',
                           'pick_factor': 'distance', 'evidence_ref': 'E0001'}]}
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=12),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))])

    monkeypatch.setattr(stage2, '_llm_call', call)
    result, _, meta = stage2.call_stage2('A', events, {'_no_smoking_prompt': '규칙'},
                                        date(2017, 11, 20), max_retry=0)
    assert [pick.order for pick in result.picks] == [1, 2]
    assert [pick.poi_id for pick in result.picks] == ['P1', 'P2']
    assert len(seen) == 2
    assert [x['properties']['order']['enum'] for x in seen[0][1]['items']['anyOf']] == [[1], [2]]
    assert seen[1][1]['items']['properties']['order']['enum'] == [2]
    assert 'minItems' not in seen[0][1]
    assert seen[0][1]['maxItems'] == 2
    assert seen[1][1]['maxItems'] == 1
    assert '누락된 order 2' in seen[1][0]
    assert '허용된 남은 order는 2' in seen[1][0]
    assert meta['s2_timing']['attempts'][0]['partial_picks_saved'] == 1


def test_stage2_retry_names_already_saved_order_error_and_remaining_order(monkeypatch):
    import stage2_poi as stage2
    from stage1_intent import Stage1Output

    events = Stage1Output.model_validate({'events': [
        {'time': '12:00', 'anchor': 'zone:11650101', 'category': '식사', 'intent': '점심'},
        {'time': '18:00', 'anchor': 'zone:11650101', 'category': '식사', 'intent': '저녁'},
    ]})
    monkeypatch.setattr(stage2, 'fetch_candidates_for_events', lambda *args, **kwargs: {
        1: [{'poi_id': 'P1', 'name': '점심 식당', 'unit_anchor': 12000,
             'price_band': 2, 'price_factor': 1.0}],
        2: [{'poi_id': 'P2', 'name': '저녁 식당', 'unit_anchor': 13000,
             'price_band': 2, 'price_factor': 1.0}],
    })
    monkeypatch.setattr(stage2, 'build_stage2_prompt', lambda *args, **kwargs:
                        '점심 후보 P1\n저녁 후보 P2')
    prompts = []

    def call(_mode, system, user, **kwargs):
        prompts.append(user)
        order = 1 if len(prompts) < 3 else 2
        answer = {'picks': [{'order': order, 'poi_id': f'P{order}',
                            'actual_spent': 12000, 'actual_satisfaction': 0.6,
                            'pick_reason': f'{len(prompts)}번째 식당 후보 중 선택한다.',
                            'pick_factor': 'distance', 'evidence_ref': 'E0001'}]}
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=12),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))])

    monkeypatch.setattr(stage2, '_llm_call', call)
    result, _, _ = stage2.call_stage2('A', events, {'_no_smoking_prompt': '규칙'},
                                     date(2017, 11, 21), max_retry=2)
    assert [pick.order for pick in result.picks] == [1, 2]
    assert len(prompts) == 3
    assert 'got 1; remaining 2' in prompts[2]
    assert '허용된 남은 order는 2' in prompts[2]
    assert prompts[2].endswith('위 실수를 반복하지 마세요.')


def test_evidence_ref_format_repair_requires_one_existing_exact_source():
    from grounded_schema import canonical_evidence_ref

    allowed = {'E0020': '실제로 제공된 한 줄', 'E0032': '다른 한 줄'}
    assert canonical_evidence_ref('E020', allowed) == 'E0020'
    assert canonical_evidence_ref('[E32]', allowed) == 'E0032'
    for invalid in ('E0033', 'E020,E032', 'E020 또는 E032', 'E9999', 'E123456789'):
        assert canonical_evidence_ref(invalid, allowed) == invalid


def test_night_interaction_resolves_source_ref(monkeypatch):
    import night_intent_llm as night
    import no_smoking_context

    monkeypatch.setattr(no_smoking_context, 'configured_context', lambda: object())
    monkeypatch.setattr(no_smoking_context, 'begin_llm_scope', lambda *args: None)
    monkeypatch.setattr(night, 'build_user_block',
                        lambda *args: '### [AGENT_A]\n- 최근 함께 점심을 먹음\n### [AGENT_B]\n- 오늘 활동 기록 없음')
    answer = {'intent': '기타', 'initiator_id': 'A', 'recipient_id': 'B',
              'topic_type': 'none', 'topic_value': None,
              'plan_signal': {'should_inject': False, 'target_day_offset': None,
                              'target_time': None, 'meeting_location_hint': None},
              'reasoning': '오늘은 확정할 구체적인 제안이 없다.', 'evidence_ref': 'E0002'}

    def call(_mode, system, user, **kwargs):
        assert '[E0002] - 오늘 활동 기록 없음' in user
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=12),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))])

    monkeypatch.setattr(night, '_llm_call', call)
    result = night._classify_intent(('A', 'B'), {'simulation_day': '2017-11-19'}, max_retry=0)
    assert result['evidence_ref'] == 'E0002'
    assert result['evidence_quote'] == '- 오늘 활동 기록 없음'
    answer['evidence_ref'] = 'E9999'
    invalid = night._classify_intent(('A', 'B'), {'simulation_day': '2017-11-19'}, max_retry=0)
    assert 'error' in invalid


class FixtureTokenizer:
    """Deterministic fake tokenization solely for boundary logic tests."""
    def apply_chat_template(self, messages, **kwargs):
        return [0] * ((len(json.dumps(messages, ensure_ascii=False)) + 3) // 4)


def packet():
    return seal({'schema_version': 1, 'kind': 'grounded_interview_packet',
                 'run_id': 'FIXTURE_RUN', 'arm': 'on', 'agent_id': 'A', 'through_day': '2017-12-16',
                 'days': ['2017-12-16'], 'missing_days': [], 'missing_night_days': [],
                 'cohort_sha256': 'a' * 64, 'source_sha256': 'b' * 64,
                 'evidence_items': [{'evidence_id': 'E1', 'day': '2017-12-16', 'kind': 'executed_receipt',
                                    'text': '가상 당구장 결제 12000원', 'value': {}, 'source_ref': {}}]})


def reply(quote='가상 당구장 결제', content=None):
    answer = {'stance': 'mixed', 'answer': '장점과 불편이 모두 있어 양면적입니다.',
              'stance_quote': '양면적입니다', 'confidence': 0.7, 'reasons': [{'evidence_id': 'E1', 'quote': quote}],
              'argument': {
                  'personal_situation': [{'claim': '당구장 결제 기록이 있다.',
                                         'evidence': [{'evidence_id': 'E1', 'quote': quote}]}],
                  'considerations': [{'claim': '이용 공간의 규칙이 내 여가 이용에 영향을 줄 수 있다.',
                                      'basis': 'inference', 'direction': 'uncertain',
                                      'personal_relevance': '당구장 결제 기록이 있어 이용 규칙을 고려할 이유가 있다.',
                                      'evidence': [{'evidence_id': 'E1', 'quote': quote}]}],
                  'weighing': '이용 관련성은 있으나 쾌적성과 편의의 실제 변화는 몰라 양면적으로 본다.',
                  'conditions': [], 'uncertainties': ['시설의 실제 연기 노출과 흡연실 유무는 모른다.']}}
    return SimpleNamespace(model='LG-fixture', choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps(answer, ensure_ascii=False) if content is None else content))])


def test_collector_keeps_valid_explicit_stance_and_durable_raw_io(tmp_path):
    def call(*args, **kwargs):
        assert len(list((tmp_path / 'calls').glob('*.request.json'))) == 1
        return reply()
    result = collector.collect_one(packet(), FixtureTokenizer(), call, tmp_path, 'LG-fixture', synthetic_fixture=True)
    assert result['response_status'] == 'answered'
    assert result['response']['stance'] == 'mixed'
    assert result['schema_version'] == 2
    assert result['provenance']['argument_contract_version'] == 2
    assert result['argument_quality']['review_required'] is True
    assert result['provenance']['synthetic_fixture']
    assert list((tmp_path / 'calls').glob('*.full_packet.json'))
    verify(json.loads(next((tmp_path / 'records').glob('*.json')).read_text(encoding='utf-8')))


def test_collector_invalid_citation_is_error_not_opposition_or_missing_vote(tmp_path):
    result = collector.collect_one(packet(), FixtureTokenizer(), lambda *a, **k: reply('없는 경험'),
                                   tmp_path, 'LG-fixture', synthetic_fixture=True)
    assert result['response_status'] == 'error'
    assert result['response'] is None
    raw = json.loads(next((tmp_path / 'calls').glob('*.response.json')).read_text(encoding='utf-8'))
    assert '없는 경험' in raw['content']


def test_empty_interview_is_no_response_not_neutral(tmp_path):
    result = collector.collect_one(packet(), FixtureTokenizer(), lambda *a, **k: reply(content=''),
                                   tmp_path, 'LG-fixture', synthetic_fixture=True)
    assert result['response_status'] == 'no_response' and result['response'] is None


def test_evidence_budget_keeps_explicit_omissions_and_no_future_items():
    original = packet()
    raw = dict(original)
    raw['evidence_items'] = [*original['evidence_items'], dict(original['evidence_items'][0], evidence_id='BIG', text='자료' * 20000)]
    selected = collector.bounded_packet(seal(raw), FixtureTokenizer(), 2200)
    assert collector.prompt_tokens(FixtureTokenizer(), selected) <= 2200
    assert selected['selection']['omitted_items'] > 0
    assert all(i['evidence_id'] != 'BIG' for i in selected['evidence_items'])


def test_context_labels_distinguish_hypothetical_from_experienced():
    assert collector.measurement_context('off', '2017-12-16') == 'hypothetical'
    assert collector.measurement_context('on', '2017-12-02') == 'hypothetical'
    assert collector.measurement_context('on', '2017-12-16') == 'experienced'
