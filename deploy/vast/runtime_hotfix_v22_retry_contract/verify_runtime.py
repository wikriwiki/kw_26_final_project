"""Exercise the actually loaded day gate with stubs; never write graph/results."""
from datetime import date
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import stage1_intent as s1
import stage2_poi as s2


def response(value):
    return SimpleNamespace(model='offline-stub',
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=15),
        choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=json.dumps(value)))])


def main():
    os.environ['SIM_PROMPT_VARIANT'] = 'no_smoking_v1'
    os.environ['POLICY_BACKTEST_DETERMINISTIC'] = '1'
    ctx = SimpleNamespace(persona={}, prompt_timing={}, appointment=[], zone_candidates=[])
    invalid = {'events': [{'time': '08:00', 'anchor': 'residence', 'category': '집',
        'intent': '휴식', 'reasoning': '오늘 집에서 쉰다.', 'trigger': 'none', 'evidence_ref': 'E9999'}]}
    counts = []
    with patch.object(s1, '_format_dawn_blocks', lambda *a: '[E0001] 오늘 집에서 쉰다'):
        for day in (date(2017, 11, 23), date(2017, 11, 24)):
            requests = []
            def call(*args, **kwargs):
                requests.append(args[2])
                return response(invalid)
            with patch.object(s1, '_llm_call', call):
                try:
                    s1.call_stage1('test', day, ctx, max_retry=5, log_failures=False)
                except s1.Stage1Exhausted as exc:
                    counts.append({'day': str(day), 'calls': len(requests), 'reported': exc.attempts})
                else:
                    raise AssertionError('Invalid evidence passed')
            if day.day == 24:
                assert len(set(requests)) == 6
    assert counts == [{'day': '2017-11-23', 'calls': 2, 'reported': 6},
                      {'day': '2017-11-24', 'calls': 6, 'reported': 6}], counts
    plan = s1.Stage1Output.model_validate({'events': [
        {'time': '08:00', 'anchor': 'residence', 'category': '집', 'intent': '휴식'},
        {'time': '12:00', 'anchor': 'zone:1', 'category': '식사', 'intent': '점심'},
        {'time': '18:00', 'anchor': 'zone:1', 'category': '식사', 'intent': '저녁'},
    ]})
    candidates = {i: [{'poi_id': f'P{i}', 'name': '식당', 'unit_anchor': 12000,
                      'price_band': 2, 'price_factor': 1.0}] for i in (1, 2)}
    picks = [{'order': i, 'poi_id': f'P{i}', 'actual_spent': 12000,
              'actual_satisfaction': .6, 'pick_reason': '오늘 입력된 식사 목적에 맞는 후보를 선택한다.',
              'pick_factor': 'distance', 'evidence_ref': 'E0001'} for i in (1, 2)]
    requests = []
    def stage2_call(*args, **kwargs):
        requests.append(args[2])
        return response({'picks': picks[:1] if len(requests) == 1 else picks})
    with patch.object(s2, 'fetch_candidates_for_events', lambda *a, **k: candidates), \
         patch.object(s2, 'build_stage2_prompt', lambda *a, **k: '점심 후보 P1\n저녁 후보 P2'), \
         patch('neo4j_load._common.driver_session', side_effect=RuntimeError('offline; no graph reads')), \
         patch.object(s2, '_llm_call', stage2_call):
        result, _, meta = s2.call_stage2('test', plan, {'_no_smoking_prompt': '규칙'}, date(2017, 11, 24), max_retry=1)
    assert [p.poi_id for p in result.picks] == ['P1', 'P2']
    assert requests[1].startswith('[이번 응답에서 요청한 order] 2\n')
    assert meta['s2_timing']['retry_contract_sha256']
    print(json.dumps({'stubbed_runtime': 'passed', 'stage1': counts,
                      'stage2_calls': len(requests), 'real_model_test': False,
                      'graph_writes': 0, 'patch_sha256': meta['s2_timing']['retry_contract_sha256']}))


if __name__ == '__main__':
    main()
