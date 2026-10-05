"""One synthetic missing-order retry through the existing model, no DB writes."""
from datetime import date
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import stage1_intent as s1
import stage2_poi as s2
from verify_runtime import response

plan = s1.Stage1Output.model_validate({'events': [
    {'time': '08:00', 'anchor': 'residence', 'category': '집', 'intent': '휴식'},
    {'time': '12:00', 'anchor': 'zone:1', 'category': '식사', 'intent': '점심'},
    {'time': '18:00', 'anchor': 'zone:1', 'category': '식사', 'intent': '저녁'},
]})
candidates = {i: [{'poi_id': f'P{i}', 'name': f'가상식당{i}', 'unit_anchor': 12000,
                  'price_band': 2, 'price_factor': 1.0}] for i in (1, 2)}
initial = {'order': 1, 'poi_id': 'P1', 'actual_spent': 12000, 'actual_satisfaction': .6,
    'pick_reason': '오늘 입력된 점심 목적에 맞는 후보를 선택한다.',
    'pick_factor': 'distance', 'evidence_ref': 'E0001'}
real_call = s2._llm_call
requests = []
def dispatch(mode, system, user, **kwargs):
    requests.append({'system': system, 'user': user, 'kwargs': kwargs})
    if len(requests) == 1:
        return response({'picks': [initial]})
    result = real_call(mode, system, user, **kwargs)
    requests[-1]['response'] = result.choices[0].message.content
    return result

with patch.object(s2, 'fetch_candidates_for_events', lambda *a, **k: candidates), \
     patch.object(s2, 'build_stage2_prompt', lambda *a, **k:
        '가상 테스트 시민은 오늘 점심과 저녁 식사가 필요하며 예산은 30000원이다.\n'
        'order 1: 점심, 후보 P1, 가상식당1, 참고 식사비 12000원.\n'
        'order 2: 저녁, 후보 P2, 가상식당2, 참고 식사비 12000원.'), \
     patch('neo4j_load._common.driver_session', side_effect=RuntimeError('no database in probe')), \
     patch.object(s2, '_llm_call', dispatch):
    result, _, meta = s2.call_stage2('synthetic-retry-probe', plan, {'_no_smoking_prompt': '규칙'},
                                   date(2017, 11, 24), max_retry=1)
assert [p.poi_id for p in result.picks] == ['P1', 'P2']
assert len(requests) == 2 and 'response' in requests[1]
record = {'synthetic_probe': True, 'real_model_calls': 1, 'graph_writes': 0,
          'requests': requests, 'result': result.model_dump(), 'meta': meta}
target = Path(__file__).with_name('synthetic_probe_result.json')
target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'synthetic_real_model_retry': 'passed', 'model_calls': 1,
                  'production_error_rate_verified': False,
                  'result_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                  's2_timing': meta['s2_timing']}))
