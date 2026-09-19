"""Adapt registered development facts to typed actions; never an outcome oracle.

Known synthetic appointments get explicit executable labels. Closure requirements
stay evaluation-only: their forbidden actions remain available to the model.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from factual_stress import CASES
from grounding_stress import EXTRA
from validate_prompt_v3 import atomic


def adapt(source):
    result = copy.deepcopy(source)
    for cell in result['cells']:
        cell.update(provided_activities=[], required_activities=[], action_rules=[], evaluation_requirements=[])
        name = cell['case']
        if name == 'fixed_appointment':
            evidence = CASES[name]['appointment']
            assert evidence in cell['user']
            cell['provided_activities'] = [{'id': 'provided_appointment', 'anchors': ['zone:11680521'],
                'category': '건강', 'intent': '입력에 확정된 치과 검진 참석', 'purchase_channel': None, 'evidence': evidence}]
            cell['required_activities'] = [{'time': '14:00', 'activity_id': 'provided_appointment', 'anchor': 'zone:11680521', 'evidence': evidence}]
        elif name == 'work_conflict':
            evidence = EXTRA[name]
            assert evidence in cell['user']
            cell['required_activities'] = [{'time': '09:00', 'activity_id': 'office_work', 'anchor': 'workplace', 'evidence': evidence}]
        elif name == 'stay_home':
            assert CASES[name]['facts'] in cell['user']
            cell['evaluation_requirements'] = [{'kind': 'no_outside'}]
        elif name == 'closed_cafes':
            assert CASES[name]['facts'] in cell['user']
            cell['evaluation_requirements'] = [{'kind': 'forbid_activity', 'ids': ['cafe_dine_in','cafe_takeaway']}]
        elif name == 'distancing' and cell['arm'] == 'on':
            assert '카페는 시간과 무관하게 매장 이용 불가' in cell['user']
            assert '식당 매장 취식은 21시까지' in cell['user']
            cell['evaluation_requirements'] = [{'kind': 'forbid_activity', 'ids': ['cafe_dine_in']},
                                                {'kind': 'forbid_after', 'ids': ['meal_dine_in'], 'time': '21:00'}]
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); path = Path(args.out)
    if path.exists(): raise ValueError('Refusing overwrite')
    raw = Path(args.source).read_bytes(); result = adapt(json.loads(raw))
    result['action_adapter'] = {'source_sha256': hashlib.sha256(raw).hexdigest(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Registered development source adapter. No closure-based menu filtering. Evaluation requirements are not submitted to the model.'}
    atomic(path, result); print(hashlib.sha256(path.read_bytes()).hexdigest())
