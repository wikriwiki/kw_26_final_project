"""Attach explicit channel metadata to the neutral four-case Day0 inputs.

This adapter handles this registered source format only; a new policy/background
provider must supply its own typed availability and evidence channels.
"""
import argparse
import hashlib
import json
from pathlib import Path
from evidence_contract import evidence_atoms, ROUTINE
from validate_prompt_v3 import atomic


def prepare(source):
    result = json.loads(json.dumps(source, ensure_ascii=False))
    people = {p['id']: p for p in source['personas']}
    for cell in result['cells']:
        if cell['case'] not in {'cashback', 'grant', 'local_voucher', 'distancing'}:
            raise ValueError('Unregistered input case')
        sections = {}
        label = None
        for line in cell['user'].splitlines():
            if line.startswith('## '):
                label = line[3:]; sections[label] = []
            elif label is not None:
                sections[label].append(line)
        if '(오늘 예정된 약속 없음)' not in '\n'.join(sections['오늘 예정된 약속']):
            raise ValueError('Not the registered empty-appointment Day0 source')
        if '(최근 기억 없음' not in '\n'.join(sections['과거 방문 기억']):
            raise ValueError('Not the registered no-memory Day0 source')
        public = []
        if cell['arm'] == 'on' and cell['case'] != 'distancing':
            public.extend(sections['현재 활성 정책의 조건'])
            public.extend(sections['개인별 정책 상태'])
        if not (cell['case'] == 'distancing' and cell['arm'] == 'off'):
            public.extend(line for line in sections['사회 배경'] if line.startswith('- ') and '서울 신규 확진' not in line)
        atoms = [s for s in evidence_atoms('\n'.join(public)) if s != ROUTINE]
        if not set(atoms) <= set(evidence_atoms(cell['user'])):
            raise ValueError('Evidence is not an exact input span')
        cell['allowed_triggers'] = ['none', 'lifestyle', 'mood'] + (['policy'] if atoms else [])
        cell['trigger_evidence'] = {'policy': atoms} if atoms else {}
        cell['fixed_times'] = []
        person = people[cell['aid']]
        minutes = person.get('commute_min')
        cell['minimum_transitions'] = []
        if person.get('work_poi_id') and minutes is not None:
            if isinstance(minutes, bool) or float(minutes) < 0:
                raise ValueError('Invalid commute duration')
            import math
            duration = math.ceil(float(minutes))
            cell['minimum_transitions'] = [{'from_anchor': a, 'to_anchor': b, 'minimum_minutes': duration}
                                           for a, b in [('residence', 'workplace'), ('workplace', 'residence')]]
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); out = Path(args.out)
    if out.exists(): raise ValueError('Refusing overwrite')
    raw = Path(args.source).read_bytes(); result = prepare(json.loads(raw))
    result['typed_channel_provenance'] = {'source_sha256': hashlib.sha256(raw).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'user_text_unchanged': True,
        'scope': 'Public restrictions and active policy facts may support policy labels; ordinary personal information and infection counts alone may not. No invented appointments/rumors in this registered Day0 input.',
        'commute_assumption': 'Treat supplied commute minutes as a lower bound for consecutive home/work anchors in this synthetic scenario. Intermediate POIs, route choice and transit schedules are not fully modelled.'}
    atomic(out, result); print(hashlib.sha256(out.read_bytes()).hexdigest())
