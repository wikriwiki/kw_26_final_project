"""Development checks for factual entailment, separate from policy-effect scoring."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from factual_stress import CASES
from validate_prompt_v3 import atomic, digest


EXTRA = {
    'health_history': '평소 업종별 지출 중 건강 업종 비중은 35%다. 이는 과거 결제 구성이고 진단·복용약·정기 검진·오늘 필요한 의료 구매에 관한 정보는 없다.',
    'household': '정형 가구 정보: 배우자와 함께 사는 2인 가구이며 자녀는 없다. 역사 팟캐스트와 집에서 요리를 즐긴다.',
    'work_conflict': '정형 오늘 근무 정보: 09:00에 workplace로 출근해서 17:00까지 사무실에서 일하기로 확정했다. 집에서 직장까지 40분. 생활 서술에는 과거 재택 경험과 집에서 일하고 싶은 마음이 있다. 오늘의 재택근무 허가는 없다.',
    'working_appliance': '냉장고를 12년째 쓰고 있으며 현재 정상 작동한다. 보통 교체 주기는 11년이라고 한다. 고장 신고나 구매 예약은 없고 수리·교체가 필요한 상태라는 정보도 없다.',
}


def prepare(path):
    cases = {name: {'facts': spec['facts'], 'appointment': spec['appointment']} for name, spec in CASES.items()}
    cases.update({name: {'facts': text + ' 추가 영업·이동 제한 없음. 활성 제도·별도 지갑 없음.', 'appointment': '없음'} for name, text in EXTRA.items()})
    cells = []
    personas = []
    for name, spec in cases.items():
        aid = 'SYNTH_GROUND_' + name
        work = name == 'work_conflict'
        household = '가구 구성은 아래 정형 정보와 같다.' if name == 'household' else '한 사람 가구.'
        user = f'''## 개인 정보
ID: {aid}. 성인 40대. {household}
{'직장 있음: zone:11680640, 오늘 근무 조건은 아래 정형 정보를 따른다.' if work else '직장 없음. 오늘 출근 의무 없음.'}
현재 현금 80000원. 평소 하루 소비규모 20000원. 소비 성향은 보통.
독서·요리·동네 산책을 즐긴다. 아래에 적힌 것 외에는 질병·통증·고장·구매 예약 정보가 없다.
## 거주 및 장소 후보
거주지는 11680521. 외출 가능 후보 11680521과 11680640.
외출 anchor 허용값: "zone:11680521", "zone:11680640".
## 오늘의 조건
{spec['facts']}
## 오늘 약속
{spec['appointment']}
## 기억과 소문
과거 방문 기억 없음. 소문·지인 추천 없음.
## 오늘
2026-09-21 월요일 weekday. 오늘 하루 계획을 작성한다.
'''
        personas.append({'id': aid, 'work_poi_id': 'SYNTH_WORK' if work else None})
        cells.append({'aid': aid, 'case': name, 'arm': 'off', 'date': '2026-09-21',
                      'zones': ['11680521', '11680640'], 'user': user, 'context_sha256': digest(user)})
    target = Path(path)
    if target.exists():
        raise ValueError('Refusing overwrite')
    atomic(target, {'personas': personas, 'cells': cells, 'scope': 'Eight development cases, not unseen confirmation or policy effect targets.'})
    return hashlib.sha256(target.read_bytes()).hexdigest()


def audit(folder):
    folder = Path(folder)
    rows = [json.loads(s) for s in (folder/'responses.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    config = manifest['config']
    names = set(CASES) | set(EXTRA)
    expected = {(c['id'], seed, case) for c in config['candidates'] for seed in config['seeds'] for case in names}
    actual = Counter((r['variant'], r['replicate'], r['case']) for r in rows)
    if set(actual) != expected or any(n != 1 for n in actual.values()):
        raise ValueError('Incomplete or duplicate matrix')
    results = []
    for row in rows:
        failures = [] if row['valid'] else ['output_contract']
        try:
            events = json.loads(row['raw'])['events']
        except (KeyError, ValueError):
            events = []
            failures.append('missing_events')
        name = row['case']
        if any(e['trigger'] == 'rumor' for e in events):
            failures.append('absent_rumor')
        if name not in {'fixed_appointment', 'work_conflict'} and any(e['trigger'] == 'appointment' for e in events):
            failures.append('absent_appointment')
        if name not in {'stay_home', 'closed_cafes'} and any(e['trigger'] == 'policy' for e in events):
            failures.append('absent_policy')
        if name == 'stay_home' and any(e['anchor'] != 'residence' for e in events):
            failures.append('outside_prohibited')
        if name == 'closed_cafes' and any(e['category'] == '카페' for e in events):
            failures.append('closed_cafe')
        if name == 'fixed_appointment' and not any(e['time'] == '14:00' and e['anchor'] == 'zone:11680521' and e['category'] == '건강' for e in events):
            failures.append('missed_appointment')
        if name == 'work_conflict' and not any(e['time'] == '09:00' and e['anchor'] == 'workplace' for e in events):
            failures.append('missed_office_start')
        results.append({k: row[k] for k in ['variant', 'replicate', 'case']} | {'failures': failures})
    return {'scope': 'Machine checks only. Full manual factual review required before promotion.', 'results': results,
            'variants': {c['id']: {'checks': sum(r['variant'] == c['id'] for r in results),
                                  'passed': sum(r['variant'] == c['id'] and not r['failures'] for r in results)} for c in config['candidates']}}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare'); ap.add_argument('--audit'); ap.add_argument('--output')
    args = ap.parse_args()
    if args.prepare:
        print(prepare(args.prepare))
    else:
        if Path(args.output).exists():
            raise ValueError('Refusing overwrite')
        result = audit(args.audit); atomic(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
