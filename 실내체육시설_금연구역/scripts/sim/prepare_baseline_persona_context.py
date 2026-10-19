"""Explicit baseline-narrative repair ablation; no database or LLM mutation."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from persona_lineage import read_complete, restore_baseline


def replace_persona_section(user, rendered):
    start = '## 시민 정보\n'
    end = '\n\n## 개인별 정책 상태\n'
    if user.count(start) != 1 or user.count(end) != 1:
        raise ValueError('Ambiguous or missing persona section')
    prefix, tail = user.split(start)
    _, suffix = tail.split(end)
    return prefix + start + rendered + end + suffix


def render_persona(persona):
    from dawn_context import _format_persona
    text = _format_persona(persona)
    text = text.replace('직장: 없음', '근무 장소: 미제공 (취업 여부·근무 방식·오늘 일정은 이 값만으로 판단할 수 없음)')
    text = text.replace('평일 재택 ', '평일 집 체류 ').replace('주말 재택 ', '주말 집 체류 ')
    text = text.replace('평소 업종별 지출 구성(카드 실측)', '합성 시민에게 부여한 업종별 지출 구성')
    return ('자료 성격: 통계와 설정으로 구성한 합성 시민이다. 소비·행태 수치는 개인의 실제 거래 기록이 아닌 부여된 기준이다.\n'
            '직업과 생활 서술은 같은 원래 시민 설정에서 가져왔다. 근무·수업 시간표와 재택근무 허용 여부는 별도 입력이 없으면 미관측이다.\n' + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--allow-mapped-residence', action='store_true',
                    help='Explicitly retain later mapped POI residence despite different original administrative dong; audited')
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError('Refusing overwrite')
    raw = args.source.read_bytes()
    if len(raw) != args.source.stat().st_size:
        raise ValueError('Incomplete source input')
    source = json.loads(raw)
    if any('## 선택 가능한 활동 사전' in c['user'] for c in source['cells']):
        raise ValueError('Use prepared source, not already rendered runner inputs')
    baseline, baseline_meta = read_complete(args.baseline)
    personas, audit = restore_baseline(source['personas'], baseline, allow_mapped_residence=args.allow_mapped_residence)
    result = copy.deepcopy(source)
    result['personas'] = personas
    os.environ['EXP_DURABLES'] = '1'
    os.environ['EXP_CATLINE'] = 'fold'
    by_id = {p['id']: p for p in personas}
    by_audit = {r['id']: r for r in audit}
    from validate_prompt_v3 import digest, atomic
    for cell in result['cells']:
        rendered = render_persona(by_id[cell['aid']])
        if by_audit[cell['aid']]['residence_remapped']:
            rendered += '\n거주 위치 가정: 원래 통계상의 거주 동과 이후 연결된 주거 POI의 동이 다르다. 이 실험의 장소·이동 조건은 표시한 현재 주거 POI를 따른다. 이는 실제 이사 기록이 아니다.'
        cell['user'] = replace_persona_section(cell['user'], rendered)
        cell['context_sha256'] = digest(cell['user'])
    result['persona_overlay'] = {
        'source_sha256': hashlib.sha256(raw).hexdigest(), 'baseline_file': baseline_meta,
        'allow_mapped_residence': args.allow_mapped_residence,
        'scripts_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ['persona_lineage.py', Path(__file__).name, 'dawn_context.py']},
        'scope': 'Declared input-repair bundle: restore original qualitative prior, quarantine incompatible rich narratives, label synthetic data and unknown work location. Structured persona values, policy facts, state, menu and system prompt unchanged.',
        'limitations': ['A shared source identity is not proof of semantic or empirical validity.',
                       'Numeric fields retain their frozen later calibration; source baseline numeric values are audit-only.',
                       'The work activity vocabulary still depends on mapped work POI; no claim this fixes missing work/school schedules.'],
        'audit': audit}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic(args.out, result)
    print(json.dumps({'people': len(personas), 'cells': len(result['cells']),
                      'sha256': hashlib.sha256(args.out.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
