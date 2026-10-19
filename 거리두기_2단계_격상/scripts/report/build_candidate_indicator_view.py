"""Historical candidate proxy inventory with separate external references.

This report compares candidates within the legacy probe; it does not grade their
distance from empirical effect sizes without a matched-estimand audit.

    python scripts/report/build_candidate_indicator_view.py \
        --cand v25=data/experiments/v7_v25_pooled.json \
        --cand v30=data/experiments/v7_v30_pooled.json \
        --out experiments/v7/indicator_view_v7.md
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_indicator_comparison import CANNOT, MEASURED, POLICY

def _number(value):
    return '—' if value is None else f'{value:+.1f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', action='append', required=True, metavar='label=pooled.json')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    candidates = {}
    metadata = {}
    for item in args.cand:
        if '=' not in item:
            ap.error('--cand must be label=pooled.json')
        label, file_name = item.split('=', 1)
        if not label or label in candidates:
            ap.error('candidate labels must be nonempty and unique')
        document = json.loads(Path(file_name).read_text(encoding='utf-8'))
        candidates[label] = document['pooled']
        metadata[label] = document
    labels = list(candidates)
    lines = [
        '# 과거 프롬프트 후보 — 시뮬레이션 대리값 비교', '',
        f'> 후보: {", ".join(labels)}. 같은 프로브의 후보끼리만 내부 비교할 수 있다.',
        '> 외부 참고값은 아래 별도 표에 둔다. 모집단·관측창·반사실·분모·측정 단위의 일치가 감사되지 않아',
        '> 외부값과 시뮬레이션값의 차이·배율·같은 눈금 그래프·크기 적중 판정을 계산하지 않는다.',
        '> 이 과거 프로브의 결과로 현재 범용 프롬프트를 선정하지 않는다.', '',
    ]
    for label in labels:
        document = metadata[label]
        sizes = document.get('cells_per_case_arm') or {}
        lines.append(f'- `{label}`: {sum(sizes.values()) if sizes else "미기록"}칸, '
                     f'{document.get("draws", "미기록")}회 재표집, '
                     f'재표집 단위 `{document.get("resample_unit", "미기록")}`')
    lines.append('')
    for policy in ('P012', 'EMERGENCY', 'LOCAL_VOUCHER', 'DISTANCING'):
        name, day, real, source = POLICY[policy]
        ids = [key for key, row in MEASURED.items() if row[0] == policy]
        lines += ['---', '', f'## {name}', '',
                  f'프로브 날짜: {day} · 현실 대응: {real} · 참고 출처: {source}', '',
                  '### 외부 참고값 — 후보 점수로 사용 불가', '',
                  '| 지표 | 기대 방향 | 외부 참고값 |', '|---|:-:|---:|']
        for key in ids:
            _, measured, expected, _ = MEASURED[key]
            lines.append(f'| {key} | `{expected}` | {_number(measured)} |')
        lines += ['', '### 후보별 시뮬레이션 대리값', '',
                  '| 지표 | ' + ' | '.join(labels) + ' |',
                  '|---|' + '---|' * len(labels)]
        for key in ids:
            cells = []
            for label in labels:
                block = candidates[label].get(key)
                if block is None:
                    cells.append('산출 불가')
                else:
                    value = _number(block.get('mean'))
                    lo, hi = block.get('lo'), block.get('hi')
                    if lo is not None and hi is not None:
                        value += f' [{_number(lo)}, {_number(hi)}]'
                    cells.append(value)
            lines.append('| ' + key + ' | ' + ' | '.join(cells) + ' |')
        missing = [key for key in ids if all(key not in candidates[label] for label in labels)]
        if missing:
            lines += ['', '당시 프로브의 미산출 사유:', '']
            lines += [f'- `{key}`: {CANNOT.get(key, "산출되지 않음")}' for key in missing]
        lines.append('')
    lines += ['후보 선정에는 동일 조건의 생성 형식 검증과 새 ON/OFF 원장 실험을 사용한다. '
              '외부 효과 크기는 추정량 감사가 통과한 지표에서만 채점한다.', '']
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
