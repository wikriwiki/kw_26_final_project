"""Historical probe inventory with separate empirical and simulated values.

This legacy probe cannot establish external effect-size accuracy. Its hardcoded
reference values require source and estimand auditing before use in a new evaluation.

    python scripts/report/build_indicator_comparison.py \
        --sim data/experiments/indicators_v25_block.json --out experiments/INDICATOR_COMPARISON.md
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

# id -> (policy, 실측값 or None, 기대부호, 한 줄 설명)
MEASURED = {
    'P012-1': ('P012', 20.82, '+', '적립업종 지출 (개인 내 쌍체차)'),
    'P012-2': ('P012', 2.85, '0', '제외업종 무반응 — 방어선'),
    'P012-3': ('P012', None, '+', '문턱 도달자 비율 > 0'),
    'P012-4': ('P012', None, '+', '1인 평균 캐시백 (실측 47,880원/월)'),
    'P012-5': ('P012', 33.36, 'rank', '가전·가구 > 이·미용 (+36.23 vs +2.87)'),
    'P012-6': ('P012', 21.0, '+', '월 한도 도달 비율 (실측 21.0%)'),
    'EM-2': ('EMERGENCY', 11.1, '+', '사용가능업종 순효과 (+11.1%p)'),
    'EM-3': ('EMERGENCY', 7.3, '+', '전체 카드매출 증가 (+7.3%)'),
    'EM-4': ('EMERGENCY', 7.8, 'rank', '준내구재 > 대면서비스 (+10.8 vs +3.0%p)'),
    'LV-1': ('LOCAL_VOUCHER', None, '0', '시민 총지출 무반응 — 내부 방어선 가설; 원문은 가맹점 매출'),
    'LV-2': ('LOCAL_VOUCHER', None, '+', '거주 행정동 소비 비중 증가 — 원문 직접 관측 없음'),
    'LV-3': ('LOCAL_VOUCHER', None, '-', '타 자치구 소비 비중 감소 — 원문 직접 관측 없음'),
    'DS-1': ('DISTANCING', -14.1, '-', '음식점 지출 감소 (−14.1%)'),
    'DS-2': ('DISTANCING', 4.2, '+', '소매업 지출 증가 (+4.2%)'),
    'DS-3': ('DISTANCING', 18.3, 'rank', '소매 > 음식점 (+4.2 − (−14.1))'),
    'DS-4': ('DISTANCING', None, '-', '카페 지출 감소'),
    'DS-6': ('DISTANCING', -4.3, 'rank', '관광특구 > 발달상권 감소폭 (−8.7 vs −4.4)'),
}

# Why a cell is empty on our side. Absent here means the probe produced a number.
CANNOT = {
    'P012-3': '문턱이 **월 적격지출** 기준이다. 하루 관측으로는 도달 여부가 정의되지 않는다',
    'P012-4': '캐시백은 **월 적립액**으로 정해진다. 하루 소비로는 정의되지 않는다',
    'P012-5': '카탈로그에 **가전·가구 품목이 없다**. 최고가가 커트 20,000원이다',
    'P012-6': '한도가 **월 10만원**이다. 하루에 도달할 수 없다',
    'EM-2': 'KDI는 사용가능업종 카드매출의 전년동기 대비 증가율이 지급 후 +11.1%p 변했다고 보고했다. 이 카탈로그의 당일 off/on 비교에는 전년 대조와 동일한 적격 업종 원장이 없다',
    'EM-4': '카탈로그에 **준내구재·대면서비스 구분이 없다**',
    'DS-4': '**무정책 팔의 카페 지출이 0원**이라 변화율이 정의되지 않는다. 카페 후보는 96칸 전부에 있었고 아무도 고르지 않았다',
    'DS-6': '장소에 **상권 유형(발달상권·관광특구) 태그가 없다**. 동 코드만 있다',
}

POLICY = {
    'P012': ('P012 카드 실적 캐시백', '2021-10-25', '2021 상생소비지원금', '기획재정부·KDI (2022.9)'),
    'EMERGENCY': ('P013 정책지갑 지급', '2020-05-14', '2020 1차 긴급재난지원금', 'KDI 2020-12-22 보도자료'),
    'LOCAL_VOUCHER': ('P014 할인 구매 상품권', '2020-09-23', '지역사랑상품권 할인발행', '조세재정연구원 (2020)'),
    'DISTANCING': ('거리두기 2단계 (정책 아님 · 사회 배경)', '2020-11-24', '수도권 2단계', '서울연구원 (2021.4)'),
}


def _number(value):
    return '—' if value is None else f'{value:+.1f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    document = json.loads(Path(args.sim).read_text(encoding='utf-8'))
    pooled = document['pooled']
    scored = sum(k in pooled for k in MEASURED)
    lines = [
        '# 과거 지표 프로브 — 실측 참고값과 시뮬레이션 대리값', '',
        f'> {len(MEASURED)}개 중 당시 프로브가 산출한 값 {scored}개. 이 표는 역사적 진단이며 현재 프롬프트의 크기 정확도 채점표가 아니다.',
        f'> 입력: `{args.sim}` · 반복 {document.get("runs", "미기록")}회 · 재표집 {document.get("draws", "미기록")}회.',
        f'> 원본 산출 방식: {document.get("method", "미기록")}',
        '> 실측과 시뮬레이션은 모집단·관측창·반사실·측정 단위가 맞는지 감사되지 않았다.',
        '> 따라서 두 수치를 같은 그래프에 그리거나 차이·배율·크기 적중률을 계산하지 않는다.',
        '> 실측값은 원자료와 정의를 다시 확인한 뒤 `scoring_table.json`의 추정량 감사로 판단한다.', '',
    ]
    for policy in ('P012', 'EMERGENCY', 'LOCAL_VOUCHER', 'DISTANCING'):
        name, day, real, source = POLICY[policy]
        ids = [key for key, row in MEASURED.items() if row[0] == policy]
        lines += ['---', '', f'## {name}', '',
                  f'역사적 프로브 날짜: {day} · 현실 대응: {real} · 참고 출처: {source}', '',
                  '### 외부 참고값 — 이 프로브와 크기 비교 불가', '',
                  '| 지표 | 당시 기재한 설명 | 기대 방향 | 외부 참고값 |',
                  '|---|---|:-:|---:|']
        for key in ids:
            _, measured, expected, description = MEASURED[key]
            lines.append(f'| {key} | {description} | `{expected}` | {_number(measured)} |')
        lines += ['', '### 당시 시뮬레이션 대리값 — 내부 진단 전용', '',
                  '| 지표 | 시뮬 값 | 관측 범위/구간 | 산출 상태 |',
                  '|---|---:|---|---|']
        for key in ids:
            block = pooled.get(key)
            if block is None:
                reason = CANNOT.get(key, '이 프로브에서 산출되지 않음')
                lines.append(f'| {key} | — | — | {reason} |')
                continue
            lo, hi = block.get('lo'), block.get('hi')
            if lo is not None and hi is not None:
                spread = f'보고된 구간 [{_number(lo)}, {_number(hi)}]'
            elif block.get('min') is not None and block.get('max') is not None:
                spread = f'런 범위 [{_number(block["min"])}, {_number(block["max"])}] (신뢰구간 아님)'
            else:
                spread = '범위 없음'
            lines.append(f'| {key} | {_number(block.get("mean"))} | {spread} | 산출 |')
        lines.append('')
    lines += ['시뮬레이션 후보의 외부 효과 크기 평가는 동일 추정량·기간·모집단·분모·단위의 감사가 통과한 별도 실험에서만 수행한다.', '']
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
    print('wrote', args.out, f'({scored}/{len(MEASURED)} historical proxies)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
