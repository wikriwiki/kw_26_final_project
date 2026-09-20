"""Put every answer-key indicator next to our simulation's value, one row each.

Nothing is summarised away. Seventeen indicators are listed; the nine this probe can
build carry a number and a sign verdict, and the eight it cannot carry the reason the
cell is empty. An empty cell is a result - it says the probe is not built to answer
that question, which is different from answering it wrongly.

Magnitudes are shown as a ratio, never as matching won. The catalog sells unit items
and the studies measure card sales, so only ratios are the same kind of number.

    python scripts/report/build_indicator_comparison.py \
        --sim data/experiments/indicators_v25.json --out experiments/INDICATOR_COMPARISON.md
"""
from __future__ import annotations

import argparse
import io
import json
import unicodedata
from pathlib import Path

WIDTH = 18

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
    'LV-1': ('LOCAL_VOUCHER', 0.0, '0', '소비지출 규모 무반응 — 방어선'),
    'LV-2': ('LOCAL_VOUCHER', None, '+', '거주 행정동 소비 비중 증가'),
    'LV-3': ('LOCAL_VOUCHER', None, '-', '타 자치구 소비 비중 감소'),
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
    'EM-2': '실측은 지급 전(−4.0%)과 지급 후(+7.1%)의 차이인데, **우리 off 팔이 곧 지급 전**이라 견줄 기준선이 없다 (`scoring_table` 에 `not_scorable` 로 기록돼 있다)',
    'EM-4': '카탈로그에 **준내구재·대면서비스 구분이 없다**',
    'DS-4': '**무정책 팔의 카페 지출이 0원**이라 변화율이 정의되지 않는다. 카페 후보는 96칸 전부에 있었고 아무도 고르지 않았다',
    'DS-6': '장소에 **상권 유형(발달상권·관광특구) 태그가 없다**. 동 코드만 있다',
}

POLICY = {
    'P012': ('P012 카드 실적 캐시백', '2021-10-25', '2021 상생소비지원금', '기획재정부·KDI (2022.9)'),
    'EMERGENCY': ('P013 정책지갑 지급', '2020-05-14', '2020 1차 긴급재난지원금', 'KDI FOCUS'),
    'LOCAL_VOUCHER': ('P014 할인 구매 상품권', '2020-09-23', '지역사랑상품권 할인발행', '조세재정연구원 (2020)'),
    'DISTANCING': ('거리두기 2단계 (정책 아님 · 사회 배경)', '2020-11-24', '수도권 2단계', '서울연구원 (2021.4)'),
}


def cells_wide(text):
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)


def pad(text, width):
    return text + ' ' * max(0, width - cells_wide(text))


def bar(value, span):
    """Diverging bar with zero fixed at the centre column."""
    if span <= 0:
        return ' ' * WIDTH + '│' + ' ' * WIDTH
    n = max(0, min(WIDTH, int(round(abs(value) / span * WIDTH))))
    if value >= 0:
        return ' ' * WIDTH + '│' + '█' * n + ' ' * (WIDTH - n)
    return ' ' * (WIDTH - n) + '█' * n + '│' + ' ' * WIDTH


def verdict(expect, measured, sim, spread=None):
    """Sign agreement only. Magnitude is reported separately, never as a pass mark."""
    if sim is not None and sim == 0.0 and spread == 0.0:
        # Both arms produced zero. Nothing moved because nothing was there; calling that
        # agreement with a negative expectation would be a free pass.
        return '**공허 — 양쪽 팔 모두 0**'
    if expect == '0':
        return '방어선 — 아래 주석'
    if expect == 'rank':
        if measured is None or sim is None:
            return '—'
        return '순위 일치' if (measured > 0) == (sim > 0) else '**순위 반대**'
    if sim is None:
        return '—'
    want = 1 if expect == '+' else -1
    got = 1 if sim > 0 else -1
    return '부호 일치' if want == got else '**부호 반대**'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    sim = json.loads(Path(args.sim).read_text(encoding='utf-8'))['pooled']

    L = ['# 정답지 지표 × 시뮬레이션 — 하나씩 전부', '',
         '> 시뮬 값은 **현행 프롬프트 v25 로 돌린 네 런**(seed 59001·60001·61001·62001)을',
         '> **합쳐서 한 번 나눈 값**이다(384칸). 비율의 평균이 아니라 합계의 비율이다 —',
         '> 분모가 몇 건일 때 비율의 평균은 가장 얇은 런에 끌려간다.',
         '> 괄호는 **칸 붓스트랩 95% 구간**(2,000회). 구간이 0을 지나면 부호조차 말할 수 없다.', '',
         '> **금액은 맞대지 않는다.** 카탈로그는 낱개 단가(쌀 1kg 5,000원)이고 실측은 카드매출이다.',
         '> 비율 대 비율로만 비교하고, 크기는 **배율**로만 적는다.', '']

    total = len(MEASURED)
    scored = sum(1 for k in MEASURED if k in sim)
    tally = {}
    for k, (_, m, expect, _d) in MEASURED.items():
        if k not in sim:
            tally['산출 불가'] = tally.get('산출 불가', 0) + 1
            continue
        s_ = sim[k]
        tally[verdict(expect, m, s_['mean'],
                      0.0 if s_['max'] == 0.0 and s_['min'] == 0.0 else None)] =             tally.get(verdict(expect, m, s_['mean'],
                              0.0 if s_['max'] == 0.0 and s_['min'] == 0.0 else None), 0) + 1
    L += [f'**지표 {total}개 중 시뮬이 값을 낸 것 {scored}개, 낼 수 없는 것 {total - scored}개.**',
          '빈 칸은 결과다 — 틀린 답이 아니라 **이 프로브가 답하도록 만들어지지 않은 질문**이라는 뜻이다.', '',
          '| 판정 | 개수 |', '|---|---:|']
    for k in sorted(tally, key=lambda x: -tally[x]):
        L.append(f'| {k} | {tally[k]} |')
    L += ['']

    for pol in ('P012', 'EMERGENCY', 'LOCAL_VOUCHER', 'DISTANCING'):
        name, date, real, key = POLICY[pol]
        ids = [k for k, v in MEASURED.items() if v[0] == pol]
        L += ['---', '', f'## {name}', '',
              f'- 칸의 날짜 **{date}** · 현실 대응 **{real}** · 정답지 **{key}**', '']
        vals = [abs(MEASURED[k][1]) for k in ids if MEASURED[k][1] is not None]
        vals += [abs(sim[k]['mean']) for k in ids if k in sim]
        span = max(vals) if vals else 1.0
        left, right = '← 감소', '증가 →'
        head = (' ' * 9 + ' ' * max(0, WIDTH - cells_wide(left)) + left
                + '0' + right + ' ' * max(0, WIDTH - cells_wide(right)))
        L += ['```', head]
        for k in ids:
            m = MEASURED[k][1]
            L.append(f'{k:<9}{bar(m, span) if m is not None else " " * (2*WIDTH+1)}  '
                     f'{"실측 %+.1f" % m if m is not None else "실측 수치 없음"}')
            if k in sim:
                s = sim[k]
                stable = s.get('sign_stable')
                flag = '' if stable else '  ← 0을 지난다'
                L.append(f'{"":<9}{bar(s["mean"], span)}  시뮬 {s["mean"]:+.1f}'
                         f'  [{s["min"]:+.1f}, {s["max"]:+.1f}]{flag}')
            else:
                L.append(f'{"":<9}{" " * (2*WIDTH+1)}  시뮬 — 산출 불가')
            L.append('')
        L += ['```', '',
              '| 지표 | 무엇을 재나 | 기대 | 실측 | 시뮬(합산) | 95% 구간 | 부호 | 크기 |',
              '|---|---|:-:|---:|---:|---|---|---|']
        for k in ids:
            _, m, expect, desc = MEASURED[k]
            if k in sim:
                s = sim[k]
                simtxt = f'{s["mean"]:+.1f}'
                if m and s['mean'] and expect != '0':
                    ratio = f'**{abs(s["mean"] / m):.1f}배**'
                elif expect == '0':
                    ratio = '—'
                else:
                    ratio = '—'
                vd = verdict(expect, m, s['mean'],
                             0.0 if s['max'] == 0.0 and s['min'] == 0.0 else None)
            else:
                simtxt, ratio, vd = '산출 불가', '—', '—'
            mtxt = f'{m:+.1f}' if m is not None else '수치 없음'
            if k in sim:
                s = sim[k]
                citxt = f'[{s["min"]:+.1f}, {s["max"]:+.1f}]'
                if not s.get('sign_stable'):
                    citxt += ' **0 포함**'
                    if vd in ('부호 일치', '순위 일치'):
                        vd = f'{vd} *(구간이 0을 지나 단정 못 함)*'
            else:
                citxt = '—'
            L.append(f'| {k} | {desc} | `{expect}` | {mtxt} | {simtxt} | {citxt} | {vd} | {ratio} |')
        L += ['']
        miss = [k for k in ids if k not in sim]
        if miss:
            L += ['**왜 낼 수 없나**', '']
            for k in miss:
                L.append(f'- **{k}** — {CANNOT[k]}')
            L += ['']

    stable = [k for k in MEASURED if k in sim and sim[k].get('sign_stable')]
    L += ['---', '', '## 이 표를 읽는 규칙', '',
          f'- **부호를 말할 수 있는 지표는 {len(stable)}개뿐이다** — {", ".join(stable) if stable else "없다"}.',
          f'  값을 낸 {scored}개 중 나머지는 95% 구간이 0을 지난다. 네 런 384칸을 합쳐도 그렇다',
          '- **배율은 참고값이다.** 하루 대 한 달, 낱개 단가 대 카드매출이라 같은 척도가 아니다',
          '- **방어선 지표(`0`)는 동등성 검정이 필요하다.** 신뢰구간이 0을 포함하는 것으로는 부족하고,',
          '  미리 정한 띠 **안에** 들어와야 한다. 아직 그 띠를 정하지 않았다',
          '',
          '---', '',
          '## 다음에 무엇을 고칠 것인가 — 원인별로', '',
          '### ① 관측 창이 하루다 → 월 단위로 (지표 4개가 풀린다)', '',
          'P012-3 문턱 도달 · P012-4 1인 평균 캐시백 · P012-6 한도 도달 · EM-2 지급 전후 순효과.',
          '넷 다 **정책의 회계 기간이 달(月)** 이라서 못 내는 것이지 잡음 때문이 아니다.',
          '캐시백은 월 적립이 기전 자체이므로, 창을 늘리지 않으면 이 정책은 원리상 채점되지 않는다.', '',
          '> 비용 가늠: 하루 96칸이 계획 28분이었다. 30일이면 같은 12명으로도 2,880칸이다.',
          '> 한 달을 통째로 돌리는 대신 **정책 회계에 필요한 누적만 잇는 설계**가 필요하다.', '',
          '### ② 카탈로그에 업종 계층이 없다 → 품목을 넓힌다 (지표 3개)', '',
          'P012-5 가전·가구 vs 이·미용 · EM-4 준내구재 vs 대면서비스 · DS-6 상권 유형.',
          '지금 카탈로그는 17개 낱개 품목이고 최고가가 커트 20,000원이다.',
          '실측이 쓰는 업종 구분이 아예 없어서 순위 지표를 낼 수 없다.', '',
          '### ③ 부호가 반대인 셋 — 여기가 프롬프트의 표적이다', '',
          '| 지표 | 실측 | 시뮬 | 무슨 뜻인가 |',
          '|---|---:|---:|---|',
          '| DS-1 | −14.1 | +43.3 | **거리두기를 켰는데 음식점 지출이 늘었다** |',
          '| DS-3 | +18.3 | −23.5 | 소매 > 음식점 순위가 뒤집혔다 |',
          '| LV-2 | (+) | −3.6 | 상품권을 줬는데 **동네 소비 비중이 줄었다** |',
          '',
          'DS-1 이 가장 크다. 거리두기 on 팔에서 식당 21시 제한·카페 매장 금지가 걸리는데',
          '음식점 지출이 **늘어난다.** 제약을 제약으로 읽지 않는다는 뜻이고,',
          '[TREATMENT_BINDS.md](TREATMENT_BINDS.md) 가 찾은 것과 같은 자리다 —',
          '이 12명은 카페·외식을 거의 고르지 않아서, 켜든 끄든 몇 건의 우연이 부호를 정한다.', '',
          '### ④ 런 사이 폭이 부호를 뒤집는다 → 지표 산출에 반복을 넣는다', '',
          '같은 프롬프트 v25 인데 네 런의 범위가 이렇다.', '',
          '```',
          'P012-1  -64.3 ~ +280.0      EM-3   +13.9 ~ +262.5',
          'DS-2    -41.5 ~  +83.3      LV-1   -33.3 ~ +148.6',
          '```',
          '',
          '**부호가 런마다 뒤집힌다.** 위 표의 "부호 일치" 셋도 평균이 그랬다는 것이지',
          '안정적으로 그렇다는 뜻이 아니다. 지표를 한 런에서 뽑는 지금 방식을 바꿔',
          '**반복을 지표 정의 안에 넣어야 한다.**', '',
          '### 순서', '',
          '1. **④ 먼저** — 폭을 줄이지 않으면 ③의 개선이 보이지 않는다',
          '2. **③** — 부호가 반대인 셋을 겨냥한 프롬프트/입력 수정',
          '3. **① ②** — 창과 카탈로그는 파이프라인 작업이고 비용이 크다. ③이 잡힌 뒤에',
          '']
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
    print('wrote', args.out, f'({scored}/{total} 지표에 시뮬 값)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
