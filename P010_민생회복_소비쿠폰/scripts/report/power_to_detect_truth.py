"""지금 표본으로 정답지 값을 **검출할 수 있기는 한지** 지표마다 계산한다.

거리두기는 프롬프트를 한 글자도 고치지 않고 n 을 200 → 500 으로 올린 것만으로
부호 적중이 1/4 에서 3/4 이 됐다. 표본이 모자라면 어떤 프롬프트를 써도 수렴을
보여줄 수 없다는 뜻이다. 그래서 프롬프트를 더 만들기 전에 이것부터 잰다.

    python scripts/report/power_to_detect_truth.py

계산은 이렇게 한다.

    채점표의 95% 구간 [lo, hi] → 표준오차 se = (hi - lo) / 3.92
                                 표준편차 sd = se * sqrt(n)
    정답지 효과의 금액 d = 실측% * base / 100
    필요한 n = ((1.96 + 0.84) * sd / d)^2          (양측 5%, 검정력 80%)

**이것은 정답지 값을 '맞히는' 힘이 아니라 '0 과 구별하는' 힘이다.** 부호 채점이
요구하는 것이 그것이다.
"""
from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORING = ROOT / 'data/experiments/scoring_table.json'

POLICIES = [
    ('상생소비 P012', 'P012'),
    ('긴급재난 P013', 'EMERGENCY_2020'),
    ('지역상품권 P014', 'LOCAL_VOUCHER'),
    ('거리두기', 'DISTANCING_2020'),
]

DASH = str.maketrans({'−': '-', '–': '-', '—': '-'})
Z = 1.959964 + 0.841621          # 양측 5% + 검정력 80%


def truth_pct(desc, expect):
    if expect == 'rank':
        return None
    m = re.search(r'\(실측\s*([^)]*)\)', str(desc or '').translate(DASH))
    if not m:
        return None
    body = m.group(1)
    if '원' in body and '%' not in body:
        return None
    m2 = re.search(r'([+-]?\d+(?:\.\d+)?)\s*%', body)
    return float(m2.group(1)) if m2 else None


def main():
    sc = json.loads(SCORING.read_text(encoding='utf-8'))
    # 지표마다 단위가 다르다 — 금액(원)과 몫(0~1)이 섞여 있다. 같은 자릿수로
    # 찍으면 몫이 0 으로 뭉개져 '효과가 없다'처럼 보인다. 단위를 같이 적는다.
    print('%-14s %-8s %7s %6s %12s %12s %9s  %s'
          % ('정책', '지표', '실측%', 'n', '관측 sd', '필요 효과', '필요 n', '판정'))
    print('-' * 100)
    rows = []
    for name, key in POLICIES:
        blk = sc.get(key) or {}
        inds = {i['id']: i for i in (blk.get('indicators') or []) if i.get('id')}
        # 가장 최근(가장 큰 n)의 결과 블록을 쓴다
        best = None
        for rn, rb in blk.items():
            if not (rn.startswith('result') and isinstance(rb, dict)):
                continue
            n = max([v.get('n', 0) for v in rb.values()
                     if isinstance(v, dict) and isinstance(v.get('n'), int)] or [0])
            if best is None or n > best[0]:
                best = (n, rn, rb)
        if not best:
            continue
        _, rn, rb = best
        for iid, i in inds.items():
            v = rb.get(iid)
            if not isinstance(v, dict):
                continue
            ci, n, base = v.get('ci'), v.get('n'), v.get('base')
            t = truth_pct(i.get('desc'), i.get('expect'))
            if not (isinstance(ci, list) and len(ci) == 2 and isinstance(n, int) and n > 1):
                continue
            se = (ci[1] - ci[0]) / 3.919928
            sd = se * math.sqrt(n)
            if t is None or not isinstance(base, (int, float)) or not base:
                # 정답지 금액을 모르면, 지금 관측된 효과를 0 과 가르는 힘만 본다
                d = abs(v.get('mean') or 0)
                tag = '(관측 효과 기준)'
            else:
                d = abs(t * base / 100.0)
                tag = ''
            if d <= 0:
                continue
            need = (Z * sd / d) ** 2
            verdict = '지금 n 으로 충분' if need <= n else '**%.1f배 필요**' % (need / n)
            rows.append((name, iid, t, n, sd, d, need, verdict, rn, tag))
            share = isinstance(base, (int, float)) and 0 < abs(base) <= 1.0
            fmt = ((lambda x: '%.4f(몫)' % x) if share else (lambda x: format(x, ',.0f') + '원'))
            print('%-14s %-8s %7s %6d %12s %12s %9.0f  %s %s'
                  % (name, iid, ('%+.1f' % t) if t is not None else '—',
                     n, fmt(sd), fmt(d), need, verdict, tag))
    print()
    short = [r for r in rows if r[6] > r[3]]
    print('정답지 값을 지금 표본으로 검출할 수 없는 지표: %d / %d' % (len(short), len(rows)))
    if short:
        worst = max(short, key=lambda r: r[6] / r[3])
        print('  가장 모자란 것: %s %s — 지금의 %.0f배(n=%.0f) 가 필요하다'
              % (worst[0], worst[1], worst[6] / worst[3], worst[6]))
    print()
    print('주의 — 이 계산이 말하지 않는 것')
    print('  · 정답지 값을 그대로 맞히는 힘이 아니라 **0 과 구별하는** 힘이다')
    print('  · 구간은 붓스트랩이고 정규 근사로 sd 를 되돌린 값이라 어림이다')
    print('  · 원문 창과 우리 창이 달라 실측 %를 우리 base 에 그대로 곱한 것도 어림이다')
    print('  · 그래도 **몇 배가 모자란지**는 자릿수로 읽을 수 있다')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
