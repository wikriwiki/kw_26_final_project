"""후보들을 비율이 아니라 짝지은 칸으로 비교한다 — 이 파일럿의 잡음이 그만큼 크다.

같은 프롬프트(v45)를 같은 시민·날짜·seed 로 두 번 돌렸더니 192칸 중 **38칸(19.8%)의
판정이 뒤집혔다**. 통과율로는 89.1% 와 85.9% 다. seed 를 고정해도 배치 구성이 런마다
달라 샘플러의 실제 흐름이 달라진다 — 아키텍처 문서가 "seed 지원도 비트 단위 결정성을
보장한다고 주장하지 않는다"고 적어 둔 그대로다.

그래서 통과율 차이를 그대로 읽으면 안 된다. 칸은 후보들 사이에서 짝지어져 있으므로
(같은 시민·시나리오·팔·복제) McNemar 정확검정을 쓴다.

    개선   그 후보에서 살아난 칸
    악화   그 후보에서 죽은 칸
    p      두 방향이 같은 확률에서 나왔을 확률 (이항 정확검정, 양측)

그리고 **"유의하지 않다"와 "검출력 부족"을 갈라 적는다.** 관측 불일치율이 약 30% 이므로
최소검출차는 대략 2·sqrt(불일치 칸 수)다.

    칸 수   192 → 15칸(7.9%p)   384 → 21칸(5.6%p)   1152 → 37칸(3.2%p)

+4칸을 읽으려면 2,700칸이 필요하다. 그런 차이는 **못 읽는 것이지 없는 것이 아니다.**

    python scripts/report/paired_contract_gate.py --responses run/responses.jsonl \
        --compare v5:v47 --compare v45:v47
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

DISCORDANCE = 0.30   # 관측값. 후보 쌍의 불일치 칸 비율.


def load(path):
    """{후보: {(시민, 시나리오, 팔, 복제): 통과 여부}}"""
    out = collections.defaultdict(dict)
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r['aid'], r['case'], r['arm'], r['replicate'])
            out[r['variant']][key] = bool(r.get('valid'))
    return dict(out)


def mcnemar(a, b):
    """(개선, 악화, p). 짝지은 칸만 센다."""
    keys = set(a) & set(b)
    improved = sum(1 for k in keys if not a[k] and b[k])
    worsened = sum(1 for k in keys if a[k] and not b[k])
    n = improved + worsened
    if n == 0:
        return improved, worsened, 1.0
    lo = min(improved, worsened)
    p = 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return improved, worsened, min(1.0, p)


def mde(cells, discordance=DISCORDANCE):
    """최소검출차, 칸 단위. 대략 2·sqrt(불일치 칸 수)."""
    return 2.0 * math.sqrt(max(1.0, discordance * cells))


def verdict(improved, worsened, p, cells, alpha=0.05):
    """유의 / 검출력 부족 / 유의하지 않다 — 셋을 갈라 적는다."""
    net = improved - worsened
    if p < alpha:
        return '유의'
    if abs(net) < mde(cells):
        return '검출력 부족'
    return '유의하지 않다'


def compare(data, pairs, alpha=0.05):
    rows = []
    for x, y in pairs:
        if x not in data or y not in data:
            rows.append({'pair': (x, y), 'missing': True})
            continue
        cells = len(set(data[x]) & set(data[y]))
        i, d, p = mcnemar(data[x], data[y])
        rows.append({'pair': (x, y), 'missing': False, 'improved': i, 'worsened': d,
                     'net': i - d, 'p': p, 'cells': cells, 'mde': mde(cells),
                     'verdict': verdict(i, d, p, cells, alpha)})
    return rows


def rates(data):
    return {k: (sum(v.values()), len(v)) for k, v in sorted(data.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--responses', required=True)
    ap.add_argument('--compare', action='append', required=True, metavar='A:B')
    ap.add_argument('--gate', type=float, default=0.95)
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()
    data = load(args.responses)
    if not data:
        raise SystemExit('응답이 없다: %s' % args.responses)

    print('%-6s %12s %9s' % ('후보', '엄격 통과', '통과율'))
    for name, (ok, tot) in rates(data).items():
        print('%-6s %12s %8.1f%%' % (name, '%d/%d' % (ok, tot), 100 * ok / tot))
    print()
    print('%-14s %8s %8s %8s %11s  %s' % ('비교', '개선', '악화', '순증', 'p', '판정'))
    pairs = [tuple(c.split(':', 1)) for c in args.compare]
    for r in compare(data, pairs, args.alpha):
        if r['missing']:
            print('%-14s  (후보 없음)' % ('%s → %s' % r['pair']))
            continue
        print('%-14s %8d %8d %+8d %11.4f  %s'
              % ('%s → %s' % r['pair'], r['improved'], r['worsened'], r['net'],
                 r['p'], r['verdict']))
    any_cells = len(next(iter(data.values())))
    print()
    print('칸 %d 기준 최소검출차 약 %.0f칸 (%.1f%%p) — 그보다 작은 차이는 못 읽는다'
          % (any_cells, mde(any_cells), 100 * mde(any_cells) / any_cells))
    print()
    passed = [n for n, (ok, tot) in rates(data).items() if ok / tot >= args.gate]
    print('등록된 관문 %.0f%%: %s' % (100 * args.gate, ', '.join(passed) if passed else '전부 미달'))
    if not passed:
        print('  → 전부 미달이면 어느 후보가 더 나아도 "통과했다"고 적지 않는다')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
