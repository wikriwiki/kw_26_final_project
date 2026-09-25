"""v5 가 빗나간 지표를 **빗나간 이유별로** 가른다.

    python scripts/report/why_it_misses.py

부호 적중표는 O/X 만 준다. 그런데 X 에는 서로 다른 세 가지가 섞여 있고,
**고치는 방법이 전혀 다르다.**

    ① 부호가 반대다            점추정이 기대와 반대쪽이다
                              → **프롬프트가 고칠 수 있는 유일한 자리**
    ② 부호는 맞는데 유의하지 않다  점추정은 기대 방향인데 구간이 0 을 지난다
                              → 표본의 문제다. 프롬프트를 고쳐도 안 좁혀진다
    ③ 동등성 지표가 밴드를 넘었다  기대가 '무반응'인데 구간이 밴드 밖이다
                              → 밴드 대비 얼마나 넘었는지를 같이 본다

거리두기는 프롬프트를 한 글자도 고치지 않고 n 을 200 → 500 으로 올린 것만으로
부호 적중이 1/4 → 3/4 이 됐다. ②가 많다면 프롬프트를 손대는 것은 헛일이다.

## 동등성 지표를 부호로 읽지 않는다

`expect: "0"` 은 "구간이 기준선 ±10% 밴드 **안에** 있어야 적중" 이다. 이것을
①②로 가르면 뜻이 없다 — 방향이 아니라 폭의 문제이기 때문이다. 따로 센다.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORING = ROOT / 'data/experiments/scoring_table.json'

NULL_BAND = 0.10          # score_policy 와 같은 값. 바뀌면 여기도 바꿔야 한다


def load_readings():
    """부호 적중표와 **같은 런**을 읽는다 — 두 표가 다른 런을 보면 안 된다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'sb', ROOT / 'scripts/report/sign_scoreboard.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.READINGS, m.PLACEBOS, m.exclusion_reason


def classify(expect, v):
    """(분류, 한 줄 설명). `v` 는 채점표의 지표 블록."""
    mean, ci, base = v.get('mean'), v.get('ci'), v.get('base')
    if not (isinstance(ci, list) and len(ci) == 2):
        return '자료부족', '구간이 없다'
    lo, hi = ci
    if expect == '0':
        band = NULL_BAND * abs(base) if isinstance(base, (int, float)) and base else None
        if band is None:
            return '자료부족', 'base 가 없어 밴드를 못 만든다'
        over = max(band - abs(lo) if lo < -band else 0.0,
                   abs(hi) - band if hi > band else 0.0)
        worst = max(abs(lo), abs(hi))
        return ('밴드 초과',
                '밴드 ±%s · 구간 끝 %s — %.0f%% 초과'
                % (_f(band), _f(worst), 100 * (worst / band - 1)))
    want_up = (expect == '+')
    if not isinstance(mean, (int, float)):
        return '자료부족', '점추정이 없다'
    right_sign = (mean > 0) if want_up else (mean < 0)
    crosses = (lo <= 0 <= hi)
    if right_sign and crosses:
        # 얼마나 모자란지 — 구간 반폭 대비 점추정
        half = (hi - lo) / 2.0
        return ('유의하지 않다',
                '점추정 %s 는 기대 방향 · 구간 [%s, %s] 이 0 을 지난다 (|평균|/반폭 %.2f)'
                % (_f(mean), _f(lo), _f(hi), abs(mean) / half if half else 0))
    if right_sign:
        return '적중', '기대 방향이고 구간이 0 을 안 지난다'
    return ('부호가 반대', '점추정 %s — 기대는 %s' % (_f(mean), '증가' if want_up else '감소'))


def _f(x):
    if not isinstance(x, (int, float)):
        return str(x)
    return ('%.4f' % x) if abs(x) <= 1.5 else format(x, ',.0f')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    sc = json.loads(io.open(SCORING, encoding='utf-8').read())
    readings, placebos, exclusion_reason = load_readings()

    print('# v5 가 빗나간 이유 — 프롬프트가 고칠 수 있는 자리는 어디인가')
    print()
    tally = {}
    rows = []
    for key, name, block, _why in list(readings) + list(placebos):
        blk = sc.get(key) or {}
        res = (blk.get(block) or {})
        for ind in (blk.get('indicators') or []):
            expect, iid = ind.get('expect'), ind['id']
            if expect in ('info', 'rank'):
                continue
            if exclusion_reason(key, block, ind):
                continue
            v = res.get(iid)
            if not isinstance(v, dict):
                continue
            if v.get('hit') is True:
                kind, why = '적중', ''
            else:
                kind, why = classify(expect, v)
            tally[kind] = tally.get(kind, 0) + 1
            rows.append((name, iid, expect, kind, why, ind.get('desc') or ''))

    for want in ('부호가 반대', '유의하지 않다', '밴드 초과', '자료부족', '적중'):
        got = [r for r in rows if r[3] == want]
        if not got:
            continue
        print('## %s — %d개' % (want, len(got)))
        for nm, iid, expect, _k, why, desc in got:
            print('  %-14s %-8s %-4s %s' % (nm, iid, expect, desc[:40]))
            if why:
                print('  %-14s %-8s      %s' % ('', '', why))
        print()

    print('## 읽는 법')
    n_sign = tally.get('부호가 반대', 0)
    n_power = tally.get('유의하지 않다', 0)
    print('  프롬프트가 고칠 수 있는 자리(부호가 반대)          %d개' % n_sign)
    print('  표본이 고칠 자리(부호는 맞는데 유의하지 않다)      %d개' % n_power)
    print('  폭의 문제(동등성 밴드 초과)                      %d개' % tally.get('밴드 초과', 0))
    print('  적중                                          %d개' % tally.get('적중', 0))
    print()
    print('  거리두기는 프롬프트를 한 글자도 고치지 않고 n 을 200 → 500 으로 올린 것만으로')
    print('  부호 적중이 1/4 → 3/4 이 됐다. "유의하지 않다" 가 많으면 프롬프트가 아니라')
    print('  표본이 병목이다 — 그 자리에서 후보를 가르면 잡음을 성질로 적는다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
