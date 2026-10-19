"""원문 대조 가능한 지표와 위약의 등록판정 실패를 유형별로 가른다.

    python scripts/report/why_it_misses.py

부호 적중표는 O/X 만 준다. 그런데 X 에는 서로 다른 세 가지가 섞여 있고,
**고치는 방법이 전혀 다르다.**

    ① 부호가 반대다            점추정이 기대와 반대쪽이다
                              → 정책 전달·측정·교란·프롬프트를 순서대로 점검
    ② 부호는 맞는데 유의하지 않다  점추정은 기대 방향인데 구간이 0 을 지난다
                              → 표본 크기·효과 크기·측정 잡음을 구분해야 한다
    ③ 동등성 지표가 밴드를 넘었다  기대가 '무반응'인데 구간이 밴드 밖이다
                              → 밴드 대비 얼마나 넘었는지를 같이 본다

현재 외부 방향 대응 감사가 통과한 정책 지표는 없다. 따라서 위약의 실패 유형만
남으며, 이것을 정책 프롬프트의 외부 실측 성능으로 일반화하지 않는다.

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

    print('# v5 외부 방향 감사 통과 지표와 위약의 실패 유형')
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
    print('  점추정 부호가 반대                             %d개' % n_sign)
    print('  부호는 맞는데 유의하지 않다                     %d개' % n_power)
    print('  폭의 문제(동등성 밴드 초과)                      %d개' % tally.get('밴드 초과', 0))
    print('  적중                                          %d개' % tally.get('적중', 0))
    print()
    print('  원문 방향 대응 감사 전의 정책 지표는 제외했다. 위약 실패만으로 프롬프트를')
    print('  바꿀지 결정하지 않는다. 먼저 정책 전달·측정·교란과 표본 오차를 점검한다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
