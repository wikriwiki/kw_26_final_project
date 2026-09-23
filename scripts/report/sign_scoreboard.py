"""**현행 프롬프트 v5 가 정답지에 얼마나 수렴했는가** — 전 정책 한 장.

    python scripts/report/sign_scoreboard.py

## 왜 부호인가

크기 수렴은 지금 어느 정책에서도 잴 수 없다. 원문 창이 주 단위인데 우리 창은
이틀이고, 대상·분모·대조군이 호환되지 않는다(`audit_2026_09_20`). 지표 25개 중
백분율이 붙은 것이 절반이 안 되고, 붙은 것도 우리 base 에 그대로 곱할 수 없다.

**그래서 채점은 부호로 한다.** 이 표가 그 채점의 전부이며, "정답에 수렴한다" 는
말이 지금 뜻할 수 있는 것의 전부다.

## 위약을 같이 센다

진짜 정책만 세면 **"다 오른다"고 답하는 프롬프트가 만점을 받는다.** 위약 둘을
같은 표에 올리는데, **둘은 서로 다른 것을 본다.**

    PLACEBO_FAKE     실재하지 않는 정책. 기전에 맞게 **반응해야** 적중이다(PL-1 +).
                     무반응이면 모델이 기전을 처리한 것이 아니라 실제 정책의
                     결과를 외운 것이다 — lookahead bias 검정.
                     다만 대상이 아닌 업종은 무반응이어야 한다(PL-2 0).

    PLACEBO_TIMING   정책이 없는 자리. **아무 일도 없어야** 적중이다(전부 0).
                     효과가 나오면 워밍업 추세·요일 교란 같은 설계 결함이다.

그래서 "위약은 무반응이 정답" 이라고 뭉뚱그리지 않는다. 지표마다 등록된
기댓값을 그대로 읽는다.

## 어느 런을 읽는가

정책마다 런이 여럿이다. **어느 것을 v5 의 현재 읽기로 삼는지 여기 적어 둔다** —
고르는 규칙을 코드에 숨기면 나중에 유리한 런을 고르게 된다.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORING = ROOT / 'data/experiments/scoring_table.json'

# (채점표 키, 표시 이름, 읽을 결과 블록, 고른 이유)
READINGS = [
    ('EMERGENCY_2020', '긴급재난 P013', 'result_stage3_dow_2026_09_16',
     'v5 · n=200 · 유일한 런'),
    ('P012', '상생소비 P012', 'result_r2_v5',
     'v5 · n=500 · 가장 큰 표본. result_FINAL 은 n=200'),
    ('LOCAL_VOUCHER', '지역상품권 P014', 'result_stage6_district_n500_2026_09_18',
     'v5 · n=495 · 자치구 조건이 판정에 들어간 뒤의 런'),
    ('DISTANCING_2020', '거리두기', 'result_r3_v5',
     'v5 · n=200 · 가장 최근. v50 의 v5 와 같은 창'),
    ('SECTOR_VOUCHER_2020', '8대쿠폰 P015', 'result_policy_2026_09_17_hits',
     'v5 · n=200 · 홀드아웃 런의 등록된 판정을 옮겨 적은 블록(원문은 사람이 읽는 형식)'),
]

# 위약 둘 — 기댓값이 서로 다르다. 머리말 참조.
PLACEBOS = [
    ('PLACEBO_FAKE', '위약(가짜 정책)', 'result_2026_09_16_fixed_plumbing',
     '배관을 고친 뒤의 런. 그 전 런은 표시 결함으로 대상 업종이 -10.5% 였다'),
    ('PLACEBO_TIMING', '위약(시점 엇갈림)', 'result_FINAL',
     '확정 런'),
]

# **다른 자로 잰 것은 세지 않는다.** (채점표 키, 결과 블록, 지표) → 이유.
# 지우지 않고 남겨 두는 이유: 왜 빠졌는지 표에서 바로 보여야 한다.
SUSPECT = {
    ('EMERGENCY_2020', 'result_stage3_dow_2026_09_16', 'EM-2'):
        '적격 판정이 상생 기준이었다 — 긴급재난의 사용가능업종이 아니다',
}

EXPECT_TXT = {'+': '증가', '-': '감소', '0': '무반응', 'rank': '순위', 'info': '참고'}


def read(sc, key, block):
    return ((sc.get(key) or {}).get(block) or {})


def line(iid, expect, desc, v, suspect=None):
    """한 지표 한 줄. `hit` 가 없으면 못 잰 것이다 — 빈칸으로 두고 세지 않는다.

    `suspect` 가 있으면 **다른 자로 잰 것**이라 적중/빗나감을 세지 않는다.
    지우지 않고 물음표로 남긴다 — 왜 빠졌는지 표에서 바로 보여야 한다.
    """
    if not isinstance(v, dict):
        return None, '%-7s %-5s %-46s %s' % (iid, EXPECT_TXT.get(expect, expect),
                                             desc[:46], '— 이 런에 없다')
    hit = v.get('hit')
    got = v.get('note') or v.get('got') or ''
    if isinstance(v.get('pct'), (int, float)):
        got = '%+.1f%%' % v['pct'] + (('  ' + str(got)) if got else '')
    if suspect:
        return None, '%-7s %-5s %-46s %s  %s   <- **못 셈: %s**' % (
            iid, EXPECT_TXT.get(expect, expect), desc[:46], '?', str(got)[:24], suspect)
    mark = {True: 'O', False: 'X'}.get(hit, '·')
    return hit, '%-7s %-5s %-46s %s  %s' % (iid, EXPECT_TXT.get(expect, expect),
                                            desc[:46], mark, str(got)[:40])


def section(sc, title, rows, tally):
    print(title)
    print('-' * 104)
    for key, name, block, why in rows:
        blk = sc.get(key) or {}
        v = read(sc, key, block)
        print('  %s   [%s · %s]' % (name, block, why))
        if not v:
            print('    (그 블록이 채점표에 없다)')
            continue
        for ind in (blk.get('indicators') or []):
            expect = ind.get('expect')
            if expect == 'info':
                continue
            hit, txt = line(ind['id'], expect, ind.get('desc') or '',
                            v.get(ind['id']), SUSPECT.get((key, block, ind['id'])))
            print('    ' + txt)
            if hit is True:
                tally['hit'] += 1
            elif hit is False:
                tally['miss'] += 1
            else:
                tally['none'] += 1
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    sc = json.loads(io.open(SCORING, encoding='utf-8').read())

    print('# 현행 프롬프트 v5 의 부호 적중 — 전 정책')
    print()
    print('크기 수렴은 지금 어느 정책에서도 잴 수 없다. 채점은 부호로 한다.')
    print()

    real = {'hit': 0, 'miss': 0, 'none': 0}
    fake = {'hit': 0, 'miss': 0, 'none': 0}
    section(sc, '## 실제 정책 — 정답지가 말하는 방향을 맞혔는가', READINGS, real)
    section(sc, '## 위약 — 지표마다 등록된 기댓값이 다르다 (가짜정책은 반응, 시점엇갈림은 무반응)',
            PLACEBOS, fake)

    print('## 합계')
    for nm, t in (('실제 정책', real), ('위약', fake)):
        d = t['hit'] + t['miss']
        pct = ('%.0f%%' % (100 * t['hit'] / d)) if d else '—'
        print('  %-8s 적중 %2d · 빗나감 %2d · 못 잼 %2d   →  %s'
              % (nm, t['hit'], t['miss'], t['none'], pct))
    d = real['hit'] + real['miss'] + fake['hit'] + fake['miss']
    tot = real['hit'] + fake['hit']
    print('  %-8s 적중 %2d / %2d   →  %s'
          % ('전체', tot, d, ('%.0f%%' % (100 * tot / d)) if d else '—'))
    print()
    print('읽는 법')
    print('  · 못 잰 지표(·)는 분모에서 뺀다. 관측이 모자라거나 그 런에 없던 지표다.')
    print('  · 런이 서로 다른 날·다른 n 이다. 정책 간 적중률을 **서로 비교하지 않는다.**')
    print('  · 위약을 빼고 세면 "다 오른다" 고 답하는 프롬프트가 만점을 받는다.')
    print('  · 가짜정책 PL-1 은 **반응해야** 적중이다(lookahead 검정). 무반응이 정답이 아니다.')
    print('  · 이 표는 v5 만 읽는다. 후보 비교는 각 라운드의 사전등록이 한다.')
    if SUSPECT:
        print()
        print('못 센 것 — 다른 자로 쟀다')
        for (k, b, i), why in SUSPECT.items():
            print('  %-20s %-34s %-7s %s' % (k, b, i, why))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
