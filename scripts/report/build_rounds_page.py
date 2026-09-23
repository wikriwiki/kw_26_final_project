"""후보 대결 — **어떤 프롬프트로 돌렸고 결과가 어땠는가**를 라운드별로 나란히.

    python scripts/report/build_rounds_page.py

`convergence.html` 은 현행 v5 하나만 그린다. 그러면 "그래서 후보는 어땠는가"
에 답이 없다. 라운드마다 두 팔을 **같은 축**에 놓고 실측과 함께 본다.

## 읽는 규칙을 그림에 박는다

숫자만 나란히 놓으면 "0.9%p 가까워졌으니 이겼다" 로 읽게 된다. 이 저장소가
세 번 데인 자리다. 그래서 칸마다 **런 간 이동**을 함께 적고, 이동이 그보다
작으면 회색으로 죽인다.

    DS-1     런 간 이동 0.2·0.5%p   두 번 쟀고 둘 다 작다 → 읽는다
    DS-2     13.1·2.7%p            한 수로 못 적는다 → **읽지 않는다**
    P012-1   8.2%p                 n 이 함께 바뀌어 섞여 있다 → 읽지 않는다

## 등록된 판정을 같이 싣는다

각 라운드의 사전등록이 무엇을 합격선으로 걸었고 결과가 무엇이었는지를 그림
옆에 적는다. 수치만 남기면 나중에 "그때 무엇을 약속했는지" 가 사라진다.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORING = ROOT / 'data/experiments/scoring_table.json'
TPL = ROOT / 'output/report/rounds_tpl.html'
OUT = ROOT / 'output/report/rounds.html'

DOM = 30.0
TRUTH = {'DS-1': -14.1, 'DS-2': 4.2, 'P012-1': 20.82, 'P012-2': 2.85, 'P012-6': 21.0}

# 같은 프롬프트를 두 번 돌렸을 때의 이동. **잰 것만 적는다.**
RUN_SHIFT = {'DS-1': '0.2·0.5%p', 'DS-2': '13.1·2.7%p', 'P012-1': '8.2%p*'}
READABLE = {'DS-1': True, 'DS-2': False, 'P012-1': False}

ROUNDS = [
    {'name': 'v50 · 거리두기', 'key': 'DISTANCING_2020',
     'a': ('v5', 'result_v50_v5'), 'b': ('v45', 'result_v50_v45'),
     'reg': 'DS-1 과 DS-2 가 **둘 다** v45 쪽이면 v45. 아니면 v5 유지.',
     'got': '갈렸다 — DS-1 은 v45(이동 0.9%p > 런 0.5%p), DS-2 는 v5. **v5 유지.**'},
    {'name': '라운드2 · P012', 'key': 'P012',
     'a': ('v5', 'result_r2_v5'), 'b': ('v45', 'result_r2_v45'),
     'reg': 'P012-1 에서 실측(+20.82%)에 더 가까운 쪽.',
     'got': 'v5 가 가깝다(|차| 18.9 대 20.2). 그러나 그 차이 1.3%p 가 같은 '
            '프롬프트의 런 이동 8.2%p 안이다. **v5 유지 — 증거는 약하다.**'},
    {'name': '라운드3 · 거리두기', 'key': 'DISTANCING_2020',
     'a': ('v5', 'result_r3_v5'), 'b': ('v51', 'result_r3_v51'),
     'reg': 'DS-1 이 0.5%p 넘게 가까워지고 **DS-3 순위가 유지**되면 v51.',
     'got': 'DS-1 은 0.3%p 멀어졌고 DS-3 순위를 잃었다. **두 조건 다 불충족 — v5 유지.** '
            'v51 은 소매·카페까지 전부 아래로 밀었다(소매 -15.1%) — DS-3 이 잡으려고 '
            '등록해 둔 실패 모양 그 자체다.'},
]


def _x(v):
    return 50.0 + max(-DOM, min(DOM, v)) / DOM * 50.0


def _pct(r):
    if isinstance(r, dict) and isinstance(r.get('pct'), (int, float)):
        return r['pct']
    return None


def rows_for(sc, rnd):
    blk = sc.get(rnd['key']) or {}
    A, B = blk.get(rnd['a'][1]) or {}, blk.get(rnd['b'][1]) or {}
    out = []
    for ind in (blk.get('indicators') or []):
        if ind.get('expect') == 'info':
            continue
        iid = ind['id']
        a, b = A.get(iid), B.get(iid)
        if not isinstance(a, dict) and not isinstance(b, dict):
            continue
        out.append({
            'id': iid, 'expect': ind.get('expect'),
            'desc': (ind.get('desc') or '')[:64],
            'a': _pct(a), 'b': _pct(b),
            'a_hit': (a or {}).get('hit') if isinstance(a, dict) else None,
            'b_hit': (b or {}).get('hit') if isinstance(b, dict) else None,
            'truth': TRUTH.get(iid),
            'shift': RUN_SHIFT.get(iid), 'readable': READABLE.get(iid),
        })
    return out


def _bar(r):
    p = ['<span class="zero"></span>']
    for who, v in (('a', r['a']), ('b', r['b'])):
        if v is not None:
            p.append('<span class="mk %s" style="left:%.2f%%"></span>' % (who, _x(v)))
    if r['truth'] is not None:
        p.append('<span class="mk tru" style="left:%.2f%%"></span>' % _x(r['truth']))
    return ''.join(p)


def _nums(r, na, nb):
    out = []
    if r['truth'] is not None:
        out.append('<span class="tru">실측 %+.1f</span>' % r['truth'])
    for who, nm, v, hit in (('a', na, r['a'], r['a_hit']), ('b', nb, r['b'], r['b_hit'])):
        if v is not None:
            out.append('<span class="%s">%s %+.1f</span>' % (who, nm, v))
        elif hit is not None:
            out.append('<span class="%s">%s 순위 %s</span>'
                       % (who, nm, '적중' if hit else '빗나감'))
    if r['a'] is not None and r['b'] is not None:
        out.append('<span class="memo">두 팔 차 %.1f%%p</span>' % abs(r['b'] - r['a']))
    if r['shift']:
        cls = 'shift' if r['readable'] else 'shift dead'
        out.append('<span class="%s">런 이동 %s%s</span>'
                   % (cls, r['shift'], '' if r['readable'] else ' — 읽지 않는다'))
    return ''.join(out)


def render(sc):
    out = []
    for rnd in ROUNDS:
        na, nb = rnd['a'][0], rnd['b'][0]
        out.append('  <section class="rnd"><h3>%s <em>%s 대 %s</em></h3>' % (rnd['name'], na, nb))
        out.append('    <p class="reg"><b>등록한 합격선</b> %s</p>' % rnd['reg'])
        out.append('    <p class="got"><b>결과</b> %s</p><div class="rows">' % rnd['got'])
        for r in rows_for(sc, rnd):
            dead = '' if (r['readable'] is not False) else ' dead'
            out.append(
                '    <div class="row%s"><div class="meta"><span class="id">%s</span>'
                '<span class="desc">%s</span></div>'
                '<div class="track">%s</div><div class="nums">%s</div></div>'
                % (dead, r['id'], r['desc'], _bar(r), _nums(r, na, nb)))
        out.append('  </div></section>')
    tpl = io.open(TPL, encoding='utf-8').read()
    return tpl.replace('<!--BODY-->', chr(10).join(out)).replace('<!--DOM-->', str(int(DOM)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()
    sc = json.loads(io.open(SCORING, encoding='utf-8').read())
    html = render(sc)
    io.open(a.out, 'w', encoding='utf-8', newline='\n').write(html)
    print('%s · %d bytes · 라운드 %d' % (a.out, len(html), len(ROUNDS)))
    for rnd in ROUNDS:
        rs = rows_for(sc, rnd)
        print('  %-18s 지표 %d · 읽지 않는 것 %d'
              % (rnd['name'], len(rs), sum(1 for r in rs if r['readable'] is False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
