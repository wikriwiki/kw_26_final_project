"""정답지와의 거리 — 지표별 실측·시뮬 비교 페이지를 **자료에서** 만든다.

    python scripts/report/build_convergence_page.py

숫자를 손으로 옮기지 않는다. 채점표와 부호 적중표가 읽는 **같은 런**에서
뽑아 쓰므로, 표와 그림이 서로 다른 것을 말할 수 없다.

실측 수치가 붙은 지표만 그린다 — 전체 26개 중 여섯이다. 나머지는 원문이
방향만 말하거나 백분율이 공개되어 있지 않아 대고 잴 값이 없다.

`sign_scoreboard.SUSPECT` 에 든 읽기는 **'못 셈'** 으로 칠하고 적중/빗나감을
세지 않는다 — 다른 자로 잰 값이 그림에서만 적중으로 보이면 안 된다.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORING = ROOT / 'data/experiments/scoring_table.json'
TPL = ROOT / 'output/report/convergence_tpl.html'
OUT = ROOT / 'output/report/convergence.html'

DOM = 36.0          # 축 범위 ±36% — 가장 큰 실측(+20.8)과 구간 끝(-35)을 담는다
STATE = {True: ('적중', 'ok'), False: ('빗나감', 'no'), None: ('못 셈', 'sus')}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def collect():
    sb = _load('sb', 'scripts/report/sign_scoreboard.py')
    ptt = _load('ptt', 'scripts/report/power_to_detect_truth.py')
    sc = json.loads(io.open(SCORING, encoding='utf-8').read())
    rows = []
    for key, name, block, _why in list(sb.READINGS) + list(sb.PLACEBOS):
        blk = sc.get(key) or {}
        res = blk.get(block) or {}
        for ind in (blk.get('indicators') or []):
            iid, expect = ind['id'], ind.get('expect')
            if expect == 'info':
                continue
            v = res.get(iid)
            if not isinstance(v, dict):
                continue
            t = ptt.truth_pct(ind.get('desc'), expect)
            pct, base, ci = v.get('pct'), v.get('base'), v.get('ci')
            if t is None or not isinstance(pct, (int, float)):
                continue
            ci_pct = None
            if isinstance(ci, list) and len(ci) == 2 and isinstance(base, (int, float)) and base:
                ci_pct = [100.0 * ci[0] / base, 100.0 * ci[1] / base]
            sus = (key, block, iid) in sb.SUSPECT
            rows.append({'policy': name, 'id': iid, 'truth': t, 'pct': pct,
                         'ci_pct': ci_pct, 'n': v.get('n'),
                         'hit': None if sus else v.get('hit'),
                         'desc': (ind.get('desc') or '')[:70]})
    return rows


def _x(v):
    return 50.0 + max(-DOM, min(DOM, v)) / DOM * 50.0


def _bar(r):
    p = []
    if r['ci_pct']:
        a, b = _x(r['ci_pct'][0]), _x(r['ci_pct'][1])
        p.append('<span class="ci" style="left:{:.2f}%;width:{:.2f}%"></span>'
                 .format(min(a, b), abs(b - a)))
    p.append('<span class="zero"></span>')
    p.append('<span class="mk sim" style="left:{:.2f}%"></span>'.format(_x(r['pct'])))
    p.append('<span class="mk tru" style="left:{:.2f}%"></span>'.format(_x(r['truth'])))
    return ''.join(p)


def render(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r['policy'], []).append(r)
    out = []
    for pol, rs in groups.items():
        out.append('  <section class="pol"><h3>' + pol + '</h3><div class="rows">')
        for r in rs:
            lab, cls = STATE[r['hit']]
            out.append(
                '    <div class="row"><div class="meta">'
                '<span class="id">{id}</span><span class="tag {cls}">{lab}</span>'
                '<span class="desc">{desc}</span></div>'
                '<div class="track">{bar}</div><div class="nums">'
                '<span class="tru">실측 {t:+.1f}</span><span class="sim">시뮬 {s:+.1f}</span>'
                '<span class="gap">차 {g:.1f}%p</span><span class="n">n={n}</span>'
                '</div></div>'.format(id=r['id'], cls=cls, lab=lab, desc=r['desc'],
                                      bar=_bar(r), t=r['truth'], s=r['pct'],
                                      g=abs(r['pct'] - r['truth']), n=r['n']))
        out.append('  </div></section>')
    tpl = io.open(TPL, encoding='utf-8').read()
    return tpl.replace('<!--BODY-->', '\n'.join(out)).replace('<!--DOM-->', str(int(DOM)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()
    rows = collect()
    html = render(rows)
    io.open(a.out, 'w', encoding='utf-8', newline='\n').write(html)
    print('%s · 지표 %d개 · %d bytes' % (a.out, len(rows), len(html)))
    for r in rows:
        print('  %-14s %-7s 실측 %+6.1f  시뮬 %+6.1f  %s'
              % (r['policy'], r['id'], r['truth'], r['pct'], STATE[r['hit']][0]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
