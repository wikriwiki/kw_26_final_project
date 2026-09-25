"""정답지와의 거리 — **전 정책·전 지표**의 실측·시뮬·신뢰구간 페이지.

    python scripts/report/build_convergence_page.py

숫자를 손으로 옮기지 않는다. 채점표와 부호 적중표가 읽는 **같은 런**에서
뽑아 쓰므로, 표와 그림이 서로 다른 것을 말할 수 없다.

## 왜 전부 싣는가

실측 수치가 붙은 지표는 일곱뿐이다. 나머지는 원문이 방향만 말하거나 백분율이
공개되어 있지 않다. 그렇다고 빼면 **"다른 정책은 어떤가" 에 답이 없다.**
다 싣되 칸마다 무엇을 읽을 수 있는지 표시한다.

    실측 수치 있음   실측값은 별도 표기. 정합 감사가 있을 때만 같은 축에 겹친다
    방향만          시뮬 값과 구간을 그리고 기대 방향과 맞는지로 읽는다
    아직 안 쟀다    런이 없는 정책. 왜 없는지 적는다

## 적중보다 방향을 본다

크기 수렴은 지금 어느 정책에서도 잴 수 없다(창·분모·대조군 불일치). 그래서
칸의 표시는 적중/빗나감만이 아니라 **구간이 0 을 지나는가**까지 나눈다 —
점추정이 기대 방향인데 구간이 0 을 지나는 것은 프롬프트가 아니라 표본의
문제이고, 둘을 같은 'X' 로 묶으면 그 구분이 사라진다.

`sign_scoreboard.SUSPECT` 에 든 읽기는 **'못 셈'** 으로 칠하고 세지 않는다 —
다른 자로 잰 값이 그림에서만 적중으로 보이면 안 된다.
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
Z = 1.959964 + 0.841621     # 양측 5% + 검정력 80%. power_to_detect_truth 와 같은 값
EXPECT_TXT = {'+': '증가', '-': '감소', '0': '무반응', 'rank': '순위'}

# 아직 런이 없는 정책 — 빼면 "다른 정책은 어떤가" 에 답이 없다. 왜 없는지 적는다.
PENDING = {
    'P010': ('민생회복 P010', '창을 2026-09-24 에 등록했다 — 아직 이 창으로 돌린 적 없다'),
    'GATHERING_2020': ('사적모임 제한',
                       '창을 2026-09-24 에 등록했다 — 연말 교란 때문에 기준선 런이 함께 필요하다'),
    'P016': ('농할 P016', '사전등록·런너 준비 완료 — GPU 대기 중'),
}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def collect():
    """전 정책·전 지표. 부호 적중표가 읽는 **같은 런**에서 뽑는다."""
    sb = _load('sb', 'scripts/report/sign_scoreboard.py')
    ptt = _load('ptt', 'scripts/report/power_to_detect_truth.py')
    comparison = _load('all_indicators_table', 'scripts/report/all_indicators_table.py')
    sc = json.loads(io.open(SCORING, encoding='utf-8').read())
    rows = []
    for key, name, block, _why in list(sb.READINGS) + list(sb.PLACEBOS):
        blk = sc.get(key) or {}
        res = blk.get(block) or {}
        for ind in (blk.get('indicators') or []):
            expect = ind.get('expect')
            if expect == 'info':
                continue
            iid = ind['id']
            v = res.get(iid) if isinstance(res.get(iid), dict) else {}
            t = ptt.truth_pct(ind.get('desc'), expect)
            audit = ind.get('empirical_audit') or {}
            comparable = (comparison.direct_comparison_audited(ind)
                          and expect != 'rank' and audit.get('reported_unit') == '%')
            if comparable:
                t = audit['reported_value']
            pct, base, ci = v.get('pct'), v.get('base'), v.get('ci')
            ci_pct = None
            if isinstance(ci, list) and len(ci) == 2 and isinstance(base, (int, float)) and base:
                ci_pct = [100.0 * ci[0] / base, 100.0 * ci[1] / base]
            sus = (key, block, iid) in sb.SUSPECT
            # 방향 일치 — 점추정 부호가 기대와 같은가. 동등성·순위는 해당 없음.
            dirn = None
            if expect in ('+', '-') and isinstance(pct, (int, float)):
                dirn = (pct > 0) if expect == '+' else (pct < 0)
            rows.append({
                'policy': name, 'run': block, 'id': iid, 'expect': expect,
                'expect_txt': EXPECT_TXT.get(expect, expect), 'truth': t,
                'comparable': comparable,
                'pct': pct if isinstance(pct, (int, float)) else None,
                'ci_pct': ci_pct, 'n': v.get('n'),
                'hit': None if sus else v.get('hit'), 'dir_ok': dirn,
                'crosses': bool(ci_pct and ci_pct[0] <= 0 <= ci_pct[1]),
                'suspect': sus, 'note': str(v.get('note') or v.get('got') or '')[:58],
                'desc': (ind.get('desc') or '')[:78],
                'state': 'measured' if v else 'nodata'})
    for key, (name, why) in PENDING.items():
        blk = sc.get(key) or {}
        for ind in (blk.get('indicators') or []):
            if ind.get('expect') == 'info':
                continue
            rows.append({
                'policy': name, 'run': '', 'id': ind['id'], 'expect': ind.get('expect'),
                'expect_txt': EXPECT_TXT.get(ind.get('expect'), ind.get('expect')),
                'truth': _load('ptt2', 'scripts/report/power_to_detect_truth.py')
                         .truth_pct(ind.get('desc'), ind.get('expect')),
                'pct': None, 'ci_pct': None, 'n': None, 'hit': None, 'dir_ok': None,
                'comparable': False,
                'crosses': False, 'suspect': False, 'note': why,
                'desc': (ind.get('desc') or '')[:78], 'state': 'pending'})
    return rows


def need_n(r):
    """**지금 보이는 효과를 0 과 가르려면 몇 명이 필요한가.**

    구간이 0 을 지나는 칸에 "표본이 문제" 라고만 적으면 얼마나 모자란지 모른다.
    붓스트랩 구간을 표준편차로 되돌려 필요 n 을 역산한다 —
    `power_to_detect_truth.py` 와 같은 식이다.

        se = (hi - lo) / 3.92 · sd = se x sqrt(n) · need = ((1.96+0.84) x sd / d)^2

    d 는 **지금 관측된 효과**다. 정답지 값이 아니라 우리가 보고 있는 값을 0 과
    가르는 힘을 묻는 것이다 — "이만큼 더 보면 이 부호를 단정할 수 있다".

    어림이다. 구간은 붓스트랩이고 정규 근사로 sd 를 되돌린 값이다. 그래도
    **몇 배가 모자란지는 자릿수로 읽을 수 있다.**
    """
    ci, n, pct = r.get('ci_pct'), r.get('n'), r.get('pct')
    if not (ci and n and n > 1 and isinstance(pct, (int, float)) and pct):
        return None
    sd = (ci[1] - ci[0]) / 3.919928 * (n ** 0.5)
    d = abs(pct)
    if d <= 0:
        return None
    return int(round((Z * sd / d) ** 2))


def _x(v):
    return 50.0 + max(-DOM, min(DOM, v)) / DOM * 50.0


def _bar(r):
    if r['pct'] is None:
        return '<span class="zero"></span>'
    p = []
    if r['ci_pct']:
        a, b = _x(r['ci_pct'][0]), _x(r['ci_pct'][1])
        p.append('<span class="ci" style="left:{:.2f}%;width:{:.2f}%"></span>'
                 .format(min(a, b), abs(b - a)))
    p.append('<span class="zero"></span>')
    p.append('<span class="mk sim" style="left:{:.2f}%"></span>'.format(_x(r['pct'])))
    if r['truth'] is not None and r.get('comparable'):
        p.append('<span class="mk tru" style="left:{:.2f}%"></span>'.format(_x(r['truth'])))
    return ''.join(p)


def _tag(r):
    """칸마다 **무엇을 읽을 수 있는지**. 적중/빗나감만 찍으면 표본 문제가 숨는다."""
    if r['state'] == 'pending':
        return ('아직 안 쟀다', 'wait')
    if r['suspect']:
        return ('못 셈', 'sus')
    if r['state'] == 'nodata':
        return ('이 런에 없음', 'wait')
    # 동등성 지표(기대 '무반응')는 **구간이 0 을 지나는 것이 바라는 바**다.
    # 증감 지표와 같은 경고로 찍으면 정반대로 읽힌다.
    if r['crosses'] and r['expect'] in ('+', '-'):
        return ('구간이 0 을 지남', 'sus')
    if r['hit'] is True:
        return ('등록 판정 일치', 'ok')
    if r['hit'] is False:
        return ('등록 판정 불일치', 'no')
    return ('관측부족', 'wait')


def _nums(r):
    out = []
    if r['truth'] is not None:
        out.append('<span class="tru">실측 {:+.1f}</span>'.format(r['truth']))
        if not r.get('comparable'):
            out.append('<span class="memo">직접 크기 비교 불가</span>')
    elif r['state'] != 'pending':
        out.append('<span class="memo">실측 수치 없음 · 기대 {}</span>'.format(r['expect_txt']))
    if r['pct'] is not None:
        out.append('<span class="sim">시뮬 {:+.1f}</span>'.format(r['pct']))
        if r['truth'] is not None and r.get('comparable'):
            out.append('<span class="gap">차 {:.1f}%p</span>'.format(abs(r['pct'] - r['truth'])))
    if r['ci_pct']:
        out.append('<span class="ciTxt">구간 [{:+.0f}, {:+.0f}]</span>'.format(*r['ci_pct']))
    if r['n']:
        out.append('<span class="n">n={}</span>'.format(r['n']))
    if r['dir_ok'] is not None and not r['suspect']:
        out.append('<span class="memo">방향 {}</span>'.format('일치' if r['dir_ok'] else '반대'))
    if r['crosses'] and r['expect'] in ('+', '-'):
        k = need_n(r)
        if k and r['n']:
            out.append('<span class="need">n&asymp;{:,} 이면 갈린다 (지금의 {:.1f}배)</span>'
                       .format(k, k / r['n']))
    if r['note'] and r['pct'] is None:
        out.append('<span class="memo">{}</span>'.format(r['note']))
    return ''.join(out)


def render(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['policy'], r['run']), []).append(r)
    out = []
    for (pol, run), rs in groups.items():
        out.append('  <section class="pol"><h3>' + pol + '</h3>')
        out.append('    <p class="runline">' + (run or '런 없음') + '</p><div class="rows">')
        for r in rs:
            lab, cls = _tag(r)
            out.append(
                '    <div class="row"><div class="meta">'
                '<span class="id">{id}</span><span class="exp">{exp}</span>'
                '<span class="tag {cls}">{lab}</span>'
                '<span class="desc">{desc}</span></div>'
                '<div class="track{empty}">{bar}</div>'
                '<div class="nums">{nums}</div></div>'.format(
                    id=r['id'], exp=r['expect_txt'], cls=cls, lab=lab, desc=r['desc'],
                    empty='' if r['pct'] is not None else ' empty',
                    bar=_bar(r), nums=_nums(r)))
        out.append('  </div></section>')

    m = [r for r in rows if r['state'] == 'measured']
    judged = [r for r in m if r['dir_ok'] is not None and not r['suspect']]
    tally = [
        ('%d' % len(rows), '검증지표 전체', ''),
        ('%d' % len(m), '값이 나온 지표', ''),
        ('%d / %d' % (sum(1 for r in judged if r['dir_ok']), len(judged)),
         '방향 일치 (판정 가능한 것)', ' hi'),
        ('%d' % sum(1 for r in m if r['crosses']), '구간이 0 을 지남', ''),
        ('%d' % sum(1 for r in m if r['truth'] is not None), '실측 수치가 붙은 것', ''),
        ('%d' % sum(1 for r in m if r.get('comparable') and r['pct'] is not None),
         '직접 크기 비교 가능', ''),
    ]
    stat = ''.join('<div class="stat{c}"><span class="v">{v}</span>'
                   '<span class="k">{k}</span></div>'.format(v=v, k=k, c=c)
                   for v, k, c in tally)
    tpl = io.open(TPL, encoding='utf-8').read()
    return (tpl.replace('<!--BODY-->', chr(10).join(out))
               .replace('<!--TALLY-->', stat)
               .replace('<!--DOM-->', str(int(DOM))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()
    rows = collect()
    html = render(rows)
    io.open(a.out, 'w', encoding='utf-8', newline='\n').write(html)
    m = [r for r in rows if r['state'] == 'measured']
    judged = [r for r in m if r['dir_ok'] is not None and not r['suspect']]
    print('%s · %d bytes' % (a.out, len(html)))
    print('지표 %d개 · 값 있음 %d · 방향 일치 %d/%d · 구간이 0 을 지남 %d'
          % (len(rows), len(m), sum(1 for r in judged if r['dir_ok']), len(judged),
             sum(1 for r in m if r['crosses'])))
    for r in rows:
        print('  %-14s %-7s %-5s 실측 %-7s 시뮬 %-7s %s'
              % (r['policy'], r['id'], r['expect_txt'],
                 ('%+.1f' % r['truth']) if r['truth'] is not None else '—',
                 ('%+.1f' % r['pct']) if r['pct'] is not None else '—',
                 _tag(r)[0]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
