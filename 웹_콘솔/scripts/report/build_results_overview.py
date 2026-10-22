"""실험 라운드마다 **어떤 프롬프트로 돌렸고, 실측과 얼마나 차이 나는지**를 그린다.

한 눈에 보이게 하는 것이 목적이다. 지표마다 실측 막대와 시뮬 막대를 같은 눈금에
나란히 놓고, 그 차이를 옆에 적는다.

    python scripts/report/build_results_overview.py

산출: experiments/RESULTS.md

**수치는 전부 `data/experiments/scoring_table.json` 에서 온다.** 손으로 적지 않는다.
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORING = ROOT / 'data/experiments/scoring_table.json'
OUT = ROOT / 'experiments/RESULTS.md'

# 정답지에 대고 잰 라운드는 **전부 v5 로 돌았다.** v50 사전등록이 그것을 못 박았다
# ("정답지에 대고 잰 기록은 전부 v5 다"). v40·v42·v45 는 형식 계약 관문 라인에서만
# 돌았고, v45 를 정답지에 대고 재는 것은 v50 이 처음이다.
PROMPT_BY_ROUND = {
    'default': 'v5',
    'note': 'v50 부터 v45 가 정답지에 대고 처음 돌아간다',
}

POLICIES = [
    ('P010', '민생회복 소비쿠폰', 'P010', '민생회복소비쿠폰_P010'),
    ('P012', '상생소비지원금', 'P012', '상생소비지원금_P012'),
    ('P013', '긴급재난지원금', 'EMERGENCY_2020', '긴급재난지원금_P013'),
    ('P014', '지역사랑상품권', 'LOCAL_VOUCHER', '지역사랑상품권_P014'),
    ('P015', '8대 소비쿠폰 (홀드아웃)', 'SECTOR_VOUCHER_2020', '업종형소비쿠폰_P015'),
    ('P016', '농축산물 할인쿠폰', 'P016', '농축산물할인쿠폰_P016'),
    ('—', '사회적 거리두기', 'DISTANCING_2020', '사회적거리두기_DISTANCING2020'),
]

WIDTH = 26  # 막대 한쪽 길이


# 원문은 유니코드 빼기표(−, U+2212)를 쓴다. ASCII 하이픈으로 바꾸지 않으면
# −14.1% 가 +14.1% 로 읽힌다. 실제로 한 번 그렇게 나왔다.
DASHES = {'−': '-', '–': '-', '—': '-', '－': '-'}


def norm(t):
    t = str(t or '')
    for a, b in DASHES.items():
        t = t.replace(a, b)
    return t


def truth_pct(desc, expect=None):
    """지표 설명의 '(실측 …)' 에서 퍼센트 값을 꺼낸다. 없으면 None.

    순위 지표는 값이 둘이라(A vs B) 막대 하나로 그리면 오해를 준다 — 그리지 않는다.
    """
    if expect == 'rank':
        return None
    m = re.search(r'\(실측\s*([^)]*)\)', norm(desc))
    if not m:
        return None
    body = m.group(1)
    if '원' in body and '%' not in body:
        return None                      # 금액은 퍼센트 막대에 못 올린다
    m2 = re.search(r'([+-]?\d+(?:\.\d+)?)\s*%', body)
    return float(m2.group(1)) if m2 else None


def truth_text(desc):
    """표에 그대로 적을 실측 문구 — 순위 지표의 'A vs B' 까지 살린다."""
    m = re.search(r'\(실측\s*([^)]*)\)', norm(desc))
    return m.group(1).strip() if m else None


def sim_pct(val):
    if not isinstance(val, dict):
        return None
    v = val.get('pct')
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def bar(value, scale):
    """0 을 가운데 둔 발산형 막대. 왼쪽이 감소, 오른쪽이 증가."""
    if value is None or scale <= 0:
        return ' ' * WIDTH + '│' + ' ' * WIDTH
    n = max(1, min(WIDTH, round(abs(value) / scale * WIDTH)))
    if value >= 0:
        return ' ' * WIDTH + '│' + '█' * n + ' ' * (WIDTH - n)
    return ' ' * (WIDTH - n) + '█' * n + '│' + ' ' * WIDTH


def num(v, unit='%'):
    return ('%+.1f%s' % (v, unit)) if isinstance(v, (int, float)) else '—'


def indicator_rows(block, inds):
    """지표 목록을 뼈대로, 실측과 시뮬을 한 줄씩."""
    out = []
    for i in inds:
        iid = i.get('id')
        val = block.get(iid)
        out.append({
            'id': iid,
            'expect': i.get('expect'),
            'what': re.sub(r'\s*\(실측[^)]*\)', '', str(i.get('desc') or '')).strip(' -—'),
            'truth': truth_pct(i.get('desc'), i.get('expect')),
            'truth_text': truth_text(i.get('desc')),
            'sim': sim_pct(val),
            'ci': (val or {}).get('ci') if isinstance(val, dict) else None,
            'hit': (val or {}).get('hit') if isinstance(val, dict) else None,
            'note': (val or {}).get('note') if isinstance(val, dict) else None,
            'measured': isinstance(val, dict),
        })
    return out


def draw(rows):
    vals = [abs(v) for r in rows for v in (r['truth'], r['sim']) if v is not None]
    if not vals:
        return ['```', '이 라운드에서 퍼센트로 견줄 수 있는 지표가 없다.', '```', '']
    scale = max(vals)
    L = ['```',
         ' ' * 10 + '←  %-*s0%*s  →' % (WIDTH - 1, '%.0f' % scale, WIDTH - 1, '%.0f' % scale),
         ' ' * 10 + '감소' + ' ' * (2 * WIDTH - 5) + '증가']
    for r in rows:
        tag = r['id']
        if r['truth'] is None and r['sim'] is None:
            L.append('%-9s %s   실측·시뮬 모두 퍼센트 값 없음' % (tag, ' ' * (2 * WIDTH + 1)))
            continue
        lab = num(r['truth']) if r['truth'] is not None else (
            (r['truth_text'] or '수치 없음')[:34])
        L.append('%-9s %s  실측 %s' % (tag, bar(r['truth'], scale), lab))
        gap = ('  차이 %+.1f%%p' % (r['sim'] - r['truth'])
               if (r['sim'] is not None and r['truth'] is not None) else '')
        if r['sim'] is None:
            L.append('%-9s %s  시뮬 — %s' % ('', ' ' * (2 * WIDTH + 1),
                                            '안 잼' if not r['measured'] else '값 없음'))
        else:
            zero = ''
            if isinstance(r['ci'], list) and len(r['ci']) == 2:
                if r['ci'][0] <= 0 <= r['ci'][1]:
                    zero = '  ← 구간이 0 을 지난다'
            L.append('%-9s %s  시뮬 %s%s%s' % ('', bar(r['sim'], scale), num(r['sim']), gap, zero))
        L.append('')
    L.append('```')
    return L


def table(rows):
    L = ['| 지표 | 무엇을 재나 | 기대 | 실측 | 시뮬 | 95% 구간 | 차이 | 점추정 부호 | 채점 |',
         '|---|---|:-:|---:|---:|---|---:|:-:|---|']
    for r in rows:
        ci = '—'
        if isinstance(r['ci'], list) and len(r['ci']) == 2:
            ci = '[%s, %s]' % (format(r['ci'][0], '+,.0f'), format(r['ci'][1], '+,.0f'))
            if r['ci'][0] <= 0 <= r['ci'][1]:
                ci += ' **0 포함**'
        gap = ('%+.1f%%p' % (r['sim'] - r['truth'])
               if (r['sim'] is not None and r['truth'] is not None) else '—')
        # 채점(hit)은 부호 + 유의성을 함께 본다. 점추정 부호만 맞는 칸을
        # '부호 불일치'로 적으면 사실과 다르다 — 갈라서 적는다.
        verdict = {True: '적중', False: '미적중'}.get(r['hit'],
                                                  '안 잼' if not r['measured'] else '—')
        sign = '—'
        if r['sim'] is not None and r['expect'] in ('+', '-'):
            want = 1 if r['expect'] == '+' else -1
            sign = '일치' if (r['sim'] > 0) == (want > 0) else '**반대**'
        elif r['sim'] is not None and r['truth'] is not None:
            sign = '일치' if (r['sim'] >= 0) == (r['truth'] >= 0) else '**반대**'
        L.append('| `%s` | %s | `%s` | %s | %s | %s | %s | %s | %s |' % (
            r['id'], r['what'][:46].replace('|', '·'), r['expect'],
            (r['truth_text'] or num(r['truth'])).replace('|', '·'),
            num(r['sim']), ci, gap, sign, verdict))
    L.append('')
    return L


def build():
    sc = json.loads(SCORING.read_text(encoding='utf-8'))
    L = ['# 라운드별 결과 — 실측과 얼마나 차이 나는가', '',
         '**자동 생성.** `python scripts/report/build_results_overview.py`', '',
         '지표마다 **실측 막대와 시뮬 막대를 같은 눈금에 나란히** 놓았다. '
         '왼쪽이 감소, 오른쪽이 증가이고, 0 이 가운데다.', '',
         '> **정답지에 대고 잰 라운드는 전부 프롬프트 `v5` 로 돌았다.** '
         'v40·v42·v45 는 형식 계약 관문 라인에서만 돌았고, '
         'v45 를 정답지에 대고 재는 것은 **v50 이 처음**이다(진행 중).', '',
         '> 수치는 전부 `data/experiments/scoring_table.json` 에서 온다. 손으로 적지 않는다.', '']

    # 요약표
    summary = []
    for pid, name, key, folder in POLICIES:
        blk = sc.get(key) or {}
        inds = blk.get('indicators') or []
        rounds = [(n, b) for n, b in blk.items() if n.startswith('result') and isinstance(b, dict)]
        if not rounds:
            summary.append((pid, name, '—', '—', '아직 채점 기록 없음'))
            continue
        for rn, rb in rounds:
            rows = indicator_rows(rb, inds)
            hits = sum(1 for r in rows if r['hit'] is True)
            scored = sum(1 for r in rows if r['hit'] in (True, False))
            summary.append((pid, name, rn, PROMPT_BY_ROUND['default'],
                            '부호 %d/%d' % (hits, scored) if scored else '—'))
    L += ['## 한눈에', '', '| 정책 | 라운드 | 프롬프트 | 부호 적중 |', '|---|---|:-:|---|']
    for pid, name, rn, pr, res in summary:
        L.append('| %s %s | `%s` | `%s` | %s |' % (pid, name, rn, pr, res))
    L += ['', '---', '']

    for pid, name, key, folder in POLICIES:
        blk = sc.get(key) or {}
        inds = blk.get('indicators') or []
        L += ['## %s %s' % (pid, name), '']
        if blk.get('answer_key'):
            L += ['정답지 — %s · 원문은 [`data/policy_raw_data/%s/`](../data/policy_raw_data/%s/평가항목.md)'
                  % (blk['answer_key'], folder, folder), '']
        rounds = [(n, b) for n, b in blk.items() if n.startswith('result') and isinstance(b, dict)]
        if not rounds:
            L += ['> 이 채점표에 결과 블록이 없다. '
                  '(P010 의 결과는 EXP-001 라인 — `docs/EXP001_결과분석.md`)', '']
            continue
        for rn, rb in rounds:
            L += ['### `%s` — 프롬프트 `%s`' % (rn, PROMPT_BY_ROUND['default']), '']
            if rb.get('window') or rb.get('design'):
                L += ['창 · 설계 — %s' % str(rb.get('window') or rb.get('design')), '']
            rows = indicator_rows(rb, inds)
            L += draw(rows)
            L += table(rows)
            for r in rows:
                if r['note']:
                    L.append('- `%s` — %s' % (r['id'], r['note']))
            if any(r['note'] for r in rows):
                L.append('')
            did = rb.get('did')
            if isinstance(did, dict):
                L += ['**순효과(DID)** — 정책 %s · 기준선 %s · 순수 효과 **%s**'
                      % (num(did.get('policy_effect_pct')), num(did.get('baseline_pct')),
                         num(did.get('net_pp'), '%p')), '']
        L += ['---', '']

    L += ['## 이 표를 읽을 때 주의할 것', '',
          '- **크기 검증은 아직 성립하지 않았다.** 감사 주석(`audit_2026_09_20`)이 '
          '대상·기간·결과·분모·대조군이 호환될 때만 배수를 비교하라고 했다. '
          '채점은 **부호**로 한다',
          '- **원문 창과 우리 창이 다르다.** 정답지는 한 달~40주를 재고 우리는 이틀을 잰다. '
          '차이 칸의 %p 는 같은 눈금이 아니다',
          '- **채점(적중)은 부호와 유의성을 함께 본다.** 점추정 부호가 맞아도 '
          '구간이 0 을 지나면 미적중이다 — 그래서 두 칸을 갈라 적었다',
          '- **구간이 0 을 지나면 부호조차 단정할 수 없다.** 그런 칸을 표시해 두었다',
          '- 지표별 범위 불일치(거리두기 −14.1% 는 한식 값 등)는 각 정책 폴더의 '
          '`평가항목.md` 맨 아래에 적혀 있다', '']
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stdout', action='store_true')
    a = ap.parse_args()
    text = build()
    if a.stdout:
        print(text)
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, 'w', encoding='utf-8', newline='\n').write(text)
    print('완료: %s (%d줄)' % (OUT.relative_to(ROOT), text.count('\n')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
