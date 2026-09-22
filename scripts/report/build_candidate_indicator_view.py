"""Put the measured value and several candidates side by side, one indicator per block.

This is the view that shows whether a prompt change moved the simulation toward the
published numbers or away from them. A candidate can win a process criterion - buying
more often, reaching more places - and still move an indicator further from its answer,
and only this view makes that visible.

Bands are bootstrap intervals over cells. An interval that excludes zero on the wrong
side of the measured value is worse than one that merely fails to decide: the run is
now confidently wrong rather than uninformative.

    python scripts/report/build_candidate_indicator_view.py \
        --measured-from scripts/report/build_indicator_comparison.py \
        --cand v25=data/experiments/v7_v25_pooled.json \
        --cand v30=data/experiments/v7_v30_pooled.json \
        --out experiments/v7/indicator_view_v7.md
"""
from __future__ import annotations

import argparse
import io
import json
import unicodedata
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_indicator_comparison import CANNOT, MEASURED, POLICY

WIDTH = 22


def cells_wide(text):
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)


def pad(text, width):
    return text + ' ' * max(0, width - cells_wide(text))


def bar(value, span):
    if span <= 0:
        return ' ' * WIDTH + '│' + ' ' * WIDTH
    n = max(0, min(WIDTH, int(round(abs(value) / span * WIDTH))))
    if value >= 0:
        return ' ' * WIDTH + '│' + '█' * n + ' ' * (WIDTH - n)
    return ' ' * (WIDTH - n) + '█' * n + '│' + ' ' * WIDTH


def status(expect, measured, blk):
    """Sign only, and whether the interval settles it. Never a pass mark on magnitude."""
    if blk is None:
        return '산출 불가'
    v, stable = blk['mean'], blk.get('sign_stable')
    if v == 0.0 and blk.get('lo') == 0.0 and blk.get('hi') == 0.0:
        return '공허 (양쪽 0)'
    if expect == '0':
        return '방어선'
    want = None
    if expect == 'rank':
        want = 1 if (measured or 0) > 0 else -1
    else:
        want = 1 if expect == '+' else -1
    got = 1 if v > 0 else -1
    agree = want == got
    if not stable:
        return '미결 (구간이 0 포함)'
    return '부호 일치' if agree else '**부호 반대 — 구간이 0을 제외**'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', action='append', required=True, metavar='label=pooled.json')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cands, meta = {}, {}
    for item in args.cand:
        label, path = item.split('=', 1)
        doc = json.loads(Path(path).read_text(encoding='utf-8'))
        cands[label] = doc['pooled']
        meta[label] = doc
    labels = list(cands)

    # These were hardcoded to the twelve-citizen runs (384칸 · 4 seed · 2,000회) and stayed
    # wrong for every later cohort. Read them from the pooled files that were actually passed.
    sizes = ' · '.join('%s %d칸' % (c, sum((meta[c].get('cells_per_case_arm') or {}).values()))
                       for c in labels)
    one = meta[labels[0]]
    draws = one.get('draws', 2000)
    unit = {'citizen': '시민을 통째로 재추출',
            'seed block then cell': '블록을 먼저, 그 안에서 칸을 재추출'}.get(
                one.get('resample_unit'), '칸을 재추출')
    blocks = one.get('blocks', 1)
    L = ['# 정답지 × 후보 — 지표별로 나란히', '',
         f'> 후보 {len(labels)}개: **{" · ".join(labels)}** · {sizes}.',
         f'> 괄호는 95% 붓스트랩 구간({draws:,}회, {unit}). 구간이 0을 지나면 부호조차 말할 수 없다.', '']
    if blocks < 2 and one.get('resample_unit') != 'citizen':
        L += ['> **주의 — 복제가 하나뿐이라 구간이 실제보다 좁다.** v7 에서 이 조건의 판정을 철회한 적이 있다.', '']
    L += [
         '> **금액은 맞대지 않는다.** 카탈로그는 낱개 단가, 실측은 카드매출이다. 비율만 같은 종류의 수다.', '']

    rows = []
    for pol in ('P012', 'EMERGENCY', 'LOCAL_VOUCHER', 'DISTANCING'):
        name, date, real, key = POLICY[pol]
        ids = [k for k, v in MEASURED.items() if v[0] == pol]
        L += ['---', '', f'## {name}', '',
              f'- 칸의 날짜 **{date}** · 현실 대응 **{real}** · 정답지 **{key}**', '']
        vals = [abs(MEASURED[k][1]) for k in ids if MEASURED[k][1] is not None]
        for k in ids:
            for c in labels:
                if k in cands[c]:
                    vals.append(abs(cands[c][k]['mean']))
        span = max(vals) if vals else 1.0
        L += ['```', ' ' * 12 + f'←  {span:,.0f}' + ' ' * max(0, WIDTH - 8) + '0'
              + ' ' * max(0, WIDTH - 8) + f'{span:,.0f}  →']
        for k in ids:
            m = MEASURED[k][1]
            L.append(pad('실측', 12) + (bar(m, span) if m is not None else ' ' * (2 * WIDTH + 1))
                     + '  ' + (f'{m:+.1f}' if m is not None else '수치 없음') + f'   ({k})')
            for c in labels:
                blk = cands[c].get(k)
                if blk is None:
                    L.append(pad(c, 12) + ' ' * (2 * WIDTH + 1) + '  산출 불가')
                    continue
                flag = '' if blk.get('sign_stable') else '  ← 0 포함'
                L.append(pad(c, 12) + bar(blk['mean'], span) + f"  {blk['mean']:+.1f}"
                         f"  [{blk['lo']:+.1f}, {blk['hi']:+.1f}]{flag}")
            L.append('')
        L += ['```', '', '| 지표 | 기대 | 실측 |'
              + ''.join(f' {c} |' for c in labels) + ' 판정 변화 |',
              '|---|:-:|---:|' + '---:|' * len(labels) + '---|']
        for k in ids:
            _, m, expect, _desc = MEASURED[k]
            cells = []
            sts = []
            for c in labels:
                blk = cands[c].get(k)
                cells.append('산출 불가' if blk is None else f"{blk['mean']:+.1f}")
                sts.append(status(expect, m, blk))
            change = sts[0] if len(set(sts)) == 1 else ' → '.join(sts)
            rows.append((k, expect, m, cells, sts))
            L.append(f'| {k} | `{expect}` | '
                     + (f'{m:+.1f}' if m is not None else '—') + ' | '
                     + ' | '.join(cells) + f' | {change} |')
        L += ['']
        miss = [k for k in ids if all(k not in cands[c] for c in labels)]
        if miss:
            L += ['**왜 낼 수 없나**', '']
            L += [f'- **{k}** — {CANNOT[k]}' for k in miss]
            L += ['']

    # Tally per candidate
    L += ['---', '', '## 후보별 총괄', '',
          '| | ' + ' | '.join(labels) + ' |', '|---|' + '---:|' * len(labels)]
    for want in ('부호 일치', '**부호 반대 — 구간이 0을 제외**', '미결 (구간이 0 포함)',
                 '방어선', '공허 (양쪽 0)', '산출 불가'):
        counts = []
        for i, _c in enumerate(labels):
            counts.append(sum(1 for r in rows if r[4][i] == want))
        L.append(f'| {want} | ' + ' | '.join(str(x) for x in counts) + ' |')
    L += ['', '> **부호 반대인데 구간이 0을 제외한다**는 것은 미결보다 나쁘다.',
          '> 모른다가 아니라 **확신을 갖고 틀렸다**는 뜻이기 때문이다.', '']

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
    print('wrote', args.out)
    for want in ('부호 일치', '**부호 반대 — 구간이 0을 제외**', '미결 (구간이 0 포함)'):
        counts = [sum(1 for r in rows if r[4][i] == want) for i in range(len(labels))]
        print(f'  {want:<34} ' + '  '.join(f'{c}={n}' for c, n in zip(labels, counts)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
