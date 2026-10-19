"""Draw the per-mechanism comparison for one experiment round into markdown.

The bars are computed from the recorded evidence files, never typed by hand. Text
bars are used on purpose: they render in every markdown viewer and carry no colour,
so identity never depends on hue. Zero sits in the middle; left is a decrease.

    python scripts/report/build_experiment_graphs.py --round v1 --out experiments/v1/graph_v1.md

This script reports what a run measured. It does not decide whether a prompt is
good, and it must not be pointed at an incomplete matrix without saying so.
"""
from __future__ import annotations

import argparse
import io
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIDTH = 24
MECH_KO = {'cashback': '캐시백', 'grant': '지원금', 'distancing': '거리두기', 'local_voucher': '지역화폐'}


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def bar(value: float, span: float) -> str:
    """Diverging bar with zero fixed at the centre column."""
    n = 0 if span <= 0 else min(WIDTH, int(round(abs(value) / span * WIDTH)))
    if value >= 0:
        return ' ' * WIDTH + '│' + '█' * n + ' ' * (WIDTH - n)
    return ' ' * (WIDTH - n) + '█' * n + '│' + ' ' * WIDTH


def cells_wide(text: str) -> int:
    """Terminal columns a label occupies. Hangul is double width; str len misaligns bars."""
    return sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + ' ' * max(0, width - cells_wide(text))


def axis(span: float, unit: str) -> str:
    """Scale line whose 0 sits in the same column as every bar baseline."""
    row = [' '] * (2 * WIDTH + 1)
    row[WIDTH] = '0'
    left, right = f'-{span:,.0f}{unit}', f'+{span:,.0f}{unit}'
    for i, ch in enumerate(left):
        row[i] = ch
    for i, ch in enumerate(right):
        row[WIDTH + 2 + i] = ch
    return ''.join(row)


def panel(title: str, rows: list[tuple[str, float]], unit: str, note: str = '') -> list[str]:
    """rows = [(label, signed value)]. A single series needs no legend; the title names it."""
    span = max((abs(v) for _, v in rows), default=0.0)
    width = max((cells_wide(r[0]) for r in rows), default=0)
    out = [f'**{title}**', '', '```']
    out.append(' ' * (width + 2) + axis(span, unit))
    for label, value in rows:
        out.append(f'{pad(label, width)}  {bar(value, span)}  {value:+,.0f}{unit}')
    out.append('```')
    if note:
        out += ['', note]
    out.append('')
    return out


def contrast_rows(report: dict, metric: str):
    """Scored mechanisms as (label, difference); refused ones kept aside with their reason.

    A refusal is a result. It is never rendered as a zero-height bar, which would
    read as "no effect" instead of "not comparable".
    """
    rows, refused = [], []
    for mech in sorted(report['mechanisms']):
        block = report['mechanisms'][mech]
        name = MECH_KO.get(mech, mech)
        if 'matched_contrasts' not in block:
            refused.append((name, block.get('not_scored', '사유 미기록')))
            continue
        seeds = block['matched_contrasts']['by_seed']
        for entry in seeds:
            label = name + (f" seed{entry['replicate']}" if len(seeds) > 1 else '')
            rows.append((label, float(entry['metrics'][metric]['difference'])))
    return rows, refused


def levels_rows(report: dict, metric: str) -> list[tuple[str, float, float]]:
    rows = []
    for mech in sorted(report['mechanisms']):
        block = report['mechanisms'][mech]
        if 'matched_contrasts' not in block:
            continue
        for entry in block['matched_contrasts']['by_seed']:
            m = entry['metrics'][metric]
            rows.append((MECH_KO.get(mech, mech), float(m['off_mean']), float(m['on_mean'])))
    return rows


def mechanism_differences(reports: dict) -> dict:
    """{기전: {후보: 차이}}. 채점 거부는 빠진 채로 남고, 0으로 채우지 않는다."""
    table: dict = {}
    for label, report in reports.items():
        for mech, block in report['mechanisms'].items():
            name = MECH_KO.get(mech, mech)
            if 'matched_contrasts' not in block:
                continue
            for entry in block['matched_contrasts']['by_seed']:
                table.setdefault(name, {})[label] = float(
                    entry['metrics']['total_consumption']['difference'])
    return table


def candidate_panels(reports: dict, band: dict | None) -> list[str]:
    """기전마다 후보를 나란히 놓고, 사전에 정한 잡음 폭과 대본다.

    판정은 여기서 계산한다. 결과를 본 뒤 기준을 고쳐 쓰는 일을 막기 위해서다.
    후보 간 폭이 잡음 폭 이하이면 '구별 불가'이고, 그것도 결과다.
    """
    table = mechanism_differences(reports)
    if not table:
        return []
    widths = (band or {}).get('by_mechanism', {})
    out = ['## 기전별 후보 비교 — 잡음 폭과 함께 읽는다', '']
    if band:
        out += [f"> 잣대는 v3 결과를 보기 전에 고정했다 (평균 {band['mean_width']:,.0f}원 · "
                f"최대 {band['max_width']:,.0f}원). `data/experiments/seed_variation_band_v25.json`",
                '> **후보 간 폭이 잡음 폭 이하이면 후보의 차이로 읽지 않는다.**', '']
    verdicts = []
    for name in sorted(table):
        per = table[name]
        rows = [(lbl, per[lbl]) for lbl in sorted(per)]
        spread = max(v for _, v in rows) - min(v for _, v in rows)
        bw = widths.get(name, {}).get('width')
        if bw is None:
            note = f'후보 간 폭 {spread:,.0f}원 · 이 기전의 잡음 폭은 기록에 없다 — 판정하지 않는다.'
            verdicts.append((name, spread, None, '잣대 없음'))
        elif spread <= bw:
            note = (f'후보 간 폭 **{spread:,.0f}원** ≤ 잡음 폭 {bw:,.0f}원 → '
                    f'**구별 불가.** 후보의 차이라고 말하지 않는다.')
            verdicts.append((name, spread, bw, '구별 불가'))
        else:
            note = (f'후보 간 폭 **{spread:,.0f}원** > 잡음 폭 {bw:,.0f}원 → 폭을 넘었다. '
                    f'다만 후보당 seed 하나이므로 **후보 탓이라고 단정하지 않는다** (v4로 넘긴다).')
            verdicts.append((name, spread, bw, '폭 초과'))
        out += panel(name, rows, '원', note)
    out += ['**판정 요약**', '',
            '| 기전 | 후보 간 폭 | 잡음 폭 | 판정 |', '|---|---:|---:|---|']
    for name, spread, bw, verdict in verdicts:
        out.append(f"| {name} | {spread:,.0f} | {'—' if bw is None else format(bw, ',.0f')} | {verdict} |")
    out += ['']
    if all(v[3] == '구별 불가' for v in verdicts):
        out += ['> 네 기전 모두 구별 불가다. **2차 판정으로는 후보를 고를 수 없다.** '
                '사전등록대로 1차 판정(실행 품질)으로만 고른다.', '']
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--source', action='append', required=True,
                    help='label=path to a coupled report, repeatable')
    ap.add_argument('--band', help='pre-registered seed-variation band; enables the verdict')
    args = ap.parse_args()
    band = load(args.band) if args.band else None
    reports: dict = {}

    lines = [f'# 실험 {args.round} — 정책별 결과 비교', '']
    lines += ['> 막대는 **0이 가운데**다. 왼쪽이 감소, 오른쪽이 증가.',
              '> 값은 기록된 근거 파일에서 계산했다. 손으로 적은 숫자는 없다.', '']

    for item in args.source:
        label, path = item.split('=', 1)
        report = load(path)
        reports[label] = report
        lines += [f'## {label}', '']
        complete = report.get('all_matrices_complete')
        cells, rows_n = report.get('source_cells'), report.get('purchase_rows')
        lines += [f'- 칸 {cells} · 결제 행 {rows_n} · 전체 행렬 완전: **{"예" if complete else "아니오"}**']
        if not complete:
            lines += ['- ⚠ **행렬이 불완전하다.** 아래 수치를 온전한 비교로 읽으면 안 된다.']
        lines += [f"- 범위: {report.get('scope','')}", '']

        scored, refused = contrast_rows(report, 'total_consumption')
        if scored:
            lines += panel('정책 있음 − 정책 없음 (총소비, 1인 1일)', scored, '원',
                           '시뮬레이터 안에서 같은 사람·같은 날짜의 on/off 차이다. '
                           '실제 정책 효과나 모집단 유의성이 아니다.')
        for name, reason in refused:
            lines += [f'**{name} — 채점 거부**', '', '```', f'  {reason}', '```', '',
                      '거부는 결과다. 0으로 그리지 않는다 — "효과 없음"이 아니라 "비교 불가"다.', '']

        lv = levels_rows(report, 'total_consumption')
        lines += ['**수준값 (원 / 1인 1일)**', '',
                  '| 기전 | 정책 없음 | 정책 있음 | 차이 |', '|---|---:|---:|---:|']
        for name, off, on in lv:
            lines.append(f'| {name} | {off:,.0f} | {on:,.0f} | {on-off:+,.0f} |')
        lines += ['']

    if len(reports) > 1:
        lines += candidate_panels(reports, band)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
    print(f'wrote {args.out} ({len(lines)} lines)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
