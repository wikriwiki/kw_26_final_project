"""DS-1 under both readings of the published measure, because we cannot yet tell them apart.

The published restaurant figure may or may not include delivery, and which it is decides
which of our numbers is its counterpart. Our own sector '식사' bundles dine-in, takeaway and
both delivery channels, so the one indicator we report has been answering a question we had
not settled. Splitting it is not a choice of definition - it is a refusal to make that choice
silently. Both readings are printed, side by side, every time.

    매장·포장   meal_dine_in + meal_takeaway        - bought at the restaurant
    식사 전체   the above + home/office delivery     - our existing sector

The interval resamples citizens, not cells: one citizen contributes both arms and several
cells, and those move together. It is paired, so a citizen drawn for the numerator is the
same citizen in the denominator, which removes the between-person spread that dominates at
this roster size.

The denominator gate runs first and can stop the report. An indicator computed on four
events is not an estimate, and v18's DS-4 returned [-100, -100] from a single dessert.

    python scripts/report/ds1_two_definitions.py --run asm_70001.json@buy_70001/responses.jsonl \
        --run asm_70002.json@buy_70002/responses.jsonl --min-events 20
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_indicators import purchases  # noqa: E402

STORE = ('meal_dine_in', 'meal_takeaway')
DELIVERY = ('home_delivery', 'office_delivery')
DEFINITIONS = {
    '매장·포장': set(STORE),
    '식사 전체': set(STORE) | set(DELIVERY),
}


def split(spec):
    src, _, resp = spec.partition('@')
    if not resp:
        raise ValueError('Expected source@responses, got %r' % spec)
    return src, resp


def collect(runs, case='distancing'):
    """{definition: {citizen: {arm: [amount, count]}}} - one entry per citizen, not per cell."""
    out = {name: defaultdict(lambda: {'off': [0.0, 0], 'on': [0.0, 0]}) for name in DEFINITIONS}
    for src, resp in runs:
        for rec in purchases(src, resp):
            if rec['case'] != case or rec['arm'] not in ('off', 'on'):
                continue
            for name, ids in DEFINITIONS.items():
                for it in rec['items']:
                    if it['activity_id'] in ids:
                        cell = out[name][rec['aid']][rec['arm']]
                        cell[0] += it['price']
                        cell[1] += 1
    return out


def pct(on, off):
    return None if off <= 0 else (on - off) / off * 100.0


def paired_bootstrap(per_citizen, draws=4000, seed=20260922):
    """Resample citizens as bundles; each draw keeps both arms of the drawn citizen."""
    aids = sorted(per_citizen)
    if not aids:
        return None
    rng = random.Random(seed)
    keep = []
    for _ in range(draws):
        on = off = 0.0
        for _ in aids:
            c = per_citizen[aids[rng.randrange(len(aids))]]
            off += c['off'][0]
            on += c['on'][0]
        v = pct(on, off)
        if v is not None:
            keep.append(v)
    if len(keep) < 20:
        return None
    keep.sort()
    return {'lo': keep[int(0.025 * len(keep))], 'hi': keep[int(0.975 * len(keep)) - 1]}


def report(runs, min_events=20, draws=4000):
    data = collect(runs)
    lines, verdicts = [], {}
    gate_name = '매장·포장'
    off_events = sum(c['off'][1] for c in data[gate_name].values())
    gate_ok = off_events >= min_events
    lines.append('분모 관문: 무정책 팔 매장·포장 %d건 (요구 %d건) — %s'
                 % (off_events, min_events, '통과' if gate_ok else '미달'))
    lines.append('')
    lines.append('%-12s %10s %10s %10s %10s %12s %22s'
                 % ('정의', 'off 건수', 'on 건수', 'off 금액', 'on 금액', 'DS-1', '시민 군집 짝지은 구간'))
    for name in DEFINITIONS:
        per = data[name]
        ofc = sum(c['off'][1] for c in per.values())
        onc = sum(c['on'][1] for c in per.values())
        ofa = sum(c['off'][0] for c in per.values())
        ona = sum(c['on'][0] for c in per.values())
        v = pct(ona, ofa)
        ci = paired_bootstrap(per, draws) if v is not None else None
        verdicts[name] = {'off_events': ofc, 'on_events': onc, 'off_amount': ofa,
                          'on_amount': ona, 'ds1': v, 'ci': ci}
        lines.append('%-12s %10d %10d %10s %10s %10s %22s'
                     % (name, ofc, onc, format(int(ofa), ',d'), format(int(ona), ',d'),
                        ('%+.1f%%' % v) if v is not None else '정의 안 됨',
                        ('[%+.1f, %+.1f]' % (ci['lo'], ci['hi'])) if ci else '-'))
    lines.append('')
    lines.append('시민 %d명' % len(data[gate_name]))
    if not gate_ok:
        lines.append('')
        lines.append('분모 미달이므로 DS-1 판정을 적지 않는다. 이 라운드는 지정으로도 '
                     '분모가 만들어지지 않는다는 결과로 기록한다.')
    return lines, {'gate_ok': gate_ok, 'gate_events': off_events, 'by_definition': verdicts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, help='source.json@responses.jsonl')
    ap.add_argument('--min-events', type=int, default=20)
    ap.add_argument('--draws', type=int, default=4000)
    args = ap.parse_args()
    runs = [split(s) for s in args.run]
    for src, resp in runs:
        for p in (src, resp):
            if not Path(p).exists():
                raise SystemExit('없는 파일: %s' % p)
    lines, _ = report(runs, args.min_events, args.draws)
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
