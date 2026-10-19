"""Judge v5, where only some cells could have changed at all.

Surfacing the wallet rule rewrites 24 of 96 cells - grant-on and local_voucher-on.
The other 72 are byte-identical between the arms, because cashback and distancing
carry no wallet and the off arms carry no policy. Those 72 are a placebo that the
design guarantees rather than hopes for.

So the verdict needs both halves:

    success   the changed cells rise beyond the seed spread AND the untouched cells
              do not move comparably
    failure   the changed cells stay inside the spread
    void      both halves move. Then it is not the placement; it is noise or a leak,
              and reporting the primary alone would be reporting a coincidence.

A leak is worth catching on its own: if a byte-identical cell moves, either the two
arms were not actually identical there or the run is noisier than the seed spread says.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sim'))
from plan_responsiveness import load, reachability

CHANGED = ('grant', 'local_voucher')


def seeds_in(path: str) -> list:
    found = set()
    with io.open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                found.add(json.loads(line).get('replicate'))
    return sorted(s for s in found if s is not None)


def shares(path: str, homes: dict, cases, arm: str) -> list:
    """One share per seed over the named cases and arm."""
    out = []
    for seed in seeds_in(path):
        blk = reachability(load(f'{path}@{seed}'), homes)
        picked = [v for k, v in blk.items() if k != '_all' and k in cases]
        n = sum(v[arm]['reached'] for v in picked)
        d = sum(v[arm]['cells'] for v in picked)
        out.append(n / d if d else 0.0)
    return out


def compare(control: list, treat: list) -> dict:
    spread = max((statistics.stdev(v) for v in (control, treat) if len(v) > 1), default=0.0)
    c, t = statistics.mean(control), statistics.mean(treat)
    return {'control': c, 'treatment': t, 'gap': t - c, 'seed_spread': spread,
            'moved': (t - c) > spread}


def decide(primary: dict, placebo: dict) -> str:
    if primary['moved'] and placebo['moved']:
        return ('무효 — 글자 하나 다르지 않은 칸도 같이 움직였다. 자리의 효과가 아니라 '
                '잡음이거나 두 팔이 실제로는 같지 않았다는 뜻이다')
    if primary['moved']:
        return '성공 — 바뀐 칸만 올랐고 그 폭이 seed 변동을 넘는다'
    if placebo['moved']:
        return ('실패 — 바뀐 칸은 그대로인데 안 바뀐 칸이 움직였다. 이 축의 효과가 아니다')
    return '실패 — 차이가 seed 변동 안이다. 자리를 바꿔도 안 문다'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', required=True, metavar='label=responses.jsonl')
    ap.add_argument('--treatment', required=True, metavar='label=responses.jsonl')
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    homes = {p['id']: p.get('home_dong_code')
             for p in json.loads(Path(args.frozen).read_text(encoding='utf-8'))['personas']}
    (_, cpath), (_, tpath) = [i.split('=', 1) for i in (args.control, args.treatment)]
    all_cases = set(reachability(load(f'{cpath}@{seeds_in(cpath)[0]}'), homes)) - {'_all'}
    untouched = tuple(sorted(all_cases - set(CHANGED)))

    primary = compare(shares(cpath, homes, CHANGED, 'on'), shares(tpath, homes, CHANGED, 'on'))
    placebo = compare(shares(cpath, homes, untouched, 'on'), shares(tpath, homes, untouched, 'on'))
    off = compare(shares(cpath, homes, all_cases, 'off'), shares(tpath, homes, all_cases, 'off'))

    result = {'changed_cases': list(CHANGED), 'untouched_cases': list(untouched),
              'primary_changed_cells_on': primary, 'placebo_untouched_cells_on': placebo,
              'all_off_arms': off, 'verdict': decide(primary, placebo),
              'rule': 'Fixed in experiments/v5/prompt_v5.md before the run.',
              'note': 'Reaching is necessary, not sufficient.'}
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(result, ensure_ascii=False, indent=1))
    for label, blk in (('바뀐 칸 (지원금·지역화폐 on)', primary),
                       ('위약 (캐시백·거리두기 on)', placebo),
                       ('off 팔 전체', off)):
        print(f"{label:<28} {blk['control']:.3f} → {blk['treatment']:.3f}  "
              f"({blk['gap']:+.3f}, 변동폭 {blk['seed_spread']:.3f})")
    print('판정:', result['verdict'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
