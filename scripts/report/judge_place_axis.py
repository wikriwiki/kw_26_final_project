"""Apply the pre-registered v4 verdict to the recorded runs. The rule is code, not prose.

The primary outcome is the plan, not the money: does the day put an offline purchase
where a restricted wallet would actually be accepted? In v3 that was 0-2 of 12 cells,
which is why no prompt could move the money.

The verdict was fixed before the run (experiments/v4/prompt_v4.md):

    success   on-arm reachability higher than the control AND the gap exceeds the
              candidate's own seed-to-seed spread
    failure   the gap sits inside that spread - the added sentence did nothing
    worse     the OFF arm rose too. Then the prompt did not read the policy, it just
              sent the person out, and that is spending inducement, not a win.

The last branch is the one that matters. Without it a prompt that pushes everyone
outdoors would score as a success.

    python scripts/report/judge_place_axis.py --control v25=plans_v25/responses.jsonl \
        --candidate v28=plans_v28/responses.jsonl --frozen plans_v25/frozen_inputs.json \
        --out judgement.json
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sim'))
from plan_responsiveness import arm_contrast, distance, coarse, floor_contrast, load, reachability, summarise


def homes(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return {p['id']: p.get('home_dong_code') for p in data['personas']}


def seeds_in(path: str) -> list:
    found = set()
    with io.open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                found.add(json.loads(line).get('replicate'))
    return sorted(s for s in found if s is not None)


def per_seed_reach(path: str, home: dict) -> dict:
    """Reachability for each seed separately; the spread across seeds is the yardstick."""
    out = {}
    for seed in seeds_in(path):
        rows = load(f'{path}@{seed}')
        out[seed] = reachability(rows, home)
    return out


def arm_share(per_seed: dict, arm: str) -> list:
    """One reachability share per seed, for the given arm, pooled over mechanisms."""
    out = []
    for blk in per_seed.values():
        cases = [v for k, v in blk.items() if k != '_all']
        reached = sum(v[arm]['reached'] for v in cases)
        cells = sum(v[arm]['cells'] for v in cases)
        out.append(reached / cells if cells else 0.0)
    return out


def own_floor(path: str) -> dict:
    """Same arm, consecutive seeds: this candidate's own resampling floor.

    v3 could only borrow the control's floor. With several seeds each candidate carries
    its own, so a difference between candidates is no longer read against a stranger.
    """
    seeds = seeds_in(path)
    if len(seeds) < 2:
        return {}
    pairs = {}
    for a, b in zip(seeds, seeds[1:]):
        for mech, vals in floor_contrast(load(f'{path}@{a}'), load(f'{path}@{b}')).items():
            pairs.setdefault(mech, []).extend(vals)
    return summarise(pairs)


def signal(path: str) -> dict:
    seeds = seeds_in(path)
    per = {}
    for seed in seeds:
        for mech, vals in arm_contrast(load(f'{path}@{seed}')).items():
            per.setdefault(mech, []).extend(vals)
    return summarise(per)


def verdict(control: list, candidate: list, control_off: list, candidate_off: list) -> dict:
    c_on, k_on = statistics.mean(control), statistics.mean(candidate)
    c_off, k_off = statistics.mean(control_off), statistics.mean(candidate_off)
    spread = max((statistics.stdev(v) for v in (control, candidate) if len(v) > 1), default=0.0)
    gap_on, gap_off = k_on - c_on, k_off - c_off
    if gap_off > spread and gap_on > spread:
        call = '되레 나쁨 — off 팔도 같이 올랐다. 정책을 읽은 것이 아니라 밖으로 내보낸 것이다'
    elif gap_on > spread:
        call = '성공 — on 팔만 올랐고 그 폭이 seed 변동을 넘는다'
    else:
        call = '실패 — 차이가 seed 변동 안이다. 더한 문장이 일을 하지 않았다'
    return {'control_on': c_on, 'candidate_on': k_on, 'gap_on': gap_on,
            'control_off': c_off, 'candidate_off': k_off, 'gap_off': gap_off,
            'seed_spread': spread, 'verdict': call,
            'rule': 'Fixed in experiments/v4/prompt_v4.md before the run.'}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', required=True, metavar='label=responses.jsonl')
    ap.add_argument('--candidate', required=True, metavar='label=responses.jsonl')
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    home = homes(args.frozen)
    runs = {}
    for item in (args.control, args.candidate):
        label, path = item.split('=', 1)
        runs[label] = {'path': path, 'per_seed_reach': per_seed_reach(path, home),
                       'signal': signal(path), 'own_floor': own_floor(path)}
    (c_label, _), (k_label, _) = [i.split('=', 1) for i in (args.control, args.candidate)]
    result = {
        'primary': verdict(arm_share(runs[c_label]['per_seed_reach'], 'on'),
                           arm_share(runs[k_label]['per_seed_reach'], 'on'),
                           arm_share(runs[c_label]['per_seed_reach'], 'off'),
                           arm_share(runs[k_label]['per_seed_reach'], 'off')),
        'runs': {k: {'per_seed_reach': {str(s): b for s, b in v['per_seed_reach'].items()},
                     'signal': v['signal'], 'own_floor': v['own_floor']} for k, v in runs.items()},
        'note': 'Reaching is necessary, not sufficient. A higher share means the policy can '
                'be measured at all, not that the day moved correctly.',
    }
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(result, ensure_ascii=False, indent=1))
    p = result['primary']
    print(f"도달 비율 on : {c_label} {p['control_on']:.3f} → {k_label} {p['candidate_on']:.3f}  ({p['gap_on']:+.3f})")
    print(f"도달 비율 off: {c_label} {p['control_off']:.3f} → {k_label} {p['candidate_off']:.3f}  ({p['gap_off']:+.3f})")
    print(f"seed 변동 폭 : {p['seed_spread']:.3f}")
    print(f"판정         : {p['verdict']}")
    for label, v in result['runs'].items():
        fl = v['own_floor'].get('_all', {}).get('coarse_distance_mean')
        sg = v['signal']['_all']['coarse_distance_mean']
        if fl is not None:
            print(f"  {label} 동선 신호 {sg:.3f} · 제 영점 {fl:.3f} · 초과 {sg-fl:+.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
