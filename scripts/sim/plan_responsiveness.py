"""How far does switching the policy on move the plan, against how far resampling moves it?

The consumption amount is a very noisy readout: the person-to-person SD of the paired
difference is about 7,779 KRW, so twelve people give a standard error the same size as
the effect. The plan itself is quieter, and it is the thing a prompt acts on first.

So this measures a different outcome on the same evidence:

    signal  distance between the off plan and the on plan, same seed
    floor   distance between two plans for the SAME arm, different seed

The floor matters. The model is stochastic, so two identical requests already give
different plans. A prompt that merely resamples would show an off/on distance equal to
the floor. Only a distance above the floor is a reading of the institution.

This is a screen, not a verdict. Responding is necessary for a prompt to track any
policy at all, and it is not sufficient: a prompt could move a lot and move wrongly.
Nothing here is scored against a measured effect's sign or size.

    python scripts/sim/plan_responsiveness.py --arm-pairs label=plans.jsonl [...] \
        --floor-pairs label=a.jsonl,b.jsonl --out report.json
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
from collections import defaultdict

KEY = ('aid', 'date', 'case')


def events(record: dict) -> set:
    """A plan as a set of comparable slots. Identity is time + what + where + channel."""
    plan = record.get('execution_plan') or {}
    out = set()
    for ev in plan.get('events', []):
        out.add((ev.get('time'), ev.get('activity_id'), ev.get('anchor'),
                 ev.get('category'), ev.get('purchase_channel')))
    return out


def coarse(record: dict) -> set:
    """Time dropped. A slot moved by half an hour should not read as a different plan.

    The strict key counts a 30-minute shift as a total replacement, which would
    overstate how unstable the planner is. This is the same plan seen more kindly.
    """
    plan = record.get('execution_plan') or {}
    return {(ev.get('activity_id'), ev.get('anchor'), ev.get('purchase_channel'))
            for ev in plan.get('events', [])}


def buying(record: dict) -> set:
    """Only the slots that carry a purchase channel - the part consumption comes from."""
    return {e for e in events(record) if e[4]}


def distance(a: set, b: set) -> float:
    """1 - Jaccard. 0 = identical plans, 1 = nothing in common."""
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


# The rule a restricted wallet actually uses, copied from prepare_purchase_probe.base_quote:
# offline channel, not a bar, and the zone anchor inside the home city (grant) or
# home district (prepaid voucher). If the plan never puts a purchase there, the wallet
# cannot be offered at payment time no matter what the plan stage was told.
WALLET_ZONE_PREFIX = {'grant': 2, 'local_voucher': 5}


def wallet_reachable(record: dict, home_dong_code: str | None) -> bool:
    """Does this plan put an offline purchase where this case's wallet would be accepted?

    Without a home code this falls back to 'an offline purchase at any zone anchor',
    which is a superset - necessary but not sufficient. The caller is told which it got.
    """
    width = WALLET_ZONE_PREFIX.get(record.get('case'))
    for ev in (record.get('execution_plan') or {}).get('events', []):
        anchor = str(ev.get('anchor') or '')
        if not anchor.startswith('zone:') or not ev.get('purchase_channel'):
            continue
        if ev.get('activity_id') == 'bar':
            continue
        if width is None or not home_dong_code:
            return True
        if anchor.removeprefix('zone:')[:width] == str(home_dong_code)[:width]:
            return True
    return False


def reachability(rows: dict, homes: dict | None = None) -> dict:
    """Pre-registered primary outcome: the share of cells whose plan reaches the wallet.

    Measured at 0-2 of 12 in v3, which is why a prompt could not move the money.
    """
    per = {}
    for (aid, date, case, arm), rec in rows.items():
        hit = wallet_reachable(rec, (homes or {}).get(aid))
        block = per.setdefault(case, {}).setdefault(arm, [0, 0])
        block[1] += 1
        block[0] += 1 if hit else 0
    out = {}
    for case, arms in sorted(per.items()):
        out[case] = {arm: {'reached': n, 'cells': d, 'share': n / d if d else None}
                     for arm, (n, d) in sorted(arms.items())}
    tot_n = sum(b[0] for a in per.values() for b in a.values())
    tot_d = sum(b[1] for a in per.values() for b in a.values())
    out['_all'] = {'reached': tot_n, 'cells': tot_d, 'share': tot_n / tot_d if tot_d else None,
                   'rule': 'exact home-zone rule' if homes else 'any zone anchor (superset)'}
    return out


def load(spec: str) -> dict:
    """`path` or `path@seed`. One file may hold several replicates, so the seed selects.

    Mixing replicates in one dict would silently overwrite cells that differ only by
    seed, which is exactly the comparison this module exists to make.
    """
    path, _, seed = spec.partition('@')
    want = int(seed) if seed else None
    rows, seen = {}, set()
    with io.open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('valid') is not True:
                continue
            seen.add(r.get('replicate'))
            if want is not None and r.get('replicate') != want:
                continue
            rows[tuple(r[k] for k in KEY) + (r['arm'],)] = r
    if want is None and len(seen) > 1:
        raise ValueError(f'{path} holds replicates {sorted(seen)}; select one with path@seed')
    if not rows:
        raise ValueError(f'No valid rows for {spec} (file holds {sorted(seen)})')
    return rows


def arm_contrast(rows: dict) -> dict:
    """off vs on within one run: the signal."""
    per_mech = defaultdict(list)
    for key, rec in rows.items():
        aid, date, case, arm = key
        if arm != 'off':
            continue
        on = rows.get((aid, date, case, 'on'))
        if on is None:
            continue
        per_mech[case].append((distance(events(rec), events(on)),
                               distance(buying(rec), buying(on)),
                               distance(coarse(rec), coarse(on))))
    return per_mech


def floor_contrast(a: dict, b: dict) -> dict:
    """Same arm, different seed: how far apart two draws already are."""
    per_mech = defaultdict(list)
    for key, rec in a.items():
        other = b.get(key)
        if other is None:
            continue
        per_mech[key[2]].append((distance(events(rec), events(other)),
                                 distance(buying(rec), buying(other)),
                                 distance(coarse(rec), coarse(other))))
    return per_mech


def summarise(per_mech: dict) -> dict:
    out = {}
    for mech, pairs in sorted(per_mech.items()):
        whole = [p[0] for p in pairs]
        buys = [p[1] for p in pairs]
        crude = [p[2] for p in pairs]
        out[mech] = {
            'pairs': len(pairs),
            'plan_distance_mean': statistics.mean(whole),
            'plan_distance_sd': statistics.stdev(whole) if len(whole) > 1 else None,
            'purchase_slot_distance_mean': statistics.mean(buys),
            'coarse_distance_mean': statistics.mean(crude),
            'coarse_distance_sd': statistics.stdev(crude) if len(crude) > 1 else None,
            'identical_plans': sum(1 for d in whole if d == 0.0),
        }
    allw = [p[0] for pairs in per_mech.values() for p in pairs]
    allc = [p[2] for pairs in per_mech.values() for p in pairs]
    allb = [p[1] for pairs in per_mech.values() for p in pairs]
    out['_all'] = {'pairs': len(allw), 'plan_distance_mean': statistics.mean(allw) if allw else None,
                   'plan_distance_sd': statistics.stdev(allw) if len(allw) > 1 else None,
                   'coarse_distance_mean': statistics.mean(allc) if allc else None,
                   'coarse_distance_sd': statistics.stdev(allc) if len(allc) > 1 else None,
                   'purchase_slot_distance_mean': statistics.mean(allb) if allb else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm-pairs', action='append', default=[], metavar='label=plans.jsonl[@seed]')
    ap.add_argument('--floor-pairs', action='append', default=[], metavar='label=a[@seed],b[@seed]')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    result = {
        'unit': '1 - Jaccard over plan slots (time, activity, anchor, category, channel)',
        'signal_is': 'off vs on within one run',
        'floor_is': 'same arm, two seeds - what resampling alone moves',
        'reading': 'A candidate whose signal does not exceed the floor is not reading the '
                   'institution; it is only resampling. Exceeding the floor is necessary, '
                   'not sufficient - moving is not the same as moving correctly.',
        'candidates': {}, 'floor': {},
    }
    for item in args.arm_pairs:
        label, path = item.split('=', 1)
        result['candidates'][label] = summarise(arm_contrast(load(path)))
    for item in args.floor_pairs:
        label, paths = item.split('=', 1)
        p1, p2 = paths.split(',', 1)
        result['floor'][label] = summarise(floor_contrast(load(p1), load(p2)))

    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(result, ensure_ascii=False, indent=1))
    print('wrote', args.out)
    for label, block in result['candidates'].items():
        a = block['_all']
        print(f"  {label:<10} 엄격 {a['plan_distance_mean']:.3f}  시각둔감 {a['coarse_distance_mean']:.3f}"
              f"  구매칸 {a['purchase_slot_distance_mean']:.3f}  (쌍 {a['pairs']})")
    for label, block in result['floor'].items():
        a = block['_all']
        print(f"  영점 {label:<5} 엄격 {a['plan_distance_mean']:.3f}  시각둔감 {a['coarse_distance_mean']:.3f}"
              f"  구매칸 {a['purchase_slot_distance_mean']:.3f}  (쌍 {a['pairs']})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
