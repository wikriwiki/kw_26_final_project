"""Give the errand the same standing as a work duty: a fixed appointment on the day.

Given the errand as an optional need, the model does it ten times in a hundred
(THE_TEN_PERCENT_CEILING.md). Given a work duty, the same model turns up ninety-two
times in a hundred. The difference is not the activity; it is whether the day says the
thing is settled.

`daily_resource_contract` refuses a mandatory need outright - "this contract tracks
optional needs, not unimplemented mandatory demands" - so mandatory is expressed the way
work already is: an entry in `required_activities` with a time, an anchor and a line of
evidence that appears in the citizen's own text.

What this fixes and what it leaves alone:

    fixed here          that an errand exists, its category, its hour, its place
    left to the model   what to buy there, how much, and - when the policy is on -
                        whether to keep the appointment, shorten it, or drop it

The last one is what DS-1 measures. A fixed appointment is not an instruction to spend.

The hour is chosen mechanically: the first slot that clears any existing duty by an hour
on each side, preferring afternoon. A citizen whose day has no room gets no errand rather
than an impossible one, and that is recorded.

    python scripts/sim/add_fixed_errand.py --source action_source_meal.json \\
        --out action_source_fixed_errand.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_outside_need import PROFILE_RE, PART_RE, SERVICE, chosen_errand

HEADER = '## 정책 적용 전 고정한 오늘의 조건'
# 근무가 있으면 그 구간 밖에서만 가능하다. 통근이 중앙 40분·최대 76분이므로
# 저녁 쪽이 현실적이다. 근무가 없는 사람에게는 오후가 먼저 걸린다.
SLOTS = ['14:00', '15:00', '13:00', '11:00', '10:00', '18:30', '19:00', '19:30', '18:00', '20:00']
EVIDENCE = ('실험 가정: 오늘 %s 에 %s 에서 볼일 하나가 확정되어 그 장소에 간다. '
            '실제 개인 일정을 관측한 값은 아니다. 무엇을 얼마에 살지는 정하지 않는다.')


def minutes(hhmm):
    h, m = hhmm.split(':')
    return int(h) * 60 + int(m)


def free_slot(cell, travel):
    """A time outside every fixed duty, with `travel` minutes of clearance on both sides.

    The first version only checked distance from the endpoints of a presence interval, so
    14:00 passed against a 09:00-17:00 workday - it is five hours from the start and three
    from the end, and squarely inside. Forty per cent of cells then had no feasible day.
    A presence interval is a blocked range, not two points.
    """
    blocked = []
    for row in cell.get('required_presence_intervals') or []:
        blocked.append((minutes(row['start']) - travel, minutes(row['end']) + travel))
    for row in cell.get('required_activities') or []:
        t = minutes(row['time'])
        blocked.append((t - travel, t + travel))
    for slot in SLOTS:
        t = minutes(slot)
        if all(not (lo <= t <= hi) for lo, hi in blocked):
            return slot
    return None


def zone_for(cell):
    zones = cell.get('zones') or []
    return ('zone:' + str(zones[0])) if zones else None


def add(source):
    result = copy.deepcopy(source)
    people = {p['id']: p for p in source['personas']}
    placed, skipped = {}, {}
    for cell in result['cells']:
        activity, share = chosen_errand(cell)
        anchor = zone_for(cell)
        commute = people.get(cell['aid'], {}).get('commute_min') or 0
        slot = free_slot(cell, max(60, int(commute) + 30))
        key = '%s|%s|%s' % (cell['aid'], cell['case'], cell['arm'])
        if not activity or not anchor or not slot:
            skipped[key] = {'activity': activity, 'anchor': anchor, 'slot': slot,
                            'why': 'no category' if not activity else
                                   ('no zone' if not anchor else 'no free hour')}
            continue
        evidence = EVIDENCE % (slot, anchor)
        # 증거 문장은 시민이 실제로 읽는 본문에 있어야 한다. 문법 생성기가 그것을 검사한다.
        start = cell['user'].find(HEADER)
        if start < 0:
            raise ValueError('No pre-policy condition block to add evidence to')
        brace = cell['user'].find('{', start)
        if evidence not in cell['user']:
            cell['user'] = cell['user'][:brace] + evidence + '\n' + cell['user'][brace:]
        cell.setdefault('required_activities', []).append(
            {'time': slot, 'activity_id': activity, 'anchor': anchor, 'evidence': evidence})
        cell['context_sha256'] = hashlib.sha256(cell['user'].encode('utf-8')).hexdigest()
        placed[key] = {'activity': activity, 'time': slot, 'anchor': anchor,
                       'assigned_share_percent': share}
    result['fixed_errand_provenance'] = {
        'kind': 'synthetic_assumption',
        'rule': "The citizen's own largest out-of-home category, at the first hour that "
                "clears every fixed duty by an hour on both sides, in their own zone.",
        'blind_to': 'No answer-key indicator was consulted when choosing category, hour or place.',
        'decides': 'That an errand exists, where and when.',
        'does_not_decide': 'What is bought, how much, or whether the policy changes it.',
        'placed': len(placed), 'skipped': len(skipped),
        'skipped_detail': skipped,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    source = json.loads(Path(args.source).read_text(encoding='utf-8'))
    result = add(source)
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(result, ensure_ascii=False, indent=1))
    p = result['fixed_errand_provenance']
    print('wrote', args.out)
    print('  sha256', hashlib.sha256(Path(args.out).read_bytes()).hexdigest())
    print('  확정된 칸 %d · 못 넣은 칸 %d' % (p['placed'], p['skipped']))
    import collections
    print('  활동:', dict(collections.Counter(
        v['activity'] for v in
        [x for x in _placed(result)])))
    if p['skipped']:
        print('  못 넣은 이유:', dict(collections.Counter(
            v['why'] for v in p['skipped_detail'].values())))
    return 0


def _placed(result):
    for cell in result['cells']:
        for row in cell.get('required_activities') or []:
            if row['activity_id'] in SERVICE:
                yield {'activity': row['activity_id']}


if __name__ == '__main__':
    raise SystemExit(main())
