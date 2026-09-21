"""Add one errand that cannot be finished at home, chosen from the citizen's own profile.

After the meal need went in, the simulation bought groceries and ordered delivery - both
of which end at home - and eight of the eleven categories in the citizens' own assigned
mix were still never bought. Nothing in the day required leaving the house.

So this adds one more optional need, and the activity that satisfies it is zone-only:
a haircut, a health item, a class, a leisure service. Which one is not a choice made
here. For each citizen it is the service category with the largest share in their own
assigned spending mix, and a citizen whose profile gives every service category zero
gets no errand at all.

That rule is mechanical and comes from the input, never from an answer key. The measured
studies look at restaurants, cafes, retail, hair-and-beauty, durables and commercial-zone
types; the selection below is blind to all of them, and in this cohort no citizen picks
the hair category because none of them is assigned any share of it.

    python scripts/sim/add_outside_need.py --source action_source_meal.json \
        --out action_source_outside.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from daily_resource_contract import validate

HEADER = '## 정책 적용 전 고정한 오늘의 조건'
MARKER = '\n\n## 오늘\n'

# The only purchase activities with no at-home anchor. An errand met by one of these
# cannot be completed without leaving, which is the whole point.
SERVICE = {'hair': '미용', 'health_goods': '건강',
           'leisure_service': '여가', 'education_service': '교육'}

# Two renderer generations are in use: the older cells say '…부여한 업종별 지출 구성:'
# and the newer ones '평소 업종별 지출 구성(카드 실측):'. Requiring the colon to follow
# the word directly silently matched only the first, which made every distance on the
# newer cohort vacuously zero.
PROFILE_RE = re.compile(r'업종별 지출 구성[^:\n]*:\s*(.+)')
PART_RE = re.compile(r'([가-힣·]+)\s*(\d+)%')
DESCRIPTION = ('오늘 하려고 미뤄 둔 볼일이 하나 있다. 일정과 자금에 따라 다음 날로 미룰 수도 있다.')
EVIDENCE = '실험 가정: 이 볼일은 집에서 끝낼 수 없고 해당 장소에 가야 한다.'


def assigned_service_shares(cell):
    """{activity: share} for the four service categories, read from this citizen's own text."""
    m = PROFILE_RE.search(cell['user'])
    parts = {}
    if m:
        for name, pct in PART_RE.findall(m.group(1).split('/')[0]):
            parts[name] = parts.get(name, 0) + int(pct)
    return {a: parts.get(name, 0) for a, name in SERVICE.items()}


def chosen_errand(cell):
    """The service category this citizen actually spends the most on, or None if none."""
    shares = assigned_service_shares(cell)
    best = max(shares, key=lambda a: (shares[a], a))
    return (best, shares[best]) if shares[best] > 0 else (None, 0)


def rewrite_block(user, conditions):
    start = user.find(HEADER)
    if start < 0:
        raise ValueError('No pre-policy condition block to rewrite')
    end = user.find(MARKER, start)
    if end < 0:
        raise ValueError('Condition block is not delimited by the day marker')
    block = user[start:end]
    brace = block.find('{')
    if brace < 0:
        raise ValueError('Condition block carries no JSON')
    return user[:start] + block[:brace] + json.dumps(conditions, ensure_ascii=False, indent=2) \
        + user[end:]


def add_evidence_line(user):
    start = user.find(HEADER)
    brace = user.find('{', start)
    if start < 0 or brace < 0:
        raise ValueError('No condition block to add evidence to')
    if EVIDENCE in user[start:brace]:
        return user
    return user[:brace] + EVIDENCE + '\n' + user[brace:]


def add(source):
    result = copy.deepcopy(source)
    picks = {}
    for cell in result['cells']:
        cond = cell.get('daily_conditions')
        if not cond:
            raise ValueError('Source has no daily conditions to extend')
        if any(n['id'] == 'errand' for n in cond['needs']):
            raise ValueError('Errand need already present')
        activity, share = chosen_errand(cell)
        picks[cell['aid']] = {'activity': activity, 'assigned_share_percent': share}
        if activity is None:
            continue
        cond['needs'].append({'id': 'errand', 'description': DESCRIPTION,
                              'fulfilled_by': [activity], 'desired_count': 1,
                              'mandatory': False})
        cond['assumptions'].append(
            '이 볼일은 해당 장소에서만 할 수 있다고 가정한다. 실제 개인 일정이나 예약을 나타내지 않는다.')
        validate(cond)
        cell['user'] = add_evidence_line(cell['user'])
        cell['user'] = rewrite_block(cell['user'], cond)
        cell['context_sha256'] = hashlib.sha256(cell['user'].encode('utf-8')).hexdigest()
    result['errand_provenance'] = {
        'kind': 'synthetic_assumption',
        'rule': "For each citizen, the service category with the largest share in that "
                "citizen's own assigned spending mix. Zero share means no errand.",
        'blind_to': 'No answer-key indicator was consulted when choosing.',
        'picks': picks,
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
    print('wrote', args.out)
    print('  sha256', hashlib.sha256(Path(args.out).read_bytes()).hexdigest())
    picks = result['errand_provenance']['picks']
    import collections
    print('  고른 활동:', dict(collections.Counter(
        v['activity'] or '(없음)' for v in picks.values())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
