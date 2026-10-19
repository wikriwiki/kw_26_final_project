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

# Purchase activities with no at-home anchor. An errand met by one of these cannot be
# completed without leaving, which is the whole point.
#
# 2026-09-22: the first version listed only four service categories, which left 식사·마트·
# 쇼핑 out for no stated reason - an arbitrary restriction I wrote, not one the data asked
# for. Eight of sixty citizens then got no errand at all because their mix had no service
# share. Removing the restriction gives every citizen the category they actually spend most
# on outside the house.
#
# The timing is disclosed in experiments/v27: this change was made AFTER learning that DS-1
# has no dine-in denominator, and it does supply one. What it does not do is decide how much
# of that dine-in collapses when the policy is on - that is the model's answer, and it is
# what the round tests.
SERVICE = {'hair': '미용', 'health_goods': '건강',
           'leisure_service': '여가', 'education_service': '교육',
           'meal_dine_in': '식사', 'groceries': '마트', 'convenience': '편의점',
           'cafe_dine_in': '디저트', 'other_service': '기타', 'bar': '주점'}

# Two renderer generations are in use: the older cells say '…부여한 업종별 지출 구성:'
# and the newer ones '평소 업종별 지출 구성(카드 실측):'. Requiring the colon to follow
# the word directly silently matched only the first, which made every distance on the
# newer cohort vacuously zero.
PROFILE_RE = re.compile(r'업종별 지출 구성[^:\n]*:\s*(.+)')
PART_RE = re.compile(r'([가-힣·]+)\s*(\d+)%')
DESCRIPTION = ('오늘 하려고 미뤄 둔 볼일이 하나 있다. 일정과 자금에 따라 다음 날로 미룰 수도 있다.')
# 의무판. 근무 일정과 같은 층위의 사실로 적는다 — 무엇을 사라거나 얼마를 쓰라는 말은
# 없고, 정책이 켜졌을 때 어떻게 할지도 말하지 않는다.
DESCRIPTION_MANDATORY = ('오늘 해야 하는 볼일이 하나 있다. 다음 날로 미룰 수 없다.')
EVIDENCE = '실험 가정: 이 볼일은 집에서 끝낼 수 없고 해당 장소에 가야 한다.'


def assigned_service_shares(cell):
    """{activity: share} for the four service categories, read from this citizen's own text."""
    m = PROFILE_RE.search(cell['user'])
    parts = {}
    if m:
        for name, pct in PART_RE.findall(m.group(1).split('/')[0]):
            parts[name] = parts.get(name, 0) + int(pct)
    return {a: parts.get(name, 0) for a, name in SERVICE.items()}


def chosen_errand(cell, forced=None):
    """The service category this citizen actually spends the most on, or None if none.

    `forced` overrides the rule and gives every citizen the same category. That is a
    louder assumption than the own-largest rule, so it is recorded as one: the citizen's
    own share of the forced category is still reported, including when it is zero. It is
    given identically to both arms, so it cannot create a direction - only a denominator.
    """
    shares = assigned_service_shares(cell)
    if forced:
        return (forced, shares.get(forced, 0))
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


def add(source, mandatory=False, forced=None):
    result = copy.deepcopy(source)
    picks = {}
    for cell in result['cells']:
        cond = cell.get('daily_conditions')
        if not cond:
            raise ValueError('Source has no daily conditions to extend')
        if any(n['id'] == 'errand' for n in cond['needs']):
            raise ValueError('Errand need already present')
        activity, share = chosen_errand(cell, forced)
        picks[cell['aid']] = {'activity': activity, 'assigned_share_percent': share}
        if activity is None:
            continue
        # 2026-09-22: 선택으로 주면 열에 한 번만 한다(THE_TEN_PERCENT_CEILING.md).
        # 같은 모델이 mandatory 인 근무는 92% 한다. 의무로 주는 것은 근무·통근·재고를
        # 지정하는 것과 같은 층위이고, 정책이 켜졌을 때 얼마나 줄어드는지는 여전히
        # 모델이 정한다.
        cond['needs'].append({'id': 'errand',
                              'description': DESCRIPTION_MANDATORY if mandatory else DESCRIPTION,
                              'fulfilled_by': [activity], 'desired_count': 1,
                              'mandatory': bool(mandatory)})
        cond['assumptions'].append(
            '이 볼일은 해당 장소에서만 할 수 있다고 가정한다. 실제 개인 일정이나 예약을 나타내지 않는다.')
        validate(cond)
        cell['user'] = add_evidence_line(cell['user'])
        cell['user'] = rewrite_block(cell['user'], cond)
        cell['context_sha256'] = hashlib.sha256(cell['user'].encode('utf-8')).hexdigest()
    result['errand_provenance'] = {
        'kind': 'synthetic_assumption',
        'rule': ("Every citizen gets the same category: %s. Assigned uniformly, not read "
                 "from the citizen's own mix." % forced) if forced else
                ("For each citizen, the service category with the largest share in that "
                 "citizen's own assigned spending mix. Zero share means no errand."),
        'forced_activity': forced,
        'blind_to': 'No answer-key indicator was consulted when choosing.',
        'both_arms_identical': True,
        'picks': picks,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mandatory', action='store_true',
                    help='볼일을 의무로 준다 (근무 일정과 같은 층위)')
    ap.add_argument('--activity', choices=sorted(SERVICE),
                    help='전원에게 같은 업종을 준다. 규칙이 아니라 지정임을 출처에 적는다.')
    args = ap.parse_args()
    source = json.loads(Path(args.source).read_text(encoding='utf-8'))
    result = add(source, mandatory=args.mandatory, forced=args.activity)
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(result, ensure_ascii=False, indent=1))
    print('wrote', args.out)
    print('  sha256', hashlib.sha256(Path(args.out).read_bytes()).hexdigest())
    picks = result['errand_provenance']['picks']
    import collections
    print('  고른 활동:', dict(collections.Counter(
        v['activity'] or '(없음)' for v in picks.values())))
    if args.activity:
        z = sum(1 for v in picks.values() if not v['assigned_share_percent'])
        print('  지정판 - 본인 지출에 그 업종이 0%%인 시민 %d / %d' % (z, len(picks)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
