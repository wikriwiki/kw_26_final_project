"""Add the meal need to an already-prepared action source, leaving the original untouched.

The pre-conditions base file is gone, so the meal arm is built by extending the existing
`action_source.json` rather than re-running `prepare_daily_conditions.py`. The control
arm stays exactly the file that produced every earlier round.

What it adds mirrors the laundry need already in every cell - a resource, an activity
that consumes it, a quote that refills it, and one optional need:

    meal_stock        how many meals at home the citizen has food for
    home_meal         now consumes one
    quote:groceries   refills three, available on the spot
    needs[meals]      may be met at home or away, this cell's own catalog decides which

Nothing here is chosen by looking at a published effect. `home_meal` costs nothing today,
which is why eating at home is free and the day never has a reason to buy food anywhere;
the stock alternates 0/2 by sorted citizen index exactly as the detergent doses do.

    python scripts/sim/add_meal_need.py --source action_source.json \
        --out action_source_meal.json
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
from daily_resource_contract import validate

HEADER = '## 정책 적용 전 고정한 오늘의 조건'
MARKER = '\n\n## 오늘\n'
MEAL_WORDS = ('meal', 'delivery')
ASSUMPTION = ('집에서 드는 한 끼는 집에 있는 식재료 한 끼분을 쓴다고 가정한다. 장보기 한 번은 '
              '이 가정에서 세 끼분이며 같은 자리에서 바로 쓸 수 있다. 실제 식품 용량이나 영양을 '
              '나타내지 않는다.')
DESCRIPTION = '오늘 끼니를 든다. 집에서 들 수도 있고 밖에서 들 수도 있다.'


def meal_options(cell):
    """Every way this cell can have a meal, read from its own activity dictionary."""
    from action_plan_contract import catalog
    found = [aid for aid in catalog(cell)
             if any(w in aid for w in MEAL_WORDS) or aid.endswith('_meal')]
    return sorted(set(found))


def rewrite_block(user: str, conditions: dict) -> str:
    """Put the new conditions back into the planner text the citizen actually sees."""
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


def add(source: dict, stocks: dict, meal_count: int = 2) -> dict:
    result = copy.deepcopy(source)
    ids = sorted({c['aid'] for c in result['cells']})
    if set(stocks) != set(ids):
        raise ValueError('A stock is required for every citizen in the cohort')
    for cell in result['cells']:
        cond = cell.get('daily_conditions')
        if not cond:
            raise ValueError('Source has no daily conditions to extend')
        if 'meal_stock' in cond['resources']:
            raise ValueError('Meal need already present')
        options = meal_options(cell)
        if 'home_meal' not in options:
            raise ValueError('This cell cannot have a meal at home')
        cond['resources']['meal_stock'] = {'unit': '집에서 한 끼',
                                           'opening_quantity': stocks[cell['aid']]}
        cond['activity_consumption']['home_meal'] = {'meal_stock': 1}
        cond['quote_receipts']['quote:groceries'] = {'meal_stock': 3}
        cond['quote_receipt_delay_minutes']['quote:groceries'] = 0
        cond['needs'].append({'id': 'meals', 'description': DESCRIPTION,
                              'fulfilled_by': options, 'desired_count': meal_count,
                              'mandatory': False})
        cond['assumptions'].append(ASSUMPTION)
        validate(cond)
        cell['user'] = rewrite_block(cell['user'], cond)
        cell['context_sha256'] = hashlib.sha256(cell['user'].encode('utf-8')).hexdigest()
    result['meal_need_provenance'] = {
        'kind': 'synthetic_assumption',
        'source': 'Meal stock alternates 0/2 by sorted citizen index, the same rule the '
                  'detergent doses use. Not observed, and not selected by looking at any '
                  'published policy effect.',
        'stocks': stocks, 'meal_count': meal_count,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--meal-count', type=int, default=2)
    args = ap.parse_args()
    source = json.loads(Path(args.source).read_text(encoding='utf-8'))
    ids = sorted({c['aid'] for c in source['cells']})
    stocks = {aid: (0 if i % 2 == 0 else 2) for i, aid in enumerate(ids)}
    result = add(source, stocks, args.meal_count)
    io.open(args.out, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(result, ensure_ascii=False, indent=1))
    raw = Path(args.out).read_bytes()
    print('wrote', args.out)
    print('  sha256', hashlib.sha256(raw).hexdigest())
    print('  재고 0인 시민', sum(1 for v in stocks.values() if v == 0),
          '· 2인 시민', sum(1 for v in stocks.values() if v == 2))
    sample = result['cells'][0]['daily_conditions']['needs'][-1]
    print('  끼니 선택지', sample['fulfilled_by'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
