"""Freeze a hypothetical quote experiment on previously generated citizen plans.

All prices, shop participation and payment eligibility below are explicit scenario
assumptions. This provider is NOT a reconstruction of historical merchant rules.
It never reads a desired policy effect, nor uses a daily spending anchor as price.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from activity_purchase_bridge import prepare_case
from action_plan_contract import catalog
from validate_prompt_v3 import atomic

# Identical items/prices in every arm and district. They are NOT observed prices.
BOOK = {
    'home_delivery': ('배달 식사 한 끼, 배송비 포함', 13000),
    'office_delivery': ('배달 식사 한 끼, 배송비 포함', 13000),
    'home_online_goods': ('세탁 세제 한 통, 배송비 포함', 12000),
    'meal_dine_in': ('음식점 식사 한 끼', 10000),
    'meal_takeaway': ('포장 식사 한 끼', 9000),
    'cafe_dine_in': ('매장 커피 한 잔', 4000),
    'cafe_takeaway': ('포장 커피 한 잔', 4000),
    'dessert': ('빵 한 개', 3500),
    'groceries': ('쌀 1kg 한 봉지', 5000),
    'convenience': ('생수 500ml 한 병', 1000),
    'shopping': ('면 양말 한 켤레', 4000),
    'hair': ('성인 기본 커트 1회', 20000),
    'health_goods': ('일회용 마스크 5매 한 묶음', 2500),
    'leisure_service': ('영화 관람권 한 장', 12000),
    'education_service': ('자율학습 공간 1시간 이용권', 3000),
    'bar': ('주점 주류 한 잔', 6000),
    'other_service': ('복사 10장 서비스', 1000),
}


def build_case(row, cell, persona, *, price_factor=1):
    if not row.get('eligible'): raise ValueError('Failed schedule must not be omitted or zero-filled')
    if price_factor not in {0.5, 1, 2}: raise ValueError('Unregistered price factor')
    cell = copy.deepcopy(cell)
    # Planner-specific output instructions do not belong in a purchase request.
    cell['user'] = cell['user'].split('\n\n## 오늘\n')[0]
    state = cell['synthetic_state']; mechanism = cell['case']; arm = cell['arm']
    lots = {}; offers = {}
    if arm == 'on' and mechanism == 'grant':
        lots = {pid: [{'face': value, 'own_basis': 0}] for pid, value in state['grant_remaining'].items()}
    if arm == 'on' and mechanism == 'local_voucher':
        offers = {'synthetic_unit': {'wallet_id': 'P014', 'unit_face': 10000,
                  'unit_cash_cost': 9000, 'max_units': 50}}
    specs = catalog(cell); quotes = {}
    for index, event in enumerate(row['execution_plan']['events']):
        aid = event['activity_id']; spec = specs[aid]; channel = spec['purchase_channel']
        if channel is None: continue
        if aid not in BOOK: raise ValueError('No registered hypothetical price for activity')
        # A closed establishment has no quote, not a paid substitute elsewhere.
        closed = (aid == 'bar' and '집합금지:' in cell['user']) or (
            aid in {'meal_dine_in','cafe_dine_in'} and event['time'] >= '22:00'
            and '매장 취식 22:00까지' in cell['user'])
        if closed:
            quotes[str(index)] = []; continue
        description, price = BOOK[aid]; eligible = []
        zone = event['anchor'].removeprefix('zone:')
        home = str(persona['home_dong_code'])
        if channel == 'offline' and aid != 'bar' and arm == 'on':
            if mechanism == 'grant' and zone[:2] == home[:2]: eligible = list(lots)
            if mechanism == 'local_voucher' and zone[:5] == home[:5]: eligible = ['P014']
        quotes[str(index)] = [{'id': 'quote:' + aid, 'description': description,
            'price_won': int(price * price_factor), 'channel': channel, 'eligible_wallets': eligible,
            'price_provenance': 'Hypothetical fixed item quote, not observed merchant or historical price.',
            'eligibility_provenance': 'Synthetic participating independent shop; grant within home city, prepaid within home district; offline only. Not certified historical eligibility.'}]
    case, audit = prepare_case(raw_plan=row['raw'], cell=cell, quotes_by_event=quotes,
        cash=state['balance'], wallet_lots=lots, offers=offers)
    case.update(id=row['attempt_key'], date=cell['date'])
    case['scenario_assumptions'] = [
        '이 실험의 후보는 가상의 독립 가맹점 상품이다. 가격은 관측값이 아니며 같은 품목 가격은 모든 조건에서 같다.',
        'eligible_wallets는 이 합성 거래에서 확정해 제공한 사용 자격이다. 실제 역사적 가맹점 자격을 주장하지 않는다.',
        '선불 잔액 취득 단위는 실험에서 1만원, 기존 월 구매는 없음으로 가정한다. 실제 판매 단위의 검증값이 아니다.',
        '주어진 제도가 있다면 참여 자격을 만족한다고 가정한다. 나이 세부값·본인 카드 이력은 실제로 관측되지 않았다.',
        '식재료·생활용품 재고와 현재 질병은 제공되지 않았다. 구매 검토만으로 부족·질병·구매 확정을 새 사실로 만들지 않는다.',
        '오늘의 후보 구매 여부를 고른다. 없는 상품·추가 수량·외상·차입은 선택할 수 없다.']
    return case, audit


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    if args.out.exists(): raise ValueError('Refusing overwrite')
    frozen = json.loads((args.run/'frozen_inputs.json').read_bytes())
    rows = [json.loads(line) for line in (args.run/'responses.jsonl').read_bytes().splitlines()]
    cells = {(c['aid'],c['case'],c['arm']): c for c in frozen['cells']}
    people = {p['id']: p for p in frozen['personas']}
    expected = set(cells); seen = set(); prepared = []
    for row in sorted(rows, key=lambda r: (r['aid'],r['case'],r['arm'])):
        key = (row['aid'],row['case'],row['arm'])
        if key in seen or key not in expected: raise ValueError('Source must have one frozen plan per cell')
        seen.add(key); case, audit = build_case(row, cells[key], people[row['aid']])
        prepared.append({k: row[k] for k in ['aid','case','arm','date','attempt_key']} | {'transaction_case': case, 'bridge_audit': audit})
    if seen != expected: raise ValueError('Incomplete source matrix')
    provenance = {name: hashlib.sha256((args.run/name).read_bytes()).hexdigest() for name in ['frozen_inputs.json','responses.jsonl','manifest.json']}
    atomic(args.out, {'cells': prepared, 'quote_book': BOOK, 'source_sha256': provenance,
        'provider_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Development end-to-end purchase probe with hypothetical prices/eligibility; fixed v18 plans, no actual market or empirical effect validation.'})
    print(json.dumps({'cells':len(prepared),'sha256':hashlib.sha256(args.out.read_bytes()).hexdigest()}))


if __name__ == '__main__': main()
