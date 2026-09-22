"""Finite, frozen single-citizen financial feasibility experiment; no DB writes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from urllib.request import Request, urlopen
from asset_transaction_contract import schema, inspect
from prompts.asset_transaction_v1 import SYSTEM_PROMPT
from validate_prompt_v3 import atomic, digest


def make_cases():
    def offer(): return {'O': {'wallet_id': 'V', 'unit_face': 10000, 'unit_cash_cost': 9000, 'max_units': 1}}
    def event(eid, price, wallets=(), channel='offline'):
        return {'id': eid, 'intent': '입력의 물품 한 묶음 구매 검토', 'channel': channel,
                'candidates': [{'id': 'quote:' + eid, 'price_won': price, 'eligible_wallets': list(wallets)}]}
    def case(cid, text, cash, events, lots=None, offers=None, required=None):
        return {'id': cid, 'context': text, 'cash': cash, 'wallet_lots': lots or {}, 'offers': offers or {},
                'events': events, 'requirements': required or {}}
    return [
        case('acquire_then_buy', '오늘 필요한 식료품 한 묶음만 사기로 확정했다. 아래 가격이고 추가 구매는 없다. 사용 가능한 자금을 실제로 확보해서 결제한다.',
             9000, [event('food', 10000, ['V'])], offers=offer(), required={'total_consumption': 10000, 'cash_outflow_total': 9000}),
        case('asset_only', '오늘 상품·서비스 구매는 없다. 표시된 선불 잔액 한 단위를 지금 구입하기로 확정했다. 나중에 사용할 예정이다.',
             9000, [], offers=offer(), required={'total_consumption': 0, 'asset_acquisition_cash_outflow': 9000}),
        case('future_unavailable', '현재 쓸 수 있는 자금은 없다. 별도 돈은 내일 받지만 오늘 사용할 수 없고 외상·차입도 없다. 구매는 미룰 수 있다.',
             0, [event('food', 3000)], required={'total_consumption': 0, 'cash_outflow_total': 0}),
        case('ineligible', '지갑 잔액이 있지만 이 거래에 사용할 수 없다. 다른 자금·외상·차입은 없다. 구매 희망은 연기할 수 있다.',
             0, [event('food', 3000)], lots={'V': [{'face': 10000, 'own_basis': 0}]}, required={'total_consumption': 0}),
        case('mixed_channels', '오늘 필요한 물품 두 묶음을 아래 각각의 상점에서 한 묶음씩 구매하기로 확정했다. 추가 구매는 없고 외상·차입은 없다.',
             5000, [event('local', 5000, ['V']), event('web', 5000, (), 'online')],
             lots={'V': [{'face': 5000, 'own_basis': 0}]}, required={'total_consumption': 10000, 'offline_consumption': 5000, 'online_consumption': 5000}),
        case('free_activity', '오늘 공원에서 산책만 하고 돌아온다. 구매나 선불 잔액 취득은 하지 않기로 정했다. 아래 상점이 있다는 사실만 제공된다.',
             20000, [{'id': 'walk', 'intent': '무료 산책', 'channel': 'offline', 'candidates': [{'id': 'cafe', 'price_won': 4000, 'eligible_wallets': ['V']}]}],
             offers=offer(), required={'total_consumption': 0, 'cash_outflow_total': 0}),
        case('shared_balance', '오늘 필요한 두 품목을 아래 상점에서 한 개씩 구매하기로 확정했다. 각 후보는 해당 품목 한 개다. 추가 구매는 없다.',
             0, [event('a', 2000, ['V']), event('b', 2000, ['V'])], lots={'V': [{'face': 4000, 'own_basis': 2000}]},
             required={'total_consumption': 4000, 'own_funded_consumption': 2000, 'concession_funded_consumption': 2000}),
        case('partial_redemption', '오늘 필요한 한 묶음만 표시 가격으로 구매하기로 확정했다. 현재 현금은 없으며 추가 구매는 없다.',
             0, [event('food', 4000, ['V'])], lots={'V': [{'face': 10000, 'own_basis': 9000}]},
             required={'total_consumption': 4000, 'own_funded_consumption': 3600, 'cash_outflow_total': 0}),
        case('insufficient_acquisition', '현재 현금은 아래와 같고 다른 돈은 없다. 구매 희망은 오늘로 확정된 필요가 아니며 미룰 수 있다. 외상·차입 불가.',
             8000, [event('food', 10000, ['V'])], offers=offer(), required={'total_consumption': 0, 'asset_acquisition_cash_outflow': 0}),
    ]


def invoke(job, config, base, folder):
    seed, case = job; key = digest([seed, case['id']]); started = time.monotonic()
    payload = {'model': config['model'], 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
               {'role': 'user', 'content': json.dumps({k: v for k, v in case.items() if k != 'requirements'}, ensure_ascii=False)}],
               **config['sampling'], 'max_tokens': config['max_tokens'],
               'seed': int(key[:8], 16) % 2147483647, 'chat_template_kwargs': {'enable_thinking': True},
               'response_format': {'type': 'json_schema', 'json_schema': {'name': 'asset_choices', 'strict': True, 'schema': schema(case)}}}
    atomic(folder/'attempts'/f'{key}_request.json', payload)
    row = {'case': case['id'], 'replicate': seed, 'attempt_key': key, 'request_sha256': digest(payload)}
    try:
        req = Request(base + '/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=config['timeout_seconds']) as response: response = json.load(response)
        atomic(folder/'attempts'/f'{key}_response.json', response)
        choice = response['choices'][0]; raw = choice['message'].get('content') or ''
        row.update(raw=raw, response=response)
        obj, ledger = inspect(raw, case)
        errors = ['fact:' + field for field, value in case['requirements'].items() if ledger[field] != value]
        if choice['finish_reason'] != 'stop': errors.append('incomplete_generation')
        row.update(valid=not errors, errors=errors, ledger=ledger)
    except Exception as exc: row.update(valid=False, errors=[str(exc)])
    row['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); config = json.loads(Path(args.config).read_text(encoding='utf-8')); cases = make_cases()
    folder = Path(args.out); folder.mkdir(parents=True, exist_ok=False); (folder/'attempts').mkdir(); (folder/'code').mkdir()
    names = ['validate_asset_transactions.py', 'asset_transaction_contract.py', 'asset_ledger.py', 'transaction_ledger.py']
    code = {}
    for name in names:
        raw = Path(__file__).with_name(name).read_bytes(); code[name] = hashlib.sha256(raw).hexdigest()
        (folder/'code'/name).write_bytes(raw)
    atomic(folder/'manifest.json', {'config': config, 'config_sha256': digest(config), 'cases_sha256': digest(cases),
                                  'system_sha256': digest(SYSTEM_PROMPT), 'code_sha256': code, 'registered_at': datetime.now(timezone.utc).isoformat()})
    atomic(folder/'frozen_cases.json', cases); (folder/'system.txt').write_text(SYSTEM_PROMPT, encoding='utf-8')
    base = os.environ.get('LLM_BASE_URL', 'http://localhost:8000/v1').rstrip('/')
    with urlopen(base + '/models', timeout=10) as response: assert config['model'] in [m['id'] for m in json.load(response)['data']]
    jobs = [(seed, case) for seed in config['seeds'] for case in cases]; random.Random(config['order_seed']).shuffle(jobs); rows = []
    with (folder/'responses.jsonl').open('x', encoding='utf-8') as fp, ThreadPoolExecutor(max_workers=config['workers']) as pool:
        pending = [pool.submit(invoke, job, config, base, folder) for job in jobs]
        for future in as_completed(pending):
            row = future.result(); rows.append(row); fp.write(json.dumps(row, ensure_ascii=False) + '\n'); fp.flush(); os.fsync(fp.fileno())
            print(f"completed {len(rows)}/{len(jobs)} {row['case']} valid={row['valid']} errors={row['errors']}", flush=True)
    expected = {(seed, c['id']) for seed in config['seeds'] for c in cases}
    complete = len(rows) == len(expected) and {(r['replicate'], r['case']) for r in rows} == expected
    summary = {'scope': 'Synthetic financial/factual feasibility, not policy-effect validation.', 'macro_claim': False,
               'variants': {'asset_transaction_v1': {'responses': len(rows), 'complete': complete, 'valid': sum(r['valid'] for r in rows),
               'all_pass': complete and all(r['valid'] for r in rows)}}}
    atomic(folder/'summary.json', summary); print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
