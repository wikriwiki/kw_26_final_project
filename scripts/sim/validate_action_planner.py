"""Registered typed-action experiment. Raw decisions retained; no policy effect score."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from urllib.request import urlopen
from action_plan_contract import catalog, schema, inspect
from bounded_reasoning import post, run
from validate_prompt_v3 import atomic, digest


def invoke(job, config, base, prefixes, folder):
    candidate, seed, cell = job; key = digest([candidate['id'], seed, cell['aid'], cell['case'], cell['arm']])
    row = {k: cell[k] for k in ['aid','case','arm','date']}
    row.update(variant=candidate['id'], replicate=seed, attempt_key=key); started = time.monotonic()
    try:
        sc = schema(cell); sample_seed = int(digest([seed, cell['aid'], cell['case']])[:8], 16) % 2147483647
        prefix = prefixes[(candidate['id'], cell['aid'], cell['case'], cell['arm'])]
        grammar=None
        if config.get('temporal_clock_step') is not None:
            from temporal_choice_grammar import build
            if config['max_shift_minutes'] != 0: raise ValueError('Finite clock protocol forbids post-generation time shifts')
            grammar,audit=build(cell,clock_step=config['temporal_clock_step'],last_start_not_before=config.get('last_start_not_before'),allow_zone_commitments=config.get('allow_zone_commitments',False))
            row.update(temporal_grammar_sha256=digest(grammar),temporal_grammar_audit=audit)
            atomic(folder/'attempts'/f'{key}_grammar.json',{'ebnf':grammar,'audit':audit})
        if candidate['thinking_tokens']:
            first, second = run(prefix=prefix, schema=sc, base=base, seed=sample_seed,
                thinking_tokens=candidate['thinking_tokens'], answer_tokens=config['answer_tokens'], sampling=config['sampling'],
                timeout=config['timeout_seconds'], whitespace_limit=2,ebnf=grammar,
                on_request=lambda stage,value:atomic(folder/'attempts'/f'{key}_{stage}_request.json',value),
                on_deliberation=lambda value: atomic(folder/'attempts'/f'{key}_deliberation.json', value))
            row.update(thinking_usage=first['response']['meta_info'], forced_reasoning_boundary=first['forced_reasoning_boundary'])
        else:
            import xgrammar
            request = {'text': prefix, 'sampling_params': dict(config['sampling'], sampling_seed=sample_seed,
                       max_new_tokens=config['answer_tokens'], ebnf=grammar or str(xgrammar.Grammar.from_json_schema(sc, max_whitespace_cnt=2))),
                       'require_reasoning': False, 'stream': False}
            atomic(folder/'attempts'/f'{key}_request.json', request)
            second = {'request': request, 'response': post(base, request, config['timeout_seconds'])}
        atomic(folder/'attempts'/f'{key}_answer.json', second)
        response = second['response']; row.update(raw=response['text'], answer_usage=response['meta_info'], schema_sha256=digest(sc))
        report = inspect(response['text'], cell, max_shift=config['max_shift_minutes'])
        if response['meta_info']['finish_reason']['type'] != 'stop': report['errors'].append('incomplete_generation'); report['valid'] = False
        # These requirements stay outside the request. They test supplied facts,
        # not empirical policy effect signs or magnitudes.
        executable = (report['execution_plan'] or {}).get('events', [])
        if config.get('last_start_not_before') and (not executable or executable[-1]['time']<config['last_start_not_before']):
            report['errors'].append('registered_coverage_boundary');report['valid']=False
        failures = []
        for requirement in cell.get('evaluation_requirements', []):
            if requirement['kind'] not in {'no_outside','forbid_activity','forbid_after'}: raise ValueError('Unknown evaluation requirement')
            if requirement['kind'] == 'no_outside' and any(e['anchor'] != 'residence' for e in executable): failures.append('outside_prohibited')
            elif requirement['kind'] == 'forbid_activity' and any(e['activity_id'] in requirement['ids'] for e in executable): failures.append('closed_activity')
            elif requirement['kind'] == 'forbid_after' and any(e['activity_id'] in requirement['ids'] and e['time'] >= requirement['time'] for e in executable): failures.append('closed_activity_time')
        report['factual_errors'] = failures
        report['eligible'] = report['valid'] and not failures
        row.update(report)
    except Exception as exc: row.update(valid=False, eligible=False, errors=['request_or_contract'], error=str(exc))
    row['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--tokenizer', required=True)
    args = ap.parse_args(); config = json.loads(Path(args.config).read_text(encoding='utf-8')); raw = Path(args.source).read_bytes()
    import importlib
    if config.get('prompt_module','v22') not in {'v22','v23','v24','v25','v26','v27','v28','v29','v30','v31','v32','v33','v34','v35'}: raise ValueError('Unregistered prompt module')
    system_prompt = importlib.import_module('prompts.' + config.get('prompt_module','v22')).SYSTEM_PROMPT
    assert hashlib.sha256(raw).hexdigest() == config['source_inputs_sha256']
    inputs = json.loads(raw); people = {p['id']: p for p in inputs['personas']}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=True)
    prefixes = {}; frozen = []
    for cell in inputs['cells']:
        cell['has_work'] = bool(people[cell['aid']].get('work_poi_id'))
        user = cell['user'].replace('/no_think', '').replace('/think', '')
        # v5: state the wallet's usage rule where the policy's other facts already live.
        # Off by default, so the control and the treatment share every other byte.
        if config.get('surface_wallet_acceptance'):
            from surface_acceptance import surface
            user = surface(user, cell.get('purchase_preview'))
        user += '\n\n## 선택 가능한 활동 사전\n' + json.dumps(list(catalog(cell).values()), ensure_ascii=False)
        if cell.get('required_activities'):
            user += '\n\n## 입력에 명시된 일정의 실행 표기\n' + json.dumps(cell['required_activities'], ensure_ascii=False)
        if config.get('temporal_clock_step') is not None:
            if cell.get('required_presence_intervals'):
                user += '\n\n## 입력에 명시된 장소 유지 구간\n' + json.dumps(cell['required_presence_intervals'],ensure_ascii=False)
            user += '\n\n시각은 '+str(config['temporal_clock_step'])+'분 단위 또는 위에 명시된 고정 일정 시각 중에서 고른다. 주어진 일정과 이동 시간을 지키며 저녁과 하루 마무리까지 선택한다.'
            if config.get('last_start_not_before'):
                user += '\n이 실험의 일과 표현 범위는 '+config['last_start_not_before']+' 이후의 집에서의 마무리 활동까지다. 남은 항목 수 안에 하루 뒤쪽 활동도 표현한다. 이를 위해 구매나 외출을 추가할 필요는 없다. 실제 개인의 취침 시각을 관측했다는 뜻은 아니다.'
        frozen.append(dict(cell, submitted_user=user))
        for c in config['candidates']:
            prefixes[(c['id'], cell['aid'], cell['case'], cell['arm'])] = tokenizer.apply_chat_template(
                [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=bool(c['thinking_tokens']))
            if c.get('deliberation_prefill'):
                if not c['thinking_tokens']:raise ValueError('Reasoning prefill requires explicit reasoning stage')
                prefixes[(c['id'], cell['aid'], cell['case'], cell['arm'])]+=c['deliberation_prefill']
    folder = Path(args.out); folder.mkdir(parents=True, exist_ok=False); (folder/'attempts').mkdir(); (folder/'code').mkdir()
    names = ['validate_action_planner.py','action_plan_contract.py','presence_contract.py','bounded_reasoning.py','temporal_projection.py']
    if config.get('temporal_clock_step') is not None: names.append('temporal_choice_grammar.py')
    code = {}
    for name in names:
        data = Path(__file__).with_name(name).read_bytes(); code[name] = hashlib.sha256(data).hexdigest(); (folder/'code'/name).write_bytes(data)
    atomic(folder/'manifest.json', {'config': config, 'config_sha256': digest(config), 'inputs_sha256': hashlib.sha256(raw).hexdigest(),
           'system_sha256': digest(system_prompt), 'code_sha256': code, 'registered_at': datetime.now(timezone.utc).isoformat(),
           'prefix_sha256': {'|'.join(k): digest(v) for k,v in prefixes.items()}, 'template_sha256': digest(tokenizer.chat_template)})
    atomic(folder/'frozen_inputs.json', {'personas': inputs['personas'], 'cells': frozen}); (folder/'system.txt').write_text(system_prompt, encoding='utf-8')
    base = os.environ.get('LLM_BASE_URL', 'http://localhost:8000/v1').rstrip('/').removesuffix('/v1')
    with urlopen(base + '/v1/models', timeout=10) as response: assert config['model'] in [m['id'] for m in json.load(response)['data']]
    jobs = [(c, seed, cell) for c in config['candidates'] for seed in config['seeds'] for cell in inputs['cells']]
    random.Random(config['order_seed']).shuffle(jobs); rows = []
    with (folder/'responses.jsonl').open('x', encoding='utf-8') as fp, ThreadPoolExecutor(max_workers=config['workers']) as pool:
        pending = [pool.submit(invoke, j, config, base, prefixes, folder) for j in jobs]
        for future in as_completed(pending):
            row = future.result(); rows.append(row); fp.write(json.dumps(row, ensure_ascii=False)+'\n'); fp.flush(); os.fsync(fp.fileno())
            print(f"completed {len(rows)}/{len(jobs)} {row['variant']} {row['case']} eligible={row['eligible']} errors={row['errors']} factual={row.get('factual_errors')} error={row.get('error','')}", flush=True)
    summary = {'scope': 'Typed action protocol: representational hallucinations impossible by construction, independent factual/choice checks still required. Not prompt-only or macro validation.', 'macro_claim': False, 'variants': {}}
    expected = {(s, c['aid'], c['case'], c['arm']) for s in config['seeds'] for c in inputs['cells']}
    for candidate in config['candidates']:
        rr = [r for r in rows if r['variant'] == candidate['id']]
        complete = len(rr) == len(expected) and {(r['replicate'],r['aid'],r['case'],r['arm']) for r in rr} == expected
        summary['variants'][candidate['id']] = {'responses': len(rr), 'complete': complete, 'raw_valid': sum(r.get('raw_valid',False) for r in rr),
            'valid': sum(r['valid'] for r in rr), 'eligible': sum(r['eligible'] for r in rr), 'all_pass': complete and all(r['eligible'] for r in rr),
            'adjusted_responses': sum(bool((r.get('temporal_projection') or {}).get('shifts')) for r in rr),
            'mean_generated_tokens': sum(r.get('thinking_usage',{}).get('completion_tokens',0)+r.get('answer_usage',{}).get('completion_tokens',0) for r in rr)/len(rr)}
    atomic(folder/'summary.json', summary); print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
