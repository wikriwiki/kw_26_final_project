"""Read-only independent raw ledger, matrix and actual-request audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from daily_resource_contract import settle
from forced_no_purchase import resolve
from validate_purchase_probe import decision_case, factual_errors, protocol_modules
from validate_prompt_v3 import atomic, digest


def audit(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_bytes())
    config = manifest['config']
    source_raw = (folder / 'frozen_inputs.json').read_bytes()
    if hashlib.sha256(source_raw).hexdigest() != manifest['input_sha256'] or manifest['input_sha256'] != config['source_sha256']:
        raise ValueError('Frozen source hash mismatch')
    contract, system = protocol_modules(config)
    if (folder / 'system.txt').read_text(encoding='utf-8') != system or digest(system) != manifest['system_sha256']:
        raise ValueError('Registered system prompt mismatch')
    cells = json.loads(source_raw)['cells']
    key = lambda r: (r['aid'], r['case'], r['arm'])
    lookup = {key(c): c for c in cells}
    if len(lookup) != len(cells):
        raise ValueError('Duplicate source cells')
    rows = [json.loads(line) for line in (folder / 'responses.jsonl').read_bytes().splitlines()]
    counts = Counter((r['replicate'], *key(r)) for r in rows)
    expected = {(s, *key(c)) for s in config['seeds'] for c in cells}
    if set(counts) != expected or any(n != 1 for n in counts.values()):
        raise ValueError('Incomplete/duplicate registered matrix')
    errors = []
    decisions = Counter()
    request_count = 0
    summaries = []
    for row in rows:
        cell = lookup[key(row)]
        case = cell['transaction_case']
        attempt = row['attempt_key']
        decisions[row.get('decision_source', 'missing')] += 1
        item = {k: row[k] for k in ['attempt_key', 'aid', 'case', 'arm', 'replicate']}
        try:
            if attempt != digest([row['replicate'], *key(row)]) or row['date'] != cell['date']:
                raise ValueError('Response identity mismatch')
            if row['transaction_protocol'] != config['transaction_protocol']:
                raise ValueError('Transaction protocol mismatch')
            if row.get('valid') is not True or row.get('errors'):
                raise ValueError('Recorded failed response: ' + str(row.get('errors')))
            _, ledger = contract.inspect(row['raw'], case)
            if ledger != row['ledger'] or factual_errors(ledger, cell.get('evaluation_ledger', {})):
                raise ValueError('Independent financial ledger mismatch')
            if 'daily_conditions' in case:
                physical = settle(row['raw'], case)
                if physical != row.get('resource_ledger'):
                    raise ValueError('Independent physical ledger mismatch')
                item['resource_ledger'] = physical
            requests = list((folder / 'attempts').glob(attempt + '*_request.json'))
            if row['decision_source'] == 'deterministic_unique_no_purchase':
                if resolve(case) is None or requests or not config.get('skip_forced_no_purchase'):
                    raise ValueError('Unjustified deterministic resolution')
                if row.get('model_calls') != 0:
                    raise ValueError('Deterministic model-call count')
            elif row['decision_source'] == 'model':
                if row['answer_usage']['finish_reason']['type'] != 'stop':
                    raise ValueError('Incomplete answer')
                first = folder / 'attempts' / (attempt + '_deliberation_request.json')
                prefix = json.loads(first.read_bytes())['text']
                user = json.dumps(decision_case(case, config), ensure_ascii=False)
                if digest(prefix) != manifest['prefix_sha256'][case['id']] or user not in prefix or system not in prefix:
                    raise ValueError('Actual initial request differs from frozen input or prompt')
                if len(requests) != 2:
                    raise ValueError('Expected exactly two bounded-generation requests')
                for path in requests:
                    text = json.loads(path.read_bytes())['text']
                    if not text.startswith(prefix) or any(word in text for word in ['own_basis', 'wallet_lots', 'evaluation_ledger', 'feasibility_witnesses_not_model_input']):
                        raise ValueError('Mismatched request or hidden evaluation/bookkeeping input')
                answer = json.loads((folder / 'attempts' / (attempt + '_answer.json')).read_bytes())
                if answer['response']['text'] != row['raw']:
                    raise ValueError('Raw answer archive differs from scored row')
                request_count += len(requests)
            else:
                raise ValueError('Unknown decision source')
            choice = json.loads(row['raw'])
            item.update(decision_source=row['decision_source'], acquisition_units=choice.get('acquisition_units'),
                        purchases=choice['purchases'], total_consumption=ledger['total_consumption'])
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append({'attempt_key': attempt, 'error': str(exc)})
            item['audit_error'] = str(exc)
        summaries.append(item)
    return {'rows': len(rows), 'original_matrix_complete': True, 'decision_sources': dict(decisions),
            'request_files_audited': request_count, 'independent_errors': errors, 'choices': summaries,
            'scope': 'Complete raw monetary/physical/request audit; no empirical demand or policy-effect validity claim.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError('Refusing overwrite')
    result = audit(args.run)
    atomic(args.out, result)
    print(json.dumps({k: result[k] for k in ['rows', 'decision_sources', 'request_files_audited', 'independent_errors']}))
