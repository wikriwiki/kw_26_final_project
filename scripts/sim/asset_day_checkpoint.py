"""Atomic file-only day barrier for already isolated experimental purchase paths.

Revalidates raw choices against frozen quotes before making a complete snapshot.
Does not implement policy transfers, memories, demand generation or auto recovery.
"""
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import tempfile
from asset_transaction_contract import inspect


def commit_day(root, *, day, roster, cases, rows, previous=None, state_protocol='inventory_v1', transition_templates=None):
    if date.fromisoformat(day).isoformat() != day: raise ValueError('Canonical day required')
    folder = Path(root).resolve()
    if state_protocol not in {'inventory_v1','carry_needs_v1'}:raise ValueError('Unknown daily state protocol')
    if transition_templates is not None and (state_protocol!='carry_needs_v1' or previous is None or set(transition_templates)!=set(roster)):
        raise ValueError('Transition templates require a complete continuing need-state path')
    if state_protocol=='carry_needs_v1' and previous is not None and transition_templates is None:
        raise ValueError('Explicit next-day assumption templates required')
    if not roster or len(set(roster)) != len(roster) or set(cases) != set(roster): raise ValueError('Day roster')
    by_id = {}
    for row in rows:
        if row['aid'] in by_id or row['aid'] not in cases: raise ValueError('Duplicate/unknown citizen')
        if row.get('complete') is not True: raise ValueError('Failed response cannot be zero filled')
        by_id[row['aid']] = row
    if set(by_id) != set(roster): raise ValueError('Incomplete day')
    protocols={row.get('transaction_protocol','v1') for row in rows}
    if len(protocols)!=1:raise ValueError('Mixed transaction protocols in one day')
    protocol=next(iter(protocols))
    if protocol=='v1':check=inspect
    elif protocol=='v2':
        from asset_transaction_contract_v2 import inspect as check
    elif protocol in {'v3','v4'}:
        from asset_transaction_contract_v3 import inspect as check
    else:raise ValueError('Unknown transaction protocol')
    prior = None; previous_hash = None
    if previous is not None:
        previous = Path(previous).resolve()
        if previous.parent != folder: raise ValueError('Prior state belongs to another isolated path')
        raw = previous.read_bytes(); prior = json.loads(raw); previous_hash = hashlib.sha256(raw).hexdigest()
        if prior.get('complete') is not True or (date.fromisoformat(day)-date.fromisoformat(prior['day'])).days != 1 or prior['roster'] != list(roster):
            raise ValueError('Invalid prior day; consecutive day required')
        if prior.get('transaction_protocol','v1')!=protocol:raise ValueError('Transaction protocol changed within frozen path')
        if prior.get('state_protocol','inventory_v1')!=state_protocol:raise ValueError('Daily state protocol changed within frozen path')
    ledgers = {}; states = {}
    for aid in roster:
        case = cases[aid]
        if state_protocol=='carry_needs_v1':
            if protocol not in {'v3','v4'} or 'daily_conditions' not in case:
                raise ValueError('Need-state protocol requires current choices and physical state')
            if prior is not None:
                from daily_state_transition import advance
                expected=advance(prior['frozen_cases'][aid],prior['raw_choices'][aid],transition_templates[aid])
                if any(case[k]!=expected[k] for k in expected):
                    raise ValueError('Daily need/resource/financial state discontinuity')
        if prior is not None and (case['cash'] != prior['closing_states'][aid]['cash'] or case['wallet_lots'] != prior['closing_states'][aid]['wallet_lots']):
            raise ValueError('State discontinuity; exogenous transfers need a separate registered transition')
        _, ledger = check(by_id[aid]['raw'], case)
        ledgers[aid] = ledger
        states[aid] = {'cash': ledger['closing_cash'], 'wallet_lots': ledger['closing_wallet_lots']}
        conditions = case.get('daily_conditions')
        prior_resources = prior['closing_states'][aid].get('resources') if prior else None
        if prior is not None and (conditions is None) != (prior_resources is None):
            raise ValueError('Physical-state protocol changed within path')
        if conditions is not None:
            if protocol not in {'v3','v4'}:raise ValueError('Physical-state checkpoint requires purchase-choice protocol')
            from daily_resource_contract import settle, validate
            validate(conditions)
            if prior_resources is not None:
                opening = {k:v['opening_quantity'] for k,v in conditions['resources'].items()}
                units = {k:v['unit'] for k,v in conditions['resources'].items()}
                pending = [dict(q,minute=q['minute']-1440) for q in prior['closing_states'][aid]['pending_receipts']]
                if opening != prior_resources or units != prior['closing_states'][aid]['resource_units'] or conditions.get('opening_pending_receipts',[]) != pending:
                    raise ValueError('Physical inventory/delivery state discontinuity')
            physical = settle(by_id[aid]['raw'],case)
            ledgers[aid]['resource_ledger'] = physical
            states[aid].update(resources=physical['closing_resources'],resource_units=physical['resource_units'],pending_receipts=physical['pending_receipts'])
    snapshot = {'day': day, 'roster': list(roster), 'complete': True, 'previous_sha256': previous_hash,'transaction_protocol':protocol,
                'state_protocol':state_protocol,'transition_templates':transition_templates,
                'frozen_cases': cases, 'raw_choices': {a: by_id[a]['raw'] for a in roster}, 'ledgers': ledgers, 'closing_states': states}
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (day + '.json'); lock = folder / 'day_barrier.lock'
    # Cooperative exclusive lock. A crash leaves a visible lock for explicit audit.
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.close(fd)
    temp = None
    try:
        if target.exists(): raise ValueError('Refusing day overwrite or implicit retry')
        snapshots = sorted(folder.glob('????-??-??.json'))
        if snapshots and (previous is None or snapshots[-1] != previous): raise ValueError('Prior snapshot is not the latest committed day')
        fd, temp = tempfile.mkstemp(prefix=day+'.pending-', suffix='.json', dir=folder)
        with os.fdopen(fd, 'w', encoding='utf-8') as fp:
            json.dump(snapshot, fp, ensure_ascii=False, indent=2); fp.write('\n'); fp.flush(); os.fsync(fp.fileno())
        os.replace(temp, target); temp = None
        if os.name != 'nt':
            directory = os.open(folder, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
    finally:
        if temp is not None: Path(temp).unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    return target
