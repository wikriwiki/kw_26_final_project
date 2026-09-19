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


def commit_day(root, *, day, roster, cases, rows, previous=None):
    if date.fromisoformat(day).isoformat() != day: raise ValueError('Canonical day required')
    folder = Path(root).resolve()
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
    ledgers = {}; states = {}
    for aid in roster:
        case = cases[aid]
        if prior is not None and (case['cash'] != prior['closing_states'][aid]['cash'] or case['wallet_lots'] != prior['closing_states'][aid]['wallet_lots']):
            raise ValueError('State discontinuity; exogenous transfers need a separate registered transition')
        _, ledger = check(by_id[aid]['raw'], case)
        ledgers[aid] = ledger
        states[aid] = {'cash': ledger['closing_cash'], 'wallet_lots': ledger['closing_wallet_lots']}
    snapshot = {'day': day, 'roster': list(roster), 'complete': True, 'previous_sha256': previous_hash,'transaction_protocol':protocol,
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
