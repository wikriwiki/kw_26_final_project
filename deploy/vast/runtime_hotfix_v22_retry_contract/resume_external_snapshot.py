"""Finish this handoff using independently hashed off-instance local recovery.

The Drive uploader remains alive. Its offline dump/archive phase has completed;
only network transfer remains. This does not mark Drive as verified.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

from switch_once import (HERE, RUN, REPORT, PYTHON, PIPELINE, pids, wait_absent,
                         event, run_launcher, completed_hashes, check)


def main():
    receipt = json.loads((HERE / 'local_hash_verified.json').read_text())
    assert receipt['status'] == 'verified_external_local_copy' and receipt['drive_upload_verified'] is False
    work = Path('/workspace/no-smoking-checkpoints/integration-main-v22-1154-shared-pre/progress-2017-11-23/20260928T171532Z-655e3a7a')
    raw = (work / 'checkpoint.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt['checkpoint_sha256']
    record = json.loads(raw)
    for name, expected in record['files'].items():
        assert receipt['files'][name] == expected
        path = work / name
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        assert digest == expected['sha256'] and path.stat().st_size == expected['bytes']
    # Receipt exists only after offline dump returned and Neo4j was restarted.
    with socket.create_connection(('127.0.0.1', 17791), timeout=3):
        pass
    assert not pids(b'/scripts/sim/run_simulation.py') and not pids(b'/deploy/vast/run_shared.py')
    controllers = pids(b'/retry-contract/continue_handoff.py')
    # The actual pathname includes the complete runtime directory name.
    controllers = pids(str(HERE / 'continue_handoff.py').encode())
    assert len(controllers) == 1
    os.kill(controllers[0], signal.SIGTERM)  # only its waiting parent, never killpg
    wait_absent(str(HERE / 'continue_handoff.py').encode(), 30)
    assert len(pids(b'/deploy/vast/backup_checkpoint.py')) == 1
    report = json.loads(REPORT.read_text())
    assert completed_hashes() == report['protected_hashes']
    env = dict(os.environ, NEO4J_URI='bolt://127.0.0.1:17791', NEO4J_USER='neo4j',
               NEO4J_PASSWORD=Path('/root/no-smoking-v22-neo4j-password').read_text().strip())
    report.update(local_external_recovery=receipt, drive_upload_status='pending_shared_client_quota_403')
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    (RUN / 'runtime_changes/external_local_backup_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    onstart = Path('/workspace/onstart.sh')
    prior = HERE / 'prior-onstart.sh'
    if not prior.exists():
        prior.write_bytes(onstart.read_bytes())
    temp = onstart.with_suffix('.retry-contract.tmp')
    temp.write_bytes((HERE / 'onstart.sh').read_bytes())
    temp.chmod(0o700)
    os.replace(temp, onstart)
    event('external_local_snapshot_verified_drive_upload_continues', receipt=receipt)
    run_launcher(HERE / 'resume_runtime.py')
    until = time.monotonic() + 120
    while time.monotonic() < until:
        live = pids(b'/scripts/sim/run_simulation.py')
        if len(live) == 1 and b'PYTHONPATH=' + str(HERE).encode() in Path(f'/proc/{live[0]}/environ').read_bytes():
            break
        time.sleep(2)
    else:
        raise RuntimeError('New simulator not observed')
    assert completed_hashes() == report['protected_hashes']
    report.update(status='resumed', after=check(env), new_simulator_pid=live[0],
                  model_pids=pids(b'sglang.launch_server'))
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    event('resumed_from_external_local_snapshot', simulator_pid=live[0], activation_day='2017-11-24',
          model_pids=report['model_pids'], drive_verified=False)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        event('local_snapshot_resume_failed', error_type=type(exc).__name__, error=str(exc)[:400])
        raise
