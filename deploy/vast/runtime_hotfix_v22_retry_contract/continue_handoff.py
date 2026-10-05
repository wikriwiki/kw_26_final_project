"""Continue the recorded handoff after its DB shutdown race, without resetting data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import tarfile

from switch_once import (HERE, ROOT, RUN, PROJECT, PYTHON, PIPELINE, HOOK, GUARD, REPORT,
    WATCHER, check, completed_hashes, pids, event, ensure_database, verify_remote, run_launcher)


def main():
    import fcntl
    lock = (HERE / 'switch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    guard = json.loads(GUARD.read_text())
    assert guard['activation_day'] == '2017-11-24'
    assert not pids(b'/scripts/sim/run_simulation.py') and not pids(b'/deploy/vast/run_shared.py')
    assert not pids(b'/deploy/vast/backup_checkpoint.py') and not pids(WATCHER)
    assert json.loads(PIPELINE.read_text())['status'] == 'failed'
    env = dict(os.environ, NEO4J_URI='bolt://127.0.0.1:17791', NEO4J_USER='neo4j',
        NEO4J_PASSWORD=Path('/root/no-smoking-v22-neo4j-password').read_text().strip(),
        BACKUP_NEO4J_HOME='/workspace/no-smoking-neo4j-integration-main-v22-1154-pre/off',
        BACKUP_NEO4J_BOLT_PORT='17791', BACKUP_RCLONE_BINARY='/workspace/bin/rclone',
        BACKUP_RCLONE_CONFIG='/workspace/no-smoking-drive.conf',
        BACKUP_DRIVE_REMOTE='no_smoking_drive:No_SmokingZone_EXP_Backups',
        BACKUP_CHECKPOINT_ROOT='/workspace/no-smoking-checkpoints')
    ensure_database(env)
    before = check(env, reconcile=True)
    assert max(before) == '2017-11-23'
    protected = completed_hashes()
    os.environ.update({k: v for k, v in env.items() if k.startswith('NEO4J_')})
    from day_resume import verified_completed_day
    manifest = json.loads((RUN / 'experiment_run.json').read_text())
    for day, row in before.items():
        if row['complete']:
            assert verified_completed_day(RUN, day, manifest['cohort_ids'], manifest['run_id'], manifest['arm'])
    event('retained_dates_verified_after_database_ready', dates=before)
    report = {'before': before, 'protected_hashes': protected,
              'activation_day': '2017-11-24', 'prior_handoff_error': 'DB shutdown race; original store retained'}
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    # Preserve the corrected maintenance procedure alongside the actual runtime.
    package = RUN / 'runtime_changes/v22-retry-contract-handoff-repair.tar.gz'
    with tarfile.open(package, 'w:gz') as archive:
        for name in ('state_check.py', 'switch_once.py', 'continue_handoff.py'):
            archive.add(HERE / name, arcname=name, recursive=False)
    (RUN / 'runtime_changes/handoff-repair.json').write_text(json.dumps({
        'archive': package.name, 'sha256': hashlib.sha256(package.read_bytes()).hexdigest(),
        'reason': 'Wait for stopped Neo4j to exit; reopen same store before receipt verification.'}) + '\n')
    with (HERE / 'cutover-backup.log').open('ab') as out:
        child = subprocess.Popen([str(PYTHON), str(HOOK), '--progress', '2017-11-23', str(RUN)],
            env=env, cwd=PROJECT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        event('backup_started_after_database_ready', pid=child.pid)
        result = child.wait(timeout=3000)
    assert result == 0, f'Backup failed: {result}'
    receipt = json.loads((RUN / 'recoverable_backup.json').read_text())
    assert receipt['day'] == '2017-11-23' and receipt['checkpoint_sha256'] != guard['old_checkpoint']
    remote = verify_remote(receipt, env)
    assert completed_hashes() == protected
    report.update(backup=receipt, independently_verified_remote=remote)
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    event('fresh_drive_backup_verified', receipt=receipt, remote=remote)
    subprocess.run([str(PYTHON), str(HERE / 'resume_runtime.py'), '--dry-run'], check=True, timeout=120)
    onstart = Path('/workspace/onstart.sh')
    prior = HERE / 'prior-onstart.sh'
    if not prior.exists():
        prior.write_bytes(onstart.read_bytes())
    temp = onstart.with_suffix('.retry-contract.tmp')
    temp.write_bytes((HERE / 'onstart.sh').read_bytes())
    temp.chmod(0o700)
    os.replace(temp, onstart)
    run_launcher(HERE / 'resume_runtime.py')
    until = time.monotonic() + 120
    while time.monotonic() < until:
        live = pids(b'/scripts/sim/run_simulation.py')
        if len(live) == 1:
            if b'PYTHONPATH=' + str(HERE).encode() in Path(f'/proc/{live[0]}/environ').read_bytes():
                break
        time.sleep(2)
    else:
        raise RuntimeError('New day-gated simulator was not observed')
    assert completed_hashes() == protected
    report.update(status='resumed', after=check(env), new_simulator_pid=live[0],
                  model_pids=pids(b'sglang.launch_server'))
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    event('resumed', simulator_pid=live[0], activation_day='2017-11-24',
          model_pids=report['model_pids'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        event('continuation_failed', error_type=type(exc).__name__, error=str(exc)[:400])
        raise
