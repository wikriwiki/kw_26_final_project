"""One safe process handoff; new decisions are gated to Day6, no day is reset.

Never stops the Vast instance/model. Stop writers, verify graph/file receipts,
make a fresh Drive checkpoint, verify remote hashes, and resume the same run.
If backup fails, retain evidence and resume the prior runtime instead.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from state_check import check, completed_hashes

HERE = Path(__file__).resolve().parent
PROJECT = Path('/workspace/no-smoking-project-v22-perf-final')
PYTHON = PROJECT / '.venv-no-smoking/bin/python'
ROOT = Path('/workspace/no-smoking-results')
PREFIX = 'integration-main-v22-1154'
RUN = ROOT / f'{PREFIX}-shared-pre'
PIPELINE = ROOT / f'{PREFIX}-pipeline.json'
HOOK = PROJECT / 'deploy/vast/backup_checkpoint.py'
OLD_LAUNCHER = Path('/workspace/no-smoking-runtime-hotfix-v22-skip/resume_runtime.py')
WATCHER = b'/workspace/no-smoking-runtime-hotfix-v22-skip/watch_pipeline.py'
LOG = ROOT / f'{PREFIX}-retry-contract-switch.jsonl'
GUARD = ROOT / f'{PREFIX}-retry-contract-switch-once.json'
REPORT = ROOT / f'{PREFIX}-retry-contract-preservation.json'


def event(kind, **details):
    with LOG.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'at_utc': datetime.now(timezone.utc).isoformat(),
                                'event': kind, **details}, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def pids(fragment):
    found = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if any(fragment in arg for arg in (p / 'cmdline').read_bytes().split(b'\0')):
                found.append(int(p.name))
        except OSError:
            pass
    return found


def wait_absent(fragment, timeout=120):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if not pids(fragment):
            return
        time.sleep(1)
    raise RuntimeError(f'Process still present: {fragment.decode()}')


def run_launcher(path):
    result = subprocess.run([str(PYTHON), str(path)], capture_output=True, text=True, timeout=1200)
    event('launcher_result', launcher=str(path), returncode=result.returncode,
          output=result.stdout[-500:] + result.stderr[-500:])
    if result.returncode:
        raise RuntimeError('Launcher failed')


def ensure_database(env):
    """A DB restarted by the old backup hook can share the simulator PGID.

    Wait for its graceful shutdown to finish before starting the same store in
    a separate session. Starting during shutdown would race its PID/store lock.
    """
    home = Path(env['BACKUP_NEO4J_HOME']).resolve()
    assert home == Path('/workspace/no-smoking-neo4j-integration-main-v22-1154-pre/off')
    port = int(env['BACKUP_NEO4J_BOLT_PORT'])
    def reachable():
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=2):
                return True
        except OSError:
            return False
    if reachable():
        return
    wait_absent(('--home-dir=' + str(home)).encode(), 180)
    db_env = {k: v for k, v in env.items()
              if not any(term in k.upper() for term in ('PASSWORD', 'TOKEN', 'SECRET', 'API_KEY'))}
    db_env.update(NEO4J_HOME=str(home), NEO4J_CONF=str(home / 'conf'))
    subprocess.run([str(home / 'bin/neo4j'), 'start'], env=db_env, cwd=home,
                   start_new_session=True, check=True, timeout=180)
    until = time.monotonic() + 180
    while not reachable():
        if time.monotonic() >= until:
            raise RuntimeError('Existing Neo4j store did not become available')
        time.sleep(2)
    event('existing_database_ready', home=str(home), port=port)


def verify_remote(receipt, env):
    """Independent listing: hashes must match the uploader's local checkpoint."""
    remote = receipt['remote']
    leaf = remote.rsplit('/', 1)[-1]
    matches = list(Path(env['BACKUP_CHECKPOINT_ROOT']).glob(f'{RUN.name}/**/{leaf}/checkpoint.json'))
    if len(matches) != 1:
        raise ValueError('Unique local checkpoint record required')
    record_path = matches[0]
    raw = record_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt['checkpoint_sha256']
    record = json.loads(raw)
    args = [env['BACKUP_RCLONE_BINARY'], '--config', env['BACKUP_RCLONE_CONFIG'],
            'lsjson', remote, '--hash', '--files-only', '--max-depth', '1',
            '--contimeout', '10s', '--timeout', '40s', '--retries', '1', '--low-level-retries', '1']
    result = subprocess.run(args, capture_output=True, text=True, check=True, timeout=90)
    entries = {r['Name']: r for r in json.loads(result.stdout)}
    assert {'checkpoint.json', 'committed.json', 'neo4j.dump', 'run-day.tar.gz'}.issubset(entries)
    for name in ('checkpoint.json', 'committed.json'):
        assert entries[name]['Hashes']['md5'] == hashlib.md5((record_path.parent / name).read_bytes()).hexdigest()
    for name in ('neo4j.dump', 'run-day.tar.gz'):
        assert entries[name]['Size'] == record['files'][name]['bytes']
        assert entries[name]['Hashes']['sha256'] == record['files'][name]['sha256']
    marker = json.loads((record_path.parent / 'committed.json').read_text())
    assert marker['complete'] is True and marker['checkpoint_sha256'] == receipt['checkpoint_sha256']
    return {name: {'size': value['Size'], 'hashes': value['Hashes']} for name, value in entries.items()}


def main():
    lock = (HERE / 'switch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if GUARD.exists():
        raise ValueError('One-time handoff already attempted; inspect its journal')
    # Dry-run performs all old and new payload/config hash checks.
    subprocess.run([str(PYTHON), str(HERE / 'resume_runtime.py'), '--dry-run'], check=True, timeout=120)
    state = json.loads(PIPELINE.read_text())
    assert state['status'] == 'running' and state['phase'] == 'shared-pre'
    supervisors, simulators, watchers = pids(b'/deploy/vast/run_shared.py'), pids(b'/scripts/sim/run_simulation.py'), pids(WATCHER)
    assert len(supervisors) == len(simulators) == len(watchers) == 1
    assert state['pid'] == supervisors[0]
    assert not pids(b'/deploy/vast/backup_checkpoint.py'), 'Existing backup must finish first'
    active_day = sorted((RUN / 'metrics').glob('day_*.jsonl'))[-1].stem.removeprefix('day_')
    assert active_day == '2017-11-23', 'Day6 already began; reassess activation boundary'
    env = dict(item.decode().split('=', 1) for item in Path(f'/proc/{simulators[0]}/environ').read_bytes().split(b'\0') if b'=' in item)
    assert env['SIM_POST_DAY_BACKUP_HOOK'] == str(HOOK)
    disk = os.statvfs('/workspace')
    assert disk.f_bavail * disk.f_frsize > 4_000_000_000
    # Prove existing offsite recovery remains accessible BEFORE pausing writers.
    old_receipt = json.loads((RUN / 'recoverable_backup.json').read_text())
    verify_remote(old_receipt, env)
    protected = completed_hashes()
    models = pids(b'sglang.launch_server')
    assert len(models) == 1
    guard = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'old_pids': {
        'supervisor': supervisors[0], 'simulator': simulators[0], 'watcher': watchers[0]},
        'activation_day': '2017-11-24', 'old_checkpoint': old_receipt['checkpoint_sha256'],
        'protected_hashes': protected}
    with GUARD.open('x') as stream:
        json.dump(guard, stream)
    event('preflight_passed', **guard)
    paused = False
    backup = None
    try:
        os.kill(watchers[0], signal.SIGTERM)
        wait_absent(WATCHER, 30)
        paused = True
        os.kill(supervisors[0], signal.SIGTERM)
        wait_absent(b'/deploy/vast/run_shared.py')
        wait_absent(b'/scripts/sim/run_simulation.py')
        assert not pids(b'/deploy/vast/backup_checkpoint.py')
        ensure_database(env)
        before = check(env, reconcile=True)
        assert completed_hashes() == protected
        event('writers_stopped_and_receipts_verified', dates=before)
        report = {'before': before, 'protected_hashes': protected, 'activation_day': '2017-11-24'}
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
        backup_log = HERE / 'cutover-backup.log'
        with backup_log.open('ab') as out:
            backup = subprocess.Popen([str(PYTHON), str(HOOK), '--progress', active_day, str(RUN)],
                env=env, cwd=PROJECT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            event('backup_started', pid=backup.pid)
            code = backup.wait(timeout=3000)
        if code:
            raise RuntimeError(f'Cutover backup failed with exit {code}')
        receipt = json.loads((RUN / 'recoverable_backup.json').read_text())
        assert receipt['checkpoint_sha256'] != old_receipt['checkpoint_sha256'] and receipt['day'] == active_day
        remote = verify_remote(receipt, env)
        report.update(backup=receipt, independently_verified_remote=remote)
        assert completed_hashes() == protected and pids(b'sglang.launch_server') == models
        # New onstart invokes the verified day gate after any future container reboot.
        onstart = Path('/workspace/onstart.sh')
        previous = HERE / 'prior-onstart.sh'
        previous.write_bytes(onstart.read_bytes())
        temporary = onstart.with_suffix('.retry-contract.tmp')
        temporary.write_bytes((HERE / 'onstart.sh').read_bytes())
        temporary.chmod(0o700)
        os.replace(temporary, onstart)
        event('fresh_drive_backup_verified', receipt=receipt, remote=remote)
        run_launcher(HERE / 'resume_runtime.py')
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            live = pids(b'/scripts/sim/run_simulation.py')
            if len(live) == 1 and live != simulators:
                values = Path(f'/proc/{live[0]}/environ').read_bytes()
                if b'PYTHONPATH=' + str(HERE).encode() in values:
                    break
            time.sleep(2)
        else:
            raise RuntimeError('New simulator not observed')
        assert pids(b'sglang.launch_server') == models and completed_hashes() == protected
        report.update(after=check(env), new_simulator_pid=live[0], model_pid=models[0], status='resumed')
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
        event('resumed', simulator_pid=live[0], model_pid=models[0], activation_day='2017-11-24')
    except BaseException as exc:
        event('handoff_failed', error_type=type(exc).__name__, error=str(exc)[:400])
        # No concurrent restart while a checkpoint owns Neo4j. Retain the old
        # runtime as the recovery route; never reset/restore the graph here.
        if paused and not pids(b'/scripts/sim/run_simulation.py') and not pids(b'/deploy/vast/backup_checkpoint.py'):
            if (HERE / 'prior-onstart.sh').exists():
                Path('/workspace/onstart.sh').write_bytes((HERE / 'prior-onstart.sh').read_bytes())
            ensure_database(env)
            run_launcher(OLD_LAUNCHER)
        raise


if __name__ == '__main__':
    main()
