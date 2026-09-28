#!/usr/bin/env python3
"""Instance-local guard for one No Smoking Zone pipeline.

The supervisor, model server and this watcher run on Vast, independent of the
user's laptop. A failed or vanished pipeline gets one best-effort recoverable
Drive checkpoint and remains available for diagnosis and recovery. Only a
completed pipeline with a verified remote score and final manifest may stop.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


INSTANCE_ID = 52220534
RESULTS = Path('/workspace/no-smoking-results')
CLI = Path('/workspace/bin/vast')


def write_event(path, event, **details):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as out:
        out.write(json.dumps({'at_utc': datetime.now(timezone.utc).isoformat(),
                              'event': event, **details}, ensure_ascii=False) + '\n')
        out.flush()
        os.fsync(out.fileno())


def container_key_ready():
    values = Path('/proc/1/environ').read_bytes().split(b'\0')
    key = next((v.split(b'=', 1)[1] for v in values
                if v.startswith(b'CONTAINER_API_KEY=')), None)
    label = next((v.split(b'=', 1)[1] for v in values
                  if v.startswith(b'VAST_CONTAINERLABEL=')), None)
    if not key or len(key) < 32 or label != f'C.{INSTANCE_ID}'.encode():
        raise RuntimeError('Vast container control identity is unavailable')
    target = Path.home() / '.config/vastai/vast_api_key'
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as out:
        out.write(key)
    target.chmod(0o600)


def cli(*args, timeout=30):
    if not CLI.is_file():
        raise RuntimeError('Vast CLI is unavailable')
    command = [sys.executable, str(CLI), *args]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    output = (proc.stdout + '\n' + proc.stderr).strip()
    if proc.returncode or 'Invalid user key' in output or 'Authorization Error' in output:
        raise RuntimeError(f'Vast CLI request failed: {output[:200]}')
    return output


def stop_instance(log, reason):
    if reason != 'pipeline_complete':
        write_event(log, 'automatic_stop_suppressed', reason=reason)
        return
    container_key_ready()
    status = json.loads(cli('show', 'instance', str(INSTANCE_ID), '--raw'))
    if status.get('id') != INSTANCE_ID or status.get('actual_status') != 'running':
        write_event(log, 'stop_not_needed', reason=reason,
                    actual_status=status.get('actual_status'))
        return
    write_event(log, 'requesting_vast_stop', reason=reason)
    answer = cli('stop', 'instance', str(INSTANCE_ID), timeout=45)
    if 'stopping instance' not in answer.lower() and 'success' not in answer.lower():
        raise RuntimeError(f'Vast stop was not acknowledged: {answer[:200]}')
    write_event(log, 'vast_stop_acknowledged', reason=reason)


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        return (Path(f'/proc/{pid}/cmdline').read_bytes().find(b'run_shared.py') >= 0)
    except OSError:
        return False


def active_simulation_pids():
    found = []
    for folder in Path('/proc').glob('[0-9]*'):
        try:
            argv = (folder / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(arg.endswith(b'/scripts/sim/run_simulation.py') for arg in argv):
            found.append(int(folder.name))
    return found


def quiesce_orphan_workers(log):
    pids = active_simulation_pids()
    if not pids:
        return
    write_event(log, 'terminating_orphan_simulation', pids=pids)
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 45
    while active_simulation_pids() and time.monotonic() < deadline:
        time.sleep(1)
    if active_simulation_pids():
        write_event(log, 'orphan_simulation_did_not_stop')


def model_alive():
    import urllib.request
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def drive_file_verified(local, remote):
    """Wait for the supervisor's final upload before stopping a completed run."""
    proc = subprocess.run([str(Path('/workspace/bin/rclone')), '--config',
                           '/workspace/no-smoking-drive.conf', 'md5sum', remote],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode or not proc.stdout.strip():
        return False
    expected = hashlib.md5(local.read_bytes()).hexdigest()
    return proc.stdout.split()[0].lower() == expected


def latest_run(config, phase):
    prefix = config['prefix']
    if phase not in {'shared-pre', 'post-off', 'post-on'}:
        return None
    folder = RESULTS / f'{prefix}-{phase}'
    if not (folder / 'experiment_run.json').is_file():
        return None
    days = sorted((folder / 'metrics').glob('day_*.jsonl'))
    start = '2017-11-19' if phase == 'shared-pre' else '2017-12-03'
    day = days[-1].stem.removeprefix('day_') if days else start
    label = 'pre' if phase == 'shared-pre' else 'post'
    arm = 'on' if phase == 'post-on' else 'off'
    port = config['port_base'] + (0 if phase == 'shared-pre' else 3 if arm == 'on' else 2)
    return folder, day, label, arm, port


def backup_on_failure(config, phase, log):
    current = latest_run(config, phase)
    if current is None:
        write_event(log, 'no_partial_run_to_backup', phase=phase)
        return
    folder, day, label, arm, port = current
    root = Path(f"/workspace/no-smoking-neo4j-{config['prefix']}-{label}")
    env = dict(os.environ,
               BACKUP_RCLONE_BINARY='/workspace/bin/rclone',
               BACKUP_RCLONE_CONFIG='/workspace/no-smoking-drive.conf',
               BACKUP_DRIVE_REMOTE='no_smoking_drive:No_SmokingZone_EXP_Backups',
               BACKUP_CHECKPOINT_ROOT='/workspace/no-smoking-checkpoints',
               BACKUP_NEO4J_HOME=str(root / arm),
               BACKUP_NEO4J_BOLT_PORT=str(port))
    script = Path(config['project']) / 'deploy/vast/backup_checkpoint.py'
    write_event(log, 'partial_backup_start', phase=phase, day=day)
    try:
        result = subprocess.run([sys.executable, str(script), '--progress', day, str(folder)],
                                env=env, capture_output=True, text=True, timeout=1200)
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr)[-350:])
        receipt = json.loads((folder / 'recoverable_backup.json').read_text(encoding='utf-8'))
        write_event(log, 'partial_backup_verified', day=day,
                    checkpoint_sha256=receipt.get('checkpoint_sha256'))
    except Exception as exc:
        write_event(log, 'partial_backup_failed', error=str(exc)[:400])


def watch(config_path, poll_seconds=30):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if (config.get('instance_id') != INSTANCE_ID
            or not re.fullmatch(r'integration-main-v(?:20|21|22)-1154', config.get('prefix', ''))):
        raise ValueError('Watcher is not bound to the approved Vast instance and run')
    pipeline = RESULTS / f"{config['prefix']}-pipeline.json"
    log = RESULTS / f"{config['prefix']}-watchdog.jsonl"
    write_event(log, 'watcher_started', pipeline=str(pipeline))
    missing_since = None
    unhealthy_since = None
    start = time.monotonic()
    while True:
        try:
            state = json.loads(pipeline.read_text(encoding='utf-8')) if pipeline.is_file() else None
            if state is None:
                if time.monotonic() - start > 180:
                    write_event(log, 'pipeline_never_started')
                    stop_instance(log, 'pipeline_never_started')
                    return
            elif state.get('config', {}).get('prefix') != config['prefix']:
                raise ValueError('Watcher pipeline identity changed')
            elif state.get('status') == 'complete':
                score = RESULTS / f"{config['prefix']}-shared-score.json"
                if not score.is_file() or not state.get('score_sha256'):
                    raise ValueError('Completed pipeline has no score receipt')
                remote_root = 'no_smoking_drive:No_SmokingZone_EXP_Backups/runs'
                if not (drive_file_verified(score, f'{remote_root}/{score.name}')
                        and drive_file_verified(pipeline, f'{remote_root}/{pipeline.name}')):
                    write_event(log, 'waiting_for_final_drive_upload')
                    time.sleep(poll_seconds)
                    continue
                stop_instance(log, 'pipeline_complete')
                return
            elif state.get('status') == 'failed':
                quiesce_orphan_workers(log)
                backup_on_failure(config, state.get('phase'), log)
                stop_instance(log, 'pipeline_failed')
                return
            elif not process_alive(state.get('pid')):
                missing_since = missing_since or time.monotonic()
                if time.monotonic() - missing_since >= 90:
                    quiesce_orphan_workers(log)
                    backup_on_failure(config, state.get('phase'), log)
                    stop_instance(log, 'supervisor_missing')
                    return
            else:
                missing_since = None
                if not model_alive():
                    unhealthy_since = unhealthy_since or time.monotonic()
                    if time.monotonic() - unhealthy_since >= 300:
                        write_event(log, 'model_unhealthy', phase=state.get('phase'))
                        try:
                            os.kill(state['pid'], signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        time.sleep(35)
                        quiesce_orphan_workers(log)
                        backup_on_failure(config, state.get('phase'), log)
                        stop_instance(log, 'model_server_unhealthy')
                        return
                else:
                    unhealthy_since = None
        except Exception as exc:
            write_event(log, 'watcher_error', error=str(exc)[:400])
        time.sleep(poll_seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    watch(args.config)
