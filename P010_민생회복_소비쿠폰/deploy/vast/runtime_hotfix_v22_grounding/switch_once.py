"""One controlled v22 runtime switch after a quiescent, verified Drive backup.

Run only on the existing instance. This never changes the frozen source, model,
cohort, or prior completed days. The event log contains no credentials.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


ROOT = Path('/workspace/no-smoking-results')
PREFIX = 'integration-main-v22-1154'
RUN = ROOT / f'{PREFIX}-shared-pre'
PIPELINE = ROOT / f'{PREFIX}-pipeline.json'
PYTHON = Path('/workspace/no-smoking-project-v22-perf-final/.venv-no-smoking/bin/python')
LAUNCHER = Path('/workspace/no-smoking-runtime-hotfix-v22-skip/resume_runtime.py')
HOOK = Path('/workspace/no-smoking-project-v22-perf-final/deploy/vast/backup_checkpoint.py')
PATCH_DIR = '/workspace/no-smoking-runtime-hotfix-v22-grounding'
LOG = ROOT / f'{PREFIX}-grounding-switch.jsonl'
GUARD = ROOT / f'{PREFIX}-grounding-switch-once.json'
BACKUP_KEYS = ('BACKUP_NEO4J_HOME', 'BACKUP_NEO4J_BOLT_PORT',
               'BACKUP_RCLONE_BINARY', 'BACKUP_RCLONE_CONFIG',
               'BACKUP_DRIVE_REMOTE', 'BACKUP_CHECKPOINT_ROOT')


def event(kind, **details):
    with LOG.open('a', encoding='utf-8') as out:
        out.write(json.dumps({'at_utc': datetime.now(timezone.utc).isoformat(),
                              'event': kind, **details}, ensure_ascii=False) + '\n')
        out.flush()
        os.fsync(out.fileno())


def pids(fragment):
    found = []
    for folder in Path('/proc').glob('[0-9]*'):
        try:
            args = (folder / 'cmdline').read_bytes().split(b'\0')
            if any(fragment in arg for arg in args):
                found.append(int(folder.name))
        except OSError:
            pass
    return found


def wait_absent(fragment, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pids(fragment):
            return True
        time.sleep(1)
    return not pids(fragment)


def simulator_backup_env(pid):
    values = dict(item.split(b'=', 1) for item in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
                  if b'=' in item)
    env = os.environ.copy()
    for key in BACKUP_KEYS:
        env[key] = values[key.encode()].decode()
    if values[b'SIM_POST_DAY_BACKUP_HOOK'].decode() != str(HOOK):
        raise RuntimeError('Unexpected checkpoint hook')
    return env


def preflight():
    state = json.loads(PIPELINE.read_text())
    supervisors = pids(b'/deploy/vast/run_shared.py')
    simulators = pids(b'/scripts/sim/run_simulation.py')
    watchers = pids(b'/workspace/no-smoking-runtime-hotfix-v22-skip/watch_pipeline.py')
    if state.get('status') != 'running' or state.get('phase') != 'shared-pre':
        raise RuntimeError('Pipeline is not running in shared-pre')
    if len(supervisors) != 1 or state.get('pid') != supervisors[0]:
        raise RuntimeError('Supervisor identity mismatch')
    if len(simulators) != 1 or len(watchers) != 1:
        raise RuntimeError('Expected one simulator and one no-auto-stop watcher')
    if pids(b'/deploy/vast/backup_checkpoint.py'):
        raise RuntimeError('Another graph checkpoint is running')
    day_files = sorted((RUN / 'metrics').glob('day_*.jsonl'))
    if not day_files:
        raise RuntimeError('No daily metrics to preserve')
    day = day_files[-1].stem.removeprefix('day_')
    stat = os.statvfs('/workspace')
    free_gb = stat.f_bavail * stat.f_frsize / 1e9
    if free_gb < 5:
        raise RuntimeError(f'Insufficient backup staging space: {free_gb:.2f} GB')
    if not PYTHON.is_file() or not LAUNCHER.is_file() or not HOOK.is_file():
        raise RuntimeError('Required runtime file is missing')
    env = simulator_backup_env(simulators[0])
    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as response:
        if response.status != 200:
            raise RuntimeError('Model health check failed')
    return day, supervisors[0], simulators[0], watchers[0], env, free_gb


def main():
    day, supervisor, simulator, watcher, backup_env, free_gb = preflight()
    with GUARD.open('x', encoding='utf-8') as out:
        json.dump({'started_at_utc': datetime.now(timezone.utc).isoformat(),
                   'old_supervisor_pid': supervisor, 'day': day}, out)
    event('preflight_passed', day=day, supervisor_pid=supervisor,
          simulator_pid=simulator, watcher_pid=watcher, free_gb=round(free_gb, 2))

    os.kill(watcher, signal.SIGTERM)
    if not wait_absent(b'/workspace/no-smoking-runtime-hotfix-v22-skip/watch_pipeline.py', 30):
        raise RuntimeError('Watcher did not stop; supervisor left untouched')
    os.kill(supervisor, signal.SIGTERM)
    supervisor_stopped = wait_absent(b'/deploy/vast/run_shared.py', 120)
    simulator_stopped = wait_absent(b'/scripts/sim/run_simulation.py', 30)
    event('quiesced', supervisor_stopped=supervisor_stopped,
          simulator_stopped=simulator_stopped)
    if not supervisor_stopped or not simulator_stopped:
        # A second simulator must never be started over an active one.
        raise RuntimeError('Workers did not quiesce; backup and restart blocked')

    try:
        checkpoint = subprocess.run([str(PYTHON), str(HOOK), '--progress', day, str(RUN)],
                                    cwd=HOOK.parents[2], env=backup_env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=60 * 60, check=False)
        event('checkpoint_finished', returncode=checkpoint.returncode,
              output=checkpoint.stdout[-600:])
    except Exception as exc:
        event('checkpoint_error', error_type=type(exc).__name__, error=str(exc)[:300])

    if pids(b'/scripts/sim/run_simulation.py') or pids(b'/deploy/vast/backup_checkpoint.py'):
        raise RuntimeError('Unexpected worker or checkpoint process before restart')
    launched = subprocess.run([str(PYTHON), str(LAUNCHER)], cwd=LAUNCHER.parent,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=20 * 60, check=False)
    event('launcher_finished', returncode=launched.returncode, output=launched.stdout[-600:])
    if launched.returncode:
        raise RuntimeError(f'Launcher failed with exit {launched.returncode}')
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        simulators = pids(b'/scripts/sim/run_simulation.py')
        if len(simulators) == 1:
            raw = Path(f'/proc/{simulators[0]}/environ').read_bytes()
            if b'PYTHONPATH=' + PATCH_DIR.encode() in raw:
                state = json.loads(PIPELINE.read_text())
                if state.get('status') == 'running':
                    event('resumed', day=day, simulator_pid=simulators[0],
                          supervisor_pid=state.get('pid'), grounding_path=PATCH_DIR)
                    return
        time.sleep(2)
    raise RuntimeError('New simulator with grounding patch was not observed')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        event('switch_failed', error_type=type(exc).__name__, error=str(exc)[:300])
        # Restore the no-auto-stop watcher, or the supervisor if fully idle.
        # Never launch alongside an orphan simulator or an offline dump.
        if not pids(b'/deploy/vast/backup_checkpoint.py'):
            supervisors = pids(b'/deploy/vast/run_shared.py')
            simulators = pids(b'/scripts/sim/run_simulation.py')
            if (len(supervisors) == 1 or not simulators) and LAUNCHER.is_file():
                fallback = subprocess.run([str(PYTHON), str(LAUNCHER)],
                                          cwd=LAUNCHER.parent, capture_output=True,
                                          text=True, timeout=20 * 60, check=False)
                event('fallback_launcher_finished', returncode=fallback.returncode,
                      output=fallback.stdout[-400:] + fallback.stderr[-200:])
        raise
