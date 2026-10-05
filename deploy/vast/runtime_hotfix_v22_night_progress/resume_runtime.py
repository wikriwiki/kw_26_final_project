#!/usr/bin/env python3
"""Persistent recovery controller: resume existing receipts, never stop an unfinished run."""

import argparse
import importlib.util
import socket
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


PROJECT = Path('/workspace/no-smoking-project-v22-perf-final')
RESULTS = Path('/workspace/no-smoking-results')
PREFIX = 'integration-main-v22-1154'
PYTHON = PROJECT / '.venv-no-smoking/bin/python'
CONFIG = Path('/workspace/expedited-full-1154-v22-day2-config.json')
SERVER = PROJECT / 'output/experiments/no_smoking_zone/runtime/server-config.json'
PATCH = Path('/workspace/no-smoking-runtime-hotfix-v22-skip/sitecustomize.py')
PATCH_SHA256 = 'e6b2212e007c49889e13a117e22fcb8017d26231b75fba8b810e4afc1521a486'
WATCHER = Path('/workspace/no-smoking-runtime-hotfix-v22-skip/watch_pipeline.py')
WATCHER_SHA256 = 'd2b28b92f0c080ba1420aa45eee87ce0057b2022a8dca70947ff2fe4d4a5bf1c'
NIGHT_PATCH = Path('/workspace/no-smoking-runtime-hotfix-v22-night2/sitecustomize.py')
NIGHT_PATCH_SHA256 = 'c8680ced3d27fbeaaf92a89f6e27b65fee44050280ffcddb832c2c094a4c4545'
NIGHT_RECOVERY_PATCH = Path('/workspace/no-smoking-runtime-hotfix-v22-night2-recovery/sitecustomize.py')
NIGHT_RECOVERY_PATCH_SHA256 = '500029c3259862cb5f6c284a56f34ce7f95ea31d95185c2b30682772b6cf1e91'
NIGHT_RECOVERY_CACHE = RESULTS / f'{PREFIX}-shared-pre/night2_recovery_2017-11-20.json'
NIGHT_RECOVERY_CACHE_SHA256 = '923354b4836471ce9acdba1965f083fc28a5eb33ca448251a1140f0d56c597ec'
RETRY_PATCH = Path('/workspace/no-smoking-runtime-hotfix-v22-retry/sitecustomize.py')
RETRY_PATCH_SHA256 = '97805a307f5a47b6d48f93e6019c45363114457174155890f158563ff613a422'
RETRY_STAGE1 = RETRY_PATCH.parent / 'stage1_intent.py'
RETRY_STAGE1_SHA256 = 'e77a7736743c5f43efce5363aa9eec73e601bdf71af9ba37196685647c3305bc'
RETRY_STAGE2 = RETRY_PATCH.parent / 'stage2_poi.py'
RETRY_STAGE2_SHA256 = 'ec7df34bf4a1190bed07074359ef6c34f06b3d17524ab257336439855ec750c6'
GROUNDING_PATCH = Path('/workspace/no-smoking-runtime-hotfix-v22-grounding/sitecustomize.py')
GROUNDING_PATCH_SHA256 = '7dd2a8a425724ba5c8c658c04a0fdfd9190ec3b21d06be8b23350c9844eb97d5'
GROUNDING_HELPER_SHA256 = '505b513a9c7f50dd385813817729459a03462bc0f8e8f0a4d6c728e2b62b1c7f'
GROUNDING_STAGE1_SHA256 = 'b8a131a529d6beff5bd561d8d5bc45c350489b0dcf85c683ba55372744075499'
GROUNDING_STAGE2_SHA256 = '916784a7c9c78fca7268d6c26e98feeb1ef31f535134672330ed8919687e2ef4'
CONTRACT_PATCH = Path('/workspace/no-smoking-runtime-hotfix-v22-retry-contract/sitecustomize.py')
CONTRACT_PATCH_SHA256 = '732eaab121623997c6f55a007160f1af237c1a18196210221c42d313fb502f74'
CONTRACT_MANIFEST_SHA256 = '7f0c6f2d4ed7b956b1371898b3736db511150625ae71db1f6044819520166b99'
SOURCE_SHA256 = 'e6ab8021766613426a615de4f1322b2d315864b857080d4fe047c5e2e4277f7f'
PASSWORD = Path('/root/no-smoking-v22-neo4j-password')
PIPELINE = RESULTS / f'{PREFIX}-pipeline.json'
RUN = RESULTS / f'{PREFIX}-shared-pre'
JOURNAL = RESULTS / f'{PREFIX}-resume-control.jsonl'
GUARD = RESULTS / f'{PREFIX}-resume-guard.json'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(kind, **details):
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open('a', encoding='utf-8') as out:
        out.write(json.dumps({'at_utc': datetime.now(timezone.utc).isoformat(),
                              'event': kind, **details}, ensure_ascii=False) + '\n')
        out.flush()
        os.fsync(out.fileno())


def process_pid(fragment):
    for folder in Path('/proc').glob('[0-9]*'):
        try:
            args = (folder / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(fragment == arg or (b'/' not in fragment and fragment == arg) for arg in args):
            return int(folder.name)
    return None


def healthy():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def guarded_signature(state):
    phase = state.get('phase')
    run = RESULTS / f'{PREFIX}-{phase}' if phase in {'shared-pre', 'post-off', 'post-on'} else RUN
    metrics = sorted((run / 'metrics').glob('day_*.jsonl'))
    metric_sha = sha256(metrics[-1]) if metrics else None
    payload = {'status': state.get('status'), 'error': state.get('error'),
               'phase': state.get('phase'), 'metric_sha256': metric_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate():
    config = json.loads(CONFIG.read_text())
    if config.get('prefix') != PREFIX or config.get('source_sha256') != SOURCE_SHA256:
        raise RuntimeError('Frozen v22 config identity mismatch')
    if sha256(Path(config['source_archive'])) != SOURCE_SHA256:
        raise RuntimeError('Frozen v22 source archive changed')
    if sha256(PATCH) != PATCH_SHA256:
        raise RuntimeError('Skipped-State hotfix changed')
    if sha256(WATCHER) != WATCHER_SHA256:
        raise RuntimeError('No-auto-stop watcher changed')
    if sha256(NIGHT_PATCH) != NIGHT_PATCH_SHA256:
        raise RuntimeError('Bounded Night2 retry patch changed')
    if sha256(NIGHT_RECOVERY_PATCH) != NIGHT_RECOVERY_PATCH_SHA256:
        raise RuntimeError('Night2 recovery patch changed')
    if sha256(NIGHT_RECOVERY_CACHE) != NIGHT_RECOVERY_CACHE_SHA256:
        raise RuntimeError('Night2 validated answer cache changed')
    if sha256(RETRY_PATCH) != RETRY_PATCH_SHA256:
        raise RuntimeError('Retry prompt runtime patch changed')
    if sha256(RETRY_STAGE1) != RETRY_STAGE1_SHA256 or sha256(RETRY_STAGE2) != RETRY_STAGE2_SHA256:
        raise RuntimeError('Reviewed retry prompt functions changed')
    if sha256(GROUNDING_PATCH) != GROUNDING_PATCH_SHA256:
        raise RuntimeError('Grounding runtime patch changed')
    if (sha256(GROUNDING_PATCH.parent / 'canonical_evidence_ref.py') != GROUNDING_HELPER_SHA256
            or sha256(GROUNDING_PATCH.parent / 'stage1_intent.py') != GROUNDING_STAGE1_SHA256
            or sha256(GROUNDING_PATCH.parent / 'stage2_poi.py') != GROUNDING_STAGE2_SHA256):
        raise RuntimeError('Reviewed grounding functions changed')
    if sha256(CONTRACT_PATCH) != CONTRACT_PATCH_SHA256:
        raise RuntimeError('Day-gated retry-contract entry point changed')
    manifest_path = CONTRACT_PATCH.parent / 'manifest.json'
    if sha256(manifest_path) != CONTRACT_MANIFEST_SHA256:
        raise RuntimeError('Day-gated retry-contract manifest changed')
    for name, expected in json.loads(manifest_path.read_text())['files'].items():
        if Path(name).name != name or sha256(CONTRACT_PATCH.parent / name) != expected:
            raise RuntimeError(f'Day-gated retry-contract payload changed: {name}')
    if not PYTHON.is_file() or not PASSWORD.is_file() or len(PASSWORD.read_text().strip()) < 8:
        raise RuntimeError('Runtime or Neo4j credential unavailable')
    state = json.loads(PIPELINE.read_text())
    if state.get('config') != config or state.get('status') not in {'running', 'failed', 'complete'}:
        raise RuntimeError('Unexpected pipeline identity or status')
    return state


def start_model():
    if healthy():
        return
    current = process_pid(b'sglang.launch_server')
    if current is None:
        record = json.loads(SERVER.read_text())
        argv = record['argv']
        if (argv[:3] != ['/workspace/venv_sgl/bin/python', '-m', 'sglang.launch_server']
                or record.get('model_revision') != '31e6a965d0661bbe4a8b895e22a77f8271772ba0'
                or record.get('grammar_backend') != 'outlines'):
            raise RuntimeError('Recorded model command changed')
        log = RESULTS / f'{PREFIX}-model-server-autoresume.log'
        with log.open('ab') as out:
            current = subprocess.Popen(argv, cwd=PROJECT, env=os.environ.copy(),
                                       stdin=subprocess.DEVNULL, stdout=out,
                                       stderr=subprocess.STDOUT, start_new_session=True).pid
        event('model_started', pid=current)
    deadline = time.monotonic() + 15 * 60
    while time.monotonic() < deadline:
        if healthy():
            event('model_healthy', pid=current)
            return
        if not Path(f'/proc/{current}').exists():
            raise RuntimeError('Model server exited before health check')
        time.sleep(5)
    raise RuntimeError('Model health did not recover within 15 minutes')


def start_supervisor():
    current = process_pid(str(PROJECT / 'deploy/vast/run_shared.py').encode())
    child = None
    if current is None:
        env = os.environ.copy()
        password = PASSWORD.read_text().strip()
        env.update(NEO4J_PASSWORD=password, NO_SMOKING_OFF_NEO4J_PASSWORD=password,
                   NO_SMOKING_ON_NEO4J_PASSWORD=password,
                   NO_SMOKING_V22_SKIP_PERSISTENCE_HOTFIX='1',
                   PYTHONPATH=':'.join(('/workspace/no-smoking-runtime-hotfix-v22-night-progress', str(CONTRACT_PATCH.parent), str(GROUNDING_PATCH.parent), str(RETRY_PATCH.parent),
                                        str(NIGHT_RECOVERY_PATCH.parent),
                                        str(NIGHT_PATCH.parent), str(PATCH.parent),
                                        str(PROJECT / 'scripts/sim'),
                                        str(PROJECT / 'scripts'), str(PROJECT))))
        log = RESULTS / f'{PREFIX}-continuation-hotfix.log'
        with log.open('ab') as out:
            child = subprocess.Popen([str(PYTHON), str(PROJECT / 'deploy/vast/run_shared.py'),
                                      '--config', str(CONFIG)], cwd=PROJECT, env=env,
                                     stdin=subprocess.DEVNULL, stdout=out,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        current = child.pid
        event('supervisor_started', pid=current, hotfix_sha256=PATCH_SHA256,
              retry_prompt_sha256=RETRY_PATCH_SHA256,
              grounding_sha256=GROUNDING_PATCH_SHA256,
              retry_contract_sha256=CONTRACT_MANIFEST_SHA256)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if child is not None and child.poll() is not None:
            raise RuntimeError(f'Supervisor exited during startup: {child.returncode}')
        if process_pid(str(PROJECT / 'deploy/vast/run_shared.py').encode()) != current:
            raise RuntimeError('Existing supervisor exited during startup')
        state = json.loads(PIPELINE.read_text())
        if state.get('status') == 'running' and state.get('pid') == current:
            return current
        time.sleep(2)
    raise RuntimeError('Supervisor did not enter running state')


RUNTIME_ROOT = Path('/workspace/no-smoking-runtime-hotfix-v22-night-progress')
MANIFEST_SHA256 = '19f2a2eaf2587b8ee85da0e5f14278877a86fd4de543346bd31bb325383455eb'


def validate_extension():
    raw = (RUNTIME_ROOT / 'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Night progress manifest changed')
    for name, expected in json.loads(raw)['files'].items():
        path = (RUNTIME_ROOT / name).resolve()
        if not path.is_relative_to(RUNTIME_ROOT) or sha256(path) != expected:
            raise ValueError(f'Night progress runtime changed: {name}')


def decide_action(state, supervisor, simulator, backup):
    # Never overlap a dump, a living simulator, or a supervisor finishing work.
    if backup:
        return 'backup_in_progress'
    if supervisor or simulator:
        return 'work_in_progress'
    if state.get('status') == 'complete':
        return 'verify_completion'
    return 'resume'


def current_processes():
    return {
        'supervisor': process_pid(str(PROJECT / 'deploy/vast/run_shared.py').encode()),
        'simulator': process_pid(str(RUNTIME_ROOT / 'scripts/sim/run_simulation.py').encode())
                     or process_pid(str(PROJECT / 'scripts/sim/run_simulation.py').encode()),
        'backup': process_pid(str(PROJECT / 'deploy/vast/backup_checkpoint.py').encode()),
    }


def verify_and_stop_completed(state):
    # Retain the original final offsite verification, with a local score hash check.
    score = RESULTS / f'{PREFIX}-shared-score.json'
    if not score.is_file() or sha256(score) != state.get('score_sha256'):
        raise ValueError('Completed pipeline score hash does not match')
    spec = importlib.util.spec_from_file_location('_completion_watcher', WATCHER)
    watcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher)
    remote = 'no_smoking_drive:No_SmokingZone_EXP_Backups/runs'
    if not (watcher.drive_file_verified(score, f'{remote}/{score.name}')
            and watcher.drive_file_verified(PIPELINE, f'{remote}/{PIPELINE.name}')):
        event('completion_waiting_for_drive')
        return False
    watcher.stop_instance(JOURNAL, 'pipeline_complete')
    return True


def recover_once():
    state = json.loads(PIPELINE.read_text())
    if state.get('config') != json.loads(CONFIG.read_text()):
        raise ValueError('Pipeline configuration changed')
    processes = current_processes()
    action = decide_action(state, **processes)
    if action == 'resume':
        validate()
        validate_extension()
        # Persist the failed status before the supervisor replaces its status file.
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        folder = RESULTS / f'{PREFIX}-recovery-events'
        folder.mkdir(exist_ok=True)
        (folder / (stamp + '.json')).write_text(json.dumps(state), encoding='utf-8')
        event('automatic_recovery_start', old_status=state.get('status'), phase=state.get('phase'))
        start_model()
        pid = start_supervisor()
        event('automatic_recovery_started', supervisor_pid=pid, night_progress_sha256=MANIFEST_SHA256)
    elif action == 'verify_completion':
        if verify_and_stop_completed(state):
            return 'complete'
    elif action == 'work_in_progress' and not healthy():
        # If a server exited, restore the same model while the existing workers retry.
        # A live but unresponsive server is logged rather than killing unrelated jobs.
        if process_pid(b'sglang.launch_server') is None:
            start_model()
        else:
            event('model_unhealthy_existing_process', **processes)
    return action


def main(dry_run=False):
    validate()
    validate_extension()
    if dry_run:
        state = json.loads(PIPELINE.read_text())
        processes = current_processes()
        print(json.dumps({'status': state['status'], 'action': decide_action(state, **processes),
                          'model_healthy': healthy(), **processes}))
        return
    import fcntl
    lock = (RESULTS / f'{PREFIX}-persistent-recovery.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        event('recovery_controller_already_running')
        return
    event('persistent_recovery_started', pid=os.getpid(), poll_seconds=15)
    failures = 0
    previous = None
    while True:
        try:
            action = recover_once()
            if action != previous:
                event('recovery_controller_state', action=action, **current_processes())
                previous = action
            failures = 0
            if action == 'complete':
                return
            delay = 15
        except Exception as exc:
            failures += 1
            delay = min(300, 15 * 2 ** min(failures, 5))
            event('automatic_recovery_retry', error_type=type(exc).__name__,
                  error=str(exc)[:300], retry_in_seconds=delay)
        time.sleep(delay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args().dry_run)

