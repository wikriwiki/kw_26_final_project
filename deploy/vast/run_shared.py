"""Frozen shared-pre -> restored OFF/ON -> evidence audit -> backed-up score.

All phases are child processes of this supervisor; failure blocks every later
phase. Credentials come only from the existing private environment/config.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import time
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/sim'))
from deploy.vast import neo4j_pair
from deploy.vast.backup_checkpoint import hash_file, require, upload_verified, remote_md5
from experience_provenance import atomic_json


def call(argv, *, env, log=None, timeout=None):
    kwargs = {'stdout': log, 'stderr': subprocess.STDOUT} if log is not None else {}
    child = subprocess.Popen([str(arg) for arg in argv], cwd=ROOT, env=env,
                             start_new_session=True, **kwargs)
    try:
        code = child.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, str(argv[0]))
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise


def run(config):
    prefix = config['prefix']
    require(bool(re.fullmatch(r'(integration|main)-[a-z0-9-]+', prefix)), 'Invalid run prefix')
    require(hash_file(Path(config['source_archive'])) == config['source_sha256'], 'Source archive changed')
    source = json.loads((ROOT / 'deployment-manifest.json').read_text(encoding='utf-8'))
    for name, digest in source['file_sha256'].items():
        require(hash_file(ROOT / name) == digest, f'Frozen source changed: {name}')
    if prefix.startswith('main-'):
        gate = json.loads(Path(config['gate_file']).read_text(encoding='utf-8'))
        require(gate.get('status') == 'passed' and gate.get('source_sha256') == config['source_sha256']
                and gate.get('replay_passed') is True and gate.get('restore_passed') is True
                and gate.get('full_shared_pipeline_passed') is True,
                'Main run is blocked until the same source passes integration and restore gates')
    results = Path('/workspace/no-smoking-results')
    status_path = results / (prefix + '-pipeline.json')
    previous = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else None
    if previous:
        require(previous['config'] == config, 'Cannot resume a different pipeline configuration')
        prior_pid = previous.get('pid')
        if prior_pid and prior_pid != os.getpid():
            cmdline = Path(f'/proc/{prior_pid}/cmdline')
            require(not cmdline.exists() or b'run_shared.py' not in cmdline.read_bytes(),
                    'Another supervisor is still running')
    bundle = Path(config['bundle'])
    runtime = json.loads((bundle / 'runtime.json').read_text(encoding='utf-8'))
    require(len(runtime['cohort']) == config['cohort_size'], 'Pipeline cohort size changed')
    baseline = Path(config['baseline'])
    require(hash_file(baseline) == config['baseline_sha256'], 'Baseline hash mismatch')
    grammar_mode = config.get('json_grammar_mode', 'json_schema')
    require(grammar_mode in {'json_schema', 'json_object'}, 'Invalid JSON grammar mode')
    llm_timeout_seconds = config.get('llm_timeout_seconds', 180)
    require(type(llm_timeout_seconds) is int and 30 <= llm_timeout_seconds <= 1800,
            'Invalid LLM request timeout')
    phase_timeout = config.get('phase_timeout')
    require(phase_timeout is None or (type(phase_timeout) is int and phase_timeout > 0),
            'Invalid phase timeout')
    env = dict(os.environ, PYTHONUNBUFFERED='1', POLICY_BACKTEST_DETERMINISTIC='1',
               SIM_JSON_GRAMMAR_MODE=grammar_mode,
               SIM_LLM_TIMEOUT_SECONDS=str(llm_timeout_seconds),
               SIM_TOKENIZER_PATH=str(ROOT / 'output/experiments/no_smoking_zone/runtime/tokenizer'),
               NO_SMOKING_SERVER_CONFIG=str(ROOT / 'output/experiments/no_smoking_zone/runtime/server-config.json'),
               SGLANG_BASE_URL='http://127.0.0.1:8000/v1', LLM_MODE='exaone_4_5',
               SIM_POST_DAY_BACKUP_HOOK=str(ROOT / 'deploy/vast/backup_checkpoint.py'),
               BACKUP_RCLONE_BINARY='/workspace/bin/rclone', BACKUP_RCLONE_CONFIG='/workspace/no-smoking-drive.conf',
               BACKUP_DRIVE_REMOTE='no_smoking_drive:No_SmokingZone_EXP_Backups',
               BACKUP_CHECKPOINT_ROOT='/workspace/no-smoking-checkpoints')
    rclone, rconfig = Path(env['BACKUP_RCLONE_BINARY']), Path(env['BACKUP_RCLONE_CONFIG'])
    drive = env['BACKUP_DRIVE_REMOTE']
    if previous and previous.get('status') == 'complete':
        score = results / (prefix+'-shared-score.json')
        require(hash_file(score) == previous['score_sha256'], 'Completed score has changed')
        upload_verified(rclone, rconfig, score, f'{drive}/runs/{score.name}')
        upload_verified(rclone, rconfig, status_path, f'{drive}/runs/{status_path.name}')
        print(json.dumps({'status':'already_complete', 'prefix':prefix}), flush=True)
        return
    status = {'prefix': prefix, 'config': config, 'status': 'running', 'phase': 'preparing',
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid()}
    if previous:
        status['started_at_utc'] = previous['started_at_utc']
    results.mkdir(exist_ok=True)

    def phase(name):
        status.update(phase=name, updated_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(status_path, status)
        print(json.dumps({'phase': name, 'prefix': prefix}), flush=True)

    def create_pair(label, dump, digest, port):
        existing = Path(f'/workspace/no-smoking-neo4j-{prefix}-{label}')
        if existing.exists():
            record = json.loads((existing/'pair-manifest.json').read_text(encoding='utf-8'))
            require(record['source_dump_sha256'] == digest and record['root'] == str(existing),
                    'Existing pair has a different snapshot')
            cfg = {'root': existing, 'sha256': digest, 'ports': {'off': port, 'on': port+1}}
            for arm, number in cfg['ports'].items():
                require(record['arms'][arm]['uri'] == f'bolt://127.0.0.1:{number}',
                        'Existing pair has different database ports')
                try:
                    with socket.create_connection(('127.0.0.1', number), timeout=2):
                        pass
                except OSError:
                    neo4j_pair.command(existing/arm, ['neo4j', 'start'], 180)
                    deadline=time.monotonic()+180
                    while True:
                        try:
                            with socket.create_connection(('127.0.0.1', number),timeout=2): break
                        except OSError:
                            require(time.monotonic()<deadline,'Restored DB did not start')
                            time.sleep(2)
            return cfg
        pair_env = dict(env, CLEAN_BASELINE_DUMP=str(dump), NO_SMOKING_SNAPSHOT_SHA256=digest,
            NO_SMOKING_NEO4J_ROOT=f'/workspace/no-smoking-neo4j-{prefix}-{label}',
            NO_SMOKING_OFF_BOLT_PORT=str(port), NO_SMOKING_ON_BOLT_PORT=str(port+1),
            NO_SMOKING_NEO4J_MIN_FREE_GB='10')
        cfg = neo4j_pair.settings(pair_env)
        neo4j_pair.check_input(cfg)
        neo4j_pair.execute(cfg)
        return cfg

    def run_phase(name, arm, pair, branch=None):
        phase(name)
        output = results / f'{prefix}-{name}'
        start = '2017-11-19' if branch is None else '2017-12-03'
        phase_name = 'shared_pre' if branch is None else 'post_branch'
        child = dict(env, NO_SMOKING_SNAPSHOT_SHA256=pair['sha256'],
            NO_SMOKING_OFF_NEO4J_URI=f"bolt://127.0.0.1:{pair['ports']['off']}",
            NO_SMOKING_ON_NEO4J_URI=f"bolt://127.0.0.1:{pair['ports']['on']}",
            BACKUP_NEO4J_HOME=str(pair['root']/arm), BACKUP_NEO4J_BOLT_PORT=str(pair['ports'][arm]))
        cmd = [sys.executable, ROOT/'scripts/experiments/no_smoking_zone.py', 'run',
               '--bundle', bundle, '--arm', arm, '--phase', phase_name, '--start', start,
               '--days', '14', '--workers', str(config['workers']), '--out', output]
        if branch:
            cmd += ['--branch-manifest', branch]
        manifest_path=output/'experiment_run.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
        if manifest is None or manifest.get('status') != 'complete':
            if manifest:
                cmd.append('--resume')
            with (results / f'{prefix}-{name}.run.log').open('ab') as log:
                call(cmd, env=child, log=log, timeout=phase_timeout)
        # Read-only evidence census is a gate for both the pilot and production.
        end = '2017-12-02' if branch is None else '2017-12-16'
        audit = output / 'evidence-audit.json'
        if not audit.exists():
            pending_audit = audit.with_suffix('.pending.json')
            with pending_audit.open('wb') as log:
                call([sys.executable, ROOT/'scripts/sim/interview_evidence.py', '--run-dir', output,
                      '--audit-all', '--through-day', end], env=child, log=log, timeout=3600)
            pending_audit.replace(audit)
        report = json.loads(audit.read_text(encoding='utf-8'))
        require(report.get('status') == 'passed', 'Evidence census did not pass')
        upload_verified(rclone, rconfig, audit, f'{drive}/runs/{output.name}/evidence-audit.json')
        return output

    try:
        phase('restore-shared-pre')
        pre_manifest=results/f'{prefix}-shared-pre/experiment_run.json'
        pre_complete=(pre_manifest.is_file() and json.loads(pre_manifest.read_text(encoding='utf-8')).get('status')=='complete')
        pair = ({'root':Path(f'/workspace/no-smoking-neo4j-{prefix}-pre'),
                 'sha256':config['baseline_sha256'],
                 'ports':{'off':config['port_base'],'on':config['port_base']+1}}
                if pre_complete else create_pair('pre', baseline, config['baseline_sha256'], config['port_base']))
        pre = run_phase('shared-pre', 'off', pair)
        receipt = json.loads((pre/'backup_completed_2017-12-02.json').read_text(encoding='utf-8'))
        checkpoint_dirs = Path(env['BACKUP_CHECKPOINT_ROOT']) / pre.name / '2017-12-02'
        dumps = [p for p in checkpoint_dirs.glob('*/neo4j.dump') if hash_file(p) == receipt['graph_sha256']]
        require(len(dumps) == 1, 'Expected exactly one retained verified Dec-2 dump')
        dump = dumps[0]
        require(remote_md5(rclone, rconfig, receipt['remote']+'/neo4j.dump') == hash_file(dump, 'md5'),
                'Branch dump no longer matches Drive')
        branch = results / (prefix+'-branch.json')
        if not branch.exists():
            call([sys.executable, ROOT/'scripts/experiments/no_smoking_zone.py', 'record-branch',
                  '--pre', pre, '--dump', dump, '--out', branch], env=env)
        upload_verified(rclone, rconfig, branch, f'{drive}/runs/{prefix}-branch.json')
        # This pair was created by this supervisor. Preserve its complete dump
        # on Drive before releasing the temporary imported copies on local disk.
        phase('retire-backed-up-pre-pair')
        root = pair['root'].resolve()
        require(root == Path(f'/workspace/no-smoking-neo4j-{prefix}-pre'), 'Unexpected temporary pair root')
        if root.exists():
            require((root/'pair-manifest.json').is_file() and not dump.is_relative_to(root),
                    'Refusing to remove an untracked pair or its only dump')
            for arm in ('off', 'on'):
                neo4j_pair.command(root/arm, ['neo4j', 'stop'], 180)
            shutil.rmtree(root)
        phase('restore-post-branches')
        post_pair = create_pair('post', dump, receipt['graph_sha256'], config['port_base']+2)
        off = run_phase('post-off', 'off', post_pair, branch)
        on = run_phase('post-on', 'on', post_pair, branch)
        phase('score-and-backup')
        score = results / (prefix+'-shared-score.json')
        if not score.exists():
            call([sys.executable, ROOT/'scripts/experiments/no_smoking_zone.py', 'score-shared',
                  '--pre', pre, '--off', off, '--on', on, '--branch-manifest', branch, '--out', score], env=env)
        upload_verified(rclone, rconfig, score, f'{drive}/runs/{score.name}')
        status.update(status='complete', phase='complete', score_sha256=hash_file(score),
                      completed_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(status_path, status)
        upload_verified(rclone, rconfig, status_path, f'{drive}/runs/{status_path.name}')
        print(json.dumps({'status':'complete','prefix':prefix,'score_sha256':hash_file(score)}), flush=True)
    except BaseException as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:300])
        atomic_json(status_path, status)
        raise


if __name__ == '__main__':
    def terminate(_signum, _frame):
        raise SystemExit('Pipeline interrupted')
    signal.signal(signal.SIGTERM, terminate)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    run(json.loads(args.config.read_text(encoding='utf-8')))
