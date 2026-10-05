#!/usr/bin/env python3
"""Archive retired Neo4j pairs (stopped, earlier attempts) to Drive, verify, then delete.

Run on Vast. For each pair and arm: offline dump -> SHA256/MD5 -> rclone upload
(immutable) -> remote MD5 check -> delete the local dump. Only after every arm of
a pair is verified on Drive is the pair directory removed. Never touches a pair
whose Neo4j is running, the active experiment pair, or anything outside the
explicit list. Waits while an experiment backup is running so uploads do not
compete with it.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PAIRS = {'main-v15': Path('/workspace/no-smoking-neo4j-main-v15'),
         'main-v17': Path('/workspace/no-smoking-neo4j-main-v17')}
TMP = Path('/workspace/no-smoking-retire-tmp')
RCLONE, RCONF = '/workspace/bin/rclone', '/workspace/no-smoking-drive.conf'
REMOTE = 'no_smoking_drive:No_SmokingZone_EXP_Backups/retired'
LOG = Path('/workspace/no-smoking-results/retire-old-pairs.log')
RECEIPTS = Path('/workspace/no-smoking-checkpoints/retired-pairs')


def say(msg):
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as out:
        out.write(line + '\n')


def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(argv, env=None, timeout=3600):
    r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f'{Path(argv[0]).name} exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}')
    return r.stdout


def busy(pattern):
    return subprocess.run(['pgrep', '-f', pattern], capture_output=True).returncode == 0


def jvm_using(path):
    """True if a real Java process (argv[0] is java) references this directory."""
    needle = (str(path).rstrip('/') + '/').encode()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().split(bytes(1))
        except OSError:
            continue
        if args and args[0].endswith(b'/java') and any(needle in a for a in args):
            return True
    return False


def wait_for_experiment_backup():
    while busy('deploy/vast/backup_checkpoint[.]py'):
        say('experiment backup running; waiting 60s')
        time.sleep(60)


def upload_verified(local, remote):
    wait_for_experiment_backup()
    run([RCLONE, '--config', RCONF, 'copyto', str(local), remote, '--immutable',
         '--retries', '5', '--low-level-retries', '10', '--stats', '0'], timeout=7200)
    listed = run([RCLONE, '--config', RCONF, 'md5sum', remote], timeout=600).split()
    expected = digest(local, 'md5')
    if not listed or listed[0] != expected:
        raise RuntimeError(f'Drive MD5 mismatch for {remote}: {listed[:1]} != {expected}')
    return expected


def neo_env(home):
    env = {k: v for k, v in os.environ.items()
           if not any(t in k.upper() for t in ('PASSWORD', 'TOKEN', 'SECRET', 'API_KEY'))}
    env.update(NEO4J_HOME=str(home), NEO4J_CONF=str(home / 'conf'))
    return env


def retire(name, root):
    if not root.is_dir():
        say(f'{name}: already absent')
        return
    if jvm_using(root):
        raise RuntimeError(f'{name}: a JVM is still using {root}; refusing')
    manifest = root / 'pair-manifest.json'
    receipt = {'pair': name, 'root': str(root), 'retired_at_utc': None,
               'pair_manifest_sha256': digest(manifest, 'sha256'),
               'pair_manifest': json.loads(manifest.read_text()), 'arms': {}}
    for arm in ('off', 'on'):
        home = root / arm
        # Do not trust `neo4j status`: a stale run/neo4j.pid from a crashed instance can
        # point at a reused PID (it matched a simulator thread here). Look for a real JVM
        # using this home instead, and never call `neo4j stop`.
        if jvm_using(home):
            raise RuntimeError(f'{name}/{arm}: a JVM is using {home}; refusing')
        out = TMP / f'{name}-{arm}'
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        say(f'{name}/{arm}: dumping')
        run([str(home / 'bin/neo4j-admin'), 'database', 'dump', 'neo4j', f'--to-path={out}'],
            env=neo_env(home), timeout=3600)
        dump = out / 'neo4j.dump'
        if not dump.is_file() or dump.stat().st_size == 0:
            raise RuntimeError(f'{name}/{arm}: dump missing')
        sha = digest(dump, 'sha256')
        remote = f'{REMOTE}/neo4j-{name}/{arm}/neo4j.dump'
        md5 = upload_verified(dump, remote)
        receipt['arms'][arm] = {'remote': remote, 'bytes': dump.stat().st_size, 'sha256': sha, 'md5': md5}
        say(f'{name}/{arm}: uploaded {dump.stat().st_size} bytes sha256 {sha} md5 verified')
        shutil.rmtree(out)
    receipt['retired_at_utc'] = datetime.now(timezone.utc).isoformat()
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    local_receipt = RECEIPTS / f'{name}.json'
    local_receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    upload_verified(manifest, f'{REMOTE}/neo4j-{name}/pair-manifest.json')
    upload_verified(local_receipt, f'{REMOTE}/neo4j-{name}/retirement-receipt.json')
    if root.resolve() != root or root.parent != Path('/workspace') or not root.name.startswith('no-smoking-neo4j-main-v1'):
        raise RuntimeError(f'{name}: unexpected root {root}; not deleting')
    shutil.rmtree(root)
    say(f'{name}: all arms verified on Drive; removed {root}')


def main():
    names = sys.argv[1:] or list(PAIRS)
    say(f'start: {names}')
    for name in names:
        retire(name, PAIRS[name])
    say('done; ' + run(['df', '-BG', '/workspace']).splitlines()[-1])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        say(f'FAILED: {type(exc).__name__}: {exc}')
        sys.exit(1)
