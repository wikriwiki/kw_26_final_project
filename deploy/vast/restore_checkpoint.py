"""Download, verify and reconstruct a checkpoint in a NEW recovery directory.

No live database is overwritten. The returned dump can be imported separately;
bind_resume_snapshot then records the verified recovery lineage in that new DB.
"""
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deploy.vast.backup_checkpoint import hash_file, remote_md5, require, run_quiet


def download(rclone, config, remote, local, sha=None):
    require(not local.exists(), 'Recovery download destination already exists')
    run_quiet([str(rclone), '--config', str(config), 'copyto', remote, str(local), '--stats', '0'])
    require(hash_file(local, 'md5') == remote_md5(rclone, config, remote), 'Recovery MD5 mismatch')
    if sha:
        require(hash_file(local) == sha, 'Recovery SHA256 mismatch')


def committed_record(rclone, config, remote, folder, expected=None):
    require(bool(re.fullmatch(r'[A-Za-z0-9_]+:[A-Za-z0-9_./-]+', remote)) and '..' not in remote,
            'Invalid recovery remote')
    folder.mkdir(parents=True)
    marker = folder / 'committed.json'
    metadata = folder / 'checkpoint.json'
    download(rclone, config, remote + '/committed.json', marker)
    commit = json.loads(marker.read_text(encoding='utf-8'))
    require(commit.get('complete') is True, 'Checkpoint is not committed')
    if expected:
        require(commit.get('checkpoint_sha256') == expected, 'Parent checkpoint identity changed')
    download(rclone, config, remote + '/checkpoint.json', metadata, commit['checkpoint_sha256'])
    record = json.loads(metadata.read_text(encoding='utf-8'))
    require(record['day'] == commit['day'], 'Commit date differs from checkpoint')
    return record


def unpack_run(archive, root):
    with tarfile.open(archive) as source:
        for member in source:
            parts = PurePosixPath(member.name).parts
            require(member.isfile() and len(parts) >= 2 and parts[0] == 'run'
                    and not any(p in ('..', '.', '') for p in parts), 'Unsafe recovery archive member')
            target = root.joinpath(*parts[1:])
            require(target.resolve().is_relative_to(root.resolve()), 'Recovery archive escapes destination')
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as handle, target.open('wb') as out:
                shutil.copyfileobj(handle, out)


def restore(rclone, config, remote, target):
    target = Path(target).resolve()
    require(not target.exists(), 'Recovery must use a new directory')
    target.mkdir(parents=True)
    record = committed_record(rclone, config, remote, target / 'latest')
    run = target / 'run'
    run.mkdir()
    parents = record.get('parent_checkpoints', [])
    days = [p['day'] for p in parents]
    require(days == sorted(set(days)), 'Parent checkpoint dates are duplicated or out of order')
    all_records = []
    for index, parent in enumerate(parents):
        folder = target / f'parent-{index:03d}'
        previous = committed_record(rclone, config, parent['remote'], folder, parent['checkpoint_sha256'])
        require(previous['day'] < record['day'] and previous['run_id'] == record['run_id']
                and previous['day'] == parent['day'] and previous['arm'] == record['arm']
                and previous['phase'] == record['phase'], 'Foreign parent checkpoint')
        all_records.append((parent['remote'], folder, previous))
    all_records.append((remote, target / 'latest', record))
    for prefix, folder, item in all_records:
        archive = folder / 'run-day.tar.gz'
        download(rclone, config, prefix + '/run-day.tar.gz', archive, item['files']['run-day.tar.gz']['sha256'])
        unpack_run(archive, run)
        if item.get('kind', 'complete_day') == 'complete_day':
            receipt = {'day': item['day'], 'run_id': item['run_id'], 'arm': item['arm'],
                'kind': 'complete_day', 'remote': prefix,
                'checkpoint_sha256': hash_file(folder / 'checkpoint.json'),
                'graph_sha256': item['files']['neo4j.dump']['sha256'],
                'verified_at_utc': item['created_at_utc']}
            (run / f"backup_completed_{item['day']}.json").write_text(json.dumps(receipt), encoding='utf-8')
    require(hash_file(run / 'experiment_run.json') == record['source_manifest_sha256'],
            'Restored source manifest differs from checkpoint')
    dump = target / 'neo4j.dump'
    download(rclone, config, remote + '/neo4j.dump', dump, record['files']['neo4j.dump']['sha256'])
    manifest = json.loads((run / 'experiment_run.json').read_text(encoding='utf-8'))
    proof = {'remote': remote, 'graph_sha256': hash_file(dump),
             'original_snapshot_sha256': manifest['snapshot_sha256'],
             'run_id': record['run_id'], 'arm': record['arm'], 'day': record['day'],
             'run_directory': str(run), 'restoration_verified': True}
    (target / 'restoration.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    return proof


def bind_resume_snapshot(proof, session):
    """Call only in a DB newly imported from this verified recovery dump."""
    require(proof.get('restoration_verified') is True, 'Recovery has not been verified')
    with session.begin_transaction() as tx:
        rows = list(tx.run('MATCH (s:ExperimentSnapshot) RETURN s.id AS id'))
        require(len(rows) == 1 and rows[0]['id'] == proof['graph_sha256'],
                'Recovery DB is not marked with the verified checkpoint dump')
        tx.run('MATCH (s:ExperimentSnapshot) SET s.id=$original, s.sha256=$original, '
               's.recovery_dump_sha256=$dump, s.recovery_remote=$remote',
               original=proof['original_snapshot_sha256'], dump=proof['graph_sha256'],
               remote=proof['remote']).consume()
        tx.commit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(restore(Path(os.environ['BACKUP_RCLONE_BINARY']),
          Path(os.environ['BACKUP_RCLONE_CONFIG']), args.remote, args.out)))
