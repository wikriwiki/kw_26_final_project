#!/usr/bin/env python3
"""Seed a new run with the verified, immutable v19 first day.

The day-one graph is restored separately by run_shared.py. This script only
checks the offsite checkpoint, unpacks its exact day-one files into a new run
directory, and records the source transition for later evidence audits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/sim'))
from deploy.vast.backup_checkpoint import hash_file, require
from scripts.experiments.no_smoking_zone import (
    digest, file_hash, reference_hashes, validate_server_config, write_json,
)
from evidence_integrity import verify
from experience_provenance import source_fingerprint


DAY = '2017-11-19'
NEXT = '2017-11-20'


def adopt(checkpoint: Path, receipt_path: Path, output: Path,
          previous_source_sha256: str, new_source_sha256: str,
          server_config_path: Path | None = None,
          workers: int | None = None,
          llm_timeout_seconds: int | None = None) -> dict:
    require(not output.exists(), 'New run directory already exists')
    require(checkpoint.is_dir() and not checkpoint.is_symlink(), 'Missing checkpoint directory')
    record_path = checkpoint / 'checkpoint.json'
    marker_path = checkpoint / 'committed.json'
    archive_path = checkpoint / 'run-day.tar.gz'
    graph_path = checkpoint / 'neo4j.dump'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    require(record.get('kind') == 'complete_day' and record.get('day') == DAY
            and record.get('phase') == 'shared_pre' and record.get('arm') == 'off',
            'Checkpoint is not the completed first shared day')
    require(marker == {'checkpoint_sha256': hash_file(record_path), 'day': DAY,
                       'complete': True}, 'Checkpoint commit marker differs')
    require(receipt.get('checkpoint_sha256') == marker['checkpoint_sha256']
            and receipt.get('graph_sha256') == record['files']['neo4j.dump']['sha256']
            and receipt.get('run_id') == record['run_id']
            and receipt.get('day') == DAY and receipt.get('kind') == 'complete_day',
            'First-day receipt differs from the offsite checkpoint')
    for path in (archive_path, graph_path):
        metadata = record['files'][path.name]
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_size == metadata['bytes']
                and hash_file(path) == metadata['sha256'],
                f'Checkpoint payload changed: {path.name}')
    require(len(previous_source_sha256) == len(new_source_sha256) == 64,
            'Missing source archive identity')
    code_files = sorted((ROOT / 'scripts/sim').rglob('*.py')) + [ROOT / 'scripts/experiments/no_smoking_zone.py']
    new_code_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in code_files}
    with tarfile.open(archive_path, 'r:gz') as tar:
        members = tar.getmembers()
        require(len(members) == record['archive_file_count'], 'Archive member count changed')
        require(len({member.name for member in members}) == len(members),
                'Archive contains duplicate member paths')
        for member in members:
            parts = Path(member.name).parts
            require(member.isfile() and len(parts) >= 2 and parts[0] == 'run'
                    and all(part not in {'', '.', '..'} for part in parts),
                    'Unsafe or unexpected day-one archive member')
        manifest = json.load(tar.extractfile('run/experiment_run.json'))
        archived_manifest = tar.extractfile('run/experiment_run.json').read()
        import hashlib
        require(hashlib.sha256(archived_manifest).hexdigest() == record['source_manifest_sha256'],
                'Archived manifest differs from checkpoint provenance')
        require(manifest.get('run_id') == record['run_id']
                and manifest.get('phase') == 'shared_pre'
                and manifest.get('arm') == 'off'
                and manifest.get('start') == DAY and manifest.get('days') == 14,
                'Archived run manifest is not the expected shared cohort')
        require(hash_file(archive_path) == record['files']['run-day.tar.gz']['sha256'],
                'Archive checksum changed during inspection')
        row_member = tar.extractfile(f'run/metrics/day_{DAY}.jsonl')
        rows = [json.loads(line) for line in row_member if line.strip()]
        for row in rows:
            verify(row)
        cohort = set(manifest['cohort_ids'])
        require(len(cohort) == len(manifest['cohort_ids']) == 1154
                and {r['aid'] for r in rows} == cohort and len(rows) == len(cohort)
                and all(r.get('status') == 'ok' and r.get('experience_day') == DAY
                        and r.get('experience_run_id') == record['run_id'] for r in rows),
                'Archived first day is not 1,154 unique completed agent-days')
        old_sources = {r['source_fingerprint'] for r in rows}
        require(len(old_sources) == 1, 'Archived first day mixes source versions')
        require(manifest['reference_sha256'] == reference_hashes(),
                'Statistical reference inputs changed between first and second day')
        output.mkdir(parents=True)
        try:
            for member in members:
                target = output.joinpath(*Path(member.name).parts[1:])
                require(target.is_relative_to(output), 'Archive member escapes run directory')
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open('xb') as dest:
                    shutil.copyfileobj(source, dest)
        except BaseException:
            shutil.rmtree(output)
            raise
    manifest['snapshot_sha256'] = record['files']['neo4j.dump']['sha256']
    prior_server_config = manifest['server_config']
    if server_config_path is not None:
        new_server_config = validate_server_config(json.loads(server_config_path.read_text(encoding='utf-8')))
        require(new_server_config['model_revision'] == prior_server_config['model_revision'],
                'Inherited day and new day must use identical model weights')
        manifest['server_config'] = new_server_config
        if new_server_config.get('grammar_backend') == 'outlines':
            manifest['engine_settings']['SIM_JSON_GRAMMAR_MODE'] = 'json_object'
    previous_workers = manifest['workers']
    if workers is not None:
        require(type(workers) is int and 1 <= workers <= 32, 'Invalid inherited run worker setting')
        manifest['workers'] = workers
    if llm_timeout_seconds is not None:
        require(type(llm_timeout_seconds) is int and 30 <= llm_timeout_seconds <= 1800,
                'Invalid inherited run LLM timeout')
        manifest['engine_settings']['SIM_LLM_TIMEOUT_SECONDS'] = str(llm_timeout_seconds)
    manifest['code_sha256'] = new_code_hashes
    manifest['engine_settings']['SIM_POST_DAY_BACKUP_HOOK'] = str(ROOT / 'deploy/vast/backup_checkpoint.py')
    manifest['status'] = 'failed'  # run_shared resumes after independent graph validation.
    manifest.pop('exit_code', None)
    manifest['source_transition'] = {
        'inherited_day': DAY, 'effective_day': NEXT,
        'previous_run_id': record['run_id'],
        'previous_source_fingerprint': next(iter(old_sources)),
        'new_source_fingerprint': source_fingerprint(),
        'previous_source_archive_sha256': previous_source_sha256,
        'new_source_archive_sha256': new_source_sha256,
        'previous_server_config_sha256': digest(prior_server_config),
        'new_server_config_sha256': digest(manifest['server_config']),
        'previous_workers': previous_workers,
        'new_workers': manifest['workers'],
        'new_llm_timeout_seconds': llm_timeout_seconds,
        'checkpoint_sha256': marker['checkpoint_sha256'],
        'graph_sha256': record['files']['neo4j.dump']['sha256'],
        'remote': receipt['remote'],
    }
    write_json(output / 'experiment_run.json', manifest)
    shutil.copy2(receipt_path, output / f'backup_completed_{DAY}.json')
    # The inherited graph is already a verified offsite recovery point. Keep
    # that receipt as the current recoverable backup so Day 2 can start without
    # making a redundant full graph dump before the ten-hour backup interval.
    shutil.copy2(receipt_path, output / 'recoverable_backup.json')
    return {'run_dir': str(output), 'day_one_agents': len(rows),
            'resume_from': NEXT, 'checkpoint_sha256': marker['checkpoint_sha256'],
            'graph_sha256': manifest['snapshot_sha256'],
            'new_source_fingerprint': manifest['source_transition']['new_source_fingerprint']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--previous-source-sha256', required=True)
    parser.add_argument('--new-source-sha256', required=True)
    parser.add_argument('--server-config', type=Path)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--llm-timeout-seconds', type=int)
    args = parser.parse_args()
    print(json.dumps(adopt(args.checkpoint, args.receipt, args.out,
                           args.previous_source_sha256, args.new_source_sha256,
                           args.server_config, args.workers, args.llm_timeout_seconds)))
