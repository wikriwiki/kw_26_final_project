"""Verify completed dates before skipping them; recover only torn append tails."""
import os
import json
from pathlib import Path

from evidence_integrity import canonical, verify
from experience_provenance import atomic_text


def read_metric_rows(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    result = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # A crash can interrupt ONLY the final append. Preserve those bytes
            # before discarding an unparseable tail; the DB outbox recovers it.
            if index != len(lines) - 1 or line.endswith(b'\n'):
                raise
            import hashlib
            target = path.with_name(path.name + '.torn-' + hashlib.sha256(line).hexdigest()[:12])
            if not target.exists():
                target.write_bytes(line)
            atomic_text(path, b''.join(lines[:index]).decode('utf-8'))
            return result
    if raw and not raw.endswith(b'\n'):
        atomic_text(path, raw.decode('utf-8') + '\n')
    return result


def verified_completed_day(root, day, agents, run_id, arm):
    from neo4j_load._common import driver_session
    receipt = Path(root) / f'backup_completed_{day}.json'
    if not receipt.is_file():
        return None
    backup = json.loads(receipt.read_text(encoding='utf-8'))
    if (backup.get('day'), backup.get('run_id'), backup.get('arm'), backup.get('kind')) != (
            day, run_id, arm, 'complete_day'):
        raise ValueError('Completed-day backup receipt has wrong identity')
    rows = read_metric_rows(Path(root) / f'metrics/day_{day}.jsonl')
    seen = {}
    for row in rows:
        verify(row)
        if (row.get('aid') in seen or row.get('status') not in {'ok', 'skipped'}
                or row.get('experience_run_id') != run_id or row.get('experience_day') != day):
            raise ValueError('Cannot resume from incomplete or foreign metrics')
        if row['status'] == 'skipped' and (row.get('attempts') != int(os.environ.get('EXP_AGENT_DAY_MAX_ATTEMPTS', '6'))
                or row.get('skip_kind') != 'failed_after_retries'
                or row.get('observed_behavior') is not False
                or row.get('no_smoking', {}).get('arm') != arm):
            raise ValueError('Skipped day is missing its terminal failure receipt')
        seen[row['aid']] = row
    if set(seen) != set(agents):
        raise ValueError('Completed day does not contain the frozen cohort')
    with driver_session() as session:
        stored = list(session.run('MATCH (s:State) WHERE s.day=date($day) AND s.experience_run_id=$run '
            'RETURN s.id AS id, s.agent_metrics_json AS metrics', day=day, run=run_id))
        if len(stored) != len(agents):
            raise ValueError('Restored graph and completed-day metrics disagree')
        for entry in stored:
            value = json.loads(entry['metrics'])
            if value != seen.get(value.get('aid')):
                raise ValueError('Restored graph metrics outbox differs from the archive')
    from interview_evidence import _read_night_index
    if _read_night_index(Path(root).resolve(), day) is None:
        raise ValueError('Completed day has no verified night evidence')
    saved = json.loads((Path(root) / 'summary.json').read_text(encoding='utf-8'))
    matches = [row for row in saved.get('summary', []) if row.get('day') == day]
    if len(matches) != 1:
        raise ValueError('Completed day has no unique summary')
    if (matches[0].get('ok') != sum(r['status'] == 'ok' for r in seen.values())
            or matches[0].get('skipped', matches[0].get('err', 0))
            != sum(r['status'] == 'skipped' for r in seen.values())):
        raise ValueError('Completed-day summary disagrees with terminal receipts')
    return matches[0]
