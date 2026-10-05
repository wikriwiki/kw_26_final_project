"""Read graph outboxes and verify retained days; repair only missing file copies."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path('/workspace/no-smoking-project-v22-perf-final')
RUN = Path('/workspace/no-smoking-results/integration-main-v22-1154-shared-pre')
sys.path.insert(0, str(PROJECT / 'scripts/sim'))
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(PROJECT))
from evidence_integrity import canonical, verify
from experience_provenance import atomic_text
from day_resume import read_metric_rows
from neo4j import GraphDatabase


def completed_hashes():
    result = {}
    for marker in sorted(RUN.glob('backup_completed_*.json')):
        day = marker.stem.removeprefix('backup_completed_')
        for path in (marker, RUN / f'metrics/day_{day}.jsonl', RUN / f'night2_completed_{day}.json'):
            result[str(path.relative_to(RUN))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def check(env, *, reconcile=False):
    """Caller must first stop all writers when reconcile=True."""
    manifest = json.loads((RUN / 'experiment_run.json').read_text())
    cohort = set(manifest['cohort_ids'])
    assert len(cohort) == 1154 and manifest['phase'] == 'shared_pre'
    driver = GraphDatabase.driver(env['NEO4J_URI'],
                                  auth=(env.get('NEO4J_USER', 'neo4j'), env['NEO4J_PASSWORD']))
    report = {}
    try:
        with driver.session(database=env.get('NEO4J_DATABASE', 'neo4j')) as session:
            for path in sorted((RUN / 'metrics').glob('day_*.jsonl')):
                day = path.stem.removeprefix('day_')
                rows = read_metric_rows(path) if reconcile else [json.loads(x) for x in path.read_text().splitlines() if x]
                file_rows = {}
                for row in rows:
                    verify(row)
                    assert row['aid'] in cohort and row['aid'] not in file_rows
                    assert row['experience_day'] == day and row['experience_run_id'] == manifest['run_id']
                    assert row['status'] in ('ok', 'skipped')
                    file_rows[row['aid']] = row
                graph_rows = {}
                entries = session.run('MATCH (s:State) WHERE s.day=date($day) AND s.experience_run_id=$run '
                    'RETURN s.id AS id, s.agent_metrics_json AS metrics', day=day, run=manifest['run_id'])
                for entry in entries:
                    value = json.loads(entry['metrics'])
                    verify(value)
                    aid = value['aid']
                    assert aid in cohort and aid not in graph_rows and entry['id'] == f'{aid}_{day}'
                    assert value['experience_day'] == day and value['experience_run_id'] == manifest['run_id']
                    assert value['status'] in ('ok', 'skipped')
                    graph_rows[aid] = value
                assert set(file_rows).issubset(graph_rows), f'{day}: file row has no graph outbox'
                assert all(row == graph_rows[aid] for aid, row in file_rows.items()), f'{day}: outbox mismatch'
                recovered = []
                complete = (RUN / f'backup_completed_{day}.json').exists()
                if complete:
                    assert set(file_rows) == set(graph_rows) == cohort, f'{day}: completed day changed'
                elif reconcile:
                    recovered = sorted(graph_rows.keys() - file_rows.keys())
                    if recovered:
                        # Preserve all existing valid rows/order and copy only
                        # already committed, sealed DB receipts. No recomputation.
                        rows += [graph_rows[aid] for aid in recovered]
                        atomic_text(path, ''.join(canonical(row) + '\n' for row in rows))
                        file_rows.update((aid, graph_rows[aid]) for aid in recovered)
                    assert file_rows == graph_rows
                report[day] = {'metrics': len(file_rows), 'graph_outbox': len(graph_rows),
                    'statuses': dict(Counter(r['status'] for r in file_rows.values())),
                    'outboxes_recovered': len(recovered), 'complete': complete,
                    'metrics_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    finally:
        driver.close()
    return report
