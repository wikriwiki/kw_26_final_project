"""Commit a night once every pair has a validated answer or an explicit skip."""
import json
import os
import time
from pathlib import Path

from evidence_integrity import seal, verify
from experience_provenance import atomic_json


def validate_accounting(stats, expected):
    processed, skipped = stats.get('processed'), stats.get('skipped', 0)
    if (type(processed) is not int or type(skipped) is not int or min(processed, skipped) < 0
            or stats.get('errors', 0) or processed + skipped != expected
            or stats.get('matched', expected) != expected):
        raise ValueError('Night terminal accounting differs from selected pairs')


def complete_night(today, cohort, skipped_aids, workers, out_dir):
    from night_interaction import select_interaction_pairs
    from night_intent_llm import run_intent_classification, write_conversations
    from no_smoking_context import configured_context
    from neo4j_load._common import driver_session
    import night_store
    out_dir = Path(out_dir)
    day = today.isoformat()
    runtime = configured_context()
    run_id = os.environ.get('SIM_RUN_ID') or str(out_dir.resolve())
    marker_path = out_dir / f'night2_completed_{day}.json'
    t0 = time.time()

    def count():
        with driver_session() as session:
            return session.run('MATCH (c:Conversation) WHERE c.day=date($d) RETURN count(c) AS n',
                               d=day).single()['n']

    existing = count()
    if not marker_path.exists() and runtime:
        # Includes the all-skipped case: a valid outbox can have zero conversations.
        recovered = night_store.load(today, runtime)
        if recovered is not None:
            matched = recovered.get('matched', recovered['processed'])
            validate_accounting(recovered, matched)
            atomic_json(marker_path, seal({'cohort': cohort, 'conversation_count': existing,
                'run_id': run_id, 'arm': runtime.arm, 'day': day, 'status': 'complete',
                'matched': matched, 'skipped': recovered.get('skipped', 0),
                'evidence_ref': recovered['evidence_ref']}))
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        if marker.get('cohort') != cohort or marker.get('conversation_count') != existing:
            raise ValueError('Night completion marker differs from current cohort or graph')
        if runtime:
            verify(marker)
            if (marker.get('run_id'), marker.get('arm'), marker.get('day'), marker.get('status')) != (
                    run_id, runtime.arm, day, 'complete'):
                raise ValueError('Night marker identity mismatch')
    else:
        if existing:
            raise RuntimeError('Night graph writes have no transactional completion outbox')
        options = {'seed': runtime.stable_seed(today, 'night_pairs')} if runtime else {}
        pairs = select_interaction_pairs(today, verbose=False, exclude_agents=skipped_aids, **options)
        print(f'  [Night2] {len(pairs)} pairs; reusing archived answers and six-call budgets', flush=True)
        if pairs:
            stats = run_intent_classification(today, pairs, workers=workers, verbose=True)
        else:
            written = write_conversations(today, [], skipped=[])
            stats = {'processed': 0, 'skipped': 0, 'matched': 0, 'errors': 0,
                     'evidence_ref': written.get('evidence_ref')}
        validate_accounting(stats, len(pairs))
        actual = count()
        if actual != stats['processed']:
            raise RuntimeError('Written Conversation count differs from successful pairs')
        marker = {'cohort': cohort, 'conversation_count': actual, 'matched': len(pairs),
                  'skipped': stats.get('skipped', 0)}
        if runtime:
            marker.update(run_id=run_id, arm=runtime.arm, day=day, status='complete',
                          evidence_ref=stats['evidence_ref'],
                          night_progress_sha256=os.environ.get('NO_SMOKING_NIGHT_PROGRESS_SHA256'))
            marker = seal(marker)
        atomic_json(marker_path, marker)
    print(f"  [Night2] completed: conversations={marker['conversation_count']}, "
          f"skipped={marker.get('skipped', 0)}", flush=True)
    return {'night2_elapsed_sec': time.time() - t0, 'night2_processed': marker['conversation_count'],
            'night2_skipped': marker.get('skipped', 0),
            'night2_matched': marker.get('matched', marker['conversation_count'])}
