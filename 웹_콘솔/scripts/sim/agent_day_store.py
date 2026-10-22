"""Atomic agent-day writes and a transactional metrics outbox in Neo4j State.

The existing graph uses agent/day IDs. A different run may not overwrite that
slot. Use an isolated database or an explicit offline migration for a new run.
"""
from contextlib import contextmanager
import json

from neo4j_load._common import driver_session
from evidence_integrity import EvidenceError, canonical, seal, verify
from experience_provenance import execution_fingerprint


class AlreadyCommitted(Exception):
    def __init__(self, result):
        super().__init__('agent day is already committed')
        self.result = result


READ = '''MATCH (s:State {id:$sid})
RETURN s.experience_run_id AS run_id, s.agent_metrics_json AS metrics'''


def _read(session, aid, today, run_id):
    row = session.run(READ, sid=f'{aid}_{today}').single()
    if row is None:
        return None
    if row['run_id'] != run_id or not row['metrics']:
        raise EvidenceError('existing State belongs to another run or legacy format; refusing overwrite')
    result = json.loads(row['metrics'])
    verify(result)
    if result.get('execution_fingerprint') != execution_fingerprint():
        raise EvidenceError('execution source/settings changed; use a new isolated run')
    if result.get('aid') != aid or result.get('experience_day') != str(today) or result.get('experience_run_id') != run_id:
        raise EvidenceError('committed result identity mismatch')
    return result


def load_completed(aid, today, run_id):
    with driver_session() as session:
        return _read(session, aid, today, run_id)


@contextmanager
def transaction(aid, today, run_id):
    with driver_session() as session:
        with session.begin_transaction() as tx:
            # Serialize writes for an existing agent; no global simulation lock.
            found = tx.run('''MATCH (a:Agent {id:$aid})
                SET a.execution_lock = coalesce(a.execution_lock,0) + 1
                RETURN count(a) AS n''', aid=aid).single()
            if not found or found['n'] != 1:
                raise EvidenceError('exactly one Agent is required')
            existing = _read(tx, aid, today, run_id)
            if existing is not None:
                raise AlreadyCommitted(existing)
            yield tx
            tx.commit()


def save_result(tx, result):
    result = seal(result)
    row = tx.run('''MATCH (s:State {id:$sid})
        SET s.experience_run_id=$run_id, s.agent_metrics_json=$metrics
        RETURN count(s) AS n''', sid=f"{result['aid']}_{result['experience_day']}",
        run_id=result['experience_run_id'], metrics=canonical(result)).single()
    if not row or row['n'] != 1:
        raise EvidenceError('State missing while committing metrics outbox')
    return result
