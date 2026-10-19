"""Transactional Night2 completion outbox, including its sealed evidence link."""
import json
import os
from pathlib import Path

from evidence_integrity import canonical, digest, seal, verify
from experience_provenance import execution_fingerprint
from neo4j_load._common import driver_session


def identity(day, runtime):
    out = Path(os.environ.get('SIM_OUTPUT_DIR', os.path.expanduser('~/sim_output')))
    return {'run_id': os.environ.get('SIM_RUN_ID') or str(out.resolve()),
            'arm': runtime.arm, 'day': str(day),
            'execution_fingerprint': execution_fingerprint()}


def load(day, runtime):
    expected = identity(day, runtime)
    with driver_session() as session:
        row = session.run('MATCH (n:SimulationNight {id:$id}) RETURN n.payload AS payload',
                          id=digest(expected)).single()
        if row is None:
            return None
        record = json.loads(row['payload'])
        verify(record)
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError('Night completion outbox has a foreign identity')
        count = session.run('MATCH (c:Conversation) WHERE c.day=date($day) RETURN count(c) AS n',
                            day=str(day)).single()['n']
        if count != record['processed']:
            raise ValueError('Night completion outbox differs from conversation count')
    return record


def save(tx, day, runtime, stats):
    record = seal({**identity(day, runtime), **stats})
    tx.run('CREATE (n:SimulationNight {id:$id, payload:$payload})',
           id=digest(identity(day, runtime)), payload=canonical(record)).consume()
    return record
