"""Opt-in real DB tests; use a disposable Neo4j via NEO4J_TEST_URI only."""
from contextlib import contextmanager
from datetime import date
import os
from pathlib import Path
import sys
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/sim'))
import agent_day_store as store
from evidence_contract import EvidenceError
from experience_provenance import execution_fingerprint
from plan_writer import write_plan

pytestmark = pytest.mark.skipif(not os.environ.get('NEO4J_TEST_URI'), reason='NEO4J_TEST_URI not configured')
DAY = date(2026, 9, 15)


@pytest.fixture
def graph(monkeypatch):
    from neo4j import GraphDatabase
    password = os.environ.get('NEO4J_TEST_PASSWORD')
    auth = (os.environ.get('NEO4J_TEST_USER', 'neo4j'), password) if password else None
    driver = GraphDatabase.driver(os.environ['NEO4J_TEST_URI'], auth=auth)
    driver.verify_connectivity()
    aid = 'experience_test_' + uuid4().hex
    poi = aid + '_poi'
    sid = f'{aid}_{DAY}'
    @contextmanager
    def session():
        with driver.session(database=os.environ.get('NEO4J_TEST_DATABASE', 'neo4j')) as value:
            yield value
    monkeypatch.setattr(store, 'driver_session', session)
    try:
        with session() as s:
            s.run('CREATE (:Agent {id:$aid}), (:POI {id:$poi})', aid=aid, poi=poi).consume()
        yield aid, poi, sid, session
    finally:
        # Exact UUID-owned IDs only. Never clear the database or use production config.
        with session() as s:
            s.run('MATCH (n) WHERE n.id IN $ids DETACH DELETE n', ids=[aid, poi, sid]).consume()
        driver.close()


def save_day(graph, tx):
    aid, poi, sid, _ = graph
    write_plan(aid, DAY, [{'poi_id':poi, 'order':0, 'time':'18:00',
                           'category':'식사', 'actual_spent':1000}], 'weekday', transaction=tx)
    tx.run('CREATE (:State {id:$sid, day:date($day)})', sid=sid, day=str(DAY)).consume()
    return store.save_result(tx, dict(aid=aid, experience_day=str(DAY),
        experience_run_id='integration', execution_fingerprint=execution_fingerprint(), status='ok'))


def test_plan_state_and_outbox_rollback_together(graph):
    aid, _, sid, session = graph
    with pytest.raises(RuntimeError, match='injected'):
        with store.transaction(aid, DAY, 'integration') as tx:
            save_day(graph, tx)
            raise RuntimeError('injected before commit')
    with session() as s:
        assert s.run('MATCH (n) WHERE n.id=$sid RETURN count(n) AS n', sid=sid).single()['n'] == 0
    assert store.load_completed(aid, DAY, 'integration') is None


def test_committed_outbox_replay_and_cross_run_guard(graph):
    aid, _, _, _ = graph
    with store.transaction(aid, DAY, 'integration') as tx:
        saved = save_day(graph, tx)
    assert store.load_completed(aid, DAY, 'integration') == saved
    with pytest.raises(store.AlreadyCommitted):
        with store.transaction(aid, DAY, 'integration'):
            pytest.fail('completed day must not be written again')
    with pytest.raises(EvidenceError):
        store.load_completed(aid, DAY, 'other')


def test_missing_poi_aborts_plan(graph):
    aid, _, sid, session = graph
    with pytest.raises(ValueError, match='persistence mismatch'):
        with store.transaction(aid, DAY, 'integration') as tx:
            write_plan(aid, DAY, [{'poi_id':aid + '_missing', 'time':'18:00'}],
                       'weekday', transaction=tx)
    with session() as s:
        assert s.run('MATCH (p:Plan {id:$sid}) RETURN count(p) AS n', sid=sid).single()['n'] == 0
