"""Adversarial inputs, evidence lineage and transactional recovery contracts."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from evidence_integrity import EvidenceError, money, seal, verify, canonical
from experience import receipts, observation_window, update_appraisals, visible_observations
from experience_export import build_report
from experience_provenance import atomic_json, execution_fingerprint
import agent_day_store as store


def snapshot(aid='A'):
    before = [dict(poi_id='C',category='식사',actual_spent=10000,policy_spend={'P':10000})]
    after = [dict(before[0])]
    rows = receipts(aid,'2026-09-14',before,after,[dict(id='P',type='grant')],'r')
    state = {'observations_json':observation_window([],rows)}
    proposal = dict(policy_id='P',stance='mixed',reason='지원금으로 결제했지만 정책 전체에 대해서는 판단을 유보한다.',
        evidence_ids=[rows[0]['event_id']],persona_refs=['income'],
        claims=[dict(event_id=rows[0]['event_id'],field='policy_paid',value=10000)])
    current,changes,rejected = update_appraisals(aid,'2026-09-15',state,{'income':'중'},[proposal])
    assert not rejected
    return seal(dict(aid=aid,status='ok',experience_day='2026-09-15',experience_run_id='r',
        experience_version=2,source_fingerprint='source',execution_fingerprint='config',
        decision_provenance={'prompt_sha256':'prompt','model_id':'test-model'},
        experience_group={'income':'중'},experience_policy_ids=['P'],policy_appraisals=current,
        execution_receipts=[])), state, proposal


@pytest.mark.parametrize('bad',[True,-1,1.5,float('nan'),float('inf'),10**1000,'100'])
def test_invalid_money_rejected(bad):
    with pytest.raises(EvidenceError): money(bad)


def test_corrupt_json_is_not_treated_as_empty_memory():
    with pytest.raises(EvidenceError):
        visible_observations({'observations_json':'{broken'},'2026-09-15')


def test_tampered_observation_fails_before_prompt():
    _,state,_ = snapshot()
    state['observations_json'][0]['amount'] = 999999
    with pytest.raises(EvidenceError): visible_observations(state,'2026-09-15')


def test_foreign_personal_memory_is_not_exposed():
    _,state,_ = snapshot()
    state['_experience_agent_id'] = 'B'
    with pytest.raises(EvidenceError): visible_observations(state,'2026-09-15')


@pytest.mark.parametrize('field,value',[('policy_paid',9999),('policy_eligible',1),('queue_minutes',30)])
def test_existing_id_does_not_validate_false_factual_claim(field,value):
    _,state,proposal = snapshot()
    proposal['claims'][0].update(field=field,value=value)
    current,accepted,rejected = update_appraisals('A','2026-09-15',state,{'income':'중'},[proposal])
    assert not current and not accepted and rejected[0]['code']=='fact_claim_mismatch'


def write_rows(root,rows):
    folder=root/'metrics';folder.mkdir(exist_ok=True)
    (folder/'day_2026-09-15.jsonl').write_text('\n'.join(canonical(r) for r in rows),encoding='utf-8')


def cohort(*ids):
    return dict(run_id='r',agent_ids=list(ids),execution_fingerprint='config')


def test_valid_export_has_audit_hash_and_small_group_suppression(tmp_path):
    row,_,_=snapshot();write_rows(tmp_path,[row,row])
    report=build_report(tmp_path,'2026-09-15',cohort=cohort('A'))
    verify(report)
    assert report['release_status']=='validated_simulation_export'
    assert report['quality']['duplicate_rows']==1
    assert report['groups'][0]['stances'] is None


def test_missing_sample_blocks_release_but_can_export_diagnostic(tmp_path):
    row,_,_=snapshot();write_rows(tmp_path,[row])
    with pytest.raises(EvidenceError): build_report(tmp_path,'2026-09-15',cohort=cohort('A','B'))
    result=build_report(tmp_path,'2026-09-15',cohort=cohort('A','B'),strict=False)
    assert result['release_status']=='blocked'
    assert result['quality']['missing_agents']==1


def test_resealed_snapshot_cannot_hide_changed_nested_fact(tmp_path):
    row,_,_=snapshot()
    row['policy_appraisals']['P']['claims'][0]['value']=7
    row=seal(row)  # top-level hash alone is not enough
    write_rows(tmp_path,[row])
    with pytest.raises(EvidenceError): build_report(tmp_path,'2026-09-15',cohort=cohort('A'))


def test_conflicting_duplicates_and_wrong_cohort_block(tmp_path):
    row,_,_=snapshot()
    other=seal(dict(row,experience_group={'income':'상'}))
    write_rows(tmp_path,[row,other])
    with pytest.raises(EvidenceError): build_report(tmp_path,'2026-09-15',cohort=cohort('A'))
    write_rows(tmp_path,[row])
    with pytest.raises(EvidenceError): build_report(tmp_path,'2026-09-15',cohort=dict(cohort('A'),run_id='foreign'))


def test_atomic_write_failure_preserves_previous_artifact(tmp_path,monkeypatch):
    target=tmp_path/'result.json';target.write_text('previous',encoding='utf-8')
    import experience_provenance as module
    def fail(*args): raise OSError('injected replacement failure')
    monkeypatch.setattr(module.os,'replace',fail)
    with pytest.raises(OSError): atomic_json(target,{'new':1})
    assert target.read_text()=='previous'
    assert list(tmp_path.glob('*.tmp'))==[]


class Response:
    def __init__(self,row): self.row=row
    def single(self): return self.row


class FakeTx:
    def __init__(self,database): self.db=database;self.pending=copy.deepcopy(database);self.committed=False
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def run(self,query,**params):
        if 'execution_lock' in query: return Response({'n':1})
        if query==store.READ: return Response(self.pending.get(params['sid']))
        if 'agent_metrics_json=$metrics' in query:
            self.pending[params['sid']]={'run_id':params['run_id'],'metrics':params['metrics']}
            return Response({'n':1})
        raise AssertionError('unexpected query')
    def commit(self): self.db.clear();self.db.update(self.pending);self.committed=True


class FakeSession(FakeTx):
    def begin_transaction(self): return FakeTx(self.db)


def test_agent_day_transaction_rolls_back_and_replays_outbox(monkeypatch):
    database={}
    @contextmanager
    def session(): yield FakeSession(database)
    monkeypatch.setattr(store,'driver_session',session)
    row,_,_=snapshot();row=seal(dict(row,execution_fingerprint=execution_fingerprint()))
    with pytest.raises(RuntimeError):
        with store.transaction('A','2026-09-15','r') as tx:
            store.save_result(tx,row)
            raise RuntimeError('injected failure before commit')
    assert database=={}
    with store.transaction('A','2026-09-15','r') as tx:
        saved=store.save_result(tx,row)
    assert store.load_completed('A','2026-09-15','r')==saved
    with pytest.raises(store.AlreadyCommitted):
        with store.transaction('A','2026-09-15','r'): pass
    with pytest.raises(EvidenceError): store.load_completed('A','2026-09-15','other')


def test_legacy_day_is_not_silently_overwritten(monkeypatch):
    @contextmanager
    def session(): yield FakeSession({'A_2026-09-15':{'run_id':None,'metrics':None}})
    monkeypatch.setattr(store,'driver_session',session)
    with pytest.raises(EvidenceError): store.load_completed('A','2026-09-15','r')


def test_resealed_receipt_with_impossible_accounting_blocks_export(tmp_path):
    row, _, _ = snapshot()
    event = dict(poi_id='C', category='식사', actual_spent=100, policy_spend={})
    receipt = receipts('A','2026-09-15',[event.copy()],[event.copy()],
                       [dict(id='P',type='grant')],'r')[0]
    receipt = seal(dict(receipt, own_paid=101))
    write_rows(tmp_path, [seal(dict(row, execution_receipts=[receipt]))])
    with pytest.raises(EvidenceError):
        build_report(tmp_path,'2026-09-15',cohort=cohort('A'))


def test_missing_poi_write_is_not_reported_as_completed():
    from datetime import date
    from plan_writer import write_plan
    class MissingPoi:
        def run(self, query, **params):
            return Response({'written':0})
    with pytest.raises(ValueError, match='persistence mismatch'):
        write_plan('A', date(2026,9,15), [{'poi_id':'absent'}],
                   'weekday', transaction=MissingPoi())
