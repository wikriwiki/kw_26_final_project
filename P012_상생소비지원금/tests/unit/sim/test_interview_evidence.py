"""Offline replay/citation checks; no model endpoint or graph required."""
from datetime import date, time
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/sim"))
import interview_evidence as evidence
from evidence_integrity import EvidenceError, canonical, checked_evidence_citations, digest, seal, verify
import llm_client

SOURCE = "a" * 64
COHORT = ["a", "b"]


def response(text='{"reasoning":"짧은 공개 이유"}'):
    return SimpleNamespace(id="completion-id", model="served-model", created=123,
        choices=[SimpleNamespace(index=0, finish_reason="stop",
            message=SimpleNamespace(content=text, refusal=None, reasoning_content="MUST_NOT_SAVE"))],
        usage=SimpleNamespace(prompt_tokens=10,completion_tokens=5,total_tokens=15))


@pytest.fixture(autouse=True)
def isolated_scope(monkeypatch):
    evidence.clear_evidence()
    monkeypatch.delenv("SIM_NO_SMOKING_MANIFEST", raising=False)
    monkeypatch.delenv("SIM_NO_SMOKING_ARM", raising=False)
    yield
    evidence.clear_evidence()


def manifest(root):
    (root / "experiment_run.json").write_text(canonical({
        "arm":"on", "start":"2017-12-02", "days":2, "cohort_ids":COHORT}),encoding="utf-8")


def complete_night(root, day, results=None):
    reference = evidence.commit_night_evidence(root,"run","on",day,results or [])
    marker = seal({"run_id":"run","arm":"on","day":day,"status":"complete",
                   "cohort":COHORT,"evidence_ref":reference})
    (root / f"night2_completed_{day}.json").write_text(canonical(marker),encoding="utf-8")


def write_day(root, day="2017-12-02", aid="a", *, nightly=True, executed=None):
    token = evidence.begin_evidence(root,"run","on",day,[aid],digest(COHORT),SOURCE,
                                    context={"persona":{"age":50},"day":date.fromisoformat(day)})
    try:
        evidence.set_evidence_stage("stage1")
        evidence.record_chat_call({"model":"requested-model","messages":[{"role":"user","content":"actual input"}],"seed":42},response)
        receipt = seal({"version":2,"kind":"purchase_receipt","event_id":f"EX_{aid}_{day}",
                        "agent_id":aid,"run_id":"run","observed_at":day,"amount":100,
                        "own_paid":100,"purchase_status":"purchased","policy_facts":{}})
        result = {"aid":aid,"status":"ok","experience_run_id":"run","experience_day":day,
                  "source_fingerprint":SOURCE,"execution_receipts":[receipt],"fb_missing_picks_filled":1}
        result["interview_evidence"] = evidence.archive_agent_day(result,
            decisions={"stage1":{"events":[{"order":1,"reasoning":"짧은 공개 이유","evidence_quote":"actual input","time":time(12,0)}]},
                       "stage2":{"picks":[{"order":1,"pick_reason":"engine selection"}]},
                       "stage2_meta":{"fallback_only":True}},
            executed_events=executed if executed is not None else [{"order":1,"actual_spent":100,"poi_id":"p"}])
        result = seal(result)
    finally:
        evidence.clear_evidence(token)
    folder = root / "metrics"
    folder.mkdir(exist_ok=True)
    with (folder / f"day_{day}.jsonl").open("a",encoding="utf-8") as output:
        output.write(canonical(result)+"\n")
    if nightly:
        complete_night(root,day)
    return result


def test_packet_marks_exhausted_day_as_skipped_without_fake_experience(tmp_path):
    manifest(tmp_path)
    meta = json.loads((tmp_path / "experiment_run.json").read_text())
    meta["run_id"] = "run"
    (tmp_path / "experiment_run.json").write_text(canonical(meta))
    write_day(tmp_path, "2017-12-02", "a")
    day = "2017-12-03"
    skipped = seal({"aid": "a", "status": "skipped", "skip_kind": "failed_after_retries",
                    "attempts": 6, "observed_behavior": False,
                    "experience_run_id": "run", "experience_day": day,
                    "source_fingerprint": SOURCE, "no_smoking": {"arm": "on"}})
    (tmp_path / "metrics" / f"day_{day}.jsonl").write_text(canonical(skipped) + "\n")
    complete_night(tmp_path, day)
    packet = evidence.build_packet(tmp_path, "a", day)
    assert packet["skipped_days"] == [day]
    assert packet["missing_days"] == []
    assert not any(item["day"] == day and item["kind"] == "executed_receipt"
                   for item in packet["evidence_items"])


def test_scoped_llm_client_saves_exact_request_public_output_and_model(tmp_path):
    token = evidence.begin_evidence(tmp_path,"run","on","2017-12-02",["a"],digest(COHORT),SOURCE)
    requests = []
    def send(**kwargs):
        requests.append(kwargs)
        return response()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
    try:
        llm_client.call_chat(None,"system","user",client=client,response_format={"type":"json_object"})
    finally:
        evidence.clear_evidence(token)
    rows = [json.loads(line) for line in next((tmp_path/"evidence").rglob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert rows[0]["request"] == requests[0]
    assert rows[0]["request_sha256"] == digest(requests[0])
    assert rows[1]["response"]["model"] == "served-model"
    assert "MUST_NOT_SAVE" not in canonical(rows)
    assert all(verify(row) for row in rows)


def test_unscoped_call_keeps_original_behavior():
    assert evidence.record_chat_call({"not_json_serializable":object()},lambda: "ok") == "ok"


def test_registered_facility_facts_survive_committed_archive_and_context_selection(tmp_path):
    from no_smoking_context import NoSmokingContext
    from stance_context import select_stance_context
    runtime = NoSmokingContext(arm='on', assignment_seed=1,
        cohort=[{'id':'a','smoking_status':'smoker'}],
        pois=[{'poi_id':'p','district_code':'11650','facility_type':'billiard'}])
    runtime.manifest_sha256 = 'c' * 64
    original = [{'order':1,'actual_spent':100,'poi_id':'p','category':'여가'},
                {'order':2,'poi_id':'unregistered','category':'여가','policy_target':True}]
    annotated = runtime.annotate_executed_events(original)
    assert 'facility_type' not in original[0] and original[1]['policy_target'] is True
    assert annotated[0]['facility_type'] == 'billiard'
    assert annotated[0]['poi_registry_sha256'] == runtime.manifest_sha256
    assert 'policy_target' not in annotated[1]
    manifest(tmp_path)
    write_day(tmp_path, executed=annotated)
    packet = evidence.build_packet(tmp_path, 'a', '2017-12-02')
    recorded = [i for i in packet['evidence_items'] if i['kind'] == 'executed_event' and i['value']['poi_id'] == 'p']
    assert len(recorded) == 1 and recorded[0]['value']['policy_target'] is True
    assert recorded[0]['value']['poi_registry_sha256'] == 'c' * 64
    selected = select_stance_context(packet, None, 10000, lambda _, p: len(canonical(p)) // 4)
    assert any(i['evidence_id'] == recorded[0]['evidence_id'] for i in selected['evidence_items'])
    assert selected['selection']['selected_relevance_counts']['explicit_target'] >= 1


def test_facility_annotation_requires_frozen_registry_provenance():
    from no_smoking_context import NoSmokingContext
    runtime = NoSmokingContext(arm='off', assignment_seed=1,
        cohort=[{'id':'a','smoking_status':'unknown'}],
        pois=[{'poi_id':'p','district_code':'11650','facility_type':'billiard'}])
    with pytest.raises(ValueError, match='registry hash'):
        runtime.annotate_executed_events([{'poi_id':'p'}])


def test_package_and_flat_imports_share_scope():
    from scripts.sim import interview_evidence as packaged
    assert packaged is evidence


def test_request_write_failure_prevents_network_and_completion(tmp_path,monkeypatch):
    evidence.begin_evidence(tmp_path,"run","on","2017-12-02",["a"],digest(COHORT),SOURCE)
    monkeypatch.setattr(evidence.os,"fsync",lambda _: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(EvidenceError,match="durably"):
        evidence.record_chat_call({},lambda: pytest.fail("must not contact endpoint"))
    with pytest.raises(EvidenceError,match="persistence failures"):
        evidence.record_chat_call({},lambda: pytest.fail("must remain failed"))


def test_failed_http_call_is_not_a_citizens_reason_and_does_not_leak_error_text(tmp_path):
    evidence.begin_evidence(tmp_path,"run","on","2017-12-02",["a"],digest(COHORT),SOURCE)
    def fail():
        raise RuntimeError("https://secret:password@host")
    with pytest.raises(RuntimeError):
        evidence.record_chat_call({},fail)
    text = next((tmp_path/"evidence").rglob("*.jsonl")).read_text(encoding="utf-8")
    result = json.loads(text.splitlines()[-1])
    assert result['status'] == 'error' and result['error_type'] == 'RuntimeError'
    assert 'password' not in text


def test_packet_separates_actual_fallback_and_public_statements(tmp_path):
    manifest(tmp_path)
    write_day(tmp_path)
    packet = evidence.build_packet(tmp_path,"a","2017-12-02")
    verify(packet)
    assert packet['missing_days'] == packet['missing_night_days'] == []
    kinds = {item['kind'] for item in packet['evidence_items']}
    assert {'executed_receipt','executed_event','stated_rationale','fallback_diagnostic','model_call_reference'} <= kinds
    rationales = [item['value'] for item in packet['evidence_items'] if item['kind']=='stated_rationale']
    assert {row['origin'] for row in rationales} == {'engine_fallback','postprocessed_public_statement_unverified'}
    assert all(row['reason_status']=='subjective_unverified' for row in rationales)


def test_future_days_and_other_agent_evidence_are_not_exposed(tmp_path):
    manifest(tmp_path)
    write_day(tmp_path,aid="a")
    write_day(tmp_path,aid="b")
    write_day(tmp_path,day="2017-12-03",aid="a")
    packet = evidence.build_packet(tmp_path,"a","2017-12-02")
    assert all(row['agent_id']=='a' and row['day']=='2017-12-02' for row in packet['evidence_items'])
    with pytest.raises(EvidenceError,match="future"):
        evidence.build_packet(tmp_path,"a","2017-12-02",days=['2017-12-03'])


@pytest.mark.parametrize('missing_night',[False,True])
def test_incomplete_history_blocks_default_but_audit_can_report(tmp_path,missing_night):
    manifest(tmp_path)
    write_day(tmp_path,nightly=not missing_night)
    through = '2017-12-02' if missing_night else '2017-12-03'
    with pytest.raises(EvidenceError,match="incomplete interview history"):
        evidence.build_packet(tmp_path,'a',through)
    packet=evidence.build_packet(tmp_path,'a',through,allow_incomplete=True)
    assert packet['missing_night_days'] if missing_night else packet['missing_days']


def test_orphan_journal_without_committed_metrics_cannot_be_exported(tmp_path):
    manifest(tmp_path)
    evidence.begin_evidence(tmp_path,'run','on','2017-12-02',['a'],digest(COHORT),SOURCE)
    evidence.record_chat_call({},response)
    with pytest.raises(EvidenceError,match='no committed'):
        evidence.build_packet(tmp_path,'a','2017-12-02')


@pytest.mark.parametrize('mutation',['alter_request','remove_response','wrong_arm','wrong_receipt'])
def test_corrupt_or_foreign_committed_evidence_is_rejected(tmp_path,mutation):
    manifest(tmp_path)
    row=write_day(tmp_path)
    path=tmp_path/row['interview_evidence']['path']
    records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if mutation=='alter_request':
        records[0]['request']['model']='tampered'
        records[0]=seal(records[0])
    elif mutation=='remove_response':
        records.pop(1)
    else:
        if mutation=='wrong_arm': records[-1]['arm']='off'
        else: records[-1]['receipts']=[]
        records[-1]=seal(records[-1])
        row['interview_evidence']['integrity_sha256']=records[-1]['integrity_sha256']
        (tmp_path/'metrics/day_2017-12-02.jsonl').write_text(canonical(seal(row))+'\n',encoding='utf-8')
    path.write_text('\n'.join(canonical(record) for record in records)+'\n',encoding='utf-8')
    with pytest.raises(EvidenceError):
        evidence.build_packet(tmp_path,'a','2017-12-02')


def test_typed_citations_cannot_turn_one_into_true_or_invent_a_quote(tmp_path):
    manifest(tmp_path)
    write_day(tmp_path)
    packet=evidence.build_packet(tmp_path,'a','2017-12-02')
    item=next(row for row in packet['evidence_items'] if row['kind']=='executed_receipt')
    valid={'evidence_id':item['evidence_id'],'pointer':'/amount','value':100}
    assert checked_evidence_citations([valid],packet)[0]['source_kind']=='executed_receipt'
    for changed in ({**valid,'value':True},{**valid,'pointer':'/missing'},
                    {'evidence_id':item['evidence_id'],'quote':'invented','start':0,'end':8},
                    {**valid,'evidence_id':'not-exposed'}):
        with pytest.raises(EvidenceError): checked_evidence_citations([changed],packet)
    exact={'evidence_id':item['evidence_id'],'quote':item['text'][:10],'start':0,'end':10}
    assert checked_evidence_citations([exact],packet)


def test_shared_night_interaction_is_visible_only_to_its_participants(tmp_path):
    manifest(tmp_path)
    write_day(tmp_path,aid='a')
    write_day(tmp_path,aid='b')
    token=evidence.begin_evidence(tmp_path,'run','on','2017-12-02',['a'],digest(COHORT),SOURCE)
    try:
        evidence.record_chat_call({},response)
        result={'intent':'정보공유','reasoning':'짧은 설명'}
        result['interview_evidence']=evidence.archive_interaction(result)
    finally: evidence.clear_evidence(token)
    result['conversation_id']='conversation-1'
    complete_night(tmp_path,'2017-12-02',[result])
    with evidence.packet_session(tmp_path,'2017-12-02') as session:
        a=session.build_packet('a')
        b=session.build_packet('b')
    assert any(row['kind']=='social_interaction' for row in a['evidence_items'])
    assert not any(row['kind']=='social_interaction' for row in b['evidence_items'])


def test_indexed_census_matches_single_packet_without_rescanning_metrics(tmp_path,monkeypatch):
    manifest(tmp_path)
    write_day(tmp_path,aid='a')
    write_day(tmp_path,aid='b')
    expected=evidence.build_packet(tmp_path,'a','2017-12-02')
    with evidence.packet_session(tmp_path,'2017-12-02') as session:
        original=Path.open
        def no_metrics_open(self,*args,**kwargs):
            if self.parent.name=='metrics': pytest.fail('metrics rescanned after indexing')
            return original(self,*args,**kwargs)
        monkeypatch.setattr(Path,'open',no_metrics_open)
        actual=session.build_packet('a')
        assert actual['evidence_items']==expected['evidence_items']
        assert session.build_packet('b')['agent_id']=='b'


def test_export_does_not_overwrite_original_evidence(tmp_path):
    manifest(tmp_path)
    write_day(tmp_path)
    with pytest.raises(EvidenceError,match='overwrite'):
        evidence.export_packet(tmp_path,'a','2017-12-02',tmp_path/'experiment_run.json')


def test_grounded_interview_rejects_unexposed_citation(tmp_path,monkeypatch):
    import interview_agent
    manifest(tmp_path)
    write_day(tmp_path)
    packet=evidence.build_packet(tmp_path,'a','2017-12-02')
    monkeypatch.setattr(interview_agent,'_llm_call',lambda *args,**kwargs:response(json.dumps({
        'answer':'made up','citations':[{'evidence_id':'foreign','pointer':'','value':{}}]})))
    with pytest.raises(EvidenceError,match='unexposed'):
        interview_agent.ask_grounded(packet,'왜?')


def test_legacy_prompt_does_not_demand_thought_expansion_or_policy_benefits():
    import interview_agent
    assert '반드시 인용·확장' not in interview_agent.INTERVIEW_SYSTEM
    assert '기록만으로 알 수 없습니다' in interview_agent.INTERVIEW_SYSTEM


@pytest.mark.parametrize('as_of,visible',[('2017-12-02',True),('2017-12-03',False),('2017-12-04',False)])
def test_prior_stance_prompt_never_exposes_today_or_future_measurement(as_of,visible):
    from experience import prompt_block
    receipt=seal({'version':2,'kind':'purchase_receipt','event_id':'EX_prior','agent_id':'a',
                  'run_id':'run','observed_at':'2017-12-01','amount':100,'own_paid':100,
                  'purchase_status':'purchased','policy_facts':{'P':{'paid':0,'eligible_under_modeled_rules':True}}})
    appraisal=seal({'policy_id':'P','agent_id':'a','stance':'support','as_of':as_of})
    state={'_experience_agent_id':'a','experience_run_id':'run','observations_json':[receipt],
           'policy_appraisals_json':{'P':appraisal}}
    rendered=prompt_block(state,date(2017,12,3))
    assert ('"as_of": "'+as_of+'"' in rendered) is visible


def test_census_audit_verifies_every_agent_and_counts_real_quote_coverage(tmp_path):
    manifest(tmp_path)
    write_day(tmp_path,aid='a')
    write_day(tmp_path,aid='b')
    audit=evidence.audit_run_evidence(tmp_path,'2017-12-02')
    assert audit['status']=='passed'
    assert audit['expected_agent_days']==audit['verified_agent_days']==2
    assert audit['verified_packets']==2
    assert audit['rationale_coverage']['exact_input_quotes']==2
    assert audit['rationale_coverage']['engine_fallback_statements']==2
    assert audit['gpu_or_llm_called'] is False


def test_census_audit_reports_missing_agents_instead_of_neutral(tmp_path):
    manifest(tmp_path)
    write_day(tmp_path,aid='a')
    audit=evidence.audit_run_evidence(tmp_path,'2017-12-02')
    assert audit['status']=='blocked' and audit['missing_agent_days']==1
    assert audit['failures'][0]['agent_id']=='b'


def test_census_audit_rejects_bad_rationale_quotes_even_if_records_resealed(tmp_path):
    manifest(tmp_path)
    row=write_day(tmp_path)
    path=tmp_path/row['interview_evidence']['path']
    records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    records[-1]['stated_rationales'][0]['public_statement']['evidence_quote']='never appeared'
    records[-1]=seal(records[-1])
    path.write_text('\n'.join(canonical(record) for record in records)+'\n',encoding='utf-8')
    row['interview_evidence']['integrity_sha256']=records[-1]['integrity_sha256']
    (tmp_path/'metrics/day_2017-12-02.jsonl').write_text(canonical(seal(row))+'\n',encoding='utf-8')
    audit=evidence.audit_run_evidence(tmp_path,'2017-12-02')
    assert audit['status']=='blocked' and audit['invalid_quote_agents']==1


def test_real_process_archives_journal_before_commit_and_clears_scope(tmp_path,monkeypatch):
    from contextlib import nullcontext
    import run_simulation as runner
    from dawn_context import DawnContext
    from no_smoking_context import NoSmokingContext
    runtime=NoSmokingContext(arm='on',cohort=[{'id':'A','smoking_status':'smoker'}],
        pois=[{'poi_id':'P','district_code':'11650','facility_type':'billiard'}],assignment_seed=20171203)
    runtime.manifest_sha256='c'*64
    monkeypatch.setattr(runner,'OUT_DIR',tmp_path)
    monkeypatch.setattr(runner,'configured_context',lambda:runtime)
    monkeypatch.setattr(runner,'build_environment',lambda *a:{})
    monkeypatch.setattr(runner,'build_dawn_context',lambda *a:DawnContext(
        persona={'income':'중','daily_wd':30000,'daily_we':30000},state={'balance':100000},policy=[]))
    def stage1(*args,**kwargs):
        evidence.record_chat_call({'model':'fixture','messages':[{'role':'user','content':'observed context'}]},response)
        return SimpleNamespace(policy_appraisals=[],model_dump=lambda:{'events':[
            {'reasoning':'short explanation','evidence_quote':'observed context'}]}),{
            'tokens_in':1,'tokens_out':1,'attempt':0,'model_id':'fixture','prompt_sha256':'b'*64}
    monkeypatch.setattr(runner,'call_stage1',stage1)
    monkeypatch.setattr(runner,'call_stage2',lambda *a,**k:(SimpleNamespace(model_dump=lambda:{'picks':[]}),{},
                                                        {'tokens_in':0,'tokens_out':0,'skipped':True}))
    monkeypatch.setattr(runner,'merge_to_final_events',lambda *a,**k:[{
        'poi_id':'P','category':'여가','actual_spent':10000,'actual_satisfaction':.7,
        'policy_spend':{},'price_factor':1}])
    monkeypatch.setattr(runner.agent_day_store,'load_completed',lambda *a:None)
    monkeypatch.setattr(runner.agent_day_store,'transaction',lambda *a:nullcontext(object()))
    committed=[]
    def commit(tx,result):
        reference=result['interview_evidence']
        archived,_=evidence._load_journal(tmp_path,reference)
        assert archived['receipts']==result['execution_receipts']
        committed.append(result)
        return seal(result)
    monkeypatch.setattr(runner.agent_day_store,'save_result',commit)
    monkeypatch.setattr(runner,'write_plan',lambda *a,**k:('plan',1))
    monkeypatch.setattr(runner,'night_finalize_yesterday',lambda *a,**k:0)
    monkeypatch.setattr(runner,'night_create_state',lambda *a,**k:{'balance':90000,'mood':.5,'fatigue':.3})
    result=runner.process_one('A',date(2017,12,3),0)
    assert result['status']=='ok',result
    assert len(committed)==1 and evidence._ACTIVE.get() is None


def test_budget_rejection_is_durable_and_never_contacts_server(tmp_path,monkeypatch):
    import prompt_budget
    evidence.begin_evidence(tmp_path,'run','on','2017-12-02',['a'],digest(COHORT),SOURCE)
    monkeypatch.setattr(prompt_budget,'check_request_budget',lambda request: (_ for _ in ()).throw(ValueError('too long')))
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:pytest.fail('network'))))
    with pytest.raises(ValueError,match='too long'):
        llm_client.call_chat(None,'system','user',client=client)
    rows=[json.loads(line) for line in next((tmp_path/'evidence').rglob('*.jsonl')).read_text(encoding='utf-8').splitlines()]
    assert [row['kind'] for row in rows]==['llm_request','llm_response']
    assert rows[-1]['status']=='error' and rows[-1]['error_type']=='ValueError'


def test_execution_fingerprint_captures_guard_contract_not_machine_tokenizer_path(tmp_path,monkeypatch):
    import experience_provenance as provenance
    import no_smoking_context
    import prompt_budget
    tokenizer_manifest=tmp_path/'tokenizer_manifest.json'
    tokenizer_manifest.write_text('{"revision":"one"}',encoding='utf-8')
    monkeypatch.setattr(prompt_budget,'MANIFEST',tokenizer_manifest)
    monkeypatch.setattr(no_smoking_context,'configured_context',lambda:SimpleNamespace(arm='on',manifest_sha256='b'*64))
    monkeypatch.setenv('SIM_TOKENIZER_PATH','machine-one/tokenizer')
    initial=provenance.execution_fingerprint()
    monkeypatch.setenv('SIM_TOKENIZER_PATH','machine-two/tokenizer')
    assert provenance.execution_fingerprint()==initial
    monkeypatch.setenv('SIM_PROMPT_TOKEN_GUARD','required')
    guarded=provenance.execution_fingerprint()
    assert guarded!=initial
    tokenizer_manifest.write_text('{"revision":"two"}',encoding='utf-8')
    assert provenance.execution_fingerprint()!=guarded


def test_packet_rejects_metrics_from_another_named_run(tmp_path):
    manifest(tmp_path)
    write_day(tmp_path)
    path=tmp_path/'experiment_run.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    data['run_id']='foreign-run'
    path.write_text(canonical(data),encoding='utf-8')
    with pytest.raises(EvidenceError,match='committed snapshot'):
        evidence.build_packet(tmp_path,'a','2017-12-02')


def test_audit_rejects_future_date_before_reporting_denominators(tmp_path):
    manifest(tmp_path)
    with pytest.raises(EvidenceError,match='outside'):
        evidence.audit_run_evidence(tmp_path,'2017-12-04')
