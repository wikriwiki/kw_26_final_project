"""Circumstances, provenance and balanced time coverage without stance targeting."""
import copy
from datetime import date,timedelta
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from evidence_integrity import EvidenceError,canonical,seal,verify
from stance_context import MAX_ITEM_EVALUATIONS,select_stance_context


START=date(2017,11,19)
DAYS=[(START+timedelta(days=i)).isoformat() for i in range(28)]


def item(eid,day,kind,value):
    return {'evidence_id':eid,'agent_id':'A','day':day,'kind':kind,
            'value':value,'text':canonical(value),'source_ref':{'field':eid}}


def packet(items):
    return seal({'schema_version':1,'kind':'grounded_interview_packet','run_id':'R','arm':'on',
                 'agent_id':'A','through_day':DAYS[-1],'days':DAYS,'evidence_items':items,
                 'cohort_sha256':'a'*64,'source_sha256':'b'*64,'missing_days':[],'missing_night_days':[]})


class Counter:
    def __init__(self, nonlinear=False):
        self.calls=[]
        self.nonlinear=nonlinear

    def __call__(self,tokenizer,data):
        self.calls.append(len(data['evidence_items']))
        rendered={'evidence':[{key:row[key] for key in ('evidence_id','day','kind','text')}
                              for row in data['evidence_items']],'selection':data['selection']}
        return 100+len(canonical(rendered))//5+(12*len(data['evidence_items'])**2 if self.nonlinear else 0)


def select(data,budget=20000,counter=None):
    return select_stance_context(data,None,budget,counter or Counter())


def profile(day=DAYS[-1],**overrides):
    persona={'id':'A','age_group':'40대','gender':'여성','job':'간호사','income':'중',
             'life_stage':'성인','smoking_status':'unknown','daily_wd':34000,'daily_we':42000,
             'home_h_wd':12,'home_h_we':15,'commute_min':45,'work_dong':'상계동',
             'lifestyle':'교대근무 후 쉬는 시간을 확보하려 한다.','tendency':'절약형',
             'nv_hobbies':['수영'],'nv_family':None,'nv_skills':['간호'],'nv_career':'전문성 유지',
             'nv_summary':'원문에 부여된 가상 시민 설명.'}
    persona.update(overrides)
    return item('context_'+day,day,'context',{'persona':persona,'state':{
        'balance':90000,'month_spent':120000,'energy':.4,'mood':.6,'fatigue':.7,'yest_sat':.3},'memory':[]})


def event(index,day,sat=.6,**extra):
    return item('event_'+str(index),day,'executed_event',dict(
        order=index,poi_id='P'+str(index),category='여가',actual_satisfaction=sat,actual_spent=10000,**extra))


def test_rich_recorded_personal_resources_and_state_survive_without_invention():
    original=packet([profile()])
    before=copy.deepcopy(original)
    selected=select(original)
    verify(selected)
    assert original==before
    projected={row['value']['group']:row for row in selected['evidence_items']}
    assert projected['work_and_location']['value']['values']['commute_min']==45
    assert projected['resources_and_routine']['value']['values']['daily_wd']==34000
    assert projected['prior_state_at_dawn']['value']['values']['fatigue']==.7
    assert projected['lifestyle']['value']['values']['nv_hobbies']==['수영']
    assert 'smoking_status' in projected['identity_and_smoking']['value']['unknown_fields']
    assert 'nv_family' in projected['recorded_background']['value']['unknown_fields']
    assert projected['recorded_background']['value']['values']['nv_family'] is None
    assert 'children' not in canonical(selected) and '외향' not in canonical(selected)
    for row in selected['evidence_items']:
        assert row['source_ref']['parent_evidence_id']==original['evidence_items'][0]['evidence_id']
        assert row['source_ref']['parent_packet_sha256']==original['integrity_sha256']


def test_current_unknown_is_not_filled_from_an_older_persona():
    selected=select(packet([profile(DAYS[0],job='회사원'),profile(DAYS[-1],job=None)]))
    work=next(row for row in selected['evidence_items'] if row['value'].get('group')=='work_and_location')
    assert work['value']['values']['job'] is None
    assert 'job' in work['value']['unknown_fields']


def test_metadata_logs_are_excluded_and_original_experiences_are_verbatim():
    actual=event(1,DAYS[0])
    data=packet([actual,item('log1',DAYS[-1],'model_call_reference',{'model':'model'}),
                 item('log2',DAYS[-1],'fallback_diagnostic',{'fallback':10})])
    selected=select(data)
    assert selected['evidence_items']==[actual]
    assert selected['selection']['excluded_metadata_items']==2
    assert selected['selection']['omitted_items']==2


def test_early_middle_late_and_both_satisfaction_extremes_are_available():
    rows=[profile()]
    for offset in (0,13,27):
        rows.extend([event(offset*10+1,DAYS[offset],.1),event(offset*10+2,DAYS[offset],.9)])
    selected=select(packet(rows))
    actual=[row for row in selected['evidence_items'] if row['kind']=='executed_event']
    assert {row['day'] for row in actual}=={DAYS[0],DAYS[13],DAYS[-1]}
    assert {row['value']['actual_satisfaction'] for row in actual}=={.1,.9}
    assert set(selected['selection']['selected_period_counts'])=={'early','middle','late'}


def test_smoking_leisure_or_reason_text_alone_does_not_prove_policy_target_exposure():
    unknown=event(1,DAYS[0],reason='금연 정책에 반대한다')
    topical=event(2,DAYS[13],facility_type='billiard')
    target=event(3,DAYS[-1],facility_type='screen_golf',district_code='11350')
    selected=select(packet([unknown,topical,target]))
    assert selected['selection']['selected_relevance_counts']=={
        'unknown':1,'facility_topic_location_unknown':1,'explicit_target':1}


def test_historical_memory_keeps_asof_context_day_and_recorded_occurrence():
    context=profile()
    memory={'id':'M1','type':'visited','day':'2017-11-01','satisfaction':.2,'summary':'저장된 짧은 방문 요약',
            'category':'당구장','poi_name':'시설 A'}
    context['value']['memory']=[memory]
    context['text']=canonical(context['value'])
    original=packet([context])
    selected=select(original)
    remembered=next(row for row in selected['evidence_items'] if row['kind']=='remembered_experience')
    assert remembered['day']==DAYS[-1] and remembered['day'] in selected['days']
    assert remembered['source_occurrence_day']=='2017-11-01'
    assert remembered['value']['values']==memory
    assert remembered['source_ref']['parent_pointer']=='/memory/0'
    assert selected['days']==original['days']


def test_repeated_memories_are_deduplicated_without_turning_rumor_into_fact():
    first=profile(DAYS[10])
    last=profile()
    base={'id':'R1','type':'rumor','day':DAYS[0],'summary':'전해 들은 정보','source':'B'}
    first['value']['memory']=[dict(base,days_ago=10)]
    last['value']['memory']=[dict(base,days_ago=27)]
    for row in (first,last): row['text']=canonical(row['value'])
    selected=select(packet([first,last]))
    heard=[row for row in selected['evidence_items'] if row['kind']=='reported_information']
    assert len(heard)==1 and heard[0]['value']['values']['days_ago']==27
    assert heard[0]['value']['source_kind']=='heard_or_stored_information_not_verified_event'


@pytest.mark.parametrize('mutation',['future','foreign_agent','foreign_run','foreign_arm','outside_days','duplicate'])
def test_bad_identity_or_temporal_scope_rejected(mutation):
    actual=event(1,DAYS[0])
    data=packet([actual])
    if mutation=='future': data['evidence_items'][0]['day']='2017-12-17'
    elif mutation=='outside_days': data['evidence_items'][0]['day']='2017-11-18'
    elif mutation=='duplicate': data['evidence_items'].append(copy.deepcopy(actual))
    else: data['evidence_items'][0][{'foreign_agent':'agent_id','foreign_run':'run_id','foreign_arm':'arm'}[mutation]]='foreign'
    with pytest.raises(EvidenceError): select(seal(data))


def test_future_memory_does_not_slip_in_through_a_valid_context_timestamp():
    context=profile(DAYS[5])
    context['value']['memory']=[{'id':'future','day':DAYS[6],'type':'visited','summary':'future'}]
    with pytest.raises(EvidenceError,match='earlier'):
        select(packet([context]))


def test_whole_large_item_is_skipped_without_truncating_source_text():
    huge=event(1,DAYS[0],description='x'*100000)
    small=event(2,DAYS[13])
    selected=select(packet([huge,small]),budget=1500)
    assert small in selected['evidence_items']
    assert huge not in selected['evidence_items']
    assert selected['selection']['skipped_for_budget']>=1
    assert selected['selection']['selected_period_counts'].get('early',0)==0


def test_exact_final_budget_with_nonadditive_cost_uses_logarithmic_prefix_checks():
    rows=[event(index,DAYS[index%28]) for index in range(45)]
    counter=Counter(nonlinear=True)
    selected=select(packet(rows),budget=2600,counter=counter)
    assert counter(None,selected)<=2600
    assert len(counter.calls)<=len(rows)+12
    assert selected['selection']['selected_items']==len(selected['evidence_items'])
    assert selected['selection']['skipped_for_budget']>0


def test_candidate_work_cap_is_explicit_and_never_claims_full_coverage():
    data=packet([event(index,DAYS[index%28]) for index in range(400)])
    counter=Counter()
    selected=select(data,budget=100000,counter=counter)
    audit=selected['selection']
    assert audit['evaluated_candidates']==MAX_ITEM_EVALUATIONS
    assert audit['not_evaluated_due_to_work_limit']==400-MAX_ITEM_EVALUATIONS
    assert len(counter.calls)<=MAX_ITEM_EVALUATIONS+10
    assert audit['omitted_items']==400-len(selected['evidence_items'])


def test_record_id_order_does_not_create_a_recency_preference():
    rows=[event(index,DAYS[index%28]) for index in range(36)]
    original=select(packet(rows),budget=2500)
    altered=copy.deepcopy(rows)
    for index,row in enumerate(altered): row['evidence_id']=f'zzz_{999-index:04d}'
    selected=select(packet(altered),budget=2500)
    assert [row['value'] for row in selected['evidence_items']]==[row['value'] for row in original['evidence_items']]


def test_changing_stance_labels_or_reference_outcomes_does_not_drive_selection():
    data=packet([event(index,DAYS[index%28]) for index in range(36)])
    original=select(data,budget=2500)
    altered=seal({**data,'stance':'oppose','ground_truth_delta':-99})
    assert [row['evidence_id'] for row in select(altered,budget=2500)['evidence_items']]==[
        row['evidence_id'] for row in original['evidence_items']]


def test_insufficient_metadata_budget_fails_explicitly():
    with pytest.raises(EvidenceError,match='metadata'):
        select(packet([profile()]),budget=10)


def test_generic_golf_range_cannot_be_claimed_indoor_from_district_alone():
    generic=event(1,DAYS[0],facility_type='골프연습장',district_code='11650')
    selected=select(packet([generic]))
    assert selected['selection']['selected_relevance_counts']=={'facility_topic_location_unknown':1}


def test_explicit_registry_target_requires_a_recorded_registry_hash():
    unverified=event(1,DAYS[0],policy_target=True)
    verified=event(2,DAYS[13],policy_target=True,poi_registry_sha256='c'*64)
    selected=select(packet([unverified,verified]))
    assert selected['selection']['selected_relevance_counts']=={'unknown':1,'explicit_target':1}
