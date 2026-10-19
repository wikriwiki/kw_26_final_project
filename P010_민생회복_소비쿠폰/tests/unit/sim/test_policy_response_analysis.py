import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from analyze_policy_response import analyze
from evidence_integrity import seal, canonical, EvidenceError
from experience import receipts, observation_window, update_appraisals


def fixture(root):
    design = dict(schema_version=1, policy_id='P', population={'A':'working'},
                  start='2026-09-14', end='2026-09-14',assessment_day='2026-09-15',
                  min_group_size=1,pairs=[dict(seed=7,baseline={'path':'b','run_id':'b'},
                                               policy={'path':'p','run_id':'p'})])
    for arm, amount in [('b',100),('p',80)]:
        folder = root/arm; (folder/'metrics').mkdir(parents=True)
        records = []
        for day in ['2026-09-14','2026-09-15']:
            event = dict(poi_id='shop',category='food',actual_spent=amount,policy_spend={'P':50} if arm=='p' else {})
            current = receipts('A',day,[event.copy()],[event.copy()],
                               [dict(id='P',type='grant')] if arm=='p' else [],arm)
            stances = {}
            if records and arm=='p':
                proposal = dict(policy_id='P',stance='oppose',reason='혜택을 받았지만 반대한다.',
                                persona_refs=['income'],evidence_ids=[records[0]['event_id']],
                                claims=[dict(event_id=records[0]['event_id'],field='policy_paid',value=50)])
                stances,_,reject = update_appraisals('A',day,{'observations_json':observation_window([],records)},
                                                     {'income':'middle'},[proposal])
                assert not reject
            row = seal(dict(aid='A',status='ok',experience_version=2,experience_day=day,
                experience_run_id=arm,source_fingerprint='source',execution_fingerprint=arm,
                decision_provenance={'prompt_sha256':'prompt','model_id':'model'},
                receipt_scope='all_modeled_offline_commerce_v1',execution_receipts=current,
                experience_group={'income':'middle'},experience_policy_ids=['P'] if arm=='p' else [],
                policy_appraisals=stances))
            (folder/'metrics'/f'day_{day}.jsonl').write_text(canonical(row),encoding='utf-8')
            (folder/f'cohort_{day}.json').write_text(canonical(dict(day=day,run_id=arm,agent_ids=['A'],execution_fingerprint=arm)))
            records = current
    return design


def test_baseline_receipts_exist_without_policy_and_pairing_uses_window_only(tmp_path):
    design = fixture(tmp_path)
    result = analyze(design,tmp_path)
    group = result['groups'][0]
    assert group['mean_offline_spend_delta'] == -20  # excludes assessment day's activity
    assert group['mean_offline_own_paid_delta'] == -70
    assert group['policy_users'] == 1
    assert group['stance_counts'] == {'oppose':1}  # use is not support
    assert result['claim_level'] == 'simulation_scenario_only'
    assert result['replication_status'] == 'single_replication'


def test_unmeasured_is_not_neutral(tmp_path):
    design = fixture(tmp_path)
    path = tmp_path/'p/metrics/day_2026-09-15.jsonl'
    row=json.loads(path.read_text(encoding='utf-8'));row['policy_appraisals']={}
    path.write_text(canonical(seal(row)),encoding='utf-8')
    group=analyze(design,tmp_path)['groups'][0]
    assert group['unmeasured']==1 and group['measurement_coverage']==0
    assert group['stance_share_among_measured'] is None


@pytest.mark.parametrize('mutation',['missing_agent','same_day','reused_seed','legacy_receipts'])
def test_invalid_comparison_rejected(tmp_path,mutation):
    design=fixture(tmp_path)
    if mutation=='missing_agent': design['population']['B']='working'
    if mutation=='same_day': design['assessment_day']=design['end']
    if mutation=='reused_seed': design['pairs']*=2
    if mutation=='legacy_receipts':
        path=tmp_path/'b/metrics/day_2026-09-14.jsonl'
        row=json.loads(path.read_text(encoding='utf-8'));row.pop('receipt_scope')
        path.write_text(canonical(seal(row)),encoding='utf-8')
    with pytest.raises(EvidenceError): analyze(design,tmp_path)


def test_small_group_suppression(tmp_path):
    design=fixture(tmp_path);design['min_group_size']=5
    group=analyze(design,tmp_path)['groups'][0]
    assert group['suppressed'] and 'stance_counts' not in group


def test_embedded_evidence_must_exist_in_executed_ledger(tmp_path):
    design=fixture(tmp_path)
    path=tmp_path/'p/metrics/day_2026-09-15.jsonl'
    row=json.loads(path.read_text(encoding='utf-8'))
    appraisal=row['policy_appraisals']['P']
    appraisal['evidence_snapshot'][0]=seal(dict(appraisal['evidence_snapshot'][0],poi_id='invented_shop'))
    row['policy_appraisals']['P']=seal(appraisal)
    path.write_text(canonical(seal(row)),encoding='utf-8')
    with pytest.raises(EvidenceError,match='archived execution'):
        analyze(design,tmp_path)


def test_missing_response_bounds_and_report_are_explicit(tmp_path):
    from analyze_policy_response import render_report
    design=fixture(tmp_path)
    design['population']['B']='working'
    for arm in ('b','p'):
        for day in ('2026-09-14','2026-09-15'):
            path=tmp_path/arm/'metrics'/f'day_{day}.jsonl'
            row=json.loads(path.read_text(encoding='utf-8'))
            other=copy.deepcopy(row);other['aid']='B';other['policy_appraisals']={}
            other['execution_receipts']=[seal(dict(r,agent_id='B',event_id=r['event_id']+'_B')) for r in other['execution_receipts']]
            path.write_text(canonical(row)+'\n'+canonical(seal(other)),encoding='utf-8')
            cohort_path=tmp_path/arm/f'cohort_{day}.json'
            cohort=json.loads(cohort_path.read_text());cohort['agent_ids'].append('B')
            cohort_path.write_text(canonical(cohort))
    result=analyze(design,tmp_path)
    group=result['groups'][0]
    assert group['stance_share_among_measured']['oppose']==1
    assert group['population_share_bounds']['oppose']=={'lower':0.5,'upper':1}
    assert group['population_share_bounds']['support']=={'lower':0,'upper':0.5}
    assert group['missingness_reasons']=={'no_appraisal':1}
    assert '신뢰구간이 아니다' in render_report(result)


def test_settings_change_between_days_blocks_analysis(tmp_path):
    design=fixture(tmp_path)
    path=tmp_path/'p/metrics/day_2026-09-15.jsonl'
    row=json.loads(path.read_text(encoding='utf-8'));row['execution_fingerprint']='changed'
    path.write_text(canonical(seal(row)),encoding='utf-8')
    path=tmp_path/'p/cohort_2026-09-15.json'
    cohort=json.loads(path.read_text());cohort['execution_fingerprint']='changed'
    path.write_text(canonical(cohort))
    with pytest.raises(EvidenceError,match='within a run'):
        analyze(design,tmp_path)
