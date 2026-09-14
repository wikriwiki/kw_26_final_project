import copy
import json
from datetime import date
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from experience import (receipts, observation_window, prompt_block, update_appraisals,
                        aggregate, visible_observations)
from report_experience import build_report


def receipt_state():
    decisions = [dict(poi_id='C', category='식사', actual_spent=10000, policy_spend={'P': 10000})]
    executed = [dict(decisions[0], actual_spent=6000, policy_spend={'P': 6000}, purchase_status='reduced')]
    policy = [dict(id='P', type='grant')]
    records = receipts('A', '2026-09-14', decisions, executed, policy, 'run1')
    return records, {'observations_json': json.dumps(observation_window([], records))}


def proposal(eid, **kwargs):
    return dict(policy_id='P', stance='mixed', reason='혜택은 있지만 개인 부담도 고려한다.',
                evidence_ids=[eid], persona_refs=['income'], **kwargs)


def test_receipt_is_final_fact_and_plan_is_archived_separately():
    records, state = receipt_state()
    assert records[0]['amount'] == 6000
    assert records[0]['decision']['planned_amount'] == 10000
    assert 'decision' not in json.loads(state['observations_json'])[0]
    assert prompt_block(state, date(2026, 9, 14)) == ''
    assert records[0]['event_id'] in prompt_block(state, date(2026, 9, 15))


@pytest.mark.parametrize('kind', ['future', 'foreign_agent', 'foreign_policy', 'missing_id', 'missing_persona', 'malformed'])
def test_invalid_evidence_cannot_update_policy_stance(kind):
    records, state = receipt_state()
    item = proposal(records[0]['event_id'])
    today, aid = '2026-09-15', 'A'
    if kind == 'future': today = '2026-09-14'
    if kind == 'foreign_agent': aid = 'B'
    if kind == 'foreign_policy': item['policy_id'] = 'OTHER'
    if kind == 'missing_id': item['evidence_ids'] = ['unknown']
    if kind == 'missing_persona': item['persona_refs'] = ['invented']
    if kind == 'malformed': item['stance'] = []
    current, accepted, rejected = update_appraisals(aid, today, state, {'income': '중'}, [item])
    assert current == {} and accepted == [] and rejected


def test_valid_update_carries_without_fabricating_new_measurement():
    records, state = receipt_state()
    current, changes, errors = update_appraisals('A', '2026-09-15', state, {'income': '중'}, [proposal(records[0]['event_id'])])
    assert not errors and changes[0]['previous_stance'] is None
    state['policy_appraisals_json'] = json.dumps(current)
    carried, changes, errors = update_appraisals('A', '2026-09-16', state, {'income': '중'}, [])
    assert carried == current and not changes
    assert carried['P']['as_of'] == '2026-09-15'


def test_malformed_optional_appraisal_does_not_retry_valid_plan():
    from stage1_intent import Stage1Output
    output = Stage1Output.model_validate({'events': [dict(time='08:00', anchor='residence',
        category='집', intent='휴식')], 'policy_appraisals': 'bad metadata'})
    current, accepted, rejected = update_appraisals('A', '2026-09-15', {}, {}, output.policy_appraisals)
    assert not current and not accepted
    assert rejected == [{'code': 'invalid_appraisal_payload'}]


def test_diagnostics_are_not_observed_rejection_experiences():
    decisions = [dict(poi_id='C', category='식사', actual_spent=10000, policy_spend={'P':10000})]
    events = [dict(decisions[0], policy_spend={}, coupon_eligible=False)]
    records = receipts('A', '2026-09-14', decisions, events, [dict(id='P',type='grant',poi_restricted=True)], 'run1')
    assert records[0]['diagnostics'][0]['code'] == 'invalid_payment_request'
    observation = observation_window([], records)[0]
    assert 'diagnostics' not in observation
    assert observation['policy_facts']['P']['paid'] == 0


def test_group_denominators_and_duplicate_failed_rows(tmp_path):
    records, state = receipt_state()
    current, _, _ = update_appraisals('A','2026-09-15',state,{'income':'중'},[proposal(records[0]['event_id'])])
    row = dict(aid='A',status='ok',experience_day='2026-09-15',experience_run_id='r',
               experience_policy_ids=['P'],experience_group={'income':'중'},policy_appraisals=current)
    second = dict(row, aid='B', policy_appraisals={})
    failed = dict(row, aid='C', status='error')
    path = tmp_path / 'metrics'; path.mkdir()
    (path / 'day_2026-09-15.jsonl').write_text('\n'.join(json.dumps(r) for r in [row,row,second,failed]), encoding='utf-8')
    report = build_report(tmp_path, '2026-09-15')
    assert report['quality']['failed_agents'] == 1
    assert report['groups'][0]['agents'] == 2
    assert report['groups'][0]['measured'] == 1
    assert report['groups'][0]['unmeasured'] == 1
    with pytest.raises(ValueError):
        aggregate([row, dict(second, experience_run_id='different')])


def test_two_day_runner_connects_receipts_to_existing_stage1(monkeypatch):
    # DB and LLM are replaced at their boundaries; the real process_one,
    # consumption, validator, experience updates and metrics path are exercised.
    import run_simulation as runner
    from dawn_context import DawnContext
    snapshot = {'balance': 100000}
    calls = []
    policy = dict(id='P',type='grant',name='지원금',effective_from='2026-09-14',
                  from_=date(2026,9,14),income_grants={'중':50000})
    def dawn(aid, today):
        return DawnContext(persona={'income':'중','daily_wd':30000,'daily_we':30000},
                           state=copy.deepcopy(snapshot), policy=[policy])
    def stage1(aid, today, ctx):
        calls.append(('stage1', str(today)))
        observed = visible_observations(ctx.state, today)
        updates = [proposal(observed[-1]['event_id'])] if observed else []
        return SimpleNamespace(policy_appraisals=updates), {'tokens_in':1,'tokens_out':1,'attempt':0}
    def stage2(*args, **kwargs):
        calls.append(('stage2', str(args[3])))
        return None, {}, {'tokens_in':1,'tokens_out':1}
    def write_state(aid, today, **kwargs):
        for key in ('observations','policy_appraisals'):
            snapshot[key+'_json'] = json.dumps(kwargs[key])
        snapshot['grant_received'] = kwargs['grant_received']
        snapshot['grant_remaining'] = kwargs['grant_remaining']
        return {'balance':100000,'mood':.5,'fatigue':.3}
    monkeypatch.setattr(runner, 'build_dawn_context', dawn)
    monkeypatch.setattr(runner, 'build_environment', lambda *a: {})
    monkeypatch.setattr(runner, 'call_stage1', stage1)
    monkeypatch.setattr(runner, 'call_stage2', stage2)
    monkeypatch.setattr(runner, 'merge_to_final_events', lambda *a,**k: [dict(
        poi_id='C',category='식사',actual_spent=12000,actual_satisfaction=.8,
        policy_spend={'P':12000},coupon_eligible=True,price_factor=1)])
    monkeypatch.setattr(runner, 'write_plan', lambda *a,**k: ('plan',1))
    monkeypatch.setattr(runner, 'night_finalize_yesterday', lambda *a: 0)
    monkeypatch.setattr(runner, 'night_create_state', write_state)
    first = runner.process_one('A', date(2026,9,14),0)
    second = runner.process_one('A', date(2026,9,15),1)
    assert first['status'] == 'ok', first
    assert second['status'] == 'ok', second
    assert first['execution_receipts'] and not first['policy_appraisals']
    assert second['policy_appraisals']['P']['evidence_ids'] == [first['execution_receipts'][0]['event_id']]
    assert calls == [('stage1','2026-09-14'),('stage2','2026-09-14'),
                     ('stage1','2026-09-15'),('stage2','2026-09-15')]
    assert aggregate([first,second])[0]['measured'] == 1
