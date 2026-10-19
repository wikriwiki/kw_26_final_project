from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from prepare_daily_conditions import prepare


def inputs():
    return {'personas':[{'id':'A','job':'학생','home_dong_code':'Z','work_poi_id':None}],
            'cells':[{'aid':'A','case':'arbitrary','arm':arm,'date':'2026-09-21','zones':['Z','OTHER'],
                      'user':'정책별 입력 '+arm+'\n\n## 오늘\n2026-09-21','required_activities':[]} for arm in ['off','on']]}


def profiles():
    return {'A':{'job':'학생','detergent_doses':0,'duty':{'kind':'school','start':'09:00','end':'15:00'}}}


def test_paired_states_are_identical_and_original_source_is_unchanged():
    source=inputs();before=deepcopy(source)
    result=prepare(source,profiles())
    assert source==before
    a,b=result['cells']
    assert a['daily_conditions']==b['daily_conditions']
    assert a['required_activities']==b['required_activities']
    assert a['required_activities'][0]['anchor']=='zone:Z'
    assert a['daily_conditions']['needs'][0]['mandatory'] is False
    assert all(spec['evidence'] in a['user'] for spec in a['provided_activities'])


def test_missing_profile_or_mismatching_job_refused():
    with pytest.raises(ValueError,match='roster'):prepare(inputs(),{})
    p=profiles();p['A']['job']='unrelated'
    with pytest.raises(ValueError,match='job'):prepare(inputs(),p)


def test_unknown_work_site_is_not_fabricated_or_silently_dropped():
    p=profiles();p['A']['duty']['kind']='work'
    with pytest.raises(ValueError,match='mapped site'):prepare(inputs(),p)


def test_existing_commitment_and_absent_school_zone_require_reconciliation():
    s=inputs();s['cells'][0]['required_activities']=[{}]
    with pytest.raises(ValueError,match='reconciliation'):prepare(s,profiles())
    s=inputs();s['cells'][0]['zones']=['OTHER']
    with pytest.raises(ValueError,match='school zone'):prepare(s,profiles())
