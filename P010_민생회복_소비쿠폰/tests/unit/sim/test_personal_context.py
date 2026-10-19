"""Rich personal facts only in the smoking experiment; legacy bytes preserved."""
from datetime import date
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
import dawn_context
from personal_context import render_personal_context,render_personal_state
from prompts.no_smoking_v1 import format_dawn_blocks
from stage2_poi import build_stage2_prompt


@pytest.fixture(autouse=True)
def isolated_render(monkeypatch):
    monkeypatch.delenv('SIM_NO_SMOKING_MANIFEST',raising=False)
    monkeypatch.delenv('SIM_NO_SMOKING_ARM',raising=False)
    monkeypatch.setenv('EXP_DURABLES','0')


def persona():
    return {'id':'A','age_group':'40대','gender':'여성','income':'중','job':'간호사','life_stage':'성인',
            'daily_wd':30000,'daily_we':40000,'lifestyle':'교대근무 후 산책','tendency':'절약형',
            'home_dong':'상계동','home_dong_code':'11350611','home_poi':'집',
            'home_h_wd':12,'home_h_we':15,'mobility':3,'delivery_days':2}


def stage2(p,state=None):
    ev=SimpleNamespace(time='12:00',category='여가',sub_category='당구장',intent='시설 이용',
                       reasoning='짧은 설명',anchor='zone:11350611')
    candidates={0:[{'poi_id':'P','name':'시설','distance_m':100,'price_factor':1,
                   'avg_satisfaction':.5,'visit_count':0,'known':False}]}
    return build_stage2_prompt([ev],candidates,p,state=state)


def test_actual_stage1_and_stage2_preserve_lifestyle_first_line_and_long_original():
    p=persona()
    p.update(_no_smoking_prompt='오늘 시설 이용 규칙 원문',smoking_status='non_smoker',
             lifestyle='누락되면 안 되는 생활 서술 첫 줄\n'+('교대근무 후 수영하며 시간을 보낸다. '*30),
             nv_hobbies=['수영','독서'],nv_career='간호 전문성을 유지한다',nv_skills=['응급 간호'],
             work_dong='중계동',work_dong_code='11350621',commute_min=45)
    state={'balance':50000,'energy':.4,'mood':.5,'fatigue':.7,'yest_sat':.3}
    blocks=dawn_context.DawnContext(persona=p,state=state).to_prompt_blocks(date(2017,12,3))
    s1=format_dawn_blocks(blocks,date(2017,12,3),'weekend','일')
    s2=stage2(p,state)
    for rendered in (s1,s2):
        assert p['lifestyle'] in rendered
        assert '수영' in rendered and '독서' in rendered
        assert '간호 전문성을 유지한다' in rendered and '응급 간호' in rendered
        assert '통근시간(분): 45' in rendered
        assert '직장 동 코드: 11350621' in rendered
        assert '거주 동 코드: 11350611' in rendered
        assert '거주 장소: 집' in rendered
        assert '평일 집 체류시간(시간/일): 12' in rendered
        assert '평일 재택' not in rendered
        assert '오늘 시설 이용 규칙 원문' in rendered
        assert '모형 fatigue: 0.7' in rendered


def test_unknown_resources_and_workplace_do_not_become_zero_or_absent():
    p={'id':'A','_no_smoking_prompt':'시설 규칙','daily_wd':None,'daily_we':None,
       'home_h_wd':None,'home_h_we':None,'work_dong':None,'job':None,'smoking_status':'unknown'}
    state={'balance':None,'energy':None,'mood':None,'fatigue':None}
    s1=dawn_context.DawnContext(persona=p,state=state).to_prompt_blocks(date(2017,12,3))
    s2=stage2(p,state)
    for rendered in (s1['persona'],s2):
        assert '평일 소비규모(원/일): 알 수 없음' in rendered
        assert '평일 집 체류시간(시간/일): 알 수 없음' in rendered
        assert '직장 동: 알 수 없음' in rendered
        assert '직장: 없음' not in rendered
        assert '부여된 흡연 상태: 알 수 없음(기록값: unknown)' in rendered
        assert '평일 소비규모(원/일): 0' not in rendered
    assert '모형 에너지: 알 수 없음' in s1['state']
    assert '직전 상태의 개인 잔액(원): 알 수 없음' in s2


def test_explicit_zero_and_empty_collections_remain_observed_values():
    rendered=render_personal_context({'id':'A','daily_wd':0,'home_h_wd':0,'nv_hobbies':[]},
                                     {'balance':0,'energy':0,'grant_remaining':{}})
    assert '평일 소비규모(원/일): 0' in rendered
    assert '평일 집 체류시간(시간/일): 0' in rendered
    assert '기록된 취미·관심사: []' in rendered
    assert '직전 상태의 개인 잔액(원): 0' in rendered
    assert '모형 에너지: 0' in rendered
    assert '별도 정책 지갑 기록: {}' in rendered


def test_only_explicit_allowlisted_personal_fields_are_exposed():
    rendered=render_personal_context({'id':'A','secret':'do-not-include','ground_truth':.1354,
        'policy_opinion':'invented support','nv_family':'기록에 부여된 가족 정보'})
    assert 'do-not-include' not in rendered and '0.1354' not in rendered and 'invented support' not in rendered
    assert '기록에 부여된 가족 정보' in rendered
    assert '혼인 정보: 알 수 없음' in rendered


def test_legacy_stage1_persona_bytes_are_unchanged():
    assert dawn_context._format_persona(persona())==(
        'ID: A\n인구학: 40대 여성 / 직업: 간호사 / 생애주기: 성인 / 소득: 중\n'
        '소비: 평일 30,000원, 주말 40,000원 (주말/평일 1.00배) / 성향: 절약형\n'
        '평소 업종별 지출 구성: (미상)\n'
        '행태: 배달 2일/월, 평일 재택 12.0h, 주말 재택 15.0h, 이동성 분위 3\n'
        '거주: 상계동 (11350611) — 집\n직장: 없음\n라이프스타일: 교대근무 후 산책')


def test_legacy_stage2_persona_bytes_are_unchanged():
    assert stage2(persona(),{'balance':50000})==(
        '## 에이전트 정보\n교대근무 후 산책\n'
        '평소 1일 소비규모(스케일 참고, 총액 아님): 평일 30,000원 / 주말 40,000원 / 현재 잔액: 50,000원 / 소비성향: 절약형 / 소득분위: 중\n\n'
        '다음 이벤트별 candidates 중에서 POI를 선택하고 소비액·만족도를 설정하세요.\n\n'
        '### 이벤트 0 | 12:00 | zone:11350611 | 여가/당구장 | 시설 이용\n'
        '    P | 시설 |  |  | avg_sat=0.50 \n\n'
        '각 이벤트의 order·poi_id·actual_spent·actual_satisfaction·pick_reason·pick_factor를 JSON으로 출력하세요. /no_think')


def test_legacy_state_default_remains_unchanged():
    assert dawn_context._format_state({'balance':50000,'energy':.8,'mood':.5,'fatigue':.3,'month_spent':0,'yest_sat':.6})==(
        '잔액(내 돈): 50,000원 / 이번달 누적지출: 0원\n에너지: 0.80, mood: 0.50, fatigue: 0.30\n'
        '어제 평균 만족도: 0.60\n정책 라이프사이클: {}')
