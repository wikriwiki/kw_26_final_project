"""Explicit personal facts for the opt-in smoking experiment.

Missing values never become zero or an absent workplace. Assigned biography is
retained as a profile, not converted into observed experience or hidden motives.
The shared exact-token request guard handles long profiles; this renderer never
silently truncates or invents private facts.
"""
from __future__ import annotations

import json

try:
    from .stance_context import PERSONA_GROUPS
except ImportError:
    from stance_context import PERSONA_GROUPS


LABELS = {
    'id':'시민 ID', 'age':'기록 연령', 'age_group':'연령대', 'gender':'성별', 'sex':'성별 별도 기록',
    'life_stage':'생애주기', 'smoking_status':'부여된 흡연 상태',
    'income':'소득 구간', 'spend_decile':'소비 규모 분위',
    'daily_wd':'평일 소비규모(원/일)', 'daily_we':'주말 소비규모(원/일)',
    'we_wd_ratio':'기록된 주말/평일 소비 비율', 'cat_ratio_wd':'평일 업종별 소비 구성(원문)',
    'cat_ratio_we':'주말 업종별 소비 구성(원문)', 'delivery_days':'배달 이용(일/월)',
    'home_h_wd':'평일 집 체류시간(시간/일)', 'home_h_we':'주말 집 체류시간(시간/일)',
    'mobility':'기록된 이동성 분위', 'job':'직업(기록값)', 'work_dong':'직장 동',
    'work_dong_code':'직장 동 코드', 'work_poi':'직장 장소', 'commute_min':'통근시간(분)',
    'home_dong':'거주 동', 'home_dong_code':'거주 동 코드', 'home_poi':'거주 장소', 'home_gu':'거주 자치구',
    'lifestyle':'생활 서술(원문)', 'tendency':'부여된 소비성향', 'nv_hobbies':'기록된 취미·관심사',
    'nv_cultural':'부여된 문화 배경', 'nv_education':'부여된 교육 배경',
    'nv_marital':'부여된 혼인 정보', 'nv_family':'부여된 가족 정보',
    'nv_career':'기록된 직업 목표', 'nv_skills':'기록된 기술·역량', 'nv_summary':'부여된 개인 서술(원문)',
    'balance':'직전 상태의 개인 잔액(원)', 'month_spent':'직전 상태의 월 누적 지출(원)',
    'energy':'모형 에너지', 'mood':'모형 mood', 'fatigue':'모형 fatigue',
    'yest_sat':'직전 평균 만족도', 'grant_remaining':'별도 정책 지갑 기록',
}
GROUP_LABELS = {
    'identity_and_smoking':'기본 상황과 흡연 속성', 'resources_and_routine':'소비 자원과 평소 생활시간',
    'work_and_location':'직업·통근·생활 위치', 'lifestyle':'생활 방식과 관심사',
    'recorded_background':'부여된 배경 정보', 'recorded_career_and_skills':'직업 목표와 역량',
    'assigned_narrative':'개인 서술',
}
STATE_FIELDS = ('balance','month_spent','energy','mood','fatigue','yest_sat','grant_remaining')


def recorded(value):
    if value is None or (isinstance(value,str) and not value.strip()):
        return '알 수 없음(미기록)'
    if isinstance(value,str):
        if value.strip().casefold() in {'unknown','미상','정보 없음','알 수 없음'}:
            return f'알 수 없음(기록값: {value})'
        return value
    return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def render_personal_state(state):
    state=state or {}
    lines=['직전까지 기록된 모의 상태이며 실제 감정 진단이나 오늘의 완료 상태가 아니다.']
    lines.extend(f"- {LABELS[key]}: {recorded(state.get(key))}" for key in STATE_FIELDS)
    return '\n'.join(lines)


def render_personal_context(persona, state=None, *, include_smoking_rule=True):
    lines=[
        '다음은 모형에 부여된 개인 기록이다. 기록된 생활 상황을 선택의 근거로 쓸 수 있지만 실제 사람의 관측된 입장이나 새 경험은 아니다.',
        '미기록은 알 수 없음이며 0이나 없음으로 바꾸지 않는다. 집 체류시간은 집에 머무는 시간이며 재택근무 여부를 뜻하지 않는다.',
        '평소 소비규모는 자원·습관의 참고값이며 오늘 반드시 쓸 금액이 아니다. 정형 위치·시간과 생활 서술이 충돌하면 정형 기록을 우선하고 불확실성을 남긴다.',
    ]
    for group,fields in PERSONA_GROUPS.items():
        lines.append('### '+GROUP_LABELS[group])
        lines.extend(f"- {LABELS[key]}: {recorded(persona.get(key))}" for key in fields)
    if state is not None:
        lines.extend(['### 직전 상태',render_personal_state(state)])
    if include_smoking_rule and persona.get('_no_smoking_prompt'):
        lines.extend(['### 오늘의 시설 이용 규칙(제공된 원문)',persona['_no_smoking_prompt']])
    return '\n'.join(lines)
