"""Freeze policy-independent day assumptions on the coherent development cohort.

These are declared conditional scenarios, not observed work/school schedules or
household inventory. No policy name, arm, benefit size or result determines them.
"""
import argparse
import copy
from datetime import date
import hashlib
import json
from pathlib import Path
from daily_resource_contract import validate
from validate_prompt_v3 import atomic, digest


def prepare(source, profiles):
    result = copy.deepcopy(source)
    people = {p['id']: p for p in result['personas']}
    if set(profiles) != set(people):
        raise ValueError('Explicit full-cohort profile roster required')
    for cell in result['cells']:
        p = people[cell['aid']]; profile = profiles[cell['aid']]
        if profile['job'] != p['job'] or date.fromisoformat(cell['date']).weekday() >= 5:
            raise ValueError('Profile job or declared weekday scenario mismatch')
        if 'daily_conditions' in cell or cell.get('required_activities'):
            raise ValueError('Existing daily condition/commitment needs separate reconciliation')
        conditions = {'provenance': {'kind': 'synthetic_assumption',
            'source': 'Registered daily-condition development profiles; unobserved personal schedule and stock are explicit scenario assumptions.'},
            'resources': {'detergent_dose': {'unit': '세탁 1회분', 'opening_quantity': profile['detergent_doses']}},
            'activity_consumption': {'provided_laundry': {'detergent_dose': 1}},
            'quote_receipts': {'quote:home_online_goods': {'detergent_dose': 10}},
            'quote_receipt_delay_minutes': {'quote:home_online_goods': 90},
            'needs': [{'id': 'laundry', 'description': '밀린 빨래 1회분이 있어 오늘 세탁하고 싶다. 일정과 자금에 따라 다음 날로 미룰 수도 있다.',
                       'fulfilled_by': ['provided_laundry'], 'desired_count': 1, 'mandatory': False}],
            'assumptions': ['세탁기는 이용 가능하고 그 외 세탁 비용은 이 실험에서 추가 발생하지 않는다고 가정한다.',
                '제시된 세제 한 통은 이 가정에서 세탁 10회분이며 온라인 구입 90분 뒤 도착한다. 실제 상품 용량이나 배송 기록이 아니다.',
                '다른 물품·식재료·열량·가사 전체의 재고는 아직 모델링하지 않는다. 집안 정리 활동만으로 이 세탁을 완료한 것으로 보지 않는다.']}
        validate(conditions)
        laundry_evidence = '실험 가정: provided_laundry는 집에서 세탁 1회를 하는 활동이며, 실제 시작 전에 세제 1회분이 있어야 한다.'
        cell.setdefault('provided_activities', []).append({'id':'provided_laundry','anchors':['residence'],
            'category':'집','intent':'집에서 밀린 빨래 1회 세탁','purchase_channel':None,
            'billing_status':'no_transaction','evidence':laundry_evidence})
        duty = profile['duty']
        evidence = [laundry_evidence]
        if duty['kind'] != 'none':
            if duty['kind'] == 'work':
                if not p.get('work_poi_id'):raise ValueError('Explicit work-site profile requires a mapped site')
                anchor='workplace'; label='근무'; category='직장'
            elif duty['kind'] == 'school':
                anchor='zone:'+p['home_dong_code']; label='수업'; category='교육'
                if p['home_dong_code'] not in cell['zones']:raise ValueError('Declared school zone unavailable')
                for a,b in [('residence',anchor),(anchor,'residence')]:
                    cell.setdefault('minimum_transitions', []).append({'from_anchor':a,'to_anchor':b,'minimum_minutes':30})
            else:raise ValueError('Unknown explicitly supplied duty kind')
            duty_text = f"실험 가정: 오늘 {duty['start']}부터 {duty['end']}까지 {anchor}에서 {label} 일정이 확정되어 그 장소에 머문다. 실제 개인 시간표를 관측한 값은 아니다. 이 일정 자체의 당일 별도 결제는 없다."
            evidence.append(duty_text)
            activity='provided_'+duty['kind']+'_duty'
            cell['provided_activities'].append({'id':activity,'anchors':[anchor],'category':category,
                'intent':label+' 일정 참석','purchase_channel':None,'billing_status':'no_transaction','evidence':duty_text})
            cell.setdefault('required_activities', []).append({'time':duty['start'],'activity_id':activity,'anchor':anchor,'evidence':duty_text})
            cell.setdefault('required_presence_intervals', []).append({'start':duty['start'],'end':duty['end'],'anchor':anchor,'evidence':duty_text})
            conditions['duty']={'kind':duty['kind'],'start':duty['start'],'end':duty['end'],'anchor':anchor}
        else:
            conditions['duty']={'kind':'none','meaning':'이 조건에는 별도로 확정한 근무·수업 일정이 없다.'}
        cell['daily_conditions']=conditions
        marker='\n\n## 오늘\n'
        if cell['user'].count(marker)!=1:raise ValueError('Planner context boundary')
        block='\n\n## 정책 적용 전 고정한 오늘의 조건\n'+'\n'.join(evidence)+'\n'+json.dumps(conditions,ensure_ascii=False,indent=2)
        cell['user']=cell['user'].replace(marker,block+marker)
        cell['context_sha256']=digest(cell['user'])
    paired={}
    for cell in result['cells']:
        key=(cell['aid'],cell['case'],cell['date'])
        value=cell['daily_conditions']
        if key in paired and value!=paired[key]:raise ValueError('Policy-dependent initial daily conditions')
        paired[key]=value
    return result


def main():
    ap=argparse.ArgumentParser()
    for arg in ['source','profiles','out']:ap.add_argument('--'+arg,type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Refusing overwrite')
    raw=args.source.read_bytes();profile_raw=args.profiles.read_bytes()
    if len(raw)!=args.source.stat().st_size or len(profile_raw)!=args.profiles.stat().st_size:raise ValueError('Incomplete source file')
    result=prepare(json.loads(raw),json.loads(profile_raw)['profiles'])
    result['daily_conditions_provenance']={'source_sha256':hashlib.sha256(raw).hexdigest(),
        'profiles_sha256':hashlib.sha256(profile_raw).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Declared work/school and one household resource/optional need. Identical pre-policy day conditions. Not a complete demand model or observed routine.'}
    args.out.parent.mkdir(parents=True,exist_ok=True);atomic(args.out,result)
    print(json.dumps({'cells':len(result['cells']),'sha256':hashlib.sha256(args.out.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
