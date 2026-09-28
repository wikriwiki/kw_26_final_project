"""Make policy-vintage population targets from preserved official sources.

Does not calibrate a roster, write to Neo4j, or allow a model call. Population
counts and annual household-income covariates stay separate from policy effects.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from tools.build_seoul_population_frames_20260928 import evidence, sha, write

ROOT = Path(__file__).resolve().parents[1]
PERIODS = {'P013':'202004','P016':'202006','P014':'202008',
           'DISTANCING_2020':'202010','P012':'202109','P010':'202506'}
INCOME_YEARS = {'P013':2019,'P016':2019,'P014':2019,
                'DISTANCING_2020':2019,'P012':2021,'P010':2024}
FIELDS = ('sex','age_band','admin_dong')


def demographic_counts(records, period):
    if not records or any(str(r['stats_ym']) != period for r in records):
        raise ValueError('Official demographic source period differs')
    all_age_codes = {-1, *range(5,101,5), 101}
    adult_age_codes = {*range(25,101,5), 101}
    if any(int(r['agegrd_cd']) not in all_age_codes for r in records):
        raise ValueError('Unknown official five-year age code')
    if any(r['sex_nm'] not in ('남자','여자','계') for r in records):
        raise ValueError('Unknown official sex category')
    selected = [r for r in records if r['epmndn_nm'] != '계' and
                r['sex_nm'] in ('남자','여자') and int(r['agegrd_cd']) >= 25]
    keys = {(r['admdong_cd'],r['sex_nm'],r['agegrd_cd']) for r in selected}
    if len(keys) != len(selected):
        raise ValueError('Official demographic source has duplicate cells')
    codes = {r['admdong_cd'] for r in selected}
    expected_keys = {(code,sex,str(age)) for code in codes for sex in ('남자','여자')
                     for age in adult_age_codes}
    if {(str(code),sex,str(age)) for code,sex,age in keys} != expected_keys:
        raise ValueError('Incomplete dong sex by adult five-year age grid')
    counts = {key:Counter() for key in FIELDS}
    joint = []
    for row in selected:
        code = str(row['admdong_cd'])
        if len(code) != 10 or not code.isdigit():
            raise ValueError('Native source code must remain 10 digits')
        number = int(row['popl_cnt'])
        if number < 0:
            raise ValueError('Negative demographic count')
        lower_age = 100 if int(row['agegrd_cd']) == 101 else int(row['agegrd_cd']) - 5
        age = f'{lower_age//10*10}대' if lower_age < 60 else '60세 이상'
        sex = {'남자':'M','여자':'F'}[row['sex_nm']]
        counts['sex'][sex] += number
        counts['age_band'][age] += number
        counts['admin_dong'][code] += number
        joint.append({'admin_dong':code,'admin_dong_name':row['epmndn_nm'],
                      'gu_name':row['sigungu_nm'],'sex':sex,'age_band':age,
                      'age_group5':row['agegrd_nm'],'count':number})
    totals = {sum(v.values()) for v in counts.values()}
    if len(totals) != 1 or not next(iter(totals)) > 0:
        raise ValueError('Demographic totals do not reconcile')
    return counts, joint, totals.pop()


def build(policy, period, source_dir, out, manifest, alignment_path):
    source_entry = next(r for r in manifest['results'] if r['policy'] == policy)
    if source_entry['period'] != period:
        raise ValueError('Source manifest period mismatch')
    csv_path = source_dir / f'official_resident_sex_age_dong_{period}.from_json.csv'
    source_record = next(r for r in source_entry['files'] if r['path'].endswith('.from_json.csv'))
    if sha(csv_path) != source_record['sha256']:
        raise ValueError('Official derived CSV SHA mismatch')
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        counts, joint, total = demographic_counts(list(csv.DictReader(stream)), period)
    year = int(period[:4])
    targets = {}
    for field, counter in counts.items():
        targets[field] = {'proportions':{k:v/total for k,v in sorted(counter.items())},
                          'source':evidence(csv_path),'population_unit':'resident_person',
                          'reference_year':year,'source_population_unit':'resident_person',
                          'source_reference_year':year,'source_reference_period':period,
                          'source_url':source_entry['source_page'],
                          'field_definition':f'{period} 서울 내국인 주민등록 인구 중 20세 이상 {field}'}
    targets['admin_dong']['code_system'] = f'SeoulDataHub_KOSIS_DT_1B04005N_native_10digit_{period}'
    income_year = INCOME_YEARS[policy]
    income_path = out / f'income_distribution_{income_year}.json'
    income = json.loads(income_path.read_text(encoding='utf-8'))
    if income['reference_year'] != income_year or income['missing_income_rows'] != 0:
        raise ValueError('Income provenance or household join failed')
    targets['income_band'] = {'proportions':income['proportions'],
                             'source':evidence(income_path),'population_unit':'resident_person',
                             'reference_year':year,'source_population_unit':'resident_person',
                             'source_reference_year':income_year,'source_url':income['source_url'],
                             'field_definition':income['income_definition']+'; 가구원 원가중치, 20세 이상',
                             'population_alignment':{
                                 'method':'시행 이전에 측정된 연간 소득 주변분포를 초기 공변량으로 사용하는 명시적 시점 이월 가정. 시행 전월 소득분포 실측으로 간주하지 않음.',
                                 'source_evidence':[evidence(alignment_path)]}}
    path = out / f'policy_population_targets_{policy}_{period}.json'
    write(path, {'schema':'population_calibration_frame_v1','policy':policy,
                 'reference_year':year,'reference_period':period,
                 'population_unit':'resident_person','age_scope':'20세 이상 내국인 주민등록 시민',
                 'resident_count':total,'administrative_dong_count':len(counts['admin_dong']),
                 'targets':targets,'demographic_joint_counts':joint,
                 'source_manifest_evidence':evidence(source_dir/'official_population_months_manifest.json'),
                 'status':'official_targets_ready_not_a_calibrated_cohort',
                 'model_calls_allowed':False,
                 'limitations':[
                     '성별×5세연령×행정동 결합분포는 공식 자료에서 관측하였으나 소득을 포함한 4차원 결합분포는 관측하지 못함.',
                     f'소득은 {income_year}년 조사 표본의 가구원 가중 분포. 시행 전월 소득을 직접 측정한 값이 아님.',
                     '현재 Agent의 실제 거주 anchor와 공식 통계 코드 namespace·정책시점 경계를 연결한 감사 전에는 사용할 수 없음.',
                     '20세 미만과 외국인 시민은 이 프레임에 포함되지 않음. 전체 시민/전체 수혜자 결과로 확장할 수 없음.',
                     '후보 시민에 소득을 합성 배정하고 네 주변분포·결합분포·실행 입력 전달을 확인한 후 별도 실험을 사전등록해야 함.']})
    return {'policy':policy,'period':period,'income_source_year':income_year,
            'administrative_dongs':len(counts['admin_dong']),'adult_population':total,
            **evidence(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir',type=Path,default=ROOT/'output/population_frame_sources_20260928')
    parser.add_argument('--out',type=Path,default=ROOT/'output/population_matching_20260928')
    args = parser.parse_args()
    manifest = json.loads((args.source_dir/'official_population_months_manifest.json').read_text(encoding='utf-8'))
    alignment_path = args.out/'prepolicy_income_alignment.json'
    write(alignment_path, {'schema':'annual_income_population_alignment_v1',
                          'income_definition':'개인 근로소득이나 지출 분위가 아닌 세전 월평균 총 가구소득',
                          'income_years_by_policy':INCOME_YEARS,
                          'method':'정책 시행 이전의 연간 조사 주변분포를 초기 공변량으로 이월. 그 이후 변화를 관측한 값으로 주장하지 않음.',
                          'assumptions':['해당 연간 소득 구성비를 초기 시민 합성에 적용하는 시점 이월 가정',
                                         '자치구×성별×연령 내 행정동별 소득 조건부 분포는 별도 관측되지 않음'],
                          'limitations':['시행 전월의 동별 소득 실측 4차원 결합분포가 아님',
                                         '팬데믹과 임금·물가 변화에 대한 민감도 분석 필요'],
                          'source_evidence':[evidence(args.out/f'income_distribution_{y}.json')
                                             for y in sorted(set(INCOME_YEARS.values()))],
                          'policy_outcome_used':False,'model_calls_allowed':False})
    results = [build(policy,period,args.source_dir,args.out,manifest,alignment_path)
               for policy,period in PERIODS.items()]
    write(args.out/'policy_population_targets_manifest.json',
          {'schema':'six_policy_population_target_manifest_v1','results':results,
           'calibrated_cohort_completed':False,'population_matched_simulation_completed':False,
           'model_calls_allowed':False})
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
