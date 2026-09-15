"""Paired scenario analysis with fixed population and explicit denominators.

This estimates differences inside the simulator, not human causal effects.
Run descriptors declare pairing; they do not prove identical input databases.
"""
import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from statistics import mean

from evidence_contract import EvidenceError, iso_day, seal, digest
from experience_export import build_report, validate_snapshot
from experience import observation_window
from experience_provenance import atomic_json, atomic_text


def dates(start, end):
    start, end = date.fromisoformat(iso_day(start)), date.fromisoformat(iso_day(end))
    if end < start:
        raise EvidenceError('reversed observation window')
    return [(start + timedelta(days=i)).isoformat() for i in range((end-start).days+1)]


def load_run(root, descriptor, days, population):
    folder = (root / descriptor['path']).resolve()
    rows, provenance = {}, []
    for day in days:
        cohort_path = folder / f'cohort_{day}.json'
        cohort = json.loads(cohort_path.read_text(encoding='utf-8'))
        if cohort.get('day') != day or cohort.get('run_id') != descriptor['run_id'] or set(cohort['agent_ids']) != set(population):
            raise EvidenceError('day/run/population differs from registered design')
        quality = build_report(folder, day, cohort=cohort)
        path = folder / 'metrics' / f'day_{day}.jsonl'
        raw = path.read_bytes()
        # Check that the gate and the loader saw the same artifact.
        if hashlib.sha256(raw).hexdigest() != quality['source_sha256']:
            raise EvidenceError('metrics changed during analysis')
        for line in raw.decode('utf-8').splitlines():
            row = json.loads(line)
            if row.get('status') != 'ok':
                continue
            validate_snapshot(row, day)
            if row.get('receipt_scope') != 'all_modeled_offline_commerce_v1':
                raise EvidenceError('legacy policy-only receipts cannot support baseline comparisons')
            ids = [r['event_id'] for r in row['execution_receipts']]
            if len(ids) != len(set(ids)):
                raise EvidenceError('duplicate executed receipt')
            rows[(row['aid'], day)] = row
        provenance.append({'day':day, 'metrics_sha256':quality['source_sha256'],
                           'cohort_sha256':quality['cohort_sha256']})
    if len({r['execution_fingerprint'] for r in rows.values()}) != 1:
        raise EvidenceError('execution settings changed within a run')
    return rows, provenance


STANCES = ('support','oppose','mixed','uncertain')


def assessment_status(appraisal, window, assessment, archive):
    if not appraisal:
        return 'no_appraisal'
    if not window[0] < appraisal['as_of'] <= assessment:
        return 'outside_window'
    for observation in appraisal['evidence_snapshot']:
        if not window[0] <= observation['observed_at'] <= window[-1]:
            return 'outside_window'
        # A self-consistent embedded snapshot is not proof of an executed event.
        actual = archive.get(observation['event_id'])
        if actual is None or actual != observation:
            raise EvidenceError('appraisal evidence differs from archived execution')
    return 'measured'


def analyze(design, root):
    if design.get('schema_version') != 1:
        raise EvidenceError('unsupported analysis design')
    policy = design['policy_id']
    population = design['population']  # agent -> pre-treatment group
    if not isinstance(policy,str) or not policy or not isinstance(population,dict) or not population:
        raise EvidenceError('policy and fixed population required')
    if any(not isinstance(k,str) or not k or not isinstance(v,str) or not v for k,v in population.items()):
        raise EvidenceError('population requires agent IDs and fixed group labels')
    window = dates(design['start'], design['end'])
    assessment = iso_day(design['assessment_day'])
    if assessment != (date.fromisoformat(window[-1])+timedelta(days=1)).isoformat():
        raise EvidenceError('assessment must be the next Dawn after the observation window')
    min_group = design.get('min_group_size',5)
    if type(min_group) is not int or min_group < 1:
        raise EvidenceError('invalid minimum group size')
    pairs = design['pairs']
    if not isinstance(pairs,list) or not pairs:
        raise EvidenceError('at least one paired replication required')
    all_days = window + [assessment]
    seen_seeds, seen_runs, results, inputs = set(), set(), [], []
    sources, models = set(), set()
    for pair in pairs:
        seed = pair['seed']
        if type(seed) is not int or seed in seen_seeds:
            raise EvidenceError('replications require distinct integer seed labels')
        seen_seeds.add(seed)
        arms = {}
        for arm in ('baseline','policy'):
            desc = pair[arm]
            if desc['run_id'] in seen_runs:
                raise EvidenceError('a run cannot be reused as an independent replication')
            seen_runs.add(desc['run_id'])
            arms[arm], hashes = load_run(root,desc,all_days,population)
            inputs.append({'seed':seed,'arm':arm,'run_id':desc['run_id'],'artifacts':hashes})
            for row in arms[arm].values():
                sources.add(row['source_fingerprint'])
                models.add(row['decision_provenance']['model_id'])
                if arm == 'baseline' and policy in (set(row['experience_policy_ids']) | set(row['policy_appraisals'])):
                    raise EvidenceError('baseline contains the target policy')
                if arm == 'baseline' and any(policy in r['policy_facts'] for r in row['execution_receipts']):
                    raise EvidenceError('baseline receipt contains the target policy')
        for key in arms['baseline']:
            before, after = arms['baseline'][key], arms['policy'][key]
            if set(before['experience_policy_ids']) != set(after['experience_policy_ids']) - {policy}:
                raise EvidenceError('non-target policy exposure differs across scenarios')
        groups = {}
        if not any(policy in row['experience_policy_ids'] for row in arms['policy'].values()):
            raise EvidenceError('policy scenario never contains the target policy')
        for aid, group in population.items():
            bucket = groups.setdefault(group, [])
            totals, target_receipts = {}, []
            for arm, data in arms.items():
                receipts = [r for day in window for r in data[(aid,day)]['execution_receipts']]
                totals[arm] = {'spend':sum(r['amount'] for r in receipts),
                               'own_paid':sum(r['own_paid'] for r in receipts)}
                if arm == 'policy':
                    target_receipts = [r for r in receipts if policy in r['policy_facts']]
            final = arms['policy'][(aid,assessment)]
            appraisal = final['policy_appraisals'].get(policy)
            # Report only assessments from this observation window; old views
            # remain separately unmeasured for this study.
            archive = {}
            for day in window:
                for receipt in arms['policy'][(aid,day)]['execution_receipts']:
                    observed = observation_window([], [receipt])[0]
                    if observed['event_id'] in archive:
                        raise EvidenceError('executed event ID reused across days')
                    archive[observed['event_id']] = observed
            status = assessment_status(appraisal, window, assessment, archive)
            measured = status == 'measured'
            incompatible = any(not r['policy_facts'][policy]['eligible_under_modeled_rules'] for r in target_receipts)
            bucket.append({'spend_delta':totals['policy']['spend']-totals['baseline']['spend'],
                           'own_paid_delta':totals['policy']['own_paid']-totals['baseline']['own_paid'],
                           'observed':bool(target_receipts),
                           'used':any(r['policy_facts'][policy]['paid'] > 0 for r in target_receipts),
                           'incompatible':incompatible,
                           'assessment_status':status,
                           'as_of':appraisal['as_of'] if measured else None,
                           'stance':appraisal['stance'] if measured else None})
        for group, members in sorted(groups.items()):
            n = len(members)
            counts = Counter(m['stance'] for m in members if m['stance'] is not None)
            measured_n = sum(counts.values())
            strata = []
            for incompatible in (False,True):
                subset = [m for m in members if m['observed'] and m['incompatible'] == incompatible]
                measured_subset = [m for m in subset if m['stance'] is not None]
                strata.append({'had_modeled_ineligible_purchase':incompatible,'agents':len(subset),
                               'measured':len(measured_subset),
                               'opposition_share_among_measured':
                                   sum(m['stance']=='oppose' for m in measured_subset)/len(measured_subset)
                                   if len(measured_subset) >= min_group else None})
            result = {'seed':seed,'group':group if n >= min_group else 'suppressed',
                      'agents':n,'suppressed':n < min_group}
            if n >= min_group:
                result.update(mean_offline_spend_delta=mean(m['spend_delta'] for m in members),
                              mean_offline_own_paid_delta=mean(m['own_paid_delta'] for m in members),
                              observed_agents=sum(m['observed'] for m in members),
                              policy_users=sum(m['used'] for m in members),
                              transaction_observation_rate=sum(m['observed'] for m in members)/n,
                              policy_use_rate=sum(m['used'] for m in members)/n,
                              ineligible_purchase_agents=sum(m['incompatible'] for m in members),
                              measured=measured_n,unmeasured=n-measured_n,
                              measurement_coverage=measured_n/n,
                              missingness_reasons=dict(Counter(m['assessment_status'] for m in members if m['stance'] is None)),
                              appraisal_dates=dict(Counter(m['as_of'] for m in members if m['as_of'])) if measured_n >= min_group else None,
                              population_share_bounds={k:{'lower':counts[k]/n,'upper':(counts[k]+n-measured_n)/n}
                                                       for k in STANCES} if measured_n >= min_group else None,
                              stance_counts=dict(counts) if measured_n >= min_group else None,
                              stance_share_among_measured={k:v/measured_n for k,v in counts.items()}
                                  if measured_n >= min_group else None,
                              experience_stance_association=strata)
            results.append(result)
    if len(sources) != 1 or len(models) != 1:
        raise EvidenceError('source or model differs across paired experiments')
    stability = []
    for group in sorted(set(population.values())):
        rows = [r for r in results if not r['suppressed'] and r['group']==group]
        if not rows:
            continue
        for metric in ('mean_offline_spend_delta','mean_offline_own_paid_delta','measurement_coverage',
                       'transaction_observation_rate','policy_use_rate'):
            values = [r[metric] for r in rows]
            stability.append({'group':group,'metric':metric,'replications':len(values),
                              'mean':mean(values),'min':min(values),'max':max(values),
                              'direction':'positive' if min(values)>0 else 'negative' if max(values)<0 else 'zero' if min(values)==max(values)==0 else 'varies_or_touches_zero',
                              'interpretation':'between-run range, not a confidence interval'})
        for stance in STANCES:
            measured_rows = [r for r in rows if r['stance_share_among_measured'] is not None]
            values = [r['stance_share_among_measured'].get(stance,0) for r in measured_rows]
            stability.append({'group':group,'metric':f'{stance}_share_among_measured',
                              'replications':len(values),'excluded_replications':len(rows)-len(values),
                              'mean':mean(values) if values else None,
                              'min':min(values) if values else None,'max':max(values) if values else None,
                              'interpretation':'equal-weight run summaries; measured people can differ between runs'})
    return seal({'schema_version':1,'measure':'paired_simulated_policy_response',
                 'claim_level':'simulation_scenario_only',
                 'replication_status':'single_replication' if len(pairs)==1 else 'replicated_descriptive',
                 'design_sha256':digest(design),'inputs':inputs,'groups':results,'stability':stability,
                 'release_status':'descriptive_only_unverified_pairing',
                 'population_bounds_interpretation':'Worst-case missing-response bounds, not confidence intervals.',
                 'limitations':['Pairing and seed labels are declared, not verified RNG or database identities.',
                                'Offline modeled transactions only; online spending and travel are excluded.',
                                'Ineligible purchase is not a payment rejection or proof of unavailable alternatives.',
                                'Experience/stance associations are descriptive, not causal attribution.',
                                'No human survey validation or calibrated predictive uncertainty supplied.']})


def render_report(report):
    def cell(value):
        if value is None:
            return '미측정/억제'
        if isinstance(value,float):
            return f'{value:.4f}'
        return str(value).replace('|','/').replace('\n',' ')
    lines = ['# 정책 반응 시나리오 분석', '',
             '이 결과는 모형 내부의 기술적 비교다. 실제 여론 예측 성능이나 정책의 현실 인과효과를 검증한 결과가 아니다.', '',
             f"설계 해시: `{report['design_sha256']}`", '',
             '## 집단별 결과', '',
             '구매액·본인 부담 차이는 관측 기간의 정책 실행 − 기준 실행이다. 평가일 거래는 제외한다.', '',
             '| 시드 | 집단 | 인원 | 측정 | 미측정 | 구매액 차이(원/인) | 본인 부담 차이(원/인) |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in report['groups']:
        lines.append('| '+' | '.join(cell(row.get(k)) for k in
                     ('seed','group','agents','measured','unmeasured','mean_offline_spend_delta','mean_offline_own_paid_delta'))+' |')
    lines += ['', '## 입장과 결측', '',
              '응답자 비율과 전체 집단의 가능한 범위를 구분한다. 범위는 미측정자에 대한 최악 경우 경계이며 신뢰구간이 아니다.', '',
              '| 시드 | 집단 | 입장 | 측정자 중 비율 | 전체 인구 하한 | 전체 인구 상한 |',
              '|---|---|---|---:|---:|---:|']
    for row in report['groups']:
        if row.get('population_share_bounds') is None:
            continue
        for stance, bounds in row['population_share_bounds'].items():
            values = [row['seed'],row['group'],stance,row['stance_share_among_measured'].get(stance,0),bounds['lower'],bounds['upper']]
            lines.append('| '+' | '.join(map(cell,values))+' |')
    lines += ['', '## 경험과 입장의 연관', '',
              '정책 관련 거래를 관측한 사람만 두 경험 집단으로 나눈다. 반대율의 분모는 각 경험 집단의 측정자이며 인과효과를 뜻하지 않는다.', '',
              '| 시드 | 집단 | 사용 불가 거래 관측 | 경험 인원 | 측정 인원 | 측정자 반대율 |',
              '|---|---|---|---:|---:|---:|']
    for row in report['groups']:
        for stratum in row.get('experience_stance_association',[]):
            values=[row['seed'],row['group'],stratum['had_modeled_ineligible_purchase'],
                    stratum['agents'],stratum['measured'],stratum['opposition_share_among_measured']]
            lines.append('| '+' | '.join(map(cell,values))+' |')
    lines += ['', '## 반복 실행 범위', '',
              '시드별 결과를 동일 가중치로 요약한다. 범위는 신뢰구간이 아니며 한 번 실행한 결과로 안정성을 판단할 수 없다.', '',
              '| 집단 | 지표 | 유효 반복 | 평균 | 최소 | 최대 |', '|---|---|---:|---:|---:|---:|']
    for row in report['stability']:
        lines.append('| '+' | '.join(cell(row.get(k)) for k in ('group','metric','replications','mean','min','max'))+' |')
    lines += ['', '## 해석 제한', '',
              '- 거래 이용은 찬성이 아니며, 사용 불가 거래는 실제 결제 거절이나 대체 장소 부재를 증명하지 않는다.',
              '- 거래 경험과 입장의 연관은 기술적 결과다. 원인으로 단정하지 않는다.',
              '- 초기 DB·시드 적용·전체 서빙 조건의 동일성은 아직 자동 입증되지 않았다.',
              '- 온라인 지출·이동·대기·신청 경험과 인간 설문 정확도는 이 결과의 검증 범위 밖이다.',
              '- 근거 원장, 결측 사유, 측정일, 원본 파일 해시는 함께 생성한 JSON에서 감사할 수 있다.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', help='optional human-readable Markdown report')
    args = parser.parse_args()
    path = Path(args.design).resolve()
    output = Path(args.out).resolve()
    report_path = Path(args.report).resolve() if args.report else None
    if report_path and (report_path in (path, output) or report_path.suffix.lower() != '.md'):
        parser.error('report must be a separate Markdown file')
    if output == path or output.suffix.lower() != '.json':
        parser.error('output must be a separate JSON artifact')
    design = json.loads(path.read_text(encoding='utf-8'))
    # Keep output outside run directories so no original evidence is overwritten.
    for pair in design['pairs']:
        for arm in ('baseline','policy'):
            if output.is_relative_to((path.parent/pair[arm]['path']).resolve()):
                parser.error('output must be outside input run directories')
            if report_path and report_path.is_relative_to((path.parent/pair[arm]['path']).resolve()):
                parser.error('report must be outside input run directories')
    result = analyze(design,path.parent)
    atomic_json(output, result)
    if report_path:
        atomic_text(report_path, render_report(result))
    print(output)


if __name__ == '__main__':
    main()
