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
from experience_provenance import atomic_json


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
    return rows, provenance


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
            measured = bool(appraisal and window[0] < appraisal['as_of'] <= assessment and
                            all(window[0] <= r['observed_at'] <= window[-1]
                                for r in appraisal['evidence_snapshot']))
            incompatible = any(not r['policy_facts'][policy]['eligible_under_modeled_rules'] for r in target_receipts)
            bucket.append({'spend_delta':totals['policy']['spend']-totals['baseline']['spend'],
                           'own_paid_delta':totals['policy']['own_paid']-totals['baseline']['own_paid'],
                           'observed':bool(target_receipts),
                           'used':any(r['policy_facts'][policy]['paid'] > 0 for r in target_receipts),
                           'incompatible':incompatible,
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
                              'interpretation':'between-run range, not a confidence interval'})
    return seal({'schema_version':1,'measure':'paired_simulated_policy_response',
                 'claim_level':'simulation_scenario_only',
                 'replication_status':'single_replication' if len(pairs)==1 else 'replicated_descriptive',
                 'design_sha256':digest(design),'inputs':inputs,'groups':results,'stability':stability,
                 'limitations':['Pairing and seed labels are declared, not verified RNG or database identities.',
                                'Offline modeled transactions only; online spending and travel are excluded.',
                                'Ineligible purchase is not a payment rejection or proof of unavailable alternatives.',
                                'Experience/stance associations are descriptive, not causal attribution.',
                                'No human survey validation or calibrated predictive uncertainty supplied.']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = Path(args.design).resolve()
    output = Path(args.out).resolve()
    if output == path or output.suffix.lower() != '.json':
        parser.error('output must be a separate JSON artifact')
    design = json.loads(path.read_text(encoding='utf-8'))
    # Keep output outside run directories so no original evidence is overwritten.
    for pair in design['pairs']:
        for arm in ('baseline','policy'):
            if output.is_relative_to((path.parent/pair[arm]['path']).resolve()):
                parser.error('output must be outside input run directories')
    atomic_json(output, analyze(design,path.parent))
    print(output)


if __name__ == '__main__':
    main()
