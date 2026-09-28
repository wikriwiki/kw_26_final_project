"""CPU-only, post-result PDF proxies from preserved P010/P012 ledgers.

Never invokes a model, connects to a server, changes a graph, or rescales a
frozen core result. Catalog bytes must be fixed before this helper is run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

CATALOG_SHAS = {
    'P010': '1c1576e6c04bc13596a905c4c2b9a9c3c5d66f7cdbc70665dec8c4e26458c530',
    'P012': '97cdc74c62d952d6faf5abac3b6c9f6f3b361629d811dd46538467df0b553150',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path, root):
    return path.resolve().relative_to(root.resolve()).as_posix()


def evidence(path, root):
    return {'path': relative(path, root), 'sha256': sha(path)}


def load_ledger(path, expected_arm):
    manifest_path = path.with_suffix(path.suffix + '.manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['output_sha256'].lower() != sha(path):
        raise ValueError(f'ledger SHA mismatch: {path}')
    rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    keys = [(x['aid'], x['day']) for x in rows]
    if len(keys) != len(set(keys)) or len(rows) != manifest['rows']:
        raise ValueError(f'nonunique/count-mismatched ledger: {path}')
    if any(x['arm'] != expected_arm for x in rows):
        raise ValueError(f'arm mismatch: {path}')
    return rows, manifest


def percent_change(on, off):
    return 100.0 * (on / off - 1.0) if off > 0 else None


def sums(rows, field, names=None):
    if names is None:
        return sum(x[field] for x in rows)
    return sum(sum(x[field].get(name, 0) for name in names) for x in rows)


def result(id, value, unit, formula, raw, n, scope, notes):
    return {'id': id, 'simulation': value, 'simulation_unit': unit,
            'formula': formula, 'raw_components': raw, 'sample_citizens': n,
            'scope_note': scope+' Current roster is spending-decile stratified only; this is not income-distribution matching.', 'quality_notes': notes, 'ci': None,
            'direct_gap_allowed': False, 'direction_comparable': False}


def inputs(root, policy):
    catalog_path = root / f'experiments/pdf_benchmark_expansion_20260928/{policy.lower()}_catalog.json'
    if sha(catalog_path) != CATALOG_SHAS[policy]:
        raise ValueError('Catalog changed after definitions were fixed')
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    if sha(root / catalog['source']['path']) != catalog['source']['sha256']:
        raise ValueError('Source PDF changed')
    base = root / f'output/recovery_20260928/multipolicy_v53/{policy.lower()}'
    paths = {arm: base / arm / arm / 'sector.ledger.jsonl' for arm in ['on', 'off']}
    ledgers = {}
    manifests = {}
    proofs = [evidence(catalog_path, root), evidence(root / catalog['source']['path'], root)]
    for arm, path in paths.items():
        ledgers[arm], manifests[arm] = load_ledger(path, arm)
        proofs.extend([evidence(path, root), evidence(path.with_suffix(path.suffix + '.manifest.json'), root)])
    if {(x['aid'], x['day']) for x in ledgers['on']} != {(x['aid'], x['day']) for x in ledgers['off']}:
        raise ValueError('Paired roster/date support differs')
    for key in ['cohort_sha256', 'roster_sha256']:
        if manifests['on'][key] != manifests['off'][key]:
            raise ValueError(f'Paired {key} mismatch')
    for key in ['system_prompt_sha256', 'stage2_system_prompt_sha256', 'baseline_income_map_sha256']:
        if manifests['on']['prompt_provenance'].get(key) != manifests['off']['prompt_provenance'].get(key):
            raise ValueError(f'Paired {key} mismatch')
    if any(manifests[a]['prompt_provenance'].get('prompt_variant') != 'v53' for a in ['on', 'off']):
        raise ValueError('Expected v53 pair')
    numeric_path = root / f'output/multi_policy_v53_20260928/{policy.lower()}/numeric.json'
    proofs.append(evidence(numeric_path, root))
    return catalog_path, catalog, base, ledgers, manifests, numeric_path, proofs


def p010(root):
    cp, cat, base, ledgers, manifests, numeric, proofs = inputs(root, 'P010')
    window = (manifests['on']['effective_from'], manifests['on']['effective_until'])
    rows = [x for x in ledgers['on'] if window[0] <= x['day'] <= window[1]]
    n = len({x['aid'] for x in rows})
    observed_window = (min(x['day'] for x in rows), max(x['day'] for x in rows))
    grants = {}
    for x in ledgers['on']:
        grants[x['aid']] = max(grants.get(x['aid'], 0), x['grant_received_cumulative'])
    denominator = sum(grants.values())
    if denominator <= 0:
        raise ValueError('No issued-grant denominator')
    funded_total = sums(rows, 'policy_funded_won')
    positive_cd = sum(x['policy_funded_won'] > 0 for x in rows)
    outputs = []
    maps = []
    for entry in cat['entries']:
        if not entry['id'].startswith('P010-X-F19-USE-'):
            continue
        names = entry['simulation']['poi_mapping']
        maps.extend(names)
        amount = sums(rows, 'funded_by_sub', names)
        outputs.append(result(entry['id'], 100 * amount / denominator, '%',
            entry['simulation']['proposed_formula'],
            {'policy_funded_won': amount, 'grant_issued_won': denominator,
             'all_policy_funded_won': funded_total, 'policy_funded_positive_citizen_days': positive_cd,
             'citizen_days': len(rows), 'mapped_subclasses': names}, n,
            f'v53 first round, Seoul {n} citizens; observed {observed_window[0]}..{observed_window[1]}; wallet payments / issued grants.',
            ['Post-result exploratory. Empirical figure 19 is approximate, first/second-round weighted survey use/plans.',
             'Only two positive policy-funded citizen-days; most zeros reflect where these few wallet payments occurred, not a tested population response.',
             'Denominator is issued grant amount, not redeemed amount. No same-estimand gap/directional-hit/size accuracy.']))
    if len(maps) != len(set(maps)):
        raise ValueError('Figure-19 item proxy groups overlap')
    if sum(r['raw_components']['policy_funded_won'] for r in outputs) > funded_total:
        raise ValueError('Item-funded sums exceed all funded total')
    return package(root, 'P010', cp, numeric, proofs, outputs)


def p012(root):
    cp, cat, base, ledgers, manifests, numeric, proofs = inputs(root, 'P012')
    on, off = ledgers['on'], ledgers['off']
    n = len({x['aid'] for x in on})
    if len(on) != 31 * n or {x['day'] for x in on} != {f'2021-10-{d:02}' for d in range(1, 32)}:
        raise ValueError('Expected complete October citizen-day support')
    lookup = {x['id']: x for x in cat['entries']}
    out = []
    def add_growth(id, label, on_won, off_won, mapping=None, note=''):
        item = lookup[id]
        out.append(result(id, percent_change(on_won, off_won), '%',
            '100 * (paired October ON mapped spend / OFF mapped spend - 1); OFF=0 gives null.',
            {'on_won': on_won, 'off_won': off_won, 'citizen_days_each_arm': len(on), 'mapped_subclasses': mapping},n,
            f'October 2021, {n} identical individual citizens ×31 days, paired ON/OFF; {label}.',
            ['Post-result exploratory; source is household recipient/nonrecipient triple difference with September and 2019 controls; simulation is different paired percent growth.',
             'Source log coefficient is not raw percent growth. Card-bank sector crosswalk is provisional, not verified.',
             note] if note else ['Post-result exploratory; household/recipient/month/year estimand differs; bank sector crosswalk is provisional.']))
    add_growth('P012-X-T4-7-59-1-M10','P012-flag eligible offline spending',sums(on,'sangsaeng_eligible_offline_spent'),sums(off,'sangsaeng_eligible_offline_spent'))
    covered=[]
    mappings = {}
    for id in ['P012-X-T4-7-59-2-M10','P012-X-T4-7-59-3-M10','P012-X-T4-7-59-4-M10',
               'P012-X-T4-7-60-1-M10','P012-X-T4-7-60-2-M10','P012-X-T4-7-60-3-M10']:
        mapping = lookup[id]['simulation']['poi_mapping']
        mappings[id]=mapping
        covered.extend(mapping)
        add_growth(id,lookup[id]['label'],sums(on,'by_sub',mapping),sums(off,'by_sub',mapping),mapping,
                   'by_sub group sums include offline purchases irrespective of per-transaction P012 eligibility; per-sub eligibility is not stored in this ledger.')
    if len(covered)!=len(set(covered)):
        raise ValueError('P012 six declared POI proxy groups overlap')
    observed = {name for row in on+off for name in row['by_sub']}
    remainder = sorted(observed-set(covered))
    add_growth('P012-X-T4-7-60-4-M10','other classified OFFLINE spend after the six disjoint maps',sums(on,'by_sub',remainder),sums(off,'by_sub',remainder),remainder,
               'Residual POI proxy includes medical/pharmacy etc and can include ineligible receipts; it is not the bank exact eligible-industry other bucket.')
    add_growth('P012-X-T4-7-61-1-M10','online plus ineligible offline account',
               sums(on,'total_spent')-sums(on,'sangsaeng_eligible_offline_spent'),
               sums(off,'total_spent')-sums(off,'sangsaeng_eligible_offline_spent'),
               note='Complementary accounting proxy; bank excluded-industry merchant classification is not reproduced.')
    add_growth('P012-X-T4-7-61-3-M10','entire online account',sums(on,'online_spent'),sums(off,'online_spent'),
               note='Source ONLINE coefficient is -0.0030, distinct from excluded TOTAL +0.0285. Online channel allocation is shaped by the consumption engine.')
    add_growth('P012-X-T4-4-5-M10','total individual recorded spending',sums(on,'total_spent'),sums(off,'total_spent'))
    add_growth('P012-X-T4-6-57-3-M10','Seoul-only total individual recorded spending',sums(on,'total_spent'),sums(off,'total_spent'),
               note='All simulation citizens live in Seoul; exact household panel source population is still different.')
    cashback_path=base/'on/on/cashback.ledger.jsonl'
    cash_rows,cash_manifest=load_ledger(cashback_path,'on')
    proofs.extend([evidence(cashback_path,root),evidence(cashback_path.with_suffix(cashback_path.suffix+'.manifest.json'),root)])
    final=[x for x in cash_rows if x['day']=='2021-10-31']
    if len(final)!=n or {x['aid'] for x in final}!={x['aid'] for x in on}:
        raise ValueError('End-month cashback roster incomplete')
    amounts=[float(x['cashback_accrued_won']) for x in final]
    if any(x<0 or x>100000 for x in amounts):
        raise ValueError('Rule-accrual outside 0..100000 cap')
    positive=[x for x in amounts if x>0]
    total=sum(positive)
    total_bin_count=0
    for item in cat['entries']:
        if not item['id'].startswith('P012-X-T31-M10-BIN'):
            continue
        e=item['empirical'];lo=e['lower_won'];hi=e['upper_won']
        count=sum(x==100000 if e['bin_index']==11 else (x>0 and lo<=x<hi) for x in amounts)
        total_bin_count+=count
        out.append(result(item['id'],100*count/len(positive) if positive else None,'%',
            item['simulation']['proposed_formula'],
            {'count':count,'recipients':len(positive),'sample_citizens':n,'total_cashback_accrual_won':total,
             'lower_won':lo,'upper_won':hi,'cashback_zero_citizens':n-len(positive)},n,
            '2021-10-31 rule-based cashback accrual distribution, positive-accrual denominator; actual payment not simulated.',
            ['Post-result exploratory; source finalized paid recipient distribution is reconstructed from rounded administrative counts.',
             'Only 11 positive recipients out of 12; a zero sample bin is not evidence of a zero population proportion.']))
    if total_bin_count!=len(positive):
        raise ValueError('Cashback bins do not partition positive accrual recipients')
    out.append(result('P012-X-T31-M10-MEAN',total/len(positive) if positive else None,'KRW per positive recipient',
        'sum(end-month rule-accrual) / positive-accrual citizens',
        {'total_cashback_accrual_won':total,'recipients':len(positive),'sample_citizens':n},n,
        'October end-month accrued reward per positive-accrual individual, not actual next-month payment.',
        ['Post-result exploratory; source mean is actual nationwide administrative paid reward / positive recipient-months.']))
    delta=sums(on,'total_spent')-sums(off,'total_spent')
    out.append(result('P012-X-IV-C4',delta/total*10 if total>0 else None,'thousand KRW per 10000 KRW cashback',
        '(sum(ON total_spent)-sum(OFF total_spent))/sum(ON end-month rule-accrual)*10',
        {'on_won':sums(on,'total_spent'),'off_won':sums(off,'total_spent'),'paired_difference_won':delta,
         'total_cashback_accrual_won':total,'recipients':len(positive)},n,
        'Aggregate grant-normalized paired October response per rule-accrual; not IV or a marginal receipt effect.',
        ['Post-result exploratory; self-selection and household/year/source-instrument estimation are not reproduced.',
         'The denominator is rule-accrual not an actual payment transaction; no 2SLS accuracy or multiplier validation.']))
    return package(root,'P012',cp,numeric,proofs,out)


def package(root,policy,catalog,numeric,proofs,rows):
    return {'schema':'pdf_benchmark_simulation_v1','policy':policy,
            'created_at':datetime.now(timezone.utc).isoformat(),
            'timing':'post-result exploratory','analysis_timing':'Defined after policy results; CPU-only descriptive proxies; not preregistered validation',
            'catalog_path':relative(catalog,root),'catalog_sha256':sha(catalog),
            'numeric_path':relative(numeric,root),'numeric_sha256':sha(numeric),
            'source_evidence':proofs,'rows':rows,
            'restrictions':['No model calls or server changes. Frozen core numeric results unchanged.',
                            'No direct errors, direction hit or effect-size accuracy; scope differences remain explicit.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--output-dir',type=Path,default=Path('output/pdf_benchmark_expansion_20260928'))
    args=parser.parse_args()
    root=args.root.resolve();dest=args.output_dir if args.output_dir.is_absolute() else root/args.output_dir
    dest.mkdir(parents=True,exist_ok=True)
    for policy,build in [('P010',p010),('P012',p012)]:
        obj=build(root)
        path=dest/f'{policy.lower()}_simulation.json'
        path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(policy,'rows',len(obj['rows']),'numeric',sum(x['simulation'] is not None for x in obj['rows']),'SHA',sha(path))


if __name__=='__main__': main()
