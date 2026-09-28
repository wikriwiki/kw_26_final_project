"""CPU-only candidate support using direct MOIS and documented dong changes.

The result is a grouping proposal, not historical polygon verification or a
selected/calibrated sample. No persona/home, source frame or graph is edited.
"""
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/population_frame_sources_20260928'
CROOT=Path('C:/Users/Administrator/Documents/kw26_a100_recovery_20260928/population_frame_sources')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    manifestpath=OUT/'official_mois_population_months_manifest.json'
    changepath=OUT/'official_dong_change_relations_audit.json'
    graphpath=OUT/'graph_full_population_code_projection_20260928.json'
    manifest=json.loads(manifestpath.read_text(encoding='utf-8'))
    graph=json.loads(graphpath.read_text(encoding='utf-8'))
    graphcodes={r['code']for r in graph['dongs']}
    evidence=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)}for p in[manifestpath,changepath,graphpath]]
    results=[]
    for source in manifest['results']:
        month=source['reference_month']
        path=OUT/f'official_mois_age_seoul_{month}.long.from_csv.csv'
        assert sha(path)==next(e['sha256']for e in source['source_evidence']if e['path']==str(path.relative_to(ROOT)))
        evidence.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
        raw=list(csv.DictReader(path.open(encoding='utf-8-sig')))
        targets={r['mois_admin_code10']for r in raw if r['epmndn_nm']!='계'}
        mapping={c:c+'00'for c in graphcodes}
        # Every requested month precedes the July2025 Yongsin split.
        if month<'202507':mapping.update({'11230515':'1123053600','11230533':'1123053600'})
        if month<'202107':
            mapping.update({'11740525':'1174052000','11740526':'1174051500'})
        if month<'202212':mapping.update({'11680675':'1168074000'})
        assert set(mapping.values())==targets
        candidates=[a for a in graph['agents']if a['sex']in{'M','F'}and
                    isinstance(a['age'],int)and not isinstance(a['age'],bool)and a['age']>=20 and
                    len(a['anchor_ids'])==1 and len(a['anchor_codes'])==1 and
                    a['anchor_codes']==a['linked_codes']and a['anchor_codes'][0]in mapping]
        home_counts=Counter(mapping[a['anchor_codes'][0]]for a in candidates)
        population=Counter()
        labels={}
        for r in raw:
            if r['epmndn_nm']=='계':continue
            labels[r['mois_admin_code10']]={'gu':r['sigungu_nm'],'dong':r['epmndn_nm']}
            if r['sex_nm']in{'남자','여자'}and int(r['agegrd_cd'])>=25:
                population[r['mois_admin_code10']]+=int(r['popl_cnt'])
        absent=[{'mois_code10':c,**labels[c],'official_adult_population':population[c],
                 'current_graph_codes8':[g for g,t in mapping.items()if t==c]}
                for c in sorted(targets)if not home_counts[c]]
        results.append({'policy':source['policy'],'reference_month':month,
            'historical_target_dongs':len(targets),'current_graph_dongs':len(graphcodes),
            'all_code_groups_have_direct_MOIS_or_documented_change_relation':True,
            'candidate_anchor_priority_adult_count':len(candidates),
            'raw_code_mismatch_candidates_kept_as_warnings':sum(a['raw_residence_code']!=a['anchor_codes'][0]for a in candidates),
            'target_dongs_with_at_least_one_candidate':len(home_counts),
            'target_dongs_without_candidate':absent,
            'target_adult_population':sum(population.values()),
            'population_in_target_dongs_without_candidate':sum(population[r['mois_code10']]for r in absent),
            'home_candidate_counts_by_historical_group':dict(sorted(home_counts.items())),
            'proposed_graph_current_code8_to_target_mois_code10':dict(sorted(mapping.items())),
            'official_historical_polygon_verified':False,'profile_narrative_identity_verified':False,
            'four_axis_distribution_fitted':False,'new_model_call_gate_pass':False,
            'limitations':['Administrative change relation only. Contemporary2026home POI coordinates have not been checked against historical administrative polygons.',
                'Some target dongs have no present residence-anchored adult candidates; weighting cannot create absent residents.',
                'The original generated income is not empirical income; freeze observed-income assignment separately.',
                'No sample was selected and existing pilots remain unweighted technical small cohorts.']})
    obj={'schema':'mois_candidate_group_support_v1','generator':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':sha(Path(__file__))},
         'source_evidence':evidence,'graph_modified':False,'model_calls':0,'sample_selected':False,
         'existing_frames_and_pilots_modified':False,'results':results}
    out=OUT/'mois_candidate_group_support.json';out.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    dest=CROOT/out.name;shutil.copyfile(out,dest);assert sha(out)==sha(dest)
    print(json.dumps([{k:r[k]for k in['policy','reference_month','historical_target_dongs','candidate_anchor_priority_adult_count','raw_code_mismatch_candidates_kept_as_warnings','target_dongs_with_at_least_one_candidate','population_in_target_dongs_without_candidate','new_model_call_gate_pass']}for r in results],ensure_ascii=False,indent=2))
    print('SHA',sha(out),'C/G PASS')


if __name__=='__main__':main()
