"""Quantify verified population/agent support; does not select a sample."""
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/population_frame_sources_20260928'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    audit_path = OUT/'admin_code_crosswalk_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    graph_path = OUT/'graph_full_population_code_projection_20260928.json'
    graph = json.loads(graph_path.read_text(encoding='utf-8'))
    result = []
    evidence = [{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [audit_path,graph_path]]
    for case in audit['policy_source_crosswalks']:
        month=case['reference_month']
        path=OUT/f'official_resident_sex_age_dong_{month}.source.json'
        evidence.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
        raw=json.loads(path.read_text(encoding='utf-8'))['dataTypeList']
        populations=Counter()
        for r in raw:
            if r['epmndn_nm']!='계' and r['sex_nm'] in {'남자','여자'} and r['agegrd_cd']>=25:
                populations[r['admdong_cd']]+=r['popl_cnt']
        accepted={r['native_population_code10']:r['candidate_graph_code8'] for r in case['rows']
                  if r['status']=='official_code_identity_same_current_graph'}
        supported=sum(populations[c] for c in accepted)
        total=sum(populations.values())
        codes=set(accepted.values())
        candidates=[r for r in graph['agents'] if r['sex'] in {'M','F'} and
                    isinstance(r['age'],int) and not isinstance(r['age'],bool) and r['age']>=20 and
                    len(r['anchor_ids'])==1 and r['anchor_codes']==[r['raw_residence_code']] and
                    r['linked_codes']==[r['raw_residence_code']] and r['raw_residence_code'] in codes]
        strict_home_counts=Counter(r['raw_residence_code']for r in candidates)
        anchor_candidates=[r for r in graph['agents'] if r['sex'] in {'M','F'} and
                    isinstance(r['age'],int) and not isinstance(r['age'],bool) and r['age']>=20 and
                    len(r['anchor_ids'])==1 and len(r['anchor_codes'])==1 and
                    r['anchor_codes']==r['linked_codes'] and r['anchor_codes'][0] in codes]
        anchor_home_counts=Counter(r['anchor_codes'][0]for r in anchor_candidates)
        unsupported_candidate_dongs=[{'native_code10':code10,'graph_code8':code8,
                                     'official_adult20plus_population':populations[code10]}
                                    for code10,code8 in accepted.items() if not strict_home_counts[code8]]
        missing=[{'native_code10':r['native_population_code10'],'dong':r['population_name'],
                  'status':r['status'],'official_adult20plus_population':populations[r['native_population_code10']]}
                 for r in case['rows']if r['status']!='official_code_identity_same_current_graph']
        row={'policy':case['policy'],'reference_month':month,'age_scope':'20세 이상',
             'native_source_adult_population':total,'verified_common_code_adult_population':supported,
             'verified_common_code_population_share':supported/total,
             'unmapped_source_adult_population':total-supported,
             'candidate_agents_passing_exact_age_sex_home_identity_and_common_code':len(candidates),
             'candidate_home_dongs':len({r['raw_residence_code']for r in candidates}),
             'verified_code_dongs_without_strict_candidate':unsupported_candidate_dongs,
             'population_in_verified_code_dongs_without_strict_candidate':sum(r['official_adult20plus_population']for r in unsupported_candidate_dongs),
             'anchor_priority_candidate_count_before_profile_validation':len(anchor_candidates),
             'anchor_priority_home_dongs_before_profile_validation':len(anchor_home_counts),
             'anchor_priority_needs_frozen_profile_identity_validation':True,
             'unresolved_source_dongs':missing,
             'population_frame_representative_or_sampling_ready':False,
             'scope_note':'Code-identity support count only, not historical boundary verification. No sample selected. A same-code district may have changed jurisdiction. Native inactive codes, boundary harmonization, home profile identity, observed income and distribution fitting remain gates.'}
        result.append(row)
    obj={'schema':'admin_code_population_support_v1','source_evidence':evidence,
         'generator':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':sha(Path(__file__))},
         'graph_modified':False,'model_calls':0,'sample_selected':False,'results':result}
    path=OUT/'admin_code_population_support.json';path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    dest=Path('C:/Users/Administrator/Documents/kw26_a100_recovery_20260928/population_frame_sources')/path.name
    shutil.copyfile(path,dest);assert sha(path)==sha(dest)
    print(json.dumps(result,ensure_ascii=False,indent=2));print('SHA',sha(path),'C/G PASS')


if __name__=='__main__':main()
