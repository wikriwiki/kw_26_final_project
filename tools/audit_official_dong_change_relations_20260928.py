"""Preserve official change evidence; distinguish code, group and geometry gates."""
from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/population_frame_sources_20260928'
SOURCES = OUT / 'official_code_sources'
CROOT = Path('C:/Users/Administrator/Documents/kw26_a100_recovery_20260928/population_frame_sources')
PUBLIC = [
    ('gangdong_sangil2_split.html', 'https://www.gangdong.go.kr/web/newportal/press/10794', ['강일동에서 분동', '고덕로']),
    ('gangdong_sangil1_history.html', 'https://www.gangdong.go.kr/web/dongrenew/contents/sangil_010_010', ['상일제1동', '명칭 변경']),
    ('gangnam_ilwon2_rename.html', 'https://www.gangnam.go.kr/board/B_000031/1072853/view.do?mid=ID01_0313', ['일원', '개포', '명칭 변경']),
    ('gangbuk_ordinance_2019.html', 'https://www.law.go.kr/LSW/ordinInfoP.do?ordinSeq=1683451', []),
]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(record):
    name, url, required = record
    path = SOURCES/name
    if not path.exists():
        request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(request,timeout=25)as response:
            path.write_bytes(response.read())
    raw = path.read_bytes()
    text = html.unescape(re.sub(r'<[^>]+>', ' ', raw.decode('utf-8',errors='replace')))
    text = re.sub(r'\s+', ' ', text)
    for word in required:
        if word not in text:
            raise ValueError(f'Official source contents missing expected phrase: {name}: {word}')
    return {'path':str(path.relative_to(ROOT)), 'url':url, 'sha256':sha(path),
            'bytes':len(raw), 'required_content_checks_pass':True}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    SOURCES.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=4)as executor:
        official=list(executor.map(fetch,PUBLIC))
    frozen_paths=[OUT/'admin_code_crosswalk_audit.json',OUT/'admin_code_population_support.json',
                  OUT/'graph_full_population_code_projection_20260928.json',
                  SOURCES/'mois_notice_118483.html',SOURCES/'KIKcd_H.20250701(말소코드포함).xlsx',
                  SOURCES/'KIKmix.20250701(말소코드포함).xlsx',
                  ROOT/'scripts/sim/population_profile.py', ROOT/'scripts/sim/dawn_context.py',
                  ROOT/'scripts/sim/run_simulation.py']
    graph=json.loads(frozen_paths[2].read_text(encoding='utf-8'))
    mismatches=[a for a in graph['agents']if len(a['anchor_codes'])==1 and
                a['raw_residence_code']!=a['anchor_codes'][0]]
    assert len(mismatches)==444
    assert sum((a['raw_residence_code']or'')[:5]!=a['anchor_codes'][0][:5]for a in mismatches)==0
    def relation(label,effective,old,new,source,meaning,limits):
        return {'label':label,'effective_date':effective,'prior_mois_codes10':old,
                'current_graph_codes8':new,'official_change_type':meaning,
                'evidence':source,'source_namespace_alias_verified':False,
                'historical_polygon_identity_verified':False,
                'limits':limits,'whole_city_sampling_gate_pass':False}
    relations=[
        relation('용신동 분동','2025-07-01',['1123053600'],['11230515','11230533'],
                 ['official_code_sources/mois_notice_118483.html','official_code_sources/KIKcd_H.20250701(말소코드포함).xlsx'],
                 'Official MOIS notice explicitly names abolition of 용신동 and creation of 신설동/용두동 by split. Aggregation of successor dongs is the documented relation; do not split old population arbitrarily.',
                 'Legal code identity alone cannot identify exact sub-dong boundaries. Old population can target the aggregate after full vintage relation audit, not invented daughter shares.'),
        relation('상일동 명칭 변경','2021-07-01',['1174052000'],['11740525'],
                 ['official_code_sources/gangdong_sangil1_history.html','official_code_sources/KIKcd_H.20250701(말소코드포함).xlsx'],
                 'Gangdong official history documents former 상일동 renamed 상일제1동.',
                 'Never aggregate 상일1동 and 상일2동 to former 상일동: 상일2동 was split from 강일동. Source jurisdiction/polygon audit still required.'),
        relation('강일동에서 상일2동 분리','2021-07-01',['1174051500'],['11740515','11740526'],
                 ['official_code_sources/gangdong_sangil2_split.html','official_code_sources/KIKcd_H.20250701(말소코드포함).xlsx'],
                 'Gangdong official press release documents 강일동 split into current 강일동 north of 고덕로 and 상일2동 south of 고덕로. Same-code 강일동 therefore changed territory.',
                 'Before July 2021, both current successor territories must be treated as prior 강일동 aggregate. Code identity on 강일동 alone is insufficient. Exact polygon continuity is not established here.'),
        relation('일원2동 명칭 변경','2022-12-23',['1168074000'],['11680675'],
                 ['official_code_sources/gangnam_ilwon2_rename.html','official_code_sources/KIKcd_H.20250701(말소코드포함).xlsx'],
                 'Gangnam official press release documents 일원2동 renamed 개포3동, with legal addresses unchanged; MOIS lifecycle confirms code dates.',
                 'This documents prior active 일원2동 to new active 개포3동. It does not explain why 2025 DataHub uses inactive old 개포제3동 code1168068000. That native statistical alias remains unresolved.'),
    ]
    obj={'schema':'official_dong_change_relations_audit_v1','created_at':datetime.now(timezone.utc).isoformat(),
         'generator':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':sha(Path(__file__))},
         'source_evidence':official+[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size}for p in frozen_paths],
         'model_calls':0,'graph_modified':False,'existing_pilot_results_changed':False,
         'overall_whole_city_sampling_gate_pass':False,
         'documented_change_relations':relations,
         'inactive_native_aliases_unresolved':[
            {'native_code10':'1130563000','label':'수유3동','mois_active_code10':'1130563500',
             'status':'not_verified_KOSIS_native_to_MOIS_correspondence',
             'evidence_note':'MOIS lifecycle and official ordinance support the administrative name change, but neither states DataHub/KOSIS native code correspondence. Name uniqueness is not accepted as alias proof.'},
            {'native_code10':'1168068000','label':'개포3동','mois_active_code10':'1168067500',
             'status':'not_verified_KOSIS_native_to_MOIS_correspondence',
             'evidence_note':'Native code belongs to former 개포제3동 abolished in2009; contemporary source label alone cannot prove mapping to current 개포3동 created2022.'}],
         'anchor_priority_assessment':{'raw_anchor_mismatches':444,'different_gu_code_prefix':0,
             'all_mismatch_agents_must_be_excluded':False,
             'condition':'Require exactly one LIVES_AT residence and same home POI.dong_code/linked Dong.code; freeze actual anchor code in profile and validate sex/exact age/home identity before any model call. Preserve original aid/property as audit warnings.',
             'runtime_paths_use_actual_anchor':['dawn_context.PERSONA_CYPHER','run_simulation.run_day profile graph preflight','population_profile.verify_graph_projection','population_profile.bind_persona'],
             'still_exclude_missing_home_or_exact_age':True,'generated_income_not_empirical':True,
             'narrative_geographic_consistency_not_audited':True},
         'remaining_requirements':['Official KOSIS/DataHub-native code correspondence for stale aliases, not name matching',
             'Full historical jurisdiction continuity, including changes retaining the same code',
             'Candidate support for target dongs/cells with no current home anchors',
             'Observed income assignment and all four-axis frozen distribution fit',
             'Profile narrative geographic consistency; current2026POI cannot be assumed historical geography']}
    out=OUT/'official_dong_change_relations_audit.json'
    out.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    for source in official:
        p=ROOT/source['path'];dest=CROOT/'official_code_sources'/p.name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);assert sha(p)==sha(dest)
    dest=CROOT/out.name;shutil.copyfile(out,dest);assert sha(out)==sha(dest)
    print(json.dumps({'relations':len(relations),'aliases_unresolved':2,'whole_city_gate':False,
                      'sha256':sha(out),'C_G_verified':True},ensure_ascii=False))


if __name__=='__main__':main()
