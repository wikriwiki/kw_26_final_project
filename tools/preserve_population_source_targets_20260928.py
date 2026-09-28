"""Copy frozen source/target evidence between this user's C: and G: stores."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
C_BASE = Path('C:/Users/Administrator/Documents/kw26_a100_recovery_20260928')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def copy_pair(pair):
    source, target = pair
    expected = sha(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(target) != expected:
            raise ValueError(f'Existing frozen copy differs: {target.name}')
    else:
        shutil.copyfile(source,target)
    actual = sha(target)
    if actual != expected:
        raise ValueError(f'Copy failed SHA verification: {target.name}')
    return {'source':str(source),'copy':str(target),'sha256':expected,
            'copy_sha256':actual,'bytes':source.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-mois',action='store_true')
    args = parser.parse_args()
    survey = C_BASE/'population_frame_sources'
    pairs = [(survey/f'seoul_survey_raw_{y}.zip',ROOT/'output/population_frame_sources_20260928'/f'seoul_survey_raw_{y}.zip')
             for y in (2019,2021,2024)]
    targets = ROOT/'output/population_matching_20260928'
    names = [p.name for p in sorted(targets.glob('income_distribution_*.json'))]
    names += [p.name for p in sorted(targets.glob('policy_population_targets_*.json'))]
    if args.include_mois:
        names += [p.name for p in sorted(targets.glob('policy_mois_population_targets_*.json'))]
    names += ['prepolicy_income_alignment.json']
    pairs += [(targets/name,C_BASE/'population_matching_targets'/name) for name in names]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(copy_pair,pairs))
    payload = json.dumps({'schema':'population_source_target_preservation_v1',
                          'raw_archives_are_official_public_data':True,'model_calls':0,
                          'source_workbooks_modified':False,'files':results},ensure_ascii=False,indent=2)+'\n'
    manifest_name = 'source_targets_preservation_mois_manifest.json' if args.include_mois else 'source_targets_preservation_manifest.json'
    for p in [targets/manifest_name,C_BASE/'population_matching_targets'/manifest_name]:
        if p.exists() and p.read_text(encoding='utf-8') != payload:
            raise ValueError('Frozen preservation manifest differs')
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(payload,encoding='utf-8')
    print(json.dumps({'files':len(results),'bytes':sum(r['bytes'] for r in results),
                      'all_SHA256_pairs_match':True,'manifest_sha256':sha(targets/manifest_name)},ensure_ascii=False))


if __name__ == '__main__':
    main()
