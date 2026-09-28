"""Preserve exact-period public Seoul DataHub source JSON and derived CSV.

Reads the documented public UI's dataTypeList endpoint. Does not infer historical
vintages from newer annual tables, change graph data, or truncate dong codes.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output' / 'population_frame_sources_20260928'
BACKUP = Path('C:/Users/Administrator/Documents/kw26_a100_recovery_20260928/population_frame_sources')
ENDPOINT = 'https://data.seoul.go.kr/bsp/wgs/dataset/dataTypeList.do'
SOURCE_PAGE = 'https://data.seoul.go.kr/bsp/wgs/dataView/data300View/1.do'
MONTHS = {'P013': '202004', 'P016': '202006', 'P014': '202008',
          'DISTANCING_2020': '202010', 'P012': '202109', 'P010': '202506'}
COLS = ['stats_ym', 'admdong_cd', 'sigungu_nm', 'epmndn_nm', 'sex_nm',
        'agegrd_cd', 'agegrd_nm', 'popl_cnt']


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_data(month: str, page: int = 1, size: int = 50000) -> bytes:
    payload = {'id': '1', 'rowFilterList': [
        {'varColumn': 'stats_ym', 'varSign': 'g', 'varValue': month}],
        'currentPage': page, 'pagePer': size}
    req = Request(ENDPOINT, data=json.dumps(payload).encode('utf-8'),
                  headers={'Content-Type': 'application/json',
                           'User-Agent': 'Mozilla/5.0', 'Referer': SOURCE_PAGE})
    with urlopen(req, timeout=90) as response:
        return response.read()


def validate(rows: list[dict], month: str) -> dict:
    assert rows and all(str(r['stats_ym']) == month for r in rows)
    keys = [(r['admdong_cd'], r['sex_nm'], r['agegrd_cd']) for r in rows]
    assert len(set(keys)) == len(keys), 'Duplicate region/sex/age source keys'
    assert all(set(COLS) <= set(r) for r in rows)
    assert all(isinstance(r['popl_cnt'], int) and r['popl_cnt'] >= 0 for r in rows)
    assert all(len(str(r['admdong_cd'])) == 10 for r in rows)
    sexes = {r['sex_nm'] for r in rows}
    assert sexes == {'계', '남자', '여자'}, sexes
    age_codes = {r['agegrd_cd'] for r in rows}
    assert age_codes == {-1, *range(5, 101, 5), 101}, age_codes
    lookup = {(r['admdong_cd'], r['sex_nm'], r['agegrd_cd']): r['popl_cnt'] for r in rows}
    regions = {r['admdong_cd'] for r in rows}
    region_detail = {r['admdong_cd']: (r['sigungu_nm'], r['epmndn_nm']) for r in rows}
    dong_codes = {code for code, (_, name) in region_detail.items() if name != '계'}
    gu_codes = {code for code, (gu, name) in region_detail.items() if name == '계' and gu != '계'}
    city_codes = {code for code, (gu, name) in region_detail.items() if name == '계' and gu == '계'}
    assert len(city_codes) == 1 and len(gu_codes) == 25
    city = next(iter(city_codes))
    for code in regions:
        assert len([k for k in keys if k[0] == code]) == 66
        for age in age_codes:
            assert lookup[(code, '계', age)] == lookup[(code, '남자', age)] + lookup[(code, '여자', age)]
        for sex in sexes:
            assert lookup[(code, sex, -1)] == sum(lookup[(code, sex, a)] for a in age_codes if a != -1)
    for age in age_codes:
        for sex in sexes:
            assert lookup[(city, sex, age)] == sum(lookup[(code, sex, age)] for code in dong_codes)
            assert lookup[(city, sex, age)] == sum(lookup[(code, sex, age)] for code in gu_codes)
    return {'rows': len(rows), 'unique_source_keys': len(set(keys)),
            'administrative_dongs': len(dong_codes), 'gu_aggregates': len(gu_codes),
            'city_aggregates': len(city_codes), 'sex_categories': sorted(sexes),
            'age_categories': len(age_codes), 'seoul_registered_nationals': lookup[(city, '계', -1)],
            'gender_and_age_totals_reconcile': True,
            'dong_gu_city_totals_reconcile': True,
            'native_administrative_code_length': 10,
            'actor_8_digit_crosswalk_applied': False,
            'period_key_validated': month}


def one(spec: tuple[str, str]) -> dict:
    policy, month = spec
    t = time.monotonic()
    raw_path = OUTPUT / f'official_resident_sex_age_dong_{month}.source.json'
    if month == '202004' and not raw_path.exists():
        old = ROOT / 'tmp/population2020_endpoint_probe/1_202004_big.json'
        if old.exists(): shutil.copyfile(old, raw_path)
    if not raw_path.exists(): raw_path.write_bytes(request_data(month))
    obj = json.loads(raw_path.read_bytes())
    rows = obj['dataTypeList']
    audit = validate(rows, month)
    # The response count is a UI preview cap, not a trustworthy full row count.
    audit['ui_response_count_field'] = obj.get('dataTypeListCount')
    second_raw = request_data(month, page=2)
    second_path = OUTPUT / f'official_resident_sex_age_dong_{month}.end_page.json'
    second_path.write_bytes(second_raw)
    assert json.loads(second_raw)['dataTypeList'] == [], 'Another full page exists'
    audit['next_50000_row_page_empty'] = True
    # Source-order rows and original codes are kept; aggregates remain explicitly
    # present in raw/CSV and must be excluded by a downstream frame builder.
    csv_path = OUTPUT / f'official_resident_sex_age_dong_{month}.from_json.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in COLS} for row in rows)
    copied = []
    for path in [raw_path, second_path, csv_path]:
        dest = BACKUP / path.name
        shutil.copyfile(path, dest)
        assert sha(path) == sha(dest)
        copied.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                       'bytes': path.stat().st_size, 'c_copy': str(dest),
                       'c_sha256': sha(dest)})
    print(policy, month, json.dumps(audit, ensure_ascii=False),
          'seconds', round(time.monotonic() - t, 1), flush=True)
    return {'policy': policy, 'period': month,
            'period_semantics': 'Actual source stats_ym; policy preceding month, no year shifting',
            'source_page': SOURCE_PAGE, 'endpoint': ENDPOINT,
            'source_request': {'id': '1', 'rowFilterList': [
                {'varColumn': 'stats_ym', 'varSign': 'g', 'varValue': month}],
                'currentPage': 1, 'pagePer': 50000},
            'audit': audit, 'files': copied,
            'csv_is_derived_from_preserved_source_json': True}


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    BACKUP.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(one, MONTHS.items()))
    manifest = {'schema': 'official_population_month_sources_v1',
                'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'generator': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha(Path(__file__))},
                'source_dataset_id': 1, 'source_title': '주민등록인구 (내국인 성별/연령별/월별/동별) 통계',
                'purpose': 'Future representative sample frames; completed pilot results remain unchanged',
                'population': 'Registered Korean nationals. Foreign residents excluded; aggregates not sampling units.',
                'age_resolution': 'Five-year bands, plus 100+, not exact individual ages',
                'code_boundary_gate': '10-digit native official administrative codes. An explicit vintage-specific actor-code crosswalk is still required.',
                'income_gate': 'This source contains no income; Seoul Survey source and household-to-person estimand audit required separately.',
                'results': results}
    path = OUTPUT / 'official_population_months_manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    dest = BACKUP / path.name
    shutil.copyfile(path, dest)
    assert sha(path) == sha(dest)
    print('MANIFEST', str(path), sha(path), 'C/G verified', flush=True)


if __name__ == '__main__': main()
