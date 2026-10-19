"""Preserve public survey originals for pre-policy income covariates.

Download sequences are taken from the official OA-15564 public download page.
Never overwrite a source archive or infer a year from a newer income table.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile

SOURCE = Path('C:/Users/Administrator/Documents/kw26_a100_recovery_20260928/population_frame_sources')
SEQUENCES = {2019: '15', 2024: '26'}
PAGE = 'https://data.seoul.go.kr/dataList/OA-15564/F/1/datasetView.do'
ENDPOINT = 'https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover(year, seq):
    path = SOURCE / f'seoul_survey_raw_{year}.zip'
    if not path.exists():
        request = Request(ENDPOINT, data=urlencode({'infId':'OA-15564', 'infSeq':'1',
                          'seqNo':'', 'seq':seq}).encode('ascii'),
                          headers={'User-Agent':'Mozilla/5.0', 'Referer':PAGE})
        partial = path.with_suffix('.partial')
        with urlopen(request, timeout=90) as response, partial.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        with zipfile.ZipFile(partial) as archive:
            assert archive.testzip() is None
        partial.rename(path)
    entries = []
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        for item in archive.infolist():
            name = item.filename
            if not item.flag_bits & 0x800:
                try:
                    name = name.encode('cp437').decode('cp949')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            entries.append({'original_member_name':name, 'bytes':item.file_size})
            kind = 'households' if '가구주' in name else 'members' if '가구원' in name else None
            if kind and name.endswith('.xlsx'):
                extracted = SOURCE / 'extracted' / f'{year}_{kind}_combined.xlsx'
                content = archive.read(item)
                extracted.parent.mkdir(parents=True, exist_ok=True)
                if extracted.exists() and extracted.read_bytes() != content:
                    raise ValueError('Extracted official workbook would change')
                extracted.write_bytes(content)
            if (year == 2019 and '조사표_가구조사.pdf' in name or
                    year == 2024 and '조사(가구주조사).pdf' in name):
                extracted = SOURCE / 'extracted' / f'{year}_household_questionnaire.pdf'
                content = archive.read(item)
                if extracted.exists() and extracted.read_bytes() != content:
                    raise ValueError('Extracted official questionnaire would change')
                extracted.write_bytes(content)
    result = {'year':year, 'source_url':PAGE, 'download_sequence':seq,
              'path':str(path), 'sha256':sha(path), 'bytes':path.stat().st_size,
              'zip_crc_pass':True, 'members':entries}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    SOURCE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda item:recover(*item), SEQUENCES.items()))
    target = SOURCE / 'prepolicy_income_sources_manifest.json'
    data = json.dumps({'schema':'prepolicy_income_source_archives_v1', 'sources':results},
                      ensure_ascii=False, indent=2) + '\n'
    if target.exists() and target.read_text(encoding='utf-8') != data:
        raise ValueError('Source manifest is already frozen with different contents')
    target.write_text(data, encoding='utf-8')


if __name__ == '__main__':
    main()
