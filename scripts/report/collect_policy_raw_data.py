"""실험 중인 정책의 **실제 문서**를 정책마다 한 폴더에 모은다.

JSON 메타데이터가 아니라 PDF·HWP·표 같은 원문이다. 넘겨받는 사람이 정책을 직접
읽고(정책원문), 결과가 어땠는지 확인하고(정답지), 우리가 무엇으로 채점하는지
볼 수 있어야 한다(평가항목).

    data/policy_raw_data/<정책명>_<PXXX>/
        평가항목.md          채점 항목 · 실측값 · 시뮬 값 — 이 폴더의 요약
        정책원문_*           제도의 규칙 원문 (정부 보도자료·시행방안)
        정답지_*             효과를 잰 연구 원문 — 우리가 맞히려는 값
        환경자료/            시뮬이 배경으로 쓰는 자료 (있는 경우)
        MANIFEST.md         어느 파일이 어디서 왔는지, sha256 과 크기

문서는 두 갈래다.

    사본      저장소 안의 파일을 복사한 것. `--check` 가 원본과 sha256 을 맞춰 본다
    내려받음  공공기관에서 직접 받은 것. 저장소에 원본이 없으므로 출처 URL 과
              sha256 을 `_sources.json` 에 적어 두고 그 값으로 검사한다

**없는 것은 없다고 적는다.** 홀드아웃(P015)은 정답지가 봉인이라 넣지 않고,
사적모임 제한의 정답지는 유료 DB(KISS)에만 있어 DOI 만 적는다.

    python scripts/report/collect_policy_raw_data.py
    python scripts/report/collect_policy_raw_data.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/policy_raw_data'
SCORING = ROOT / 'data/experiments/scoring_table.json'
COVID = ROOT / 'data/experiments/covid_support_2021'
SOURCES = OUT / '_sources.json'

# (저장소 안의 원본 경로, 폴더 안에서의 이름, 한 줄 설명)
P010_DOCS = [
    ('한국은행_이슈노트.pdf', '정답지_한국은행_이슈노트_2026-13.pdf',
     '한국은행 이슈노트 2026-13 — 소비쿠폰의 한계소비성향을 잰 글. 우리가 맞히려는 정답지'),
    ('data/neo4j_load/policies/P010.json', '시뮬정의_P010.json',
     '시뮬레이터에 실린 정책 정의. **동결이다** — 이 정책의 프롬프트도 JSON 도 고치지 않는다'),
]

P012_DOCS = [
    ('docs/references/1._상생소비지원금_시행방안(최종)_배포용.pdf',
     '정책원문_상생소비지원금_시행방안_배포용.pdf',
     '기획재정부 시행방안 배포본 — 제도의 규칙 원문'),
    ('docs/references/상생소비지원금효과분석.pdf',
     '정답지_KDI_상생소비지원금_효과분석_2022.pdf',
     '기획재정부·KDI 효과분석(2022.9) — 우리가 맞히려는 정답지'),
    ('output/validation_reference/sources/kdi_cashback_2022.txt',
     '정답지_KDI_효과분석_추출텍스트.txt',
     '위 PDF 에서 뽑은 본문 — 검색용'),
    ('data/neo4j_load/policies/P012.json', '시뮬정의_P012.json',
     '시뮬레이터에 실린 정책 정의. 위 시행방안을 이 형식으로 옮긴 것'),
]

P013_DOCS = [('data/neo4j_load/policies/P013.json', '시뮬정의_P013.json',
              '시뮬레이터에 실린 정책 정의 — 전 국민 지급 지갑')]
P014_DOCS = [('data/neo4j_load/policies/P014.json', '시뮬정의_P014.json',
              '시뮬레이터에 실린 정책 정의 — 할인 구매 상품권')]
P015_DOCS = [('data/neo4j_load/policies/P015.json', '시뮬정의_P015.json',
              '시뮬레이터에 실린 정책 정의 — 업종형 쿠폰. **홀드아웃**이다')]

DISTANCING_DOCS = [
    ('data/experiments/covid_support_2021/distancing_schedule.json',
     '거리두기_단계표_2020-2021.json',
     '2020-05~2021-01 수도권 단계 전환표. 채점 구간(2020-11-24 2단계)이 여기 들어 있다'),
]

DISTANCING_ENV = [
    ('distancing_0823.html', '2021 거리두기 재연장 공지'),
    ('distancing_0906.html', '2021-09-06 4주 연장 (보건복지부)'),
    ('distancing_1004.html', '2021-10-04 연장 (경기도)'),
    ('distancing_1018.html', '2021-10-18 유지 공지'),
    ('reopening_1101.html', '2021-11-01 단계적 일상회복'),
    ('restrictions_1206.html', '2021-12-06 특별방역대책'),
    ('restrictions_1218.html', '2021-12-18 거리두기 강화'),
    ('seoul_district_cases.xlsx', '서울시 자치구별 확진자·사망자'),
    ('seoul_city_cases.csv', '서울시 확진자 발생 현황'),
]

# 내려받은 문서 — (폴더 안 이름, 출처 URL, 설명).
FETCHED = {
    '민생회복소비쿠폰_P010': [
        ('정책원문_민생회복소비쿠폰_지급시작_행안부_20250705.pdf',
         'https://www.mois.go.kr/frt/bbs/type010/commonSelectBoardArticle.do'
         '?bbsId=BBSMSTR_000000000008&nttId=118705',
         '행정안전부 2025-07-05 보도자료 — 지급 대상·금액·사용처의 규칙 원문'),
        ('집행결과_민생회복소비쿠폰_최종집계_20251205.pdf',
         'https://www.korea.kr/briefing/pressReleaseView.do?newsId=156733265',
         '정책브리핑 2025-12-05 — 실제로 얼마가 누구에게 갔는지의 집행 결과'),
    ],
    '긴급재난지원금_P013': [
        ('정책원문_긴급재난지원금_신청및지급방안_행안부_20200429.hwp',
         'https://www.mois.go.kr/frt/bbs/type010/commonSelectBoardArticle.do'
         '?bbsId=BBSMSTR_000000000008&nttId=76947',
         '행정안전부 2020-04-29 「긴급재난지원금」 신청 및 지급 방안 — 규칙 원문'),
        ('정답지_KDI_FOCUS_1차긴급재난지원금_효과와시사점_2020.pdf',
         'https://www.kdi.re.kr/research/focusView?pub_no=16851',
         'KDI FOCUS 제281호(2020-06), 김미루·오윤해 — 우리가 맞히려는 정답지'),
    ],
    '지역사랑상품권_P014': [
        ('정책원문_지역사랑상품권_발행지원사업_종합지침_20210122.pdf',
         'https://www.mois.go.kr/frt/sub/a06/b07/localVoucher/screen.do',
         '행정안전부 발행지원 사업 종합지침 — 할인율·가맹점 범위의 규칙 원문'),
        ('정답지_조세재정연구원_지역화폐가_지역경제에_미친_영향_2020.pdf',
         'https://www.kipf.re.kr/uloads/kiPublish/202012/FILE_202102020125031083.pdf',
         '송경호·이환웅(2020) 한국조세재정연구원 — 우리가 맞히려는 정답지'),
    ],
    '업종형소비쿠폰_P015': [
        ('정책원문_2020년_하반기_경제정책방향.pdf',
         'https://www.korea.kr/briefing/pressReleaseView.do?newsId=156393266',
         '8대 소비쿠폰이 실린 본문(2020-06-01). 업종별 할인율·한도의 규칙 원문'),
        ('정책원문_하반기경제정책방향_보도자료_20200601.hwp',
         'https://www.korea.kr/briefing/pressReleaseView.do?newsId=156393266',
         '위 대책의 보도자료 — 요약본'),
    ],
    '사회적거리두기_DISTANCING2020': [
        ('정답지_서울연구원_코로나19_서울_경제적손실_2021.pdf',
         'https://www.si.re.kr/node/64651',
         '서울연구원 정책리포트 322호(2021.4), 신한카드 서울 패널 — '
         '한식 -14.1% 등 우리가 맞히려는 정답지'),
        ('정책원문_중대본회의_보도자료_20201127.hwp',
         'https://www.korea.kr/briefing/pressReleaseView.do?newsId=156423736',
         '중앙재난안전대책본부 2020-11-27 — 채점 구간(11/24~25)에 시행 중이던 '
         '수도권 2단계 조치를 담은 공식 문서'),
    ],
    '사적모임제한_GATHERING2020': [
        ('정책원문_중대본회의_보도자료_20201127.hwp',
         'https://www.korea.kr/briefing/pressReleaseView.do?newsId=156423736',
         '중앙재난안전대책본부 2020-11-27 — 모임·행사 인원 제한이 들어 있는 문서'),
    ],
}

PLAN = [
    {'folder': '민생회복소비쿠폰_P010', 'scoring': 'P010',
     'rounds': 'EXP-001 · P010 라운드 (프롬프트 동결)',
     'docs': P010_DOCS, 'env': [], 'missing': [],
     'note': '2025년 전 국민 소비쿠폰. 지갑이 생기므로 총액이 늘어야 하는 기전이다. '
             '**이 정책의 프롬프트와 JSON 은 동결이다.**'},
    {'folder': '상생소비지원금_P012', 'scoring': 'P012',
     'rounds': 'v30(중단) · v43(취소) · v44(완료)',
     'docs': P012_DOCS, 'env': [], 'missing': [],
     'note': '2021년 카드 실적 캐시백. 2분기 월평균 대비 3% 초과분의 10%를 다음 달에 환급.'},
    {'folder': '긴급재난지원금_P013', 'scoring': 'EMERGENCY_2020',
     'rounds': 'stage3 (2026-09-16)',
     'docs': P013_DOCS, 'env': [], 'missing': [],
     'note': '2020년 1차 전 국민 지급. 새 돈이 지갑에 들어오는 기전이라 총액이 늘어야 한다.'},
    {'folder': '지역사랑상품권_P014', 'scoring': 'LOCAL_VOUCHER',
     'rounds': 'stage3 · stage5 (2026-09-16~18)',
     'docs': P014_DOCS, 'env': [], 'missing': [],
     'note': '할인 구매 상품권. 새 돈이 아니라 가격 할인이므로 총액이 아니라 '
             '**어디서 쓰는가**가 움직여야 한다.'},
    {'folder': '업종형소비쿠폰_P015', 'scoring': 'SECTOR_VOUCHER_2020',
     'rounds': 'baseline(2026-09-16) · policy(2026-09-17)',
     'docs': P015_DOCS, 'env': [],
     'missing': [('정답지', '**봉인이다.** 지표는 돌리기 전에 정책 설계만 보고 적었고 '
                  '정답값은 채점 1회에만 개봉한다. 이 폴더에 정답지를 넣지 않는 것이 설계다')],
     'note': '2020년 8대 소비쿠폰(농수산·외식·숙박·여행·체육 등). **홀드아웃**이다.'},
    {'folder': '사회적거리두기_DISTANCING2020', 'scoring': 'DISTANCING_2020',
     'rounds': 'v50 — 지금 돌고 있다 (v5 대 v45 정답지 평가)',
     'docs': DISTANCING_DOCS, 'env': DISTANCING_ENV,
     'missing': [('2020-11-22 격상 발표 원문',
                  '수도권 2단계 격상을 발표한 그날의 중대본 보도자료. '
                  '11-27 회의 문서로 대신했다 — 같은 조치가 시행 중이던 문서다')],
     'note': '2020년 11월 수도권 2단계. 정책 JSON 이 아니라 사회 배경이고 '
             'environment(covid_2021) 가 규제를 실어 온다.'},
    {'folder': '사적모임제한_GATHERING2020', 'scoring': 'GATHERING_2020',
     'rounds': '아직 돌리지 않았다 — 지표만 등록돼 있다',
     'docs': [], 'env': [],
     'missing': [('정답지 원문',
                  '김다미·김용규(2023) 정보통신정책연구 30(3) 35-54, '
                  'DOI 10.37793/ITPR.30.3.2. 유료 DB(KISS)에만 있어 내려받지 못했다 — '
                  'https://kiss.kstudy.com/Detail/Ar?key=4045482')],
     'note': '인원 제한. **방어선 정책이다** — 논문이 총지출·이동에 유의한 반응을 '
             '찾지 못했으므로 우리 시뮬도 움직이지 않아야 맞는 것이다.'},
]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def real_pdf(p):
    """G: 드라이브 껍데기(메타만 있고 내용 없음)가 아닌지 본다."""
    p = Path(p)
    if p.suffix.lower() != '.pdf':
        return True
    size = p.stat().st_size
    with io.open(p, 'rb') as fh:
        head = fh.read(8)
        fh.seek(max(0, size - 2048))
        tail = fh.read()
    return head.startswith(b'%PDF') and b'%%EOF' in tail


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    io.open(path, 'w', encoding='utf-8', newline='\n').write(text)


def indicators_md(scoring, key):
    b = scoring.get(key) or {}
    lines = []
    if b.get('answer_key'):
        lines += ['**정답지 출처** — %s' % b['answer_key'], '']
    inds = b.get('indicators') or []
    if inds:
        lines += ['| 지표 | 기대 | 무엇을 재는가 |', '|---|---|---|']
        for i in inds:
            lines.append('| `%s` | %s | %s |' % (
                i.get('id', '?'), i.get('expect', '?'),
                str(i.get('desc', '')).replace('|', '·')))
        lines.append('')
    return lines


def split_desc(desc):
    """지표 설명은 '무엇 (실측 X)' 꼴이다. 재는 것과 정답지 값을 갈라 적는다."""
    d = str(desc or '')
    m = re.search(r'\(실측\s*([^)]*)\)', d)
    if not m:
        return d.strip(' -'), '—'
    return (d[:m.start()] + d[m.end():]).strip(' -—'), m.group(1).strip()


def results_md(scoring, key):
    """채점표의 result_* 블록을 사람이 읽게 편다.

    표의 뼈대는 **지표 목록**이다 — 결과 블록만 돌면 못 잰 지표가 통째로 사라진다.
    지표마다 담긴 것이 다르다(비율만·금액만·이유만). 가진 것만 적고 없는 것은 '—'.
    """
    b = scoring.get(key) or {}
    got = {k: v for k, v in b.items() if k.startswith('result')}
    inds = [i for i in (b.get('indicators') or []) if i.get('id')]
    if not got:
        return ['채점표(`data/experiments/scoring_table.json`)에 결과 블록이 없다. '
                '라운드를 돌지 않았거나, 결과가 다른 곳(`experiments/`)에만 있다는 뜻이다.',
                '']
    out = []
    for name, block in got.items():
        if not isinstance(block, dict):
            continue
        out.append('### `%s`' % name)
        if block.get('design'):
            out += ['', str(block['design']), '']
        if block.get('window'):
            out += ['창 `%s`' % block['window'], '']
        rows, notes = [], []
        for i in inds:
            ind = i['id']
            val = block.get(ind)
            what, truth = split_desc(i.get('desc'))
            if not isinstance(val, dict):
                rows.append('| `%s` | %s | — | — | %s | 안 잼 |' % (ind, what, truth))
                continue
            rows.append('| `%s` | %s | %s | %s | %s | %s |' % (
                ind, what, _sim(val), _ci(val.get('ci')),
                str(val.get('실측', truth)), _hit(val.get('hit'))))
            if val.get('note'):
                notes.append('- `%s` — %s' % (ind, str(val['note'])))
        if rows:
            out += ['| 지표 | 무엇을 재는가 | 시뮬 | 구간 | 실측(정답지) | 판정 |',
                    '|---|---|---|---|---|---|'] + rows + ['']
        if notes:
            out += notes + ['']
        did = block.get('did')
        if isinstance(did, dict):
            out += ['**순효과(DID)** — 정책 %s · 기준선 %s · 순수 효과 **%s**' % (
                _pct(did.get('policy_effect_pct')), _pct(did.get('baseline_pct')),
                _pct(did.get('net_pp'), '%p')), '']
            if did.get('note'):
                out += ['> ' + str(did['note']), '']
                sup = SUPERSEDED.get(key)
                if sup and sup[0] in str(did['note']):
                    out += ['> ', '> ⚠️ ' + sup[1], '']
        if block.get('drift_removed'):
            out += ['**드리프트** — ' + str(block['drift_removed']), '']
        if block.get('판정'):
            out += ['**판정** — ' + str(block['판정']).replace(chr(10), ' '), '']
        out.append('')
    return out


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _pct(v, unit='%'):
    return (format(v, '+.1f') + unit) if _num(v) else '—'


# 채점표가 보관한 해석 중, 이후 라운드가 무효로 만든 것. 그대로 실으면 안 된다.
SUPERSEDED = {
    'P012': ('크기가 절반인 것은 우리 정책 구간이 2일이고 실측은 한 달이기 때문',
             '**이 설명은 2026-09-22 에 성립하지 않게 됐다.** v44 문턱 탐침에서 모델이 '
             '문턱까지 남은 거리에 방향 없이 반응하는 것을 확인했다 — 누적 기간을 늘려도 '
             '크기가 커질 이유가 없다. `experiments/v44/result_v44.md` 참조.'),
}


def _sim(val):
    if _num(val.get('pct')):
        return _pct(val['pct'])
    if _num(val.get('mean')):
        return format(int(val['mean']), '+,d') + '원'
    return '—'


def _ci(ci):
    if isinstance(ci, list) and len(ci) == 2 and all(_num(x) for x in ci):
        return '[%s, %s]' % (format(int(ci[0]), '+,d'), format(int(ci[1]), '+,d'))
    return '—'


def _hit(v):
    return {True: '맞음', False: '틀림'}.get(v, '—')


def build(check_only=False):
    scoring = json.loads(SCORING.read_text(encoding='utf-8'))
    # 내려받은 문서는 저장소에 원본이 없다. 무엇을 어디서 받았고 그때 sha256 이
    # 무엇이었는지를 여기 적어 두어야 나중에 바뀌었는지 알 수 있다.
    sources = json.loads(SOURCES.read_text(encoding='utf-8')) if SOURCES.exists() else {}
    problems, report = [], []
    for e in PLAN:
        out = OUT / e['folder']
        rows = []

        for src_rel, name, desc in e['docs']:
            src = ROOT / src_rel
            if not src.exists():
                problems.append('%s: 원본 없음 %s' % (e['folder'], src_rel))
                continue
            if not real_pdf(src):
                problems.append('%s: %s 가 껍데기다 (내용 없음)' % (e['folder'], src_rel))
                continue
            digest, size = sha256(src), src.stat().st_size
            dst = out / name
            if check_only:
                if not dst.exists():
                    problems.append('%s: 사본 없음 %s' % (e['folder'], name))
                elif sha256(dst) != digest:
                    problems.append('%s: 사본이 원본과 다르다 %s' % (e['folder'], name))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            rows.append((name, src_rel, digest, size, desc))

        for name, url, desc in FETCHED.get(e['folder'], []):
            dst = out / name
            if not dst.exists():
                problems.append('%s: 내려받은 문서가 폴더에 없다 — %s' % (e['folder'], name))
                continue
            if not real_pdf(dst):
                problems.append('%s: %s 가 껍데기다 (내용 없음)' % (e['folder'], name))
                continue
            digest, size = sha256(dst), dst.stat().st_size
            known = (sources.get(e['folder']) or {}).get(name)
            if known and known.get('sha256') != digest:
                problems.append('%s: 내려받은 문서가 기록과 다르다 — %s' % (e['folder'], name))
            elif check_only and not known:
                problems.append('%s: 출처 기록이 없다 — %s' % (e['folder'], name))
            if not check_only:
                sources.setdefault(e['folder'], {})[name] = {
                    'url': url, 'sha256': digest, 'size': size, 'desc': desc}
            rows.append((name, url, digest, size, desc))

        for fname, desc in e['env']:
            src = COVID / 'sources' / fname
            if not src.exists():
                continue
            digest, size = sha256(src), src.stat().st_size
            dst = out / '환경자료' / fname
            if not check_only:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            rows.append(('환경자료/%s' % fname,
                         str(src.relative_to(ROOT)), digest, size, desc))

        if not check_only:
            write(out / '평가항목.md', eval_md(e, scoring, rows))
            write(out / 'MANIFEST.md', manifest_md(e, rows))
        report.append((e, rows))

    if not check_only:
        write(OUT / 'README.md', index_md(report))
        write(SOURCES, json.dumps(sources, ensure_ascii=False, indent=2, sort_keys=True))
    return report, problems


def eval_md(e, scoring, rows):
    lines = ['# %s — 평가 항목' % e['folder'].split('_')[0], '',
             e['note'], '',
             '**실험 라운드** — %s' % e['rounds'], '', '---', '']
    lines += ['## 우리가 채점하는 항목', '']
    lines += indicators_md(scoring, e['scoring'])
    lines += ['## 지금까지 잰 것', '']
    lines += results_md(scoring, e['scoring'])
    lines += ['---', '', '## 이 폴더의 문서', '', '| 파일 | 무엇인가 |', '|---|---|']
    for name, _src, _d, size, desc in rows:
        lines.append('| `%s` | %s (%s) |' % (name, desc, _human(size)))
    lines.append('')
    if e['missing']:
        lines += ['## 이 폴더에 없는 것', '',
                  '숨기지 않고 적는다. 못 구한 것, 유료인 것, 일부러 넣지 않은 것이 섞여 있다.',
                  '', '| 무엇 | 사정 |', '|---|---|']
        for what, why in e['missing']:
            lines.append('| %s | %s |' % (what, why))
        lines.append('')
    lines += ['---', '',
              '> 실측값은 **프롬프트에 들어가지 않는다.** 방향도 크기도 모델에게 주지 않으며,',
              '> 금지어 단위 테스트가 그것을 고정한다. 여기 적힌 수치는 채점용이다.']
    return '\n'.join(lines) + '\n'


def manifest_md(e, rows):
    lines = ['# %s — 파일 출처' % e['folder'], '',
             '다시 만들려면 `python scripts/report/collect_policy_raw_data.py`.',
             '**원본**이 `http` 로 시작하면 공공기관에서 직접 내려받은 것이고,',
             '경로면 저장소 안의 파일을 복사한 것이다.', '',
             '| 파일 | 원본 | 크기 | sha256 |', '|---|---|---:|---|']
    for name, src, digest, size, _desc in rows:
        where = ('[%s](%s)' % (src.split('//')[-1].split('/')[0], src)
                 if str(src).startswith('http') else '`%s`' % src)
        lines.append('| `%s` | %s | %s | `%s` |' % (name, where, _human(size), digest))
    lines += ['', 'PDF 는 복사 전에 실제 내용이 있는지 확인했다 — `%PDF` 로 시작하고',
              '`%%EOF` 로 끝나는지. G: 드라이브의 껍데기 파일을 거르기 위해서다.']
    return '\n'.join(lines) + '\n'


def index_md(report):
    lines = ['# 정책 원문 자료', '',
             'JSON 메타데이터가 아니라 실제 문서다. **정책이 무엇인지**(정책원문)와',
             '**결과가 어땠는지**(정답지)를 각각 원문으로 둔다.', '',
             '```',
             'data/policy_raw_data/<정책명>_<PXXX>/',
             '    평가항목.md      채점 항목과 실측값 · 이 폴더의 요약',
             '    <문서들>        실제 파일 (PDF·xlsx·csv)',
             '    환경자료/        시뮬이 배경으로 쓰는 자료 (있는 경우)',
             '    MANIFEST.md     어느 파일이 어디서 왔는지 + sha256',
             '```', '',
             '| 폴더 | 실험 라운드 | 정책원문 | 정답지 원문 | 문서 |',
             '|---|---|---|---|---:|']
    for e, rows in report:
        names = [r[0] for r in rows]
        law = '있음' if any(n.startswith('정책원문') for n in names) else '—'
        key = '있음' if any(n.startswith('정답지') for n in names) else None
        if key is None:
            why = ' '.join(w for w, _ in e['missing'])
            key = ('**봉인**' if '정답지' in why and '봉인' in str(e['missing'])
                   else '**없음**')
        lines.append('| [`%s/`](%s/평가항목.md) | %s | %s | %s | %d개 |' % (
            e['folder'], e['folder'], e['rounds'], law, key, len(rows)))
    lines += ['', '## 읽는 규칙', '',
              '- **출처를 남겼다.** 각 폴더의 `MANIFEST.md` 에 원본 경로 또는 URL 과 sha256 이 있다',
              '- **실측값은 프롬프트에 들어가지 않는다.** 방향도 크기도 모델에게 주지 않는다',
              '- **없는 것은 없다고 적었다.** 홀드아웃(P015)의 정답지는 봉인이고,',
              '  사적모임 제한의 정답지는 유료 DB 에만 있다 — 각 폴더 맨 아래를 볼 것',
              '- 다시 만들려면 `python scripts/report/collect_policy_raw_data.py`,',
              '  사본이 원본과 같은지 보려면 `--check`']
    return '\n'.join(lines) + '\n'


def _human(n):
    for unit in ('B', 'KB', 'MB'):
        if n < 1024 or unit == 'MB':
            return '%.0f%s' % (n, unit) if unit == 'B' else '%.1f%s' % (n, unit)
        n /= 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    report, problems = build(args.check)
    for e, rows in report:
        print('%-34s 문서 %d개%s' % (e['folder'], len(rows),
              ' · 없는 것 %d' % len(e['missing']) if e['missing'] else ''))
    print()
    if problems:
        print('문제 %d건' % len(problems))
        for p in problems:
            print('  ' + p)
        return 1
    print('원본과 어긋난 사본 없음' if args.check else '완료: data/policy_raw_data/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
