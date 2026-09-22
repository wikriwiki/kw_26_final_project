"""현재 실험 중인 정책의 **실제 문서**를 한 폴더에 모은다.

JSON 메타데이터가 아니라 PDF·HTML·표 같은 원문이다. 넘겨받는 사람이 정책을 직접
읽고, 우리가 무엇으로 채점하는지 확인할 수 있어야 한다.

    data/policy_raw_data/<정책명>_<PXXX>/
        평가항목.md          우리가 채점하는 항목과 실측값 — 이 폴더의 요약이기도 하다
        <문서들>            실제 파일 (PDF·xlsx·csv 등)
        환경자료/            시뮬이 배경으로 쓰는 자료 (있는 경우)
        MANIFEST.md         어느 파일이 어디서 왔는지, sha256 과 크기

**현재 실험 중인 정책만 넣는다.** 지금은 둘이다.

    상생소비지원금 (P012)       v30·v43·v44 라운드
    사회적거리두기 (2020-11)     v50 라운드 — 지금 돌고 있다

없는 것은 없다고 적는다. 거리두기는 채점 구간(2020-11-24 수도권 2단계)의 고시 원문과
정답지(서울연구원 2021.4) 원문이 저장소에 없다. 대신 구조화된 단계표가 있고,
그 사실을 `평가항목.md` 에 적는다.

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

# (원본 경로, 폴더 안에서의 이름, 한 줄 설명)
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
    ('data/neo4j_load/policies/P012.json',
     '시뮬정의_P012.json',
     '시뮬레이터에 실린 정책 정의. 위 시행방안을 이 형식으로 옮긴 것'),
]

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

PLAN = [
    {'folder': '상생소비지원금_P012', 'scoring': 'P012',
     'rounds': 'v30(중단) · v43(취소) · v44(완료)',
     'docs': P012_DOCS, 'env': [],
     'missing': [],
     'note': '2021년 카드 실적 캐시백. 2분기 월평균 대비 3% 초과분의 10%를 다음 달에 환급.'},
    {'folder': '사회적거리두기_DISTANCING2020', 'scoring': 'DISTANCING_2020',
     'rounds': 'v50 — 지금 돌고 있다 (v5 대 v45 정답지 평가)',
     'docs': DISTANCING_DOCS, 'env': DISTANCING_ENV,
     'missing': [
         ('채점 구간의 고시 원문', '2020-11-24 수도권 2단계 방역수칙 전문. '
          '단계표에 옮겨져 있으나 원문 파일은 저장소에 없다'),
         ('정답지 원문', '서울연구원 「코로나19 확산이 서울 지역에 미친 경제적 손실」'
          '(2021.4, 신한카드 서울 패널). 수치는 채점표에 옮겨져 있으나 PDF 는 없다'),
     ],
     'note': '2020년 11월 수도권 2단계. 정책 JSON 이 아니라 사회 배경이고 '
             'environment(covid_2021) 가 규제를 실어 온다.'},
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
        return ['아직 채점 기록이 없다.', '']
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
        lines += ['## 저장소에 없는 것', '',
                  '숨기지 않고 적는다. 아래는 수치만 채점표에 옮겨져 있고 원문 파일이 없다.',
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
             '복사본이다. 원본을 고치면 다시 만들어야 한다 —',
             '`python scripts/report/collect_policy_raw_data.py`', '',
             '| 파일 | 원본 | 크기 | sha256 |', '|---|---|---:|---|']
    for name, src, digest, size, _desc in rows:
        lines.append('| `%s` | `%s` | %s | `%s` |' % (name, src, _human(size), digest))
    lines += ['', 'PDF 는 복사 전에 실제 내용이 있는지 확인했다 — `%PDF` 로 시작하고',
              '`%%EOF` 로 끝나는지. G: 드라이브의 껍데기 파일을 거르기 위해서다.']
    return '\n'.join(lines) + '\n'


def index_md(report):
    lines = ['# 정책 원문 자료 — 현재 실험 중인 정책만', '',
             'JSON 메타데이터가 아니라 실제 문서다. 정책을 직접 읽고, 우리가 무엇으로',
             '채점하는지 확인할 수 있다.', '',
             '```',
             'data/policy_raw_data/<정책명>_<PXXX>/',
             '    평가항목.md      채점 항목과 실측값 · 이 폴더의 요약',
             '    <문서들>        실제 파일 (PDF·xlsx·csv)',
             '    환경자료/        시뮬이 배경으로 쓰는 자료 (있는 경우)',
             '    MANIFEST.md     어느 파일이 어디서 왔는지 + sha256',
             '```', '',
             '| 폴더 | 실험 라운드 | 문서 | 정답지 원문 |', '|---|---|---:|---|']
    for e, rows in report:
        has_key = any('정답지' in r[0] for r in rows)
        lines.append('| [`%s/`](%s/평가항목.md) | %s | %d개 | %s |' % (
            e['folder'], e['folder'], e['rounds'], len(rows),
            '있음' if has_key else '**없음** — 수치만 채점표에'))
    lines += ['', '## 읽는 규칙', '',
              '- **원본은 옮기지 않았다.** 각 폴더의 `MANIFEST.md` 에 원본 경로와 sha256 이 있다',
              '- **실측값은 프롬프트에 들어가지 않는다.** 방향도 크기도 모델에게 주지 않는다',
              '- **없는 것은 없다고 적었다.** 거리두기는 채점 구간 고시 원문과 정답지 PDF 가 없다',
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
