"""정책마다 폴더 하나 — 한글 이름으로, 관련 파일을 `policy_data/` 아래 모은다.

넘겨받는 사람이 `P012` 가 무엇인지 몰라도 폴더 이름만 보고 찾을 수 있어야 한다.
그래서 폴더는 정책의 한글 이름으로 짓고, 그 안에 한 장짜리 설명과 자료 폴더를 둔다.

    data/상생소비지원금/
        README.md                   한 장 요약 — 무엇이고, 정답지가 무엇이고, 우리가 무엇을 쟀는가
        policy_data/
            policy.json             정책 정의 (원본의 사본 + sha256)
            answer_key.json         정답지 — 출처·지표·실측값
            results.json            우리 측정 결과 (있는 것만)
            variants/               섭동판 (있는 것만)
            related/                그 정책을 돌리는 데 쓰인 부속 자료 (있는 것만)
            SOURCES.md              어느 파일이 어디서 왔는지, sha256 과 함께

**원본은 옮기지 않는다.** `data/neo4j_load/policies/` 가 그대로 정본이고 여기 있는
것은 sha256 을 단 사본이다. 사본이 정본과 어긋나면 `--check` 가 알려 준다.

    python scripts/report/organize_policy_data.py            # 만든다
    python scripts/report/organize_policy_data.py --check    # 사본이 정본과 같은지만 본다
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_POLICIES = ROOT / 'data/neo4j_load/policies'
SCORING = ROOT / 'data/experiments/scoring_table.json'
COVID = ROOT / 'data/experiments/covid_support_2021'
# data/policies/ 는 정책 적재 파이프라인이 쓰는 폴더다(inbox·processed·failed).
# 거기 끼워 넣지 않는다. 요청받은 자리도 data/<정책명>/ 이다.
OUT_ROOT = ROOT / 'data'

# 폴더 이름은 사람이 읽는 이름이다. 파일 이름(P0xx)은 안에 적는다.
# kind: policy(실제 정책) · holdout(봉인) · placebo(위약) · stress(부하 시험) · backdrop(사회 배경)
PLAN = [
    {'folder': '상생소비지원금', 'file': 'P012.json', 'scoring': 'P012', 'kind': 'policy',
     'variants': ['P012_HIGH.json', 'P012_LOW.json'],
     'note': '2021년 카드 실적 캐시백. 2분기 월평균 대비 3% 초과분의 10%를 다음 달에 환급.'},
    {'folder': '민생회복소비쿠폰_1차', 'file': 'P010.json', 'scoring': 'P010', 'kind': 'policy',
     'note': '2025년 정책지갑 지급. 프롬프트가 동결돼 있다 — prompts/p010 을 건드리지 않는다.'},
    {'folder': '민생회복소비쿠폰_2차_백테스트', 'file': 'P011.json', 'scoring': None,
     'kind': 'stress', 'note': '부하 시험용. 실제 정책의 재현이 아니다.'},
    {'folder': '긴급재난지원금_1차', 'file': 'P013.json', 'scoring': 'EMERGENCY_2020',
     'kind': 'policy', 'note': '2020년 1차 긴급재난지원금. 정책지갑 지급 기전.'},
    {'folder': '서울사랑상품권', 'file': 'P014.json', 'scoring': 'LOCAL_VOUCHER',
     'kind': 'policy', 'note': '지역사랑상품권 할인 발행. 자치구 안에서만 쓸 수 있다.'},
    {'folder': '8대소비쿠폰_홀드아웃', 'file': 'P015.json', 'scoring': 'SECTOR_VOUCHER_2020',
     'kind': 'holdout',
     'note': '홀드아웃. 채점을 한 번만 했고 다시 부르지 않는다. 지표는 돌리기 전에 정책 설계만 보고 적었다.'},
    {'folder': '위약_장보기환급', 'file': 'P090.json', 'scoring': 'PLACEBO_FAKE',
     'kind': 'placebo', 'note': '위약. 있지도 않은 정책에 반응하는지 보는 검정이다.'},
    {'folder': '보행친화거리', 'file': 'P008.json', 'scoring': None, 'kind': 'policy',
     'note': '시설 기전. 소비 지표로 채점하지 않는다.'},
    {'folder': '사회적거리두기_2단계', 'file': None, 'scoring': 'DISTANCING_2020',
     'kind': 'backdrop',
     'related': ['distancing_schedule.json', 'seoul_cases_daily.json',
                 'seoul_vaccination_review.json', 'national_support_rules.json'],
     'note': '정책 JSON 이 아니라 사회 배경이다. environment(covid_2021) 가 규제를 실어 온다.'},
]

KIND_LABEL = {'policy': '정책', 'holdout': '홀드아웃 (봉인)', 'placebo': '위약',
              'stress': '부하 시험', 'backdrop': '사회 배경 (정책 아님)'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    io.open(path, 'w', encoding='utf-8', newline='\n').write(text)


def dump(path, obj):
    write(path, json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def answer_key(scoring, key):
    """채점표에서 그 정책의 정답지만 떼어 낸다."""
    if not key or key not in scoring:
        return None
    b = scoring[key]
    out = {'scoring_table_key': key}
    for f in ('mechanism', 'answer_key', 'indicators', 'window', 'window_note',
              'window_stage2', 'window_stage3', 'not_scorable', 'dow_confound'):
        if f in b:
            out[f] = b[f]
    return out


def our_results(scoring, key):
    """우리가 잰 결과. 채점표에 result_* 로 들어 있는 것만 가져온다."""
    if not key or key not in scoring:
        return None
    b = scoring[key]
    got = {k: v for k, v in b.items() if k.startswith('result')}
    return got or None


def readme(entry, pol, ak, res, files):
    name = (pol or {}).get('name') or entry['folder']
    lines = ['# %s' % name, '']
    lines.append('**%s**' % KIND_LABEL[entry['kind']])
    if pol:
        lines[-1] += ' · `%s` · 기전 `%s` · %s ~ %s' % (
            entry['file'].replace('.json', ''), pol.get('type', '?'),
            pol.get('effective_from', '?'), pol.get('effective_until', '?'))
    lines += ['', entry['note'], '', '---', '']

    if pol and pol.get('description'):
        lines += ['## 정책이 무엇을 하는가', '',
                  ' '.join(str(pol['description']).split()), '']

    if ak:
        lines += ['## 정답지', '']
        if ak.get('answer_key'):
            lines += ['출처 — %s' % ak['answer_key'], '']
        inds = ak.get('indicators') or []
        if inds:
            lines += ['| 지표 | 기대 | 무엇을 재는가 |', '|---|---|---|']
            for i in inds:
                lines.append('| `%s` | %s | %s |' % (
                    i.get('id', '?'), i.get('expect', '?'),
                    str(i.get('desc', '')).replace('|', '·')))
            lines.append('')
    else:
        lines += ['## 정답지', '', '없다 — 채점 대상이 아니다.', '']

    lines += ['## 이 폴더에 무엇이 있는가', '', '```']
    for rel in files:
        lines.append('policy_data/%s' % rel)
    lines += ['```', '',
              '`SOURCES.md` 에 각 파일이 어디서 왔는지 sha256 과 함께 적혀 있다.', '']

    if res:
        lines += ['## 우리가 잰 것', '',
                  '`policy_data/results.json` 에 채점표가 보관한 측정 기록이 그대로 들어 있다.',
                  '**한 런의 관측이며 검증 완료를 뜻하지 않는다** — 크기 비교의 조건은',
                  '`data/experiments/scoring_table.json` 의 `MAGNITUDE_CRITERION.audit_2026_09_20`',
                  '을 따른다.', '']

    lines += ['---', '',
              '> 원본은 옮기지 않았다. `data/neo4j_load/policies/` 가 정본이고 여기 있는 것은',
              '> sha256 을 단 사본이다. 어긋나면 `organize_policy_data.py --check` 가 알려 준다.']
    return '\n'.join(lines) + '\n'


def build(check_only=False):
    scoring = json.loads(SCORING.read_text(encoding='utf-8')) if SCORING.exists() else {}
    rows, problems = [], []
    for entry in PLAN:
        out = OUT_ROOT / entry['folder']
        pdir = out / 'policy_data'
        files, sources = [], []
        pol = None

        if entry['file']:
            src = SRC_POLICIES / entry['file']
            if not src.exists():
                problems.append('%s: 원본 없음 %s' % (entry['folder'], src))
                continue
            pol = json.loads(src.read_text(encoding='utf-8'))
            digest = sha256(src)
            if check_only:
                cp = pdir / 'policy.json'
                if not cp.exists():
                    problems.append('%s: 사본 없음' % entry['folder'])
                elif sha256(cp) != digest:
                    problems.append('%s: 사본이 정본과 다르다' % entry['folder'])
            else:
                shutil.copyfile(src, _mk(pdir / 'policy.json'))
            files.append('policy.json')
            sources.append(('policy.json', src.relative_to(ROOT), digest))

        for v in entry.get('variants') or []:
            src = SRC_POLICIES / v
            if not src.exists():
                continue
            if not check_only:
                shutil.copyfile(src, _mk(pdir / 'variants' / v))
            files.append('variants/%s' % v)
            sources.append(('variants/%s' % v, src.relative_to(ROOT), sha256(src)))

        for r in entry.get('related') or []:
            src = COVID / r
            if not src.exists():
                continue
            if not check_only:
                shutil.copyfile(src, _mk(pdir / 'related' / r))
            files.append('related/%s' % r)
            sources.append(('related/%s' % r, src.relative_to(ROOT), sha256(src)))

        ak = answer_key(scoring, entry.get('scoring'))
        res = our_results(scoring, entry.get('scoring'))
        if not check_only:
            if ak:
                dump(_mk(pdir / 'answer_key.json'), ak)
            if res:
                dump(_mk(pdir / 'results.json'), res)
        if ak:
            files.append('answer_key.json')
        if res:
            files.append('results.json')

        if not check_only:
            write(pdir / 'SOURCES.md', _sources_md(entry, sources))
            files.append('SOURCES.md')
            write(out / 'README.md', readme(entry, pol, ak, res, files))
        rows.append({'entry': entry, 'pol': pol, 'ak': ak, 'res': res, 'files': files})

    if not check_only:
        write(ROOT / 'data/POLICIES.md', _index_md(rows))
    return rows, problems


def _mk(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sources_md(entry, sources):
    lines = ['# 이 폴더의 파일이 어디서 왔는가', '',
             '복사본이다. 원본을 고치면 여기도 다시 만들어야 한다 —',
             '`python scripts/report/organize_policy_data.py`', '',
             '| 파일 | 원본 | sha256 |', '|---|---|---|']
    for name, src, digest in sources:
        lines.append('| `%s` | `%s` | `%s` |' % (name, src, digest))
    lines += ['', '`answer_key.json` 과 `results.json` 은 복사가 아니라',
              '`data/experiments/scoring_table.json` 에서 이 정책 부분만 떼어 낸 것이다.']
    return '\n'.join(lines) + '\n'


def _index_md(rows):
    lines = ['# 정책 자료 — 정책마다 폴더 하나', '',
             '폴더 이름은 정책의 한글 이름이다. 각 폴더 안에 한 장짜리 `README.md` 와',
             '자료를 담은 `policy_data/` 가 있다.', '',
             '```',
             'data/<정책명>/',
             '    README.md              무엇이고 정답지가 무엇인지 한 장',
             '    policy_data/',
             '        policy.json        정책 정의 (원본의 사본)',
             '        answer_key.json    정답지 — 출처·지표·실측값',
             '        results.json       우리 측정 기록 (있는 것만)',
             '        variants/          섭동판 (있는 것만)',
             '        related/           부속 자료 (있는 것만)',
             '        SOURCES.md         어느 파일이 어디서 왔는지 + sha256',
             '```', '',
             '| 폴더 | 성격 | 기전 | 정답지 | 지표 |', '|---|---|---|---|---:|']
    for r in rows:
        e, pol, ak = r['entry'], r['pol'], r['ak']
        src = (ak or {}).get('answer_key') or '—'
        n = len((ak or {}).get('indicators') or [])
        lines.append('| [`%s/`](%s/README.md) | %s | %s | %s | %d |' % (
            e['folder'], e['folder'], KIND_LABEL[e['kind']],
            (pol or {}).get('type', '—'), src, n))
    lines += ['', '## 읽는 규칙', '',
              '- **원본은 `data/neo4j_load/policies/` 다.** 여기 있는 것은 sha256 을 단 사본이고,',
              '  어긋나면 `organize_policy_data.py --check` 가 알려 준다',
              '- **실측값은 프롬프트에 들어가지 않는다.** 방향도 크기도 주지 않는다',
              '- **홀드아웃(8대소비쿠폰)은 채점을 한 번만 했다.** 다시 부르지 않는다',
              '- **`results.json` 은 한 런의 관측이다.** 크기 비교의 조건은 채점표의',
              '  `MAGNITUDE_CRITERION.audit_2026_09_20` 을 따른다']
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='사본이 정본과 같은지만 본다')
    args = ap.parse_args()
    rows, problems = build(check_only=args.check)
    for r in rows:
        e = r['entry']
        print('%-28s %-16s 파일 %d개%s' % (
            e['folder'], KIND_LABEL[e['kind']], len(r['files']),
            ' · 정답지 지표 %d' % len((r['ak'] or {}).get('indicators') or [])
            if r['ak'] else ' · 정답지 없음'))
    print()
    if problems:
        print('문제 %d건' % len(problems))
        for p in problems:
            print('  ' + p)
        return 1
    print('정본과 어긋난 사본 없음' if args.check else '완료: data/<정책명>/ · 목차 data/POLICIES.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
