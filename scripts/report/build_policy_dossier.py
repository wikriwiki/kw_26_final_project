"""Gather everything known about one policy into a single folder per policy.

The pieces were in four places: the definition the loader reads, the indicator list in
`scoring_table.json`, the measured values hardcoded in `build_indicator_comparison.py`,
and whatever round last produced a simulation value. Answering "what does P012 claim and
how close are we?" meant opening all four, so this writes one folder per policy holding
all of it.

The policy definition is COPIED, not moved. `data/neo4j_load/policies/` stays the single
source of truth because the graph loader reads it and P010 is frozen; the copy here
carries the source path and its sha256 so drift is detectable rather than silent.

    python scripts/report/build_policy_dossier.py \
        --result v25=data/experiments/v18_v25_pooled.json \
        --result v34=data/experiments/v18_v34_pooled.json \
        --round v18
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_indicator_comparison import CANNOT, MEASURED, POLICY

ROOT = Path(__file__).resolve().parents[2]
LOADER = ROOT / 'data' / 'neo4j_load' / 'policies'
SCORING = ROOT / 'data' / 'experiments' / 'scoring_table.json'

# The three names disagree because they grew apart: MEASURED calls the 2020 payout
# 'EMERGENCY', the scoring table calls it 'EMERGENCY_2020', and the loader calls it P013.
# Stated here rather than inferred, because it is the join everything else depends on.
POLICIES = {
    'P010': {'measured': None, 'scoring': 'P010', 'file': 'P010.json',
             'title': '민생회복 소비쿠폰 1차', 'kind': 'policy'},
    'P011': {'measured': None, 'scoring': None, 'file': 'P011.json',
             'title': '민생회복 소비쿠폰 2차 (스트레스테스트)', 'kind': 'stress_test'},
    'P012': {'measured': 'P012', 'scoring': 'P012', 'file': 'P012.json',
             'title': '상생소비지원금 (카드 실적 캐시백)', 'kind': 'policy',
             'variants': ['P012_HIGH.json', 'P012_LOW.json']},
    'P013': {'measured': 'EMERGENCY', 'scoring': 'EMERGENCY_2020', 'file': 'P013.json',
             'title': '긴급재난지원금 1차 (정책지갑 지급)', 'kind': 'policy'},
    'P014': {'measured': 'LOCAL_VOUCHER', 'scoring': 'LOCAL_VOUCHER', 'file': 'P014.json',
             'title': '지역사랑상품권 (할인 구매)', 'kind': 'policy'},
    'P015': {'measured': None, 'scoring': 'SECTOR_VOUCHER_2020', 'file': 'P015.json',
             'title': '업종형 소비쿠폰 — 홀드아웃 (정답 봉인)', 'kind': 'holdout',
             'holdout_of': '_holdout'},
    'P090': {'measured': None, 'scoring': 'PLACEBO_FAKE', 'file': 'P090.json',
             'title': '(가짜) 장보기 환급 — 위약', 'kind': 'placebo'},
    'DISTANCING_2020': {'measured': 'DISTANCING', 'scoring': 'DISTANCING_2020', 'file': None,
                        'title': '거리두기 2단계 — 사회 배경 (정책 아님)', 'kind': 'background'},
}

EXPECT_TEXT = {'+': '증가', '-': '감소', '0': '무반응 (방어선)', 'rank': '순위'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def answer_key(pid, spec, table):
    """Every indicator this policy is judged on, with its measured value and why."""
    entry = table.get(spec['scoring']) or {} if spec['scoring'] else {}
    # The holdout's indicator list is prose ("정답값은 여전히 봉인"), not rows. Keeping the
    # sealed wrapper beside it is the point: the folder must show that a key exists and
    # that it has not been opened.
    rows = entry.get('indicators')
    listed = {i['id']: i for i in rows} if isinstance(rows, list) else {}
    ids = [k for k, v in MEASURED.items() if v[0] == spec['measured']] if spec['measured'] else []
    for extra in listed:
        if extra not in ids:
            ids.append(extra)
    out = []
    for iid in ids:
        pol, value, expect, desc = MEASURED.get(iid, (None, None, None, None))
        row = listed.get(iid, {})
        out.append({
            'id': iid,
            'description': desc or row.get('desc'),
            'expected_sign': expect or row.get('expect'),
            'expected_meaning': EXPECT_TEXT.get(expect or row.get('expect')),
            'measured_value': value,
            'measured_unit': '%' if value is not None else None,
            'scale': row.get('scale'),
            'metric': row.get('metric'),
            'simulation_cannot_produce': CANNOT.get(iid),
        })
    name, date, real, source = POLICY.get(spec['measured'], (None, None, None, None))
    result = {
        'policy_id': pid,
        'title': spec['title'],
        'kind': spec['kind'],
        'cell_date': date,
        'real_world_counterpart': real,
        'answer_key_source': source or entry.get('answer_key'),
        'mechanism': entry.get('mechanism'),
        'indicators': out,
        'note': ('실측값과 기대부호는 공개된 연구의 결과다. 프롬프트에는 들어가지 않는다 — '
                 '검사로 고정돼 있다.'),
    }
    if spec.get('holdout_of'):
        result['sealed'] = table.get(spec['holdout_of'], {}).get(spec['scoring'], {})
        result['sealed_note'] = ('정답값은 봉인돼 있다. 홀드아웃은 최종 채점 1회에만 개봉하고, '
                                 '이미 한 번 썼으면 다시 돌리지 않는다.')
    return result


def results_for(pid, spec, runs, round_label):
    """What our simulation produced for this policy's indicators, per candidate."""
    ids = [k for k, v in MEASURED.items() if v[0] == spec['measured']] if spec['measured'] else []
    if not ids:
        return None
    per = {}
    for label, pooled in runs.items():
        rows = {}
        for iid in ids:
            block = pooled.get(iid)
            if block is None:
                rows[iid] = {'value': None, 'reason': CANNOT.get(iid) or '이 라운드에서 값 없음'}
                continue
            lo, hi = block.get('lo'), block.get('hi')
            stable = block.get('sign_stable')
            measured = MEASURED[iid][1]
            expect = MEASURED[iid][2]
            if not stable:
                verdict = '미결 — 구간이 0을 지난다'
            elif expect == 'rank':
                verdict = '순위 일치' if (block['mean'] > 0) == ((measured or 0) > 0) else '순위 반대'
            elif expect == '0':
                verdict = '방어선 — 동등성 띠 미정'
            else:
                want = 1 if expect == '+' else -1
                verdict = '부호 일치' if (1 if block['mean'] > 0 else -1) == want else '부호 반대'
            degenerate = lo is not None and hi is not None and lo == hi
            if degenerate:
                verdict += ' · **퇴화** — 구간 폭이 0이다. 관측이 한두 건이라는 뜻이지 신호가 아니다'
            rows[iid] = {'value': block['mean'], 'ci95': [lo, hi],
                         'sign_resolved': bool(stable) and not degenerate,
                         'degenerate_interval': degenerate, 'verdict': verdict}
        per[label] = rows
    return {'round': round_label, 'candidates': per,
            'note': ('구간은 95% 붓스트랩이다. 0을 지나면 부호조차 말할 수 없다. '
                     '금액은 맞대지 않는다 — 카탈로그는 낱개 단가, 실측은 카드매출이다.')}


def readme(pid, spec, key, res):
    L = ['# %s — %s' % (pid, spec['title']), '']
    if key['real_world_counterpart']:
        L += ['- **현실 대응** %s' % key['real_world_counterpart'],
              '- **칸의 날짜** %s' % (key['cell_date'] or '—'),
              '- **정답지 출처** %s' % (key['answer_key_source'] or '—')]
    if key['mechanism']:
        L.append('- **기전** %s' % key['mechanism'])
    L += ['', '## 이 폴더에 무엇이 있나', '',
          '| 파일 | 내용 |', '|---|---|',
          '| `policy.json` | 정책 정의 **사본**. 원본은 `data/neo4j_load/policies/` 이고 그쪽이 기준이다 |',
          '| `answer_key.json` | 이 정책이 채점되는 지표 전수 — 기대부호·실측값·산출 불가 사유 |']
    if res:
        L.append('| `results.json` | 우리 시뮬레이션 값 (%s 라운드) |' % res['round'])
    if spec.get('variants'):
        L.append('| `variants/` | 강도 변형 %s |' % ', '.join(spec['variants']))
    if key.get('sealed'):
        L += ['', '## 정답 봉인', '',
              '이 정책은 **홀드아웃**이다. 정답값은 봉인돼 있고 최종 채점 한 번에만 연다.',
              '`answer_key.json` 의 `sealed` 에 설계(관측창·채점 업종·제외 사유)만 담겨 있고 값은 없다.', '']
    if not key['indicators']:
        L += ['', '## 지표', '',
              '**채점 지표가 없다.** 이 항목은 %s 이라 정답지와 맞대지 않는다.'
              % {'stress_test': '부하 시험용', 'holdout': '봉인된 홀드아웃',
                 'placebo': '위약(효과가 없어야 정상)', 'policy': '정책'}.get(key['kind'], key['kind']), '']
    else:
        L += ['', '## 지표', '', '| 지표 | 기대 | 실측 | 설명 | 우리가 낼 수 있나 |', '|---|:-:|---:|---|---|']
    for row in key['indicators']:
        m = ('%+.2f' % row['measured_value']) if row['measured_value'] is not None else '수치 없음'
        can = '—' if not row['simulation_cannot_produce'] else '**불가** — ' + row['simulation_cannot_produce']
        L.append('| %s | `%s` | %s | %s | %s |'
                 % (row['id'], row['expected_sign'] or '?', m, row['description'] or '', can))
    if res:
        L += ['', '## 현재 판정 (%s)' % res['round'], '', '| 지표 | ' +
              ' | '.join(res['candidates']) + ' |', '|---|' + '---|' * len(res['candidates'])]
        for row in key['indicators']:
            cells = []
            for label in res['candidates']:
                r = res['candidates'][label].get(row['id'], {})
                cells.append('산출 불가' if r.get('value') is None
                             else '%+.1f · %s' % (r['value'], r['verdict']))
            L.append('| %s | %s |' % (row['id'], ' | '.join(cells)))
    L += ['', '> 실측값은 프롬프트에 들어가지 않는다. 방향도 크기도 주지 않는 것이 이 실험의 규칙이고,',
          '> 금지어 목록을 둔 단위 테스트로 고정돼 있다.', '']
    return '\n'.join(L)


KIND_TEXT = {'policy': '정책', 'stress_test': '부하 시험', 'holdout': '홀드아웃 (봉인)',
             'placebo': '위약', 'background': '사회 배경 (정책 아님)'}


def write_index(out_root, index, round_label):
    """One table naming every policy folder, so nobody has to open eight READMEs to look."""
    L = ['# 정책 한 곳 모음', '',
         '정책마다 폴더 하나다. 정의·정답지·우리 결과가 그 안에 함께 있다.',
         '정책 정의의 원본은 `data/neo4j_load/policies/` 이고 여기 있는 것은 sha256 을 단 사본이다.', '',
         '| 폴더 | 성격 | 현실 대응 | 정답지 | 지표 | %s 결과 |' % round_label,
         '|---|---|---|---|---:|---|']
    for pid, spec, key, res in index:
        scored, resolved = set(), set()
        if res:
            for rows in res['candidates'].values():
                for iid, r in rows.items():
                    if r.get('value') is not None:
                        scored.add(iid)
                    if r.get('sign_resolved'):
                        resolved.add(iid)
        cell = ('—' if not res else '값 %d/%d · 부호 정해짐 %d'
                % (len(scored), len(key['indicators']), len(resolved)))
        L.append('| [`%s/`](%s/README.md) | %s | %s | %s | %d | %s |'
                 % (pid, pid, KIND_TEXT.get(key['kind'], key['kind']),
                    key['real_world_counterpart'] or '—',
                    (key['answer_key_source'] or '—').replace('|', '/'),
                    len(key['indicators']), cell))
    L += ['', '## 읽는 규칙', '',
          '- **실측값은 프롬프트에 들어가지 않는다.** 방향도 크기도 주지 않으며 금지어 단위 테스트로 고정돼 있다',
          '- **금액을 맞대지 않는다.** 카탈로그는 낱개 단가이고 실측은 카드매출이라 비율만 같은 종류의 수다',
          '- **구간이 0을 지나면 부호조차 말할 수 없다.** "값이 있다"와 "부호가 정해졌다"는 다른 칸이다',
          '- **홀드아웃은 한 번만 연다.** P015 는 이미 썼으므로 다시 돌리지 않는다', '']
    text = chr(10).join(L)
    io.open(Path(out_root) / 'POLICIES.md', 'w', encoding='utf-8',
            newline=chr(10)).write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', action='append', default=[], metavar='label=pooled.json')
    ap.add_argument('--round', default='최근')
    ap.add_argument('--out-root', default=str(ROOT / 'data'))
    args = ap.parse_args()

    runs = {}
    for item in args.result:
        label, path = item.split('=', 1)
        runs[label] = json.loads(Path(path).read_text(encoding='utf-8'))['pooled']

    table = json.loads(SCORING.read_text(encoding='utf-8'))
    out_root = Path(args.out_root)
    index = []
    for pid, spec in POLICIES.items():
        folder = out_root / pid
        folder.mkdir(parents=True, exist_ok=True)
        key = answer_key(pid, spec, table)
        res = results_for(pid, spec, runs, args.round) if runs else None

        if spec['file']:
            src = LOADER / spec['file']
            if src.exists():
                body = json.loads(src.read_text(encoding='utf-8'))
                wrapped = {'_source': {'path': 'data/neo4j_load/policies/' + spec['file'],
                                       'sha256': sha256(src),
                                       'note': '사본이다. 고칠 일이 있으면 원본을 고치고 이 파일을 다시 만든다.'},
                           'policy': body}
                io.open(folder / 'policy.json', 'w', encoding='utf-8', newline='\n').write(
                    json.dumps(wrapped, ensure_ascii=False, indent=1))
        for var in spec.get('variants', []):
            src = LOADER / var
            if src.exists():
                (folder / 'variants').mkdir(exist_ok=True)
                io.open(folder / 'variants' / var, 'w', encoding='utf-8', newline='\n').write(
                    src.read_text(encoding='utf-8'))

        io.open(folder / 'answer_key.json', 'w', encoding='utf-8', newline='\n').write(
            json.dumps(key, ensure_ascii=False, indent=1))
        if res:
            io.open(folder / 'results.json', 'w', encoding='utf-8', newline='\n').write(
                json.dumps(res, ensure_ascii=False, indent=1))
        io.open(folder / 'README.md', 'w', encoding='utf-8', newline='\n').write(
            readme(pid, spec, key, res))
        index.append((pid, spec, key, res))
        print('%-16s 지표 %d개%s' % (pid, len(key['indicators']),
                                     ' · 결과 있음' if res else ''))

    write_index(out_root, index, args.round)
    print('색인  data/POLICIES.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
