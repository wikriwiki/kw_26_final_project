"""범위의 산술 한 줄이 **계획을 바꾸기는 하는가** — 전체 라운드 전에 싸게 본다.

    python scripts/sim/scope_fact_probe.py --frozen <frozen_inputs.json> --out <dir>

사전등록: `experiments/scope_fact/prereg.md`

## 왜 탐침을 먼저 돌리나

후보를 만들었다고 라운드를 돌리면 안 된다. **P010 역진의 벽에서 서술 프롬프트가
무효인 것을 이미 겪었다** — 문장을 더해도 행동이 한 글자도 안 달라지는 일이
실제로 있었다. 그러면 12일 × 200명 × 두 팔이 통째로 낭비다.

그래서 먼저 묻는다. **이 줄이 아무것도 바꾸지 않는가.**

동결된 파일럿 맥락(`frozen_inputs.json`)의 캐시백 셀에서 **그 한 줄만** 넣고
빼서 같은 시민을 두 번 부른다. 바뀌는 것이 그 줄뿐이므로, 계획이 달라지면
그 줄 때문이고 안 달라지면 그 줄은 무력하다.

## 읽는 것

    **쌍별 부호 검정이 판정이다.** `changed` 는 관문이 못 된다 —
    temperature 0.7 에서는 프롬프트가 한 글자만 달라져도 토큰 경로가 갈리므로
    **의미 없는 줄을 끼워도 계획이 달라진다.** 그것을 영가설 대조로 확인했다.

    sign           같은 시민의 off/on 을 맞대 기대 방향 쌍이 몇인지 센다.
                   동전 던지기면 절반이다 — 양측 이항 p 로 읽는다
    changed        계획이 조금이라도 달라진 시민 수 — **참고로만 둔다**
    propensity     최상위 소비성향 delta. 총액 스칼라가 움직이는지
    excluded_ev    제외업종 이벤트 수 delta. 줄이기를 멈추는지가 이 가설의 핵심
    eligible_ev    적립업종 이벤트 수 delta. 함께 보지 않으면 그냥 다 늘었는지 모른다

## 순환이 아니다

바꾸는 것은 제도 산식의 성질 한 줄이고, 재는 것은 계획의 내용이다. 측정
공식(순효과·밴드·MPC)을 입력으로 넣지 않는다. 방향을 지시하는 문장도 없다.
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

# 코드가 켜질 때 붙는 문구 그대로. 여기서 손으로 고치면 탐침과 본런이 달라진다 —
# 한 곳에서 읽어 오는 것이 옳지만, 탐침은 서버의 맥락 파일만 쓰므로 문구를
# 복사하고 그 일치를 시험으로 못 박는다(tests/unit/sim/test_scope_fact.py).
#
# **후보를 자료로 적는다.** 다음 후보마다 스크립트를 복사하면 복사본끼리 문구가
# 어긋나고 무엇을 쟀는지 흐려진다. 코드는 하나만 둔다.
#
#   line    끼울 문장 (코드 렌더와 **바이트가 같아야 한다**)
#   tail    끼울 자리 표식
#   case    동결 맥락의 어느 셀을 쓰는가
#   expect  지표별 **기대 방향**. {'propensity': 'down', ...}
#           적지 않으면 전부 'up' 으로 찍히는데, 그러면 "소비성향이 내려가야
#           한다" 는 후보에서 표가 정반대로 읽힌다. 양측 p 는 같아서 판정은
#           안 바뀌지만 **이름이 틀리면 사람이 틀리게 읽는다.** 실제로 그랬다.
#   mode    끼우는 방식. 블록마다 글의 모양이 다르다
#             'pipe'         정책 줄 — ` | ` 로 이어진다. 표식 **앞**에 칸을 하나 넣는다
#             'bullet_after' 환경 블록 — `- ` 로 시작하는 줄 목록. 표식 줄 **다음 줄**에
#                            새 항목을 넣는다
CANDIDATES = {
    'scope_fact': {
        'line': ("문턱은 적립업종 지출만 센다 — 제외업종(대형마트·백화점·온라인 등)에서 "
                 "줄여도 문턱은 가까워지지 않고, 거기서 쓴 돈이 환급을 깎지도 않는다"),
        'tail': "못 넘기면 이번 달 혜택은 사라짐",
        'case': 'cashback',
        'mode': 'pipe',
        'expect': {'excluded_ev': 'up', 'propensity': 'up', 'n_events': 'up'},
        'why': '제외업종에서 줄여도 문턱은 가까워지지 않는다 — 제도 산식의 성질',
    },
    # **영가설 대조.** 같은 자리에 같은 길이로 끼우되 **행동과 무관한** 사실이다.
    # temperature 0.7 에서는 아무 줄이나 끼워도 토큰 경로가 갈려 계획이 달라진다 —
    # 그러면 changed 관문이 무엇이든 통과시킨다. 이 줄이 후보와 같은 changed 를
    # 내면 그 관문은 잡음을 센 것이고, 판정을 방향 지표로 옮겨야 한다.
    # 코드 렌더에는 없는 문장이므로 레지스트리 대조 시험에서 면제한다.
    'null_control': {
        'line': ('이 안내는 카드사 앱과 홈페이지에서도 같은 내용으로 볼 수 있으며, '
                 '문의는 고객센터에서 받는다'),
        'tail': '못 넘기면 이번 달 혜택은 사라짐',
        'mode': 'pipe',
        'case': 'cashback',
        'why': '행동과 무관한 사실 — changed 관문이 잡음을 세는지 본다',
        'is_null': True,
    },
    # 후보 2 — 확진 수준에 기준을 같이 준다. 동결 맥락의 거리두기 셀(2020-11-24)에
    # 맞춘 값이다. 날짜가 다른 맥락에 쓰려면 후보를 따로 등록해야 한다 —
    # 배수를 손으로 바꾸면 탐침과 본런이 다른 숫자를 말하게 된다.
    'case_trend_2020_11_24': {
        'line': '2주 전 7일 평균은 42명 — 지금은 그 2.6배',
        'tail': '최근 7일 평균',
        'mode': 'bullet_after',
        'case': 'distancing',
        # 제약에 더 반응한다는 가설이므로 **소비성향은 내려가야** 한다.
        'expect': {'propensity': 'down', 'excluded_ev': 'down', 'n_events': 'down'},
        'why': '수준만 주고 기준을 안 주면 해석할 수 없다 — 같은 원자료에서 센 배수',
    },
}

# 기존 이름은 그대로 둔다 — 시험과 런너가 쓴다.
SCOPE_LINE = CANDIDATES['scope_fact']['line']
TAIL = CANDIDATES['scope_fact']['tail']

# 상생 제외업종 — 계획의 sub_category 로 세기 위한 목록.
# sangsaeng_eligibility 의 제외 기준을 세분류 이름으로 옮긴 것이다.
EXCLUDED_SUBS = {
    '종합소매', '슈퍼마켓', '기타상품',          # 대형마트·기업형슈퍼·백화점
    '시계·귀금속',                              # 환금성
    '부동산', '법무', '회계·세무',               # 비소비 서비스
}


def insert(user: str, line: str, tail: str, mode: str = 'pipe') -> str:
    """맥락에 한 줄을 끼운다. **넣을 자리가 없으면 그대로 둔다.**

    그대로 두는 것이 중요하다 — 억지로 끼우면 앞뒤가 안 맞는 맥락이 되고,
    그러면 그 줄이 아니라 **어색한 글**에 대한 반응을 재게 된다.

        pipe          ` | ` 로 이어진 정책 줄. 표식 앞에 칸을 하나 넣는다
        bullet_after  `- ` 줄 목록인 환경 블록. 표식이 든 줄 **다음**에 항목을 넣는다
    """
    if line in user:
        return user
    if mode == 'pipe':
        if tail not in user:
            return user
        return user.replace(" | " + tail, " | " + line + " | " + tail, 1)
    if mode == 'bullet_after':
        out = []
        done = False
        for ln in user.split(chr(10)):
            out.append(ln)
            if not done and tail in ln and ln.lstrip().startswith('-'):
                indent = ln[:len(ln) - len(ln.lstrip())]
                out.append('%s- %s' % (indent, line))
                done = True
        return chr(10).join(out) if done else user
    raise ValueError('모르는 끼우기 방식: %r' % mode)


def add_scope(user: str) -> str:
    """기존 이름 — `scope_fact` 후보를 끼운다."""
    return insert(user, SCOPE_LINE, TAIL)


def build(frozen: dict, case: str = 'cashback', arm: str = 'on',
          candidate: str = 'scope_fact') -> list[dict]:
    """같은 시민의 off/on 한 쌍씩. 바뀌는 것은 그 한 줄뿐이다."""
    spec = CANDIDATES[candidate]
    case = spec.get('case') or case
    out = []
    for c in frozen.get('cells') or []:
        if c.get('case') != case or c.get('arm') != arm:
            continue
        base = c['user']
        withline = insert(base, spec['line'], spec['tail'],
                          spec.get('mode', 'pipe'))
        if withline == base:
            continue                     # 문턱 줄이 없는 셀 — 이 가설과 무관하다
        for side, text in (('off', base), ('on', withline)):
            out.append({'aid': c['aid'], 'case': case, 'candidate': candidate, 'side': side,
                        'date': c.get('date'), 'zones': c.get('zones'),
                        'user': text})
    return out


# ------------------------------------------------------------------ 응답 읽기
_PROP = re.compile(r'"daily_propensity"\s*:\s*([0-9.]+)')


def parse(raw: str) -> dict | None:
    """계획에서 소비성향과 업종별 이벤트 수를 뽑는다."""
    if not raw:
        return None
    prop = None
    m = _PROP.search(raw)
    if m:
        try:
            prop = float(m.group(1))
        except ValueError:
            prop = None
    subs = re.findall(r'"sub_category"\s*:\s*"([^"]*)"', raw)
    cats = re.findall(r'"category"\s*:\s*"([^"]*)"', raw)
    return {'propensity': prop,
            'n_events': len(cats),
            'excluded_ev': sum(1 for s in subs if s in EXCLUDED_SUBS),
            'subs': subs, 'cats': cats,
            'sha': __import__('hashlib').sha256(raw.encode('utf-8')).hexdigest()[:12]}


def compare(rows: list[dict]) -> dict:
    """시민별로 off/on 을 맞대고 요약한다."""
    by = {}
    _rows = list(rows)
    for r in _rows:
        if r.get('error'):
            continue
        p = parse(r.get('raw'))
        if p is None:
            continue
        # 시드까지 키에 넣는다 — 시드별로 나눠 부르지 않아도 24쌍이 온전히 남는다.
        by.setdefault((r['aid'], r.get('seed')), {})[r['side']] = p
    paired = {k: v for k, v in by.items() if 'off' in v and 'on' in v}
    changed = [k for k, v in paired.items() if v['off']['sha'] != v['on']['sha']]

    def d(key):
        vals = [v['on'][key] - v['off'][key] for v in paired.values()
                if v['on'].get(key) is not None and v['off'].get(key) is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    return {'paired': len(paired), 'changed': len(changed),
            'changed_aids': sorted(str(k) for k in changed),
            'd_propensity': d('propensity'), 'd_excluded': d('excluded_ev'),
            'd_events': d('n_events'),
            'sign_excluded': sign_test(paired, 'excluded_ev', _up(_rows, 'excluded_ev')),
            'sign_propensity': sign_test(paired, 'propensity', _up(_rows, 'propensity')),
            'sign_events': sign_test(paired, 'n_events', _up(_rows, 'n_events'))}


def _up(rows, key) -> bool:
    """이 응답들이 속한 후보의 **기대 방향**. 모르면 'up' 으로 둔다.

    응답 줄에 후보 이름이 실려 있다(build 가 넣는다). 없으면 옛 자료이므로
    기존 동작(증가 기대)을 유지한다 — 다시 세었을 때 수가 달라지면 안 된다.
    """
    name = next((r.get('candidate') for r in rows if r.get('candidate')), None)
    spec = CANDIDATES.get(name or '', {})
    return (spec.get('expect') or {}).get(key, 'up') != 'down'


def sign_test(pairs, key, want_up=True):
    """같은 시민의 off/on 차이가 **기대 방향인 쌍이 몇인가.**

    평균만 보면 한쪽으로 크게 튄 한 사람이 전체를 끌고 간다. 쌍마다 부호만
    세면 그 영향이 빠진다. 동점(둘 다 0)은 빼고 센다 — 제외업종처럼 애초에
    아무도 안 가는 지표는 동점이 대부분이고, 그것을 분모에 넣으면 p 가
    실제보다 좋아 보인다. **동점 수를 함께 돌려주는 이유다.**
    """
    import math
    d = [(v['on'][key] - v['off'][key]) for v in pairs.values()
         if v['on'].get(key) is not None and v['off'].get(key) is not None]
    up = sum(1 for x in d if x > 0)
    dn = sum(1 for x in d if x < 0)
    tie = sum(1 for x in d if x == 0)
    n = up + dn
    if n:
        k = max(up, dn)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n))
    else:
        p = 1.0
    return {'hit': up if want_up else dn, 'miss': dn if want_up else up,
            'tie': tie, 'n': n, 'p': p, 'dir': '증가' if want_up else '감소',
            'mean': (sum(d) / len(d)) if d else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--responses', default='',
                    help='이미 받은 응답 jsonl 이 있으면 그것을 읽어 요약만 한다')
    ap.add_argument('--candidate', default='scope_fact',
                    choices=sorted(CANDIDATES),
                    help='어느 후보를 끼울 것인가. 후보는 CANDIDATES 에 자료로 적는다')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.responses:
        rows = [json.loads(ln) for ln in io.open(a.responses, encoding='utf-8')
                if ln.strip()]
        s = compare(rows)
        print('쌍 %d · 계획이 달라진 시민 %d명 (참고 — 관문이 못 된다)'
              % (s['paired'], s['changed']))
        print()
        print('쌍별 부호 검정 — **이것이 판정이다**')
        for k, lbl in (('sign_excluded', '제외업종 이벤트'),
                       ('sign_propensity', '소비성향'), ('sign_events', '전체 이벤트')):
            t = s[k]
            print('  %-14s 기대방향(%s) %d · 반대 %d · 동점 %d   평균 %s   양측 p=%.3f  %s'
                  % (lbl, t.get('dir', '?'), t['hit'], t['miss'], t['tie'],
                     ('%+.4f' % t['mean']) if t['mean'] is not None else '—',
                     t['p'], '**갈린다**' if t['p'] < 0.05 else '동전 던지기와 구별 안 됨'))
        io.open(out / 'summary.json', 'w', encoding='utf-8', newline='\n').write(
            json.dumps(s, ensure_ascii=False, indent=1))
        print()
        if s['changed'] == 0:
            print('**계획이 하나도 안 달라졌다 — 이 줄은 무력하다. 라운드를 돌리지 않는다.**')
        return 0

    frozen = json.loads(Path(a.frozen).read_text(encoding='utf-8'))
    cells = build(frozen, candidate=a.candidate)
    io.open(out / 'cells.json', 'w', encoding='utf-8', newline='\n').write(
        json.dumps({'cells': cells}, ensure_ascii=False, indent=1))
    print('후보 %s — %s' % (a.candidate, CANDIDATES[a.candidate]['why']))
    print('시민 %d · 칸 %d (한 시민당 off/on 두 칸)'
          % (len({c['aid'] for c in cells}), len(cells)))
    print('wrote', out / 'cells.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
