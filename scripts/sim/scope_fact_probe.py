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

    changed        계획이 조금이라도 달라진 시민 수 — **0 이면 후보를 버린다**
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
SCOPE_LINE = ("문턱은 적립업종 지출만 센다 — 제외업종(대형마트·백화점·온라인 등)에서 "
              "줄여도 문턱은 가까워지지 않고, 거기서 쓴 돈이 환급을 깎지도 않는다")
TAIL = "못 넘기면 이번 달 혜택은 사라짐"

# 상생 제외업종 — 계획의 sub_category 로 세기 위한 목록.
# sangsaeng_eligibility 의 제외 기준을 세분류 이름으로 옮긴 것이다.
EXCLUDED_SUBS = {
    '종합소매', '슈퍼마켓', '기타상품',          # 대형마트·기업형슈퍼·백화점
    '시계·귀금속',                              # 환금성
    '부동산', '법무', '회계·세무',               # 비소비 서비스
}


def add_scope(user: str) -> str:
    """캐시백 문턱 줄에 범위 문구를 끼운다. 넣을 자리가 없으면 그대로 둔다."""
    if SCOPE_LINE in user:
        return user
    if TAIL not in user:
        return user
    return user.replace(" | " + TAIL, " | " + SCOPE_LINE + " | " + TAIL, 1)


def build(frozen: dict, case: str = 'cashback', arm: str = 'on') -> list[dict]:
    """같은 시민의 off/on 한 쌍씩. 바뀌는 것은 그 한 줄뿐이다."""
    out = []
    for c in frozen.get('cells') or []:
        if c.get('case') != case or c.get('arm') != arm:
            continue
        base = c['user']
        withline = add_scope(base)
        if withline == base:
            continue                     # 문턱 줄이 없는 셀 — 이 가설과 무관하다
        for side, text in (('off', base), ('on', withline)):
            out.append({'aid': c['aid'], 'case': case, 'side': side,
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
    for r in rows:
        if r.get('error'):
            continue
        p = parse(r.get('raw'))
        if p is None:
            continue
        by.setdefault(r['aid'], {})[r['side']] = p
    paired = {a: v for a, v in by.items() if 'off' in v and 'on' in v}
    changed = [a for a, v in paired.items() if v['off']['sha'] != v['on']['sha']]

    def d(key):
        vals = [v['on'][key] - v['off'][key] for v in paired.values()
                if v['on'].get(key) is not None and v['off'].get(key) is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    return {'paired': len(paired), 'changed': len(changed),
            'changed_aids': sorted(changed),
            'd_propensity': d('propensity'), 'd_excluded': d('excluded_ev'),
            'd_events': d('n_events')}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--responses', default='',
                    help='이미 받은 응답 jsonl 이 있으면 그것을 읽어 요약만 한다')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.responses:
        rows = [json.loads(ln) for ln in io.open(a.responses, encoding='utf-8')
                if ln.strip()]
        s = compare(rows)
        print('쌍 %d · **계획이 달라진 시민 %d명**' % (s['paired'], s['changed']))
        for k, lbl in (('d_propensity', '소비성향'), ('d_excluded', '제외업종 이벤트'),
                       ('d_events', '전체 이벤트')):
            v, n = s[k]
            print('  %-14s delta %s (n=%d)'
                  % (lbl, ('%+.4f' % v) if v is not None else '—', n))
        io.open(out / 'summary.json', 'w', encoding='utf-8', newline='\n').write(
            json.dumps(s, ensure_ascii=False, indent=1))
        print()
        if s['changed'] == 0:
            print('**계획이 하나도 안 달라졌다 — 이 줄은 무력하다. 라운드를 돌리지 않는다.**')
        return 0

    frozen = json.loads(Path(a.frozen).read_text(encoding='utf-8'))
    cells = build(frozen)
    io.open(out / 'cells.json', 'w', encoding='utf-8', newline='\n').write(
        json.dumps({'cells': cells}, ensure_ascii=False, indent=1))
    print('시민 %d · 칸 %d (한 시민당 off/on 두 칸)'
          % (len({c['aid'] for c in cells}), len(cells)))
    print('wrote', out / 'cells.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
