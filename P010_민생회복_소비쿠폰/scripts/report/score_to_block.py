"""채점 JSON 을 **채점표에 붙일 블록**으로 바꾼다.

    python scripts/report/score_to_block.py output/rounds/score_ruler_a.json \
        --key EMERGENCY_2020 --name result_ruler_a --design "..." --window "..."

## 왜 손으로 옮기지 않는가

라운드가 끝날 때마다 채점 JSON 의 수치를 채점표에 옮겨 적었다. 그 자리에서
자릿수를 한 번 틀리면 **표·그림·판정이 모두 틀린 수를 말하고**, 원본과
대조하기 전에는 아무도 모른다. 실제로 P015 는 사람이 읽는 형식으로 적혀
있어서 기계가 못 세는 채로 표에서 통째로 빠져 있었다.

이 스크립트는 채점 JSON 에서 **그대로** 뽑아 붙일 수 있는 형태로 찍는다.
반올림도 기존 블록과 같은 규칙으로 한다(금액은 정수, 몫은 소수 넷째, pct 는
소수 첫째).

## 붙이는 것은 사람이 한다

찍어 주기만 하고 채점표를 직접 고치지는 않는다. **어느 런을 읽는지가 코드에
숨으면 안 되기 때문이다** — 블록 이름과 design 문구는 사람이 정해 적는다.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _round(v, base):
    """기존 블록과 같은 자릿수. 몫(0~1.5)은 소수 넷째, 금액은 정수."""
    if not isinstance(v, (int, float)):
        return v
    share = isinstance(base, (int, float)) and 0 < abs(base) <= 1.5
    return round(v, 4) if share else int(round(v))


def build(score: dict, design: str, window: str) -> dict:
    out = {}
    if design:
        out['design'] = design
    if window:
        out['window'] = window
    for r in (score.get('results') or []):
        iid = r.get('id')
        if not iid:
            continue
        mean, base, ci, n = r.get('mean'), r.get('base'), r.get('ci'), r.get('n')
        if not isinstance(mean, (int, float)):
            # 순위·관측부족 지표 — hit 와 근거 문구만 남긴다
            out[iid] = {'hit': r.get('hit'), 'note': str(r.get('got') or '')[:80]}
            continue
        blk = {'mean': _round(mean, base)}
        if isinstance(base, (int, float)) and base:
            blk['pct'] = round(100.0 * mean / base, 1)
            blk['base'] = _round(base, base)
        if isinstance(ci, list) and len(ci) == 2:
            blk['ci'] = [_round(ci[0], base), _round(ci[1], base)]
        if n:
            blk['n'] = n
        blk['hit'] = r.get('hit')
        out[iid] = blk
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('score', help='score_*.json')
    ap.add_argument('--key', required=True, help='채점표의 정책 키 (예: EMERGENCY_2020)')
    ap.add_argument('--name', required=True, help='블록 이름 (예: result_ruler_a)')
    ap.add_argument('--design', default='', help='이 런이 무엇인지 한 줄')
    ap.add_argument('--window', default='', help='채점 창')
    a = ap.parse_args()

    score = json.loads(io.open(a.score, encoding='utf-8').read())
    blk = build(score, a.design, a.window)

    body = json.dumps({a.name: blk}, ensure_ascii=False, indent=2)
    # 채점표의 들여쓰기(정책 블록 안 4칸)에 맞춰 찍는다
    lines = body.split('\n')[1:-1]          # 바깥 중괄호를 뺀다
    print('# 채점표 "%s" 블록 안에 붙인다 — **붙이는 것은 사람이 한다**' % a.key)
    print('#   어느 런을 읽는지가 코드에 숨으면 안 되므로 여기서 파일을 고치지 않는다.')
    print('#   붙인 뒤 sign_scoreboard.READINGS 도 고칠지 판단할 것.')
    print()
    for ln in lines:
        print('  ' + ln)
    print()
    got = [k for k in blk if k not in ('design', 'window')]
    hit = sum(1 for k in got if blk[k].get('hit') is True)
    miss = sum(1 for k in got if blk[k].get('hit') is False)
    print('# 지표 %d개 · 적중 %d · 빗나감 %d · 미판정 %d'
          % (len(got), hit, miss, len(got) - hit - miss))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
