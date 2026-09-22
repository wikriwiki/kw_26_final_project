"""v50 판정기 — 등록된 규칙을 **결과가 나오기 전에** 코드로 고정한다.

두 팔의 채점 파일이 다 생기면 이 스크립트가 판정을 낸다. 사람이 그때 가서
지표를 고르거나 기준을 손보는 일을 막는 것이 목적이다.

    python scripts/report/v50_verdict.py --dir <두 score_v50_*.json 이 있는 폴더>

등록된 규칙은 `experiments/v50/prompt_v50.md` 에 있고 여기 그대로 옮긴다.

    주 지표   v45 의 DS-1·DS-2 가 v5 보다 실측에 가깝다
              |시뮬 − 실측| 이 **두 지표 모두에서** 줄어든다
    부 지표   DS-3 순위가 v45 에서도 유지된다
    결정      v45 가 더 가깝다 → 거시 현행을 v45 로 올린다
              v5 가 더 가깝다  → v5 유지. v45 는 관문용으로만 남는다
              비슷하다        → v5 유지. 바꿀 근거가 없다

그리고 `experiments/v50/v5_replication.md` 가 **결과를 보기 전에** 적어 둔
읽는 법을 함께 적용한다.

    DS-1 이 1%p 넘게 움직이면     프롬프트의 것으로 읽는다
    DS-2 가 13%p 안에서 움직이면  런으로도 설명되므로 판정 근거로 쓰지 않는다
    DS-3 순위가 뒤집히면          두 런 모두 적중했던 지표이므로 읽는다
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 정답지 — 서울연구원 정책리포트 322호. 폴더의 평가항목.md 에 원문 위치가 적혀 있다.
TRUTH = {'DS-1': -14.1, 'DS-2': +4.2}

# 런 간 이동. 같은 v5 를 같은 창에서 두 번 재 얻은 값이다(v5_replication.md).
RUN_SHIFT = {'DS-1': 0.2, 'DS-2': 13.1}

PRIMARY = ('DS-1', 'DS-2')


def pct(block, ind):
    """채점 파일에서 그 지표의 퍼센트 변화를 꺼낸다."""
    for r in block.get('results', []):
        if r.get('id') != ind:
            continue
        m, b = r.get('mean'), r.get('base')
        if isinstance(m, (int, float)) and isinstance(b, (int, float)) and b:
            return 100.0 * m / b
    return None


def rank_hit(block):
    for r in block.get('results', []):
        if r.get('id') == 'DS-3':
            return r.get('hit')
    return None


def load(d):
    out = {}
    for p in sorted(Path(d).glob('score_v50_*.json')):
        j = json.loads(p.read_text(encoding='utf-8'))
        name = j.get('label', p.stem).replace('v50_', '')
        out[name] = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/data/v50_answerkey')
    a = ap.parse_args()
    arms = load(a.dir)
    missing = [x for x in ('v5', 'v45') if x not in arms]
    if missing:
        print('아직 판정할 수 없다 — 채점 파일이 없는 팔: %s' % ', '.join(missing))
        print('있는 것: %s' % (', '.join(sorted(arms)) or '없음'))
        return 2

    print('# v50 판정 — 등록된 규칙 그대로')
    print()
    print('%-6s %10s %10s %10s %10s' % ('지표', '실측', 'v5', 'v45', '누가 가깝나'))
    closer, readable = {}, {}
    for ind in PRIMARY:
        t = TRUTH[ind]
        a5, a45 = pct(arms['v5'], ind), pct(arms['v45'], ind)
        if a5 is None or a45 is None:
            print('%-6s %10s %10s %10s   값 없음' % (ind, t, a5, a45))
            closer[ind] = None
            continue
        d5, d45 = abs(a5 - t), abs(a45 - t)
        who = 'v45' if d45 < d5 else ('v5' if d5 < d45 else '동률')
        closer[ind] = who
        moved = abs(a45 - a5)
        readable[ind] = moved > RUN_SHIFT[ind]
        print('%-6s %+9.1f%% %+9.1f%% %+9.1f%%   %-4s  (|차| %.1f → %.1f)'
              % (ind, t, a5, a45, who, d5, d45))
        print('%6s %s 두 팔의 이동 %.1f%%p · 런 간 이동 %.1f%%p → %s'
              % ('', ' ' * 32, moved, RUN_SHIFT[ind],
                 '읽는다' if readable[ind] else '**런으로도 설명된다 — 근거로 쓰지 않는다**'))
    r5, r45 = rank_hit(arms['v5']), rank_hit(arms['v45'])
    print()
    print('부 지표  DS-3 순위   v5 %s · v45 %s   → %s'
          % (r5, r45, '유지' if (r5 and r45) else '**뒤집혔다**'))

    print()
    print('## 등록된 결정')
    both = [closer.get(i) for i in PRIMARY]
    if all(x == 'v45' for x in both):
        verdict = '**v45 가 두 주 지표 모두에서 더 가깝다 → 거시 현행을 v45 로 올린다**'
    elif all(x == 'v5' for x in both):
        verdict = '**v5 가 두 주 지표 모두에서 더 가깝다 → v5 유지. v45 는 관문용으로만 남는다**'
    else:
        verdict = '**갈렸다 → 등록된 규칙은 "두 지표 모두"를 요구한다. v5 유지**'
    print('  ' + verdict)

    usable = [i for i in PRIMARY if readable.get(i)]
    print()
    print('## 이 판정이 기대는 증거')
    if not usable:
        print('  **어느 주 지표도 런 간 이동을 넘지 못했다.** 위 결정은 한 런의 관측이고,')
        print('  프롬프트의 성질로 적을 수 없다. 재현이 필요하다.')
    else:
        print('  런 간 이동을 넘은 지표: %s' % ', '.join(usable))
        print('  넘지 못한 지표: %s' % (', '.join(i for i in PRIMARY if not readable.get(i)) or '없음'))
    print()
    print('  규칙대로 **이 결과를 보고 v45 를 고치지 않는다.** 일회 평가다.')
    print('  크기 검증이 아니다 — audit_2026_09_20 이 대상·기간·결과·분모·대조군이')
    print('  맞을 때만 배수를 비교하라고 했고, 원문 창은 40주이고 우리 창은 이틀이다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
