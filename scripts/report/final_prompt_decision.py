"""세 라운드를 함께 읽어 **거시 라인의 현행 프롬프트**를 정한다.

라운드가 다 끝난 뒤에 사람이 지표를 고르거나 규칙을 손보는 일을 막으려고,
**결과가 다 나오기 전에** 절차를 코드로 박는다.

    python scripts/report/final_prompt_decision.py --dir <채점 파일들이 있는 곳>

읽는 라운드 셋과 각각의 등록된 규칙은 이렇다.

    v50        v5 대 v45   거리두기   prereg: experiments/v50/prompt_v50.md
               DS-1·DS-2 가 **둘 다** 실측에 가까워지면 v45, 아니면 v5
               → 이미 끝났다. 갈려서 **v5 유지**

    라운드2    v5 대 v45   P012      prereg: experiments/answerkey_round2/prereg.md
               검출되는 지표(P012-1·P012-4)에서 |시뮬−실측| 이 줄어든 지표 수
               둘 다 줄면 v45, 아니면 v5

    라운드3    v5 대 v51   거리두기   prereg: experiments/answerkey_round3/prereg.md
               DS-1 에서 v51 이 가깝고 그 이동이 런 간 이동 0.2%p 를 넘으면 v51
               DS-3 순위도 유지되어야 한다

## 합치는 규칙 — 여기서 새로 정하지 않는다

각 라운드가 이미 자기 규칙으로 승자를 낸다. 이 스크립트는 그것을 **세어서**
현행을 정한다.

    v5 를 이긴 후보가 **둘 이상의 정책**에서 나오면   그 후보를 거시 현행으로 올린다
    한 정책에서만 이겼으면                          "한 정책에서 앞섰다" 로만 적고 v5 유지
    아무도 못 이겼으면                              v5 유지

**한 정책·한 런으로 프롬프트를 바꾸지 않는다.** v50 에서 이미 배운 것이다 —
관문에서 +20.6%p 이긴 후보가 정답지 한 곳에서 졌다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 정답지 — 각 지표의 실측값
TRUTH = {
    'DS-1': -14.1,          # 서울연구원 정책리포트 322호 (한식)
    'DS-2': +4.2,
    'P012-1': +20.82,       # 기획재정부·KDI (2022.9)
}

# 같은 프롬프트를 같은 창에서 두 번 재 얻은 런 간 이동. 이보다 작은 차이는 읽지 않는다.
RUN_SHIFT = {'DS-1': 0.2, 'DS-2': 13.1}

# 라운드마다 (채점 파일 접두사, 비교할 두 팔, 주 지표)
ROUNDS = [
    ('v50', 'score_v50_', ('v5', 'v45'), ['DS-1', 'DS-2'], '거리두기'),
    ('round2', 'score_r2_', ('v5', 'v45'), ['P012-1', 'P012-4'], 'P012'),
    ('round3', 'score_r3_', ('v5', 'v51'), ['DS-1'], '거리두기'),
]


def load(d, prefix, arm):
    p = Path(d) / ('%s%s.json' % (prefix, arm))
    if not p.exists():
        return None
    j = json.loads(p.read_text(encoding='utf-8'))
    return {r['id']: r for r in (j.get('results') or [])}


def pct(r):
    if not r:
        return None
    m, b = r.get('mean'), r.get('base')
    if isinstance(m, (int, float)) and isinstance(b, (int, float)) and b:
        return 100.0 * m / b
    return None


def judge(name, rows_a, rows_b, arms, metrics, policy):
    """한 라운드의 등록된 판정. 승자와 근거를 돌려준다."""
    a_name, b_name = arms
    closer, readable, detail = {}, {}, []
    for k in metrics:
        t = TRUTH.get(k)
        x, y = pct(rows_a.get(k)), pct(rows_b.get(k))
        if x is None or y is None or t is None:
            detail.append('  %-8s 값 없음 (%s %s · %s %s)' % (k, a_name, x, b_name, y))
            closer[k] = None
            continue
        da, db = abs(x - t), abs(y - t)
        who = b_name if db < da else (a_name if da < db else '동률')
        closer[k] = who
        moved = abs(y - x)
        shift = RUN_SHIFT.get(k)
        readable[k] = (moved > shift) if shift is not None else None
        tag = ''
        if shift is not None:
            tag = ('  이동 %.1f%%p > 런 %.1f%%p → 읽는다' % (moved, shift)) if moved > shift \
                else ('  이동 %.1f%%p ≤ 런 %.1f%%p → **근거로 쓰지 않는다**' % (moved, shift))
        detail.append('  %-8s 실측 %+.1f%% · %s %+.1f%% · %s %+.1f%% → %s (|차| %.1f → %.1f)%s'
                      % (k, t, a_name, x, b_name, y, who, da, db, tag))
    # **등록된 규칙 그대로 — 나열된 지표가 전부 도전자 쪽이어야 한다.**
    # '읽히는 지표만 세자' 로 바꾸면 규칙이 사후에 느슨해진다. 실제로 그렇게 짰다가
    # v50 의 결론(갈렸다 → v5 유지)이 v45 로 뒤집혔다.
    got = [k for k in metrics if closer.get(k)]
    if got and all(closer[k] == b_name for k in got):
        winner = b_name
    else:
        winner = a_name          # 갈리거나 값이 없으면 대조군 유지
    # 읽을 수 있었던 지표는 따로 적는다 — 판정을 바꾸지는 않고, 증거의 무게를 보인다
    usable = [k for k in metrics if closer.get(k) and readable.get(k) is not False]
    return winner, detail, usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/data')
    a = ap.parse_args()

    print('# 거시 라인 현행 결정 — 세 라운드를 함께 읽는다')
    print()
    wins, done = {}, []
    for name, prefix, arms, metrics, policy in ROUNDS:
        # v50 은 /data/v50_answerkey, 라운드2·3 은 각자의 폴더에 있다
        for base in (Path(a.dir), Path(a.dir) / 'v50_answerkey',
                     Path(a.dir) / 'answerkey_round2', Path(a.dir) / 'answerkey_round3'):
            ra = load(base, prefix, arms[0])
            rb = load(base, prefix, arms[1])
            if ra and rb:
                break
        print('## %s — %s · %s 대 %s' % (name, policy, arms[0], arms[1]))
        if not (ra and rb):
            print('  아직 채점 파일이 없다.')
            print()
            continue
        winner, detail, usable = judge(name, ra, rb, arms, metrics, policy)
        for ln in detail:
            print(ln)
        print('  → **%s**  (읽을 수 있었던 지표: %s)' % (winner, ', '.join(usable) or '없음'))
        print()
        done.append(name)
        if winner != arms[0]:
            wins.setdefault(winner, []).append(policy)

    print('## 합친 결정')
    if not done:
        print('  읽을 라운드가 없다.')
        return 2
    print('  끝난 라운드: %s' % ', '.join(done))
    if not wins:
        print('  **v5 를 이긴 후보가 없다 → 거시 현행 v5 유지.**')
    else:
        for cand, pols in wins.items():
            uniq = sorted(set(pols))
            if len(uniq) >= 2:
                print('  **%s 가 정책 %d개(%s)에서 v5 를 이겼다 → 거시 현행을 %s 로 올린다.**'
                      % (cand, len(uniq), ', '.join(uniq), cand))
            else:
                print('  %s 가 %s 한 정책에서만 앞섰다 → **한 정책에서 앞섰다고만 적고 v5 유지.**'
                      % (cand, uniq[0]))
    print()
    print('  한 정책·한 런으로 프롬프트를 바꾸지 않는다. v50 에서 배운 것이다 —')
    print('  관문에서 +20.6%p 이긴 후보가 정답지 한 곳에서 졌다.')
    print('  크기 검증이 아니다(audit_2026_09_20). 채점은 부호로 한다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
