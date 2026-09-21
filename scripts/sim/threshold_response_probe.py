"""문턱까지 남은 거리를 바꿔 보고, 계획이 그것을 읽는지 본다.

P012 는 여섯 지표 중 실측과 가장 멀다. 원인 후보로 관측 창 길이(v30)와 정책 본문
절단(v43)을 시험 중인데, 둘 다 **모델이 문턱을 쫓기는 한다**는 것을 전제한다.
그 전제를 확인한 적이 없다.

런타임은 사람마다 이렇게 계산해 준다.

    X  2분기 월평균 앵커        Z = X × 1.03  실적 문턱
    Y  적립업종 이번달 누적      W = max(0, Z − Y)  문턱까지 남은 거리

W 가 작으면 조금만 더 쓰면 캐시백 자격이 생기고, 크면 이번 달엔 글렀다. 사람이라면
W 에 따라 다르게 움직인다. **모델이 그러는지는 재 본 적이 없다.**

이 탐침은 동결된 파일럿 맥락에서 **Y 하나만 바꿔** W 격자를 만든다. Z 와 남은 일수는
고정하므로 "이 사람이 지금까지 얼마나 썼는가"만 달라진다. 페이스(W ÷ 남은 일수)는
바뀐 W 에 맞춰 다시 계산한다 — 안 그러면 앞뒤가 안 맞는 맥락을 주고 그 불일치를
재게 된다.

읽는 것은 둘이다.

    daily_propensity      최상위 소비성향
    적립업종 이벤트 수      계획에 담긴 적립 대상 방문 수

**순환 검증이 아니다.** W 는 이미 모델의 입력이고, 여기서는 그 입력을 모델이 쓰는지를
잰다. 측정 공식을 입력으로 새로 넣는 것이 아니다. 방향을 지시하는 문장도 없다 —
바뀌는 것은 숫자 하나뿐이다.

    python scripts/sim/threshold_response_probe.py \
        --frozen /data/validation_v3/pilot_registered/frozen_inputs.json \
        --out /data/threshold_probe --variant v5
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

# 문턱 줄. 세 숫자(누적·문턱·남은 거리)와 페이스가 서로 맞물려 있다.
LINE_RE = re.compile(
    r'- (?P<pid>P\d+): 적립업종 이번달 누적 (?P<spent>[\d,]+)원 / '
    r'2분기 월평균 약 (?P<anchor>[\d,]+)원 / (?P<pct>[\d.]+)% 문턱 (?P<threshold>[\d,]+)원'
)
REMAIN_RE = re.compile(
    r'문턱까지 (?P<remaining>[\d,]+)원 남음 — 적립업종에서 이만큼 더 쓰면 캐시백 자격 시작'
    r'(?: \(이번 달 (?P<days>\d+)일 남음 · 하루 평균 (?P<pace>[\d,]+)원 페이스\))?'
)

# 사전등록한 격자 — 문턱 대비 비율이다. 절대 금액으로 하면 사람마다 문턱이 달라
# 같은 칸이 누구에게는 "거의 다 왔다"이고 누구에게는 "이번 달엔 글렀다"가 된다.
# 그리고 W 가 문턱보다 커지면 누적이 음수가 되어야 하는 모순이 생긴다(누적은 0 이 바닥).
#   0.02  거의 다 왔다       0.30  절반쯤
#   0.10  조금 남았다        0.60  멀다
#   1.00  한 푼도 안 썼다 (W = 문턱, 이 맥락에서 가능한 최대치)
GRID = (0.02, 0.10, 0.30, 0.60, 1.00)


def _num(text):
    return int(str(text).replace(',', ''))


def read_state(user):
    """맥락에서 X·Y·Z·W·남은 일수를 읽는다. 문턱 줄이 없으면 None."""
    a = LINE_RE.search(user)
    b = REMAIN_RE.search(user)
    if not a or not b:
        return None
    return {'pid': a.group('pid'), 'spent': _num(a.group('spent')),
            'anchor': _num(a.group('anchor')), 'threshold': _num(a.group('threshold')),
            'remaining': _num(b.group('remaining')),
            'days': int(b.group('days')) if b.group('days') else 0}


def set_fraction(user, fraction):
    """W 를 문턱의 `fraction` 배로 바꾼다 — Y 를 옮겨서. Z 와 남은 일수는 안 건드린다.

    누적·남은 거리·페이스를 따로 고치면 서로 어긋난 맥락이 되고, 그러면 W 의 효과가
    아니라 앞뒤가 안 맞는 글에 대한 반응을 재게 된다. 그래서 셋을 함께 다시 쓴다.
    """
    st = read_state(user)
    if st is None:
        raise ValueError('문턱 줄을 찾지 못했다')
    if not 0.0 <= fraction <= 1.0:
        raise ValueError('비율은 0~1 이어야 한다 — W 가 문턱을 넘으면 누적이 음수가 된다: %r'
                         % fraction)
    return _rewrite(user, st, int(round(st['threshold'] * fraction)))


def _rewrite(user, st, target):
    target = max(0, int(target))
    spent = max(0, st['threshold'] - target)
    out = user.replace('적립업종 이번달 누적 %s원' % format(st['spent'], ',d'),
                       '적립업종 이번달 누적 %s원' % format(spent, ',d'), 1)
    old = '문턱까지 %s원 남음' % format(st['remaining'], ',d')
    new = '문턱까지 %s원 남음' % format(target, ',d')
    if old not in out:
        raise ValueError('남은 거리 문구를 찾지 못했다')
    out = out.replace(old, new, 1)
    if st['days'] > 0:
        pace_old = '하루 평균 %s원 페이스' % format(
            int(round(st['remaining'] / st['days'])), ',d')
        pace_new = '하루 평균 %s원 페이스' % format(int(round(target / st['days'])), ',d')
        if pace_old in out:
            out = out.replace(pace_old, pace_new, 1)
    return out


def build(frozen, grid=GRID, case='cashback', arm='on'):
    """격자 칸들. 한 시민당 len(grid) 개."""
    cells = [c for c in frozen['cells'] if c['case'] == case and c['arm'] == arm]
    out = []
    for c in cells:
        st = read_state(c['user'])
        if st is None:
            continue
        for f in grid:
            out.append({'aid': c['aid'], 'case': c['case'], 'arm': c['arm'],
                        'date': c.get('date'), 'fraction': f,
                        'target_remaining': int(round(st['threshold'] * f)),
                        'zones': c.get('zones'), 'user': set_fraction(c['user'], f)})
    return out


def monotone(points):
    """(비율, 값) 목록이 문턱이 멀어질수록 줄어드는가. 같은 값은 허용한다."""
    ordered = [v for _, v in sorted(points)]
    return all(a >= b for a, b in zip(ordered, ordered[1:]))


def spread(points):
    """가장 가까울 때와 가장 멀 때의 차이. 0 이면 W 를 안 읽은 것이다."""
    if not points:
        return 0.0
    ordered = [v for _, v in sorted(points)]
    return ordered[0] - ordered[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--grid', default=','.join(str(x) for x in GRID),
                    help='문턱 대비 비율. 0~1.')
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    frozen = json.loads(Path(args.frozen).read_text(encoding='utf-8'))
    grid = tuple(float(x) for x in args.grid.split(','))
    cells = build(frozen, grid)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    io.open(out / 'cells.json', 'w', encoding='utf-8', newline='\n').write(
        json.dumps({'grid': list(grid), 'cells': cells}, ensure_ascii=False, indent=1))
    print('시민 %d · 격자(문턱 대비) %s · 칸 %d'
          % (len({c['aid'] for c in cells}), list(grid), len(cells)))
    print('wrote', out / 'cells.json')
    if args.prepare_only:
        return 0
    print('호출은 run_v44_threshold_probe.sh 가 한다 — 이 스크립트는 맥락만 만든다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
