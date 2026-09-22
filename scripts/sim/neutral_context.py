"""Experimental context renderer: factual state, without spending-pace framing.

Historical provider and completed frozen inputs stay unchanged. Policy-specific
accounting belongs here, not in the universal citizen system instruction.
"""
from datetime import date
import json
import math


def initial_state(persona, today, eligible_monthly_anchor, grant=None):
    wd = float(persona.get('daily_wd') or persona.get('daily_we') or 0)
    we = float(persona.get('daily_we') or wd)
    if not all(math.isfinite(x) and x >= 0 for x in [wd, we, eligible_monthly_anchor]):
        raise ValueError('Invalid initial spending anchors')
    monthly_total = int((wd * 5 + we * 2) / 7 * (today.day - 1))
    eligible = int(eligible_monthly_anchor / 30 * (today.day - 1))
    if eligible > monthly_total:
        raise ValueError('Eligible cumulative spending exceeds total cumulative spending')
    state = {'balance': int(wd * 39), 'month_spent': monthly_total,
             'sangsaeng_month_spent': eligible, 'energy': .7, 'mood': .5, 'fatigue': .3, 'yest_sat': .6}
    if grant is not None:
        pid, amount = grant
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError('Invalid currently available grant')
        state.update(grant_received={pid: amount}, grant_remaining={pid: amount})
    return state


def render(blocks, *, today, day_type, zones, cashback_status=None):
    clean = dict(blocks)
    clean['persona'] = clean['persona'].replace('평일 재택 ', '평일 집 체류 ').replace('주말 재택 ', '주말 집 체류 ')
    clean['policy'] = '\n'.join(s for s in clean['policy'].splitlines() if not s.startswith('- 판단 원칙:'))
    if cashback_status is not None:
        if cashback_status['eligible_month_spent_won'] > cashback_status['total_month_spent_won']:
            raise ValueError('Contradictory cumulative spending')
        clean['policy'] = json.dumps(cashback_status, ensure_ascii=False, indent=2)
    clean['zones'] = '\n'.join(s for s in clean['zones'].splitlines() if not s.startswith('평일엔 주로 생활권'))
    sections = [('현재 활성 정책의 조건', 'policy_facts'), ('사회 배경', 'environment'), ('시민 정보', 'persona'),
                ('개인별 정책 상태', 'policy'), ('장소 후보', 'zones'), ('오늘 시작 상태', 'state'),
                ('과거 방문 기억', 'memory'), ('오늘 예정된 약속', 'appointment'), ('지인', 'social'), ('알고 있는 장소', 'knows_poi')]
    text = '\n\n'.join('## ' + label + '\n' + (clean.get(key) or '(정보 없음)') for label, key in sections)
    text += f'\n\n## 오늘\n날짜: {today.isoformat()}. {day_type}. 오늘의 계획만 작성한다.'
    text += '\n외출 anchor 허용값: ' + ', '.join(json.dumps('zone:' + str(z)) for z in zones)
    return text
