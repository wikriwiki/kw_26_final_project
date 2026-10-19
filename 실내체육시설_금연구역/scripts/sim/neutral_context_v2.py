"""Facts-only renderer using explicit provider status rather than legacy hints.

The provider supplies observations and scenario assumptions separately. This
renderer does not decide eligibility, infer a desired behavior or calculate an
effect target. Legacy/frozen inputs are not rewritten.
"""
import json
from neutral_context import render as legacy_render


def render(blocks, *, today, day_type, zones, personal_policy_status):
    if not isinstance(personal_policy_status,list):raise ValueError('Explicit personal policy status list required')
    for row in personal_policy_status:
        if not isinstance(row,dict) or not row.get('id') or not isinstance(row.get('scenario_assumptions'),list):
            raise ValueError('Policy status must separate scenario assumptions')
    clean=dict(blocks)
    clean['policy']=json.dumps(personal_policy_status,ensure_ascii=False,indent=2) if personal_policy_status else '(입력한 활성 정책 없음)'
    clean['zones']='\n'.join(line.replace(' ← 주말 나들이·여가 등에 적합','') for line in clean.get('zones','').splitlines())
    return legacy_render(clean,today=today,day_type=day_type,zones=zones)
