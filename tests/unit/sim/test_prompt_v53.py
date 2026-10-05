"""A shared prompt must not encode one policy's merchant rules or grant response."""
from __future__ import annotations

import re

from scripts.sim.prompts import get, v51, v53


def test_v53_removes_the_hidden_voucher_and_grant_defaults():
    old, new = v51.SYSTEM_PROMPT, v53.SYSTEM_PROMPT
    assert '프랜차이즈 직영점·온라인 주문에는 쓸 수 없다' in old
    assert 'grant의 실제 결제수단' in old
    assert '쓸 수 있는 목돈이 생겼을 때 사람들이 흔히 하는 일' in old
    for claim in ('프랜차이즈 직영점·온라인 주문에는 쓸 수 없다',
                  '생활 주변 가게 전반이다', 'grant의 실제 결제수단',
                  '쓸 수 있는 목돈이 생겼을 때 사람들이 흔히 하는 일',
                  'subsidy·regulation·facility·campaign'):
        assert claim not in new
    assert '정책 블록에 없는 허용·제외 업종' in new
    assert '무엇을 할지 정한다. 같은 일을 그대로 하거나' in new
    assert 'Stage1에서는 오늘의 의도와 일정만 정한다' in new


def test_v53_preserves_the_output_contract_and_does_not_leak_targets():
    old, new = v51.SYSTEM_PROMPT, v53.SYSTEM_PROMPT
    assert get('v53').SYSTEM_PROMPT == new
    assert len(re.findall(r'\{"time":', old)) == len(re.findall(r'\{"time":', new)) == 14
    for token in ('[출력 형식]', 'daily_propensity', 'anchor', 'workplace',
                  'trigger enum', '이벤트 간 최소 체류 20분'):
        assert token in new
    for leak in ('20.82', '14.1%', '6.957', '47,880',
                 '캐시백', '바우처', '쿠폰', 'grant_kept_share'):
        assert leak not in new


def test_v53_changes_only_registered_spans():
    restored = v53.SYSTEM_PROMPT
    for old, new in reversed(v53.REWRITES):
        assert new in restored
        restored = restored.replace(new, old, 1)
    assert restored == v51.SYSTEM_PROMPT
