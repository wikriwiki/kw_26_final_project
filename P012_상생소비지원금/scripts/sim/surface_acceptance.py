"""State a wallet's usage rule where the policy's other facts already live.

The conditions block tells the model the wallet works "[쿠폰] 표시 POI에서만" - at POIs
carrying a mark that appears nowhere in the input. The rule that the payment stage
actually applies sits in `purchase_preview.wallet_acceptance`, buried inside the
candidate-overview JSON.

Cashback is the one mechanism whose terms are stated as plain numbers in the conditions
block, and it is the one with the widest on/off gap in reachability. That is an
observation across four mechanisms, not a test - this module exists so the test can be
run: move the rule, change nothing else.

Nothing is added here that the input did not already contain. No direction, no target,
no policy outcome. The dangling marker is replaced by the same claim in a form the
input supports.
"""
from __future__ import annotations

MARKER_CLAUSE = ' · [쿠폰] 표시 POI에서만 사용'
REGION_CLAUSE = ' · [지역] 표시 POI에서만 사용'
HEADING = '## 현재 활성 정책의 조건'
NEXT_HEADING = '## 사회 배경'


def acceptance_lines(preview: dict | None) -> list[str]:
    """One plain line per wallet, listing the places and activities that accept it."""
    out = []
    for entry in (preview or {}).get('wallet_acceptance') or []:
        anchors = list(entry.get('anchors') or [])
        acts = list(entry.get('activity_ids') or [])
        if not anchors or not acts:
            continue
        out.append(
            f"  {entry['wallet_id']} 사용처: {', '.join(anchors)} 에서의 "
            f"{', '.join(acts)} 에서만 받는다. 다른 장소·활동에서는 이 자금으로 결제할 수 없다.")
    return out


def surface(user: str, preview: dict | None) -> str:
    """Put the acceptance rule in the conditions block and drop the dangling marker.

    Returns the text unchanged when there is no wallet to describe, so the control and
    the treatment differ on exactly the cells where the rule exists.
    """
    lines = acceptance_lines(preview)
    if not lines:
        return user
    start = user.find(HEADING)
    if start < 0:
        raise ValueError('No conditions block to state the rule in')
    end = user.find(NEXT_HEADING, start)
    if end < 0:
        raise ValueError('Conditions block is not delimited')
    block = user[start:end]
    for dangling in (MARKER_CLAUSE, REGION_CLAUSE):
        block = block.replace(dangling, '')
    block = block.rstrip('\n') + '\n' + '\n'.join(lines) + '\n\n'
    return user[:start] + block + user[end:]
