"""후보 v47 — v46 에 하루를 닫는 예시를 넣는다. 네 번째 자기모순이다.

계약은 이렇게 요구한다.

    events[0].anchor == 'residence'  그리고  events[-1].anchor == 'residence'

그런데 **예시는 집에서 시작해서 마트에서 끝난다.**

    08:10 residence · 08:50 zone · … · 19:00 zone · 19:20 zone(마트)
                                                    ↑ 여기서 끝난다

시작은 보여 주고 마무리는 안 보여 준다. 모델은 계약을 지키려고 마무리를 **즉흥으로
지어내야 하고**, v41 에서 남은 실패가 정확히 그 자리에 모여 있다.

    'reason' 오타 3건   전부 하루 마지막 쪽 residence 이벤트 (취침·귀가)
    zone + 집 1건       intent="귀가 후 정리"

v5 에서 오타 23건을 셌을 때도 같았다 — 마지막 이벤트 10건, 나머지도 전부 뒤쪽,
intent 는 취침 9 · 귀가 3 · 귀가 및 휴식 3. **zone 이벤트 714개와 workplace 263개
에서는 한 번도 나지 않았다.** 즉흥으로 짓는 자리에서만 틀린다.

그래서 마무리를 예시로 보여 준다. 더하는 것은 이벤트 하나이고, 그 하나가
**필드 이름·앵커·업종·간격을 동시에 시연한다.**

    22:30 · residence · 집 · reasoning(정확한 철자) · trigger:none · 앞 이벤트와 190분

v10 이 예시를 없애 7.3% 로 무너진 것의 정확한 반대다 — 예시가 덮지 않던 자리를
예시로 덮는다.
"""
from __future__ import annotations

import re

from .p012 import format_dawn_blocks  # noqa: F401  (본문 공통)
from .v46 import SYSTEM_PROMPT as _V46

# `...` 자리 뒤에 하루를 닫는 이벤트를 놓는다. 앞 이벤트(19:20)와 190분 떨어져 있다.
CLOSER_ANCHOR = '  ...\n ],\n "daily_propensity": 0.72}'
CLOSER_NEW = (
    '  ...,\n'
    '  {"time":"22:30","anchor":"residence","category":"집","intent":"하루 마무리",\n'
    '   "reasoning":"장 본 것을 정리하고 내일 출근 준비까지 마친 뒤 자리에 누움. '
    '어제 fatigue 0.4 정도라 특별히 일찍 자야 할 이유는 없었지만 평소 리듬대로 마무리함.",\n'
    '   "trigger":"none"}\n'
    ' ],\n "daily_propensity": 0.72}')

EVENT_RE = re.compile(r'\{"time":"(\d\d):(\d\d)","anchor":"([^"]+)"')


def build() -> str:
    s = _V46
    if CLOSER_ANCHOR not in s:
        raise ValueError('v46 본문이 바뀌었다 — 예시 배열의 끝을 찾지 못했다')
    s = s.replace(CLOSER_ANCHOR, CLOSER_NEW, 1)
    # 넣고 나서 계약을 스스로 지키는지 확인한다. 예시가 규칙을 어기면 안 된다.
    a = s.index('{"events": [')
    ev = EVENT_RE.findall(s[a:s.index('\n ],\n "daily_propensity"', a)])
    if not ev:
        raise ValueError('예시 이벤트를 읽지 못했다')
    if ev[0][2] != 'residence' or ev[-1][2] != 'residence':
        raise ValueError('예시가 집에서 시작해 집에서 끝나지 않는다')
    mins = [int(h) * 60 + int(m) for h, m, _ in ev]
    if any(b - a2 < 20 for a2, b in zip(mins, mins[1:])):
        raise ValueError('예시 안에 20분 미만 간격이 있다')
    return s


SYSTEM_PROMPT = build()
