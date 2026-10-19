"""후보 v48 — v45 에 마무리 예시만 넣는다. v46 의 정렬은 넣지 않는다.

v47 라운드가 셋을 한꺼번에 넣어 놓고 갈라 읽을 수 없게 만들었다. 결과는 이랬다.

    v45  356/384  92.7%   time 11 · zone  7 · explanation 6 · anchor_category 5
    v47  339/384  88.3%   time 29 · zone 11 · explanation 3 · anchor_category 1

**겨냥한 둘은 고쳐졌고 겨냥하지 않은 하나가 그보다 크게 망가졌다.** 고친 쪽은
마무리 예시가 한 일로 보이고(explanation 6→3 · anchor_category 5→1), 망친 쪽은
예시를 시각순으로 정렬한 것이 한 일로 보인다(붙은 쌍 4→7 · 간격 위반 11→25).

그래서 **마무리 예시만** 따로 넣는다. 정렬은 없다.

    넣는다   하루를 닫는 residence 이벤트 하나 (22:30 · 집 · reasoning 정확한 철자)
    안 넣는다  예시 정렬 · 시간 규칙 문장

예시가 계약을 어기는 것(되감기는 시각)은 그대로 남는다. **알면서 남긴다** —
그것을 고치려다 더 큰 것이 깨졌고, 지금은 무엇이 무엇을 움직이는지 한 번에
하나씩 확인하는 중이다.

v49 가 시간 규칙 문장만, 이 후보가 마무리 예시만 맡는다.
"""
from __future__ import annotations

import re

from .p012 import format_dawn_blocks  # noqa: F401  (본문 공통)
from .v45 import SYSTEM_PROMPT as _V45

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
    s = _V45
    if CLOSER_ANCHOR not in s:
        raise ValueError('v45 본문이 바뀌었다 — 예시 배열의 끝을 찾지 못했다')
    s = s.replace(CLOSER_ANCHOR, CLOSER_NEW, 1)
    a = s.index('{"events": [')
    ev = EVENT_RE.findall(s[a:s.index('\n ],\n "daily_propensity"', a)])
    if not ev:
        raise ValueError('예시 이벤트를 읽지 못했다')
    if ev[0][2] != 'residence' or ev[-1][2] != 'residence':
        raise ValueError('예시가 집에서 시작해 집에서 끝나지 않는다')
    return s


SYSTEM_PROMPT = build()
