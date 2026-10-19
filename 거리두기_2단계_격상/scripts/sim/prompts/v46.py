"""후보 v46 — v45 에서 시간 규칙의 두 어긋남을 없앤다. 관문 결과가 지목한 자리다.

v41 이 사다리를 재 줬다.

    v5   131/192  68.2%
    v40  141/192  73.4%   캐시백 오염 제거   +5.2%p
    v42  165/192  85.9%   형식 고침 다섯    +12.5%p
    v45  171/192  89.1%   지갑형 어휘 제거   +3.1%p

남은 실패 21건 중 **10건이 time 하나**이고, 전부 간격이다(15분 7 · 10분 3).
빈 문자열은 0 — v42 의 ④ 가 완전히 잡았다. 그리고 time 만 틀린 응답이 10건이므로
이것만 고치면 181/192 = 94.3% 다.

나는 v42 를 만들 때 이 항목을 "프롬프트가 이미 말하고 있으니 반복하지 않는다"로
분류하고 남겨 뒀다. **분류가 틀렸다.** 있는 것은 같은 규칙이 아니라 다른 규칙이다.

    프롬프트   "이벤트 간 최소 체류 20분"      ← 얼마나 머무는가
    검사기     minute - previous < 20          ← 시작 시각이 얼마나 떨어졌는가

스무 분을 머물러도 다음 이벤트의 `time` 이 15분 뒤면 위반이다. 강조를 더할 일이
아니라 **검사되는 것을 적을 일**이다. ① 의 "식사 = 끼니" 모순과 같은 종류다.

그리고 세어 보다가 하나를 더 찾았다. **출력 형식의 예시가 스스로 그 규칙을 어긴다.**

    08:10 → 08:50 → 12:00 → 15:00 → 18:10 → 18:40 → 19:00 → 17:30 → 16:30 → 17:00 → 19:20
                                                                ↑ 되감긴다

한 `{"events": [...]}` 배열 안이다. 계약은 시각이 순증해야 한다고 요구하는데,
그것을 보여 주는 예시가 세 번 되감긴다. **이벤트 내용은 한 글자도 바꾸지 않고
순서만 시각순으로 놓는다.** 정렬하면 모든 간격이 20분 이상이 된다.

    08:10 · 08:50 · 12:00 · 15:00 · 16:30 · 17:00 · 17:30 · 18:10 · 18:40 · 19:00 · 19:20
    간격   40 · 190 · 180 ·  90 ·  30 ·  30 ·  40 ·  30 ·  20 ·  20

**둘 다 빼는 고침이다.** 문장을 더하는 것은 규칙 한 줄뿐이고, 그것도 이미 있던
문장을 검사되는 말로 바꿔 적는 것이다.
"""
from __future__ import annotations

import re

from .p012 import format_dawn_blocks  # noqa: F401  (본문 공통)
from .v45 import SYSTEM_PROMPT as _V45

DWELL_OLD = '- 이벤트 간 최소 체류 20분.'
DWELL_NEW = ('- 이벤트 간 최소 체류 20분. 그리고 **다음 이벤트의 `time` 은 앞 이벤트의 `time` '
             '보다\n  최소 20분 뒤**여야 한다 — 시각은 하루 동안 뒤로 가지 않는다.')

EVENT_RE = re.compile(r'\{"time":"(\d\d):(\d\d)"')


def _sort_example(text):
    """출력 형식 예시의 이벤트를 시각순으로 놓는다. 내용은 건드리지 않는다."""
    head = text.index('{"events": [')
    start = text.index('\n', head) + 1
    end = text.index('\n  ...\n', start)
    body = text[start:end]
    # 이벤트 하나는 `  {"time":...` 로 시작해 다음 그런 줄 직전까지다.
    marks = [m.start() for m in re.finditer(r'(?m)^  \{"time":', body)]
    if len(marks) < 2:
        raise ValueError('예시 이벤트를 찾지 못했다')
    marks.append(len(body))
    chunks = [body[a:b] for a, b in zip(marks, marks[1:])]

    def key(chunk):
        m = EVENT_RE.search(chunk)
        if not m:
            raise ValueError('이벤트에서 시각을 읽지 못했다')
        return int(m.group(1)) * 60 + int(m.group(2))

    return text[:start] + ''.join(sorted(chunks, key=key)) + text[end:]


def build() -> str:
    s = _V45
    if DWELL_OLD not in s:
        raise ValueError('v45 본문이 바뀌었다 — 체류 규칙 줄을 찾지 못했다')
    s = s.replace(DWELL_OLD, DWELL_NEW, 1)
    return _sort_example(s)


SYSTEM_PROMPT = build()
