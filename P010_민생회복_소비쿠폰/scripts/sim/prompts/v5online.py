"""v5 + **배송 몫 한 항목**. 잠겨 있던 레버를 되살린 변형 — diagnosis_04.

v5 의 본문을 **한 글자도 고치지 않고** `online_share` 블록 하나와 예시의 필드
하나만 덧댄다. 그래서 v5 를 고르면 지금까지와 완전히 같은 프롬프트가 나간다.

## 왜 이 항목인가

`stage1_intent.py:227` 에 필드와 배선이 이미 있는데 v5 프롬프트가 묻지 않아
계획 2,977건 전부 `None` 이었고, 회계는 상수 0.2535 로 떨어졌다. 폐기 근거는
"계층 구분 없이 0.2 근처로 균일" — 즉 **사람마다 안 갈린다**이지
**정책에 안 움직인다**가 아니다. 뒤의 질문은 측정된 적이 없다.

## 유도하지 않는다

블록에 정책도, 방향도, 목표 수치도 넣지 않는다. 묻는 것은 "오늘 산 것을 어디서
샀나" 하나다. 정책 사실은 이미 맥락에 들어 있고, 거기서 무엇을 추론할지는
에이전트의 몫이다 — 그 추론이 일어나는지가 바로 재려는 것이다.

예시의 `0.25` 는 스키마를 지키게 하려고 둔다. 예시 숫자로 쏠리는 것은 이미 아는
현상이고(daily_propensity 가 0.68 에 54% 몰림), 그래서 회계는 **수준이 아니라
편차만** 받는다(consumption.py 의 EXP_SPLIT_ANCHOR). 기준선 쏠림은 기준 런에서
잰 평균 M 으로 나눠 사라진다.
"""
from __future__ import annotations

from .p012 import format_dawn_blocks  # noqa: F401  (본문 공통)
from .v5 import SYSTEM_PROMPT as _V5

# 끼울 자리 표식 — v5 본문에 그대로 있는 문자열이라야 한다. 시험이 못 박는다.
_ANCHOR_AT = "[출력 형식]"
_EXAMPLE_AT = '"daily_propensity": 0.72, "grant_kept_share": 0.8}'
_NOTE_AT = "( daily_propensity·grant_kept_share에 적힌 숫자도"

ONLINE_BLOCK = """[배송으로 받는 몫 online_share — 최상위 필드]
위 events 에 적은 지출을 하나씩 되짚어, 그중 **가게에 가지 않고 집에서 주문해 배송으로 받는**
몫이 얼마나 되는지를 `online_share` (0~1)로 출력한다. 전부 직접 가서 산다면 0, 전부 주문해서
받는다면 1이다.
장 보러 나가는 대신 새벽배송을 시키거나, 옷·생필품·가전을 앱으로 주문하는 것이 여기 해당한다.
반대로 끼니를 밖에서 해결하거나 미용실·병원처럼 몸이 가야 하는 일은 배송이 될 수 없다.
사람마다 다르다 — 나가기 번거로운 사정인지, 집 앞에 그 물건을 파는 곳이 있는지, 무겁거나 급한지,
평소 어떻게 사 왔는지에 달렸다. 오늘 적은 지출의 성격을 보고 판단한다.

"""


def _build() -> str:
    s = _V5
    if _ANCHOR_AT not in s:
        raise RuntimeError("v5 본문에서 [출력 형식] 을 못 찾았다 — 본문이 바뀌었다")
    s = s.replace(_ANCHOR_AT, ONLINE_BLOCK + _ANCHOR_AT, 1)
    if _EXAMPLE_AT not in s:
        raise RuntimeError("v5 예시의 최상위 필드 줄을 못 찾았다 — 본문이 바뀌었다")
    s = s.replace(
        _EXAMPLE_AT,
        '"daily_propensity": 0.72, "grant_kept_share": 0.8, "online_share": 0.25}',
        1,
    )
    if _NOTE_AT in s:
        s = s.replace(
            _NOTE_AT, "( daily_propensity·grant_kept_share·online_share에 적힌 숫자도", 1
        )
    return s


SYSTEM_PROMPT = _build()
