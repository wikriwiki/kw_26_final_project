"""v5 + **동네 밖 몫** 한 항목. v5online 의 질문 범위를 바로잡은 변형.

## 왜 v5online 으로는 안 되는가 — 교정 탐침이 알려 줬다

v5online 은 "배송으로 받는 몫" 을 물었고 80건에서 **평균 0.14** 가 나왔다
(응답률 100%). 그런데 회계가 걷어내는 몫은 **0.7465** 다. 둘은 같은 양이 아니다.

`ELIGIBLE_SHARE_SEOUL = 0.2535` 는 서울시 상권분석의 '생활밀착 63업종' 매출
비율이고, 그 목록에 **백화점·대형마트·할인점·면세점이 없다**. 즉 회계가 걷어내는
0.7465 는 배송만이 아니라 **동네 가게가 아닌 모든 곳**이다. 코드 주석도 그렇게
적혀 있다 — "하루 지출이 전부 동네 가게 계산대에서 나가지는 않는다".

좁은 답(0.14)으로 넓은 상수(0.7465)를 흔들면 레버의 천장이 **+16.3%** 로 눌린다.
메워야 할 간격이 18.92%p 이므로 **천장이 간격보다 작다** — 돌려 봐야 못 메운다.
질문의 범위를 회계가 걷어내는 것과 맞춘다.

## 유도하지 않는다

블록은 **매장 종류**만 말한다. 정책·쿠폰·캐시백·적립·제외 같은 낱말도, 방향을
지시하는 문장도, 수치도 없다(시험이 막는다). 정책 사실은 이미 맥락에 있고,
거기서 무엇을 추론하는지가 재려는 것이다.
"""
from __future__ import annotations

from .p012 import format_dawn_blocks  # noqa: F401  (본문 공통)
from .v5 import SYSTEM_PROMPT as _V5

_ANCHOR_AT = "[출력 형식]"
_EXAMPLE_AT = '"daily_propensity": 0.72, "grant_kept_share": 0.8}'
_NOTE_AT = "( daily_propensity·grant_kept_share에 적힌 숫자도"

OFFSITE_BLOCK = """[동네 가게가 아닌 곳의 몫 online_share — 최상위 필드]
위 events 에 적은 지출을 하나씩 되짚어, 그중 **동네의 작은 가게가 아닌 곳**에서 나가는 몫이
얼마나 되는지를 `online_share` (0~1)로 출력한다. 전부 동네 가게에서 쓴다면 0, 전부 그 바깥이면 1이다.
대형마트·창고형 매장·백화점·아울렛·면세점처럼 큰 매장에서 쓰거나, 집에서 주문해 배송으로 받는
지출이 여기 해당한다. 동네 식당·카페·미용실·편의점·개인 슈퍼·학원·병원처럼 걸어서 가는 곳은 아니다.
사람마다 다르다 — 한 번에 몰아서 크게 사는 편인지, 집 앞에서 조금씩 사는 편인지, 큰 매장이
가까운지, 무겁거나 부피가 큰 물건인지, 평소 어떻게 사 왔는지에 달렸다.
오늘 적은 지출의 성격과 그 사람의 형편을 보고 판단한다.

"""


def _build() -> str:
    s = _V5
    if _ANCHOR_AT not in s:
        raise RuntimeError("v5 본문에서 [출력 형식] 을 못 찾았다 — 본문이 바뀌었다")
    s = s.replace(_ANCHOR_AT, OFFSITE_BLOCK + _ANCHOR_AT, 1)
    if _EXAMPLE_AT not in s:
        raise RuntimeError("v5 예시의 최상위 필드 줄을 못 찾았다 — 본문이 바뀌었다")
    s = s.replace(
        _EXAMPLE_AT,
        '"daily_propensity": 0.72, "grant_kept_share": 0.8, "online_share": 0.4}',
        1,
    )
    if _NOTE_AT in s:
        s = s.replace(
            _NOTE_AT, "( daily_propensity·grant_kept_share·online_share에 적힌 숫자도", 1
        )
    return s


SYSTEM_PROMPT = _build()
