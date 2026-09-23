"""v5 + **결정이 어디서 일어나는지 한 줄**. 후보 2 — experiments/plan_channel/s1_ownership.md

## 왜 이 줄인가

후보 1(문턱 줄을 Stage2 로)이 기각됐다 — 같은 이벤트를 두고 금액만 다시 매기게
해도 안 달라진다(p=0.678). 그런데 실제 정책 반응은 **이벤트가 달라져서** 나온다
(수는 11.5 -> 11.5 그대로인데 금액 +9.76%).

그리고 지금 프롬프트에는 이런 줄이 있다.

    p012.py:93  "grant의 실제 결제수단과 금액은 Stage2가 결정한다."

지갑형 정책에서는 맞는 말이다. 그런데 **적립형에는 지갑이 없다.** 결제수단을
고를 것이 없고, Stage2 는 문턱을 모른다(후보 1 에서 확인). 그래서 이 줄은
적립형에서 **아무도 책임지지 않는 자리**를 만든다 — Stage1 은 금액을 Stage2 에
넘기고, Stage2 는 제도를 모른다.

덧대는 한 줄은 **그 자리를 가리킬 뿐**이다. 방향도 수치도 없다.

## 유도가 아닌 이유

"더 쓰라"고 하지 않는다. 적립형에 결제수단 선택이 없다는 **사실**과, 그래서
제도가 닿는 곳이 '무엇을 하기로 하는가' 라는 **구조**를 적는다. 그 사정을
어떻게 받아들일지는 기존 문장들(감쇠 포함)이 그대로 정한다 — 감쇠를 빼지 않는다.
"""
from __future__ import annotations

from .candidates import make_system_prompt
from .p012 import format_dawn_blocks  # noqa: F401  (본문 공통)
from .v5 import SYSTEM_PROMPT as _V5

_ANCHOR = ("- grant의 실제 결제수단과 금액은 Stage2가 결정한다. "
           "Stage1에서는 정책이 오늘 의도에 실제로 관련된 경우에만 reasoning과 trigger에 반영한다.")

OWN_LINE = (
    "\n- 적립(캐시백)형 정책은 지갑이 따로 없어 **고를 결제수단이 없다.** "
    "그래서 Stage2가 정할 몫도 없고, 이 제도가 오늘에 닿는 자리는 "
    "**무엇을 하기로 하는가** 하나뿐이다. 그 판단은 위의 기준을 그대로 따른다."
)


def _build() -> str:
    s = _V5
    if _ANCHOR not in s:
        raise RuntimeError("v5 본문에서 결제수단 줄을 못 찾았다 — 본문이 바뀌었다")
    return s.replace(_ANCHOR, _ANCHOR + OWN_LINE, 1)


SYSTEM_PROMPT = _build()
