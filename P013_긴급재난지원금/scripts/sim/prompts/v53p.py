"""v53p — v53n 에서 출력 예시 묶음만 바꾼 판 (2026-10-07).

v53n 의 예시 하루에는 의원('허리 통증 진료')·약국·미용실·학원비·아이 신발·전기밥솥·주간 장보기가 한꺼번에 들어 있었다.
'그런 날의 예시일 뿐'이라는 단서가 있었지만 모델이 그대로 따라 써서, 10명 시험 첫날 3명이 '허리 통증 진료'를 갔고
본런에서는 의원 방문이 4일에 44%, 같은 사람 사흘 연속이 11% 였다. 예시의 "daily_propensity": 0.72 도 숫자 기준값이 된다.
예시는 형식만 보여 주는 평범한 평일 하루로 바꾸고, 가끔 있는 일은 원칙 한 줄로만 둔다. 다른 문장은 바꾸지 않는다.
"""
from __future__ import annotations

from .p012 import format_dawn_blocks  # noqa: F401
from .v53n import SYSTEM_PROMPT as _V53N

STAGE2_NEUTRAL = True

_START = "예시 (실제 dong_code는 페르소나 블록 참조 / reasoning은 페르소나 → 행동 직결이 아닌 살아있는 흐름):\n"
_END = "※ 하루 일정(events)을 먼저 다 적은 뒤, 그 구체적인 지출을 눈앞에 두고 나머지 값들을 판단해 적는다.\n"

_NEW = """예시 (형식만 보여 준다. 실제 시간·장소·업종·이유는 이 에이전트의 오늘 사정에서 나온다. dong_code는 위 zone 후보의 실제 숫자):
{"events": [
  {"time":"07:00","anchor":"residence","category":"집","intent":"기상",
   "reasoning":"평소 일어나는 시간에 일어남.",
   "trigger":"none"},
  {"time":"08:30","anchor":"workplace","category":"직장","intent":"출근",
   "reasoning":"평일이라 출근함.",
   "trigger":"lifestyle"},
  {"time":"12:00","anchor":"zone:11680111","category":"식사","sub_category":"한식","intent":"점심",
   "reasoning":"점심때가 되어 직장 근처에서 먹음.",
   "trigger":"lifestyle"},
  {"time":"18:30","anchor":"residence","category":"집","intent":"귀가",
   "reasoning":"일을 마치고 집으로 돌아옴.",
   "trigger":"none"},
  {"time":"23:30","anchor":"residence","category":"집","intent":"취침",
   "reasoning":"하루를 마무리하고 잠자리에 듦.",
   "trigger":"none"},
  ...
 ],
 "daily_propensity": <0~1 사이 숫자>, "grant_kept_share": <0~1 사이 숫자>}

※ 이 예시는 출력 형식만 보여 준다. 오늘 하루의 내용은 이 사람의 직업·생활·어제 상태·기억·약속에서 나온다.
※ 병원·약국·미용실·학원비·큰 물건 구입·장보기 같은 일은 매일 하는 일이 아니다. 그날 실제로 그럴 사정이 있을 때만 넣는다.
"""


def build() -> str:
    if _V53N.count(_START) != 1 or _V53N.count(_END) != 1:
        raise ValueError("v53n example block changed; expected one exact span")
    before, rest = _V53N.split(_START, 1)
    _, after = rest.split(_END, 1)
    return before + _NEW + _END + after


SYSTEM_PROMPT = build()
