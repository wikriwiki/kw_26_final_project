"""v53q — v53p 에서 모델이 따라 쓰던 값·지어낸 규범을 뺀 판 (2026-10-07 저녁).

v53p 본런 정책 전 주(P012·P013 각 4일, 2,000명) 검수:
- 출력 예시의 07:00 기상·08:30 출근·12:00 점심·sub_category 한식·zone:11680111 을 그대로 따라 했다
  (기상 시각 둘뿐, 출근 73~76% 08:30, 외식 98.5% 한식, 예시 동 코드를 엉뚱한 사람이 20건 사용).
- '좋은 예' 블록의 식당 이름(두부마을찬)을 '어제 먹은 곳'으로 지어냈고(400여 건, 실제 방문 0),
  미용실 예시 문구를 미용실 일정의 92% 가 그대로 썼고, 드라이기 정책 예시는 '정책이면 가전'으로 이끄는 방향 유도였다.
- 출처 없는 빈도·개수 규범(평일 6~10개, 주말 외출 두세 번, 한 달 한두 번 의원…)이 박혀 있었다.
- 공휴일(어린이날, 화)에 직장인 57% 가 출근했다 — 규칙이 '평일 + 직장 있음'만 말했다.
예시는 자리표시로만, 규범 숫자는 빼고, 공휴일·재택의 정의만 사실로 적는다. 방향을 지시하는 문장은 넣지 않는다.
"""
from __future__ import annotations

from .p012 import format_dawn_blocks  # noqa: F401
from .v53p import SYSTEM_PROMPT as _V53P

STAGE2_NEUTRAL = True


def _swap(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"v53p text changed; expected one exact span: {old[:40]!r}")
    return text.replace(old, new)


def _cut(text: str, start: str, end: str, new: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"v53p block changed: {start[:40]!r}")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before + new + end + after


def build() -> str:
    t = _V53P
    t = _swap(t, "- 하루 이벤트 수: **평일 6~10개**, 주말/공휴일 4~8개.\n", "")
    t = _swap(t, "- 평일 + 직장 있음: anchor='workplace' 체류가 09~18시 사이 누적 4시간 이상.\n",
              "- 직장이 있으면 일하는 날에는 근무 시간 동안 직장에 있다. 공휴일은 주말처럼 쉬는 날이다"
              "(그날도 일하는 직업이면 그 사정을 reasoning 에 적는다).\n"
              "- 재택근무는 직장이 아니라 집이다: anchor='residence', category='집'.\n")
    t = _cut(t, "[외출 — 자연스러운 일상 패턴 참고]\n", "- 외출 카테고리(commerce)는", "[외출]\n")
    t = _swap(t,
              "- 이런 일들은 매일은 아니지만 한 달을 놓고 보면 적지 않은 몫을 차지한다. 실제로 우리나라 성인은\n"
              "  한 달에 한두 번꼴로 의원·치과·한의원을 찾거나 약국에서 약을 타고, 한두 달에 한 번은 미용실에\n"
              "  가며, 사나흘에 한 번은 마트에서 장을 본다. 나이가 있거나 지병으로 약을 계속 타는 사람, 아이를\n"
              "  키우는 사람은 그보다 잦다.\n", "")
    t = _swap(t, "약속(appointment)은 그 시간대 우선(anchor=zone, pinned_poi).\n",
              "약속(appointment)은 그 시간대 우선(anchor=zone, pinned_poi). 약속이 끼니 때면 그 끼니는 약속 자리에서 먹는다"
              "(같은 끼니를 두 번 먹지 않는다).\n")
    t = _swap(t, "- appointment(약속) · rumor(어제·그제 소문/추천) · policy(오늘 활성인 정책 조건)\n",
              "- appointment(약속) · rumor(어제·그제 소문/추천) · policy(오늘 활성인 정책 조건)\n"
              "  (rumor 는 입력에 [rumor] 기억이 있을 때만, appointment 는 오늘 약속이 입력에 있을 때만 쓴다)\n")
    t = _swap(t, "- 약속 진입 시 상대 agent_id와 약속 잡힌 사유 명시.\n- 소문 따라간 경우 출처 agent와 topic 명시.\n",
              "- 약속이면 누구와 왜 잡은 약속인지, 들은 이야기를 따라갔으면 누구에게서 무엇을 들었는지 적는다.\n")
    t = _cut(t, "**좋은 예** (같은 카테고리라도 잔상·컨디션·정책이 사고 흐름을 만든다):\n", "trigger enum",
             "**형식** (내용은 이 에이전트의 입력에서만 나온다):\n"
             '{"time":"<HH:MM>","anchor":"zone:<dong_code>","category":"<L1>","sub_category":"<세부 업종>","intent":"<할 일>",\n'
             ' "reasoning":"<입력에 실제로 있는 기억·컨디션·약속·정책 조건 중 무엇이 오늘 이 일을 하게 했는지>","trigger":"<아래 enum 중 하나>"}\n\n')
    t = _cut(t, "예시 (형식만 보여 준다. 실제 시간·장소·업종·이유는 이 에이전트의 오늘 사정에서 나온다. dong_code는 위 zone 후보의 실제 숫자):\n",
             "※ 이 예시는 출력 형식만 보여 준다.",
             "출력 형식 (자리표시만 있다. 시간·장소·업종·개수·이유는 모두 이 에이전트의 오늘 사정에서 나온다):\n"
             '{"events": [\n'
             '  {"time":"<HH:MM>","anchor":"residence|workplace|zone:<dong_code>","category":"<L1>","sub_category":"<세부 업종, 외출일 때>","intent":"<할 일>",\n'
             '   "reasoning":"<이유>","trigger":"<enum>"},\n'
             "  ...\n"
             " ],\n"
             ' "daily_propensity": <0~1 사이 숫자>, "grant_kept_share": <0~1 사이 숫자>}\n\n')
    return t


SYSTEM_PROMPT = build()
