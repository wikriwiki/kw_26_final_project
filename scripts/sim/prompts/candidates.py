"""소비행동 프롬프트 후보 — 기전 중립판.

정책마다 프롬프트를 따로 두면 새 정책에서 어느 프롬프트가 좋은지 알 수 없다.
소비행동 프롬프트는 **정책과 무관한 사람의 성질**이어야 하고, 정책·사회배경은
그 사람에게 들어가는 입력이어야 한다.

현행 p012 의 daily_propensity 블록은 캐시백 전용 문구가 박혀 있다("캐시백은 지금
쓸 돈을 늘려주지 않는다", "문턱까지 남은 금액과 남은 일수"). 이것이 정확히 1:1
결합이다. 여기서는 그 자리를 기전 중립 문장으로 바꾼다.

## 후보 집합은 **미리 못박는다**

실효 자유도는 글자 수가 아니라 **시도한 후보 수**다. 6개를 먼저 적어 두고 그 안에서
고른다. 단계별로 후보를 추가하면 그만큼 자유도가 늘어난다는 점을 기록한다.
(docs/GENERALIZATION_METHOD.md §4 ③)

| 후보 | 축 | 무엇을 시험하는가 |
|---|---|---|
| v1 | 기준선 | 정책 = 예산·제약 중 하나 (P010 원형에 가장 가깝다) |
| v2 | 시점 | 미뤄둔 일이 앞당겨지는 축을 주면 달라지는가 |
| v3 | 대체 | 못 하게 된 일을 대체·연기·포기로 푸는 축 |
| v4 | 시점+대체 | 두 축이 함께 있을 때 |
| v5 | 조건 재정의 | 정책을 "오늘의 조건 변화"로 명시하고 무엇·언제·어디서를 함께 |
| v6 | 최소 (하한 대조군) | 정책 해석 지시를 **아예 빼면** 어떻게 되는가 |

**v6 이 대조군인 이유**: v4 가 v6 보다 나을 때만 추가 문장이 값을 한다. 비슷하면
그 문장들은 아무 일도 하지 않은 것이고, 그 사실을 알아야 과적합을 피한다.

기전별 판단 원칙(지갑/캐시백/영업제한/인원제한/할인)은 여기 넣지 않는다 —
`mechanisms/` 레지스트리가 활성 기전에 맞춰 붙인다.
"""
from __future__ import annotations

import os

# 교체 대상 — p012.SYSTEM_PROMPT 안의 이 구간을 후보 블록으로 바꾼다.
ANCHOR_HEAD = "[소비성향 daily_propensity — 최상위 필드]"
ANCHOR_TAIL = "단가와 결제수단은 Stage2가 결정하므로 여기서는 소비 의향만 표현한다."

_OPEN = """[소비성향 daily_propensity — 최상위 필드]
최상위에 오늘의 **소비성향 `daily_propensity` (0~1)** 를 출력한다. 금액이 아니라 오늘 소비하려는
성향의 비율이다. 어제 컨디션, 개인 잔액, 오늘의 조건, 일정, 실제 생활 필요를 함께 고려한다."""

_BASE = """정책은 오늘 쓸 수 있는 예산과 제약 중 하나다. 정책이 있다는 이유만으로 소비를 새로
만들지 않는다. 오늘 무엇이 필요한지는 평소 습관과 지금 상황에서 나온다."""

_TIMING = """할 일에는 오늘 꼭 해야 하는 것과 미뤄도 되는 것이 있다. 조건이 달라지면 그 순서가
바뀔 수 있다 — 미뤄두었던 일이 앞당겨지기도 하고, 하려던 일이 뒤로 밀리기도 한다."""

_SUBSTITUTE = """하려던 일을 할 수 없게 되면 셋 중 하나를 한다 — 대신할 것을 찾거나, 시간이나 장소를
옮기거나, 그만둔다. 무엇을 고를지는 그 일이 얼마나 필요했는지에 달렸다."""

_CONDITION = """정책은 오늘의 조건을 바꾼다. 쓸 수 있는 돈이 늘기도 하고, 같은 돈으로 더 살 수 있게
되기도 하고, 할 수 있는 일이 줄기도 한다. 먼저 오늘 무엇이 달라졌는지 보고, 그것이 내가
하려던 일에 닿는지 판단한다. 닿지 않으면 평소와 같다.
닿는다면 무엇을·언제·어디서 할지가 달라질 수 있다. 미뤄두었던 일이 앞당겨지기도 하고,
하려던 일이 뒤로 밀리기도 하고, 같은 일을 다른 곳에서 하기도 한다."""

_CLOSE = """얼마나 달라지는지는 사람마다 다르다 — 미뤄둔 일이 얼마나 쌓였는지, 평소 씀씀이가
어땠는지, 통장에 여유가 있었는지에 달렸다. 다만 조건이 달라졌다는 사실만으로 기계적으로
올리거나 내리지는 않는다. 실제 용건이 있을 때 달라진다.
단가와 결제수단은 Stage2가 결정하므로 여기서는 소비 의향만 표현한다."""

_CLOSE_MIN = """단가와 결제수단은 Stage2가 결정하므로 여기서는 소비 의향만 표현한다."""

_BLOCKS: dict[str, list[str]] = {
    "v1": [_OPEN, _BASE, _CLOSE],
    "v2": [_OPEN, _BASE, _TIMING, _CLOSE],
    "v3": [_OPEN, _BASE, _SUBSTITUTE, _CLOSE],
    "v4": [_OPEN, _BASE, _TIMING, _SUBSTITUTE, _CLOSE],
    "v5": [_OPEN, _CONDITION, _CLOSE],
    "v6": [_OPEN, _CLOSE_MIN],
}

NAMES = tuple(sorted(_BLOCKS))


def block(name: str) -> str:
    """후보 이름 → daily_propensity 블록 전문."""
    parts = _BLOCKS.get(name)
    if parts is None:
        raise KeyError(f"알 수 없는 후보: {name} (가능: {NAMES})")
    return "\n".join(parts)


def make_system_prompt(name: str) -> str:
    """p012 SYSTEM_PROMPT 에서 daily_propensity 블록만 후보로 교체한다.

    나머지 본문(카테고리 어휘·anchor 규칙·출력 형식·예시)은 손대지 않는다 —
    바꾸는 것은 '정책이 소비에 어떻게 닿는가' 한 블록뿐이다.
    """
    from . import p012
    s = p012.SYSTEM_PROMPT
    a = s.index(ANCHOR_HEAD)
    b = s.index(ANCHOR_TAIL, a) + len(ANCHOR_TAIL)
    return s[:a] + block(name) + s[b:]


def active() -> str:
    return (os.environ.get("EXP_PROMPT_CANDIDATE") or "v1").strip()


# =========================================================
# 자체 점검 — 후보별 길이와 차이
# =========================================================
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    base = len(block("v1"))
    print("후보 6개 — daily_propensity 블록")
    for n in NAMES:
        b = block(n)
        print(f"  {n}  {len(b):>5}자  (v1 대비 {len(b)-base:+5}자)")
    print()
    print("=== v5 전문 ===")
    print(block("v5"))
