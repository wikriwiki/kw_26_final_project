"""후보 v40 — v5 에서 캐시백 특화를 걷어낸 판. 형식 예시는 한 글자도 건드리지 않는다.

v5 는 p012 본문에서 `daily_propensity` 블록 하나만 갈아 끼운 것이고, 그 블록은 전체의
6.4% 다. **나머지 93.6% 를 그대로 물려받는다.** 그 안에 캐시백 문턱·10%·103% 와
"문턱까지 얼마 안 남아 이번 달에 사면 캐시백까지 받겠다" 는 구매 예시가 들어 있고,
거리두기·무정책·지원금 입력에도 똑같이 들어간다.

`docs/VALIDATION_ARCHITECTURE_V3.md` 가 이것을 1순위로 적었고,
`docs/VALIDATION_V3_RESULTS.md` 가 다음 후보의 설계 원칙을 적었다.

    "후속 후보는 정책 행동을 유도하는 구매 예시와 출력 형식을 설명하는 예시를
     구분해야 한다."

v10 은 그 구분 없이 **예시를 통째로 없앴고** 원문 계약 통과율이 66.7% 에서 7.3% 로
무너졌다. 장소 위반 167건이 전부 `zone:` 접두사 누락이었다. 그러니 이 후보는
형식을 지탱하는 것은 전부 남기고, 특정 정책의 유인을 설명하는 것만 뺀다.

    남긴다   anchor 규칙 · 카테고리 어휘 · 출력 형식 · 이벤트 JSON 예시의 구조
             v5 의 daily_propensity 블록 (2차 선별에서 v5 가 1위였다)
    뺀다     grant_kept_share 절 전체 — 자기보고 반사실이고, 아키텍처 문서가
             "v10에서는 생성하지 않는다. 자기보고 반사실을 인과효과 측정값으로
             쓰지 않는다" 고 적었다. 엔진에서도 진단용이다
             정책 사실을 캐시백 어휘로 적은 줄
             예시 하나의 reasoning 이 문턱을 넘기려고 사는 장면을 연기하는 것

**전제.** 캐시백의 핵심 사실("문턱을 못 넘기면 이번 달 혜택이 사라진다")이 지금은
이 오염된 본문을 통해서만 모델에게 닿는다. 런타임 정책 블록은 정책 본문을 280자에서
자르는데 P012 본문은 358자이고, 잘려 나가는 78자가 바로 그 사실이다. **절단을 먼저
고치지 않고 오염만 걷어내면 P012 는 유인을 잃는다.** 두 변경은 같이 가야 한다.

바꾼 자리는 아래 표로 고정한다. 표에 없는 글자는 v5 와 바이트 단위로 같다.
"""
from __future__ import annotations

from .candidates import make_system_prompt
from .p012 import format_dawn_blocks  # noqa: F401  (본문 공통)

# 걷어내는 절 — 머리글부터 다음 절 머리글 직전까지 통째로 뺀다.
DROP_SECTION_HEAD = '[캐시백이 없어도 했을 지출의 몫 grant_kept_share — 최상위 필드]'
DROP_SECTION_NEXT = '[출력 형식]'

# 바꿔 쓰는 줄 — (원문, 바꾼 글) 짝. 형식에 관한 것은 하나도 없다.
REWRITES = (
    (
        '- 정책 블록의 공통 사실과 개인별 적립 실적·문턱까지 남은 금액·남은 일수를 '
        '사실 그대로 읽는다.',
        '- 정책 블록의 공통 사실과 개인별 정책 상태(대상 여부·남은 혜택·조건 충족 현황)를 '
        '사실 그대로 읽는다.',
    ),
    (
        '"reasoning":"드라이기가 자꾸 꺼져 불편했지만 목돈이라 미뤄왔음. '
        '문턱까지 얼마 안 남아 이번 달에 사면 캐시백까지 받겠다 싶어 퇴근길에 들름.'
        '","trigger":"policy"}',
        '"reasoning":"드라이기가 자꾸 꺼져 불편했지만 목돈이라 미뤄왔음. '
        '오늘 정책 블록을 보니 이 업종에서 쓸 수 있다고 해서 퇴근길에 들름.'
        '","trigger":"policy"}',
    ),
    # 설명 절을 빼면서 필드는 남기면, 모델이 안내 없이 그 값을 지어낸다. 출력 예시와
    # 주석에서도 같이 뺀다. 엔진에서 선택 필드이고(default=None) 진단용이다.
    (
        '"daily_propensity": 0.72, "grant_kept_share": 0.8}',
        '"daily_propensity": 0.72}',
    ),
    (
        '( daily_propensity·grant_kept_share에 적힌 숫자도 마찬가지로 이 사람의',
        '( daily_propensity에 적힌 숫자도 마찬가지로 이 사람의',
    ),
)


def _drop_section(text: str, head: str, nxt: str) -> str:
    a = text.index(head)
    b = text.index(nxt, a)
    return text[:a] + text[b:]


def build() -> str:
    """v5 의 시스템 프롬프트에서 정책 특화만 걷어낸다."""
    s = make_system_prompt('v5')
    for old, new in REWRITES:
        if old not in s:
            raise ValueError('v5 본문이 바뀌었다 — 바꿔 쓸 줄을 찾지 못했다: %.40s' % old)
        s = s.replace(old, new)
    return _drop_section(s, DROP_SECTION_HEAD, DROP_SECTION_NEXT)


SYSTEM_PROMPT = build()
