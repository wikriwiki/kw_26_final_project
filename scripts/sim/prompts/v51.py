"""후보 v51 — 직장을 동 코드로 적는 것을 막는다. 한 줄만 고친다.

그리스디(temperature 0)에서 v45 가 남긴 23칸 중 **zone 9칸**의 정체를 원문에서
확인했다. 앞선 문서가 "허용 목록 밖 코드"라고 적어 둔 것은 **틀렸다.**
목록 밖 코드는 한 건도 없었다. 26개 이벤트 전부가 이 모양이다.

    anchor="zone:11470550"  category="직장"   intent="출근"
    anchor="zone:11470550"  category="직장"   intent="업무"
    anchor="zone:11470550"  category="직장"   intent="퇴근"

**동 코드는 허용 목록 안이고, 카테고리가 '직장'인 것이 위반이다.** 계약은
category='직장'을 anchor='workplace'에만 허용한다. 같은 칸에서 그 사람의
식사·카페·편의점 이벤트는 같은 동 코드로 전부 통과했다 — 모델은 동 코드를
제대로 골랐고, **근무를 가게처럼 적었을 뿐이다.**

9칸 중 6칸이 한 사람(50대 건설 프로젝트 매니저)이고, 나머지도 전부 직장이 있는
사람이다. 무작위로 흩어진 잡음이 아니라 **특정 문맥에서 매번 같은 실수**다.

왜 그러는가. 본문이 이미 세 군데에서 "workplace" 를 말한다.

    - "workplace": 직장 빌딩 내부에서만 일어나는 활동. category는 '직장'만.
    - anchor='workplace'일 때 category='직장'
    - 출근은 anchor='workplace' category='직장'

**적혀 있는데도 어긴다.** 그러니 규칙을 한 번 더 적는 것은 답이 아니다.
대신 바로 옆의 끌개를 본다 —

    - 직장 동 근처 외출(예: 점심 식당·퇴근길 카페) → zone:<work_dong_code>

이 줄이 '직장'과 'zone 코드'를 한 줄에서 묶는다. 모델이 여기서 직장→동 코드
결합을 가져가는 것으로 본다. 그래서 **그 줄에서** 경계를 긋는다.

고치는 자리는 하나다. 규칙을 더하지 않고, 이미 있는 줄에 그 줄이 무엇을
가리키는지 한 마디를 붙인다.

**형식은 건드리지 않는다.** 이벤트 JSON 예시 열넷, 출력 스키마, 나머지 anchor
규칙은 글자 하나 바뀌지 않는다.
"""
from __future__ import annotations

from .p012 import format_dawn_blocks  # noqa: F401  (본문 공통)
from .v45 import SYSTEM_PROMPT as _V45

REWRITES = (
    # 끌개가 있는 바로 그 줄에서 경계를 긋는다. '직장 동' 은 가게를 가리키는 말이지
    # 근무를 가리키는 말이 아니다.
    (
        '  - 직장 동 근처 외출(예: 점심 식당·퇴근길 카페) → zone:<work_dong_code>',
        '  - 직장 동 근처 외출(예: 점심 식당·퇴근길 카페) → zone:<work_dong_code>\n'
        '    (여기서 zone 이 가리키는 것은 **그 동에 있는 가게**다. 근무·회의·출퇴근 자체는\n'
        "     직장이 어느 동에 있든 anchor='workplace' 이고, 동 코드로 적지 않는다.)",
    ),
)


def build() -> str:
    s = _V45
    for old, new in REWRITES:
        if old not in s:
            raise ValueError('v45 본문이 바뀌었다 — 바꿔 쓸 줄을 찾지 못했다: %.40s' % old)
        s = s.replace(old, new, 1)
    return s


SYSTEM_PROMPT = build()
