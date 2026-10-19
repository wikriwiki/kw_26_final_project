# -*- coding: utf-8 -*-
"""날짜 → 사회 배경 dict.

`build_environment(env_id, day)` 가 dawn_context 의 `environment` 필드에 그대로
들어간다. 반환 형태는 `_format_environment` 의 계약과 같다.

    {"headline": str, "facts": [str, ...], "note": str | None}

행동 지시("더 써라"/"덜 써라")는 넣지 않는다. 상태만 제시하고 판단은 에이전트가 한다.
"""
from __future__ import annotations

from datetime import date
from typing import Callable

from .covid_2021 import build as _covid_2021
from .covid_no_distancing import build as _covid_no_distancing

# 2020-11-24 2단계 격상의 대조 환경: 11-23 의 규칙(1.5단계)을 그대로 이어 간다. 확진 소식은 그날 것.
# 11-23 까지는 covid_2021 과 똑같다. 실제로 있었던 단계를 유지할 뿐 '비교 조건'이라 적지 않는다.
_HOLD_1123 = date(2020, 11, 23)

_REGISTRY: dict[str, Callable[[date], dict]] = {
    "covid_2021": _covid_2021,
    "covid_no_distancing": _covid_no_distancing,
    "covid_2020_hold_1123": lambda day: _covid_2021(day, rule_day=min(day, _HOLD_1123)),
}


def list_environments() -> list[str]:
    return sorted(_REGISTRY)


def build_environment(env_id: str | None, day: date) -> dict:
    """환경 id 와 날짜로 배경 dict 생성. 없으면 {} — 섹션이 생략된다."""
    if not env_id:
        return {}
    fn = _REGISTRY.get(env_id)
    if fn is None:
        raise KeyError(f"알 수 없는 환경 id: {env_id} (가능: {list_environments()})")
    return fn(day) or {}
