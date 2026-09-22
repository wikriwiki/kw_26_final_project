"""소비행동 프롬프트 후보 v6 — 기전 중립판.

본문은 p012 와 같고 daily_propensity 블록만 다르다. 무엇이 다른지는
`candidates.py` 의 표를 보라. 후보 집합은 미리 못박았으므로 여기에
새 후보를 임의로 추가하지 않는다.
"""
from __future__ import annotations

from .candidates import make_system_prompt
from .p012 import format_dawn_blocks  # noqa: F401  (본문 공통)

SYSTEM_PROMPT = make_system_prompt("v6")
