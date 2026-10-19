"""v53n — v53 with one directional clause removed (2026-10-06, final prompt review before the 2,000-person runs).

"외출을 너무 보수적으로 줄이면 부자연스럽다" told every arm that cutting outings is unnatural. It is the same in the
policy and no-policy arms, but under a restriction (distancing step 2) it pushes against the very response being
measured. The descriptive sentence before it — people go out on weekdays too — stays. Nothing else changes.
Named v53n, not v54: "v54" was already used on 2026-09-28 for a Stage1 candidate screen
(experiments/multi_policy_v53_20260927/run_v54_stage1_screen.py).
"""
from __future__ import annotations

from .p012 import format_dawn_blocks  # noqa: F401
from .v53 import SYSTEM_PROMPT as _V53

STAGE2_NEUTRAL = True

_OLD = "- 사람들은 평일에도 일상적 외출(점심·간식·간단 쇼핑·운동·약 처방)을 한다. 외출을 너무 보수적으로 줄이면 부자연스럽다.\n"
_NEW = "- 사람들은 평일에도 일상적 외출(점심·간식·간단 쇼핑·운동·약 처방)을 한다.\n"


def build() -> str:
    if _V53.count(_OLD) != 1:
        raise ValueError('v53 outing wording changed; expected one exact span')
    return _V53.replace(_OLD, _NEW, 1)


SYSTEM_PROMPT = build()
