"""Small, provenance-first narration helper for the report adapter.

The existing report engine owns its analysis prose.  This module intentionally
does not invent a number or call an external model; it provides the generated
report with a machine-readable note about the selected analysis set.
"""
from __future__ import annotations

import json


def provenance_note(
    *,
    run_id: str,
    policy_id: str,
    analyses: list[str],
    include_interview: bool,
    snapshot_id: str,
    source_count: int,
) -> str:
    selected = ", ".join(analyses) if analyses else "없음"
    interview = "포함" if include_interview else "제외"
    return (
        f"DASOL 생성 기록 · run={run_id} · policy={policy_id} · "
        f"snapshot={snapshot_id} · 분석={selected} · 인터뷰={interview} · "
        f"검증된 원본 파일={source_count}개"
    )


def narrate(label: str, data: dict, ctx: dict, mode: str | None = None) -> str:
    """Keep the modern DASOL contract without inventing a numeric conclusion.

    The upstream branch may replace this narrow function with its configured
    LLM narrator.  Until then, the report still carries a deterministic
    explanation that points readers to the computed table/chart.  No value is
    rounded, inferred, or created here.
    """

    keys = ", ".join(str(key) for key in data.keys()) if isinstance(data, dict) else "계산 결과"
    policy_name = ctx.get("name") or ctx.get("id") or "선택한 policy"
    return f"{policy_name}의 {label} 계산이 완료되었습니다. 원본 계산 결과의 키: {keys}. 숫자 해석은 아래 표·차트와 원본 근거를 함께 확인하세요."
