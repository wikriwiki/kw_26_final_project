"""Compatibility bridge for the two checked-in DASOL renderer contracts.

``origin/dasol`` exposes policy-context dispatchers (``run_section2`` and
``run_section3``), while the older protected renderer exposes section
functions directly.  The web adapter must not silently call the wrong
signature, so the modern branch is selected only when its full contract is
present; otherwise the legacy path remains explicit in ``menu.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .narrate import narrate
except ImportError:  # script execution via menu.py adds scripts/report to sys.path
    from narrate import narrate


def is_modern(generator: Any) -> bool:
    required = ("load_policy_ctx", "run_section2", "run_section3")
    return all(callable(getattr(generator, name, None)) for name in required)


def _normalized_result(result: Any, out_dir: Path) -> tuple[dict[str, Any], list[Path], list[dict[str, Any]] | None]:
    if result is None:
        raise ValueError("DASOL 분석 함수가 결과 없이 반환했습니다")
    if isinstance(result, dict):
        data = result
        figure = data.get("fig")
        charts = [out_dir / str(figure)] if figure else []
        if isinstance(data.get("per_bucket"), dict):
            rows = [{"소득 bucket": key, **value} for key, value in data["per_bucket"].items() if isinstance(value, dict)]
        else:
            rows = None
        return data, charts, rows
    if isinstance(result, tuple) and len(result) == 2:
        data, figure_value = result
        if not isinstance(data, dict):
            raise ValueError("DASOL 분석 결과의 data가 객체가 아닙니다")
        if isinstance(figure_value, dict):
            charts = [out_dir / str(name) for name in figure_value.values() if name]
        elif figure_value:
            charts = [out_dir / str(figure_value)]
        else:
            charts = []
        rows = [data.get("summary", {})] if isinstance(data.get("summary"), dict) and data.get("summary") else None
        return data, charts, rows
    raise ValueError("DASOL 분석 함수의 반환 형태를 인식하지 못했습니다")


def build_modern(
    generator: Any,
    *,
    policy_path: Path,
    start: Any,
    days: int,
    policy_from: str | None,
    analysis_ids: list[str],
    include_interview: bool,
    work_dir: Path,
) -> tuple[str, str]:
    """Run the origin/dasol context-dispatch flow and return HTML + Markdown."""

    ctx = generator.load_policy_ctx(policy_path)
    if policy_from:
        ctx = {**ctx, "effective_from": policy_from}
    s1 = generator.section1_conditions(start, days, policy_from or ctx.get("effective_from"))
    sections: list[dict[str, Any]] = []
    effective_from = policy_from or ctx.get("effective_from")

    for analysis_id in analysis_ids:
        if analysis_id == "sales":
            result = generator.run_section2(ctx, start, days, effective_from, work_dir)
            label = "정책 시행 전후 매출 효과 (DID)"
        elif analysis_id == "spillover":
            result = generator.run_section3(ctx, start, days, work_dir)
            label = "간접 영향 (Spillover)"
        elif analysis_id == "triggers":
            result = generator.section4_1_triggers(start, days, work_dir)
            label = "방문 목적·이동 패턴 (trigger 분포)"
        elif analysis_id == "regulars":
            result = generator.section4_2_regulars(start, days, work_dir)
            label = "단골 vs 신규"
        elif analysis_id == "satisfaction":
            result = generator.section4_3_satisfaction(start, days, work_dir)
            label = "만족도·피드백 (동기별)"
        else:
            raise ValueError(f"지원하지 않는 DASOL 분석 ID: {analysis_id}")
        if result is None:
            continue
        data, chart_paths, table_rows = _normalized_result(result, work_dir)
        sections.append(
            {
                "title": label,
                "narration": narrate(label, data, ctx),
                "data": data,
                "table_rows": table_rows,
                "chart_paths": chart_paths,
            }
        )

    interview = None
    if include_interview:
        interview = generator.section5_interviews(start, days, work_dir, ctx, mode=None)
    html = generator.build_html(ctx, s1, sections, interview)

    lines = [
        f"# DASOL 보고서 — {ctx.get('name') or ctx.get('id') or 'policy'}",
        "",
        f"기간: {s1.get('기간', '')}",
        "",
        "## 계산된 분석",
        "",
    ]
    for section in sections:
        lines.extend(
            [
                f"### {section['title']}",
                "",
                str(section["narration"]),
                "",
                "```json",
                json.dumps(section["data"], ensure_ascii=False, indent=2, default=str),
                "```",
                "",
            ]
        )
    if include_interview:
        lines.extend(["## 인터뷰 부록", "", json.dumps(interview, ensure_ascii=False, indent=2, default=str), ""])
    return html, "\n".join(lines)
