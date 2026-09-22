#!/usr/bin/env python3
"""보고서 v2 CLI — run snapshot 파일만으로 상세 분석 보고서를 만든다.

시뮬레이션 엔진(`scripts/sim/*`)과 Neo4j 를 건드리지 않는다. 읽기만 한다.

사용 예
-------
    python scripts/report/build_report_v2.py \
        --run-id FINAL \
        --run-root "C:/Users/srdyh/gpu_exp_data/20260802/out_FINAL" \
        --policy-json data/neo4j_load/policies/P010.json \
        --policy-from 2025-07-28 \
        --out output/sim/report/FINAL_REPORT_V2.html

출력
----
``<out>.html``           단일 HTML (인라인 SVG, 외부 리소스 없음)
``<out>.md``             같은 내용의 Markdown
``<out>.data.json``      계산 결과 원본 + 일관성 검사 결과 (재현·검증용)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # 패키지로 import 될 때
    from . import analytics, consistency, narrator, render_v2
    from .narrate import provenance_note
except ImportError:  # 스크립트로 직접 실행될 때
    import analytics  # type: ignore
    import consistency  # type: ignore
    import narrator  # type: ignore
    import render_v2  # type: ignore
    from narrate import provenance_note  # type: ignore


DEFAULT_RUN_DIRS = {
    "BASE": ("out_BASE",),
    "FINAL": ("out_FINAL",),
    "BASE7500": ("rescue/out_BASE7500", "out_BASE7500"),
}


def resolve_run_root(run_id: str, run_root: Path | None, data_root: Path | None) -> Path:
    if run_root is not None:
        return run_root.resolve()
    if data_root is None:
        raise SystemExit("--run-root 또는 --data-root 중 하나는 있어야 합니다")
    for candidate in DEFAULT_RUN_DIRS.get(run_id, (f"out_{run_id}",)):
        path = (data_root / candidate).resolve()
        if path.is_dir():
            return path
    raise SystemExit(f"run 산출물 디렉터리를 찾지 못했습니다: {run_id} (data-root={data_root})")


def read_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("정책 JSON 은 객체여야 합니다")
    nested = payload.get("policy")
    return nested if isinstance(nested, dict) else payload


def source_paths_of(run_root: Path, days: list[str]) -> list[str]:
    paths = ["events.jsonl", "summary.json", "poi_summary.json"]
    existing = [p for p in paths if (run_root / p).is_file()]
    for day in days:
        for relative in (f"metrics/day_{day}.jsonl", f"timing/day_{day}.json"):
            if (run_root / relative).is_file():
                existing.append(relative)
    return existing


def generate(
    *,
    run_id: str,
    run_root: Path,
    policy: dict[str, Any],
    policy_from: str | None,
    start: str | None,
    days: int | None,
    out_path: Path,
    use_llm: bool = True,
    snapshot_id: str = "",
    sections: list[str] | None = None,
) -> dict[str, Any]:
    print("[1/5] run snapshot 스캔", file=sys.stderr, flush=True)
    bundle = analytics.build_bundle(
        run_id=run_id,
        run_root=run_root,
        policy=policy,
        policy_from=policy_from,
        start=start,
        days=days,
    )

    print("[2/5] 일관성 검증", file=sys.stderr, flush=True)
    checks = consistency.run_checks(bundle)

    print("[3/5] 해설 생성", file=sys.stderr, flush=True)
    narration = narrator.narrate_report(bundle, checks, enabled=use_llm)

    print("[4/5] HTML/Markdown 렌더", file=sys.stderr, flush=True)
    sources = source_paths_of(run_root, bundle["meta"]["days"])
    provenance = provenance_note(
        run_id=run_id,
        policy_id=str(policy.get("id") or "—"),
        analyses=["timeseries", "overlay", "categories", "did", "did_by_category", "deciles", "structure"],
        include_interview=False,
        snapshot_id=snapshot_id or "run-files",
        source_count=len(sources),
    )
    html = render_v2.build_html(
        bundle,
        checks,
        narration,
        policy=policy,
        provenance=provenance,
        run_id=run_id,
        source_paths=sources,
        sections=sections,
    )
    markdown = render_v2.build_markdown(bundle, checks, narration, policy=policy)

    print("[5/5] 저장", file=sys.stderr, flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    out_path.with_suffix(".md").write_text(markdown, encoding="utf-8")
    data_path = out_path.with_suffix(".data.json")
    data_path.write_text(
        json.dumps(
            {"bundle": bundle, "consistency": checks, "narration": narration, "provenance": provenance},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {
        "html": str(out_path),
        "markdown": str(out_path.with_suffix(".md")),
        "data": str(data_path),
        "consistent": checks["consistent"],
        "failed_checks": checks["failed_ids"],
        "used_llm": narration["used_llm"],
        "llm": narration["llm"],
        "sections": sections,
        "did_absolute": (bundle.get("did") or {}).get("did_absolute"),
        "days": bundle["meta"]["day_count"],
        "events": bundle["meta"]["event_rows"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DASOL 보고서 v2 생성기")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--policy-json", required=True, type=Path)
    parser.add_argument("--policy-from")
    parser.add_argument("--start")
    parser.add_argument("--days", type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--snapshot-id", default="")
    parser.add_argument(
        "--section",
        action="append",
        dest="sections",
        help="포함할 절 ID (s1~s11). 생략하면 전부. s1/s10/s11 은 항상 포함된다.",
    )
    parser.add_argument("--no-llm", action="store_true", help="해설 LLM 을 호출하지 않는다")
    args = parser.parse_args(argv)

    if args.days is not None and not 1 <= args.days <= 365:
        parser.error("--days 는 1~365 여야 합니다")
    if args.start:
        try:
            date.fromisoformat(args.start)
        except ValueError:
            parser.error("--start 는 YYYY-MM-DD 여야 합니다")
    if args.policy_from:
        try:
            date.fromisoformat(args.policy_from)
        except ValueError:
            parser.error("--policy-from 은 YYYY-MM-DD 여야 합니다")
    if not args.policy_json.is_file():
        parser.error(f"정책 JSON 을 찾을 수 없습니다: {args.policy_json}")

    run_root = resolve_run_root(args.run_id, args.run_root, args.data_root)
    policy = read_policy(args.policy_json.resolve())
    try:
        result = generate(
            run_id=args.run_id,
            run_root=run_root,
            policy=policy,
            policy_from=args.policy_from,
            start=args.start,
            days=args.days,
            out_path=args.out.resolve(),
            use_llm=not args.no_llm,
            snapshot_id=args.snapshot_id,
            sections=list(dict.fromkeys(args.sections)) if args.sections else None,
        )
    except analytics.AnalyticsError as exc:
        print(f"보고서를 만들 수 없습니다: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["consistent"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
