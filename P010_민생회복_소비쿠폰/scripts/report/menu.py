#!/usr/bin/env python3
"""Non-interactive DASOL report runner used by the web report job.

Only structured arguments are accepted by the API.  The command reuses the
existing ``generate_final_report`` calculation and HTML functions; it never
accepts a shell command, an arbitrary data root, or a user-supplied executable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SIM_DIR = REPO_ROOT / "scripts" / "sim"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SIM_DIR))

from build_html import append_provenance_html, build_report_html  # noqa: E402
from catalog import CATALOG, applicable_ids  # noqa: E402
from engine import build_modern, is_modern  # noqa: E402
from narrate import provenance_note  # noqa: E402
from snapshot import SnapshotError, report_source_paths, verify_manifest  # noqa: E402


def _empty_trigger() -> tuple[dict[str, Any], str]:
    return ({"distribution": {}, "distribution_pct": {}, "total": 0}, "")


def _empty_regulars() -> tuple[dict[str, Any], str]:
    return ({"frequency": {}, "source": {}, "total": 0}, "")


def _empty_satisfaction() -> tuple[dict[str, Any], str]:
    return ({"by_trigger": [], "by_category": []}, "")


def _read_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("정책 JSON은 객체여야 합니다")
    policy = payload.get("policy")
    return policy if isinstance(policy, dict) else payload


def _selected_ids(policy: dict[str, Any], raw: list[str] | None, use_all: bool) -> list[str]:
    allowed = applicable_ids(policy)
    if use_all or not raw:
        return allowed
    selected = list(dict.fromkeys(raw))
    invalid = [item for item in selected if item not in allowed]
    if invalid:
        raise ValueError(f"적용할 수 없는 분석입니다: {', '.join(invalid)}")
    return selected


def build_report(
    *,
    run_id: str,
    policy_id: str,
    start: date,
    days: int,
    policy_path: Path,
    policy_from: str | None,
    analysis_ids: list[str],
    include_interview: bool,
    snapshot_manifest: Path,
    data_root: Path | None,
    out_path: Path,
) -> tuple[Path, Path]:
    manifest = verify_manifest(
        snapshot_manifest,
        expected_run_id=run_id,
        requested_start=start,
        requested_days=days,
        data_root=data_root,
    )
    # The protected generator has one legacy file-backed branch (P009/DID).
    # Bind that branch to the exact same verified run root as the manifest.
    os.environ["SIM_OUTPUT_DIR"] = str(Path(manifest["root"]).resolve())
    import generate_final_report as generator

    policy = _read_policy(policy_path)
    effective_from = policy_from or policy.get("effective_from")
    source_paths = report_source_paths(manifest)
    with tempfile.TemporaryDirectory(prefix="dasol-report-") as temp:
        work_dir = Path(temp)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if is_modern(generator):
            print("[1/4] DASOL context + immutable snapshot 확인", file=sys.stderr, flush=True)
            html, markdown = build_modern(
                generator,
                policy_path=policy_path,
                start=start,
                days=days,
                policy_from=policy_from,
                analysis_ids=analysis_ids,
                include_interview=include_interview,
                work_dir=work_dir,
            )
        else:
            print("[1/4] 기존 엔진의 조건 요약", file=sys.stderr, flush=True)
            s1 = generator.section1_conditions(start, days, effective_from)

            print("[2/4] 선택한 분석 계산", file=sys.stderr, flush=True)
            if "sales" in analysis_ids and effective_from:
                if policy.get("income_grants") and hasattr(generator, "section2b_income_did_p009"):
                    # Preserve the protected renderer's P009-specific DID
                    # dispatch instead of silently substituting the facility
                    # (Gangnam/non-Gangnam) comparison.
                    s2 = generator.section2b_income_did_p009(start, days, effective_from, work_dir)
                else:
                    s2 = generator.section2_before_after(start, days, effective_from, work_dir)
            else:
                s2 = None
            s3 = generator.section3_spillover(start, days, work_dir) if "spillover" in analysis_ids else None
            s41 = generator.section4_1_triggers(start, days, work_dir) if "triggers" in analysis_ids else _empty_trigger()
            s42 = generator.section4_2_regulars(start, days, work_dir) if "regulars" in analysis_ids else _empty_regulars()
            s43 = generator.section4_3_satisfaction(start, days, work_dir) if "satisfaction" in analysis_ids else _empty_satisfaction()

            s5: dict[str, Any] = {"grant": {"error": "interview excluded by request"}}
            if include_interview:
                print("[3/4] 인터뷰 분석", file=sys.stderr, flush=True)
                s5 = generator.section5_interviews(start, days, work_dir)
            else:
                print("[3/4] 인터뷰 제외", file=sys.stderr, flush=True)

            chart_dir = out_path.parent / f"{out_path.stem}.d"
            chart_dir.mkdir(parents=True, exist_ok=True)
            for chart in work_dir.glob("*.png"):
                shutil.copy2(chart, chart_dir / chart.name)

            markdown = generator.build_markdown(
                start,
                days,
                effective_from,
                s1,
                s2,
                s3,
                s41,
                s42,
                s43,
                s5,
                chart_dir.name,
            )
        provenance = provenance_note(
            run_id=run_id,
            policy_id=policy_id,
            analyses=analysis_ids,
            include_interview=include_interview,
            snapshot_id=str(manifest["snapshot_id"]),
            source_count=len(manifest.get("files", [])),
        )
        source_lines = "\n".join(
            f"- [{source}](/api/runs/{quote(run_id, safe='')}/artifacts/{quote(source, safe='/')})"
            for source in source_paths
        )
        markdown = f"{markdown.rstrip()}\n\n---\n\n### 생성 provenance\n\n{provenance}\n\n원본 근거:\n{source_lines}\n"
        out_md = out_path.with_suffix(".md")
        out_md.write_text(markdown, encoding="utf-8")
        if is_modern(generator):
            html = append_provenance_html(html, provenance=provenance, run_id=run_id, source_paths=source_paths)
        else:
            html = build_report_html(
                generator,
                start,
                days,
                effective_from,
                s1,
                s2,
                s3,
                s41,
                s42,
                s43,
                s5,
                work_dir,
                provenance=provenance,
                run_id=run_id,
                source_paths=source_paths,
            )
        out_path.write_text(html, encoding="utf-8")

    print("[4/4] self-contained HTML/Markdown 저장 완료", file=sys.stderr, flush=True)
    return out_path, out_md


def main() -> int:
    parser = argparse.ArgumentParser(description="DASOL report generator")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--policy-json", required=True, type=Path)
    parser.add_argument("--policy-from")
    parser.add_argument("--analysis", action="append", dest="analyses")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--include-interview", action="store_true")
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.days < 1 or args.days > 365:
        parser.error("days는 1~365 범위여야 합니다")
    if not args.policy_json.is_file():
        parser.error("policy JSON을 찾을 수 없습니다")

    policy = _read_policy(args.policy_json)
    selected = _selected_ids(policy, args.analyses, args.all)
    if not selected:
        parser.error("적용 가능한 분석이 없습니다")
    if args.policy_from:
        date.fromisoformat(args.policy_from)

    try:
        build_report(
            run_id=args.run_id,
            policy_id=args.policy_id,
            start=date.fromisoformat(args.start),
            days=args.days,
            policy_path=args.policy_json.resolve(),
            policy_from=args.policy_from,
            analysis_ids=selected,
            include_interview=args.include_interview,
            snapshot_manifest=args.snapshot_manifest.resolve(),
            data_root=args.data_root.resolve() if args.data_root else None,
            out_path=args.out.resolve(),
        )
    except SnapshotError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
