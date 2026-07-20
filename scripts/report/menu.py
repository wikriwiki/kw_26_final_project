#!/usr/bin/env python3
"""대화형 메뉴 — 분석 함수를 사용자가 골라 실행하고 최종 HTML 보고서를 조립한다.

기존 generate_final_report.py는 P008/P009 전용으로 섹션이 고정돼 있었다.
이 스크립트는 그 섹션 함수(숫자·차트 계산)를 그대로 재사용하되:
  - 어떤 분석을 포함할지는 사용자가 메뉴에서 직접 고르고 (catalog.py)
  - 해설 문장은 좁은 역할의 LLM 호출로 생성한다 (narrate.py) — 국내 모델 교체 가능
  - 인터뷰는 기존 section5_interviews를 그대로 사용 (이미 LLM 기반, trace 인용 규칙 내장)

CLI:
  python scripts/report/menu.py \\
      --start 2026-05-25 --days 4 --policy-from 2026-05-27 \\
      --out docs/FINAL_REPORT.html

  # 메뉴 없이 적용 가능한 분석을 전부 실행:
  python scripts/report/menu.py --start 2026-05-25 --days 4 --policy-from 2026-05-27 --all
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))

from build_html import build_report_html  # noqa: E402
from catalog import applicable_specs  # noqa: E402
from narrate import narrate  # noqa: E402

import generate_final_report as gfr  # noqa: E402


def _select_analyses(specs, use_all: bool):
    if use_all:
        return specs
    print("\n" + "=" * 60, file=sys.stderr)
    print("적용 가능한 분석", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for i, s in enumerate(specs, 1):
        print(f"  [{i}] {s.label}\n        {s.description}", file=sys.stderr)
    raw = input("\n실행할 분석 번호 (쉼표 구분, 전체=엔터) [all]: ").strip()
    if not raw or raw.lower() == "all":
        return specs
    idxs = [int(x) for x in raw.split(",") if x.strip()]
    return [specs[i - 1] for i in idxs if 1 <= i <= len(specs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--days", type=int, required=True)
    ap.add_argument("--policy-from", help="정책 시행일 YYYY-MM-DD (없으면 섹션2 스킵)")
    ap.add_argument("--out", default="docs/FINAL_REPORT.html")
    ap.add_argument("--all", action="store_true", help="메뉴 없이 적용 가능한 분석을 모두 실행")
    ap.add_argument("--model", default="exaone",
                     help="narrate()·인터뷰에 쓸 LLM_MODE (llm_client.MODELS 키). "
                          "국내 트랙 기본값: exaone (EXAONE 4.0 32B AWQ)")
    ap.add_argument("--skip-interview", action="store_true")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)

    policy_path = gfr.select_policy_json()
    ctx = gfr.load_policy_ctx(policy_path)

    specs = applicable_specs(ctx)
    if not specs:
        print("[오류] 이 정책 ctx에 적용 가능한 분석이 없습니다.", file=sys.stderr)
        sys.exit(1)
    chosen = _select_analyses(specs, args.all)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        s1 = gfr.section1_conditions(start, args.days, args.policy_from)

        sections = []
        for spec in chosen:
            print(f"\n[실행] {spec.label} ...", file=sys.stderr)
            result = spec.run(ctx, start, args.days, args.policy_from, out_dir)
            if result is None:
                print("  → 스킵 (조건 불충족 — 예: policy-from 없음)", file=sys.stderr)
                continue
            print(f"  [narrate] {spec.label} 해설 생성 중 ...", file=sys.stderr)
            narration = narrate(result.label, result.data, ctx, mode=args.model)
            sections.append({
                "title": result.label,
                "narration": narration,
                "table_rows": result.table_rows,
                "chart_paths": result.chart_paths,
            })

        interview = None
        if not args.skip_interview:
            print("\n[인터뷰] 대표 1명 인터뷰 생성 중 ...", file=sys.stderr)
            interview = gfr.section5_interviews(start, args.days, out_dir, ctx, mode=args.model)

        html = build_report_html(ctx, s1, sections, interview)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"\n완료 → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
