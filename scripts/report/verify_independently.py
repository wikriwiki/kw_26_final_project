#!/usr/bin/env python3
"""보고서를 만든 코드를 **쓰지 않고** 원본에서 다시 세어 대조한다.

왜 따로 만드나
--------------
`consistency.py` 는 보고서 안의 값들이 서로 어긋나지 않는지 본다. 훌륭하지만
한계가 있다 — 같은 모듈로 두 번 계산하면 **같은 버그를 두 번 얻는다.**
`analytics.py` 의 집계가 통째로 틀려도, 그 틀린 값들끼리는 완벽하게 일치한다.

그래서 이 스크립트는 `analytics.py` 를 import 하지 않는다. `events.jsonl` 을
표준 라이브러리만으로 다시 읽어 이중차분을 처음부터 계산하고, 보고서가 실제로
실은 값(`<out>.data.json`)과 대조한다. 두 구현이 독립적으로 같은 숫자에
도달해야만 통과다.

사용
----
    python scripts/report/verify_independently.py \
        --run-root  "C:/Users/srdyh/gpu_exp_data/20260802/out_FINAL" \
        --data-json output/sim/report/FINAL_REPORT_V3.data.json \
        --policy-json data/neo4j_load/policies/P010.json

종료 코드: 0 전부 일치 · 1 불일치 · 2 입력을 읽을 수 없음
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable

#: 금액은 소수 둘째 자리에서 반올림해 저장된다. 항목 수에 비례한 누적분과
#: 크기에 비례한 상대오차만 허용한다 — 그 이상은 실패다.
ABS_TOLERANCE = 1.0
REL_TOLERANCE = 1e-6


def _read_events(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "events.jsonl"
    if not path.is_file():
        raise SystemExit(f"events.jsonl 이 없습니다: {path}")
    rows = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _amount(row: dict[str, Any]) -> float:
    value = row.get("amt", row.get("amount"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _l1(row: dict[str, Any]) -> str:
    return str(row.get("l1") or row.get("cat_l1") or "미분류")


def _axis(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return str(value)
    return "미분류"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--data-json", required=True, type=Path)
    parser.add_argument("--policy-json", required=True, type=Path)
    args = parser.parse_args()

    try:
        report = json.loads(args.data_json.read_text(encoding="utf-8"))
        policy = json.loads(args.policy_json.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"입력을 읽지 못했습니다: {exc}", file=sys.stderr)
        return 2
    policy = policy.get("policy") if isinstance(policy.get("policy"), dict) else policy
    bundle = report.get("bundle") or {}
    did_report = bundle.get("did")
    if not did_report:
        print("보고서에 이중차분 결과가 없어 대조할 수 없습니다.", file=sys.stderr)
        return 2

    treat = {str(c) for c in (policy.get("benefit_categories") or policy.get("target_cats") or [])}
    if not treat:
        print("정책에 대상 업종이 없어 처치군을 만들 수 없습니다.", file=sys.stderr)
        return 2
    cut_raw = (bundle.get("period") or {}).get("policy_from")
    if not cut_raw:
        print("보고서에 정책 시행일이 없습니다.", file=sys.stderr)
        return 2
    cut = date.fromisoformat(str(cut_raw))

    rows = _read_events(args.run_root)
    # 보고서가 본 것과 **같은 창**만 본다. 창이 다르면 값이 다른 것이 당연하다.
    window = set((bundle.get("meta") or {}).get("days") or [])
    if window:
        rows = [row for row in rows if str(row.get("day")) in window]
    days = sorted({str(row.get("day")) for row in rows})
    pre = [d for d in days if date.fromisoformat(d) < cut]
    post = [d for d in days if date.fromisoformat(d) >= cut]
    if not pre or not post:
        print("사전 또는 사후 기간이 비어 대조할 수 없습니다.", file=sys.stderr)
        return 2

    def daily(daylist: list[str], keep: Callable[[dict[str, Any]], bool]) -> float:
        wanted = set(daylist)
        return sum(_amount(r) for r in rows if str(r.get("day")) in wanted and keep(r)) / len(daylist)

    t0 = daily(pre, lambda r: _l1(r) in treat)
    t1 = daily(post, lambda r: _l1(r) in treat)
    c0 = daily(pre, lambda r: _l1(r) not in treat)
    c1 = daily(post, lambda r: _l1(r) not in treat)
    if not c0:
        print("대조군의 사전 소비가 0 이라 반사실을 만들 수 없습니다.", file=sys.stderr)
        return 2
    growth = c1 / c0
    did = t1 - t0 * growth

    failures: list[str] = []

    def compare(label: str, mine: float, theirs: Any, *, scale: int = 1) -> None:
        if theirs is None:
            failures.append(label)
            print(f"FAIL {label:34s} 독립={mine:>18,.2f}  보고서=없음")
            return
        allowed = ABS_TOLERANCE * scale + abs(mine) * REL_TOLERANCE
        ok = abs(mine - float(theirs)) <= allowed
        if not ok:
            failures.append(label)
        print(f"{'OK  ' if ok else 'FAIL'} {label:34s} 독립={mine:>18,.2f}  보고서={float(theirs):>18,.2f}")

    def axis_sum(pick: Callable[[dict[str, Any]], str]) -> float:
        buckets: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        pre_set = set(pre)
        for row in rows:
            if _l1(row) not in treat:
                continue
            buckets[pick(row)][0 if str(row.get("day")) in pre_set else 1] += _amount(row)
        return sum(v[1] / len(post) - (v[0] / len(pre)) * growth for v in buckets.values())

    print(f"구간 {days[0]} ~ {days[-1]} · 사전 {len(pre)}일 · 사후 {len(post)}일 · 이벤트 {len(rows):,}건")
    print(f"처치군 {sorted(treat)}")
    print()

    compare("전체 소비금액", sum(_amount(r) for r in rows), (bundle.get("totals") or {}).get("amt"), scale=len(days))
    compare("처치군 사전 일평균", t0, did_report.get("treat_pre"))
    compare("처치군 사후 일평균", t1, did_report.get("treat_post"))
    compare("대조군 사전 일평균", c0, did_report.get("control_pre"))
    compare("대조군 사후 일평균", c1, did_report.get("control_post"))
    compare("반사실", t0 * growth, did_report.get("counterfactual_post"))
    compare("이중차분", did, did_report.get("did_absolute"))

    compare(
        "대상 업종별 DID 합",
        axis_sum(_l1),
        sum(r["did_absolute"] for r in (bundle.get("did_by_category") or []) if r.get("targeted")),
        scale=len(treat),
    )
    sub = bundle.get("did_by_subcategory") or {}
    if sub.get("available"):
        compare(
            "세부업종 DID 합",
            axis_sum(lambda r: f'{_l1(r)}|{_axis(r, "l2", "sub", "cat_l2")}'),
            sum(i["did_absolute"] for i in sub.get("items") or []),
            scale=max(len(sub.get("items") or []), 1),
        )
    region = bundle.get("did_by_region") or {}
    if region.get("available"):
        compare(
            "지역별 DID 합",
            axis_sum(lambda r: _axis(r, "gu", "district", "sgg")),
            region.get("total_did"),
            scale=max(len(region.get("items") or []), 1),
        )
    daytype = bundle.get("did_by_daytype") or {}
    if daytype.get("available"):
        compare(
            "요일유형별 DID 합",
            axis_sum(lambda r: _axis(r, "day_type", "daytype")),
            daytype.get("total_did"),
            scale=max(len(daytype.get("items") or []), 1),
        )

    cf = bundle.get("did_counterfactual_daily") or {}
    if cf.get("available"):
        per_day: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for row in rows:
            per_day[str(row.get("day"))][0 if _l1(row) in treat else 1] += _amount(row)
        gaps = [per_day[d][0] - t0 * (per_day[d][1] / c0) for d in post]
        compare("일자별 격차 평균", sum(gaps) / len(gaps), cf.get("mean_gap_post"), scale=len(post))
        compare("일자별 격차 누적", sum(gaps), cf.get("cumulative_gap_total"), scale=len(post))

    overlay = bundle.get("overlay") or {}
    if overlay.get("available"):
        overall = overlay["overall"]
        span = min(len(pre), len(post))
        pre_window, post_window = set(pre[-span:]), set(post[:span])
        compare(
            "겹쳐보기 사전 합",
            sum(_amount(r) for r in rows if str(r.get("day")) in pre_window),
            sum(overall["pre"]),
            scale=span,
        )
        compare(
            "겹쳐보기 사후 합",
            sum(_amount(r) for r in rows if str(r.get("day")) in post_window),
            sum(overall["post"]),
            scale=span,
        )
        compare(
            "겹쳐보기 누적선 끝",
            sum(_amount(r) for r in rows if str(r.get("day")) in post_window),
            (overall.get("post_cumulative") or [None])[-1],
            scale=span,
        )

    paid = 0.0
    for row in rows:
        raw = row.get("sp", row.get("policy_spend"))
        if isinstance(raw, str) and raw.strip():
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}
        if isinstance(raw, dict):
            for value in raw.values():
                try:
                    paid += float(value)
                except (TypeError, ValueError):
                    pass
    compare("정책 지급 총액", paid, (bundle.get("totals") or {}).get("policy_paid"), scale=len(days))

    counts = (report.get("consistency") or {}).get("counts") or {}
    print()
    print(f"보고서 자체 항등식: {counts}")
    if failures:
        print(f"독립 재계산 결과: {len(failures)}건 불일치 — {', '.join(failures)}")
        return 1
    print("독립 재계산 결과: 전부 일치")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
