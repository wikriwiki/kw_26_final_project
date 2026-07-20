"""분석 카탈로그 — 정책 ctx에 따라 어떤 분석이 적용 가능한지 판단하고 실행한다.

숫자 계산과 "이 정책엔 DID가 가능한가" 같은 판단은 전부 여기(파이썬 코드)가 담당한다.
LLM(narrate.py)에는 이미 계산이 끝난 숫자만 넘긴다 — 국내 소형 모델로 교체해도
판단 오류(잘못된 대조군, 없는 숫자 생성) 위험이 없도록 하기 위함.

숫자·차트 생성 로직은 새로 만들지 않고 generate_final_report.py의 기존 섹션 함수를
그대로 재사용한다 (섹션2/3 dispatch에 이미 "대조군 있는가 → DID 채택" 판단이
코드로 박혀 있음).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))

import generate_final_report as gfr  # noqa: E402


@dataclass
class AnalysisResult:
    id: str
    label: str
    data: dict                                       # narrate()에 넘길 raw 숫자
    chart_paths: list[Path] = field(default_factory=list)
    table_rows: list[dict] | None = None              # HTML 표로 그대로 렌더


@dataclass
class AnalysisSpec:
    id: str
    label: str
    description: str
    applicable: Callable[[dict], bool]
    run: Callable[[dict, date, int, str | None, Path], AnalysisResult | None]


# ─────────────────────────────────────────────────────────────
# 정책 시행 전후 매출 효과 — 섹션2 dispatch(코드)가 DID 채택 여부를 이미 결정
# ─────────────────────────────────────────────────────────────
def _sales_applicable(ctx: dict) -> bool:
    return bool(ctx.get("income_grants") or ctx.get("target_cats"))


def _sales_run(ctx, start, days, policy_from, out_dir):
    result = gfr.run_section2(ctx, start, days, policy_from, out_dir)
    if result is None:
        return None

    if ctx.get("income_grants"):
        # section2b_income_did_p009 반환 형태 — 단일 dict: {per_bucket, grant_aids_count, fig}
        data = result
        charts = [out_dir / data["fig"]] if data.get("fig") else []
        label = "소득 군집별 정책 효과 (DID)"
        rows = [{"소득 bucket": bucket, **vals}
                for bucket, vals in data.get("per_bucket", {}).items()]
    else:
        # section2_before_after 반환 형태 — (data, {차트명: 파일명, ...}) 튜플
        data, figs = result
        charts = [out_dir / name for name in figs.values()]
        label = "지역×업종 정책 시행 전후 매출 (DID)"
        rows = [data.get("summary", {})] if data.get("summary") else None

    return AnalysisResult(id="sales", label=label, data=data,
                           chart_paths=charts, table_rows=rows)


# ─────────────────────────────────────────────────────────────
# 간접 영향(Spillover) — 지역 한정 정책일 때만
# ─────────────────────────────────────────────────────────────
def _spillover_applicable(ctx: dict) -> bool:
    return bool([d for d in ctx.get("target_districts", []) if d != "서울특별시"])


def _spillover_run(ctx, start, days, policy_from, out_dir):
    result = gfr.run_section3(ctx, start, days, out_dir)
    if result is None:
        return None
    data, fig = result
    return AnalysisResult(id="spillover", label="간접 영향 (Spillover)",
                           data=data, chart_paths=[out_dir / fig])


# ─────────────────────────────────────────────────────────────
# 아래 3개는 정책 유형과 무관하게 항상 적용 가능
# ─────────────────────────────────────────────────────────────
def _triggers_run(ctx, start, days, policy_from, out_dir):
    data, fig = gfr.section4_1_triggers(start, days, out_dir)
    rows = [{"동기": k, "건수": v, "비율(%)": data["distribution_pct"][k]}
            for k, v in data["distribution"].items()]
    return AnalysisResult(id="triggers", label="방문 목적·이동 패턴 (trigger 분포)",
                           data=data, chart_paths=[out_dir / fig], table_rows=rows)


def _regulars_run(ctx, start, days, policy_from, out_dir):
    data, fig = gfr.section4_2_regulars(start, days, out_dir)
    rows = [{"구분": k, "관계 수": v} for k, v in data["frequency"].items()]
    return AnalysisResult(id="regulars", label="단골 vs 신규",
                           data=data, chart_paths=[out_dir / fig], table_rows=rows)


def _satisfaction_run(ctx, start, days, policy_from, out_dir):
    data, fig = gfr.section4_3_satisfaction(start, days, out_dir)
    rows = [{"동기": r["trigger"], "평균 만족도": r["avg_sat"], "표본 수": r["n"]}
            for r in data["by_trigger"]]
    return AnalysisResult(id="satisfaction", label="만족도·피드백 (동기별)",
                           data=data, chart_paths=[out_dir / fig], table_rows=rows)


CATALOG: list[AnalysisSpec] = [
    AnalysisSpec(
        "sales", "정책 시행 전후 매출 효과 (DID)",
        "처치군/대조군 정의와 DID 채택 여부는 정책 ctx 구조(income_grants·target_cats)로 "
        "코드가 자동 결정한다 — grant형이면 소득 군집별, 그 외엔 지역×업종별.",
        _sales_applicable, _sales_run,
    ),
    AnalysisSpec(
        "spillover", "간접 영향 (Spillover)",
        "정책 대상이 특정 자치구(전서울이 아님)일 때만 적용 — 인접·원거리 자치구 비교.",
        _spillover_applicable, _spillover_run,
    ),
    AnalysisSpec(
        "triggers", "방문 목적·이동 패턴 (trigger 분포)",
        "정책 유형과 무관하게 항상 적용 가능.",
        lambda ctx: True, _triggers_run,
    ),
    AnalysisSpec(
        "regulars", "단골 vs 신규",
        "정책 유형과 무관하게 항상 적용 가능.",
        lambda ctx: True, _regulars_run,
    ),
    AnalysisSpec(
        "satisfaction", "만족도·피드백 (동기별)",
        "정책 유형과 무관하게 항상 적용 가능.",
        lambda ctx: True, _satisfaction_run,
    ),
]


def applicable_specs(ctx: dict) -> list[AnalysisSpec]:
    return [s for s in CATALOG if s.applicable(ctx)]
