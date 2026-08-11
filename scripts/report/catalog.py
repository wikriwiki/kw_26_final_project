"""DASOL analysis catalog.

The catalog decides which analyses are meaningful for a policy.  It does not
calculate metrics and it never asks an LLM to decide whether an analysis is
valid.  The web API uses this same shape to render disabled menu items.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnalysisSpec:
    id: str
    label: str
    description: str


CATALOG: tuple[AnalysisSpec, ...] = (
    AnalysisSpec(
        "sales",
        "정책 시행 전후 매출 효과",
        "정책 시행일이 있는 완료 run에서만 기존 정책효과 계산 함수를 실행합니다.",
    ),
    AnalysisSpec(
        "spillover",
        "간접 영향 (spillover)",
        "특정 자치구 대상 정책일 때만 인접·원거리 지역 비교를 엽니다.",
    ),
    AnalysisSpec(
        "triggers",
        "방문 목적·이동 패턴",
        "외출 이벤트의 trigger 분포를 집계합니다.",
    ),
    AnalysisSpec(
        "regulars",
        "단골 vs 신규",
        "KNOWS_POI 방문 빈도와 출처를 비교합니다.",
    ),
    AnalysisSpec(
        "satisfaction",
        "만족도·피드백",
        "trigger·업종별 만족도와 표본 수를 표시합니다.",
    ),
)


def _policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(policy, dict):
        return {}
    nested = policy.get("policy")
    return nested if isinstance(nested, dict) else policy


def _districts(policy: dict[str, Any]) -> list[str]:
    value = policy.get("target_districts")
    return [str(item) for item in value] if isinstance(value, list) else []


def _target_categories(policy: dict[str, Any]) -> list[str]:
    value = policy.get("benefit_categories") or policy.get("target_cats")
    return [str(item) for item in value] if isinstance(value, list) else []


def applicability(policy: dict[str, Any] | None) -> dict[str, tuple[bool, str | None]]:
    """Return an explicit applicability decision and its reason for every item."""

    ctx = _policy(policy)
    districts = _districts(ctx)
    target_categories = _target_categories(ctx)
    income_grants = ctx.get("income_grants")
    scoped_districts = [item for item in districts if item not in {"서울특별시", "서울", "전국"}]
    return {
        "sales": (
            bool(income_grants or target_categories),
            None
            if income_grants or target_categories
            else "정책 ctx에 income_grants 또는 benefit_categories가 없어 기존 DID 계산을 적용하지 않습니다.",
        ),
        "spillover": (
            bool(scoped_districts),
            None if scoped_districts else "특정 자치구 범위가 없어 spillover 비교를 적용하지 않습니다.",
        ),
        "triggers": (True, None),
        "regulars": (True, None),
        "satisfaction": (True, None),
    }


def catalog_payload(policy: dict[str, Any] | None) -> dict[str, Any]:
    decisions = applicability(policy)
    items = []
    for spec in CATALOG:
        applicable, reason = decisions[spec.id]
        items.append(
            {
                "id": spec.id,
                "label": spec.label,
                "description": spec.description,
                "applicable": applicable,
                "disabled_reason": reason,
                "unknown": [],
            }
        )
    return {"items": items, "unknown": []}


def applicable_ids(policy: dict[str, Any] | None) -> list[str]:
    decisions = applicability(policy)
    return [spec.id for spec in CATALOG if decisions[spec.id][0]]


# --------------------------------------------------------------------------- #
# 보고서 v2 — 절 단위 카탈로그
# --------------------------------------------------------------------------- #
#
# v2 엔진은 Neo4j 가 아니라 run snapshot 파일만 읽는다. 그래서 "적용 가능"의
# 판정 근거도 정책 JSON + run 산출물의 존재 여부다. 여기서도 계산은 하지 않는다.

V2_CATALOG: tuple[AnalysisSpec, ...] = (
    AnalysisSpec("s1", "분석 개요", "무엇을 어떤 기준으로 분석했는지 밝힙니다. 항상 포함됩니다."),
    AnalysisSpec("s2", "정책 사양", "적용한 정책의 내용과 분위별 지급액을 표로 싣습니다."),
    AnalysisSpec("s3", "소비 총량 추이", "일자별 총액·건수·객단가 추이."),
    AnalysisSpec("s4", "시행 전후 겹쳐보기", "같은 길이의 전/후 구간을 한 축에 겹쳐 차이를 면으로 보여줍니다."),
    AnalysisSpec("s5", "업종별 전후 비교", "업종별 하루 평균 소비의 전후 비교."),
    AnalysisSpec("s6", "이중차분 (DID)", "시장 전체 추세를 걷어낸 정책 순효과."),
    AnalysisSpec("s7", "업종별 이중차분", "어떤 업종에서 정책 때문에 금액이 늘었는지."),
    AnalysisSpec("s8", "분위별 효과", "소비 분위별 1인당 소비 변화."),
    AnalysisSpec("s9", "소비 구조", "결제 구성·요일별·업종별·지역별 분포."),
    AnalysisSpec("s10", "일관성 검증", "보고서에 실린 수치를 다시 계산해 대조합니다. 항상 포함됩니다."),
    AnalysisSpec("s11", "근거와 한계", "수치의 출처와 해석상의 한계. 항상 포함됩니다."),
)

V2_REQUIRED = ("s1", "s10", "s11")


def v2_applicability(
    policy: dict[str, Any] | None,
    *,
    run_artifacts: dict[str, Any] | None = None,
) -> dict[str, tuple[bool, str | None]]:
    ctx = _policy(policy)
    artifacts = run_artifacts or {}
    has_events = bool(artifacts.get("events", True))
    has_metrics = bool(artifacts.get("metrics", True))
    effective_from = ctx.get("effective_from")
    grants = ctx.get("decile_grants") or ctx.get("income_grants")

    events_reason = None if has_events else "run 산출물에 events.jsonl 이 없어 소비 이벤트를 집계할 수 없습니다."
    did_reason = (
        events_reason
        or (None if effective_from else "정책 시행일(effective_from)이 없어 사전/사후를 나눌 수 없습니다.")
    )
    return {
        "s1": (True, None),
        "s2": (True, None),
        "s3": (has_events, events_reason),
        "s4": (has_events and bool(effective_from), did_reason),
        "s5": (has_events, events_reason),
        "s6": (has_events and bool(effective_from), did_reason),
        "s7": (has_events and bool(effective_from), did_reason),
        "s8": (
            has_metrics and bool(grants),
            None
            if (has_metrics and grants)
            else (
                "metrics/day_*.jsonl 이 없어 분위별 집계를 할 수 없습니다."
                if not has_metrics
                else "정책에 분위·소득 구간 지급액이 없어 분위별 효과를 나눌 수 없습니다."
            ),
        ),
        "s9": (has_events, events_reason),
        "s10": (True, None),
        "s11": (True, None),
    }


def v2_catalog_payload(
    policy: dict[str, Any] | None,
    *,
    run_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decisions = v2_applicability(policy, run_artifacts=run_artifacts)
    items = []
    for spec in V2_CATALOG:
        applicable, reason = decisions[spec.id]
        items.append(
            {
                "id": spec.id,
                "label": spec.label,
                "description": spec.description,
                "applicable": applicable,
                "required": spec.id in V2_REQUIRED,
                "disabled_reason": reason,
                "unknown": [],
            }
        )
    return {"items": items, "required": list(V2_REQUIRED), "unknown": []}


def v2_applicable_ids(
    policy: dict[str, Any] | None,
    *,
    run_artifacts: dict[str, Any] | None = None,
) -> list[str]:
    decisions = v2_applicability(policy, run_artifacts=run_artifacts)
    return [spec.id for spec in V2_CATALOG if decisions[spec.id][0]]
