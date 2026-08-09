"""DASOL 보고서 v2 — 계산 엔진.

이 모듈은 **오직 run snapshot 파일만 읽는다.** Neo4j·GPU·LLM에 접근하지 않으며,
숫자를 만들어내지 않는다. 모든 결과에는 어떤 파일에서 나왔는지(`sources`)가 붙는다.

읽는 산출물
-----------
``<run_root>/events.jsonl``          소비 이벤트 1건 = 1행 (export 단계 산출)
``<run_root>/metrics/day_*.jsonl``   에이전트×일자 지표
``<run_root>/timing/day_*.json``     일자별 소요·정책 지급
``<run_root>/summary.json``          일자별 요약
``<run_root>/poi_summary.json``      가맹점 수

핵심 산출
---------
* 일자별 시계열 (총액·건수·정책지급·추가지출)
* 업종(L1)별 시행 전/후 비교와 증감
* **이중차분(DID)** — 정책 대상 업종(처치군) vs 비대상 업종(대조군)의 2×2,
  그리고 업종별 반사실(counterfactual) 기반 DID
* 시행 전/후를 같은 상대일 축에 겹쳐 그리기 위한 overlay 시리즈
* 분위(decile)별 지급·소비
* 위 결과들 사이의 **일관성 검증**에 필요한 원장(ledger) 값

DID 정의 (보고서·화면·문서에서 동일하게 사용한다)
------------------------------------------------
처치군 T = 정책 대상 업종 집합, 대조군 C = 그 외 업종 집합.
사전기간 P0 = 시행일 이전 일자, 사후기간 P1 = 시행일 이후(당일 포함) 일자.
각 값은 **일평균 금액**(기간 총액 / 기간 일수)으로 정규화한다.

    반사실(counterfactual)  T1* = T0 × (C1 / C0)
    DID(절대)              = T1 − T1*
    DID(상대)              = (T1/T0) − (C1/C0)

업종 c 하나에 대해서도 같은 식을 쓴다 (대조군은 언제나 비대상 업종 전체).
이 정의 덕분에 **대상 업종별 DID의 합 = 처치군 전체 DID** 가 성립하며,
`consistency.py` 가 이 항등식을 실제로 검사한다.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

DAY_FILE_RE = re.compile(r"day_(\d{4}-\d{2}-\d{2})\.jsonl$")

# events.jsonl 의 축약 키 → 보고서에서 쓰는 이름.
# 키가 없으면 그 값은 계산에서 빠지고 `unknown` 에 기록된다. 기본값을 지어내지 않는다.
EVENT_FIELDS = {
    "day": ("day",),
    "l1": ("l1", "cat_l1", "category_l1"),
    "l2": ("l2", "sub", "cat_l2"),
    "amt": ("amt", "amount"),
    "extra": ("ex", "extra", "extra_spent"),
    "eligible": ("elig", "eligible"),
    "would_buy_anyway": ("wba",),
    "policy_spend": ("sp", "policy_spend"),
    "day_type": ("day_type", "daytype"),
    "agent": ("aid", "agent_id"),
    "district": ("gu", "district", "sgg"),
    "dong": ("dong",),
}

UNCLASSIFIED = "미분류"


class AnalyticsError(Exception):
    """계산을 계속할 수 없는 상태 — 추정값으로 대체하지 않는다."""


# --------------------------------------------------------------------------- #
# 원본 읽기
# --------------------------------------------------------------------------- #


def _pick(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _number(value: Any) -> float:
    if value is None or value is False:
        return 0.0
    if value is True:
        return 1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _policy_split(value: Any) -> dict[str, float]:
    """``sp`` 는 JSON 문자열이거나 이미 dict 다. 둘 다 받는다."""
    if isinstance(value, dict):
        raw = value
    elif isinstance(value, str) and value.strip():
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return {}
    else:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): _number(v) for k, v in raw.items()}


def iter_events(run_root: Path) -> Iterator[dict[str, Any]]:
    """``events.jsonl`` 을 정규화된 행으로 흘려보낸다 (전체를 메모리에 올리지 않는다)."""
    path = Path(run_root) / "events.jsonl"
    if not path.is_file():
        raise AnalyticsError(f"events.jsonl 이 없습니다: {path}")
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            policy = _policy_split(_pick(row, EVENT_FIELDS["policy_spend"]))
            yield {
                "day": _pick(row, EVENT_FIELDS["day"]),
                "l1": _pick(row, EVENT_FIELDS["l1"]) or UNCLASSIFIED,
                "l2": _pick(row, EVENT_FIELDS["l2"]),
                "amt": _number(_pick(row, EVENT_FIELDS["amt"])),
                "extra": _number(_pick(row, EVENT_FIELDS["extra"])),
                "eligible": bool(_pick(row, EVENT_FIELDS["eligible"])),
                "wba": _pick(row, EVENT_FIELDS["would_buy_anyway"]) is True,
                "policy_by_id": policy,
                "policy_paid": sum(policy.values()),
                "day_type": _pick(row, EVENT_FIELDS["day_type"]),
                "agent": _pick(row, EVENT_FIELDS["agent"]),
                "district": _pick(row, EVENT_FIELDS["district"]),
                "_keys": tuple(row.keys()),
            }


def _empty_cell() -> dict[str, float]:
    return {
        "events": 0.0,
        "amt": 0.0,
        "policy_paid": 0.0,
        "self_paid": 0.0,
        "extra": 0.0,
        "eligible": 0.0,
        "wba": 0.0,
    }


def _add(cell: dict[str, float], row: dict[str, Any]) -> None:
    cell["events"] += 1
    cell["amt"] += row["amt"]
    cell["policy_paid"] += row["policy_paid"]
    cell["self_paid"] += max(row["amt"] - row["policy_paid"], 0.0)
    cell["extra"] += row["extra"]
    cell["eligible"] += 1.0 if row["eligible"] else 0.0
    cell["wba"] += 1.0 if row["wba"] else 0.0


def scan_events(run_root: Path) -> dict[str, Any]:
    """events.jsonl 1회 스캔으로 필요한 모든 교차표를 만든다."""
    by_day: dict[str, dict[str, float]] = defaultdict(_empty_cell)
    by_l1: dict[str, dict[str, float]] = defaultdict(_empty_cell)
    by_day_l1: dict[tuple[str, str], dict[str, float]] = defaultdict(_empty_cell)
    by_l2: dict[tuple[str, str], dict[str, float]] = defaultdict(_empty_cell)
    # 아래 셋은 **일자를 키에 포함**한다. 분석 창(start/days)을 잘랐을 때
    # 이 표들만 전체 기간을 보면 총계와 어긋난다. 같은 모집단을 보게 한다.
    by_day_daytype: dict[tuple[str, str], dict[str, float]] = defaultdict(_empty_cell)
    by_day_district: dict[tuple[str, str], dict[str, float]] = defaultdict(_empty_cell)
    policy_ids_by_day: dict[tuple[str, str], float] = defaultdict(float)
    agents_by_day: dict[str, set] = defaultdict(set)
    totals = _empty_cell()
    seen_keys: set[str] = set()
    rows = 0

    for row in iter_events(run_root):
        rows += 1
        seen_keys.update(row["_keys"])
        day = str(row["day"]) if row["day"] else UNCLASSIFIED
        l1 = str(row["l1"])
        _add(by_day[day], row)
        _add(by_l1[l1], row)
        _add(by_day_l1[(day, l1)], row)
        if row["l2"]:
            _add(by_l2[(l1, str(row["l2"]))], row)
        if row["day_type"]:
            _add(by_day_daytype[(day, str(row["day_type"]))], row)
        if row["district"]:
            _add(by_day_district[(day, str(row["district"]))], row)
        if row["agent"] is not None:
            agents_by_day[day].add(row["agent"])
        for pid, amount in row["policy_by_id"].items():
            policy_ids_by_day[(day, pid)] += amount
        _add(totals, row)

    return {
        "rows": rows,
        "totals": totals,
        "by_day": {k: dict(v) for k, v in by_day.items()},
        "by_l1": {k: dict(v) for k, v in by_l1.items()},
        "by_day_l1": {k: dict(v) for k, v in by_day_l1.items()},
        "by_l2": {k: dict(v) for k, v in by_l2.items()},
        "by_day_daytype": {k: dict(v) for k, v in by_day_daytype.items()},
        "by_day_district": {k: dict(v) for k, v in by_day_district.items()},
        "policy_paid_by_day_policy_id": dict(policy_ids_by_day),
        "agents_by_day": {k: len(v) for k, v in agents_by_day.items()},
        "event_keys": sorted(seen_keys),
    }


def read_metrics(run_root: Path, days: list[str]) -> dict[str, Any]:
    """``metrics/day_*.jsonl`` 에서 분위·만족도·잔액만 뽑는다 (전 필드 집계는 하지 않는다)."""
    root = Path(run_root) / "metrics"
    per_day: dict[str, Any] = {}
    if not root.is_dir():
        return {"available": False, "reason": f"metrics 디렉터리가 없습니다: {root}", "days": {}}
    for day in days:
        path = root / f"day_{day}.jsonl"
        if not path.is_file():
            continue
        deciles: dict[int, dict[str, float]] = defaultdict(
            lambda: {
                "agents": 0.0,
                "grant_applied_today": 0.0,
                "grant_remaining_total": 0.0,
                "policy_spend_today": 0.0,
                "spend_total": 0.0,
            }
        )
        sat_total = 0.0
        sat_n = 0
        agents = 0
        unknown_decile = 0
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                agents += 1
                raw_decile = row.get("spend_decile")
                try:
                    decile = int(raw_decile)
                except (TypeError, ValueError):
                    unknown_decile += 1
                    continue
                bucket = deciles[decile]
                bucket["agents"] += 1
                bucket["grant_applied_today"] += _number(row.get("cm_intended_grant_today"))
                bucket["grant_remaining_total"] += _number(row.get("cm_grant_carry_out"))
                bucket["policy_spend_today"] += _number(row.get("cm_policy_allocated_total"))
                bucket["spend_total"] += _number(row.get("cm_personal_total")) + _number(
                    row.get("cm_online_total")
                )
                if row.get("avg_sat") is not None:
                    sat_total += _number(row.get("avg_sat"))
                    sat_n += 1
        per_day[day] = {
            "agents": agents,
            "unknown_decile_agents": unknown_decile,
            "avg_satisfaction": round(sat_total / sat_n, 4) if sat_n else None,
            "deciles": {str(k): {key: round(val, 2) for key, val in v.items()} for k, v in sorted(deciles.items())},
        }
    return {"available": bool(per_day), "reason": None if per_day else "metrics 일자 파일이 없습니다", "days": per_day}


def read_summary(run_root: Path) -> dict[str, Any] | None:
    path = Path(run_root) / "summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_poi_summary(run_root: Path) -> dict[str, Any] | None:
    path = Path(run_root) / "poi_summary.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


# --------------------------------------------------------------------------- #
# 기간 분할
# --------------------------------------------------------------------------- #


def _iso(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def split_period(days: list[str], policy_from: str | None) -> dict[str, Any]:
    """시행일 기준으로 사전/사후를 나눈다. 시행일 당일은 **사후**에 포함한다."""
    ordered = sorted(days)
    cut = _iso(policy_from)
    if cut is None:
        return {
            "policy_from": None,
            "pre": [],
            "post": ordered,
            "usable": False,
            "reason": "정책 시행일이 없어 사전/사후를 나눌 수 없습니다.",
        }
    pre = [d for d in ordered if (_iso(d) or cut) < cut]
    post = [d for d in ordered if (_iso(d) or cut) >= cut]
    reason = None
    if not pre:
        reason = "시행일 이전 일자가 run 안에 없어 이중차분을 계산할 수 없습니다."
    elif not post:
        reason = "시행일 이후 일자가 run 안에 없어 이중차분을 계산할 수 없습니다."
    return {
        "policy_from": cut.isoformat(),
        "pre": pre,
        "post": post,
        "usable": bool(pre and post),
        "reason": reason,
    }


def target_categories(policy: dict[str, Any], observed: Iterable[str], scan: dict[str, Any]) -> dict[str, Any]:
    """정책 대상 업종을 정한다.

    1순위: 정책 JSON 의 ``benefit_categories``/``target_cats`` 중 실제 관측된 업종
    2순위: 정책 지급액이 실제로 발생한 업종 (파일 근거 기반)
    둘 다 없으면 처치군을 만들 수 없다고 명시한다 — 임의로 고르지 않는다.
    """
    observed_set = {str(item) for item in observed}
    declared = policy.get("benefit_categories") or policy.get("target_cats") or []
    declared = [str(item) for item in declared] if isinstance(declared, list) else []
    matched = [item for item in declared if item in observed_set]
    if matched:
        return {
            "source": "policy.benefit_categories",
            "categories": sorted(matched),
            "declared": declared,
            "unmatched": sorted(set(declared) - observed_set),
        }
    paid = sorted(
        (name for name, cell in scan["by_l1"].items() if cell["policy_paid"] > 0),
        key=lambda name: -scan["by_l1"][name]["policy_paid"],
    )
    if paid:
        return {
            "source": "events.jsonl 의 정책 지급 실적",
            "categories": paid,
            "declared": declared,
            "unmatched": sorted(set(declared) - observed_set),
        }
    return {
        "source": None,
        "categories": [],
        "declared": declared,
        "unmatched": sorted(set(declared) - observed_set),
    }


# --------------------------------------------------------------------------- #
# 이중차분
# --------------------------------------------------------------------------- #


def _period_cell(scan: dict[str, Any], days: list[str], categories: Iterable[str] | None) -> dict[str, float]:
    cell = _empty_cell()
    wanted = None if categories is None else {str(c) for c in categories}
    for (day, l1), value in scan["by_day_l1"].items():
        if day not in days:
            continue
        if wanted is not None and l1 not in wanted:
            continue
        for key in cell:
            cell[key] += value[key]
    return cell


def _daily_average(cell: dict[str, float], n_days: int) -> dict[str, float]:
    if n_days <= 0:
        return {key: 0.0 for key in cell}
    return {key: value / n_days for key, value in cell.items()}


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def did_two_by_two(
    scan: dict[str, Any],
    period: dict[str, Any],
    treat: list[str],
    control: list[str],
    *,
    metric: str = "amt",
) -> dict[str, Any]:
    """처치군/대조군 × 사전/사후 2×2 이중차분."""
    pre_days, post_days = period["pre"], period["post"]
    t0 = _daily_average(_period_cell(scan, pre_days, treat), len(pre_days))[metric]
    t1 = _daily_average(_period_cell(scan, post_days, treat), len(post_days))[metric]
    c0 = _daily_average(_period_cell(scan, pre_days, control), len(pre_days))[metric]
    c1 = _daily_average(_period_cell(scan, post_days, control), len(post_days))[metric]
    growth_control = _ratio(c1, c0)
    counterfactual = t0 * growth_control if growth_control is not None else None
    did_abs = (t1 - counterfactual) if counterfactual is not None else None
    growth_treat = _ratio(t1, t0)
    did_rel = (growth_treat - growth_control) if (growth_treat is not None and growth_control is not None) else None
    return {
        "metric": metric,
        "treat_categories": sorted(treat),
        "control_categories": sorted(control),
        "pre_days": len(pre_days),
        "post_days": len(post_days),
        "treat_pre": round(t0, 2),
        "treat_post": round(t1, 2),
        "control_pre": round(c0, 2),
        "control_post": round(c1, 2),
        "treat_diff": round(t1 - t0, 2),
        "control_diff": round(c1 - c0, 2),
        "treat_growth": round(growth_treat, 6) if growth_treat is not None else None,
        "control_growth": round(growth_control, 6) if growth_control is not None else None,
        "counterfactual_post": round(counterfactual, 2) if counterfactual is not None else None,
        "did_absolute": round(did_abs, 2) if did_abs is not None else None,
        "did_relative": round(did_rel, 6) if did_rel is not None else None,
        "did_pct_of_counterfactual": (
            round(did_abs / counterfactual * 100, 4)
            if (did_abs is not None and counterfactual not in (None, 0))
            else None
        ),
        "naive_before_after": round(t1 - t0, 2),
        "bias_removed": round((t1 - t0) - did_abs, 2) if did_abs is not None else None,
    }


def did_by_category(
    scan: dict[str, Any],
    period: dict[str, Any],
    treat: list[str],
    control: list[str],
    *,
    metric: str = "amt",
) -> list[dict[str, Any]]:
    """업종별 이중차분. 대조군은 항상 비대상 업종 전체다.

    같은 counterfactual 배율(대조군 성장률)을 쓰기 때문에
    ``sum(대상 업종 did_absolute) == 처치군 전체 did_absolute`` 가 성립한다.
    """
    pre_days, post_days = period["pre"], period["post"]
    control_growth = _ratio(
        _daily_average(_period_cell(scan, post_days, control), len(post_days))[metric],
        _daily_average(_period_cell(scan, pre_days, control), len(pre_days))[metric],
    )
    treat_set = {str(c) for c in treat}
    rows: list[dict[str, Any]] = []
    for l1 in sorted(scan["by_l1"]):
        pre = _daily_average(_period_cell(scan, pre_days, [l1]), len(pre_days))
        post = _daily_average(_period_cell(scan, post_days, [l1]), len(post_days))
        counterfactual = pre[metric] * control_growth if control_growth is not None else None
        did_abs = (post[metric] - counterfactual) if counterfactual is not None else None
        growth = _ratio(post[metric], pre[metric])
        rows.append(
            {
                "l1": l1,
                "targeted": l1 in treat_set,
                "pre_daily": round(pre[metric], 2),
                "post_daily": round(post[metric], 2),
                "delta": round(post[metric] - pre[metric], 2),
                "growth": round(growth, 6) if growth is not None else None,
                "growth_pct": round((growth - 1) * 100, 4) if growth is not None else None,
                "counterfactual_post": round(counterfactual, 2) if counterfactual is not None else None,
                "did_absolute": round(did_abs, 2) if did_abs is not None else None,
                "did_pct": (
                    round(did_abs / counterfactual * 100, 4)
                    if (did_abs is not None and counterfactual not in (None, 0))
                    else None
                ),
                "pre_events": round(_period_cell(scan, pre_days, [l1])["events"], 0),
                "post_events": round(_period_cell(scan, post_days, [l1])["events"], 0),
                "policy_paid": round(scan["by_l1"][l1]["policy_paid"], 2),
                "self_paid": round(scan["by_l1"][l1]["self_paid"], 2),
                "amt": round(scan["by_l1"][l1]["amt"], 2),
            }
        )
    rows.sort(key=lambda item: -(item["did_absolute"] or 0))
    return rows


def event_study(
    scan: dict[str, Any],
    period: dict[str, Any],
    treat: list[str],
    control: list[str],
    *,
    metric: str = "amt",
) -> dict[str, Any]:
    """상대일(시행일=0) 기준 처치−대조 로그격차. 사전 평균을 0으로 정규화한다."""
    cut = _iso(period.get("policy_from"))
    if cut is None:
        return {"available": False, "reason": "정책 시행일이 없습니다", "points": []}
    treat_set = {str(c) for c in treat}
    control_set = {str(c) for c in control}
    per_day: dict[str, dict[str, float]] = defaultdict(lambda: {"t": 0.0, "c": 0.0})
    for (day, l1), cell in scan["by_day_l1"].items():
        if l1 in treat_set:
            per_day[day]["t"] += cell[metric]
        elif l1 in control_set:
            per_day[day]["c"] += cell[metric]
    points: list[dict[str, Any]] = []
    for day in sorted(per_day):
        d = _iso(day)
        if d is None:
            continue
        t, c = per_day[day]["t"], per_day[day]["c"]
        gap = math.log(t / c) if (t > 0 and c > 0) else None
        points.append(
            {
                "day": day,
                "rel_day": (d - cut).days,
                "treat": round(t, 2),
                "control": round(c, 2),
                "log_gap": round(gap, 6) if gap is not None else None,
            }
        )
    pre_gaps = [p["log_gap"] for p in points if p["rel_day"] < 0 and p["log_gap"] is not None]
    baseline = sum(pre_gaps) / len(pre_gaps) if pre_gaps else None
    for point in points:
        point["normalized_gap"] = (
            round(point["log_gap"] - baseline, 6)
            if (point["log_gap"] is not None and baseline is not None)
            else None
        )
    return {
        "available": bool(points) and baseline is not None,
        "reason": None if baseline is not None else "사전기간 관측이 없어 정규화 기준선을 만들 수 없습니다",
        "baseline_log_gap": round(baseline, 6) if baseline is not None else None,
        "points": points,
    }


def overlay_series(scan: dict[str, Any], period: dict[str, Any], *, metric: str = "amt") -> dict[str, Any]:
    """시행 전/후를 **같은 상대일 축**에 겹쳐 그리기 위한 시리즈.

    사전기간은 시행일 직전일을 −1 로 두고 뒤에서부터,
    사후기간은 시행일을 0 으로 두고 앞에서부터 센다.
    두 구간을 각각 1..n 축으로 접어 같은 x 위에 올린다.
    """
    pre, post = period["pre"], period["post"]
    if not pre or not post:
        return {"available": False, "reason": period.get("reason"), "overall": [], "by_category": {}}
    span = min(len(pre), len(post))
    pre_window = pre[-span:]
    post_window = post[:span]

    def _series(days: list[str], categories: list[str] | None) -> list[float]:
        out = []
        for day in days:
            if categories is None:
                out.append(round(scan["by_day"].get(day, _empty_cell())[metric], 2))
            else:
                total = 0.0
                for c in categories:
                    total += scan["by_day_l1"].get((day, c), _empty_cell())[metric]
                out.append(round(total, 2))
        return out

    overall = {
        "labels": [f"{i + 1}일차" for i in range(span)],
        "pre_days": pre_window,
        "post_days": post_window,
        "pre": _series(pre_window, None),
        "post": _series(post_window, None),
    }
    overall["delta"] = [round(b - a, 2) for a, b in zip(overall["pre"], overall["post"])]

    by_category: dict[str, Any] = {}
    for l1 in sorted(scan["by_l1"], key=lambda name: -scan["by_l1"][name]["amt"]):
        pre_vals = _series(pre_window, [l1])
        post_vals = _series(post_window, [l1])
        by_category[l1] = {
            "pre": pre_vals,
            "post": post_vals,
            "delta": [round(b - a, 2) for a, b in zip(pre_vals, post_vals)],
        }
    return {
        "available": True,
        "reason": None,
        "window_days": span,
        "note": (
            f"사전 {len(pre)}일 중 마지막 {span}일과 사후 {len(post)}일 중 처음 {span}일을 "
            "같은 길이로 잘라 겹쳤습니다. 길이를 맞추지 않으면 두 곡선의 면적을 비교할 수 없습니다."
        ),
        "overall": overall,
        "by_category": by_category,
    }


# --------------------------------------------------------------------------- #
# 번들
# --------------------------------------------------------------------------- #


def _sorted_days(scan: dict[str, Any], start: str | None, days: int | None) -> list[str]:
    observed = sorted(d for d in scan["by_day"] if _iso(d) is not None)
    if start is None or days is None:
        return observed
    first = _iso(start)
    if first is None:
        return observed
    window = {(first + timedelta(days=i)).isoformat() for i in range(days)}
    selected = [d for d in observed if d in window]
    return selected or observed


def build_bundle(
    *,
    run_id: str,
    run_root: Path,
    policy: dict[str, Any],
    policy_from: str | None = None,
    start: str | None = None,
    days: int | None = None,
    metric: str = "amt",
) -> dict[str, Any]:
    """보고서 한 편에 필요한 모든 계산 결과를 한 번에 만든다."""
    run_root = Path(run_root)
    scan = scan_events(run_root)
    observed_days = _sorted_days(scan, start, days)
    if not observed_days:
        raise AnalyticsError("events.jsonl 에서 유효한 일자를 찾지 못했습니다")

    # 요청 창 밖의 날짜를 교차표에서 제거해 모든 섹션이 같은 모집단을 본다.
    window = set(observed_days)
    scan["by_day"] = {k: v for k, v in scan["by_day"].items() if k in window}
    scan["by_day_l1"] = {k: v for k, v in scan["by_day_l1"].items() if k[0] in window}
    recomputed_l1: dict[str, dict[str, float]] = defaultdict(_empty_cell)
    recomputed_totals = _empty_cell()
    for (day, l1), cell in scan["by_day_l1"].items():
        for key, value in cell.items():
            recomputed_l1[l1][key] += value
            recomputed_totals[key] += value
    scan["by_l1"] = {k: dict(v) for k, v in recomputed_l1.items()}
    scan["totals"] = dict(recomputed_totals)

    # 요일유형·지역·정책ID 표도 같은 창으로 접는다. 접지 않으면 이 세 표만
    # 전체 기간을 보게 되어 총계와 어긋난다 (consistency 의 policy_id_sum 이 잡는다).
    def _rollup(source: dict[tuple[str, str], dict[str, float]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = defaultdict(_empty_cell)
        for (day, key), cell in source.items():
            if day not in window:
                continue
            for name, value in cell.items():
                out[key][name] += value
        return {k: dict(v) for k, v in out.items()}

    scan["by_daytype"] = _rollup(scan["by_day_daytype"])
    scan["by_district"] = _rollup(scan["by_day_district"])
    policy_totals: dict[str, float] = defaultdict(float)
    for (day, pid), amount in scan["policy_paid_by_day_policy_id"].items():
        if day in window:
            policy_totals[pid] += amount
    scan["policy_paid_by_policy_id"] = dict(policy_totals)

    effective_from = policy_from or policy.get("effective_from")
    period = split_period(observed_days, effective_from)
    targets = target_categories(policy, scan["by_l1"].keys(), scan)
    treat = targets["categories"]
    control = [name for name in sorted(scan["by_l1"]) if name not in set(treat)]

    metrics = read_metrics(run_root, observed_days)
    summary = read_summary(run_root)
    poi = read_poi_summary(run_root)

    daily = []
    for day in observed_days:
        cell = scan["by_day"].get(day, _empty_cell())
        phase = "post" if day in period["post"] else ("pre" if day in period["pre"] else "unknown")
        agents = metrics["days"].get(day, {}).get("agents") if metrics.get("available") else None
        daily.append(
            {
                "day": day,
                "phase": phase,
                "events": int(cell["events"]),
                "amt": round(cell["amt"], 2),
                "policy_paid": round(cell["policy_paid"], 2),
                "self_paid": round(cell["self_paid"], 2),
                "extra": round(cell["extra"], 2),
                "eligible_events": int(cell["eligible"]),
                "would_buy_anyway": int(cell["wba"]),
                "avg_ticket": round(cell["amt"] / cell["events"], 2) if cell["events"] else None,
                "agents": agents,
                "per_capita": round(cell["amt"] / agents, 2) if agents else None,
                "avg_satisfaction": metrics["days"].get(day, {}).get("avg_satisfaction"),
            }
        )

    did_summary = None
    category_rows: list[dict[str, Any]] = []
    study: dict[str, Any] = {"available": False, "reason": period.get("reason"), "points": []}
    if period["usable"] and treat and control:
        did_summary = did_two_by_two(scan, period, treat, control, metric=metric)
        category_rows = did_by_category(scan, period, treat, control, metric=metric)
        study = event_study(scan, period, treat, control, metric=metric)
    elif period["usable"] and not control:
        did_summary = None
        category_rows = []
        study = {
            "available": False,
            "reason": "모든 업종이 정책 대상이라 대조군을 만들 수 없습니다",
            "points": [],
        }

    # 업종별 사전/사후 (DID 가 불가능해도 이 표는 만든다)
    pre_days, post_days = period["pre"], period["post"]
    categories = []
    for l1 in sorted(scan["by_l1"], key=lambda name: -scan["by_l1"][name]["amt"]):
        pre_cell = _period_cell(scan, pre_days, [l1])
        post_cell = _period_cell(scan, post_days, [l1])
        pre_daily = _daily_average(pre_cell, len(pre_days))
        post_daily = _daily_average(post_cell, len(post_days))
        total = scan["by_l1"][l1]
        categories.append(
            {
                "l1": l1,
                "targeted": l1 in set(treat),
                "amt": round(total["amt"], 2),
                "events": int(total["events"]),
                "policy_paid": round(total["policy_paid"], 2),
                "self_paid": round(total["self_paid"], 2),
                "extra": round(total["extra"], 2),
                "share": round(total["amt"] / scan["totals"]["amt"] * 100, 4) if scan["totals"]["amt"] else None,
                "pre_daily_amt": round(pre_daily["amt"], 2),
                "post_daily_amt": round(post_daily["amt"], 2),
                "delta_daily_amt": round(post_daily["amt"] - pre_daily["amt"], 2),
                "growth_pct": (
                    round((post_daily["amt"] / pre_daily["amt"] - 1) * 100, 4) if pre_daily["amt"] else None
                ),
                "avg_ticket": round(total["amt"] / total["events"], 2) if total["events"] else None,
                "policy_share_pct": (
                    round(total["policy_paid"] / total["amt"] * 100, 4) if total["amt"] else None
                ),
            }
        )

    decile_rows = _decile_rollup(metrics, period)
    mix = {
        "policy_paid": round(scan["totals"]["policy_paid"], 2),
        "self_paid": round(scan["totals"]["self_paid"], 2),
        "extra": round(scan["totals"]["extra"], 2),
        "amt": round(scan["totals"]["amt"], 2),
        "leverage": (
            round(scan["totals"]["amt"] / scan["totals"]["policy_paid"], 4)
            if scan["totals"]["policy_paid"]
            else None
        ),
        "deadweight_events": int(scan["totals"]["wba"]),
        "deadweight_share_pct": (
            round(scan["totals"]["wba"] / scan["totals"]["events"] * 100, 4)
            if scan["totals"]["events"]
            else None
        ),
    }

    return {
        "schema": "dasol.report.v2",
        "meta": {
            "run_id": run_id,
            "run_root": str(run_root),
            "generated_from": "events.jsonl + metrics/day_*.jsonl",
            "metric": metric,
            "policy_id": policy.get("id"),
            "policy_name": policy.get("name"),
            "policy_type": policy.get("type"),
            "policy_effective_from": policy.get("effective_from"),
            "policy_effective_until": policy.get("effective_until"),
            "policy_from_used": period.get("policy_from"),
            "requested_start": start,
            "requested_days": days,
            "days": observed_days,
            "day_count": len(observed_days),
            "event_rows": scan["rows"],
            "event_keys": scan["event_keys"],
            "poi_summary": poi,
            "summary_present": summary is not None,
        },
        "period": period,
        "targets": targets,
        "control_categories": control,
        "totals": {key: round(value, 2) for key, value in scan["totals"].items()},
        "daily": daily,
        "categories": categories,
        "did": did_summary,
        "did_by_category": category_rows,
        "event_study": study,
        "overlay": overlay_series(scan, period, metric=metric),
        "deciles": decile_rows,
        "daytype": {k: {key: round(val, 2) for key, val in v.items()} for k, v in scan["by_daytype"].items()},
        "districts": {k: {key: round(val, 2) for key, val in v.items()} for k, v in scan["by_district"].items()},
        "policy_paid_by_policy_id": {k: round(v, 2) for k, v in scan["policy_paid_by_policy_id"].items()},
        "mix": mix,
        "metrics_available": metrics.get("available", False),
        "metrics_reason": metrics.get("reason"),
        "unknown": _unknown_fields(scan, metrics, period, treat, control),
    }


def _decile_rollup(metrics: dict[str, Any], period: dict[str, Any]) -> dict[str, Any]:
    """분위별 지급·소비를 사전/사후로 나눠 합친다."""
    if not metrics.get("available"):
        return {"available": False, "reason": metrics.get("reason"), "items": []}
    buckets: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "agents_pre": 0.0,
            "agents_post": 0.0,
            "grant_pre": 0.0,
            "grant_post": 0.0,
            "spend_pre": 0.0,
            "spend_post": 0.0,
            "policy_spend_pre": 0.0,
            "policy_spend_post": 0.0,
        }
    )
    pre, post = set(period["pre"]), set(period["post"])
    for day, payload in metrics["days"].items():
        suffix = "pre" if day in pre else ("post" if day in post else None)
        if suffix is None:
            continue
        for decile, values in payload["deciles"].items():
            bucket = buckets[decile]
            bucket[f"agents_{suffix}"] += values["agents"]
            bucket[f"grant_{suffix}"] += values["grant_applied_today"]
            bucket[f"spend_{suffix}"] += values["spend_total"]
            bucket[f"policy_spend_{suffix}"] += values["policy_spend_today"]
    items = []
    for decile in sorted(buckets, key=lambda value: int(value)):
        bucket = buckets[decile]
        pre_per = bucket["spend_pre"] / bucket["agents_pre"] if bucket["agents_pre"] else None
        post_per = bucket["spend_post"] / bucket["agents_post"] if bucket["agents_post"] else None
        items.append(
            {
                "decile": int(decile),
                "agents_pre": int(bucket["agents_pre"]),
                "agents_post": int(bucket["agents_post"]),
                "grant_total": round(bucket["grant_pre"] + bucket["grant_post"], 2),
                "policy_spend_total": round(bucket["policy_spend_pre"] + bucket["policy_spend_post"], 2),
                "per_capita_pre": round(pre_per, 2) if pre_per is not None else None,
                "per_capita_post": round(post_per, 2) if post_per is not None else None,
                "per_capita_delta": (
                    round(post_per - pre_per, 2) if (pre_per is not None and post_per is not None) else None
                ),
                "per_capita_growth_pct": (
                    round((post_per / pre_per - 1) * 100, 4)
                    if (pre_per not in (None, 0) and post_per is not None)
                    else None
                ),
                "treated": bucket["grant_pre"] + bucket["grant_post"] > 0,
            }
        )
    return {"available": bool(items), "reason": None if items else "분위 정보를 가진 행이 없습니다", "items": items}


def _unknown_fields(
    scan: dict[str, Any],
    metrics: dict[str, Any],
    period: dict[str, Any],
    treat: list[str],
    control: list[str],
) -> list[str]:
    unknown: list[str] = []
    if not period["usable"]:
        unknown.append("did")
    if not treat:
        unknown.append("target_categories")
    if not control:
        unknown.append("control_categories")
    if not metrics.get("available"):
        unknown.append("deciles")
    if not scan["by_district"]:
        unknown.append("districts")
    if not scan["agents_by_day"]:
        unknown.append("per_capita")
    return unknown
