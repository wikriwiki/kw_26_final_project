"""One experiment's complete policy-indicator coverage and empirical comparison.

Usage::

    python scripts/report/build_experiment_comparison.py --score output/run/score.json
    python scripts/report/build_experiment_comparison.py --experiment suite_01 \
        --score output/run/p010.json --score output/run/p012.json \
        --out output/run/comparison.html

Every indicator in the registered scoring table appears, including policies not
run in this experiment. A numerical gap is emitted only after a *run-specific*
estimand audit; an old score cannot inherit an audit registered for a new window.
This report runs after scoring and is never an input to the citizen prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORING = ROOT / "data/experiments/scoring_table.json"
TEMPLATE = ROOT / "output/report/experiment_comparison_tpl.html"
COMPARABILITY_NOTES = ROOT / "data/experiments/indicator_comparability_notes_20260927.json"
EMPIRICAL_REGISTRY = ROOT / "experiments/multi_policy_v53_20260927/empirical_registry.json"

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.report.all_indicators_table import (  # noqa: E402
    direct_comparison_audited, truth_gap, truth_of,
)
from scripts.report.sign_scoreboard import direction_comparison_audited  # noqa: E402

EXPECT = {"+": "증가", "-": "감소", "0": "무반응", "rank": "순위", "info": "참고"}
POLICY_NAMES = {
    "P010": "민생회복소비쿠폰 P010", "P012": "상생소비지원금 P012",
    "EMERGENCY_2020": "긴급재난지원금 P013", "DISTANCING_2020": "사회적 거리두기",
    "GATHERING_2020": "사적모임 제한", "LOCAL_VOUCHER": "지역상품권 P014",
    "SECTOR_VOUCHER_2020": "8대 소비쿠폰 P015", "PLACEBO_FAKE": "가짜 정책 위약",
    "PLACEBO_TIMING": "시점 위약", "P016": "농할 할인 P016",
}
EXPLORATORY_NAMES = {
    "P010-BOK-RESTAURANT": "외식 사용액 비중", "P010-BOK-MART_FOOD": "마트·식료품 사용액 비중",
    "P010-BOK-MEDICAL": "의료 사용액 비중", "P010-BOK-BEAUTY": "미용 사용액 비중",
    "P010-BOK-ACADEMY": "학원 사용액 비중", "P010-BOK-PHARMACY": "약국 사용액 비중",
    "P014-KIPF-47121": "슈퍼마켓 매출 계수", "P014-KIPF-47129": "음식료 소매 매출 계수",
    "DS6-2023-GEO-PROXY": "상권 유형 차이의 2023년 경계 탐색 대리값",
}
POLICY_ID_TO_SCORE = {
    "P010": "P010", "P012": "P012", "P013": "EMERGENCY_2020",
    "P014": "LOCAL_VOUCHER", "P015": "SECTOR_VOUCHER_2020",
    "P016": "P016", "P090": "PLACEBO_FAKE",
}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _reference(ind):
    audit = ind.get("empirical_audit") or {}
    rank = ind.get("expect") == "rank"
    if rank and _number(audit.get("reported_gap")):
        return audit["reported_gap"], audit.get("reported_unit"), "원문 감사값"
    if not rank and _number(audit.get("reported_value")):
        return audit["reported_value"], audit.get("reported_unit"), "원문 감사값"
    if rank:
        gap = truth_gap(ind.get("desc"))
        return gap, "%p" if gap is not None else None, "서술값·정의 미감사" if gap is not None else ""
    value, unit = truth_of(ind.get("desc"))
    return value, unit, "서술값·정의 미감사" if value is not None else ""


def _simulation(ind, result):
    """Return a display value/unit and CI without guessing a denominator."""
    if not result:
        return None, None, None
    metric = ind.get("metric", "")
    audited_unit = (ind.get("empirical_audit") or {}).get("simulation_unit")
    if ind.get("expect") == "rank":
        numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", str(result.get("got") or ""))
        if len(numbers) >= 2:
            return float(numbers[0]) - float(numbers[1]), "%p", None
        return None, None, None
    mean, base = result.get("mean"), result.get("base")
    ci = result.get("ci")
    ci = ci if isinstance(ci, list) and len(ci) == 2 and all(_number(v) for v in ci) else None
    if not _number(mean):
        return None, None, None
    if metric == "mpc_amount":
        return mean, "ratio", ci
    if metric in ("threshold_reach_rate", "cap_reach_rate"):
        return 100 * mean, "%", [100 * x for x in ci] if ci else None
    if metric == "cashback_per_capita":
        return mean, "원", ci
    if (metric.startswith(("sector_share:", "sector_share_within:"))
            or metric in ("elig_spend_share", "late_night_share", "out_district_ratio",
                          "home_dong_spend_share", "out_district_spend_share")):
        if audited_unit == "%" and _number(base) and base != 0:
            return 100 * mean / base, "%", sorted(100 * x / base for x in ci) if ci else None
        return 100 * mean, "%p", [100 * x for x in ci] if ci else None
    if metric == "home_hours":
        return mean, "시간", ci
    if audited_unit == "원":
        return mean, "원", ci
    if _number(base) and base != 0:
        interval = sorted(100 * x / base for x in ci) if ci else None
        return 100 * mean / base, "%", interval
    # Paired differences with zero/absent baseline have no defined percentage.
    unit = "비율차" if "share" in metric or "ratio" in metric else "원"
    return mean, unit, ci


def _run_audited(audit, score, table_hash):
    return (audit.get("simulation_off") == score.get("off")
            and audit.get("simulation_on") == score.get("on")
            and score.get("scoring_table_sha256") == table_hash)


def _row(policy, ind, score, result, table_hash):
    audit = ind.get("empirical_audit") or {}
    truth, truth_unit, truth_kind = _reference(ind)
    sim, sim_unit, ci = _simulation(ind, result)
    mismatch = bool(result and (result.get("metric") != ind.get("metric")
                                or result.get("expect") != ind.get("expect")))
    run_ok = bool(score and _run_audited(audit, score, table_hash))
    direct = bool(score and result and not mismatch and _number(sim) and _number(truth)
                  and direct_comparison_audited(ind) and run_ok
                  and sim_unit == truth_unit
                  and result.get("empirical_period_aligned") is not False
                  and result.get("measurement_complete") is not False)
    direction_ready = bool(score and result and not mismatch and _number(sim)
                           and direction_comparison_audited(ind) and run_ok
                           and result.get("empirical_period_aligned") is not False
                           and result.get("measurement_complete") is not False)
    external_direction = None
    if direction_ready:
        if ind.get("expect") in ("+", "-", "rank"):
            external_direction = (sim > 0 if ind["expect"] in ("+", "rank") else sim < 0)
        elif (ind.get("expect") == "0" and ci
              and _number(audit.get("equivalence_band"))
              and sim_unit == audit.get("simulation_unit")):
            external_direction = max(abs(x) for x in ci) <= audit["equivalence_band"]
    if score is None:
        status, reason = "미실행", "이 실험에 이 정책의 채점 파일이 없음"
    elif result is None:
        status, reason = "결과 누락", "채점 파일에 등록 지표 결과가 없음"
    elif mismatch:
        status, reason = "정의 변경", "채점 파일의 metric/expect가 현재 등록 정의와 다름"
    elif ind.get("expect") == "info":
        status, reason = "참고 지표", "채점 대상이 아닌 진단 지표"
    elif not _number(sim):
        status = str(result.get("got") or "관측 부족")
        reason = "시뮬레이션 수치가 없어 차이를 계산할 수 없음"
    elif result.get("empirical_period_aligned") is False:
        status, reason = "기간 불일치", "원문 관측 월·정책 노출 및 시민별 관측 기간이 맞지 않음"
    elif result.get("measurement_complete") is False:
        status, reason = "측정 미완료", "채점 파일의 측정 완결성 관문 실패"
    elif direct:
        status, reason = "직접 비교", "원문·실험 창·추정량·단위 감사 통과"
    elif external_direction is not None:
        status, reason = "외부 방향 대조", "원문 방향·실험 창의 대응 감사 통과, 크기 차감은 미승인"
    elif audit.get("comparison") == "not_observed":
        status, reason = "실측 없음", audit.get("reason") or "원문에 대응 지표가 없음"
    elif audit.get("comparison") == "different_estimand":
        status, reason = "다른 추정량", audit.get("reason") or "원문과 시뮬레이션 정의가 다름"
    elif truth is None:
        status = "외부 방향 미감사" if audit.get("source") else "내부 가설"
        reason = audit.get("reason") or "원문 수치·방향의 대응 감사가 등록되지 않음"
    elif direct_comparison_audited(ind) and not run_ok:
        status, reason = "실험 창 미감사", "이 채점 파일의 OFF/ON 창과 채점표 지문에 대한 감사가 없음"
    else:
        status, reason = "정의 미감사", audit.get("reason") or "출처·추정량·기간·모집단·분모 정합 감사가 필요함"
    # An unregistered/historical result is still visible, but never earns a gap.
    gap = sim - truth if direct else None
    return {
        "policy": policy, "policy_name": POLICY_NAMES.get(policy, policy),
        "id": ind["id"], "metric": ind.get("metric"), "expect": ind.get("expect"),
        "desc": ind.get("desc", ""), "scale": ind.get("scale"),
        "source": audit.get("source"), "truth": truth, "truth_unit": truth_unit,
        "truth_kind": truth_kind, "simulation": sim, "simulation_unit": sim_unit,
        "ci": ci, "n": result.get("n") if result else None,
        "raw_mean": result.get("mean") if result else None,
        "raw_base": result.get("base") if result else None,
        "internal_hit": result.get("hit") if result and not mismatch else None,
        "raw_got": result.get("got") if result else None,
        "gap": gap, "gap_unit": truth_unit if direct else None,
        "external_direction_match": external_direction,
        "status": status, "reason": reason,
        "score_label": score.get("label") if score else None,
        "off": score.get("off") if score else None,
        "on": score.get("on") if score else None,
    }


def _annotate_comparability(rows: list[dict], notes_path: Path = COMPARABILITY_NOTES,
                           strict: bool = True) -> None:
    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    expected = {r["id"] for r in rows}
    if strict and set(notes) != expected:
        raise ValueError(f"indicator opinion coverage mismatch: missing={sorted(expected-set(notes))}, extra={sorted(set(notes)-expected)}")
    for row in rows:
        note = notes.get(row["id"], {"more_people": "개별 판단 필요",
                                     "opinion": "이 지표는 등록표의 실측 정의와 실험 설계를 확인해야 합니다."})
        if not isinstance(note, dict) or not note.get("more_people") or not note.get("opinion"):
            raise ValueError(f"incomplete indicator opinion: {row['id']}")
        row["more_people"] = note["more_people"]
        row["expert_opinion"] = note["opinion"]


def build(score_paths: list[Path], scoring_path: Path = SCORING, experiment: str = "") -> dict:
    if not score_paths:
        raise ValueError("at least one --score is required")
    if len(score_paths) > 1 and not experiment:
        raise ValueError("multiple scores require --experiment (one named experiment)")
    table_hash = _sha(scoring_path)
    table = json.loads(scoring_path.read_text(encoding="utf-8"))
    policies = {k: v for k, v in table.items() if isinstance(v, dict) and isinstance(v.get("indicators"), list)}
    scores, sources = {}, []
    for path in score_paths:
        score = json.loads(path.read_text(encoding="utf-8"))
        policy = score.get("policy")
        if policy not in policies:
            raise ValueError(f"unknown policy in {path}: {policy}")
        if policy in scores:
            raise ValueError(f"duplicate policy {policy}; separate experiments/candidates need separate reports")
        results = score.get("results")
        if not isinstance(results, list):
            raise ValueError(f"missing results list in {path}")
        ids = [r.get("id") for r in results if isinstance(r, dict)]
        if len(ids) != len(results) or len(ids) != len(set(ids)):
            raise ValueError(f"duplicate/malformed indicator IDs in {path}")
        registered = {i["id"] for i in policies[policy]["indicators"]}
        if set(ids) - registered:
            raise ValueError(f"unregistered indicator IDs in {path}: {set(ids) - registered}")
        scores[policy] = (score, {r["id"]: r for r in results})
        sources.append({"path": _display_path(path), "sha256": _sha(path), "policy": policy,
                        "scoring_table_matches": score.get("scoring_table_sha256") == table_hash,
                        "scorer_sha256": score.get("scorer_sha256"),
                        "eligibility_policy_file": score.get("eligibility_policy_file"),
                        "eligibility_policy_sha256": score.get("eligibility_policy_sha256")})
    rows = []
    for policy, spec in policies.items():
        score, results = scores.get(policy, (None, {}))
        rows.extend(_row(policy, ind, score, results.get(ind["id"]), table_hash)
                    for ind in spec["indicators"])
    _annotate_comparability(rows, strict=scoring_path.resolve() == SCORING.resolve())
    from collections import Counter
    tally = dict(Counter(r["status"] for r in rows))
    unregistered = []
    for path in sorted((ROOT / "data/neo4j_load/policies").glob("P???.json")):
        if POLICY_ID_TO_SCORE.get(path.stem) not in policies:
            policy_file = json.loads(path.read_text(encoding="utf-8"))
            unregistered.append({"id": path.stem, "name": policy_file.get("name", path.stem),
                                 "path": _display_path(path), "status": "검증지표 미등록"})
    return {"experiment": experiment or scores[next(iter(scores))][0].get("label") or score_paths[0].stem,
            "scoring_table": {"path": _display_path(scoring_path), "sha256": table_hash},
            "comparability_notes": {"path": _display_path(COMPARABILITY_NOTES),
                                     "sha256": _sha(COMPARABILITY_NOTES)},
            "score_files": sources, "indicator_count": len(rows), "policy_count": len(policies),
            "simulated_count": sum(_number(r["simulation"]) for r in rows),
            "direct_gap_count": sum(r["gap"] is not None for r in rows),
            "tally": tally, "rows": rows, "unregistered_policies": unregistered}


def build_paired_effect(effect_path: Path, scoring_path: Path = SCORING,
                        experiment: str = "", sector_path: Path | None = None) -> dict:
    """Display paired-pilot proxies without granting empirical estimand equivalence."""
    effect = json.loads(effect_path.read_text(encoding="utf-8"))
    provenance = effect.get("provenance") or {}
    arms = provenance.get("arms") or {}
    if (effect.get("policy_id") != "P013" or effect.get("complete_matrix") is not True
            or effect.get("funding_reconciled") is not True
            or provenance.get("prompt_variant") != "v53"
            or set(arms) != {"on", "off"}):
        raise ValueError("paired P013/v53 evidence and accounting gates missing")
    table_hash = _sha(scoring_path)
    table = json.loads(scoring_path.read_text(encoding="utf-8"))
    policies = {k: v for k, v in table.items()
                if isinstance(v, dict) and isinstance(v.get("indicators"), list)}
    rows = [_row(policy, ind, None, None, table_hash)
            for policy, spec in policies.items() for ind in spec["indicators"]]
    by_id = {row["id"]: row for row in rows}
    proxies = {
        "EM-2": ("eligible_offline_relative_change",
                 "eligible_offline_relative_citizen_bootstrap_95_interval",
                 "같은 시민·날짜의 적격 오프라인 지출 ON−OFF 변화율입니다. 실측은 카드매출의 전년동기 증가율 변화(%p)이므로 수치를 빼거나 크기 적중으로 판정할 수 없습니다."),
        "EM-3": ("recorded_total_relative_change",
                 "recorded_total_relative_citizen_bootstrap_95_interval",
                 "같은 시민·날짜의 총지출 ON−OFF 변화율입니다. 실측은 전국 카드매출의 전년동기 변화이므로 단위가 %로 보여도 추정량이 다릅니다."),
    }
    for indicator_id, (value_key, interval_key, reason) in proxies.items():
        value = effect.get(value_key)
        if not _number(value):
            raise ValueError(f"paired proxy missing: {value_key}")
        ci = effect.get(interval_key)
        if ci is not None and (not isinstance(ci, list) or len(ci) != 2
                               or not all(_number(x) for x in ci)):
            raise ValueError(f"invalid paired interval: {interval_key}")
        row = by_id[indicator_id]
        row.update(simulation=100 * value, simulation_unit="%",
                   ci=[100*x for x in ci] if ci else None,
                   n=effect.get("citizens"), status="간접 대리지표", reason=reason,
                   score_label=experiment or "P013 v53 소규모 쌍체 파일럿",
                   off=f"{effect['effect_start']}:{effect['effect_end']}",
                   on=f"{effect['effect_start']}:{effect['effect_end']}",
                   proxy_direction_same=(value > 0 if _number(row["truth"]) and row["truth"] > 0
                                         else None))
    by_id["EM-4"].update(status="업종별 측정 없음",
                         reason="P013 양팔은 실행했지만 업종별 쌍체 원장이 없거나 한 업종의 OFF 지출이 0원이면 준내구재·대면서비스 변화율 순위를 계산할 수 없습니다.",
                         score_label=experiment or "P013 v53 소규모 쌍체 파일럿",
                         off=f"{effect['effect_start']}:{effect['effect_end']}",
                         on=f"{effect['effect_start']}:{effect['effect_end']}")
    sector = None
    if sector_path:
        sector = json.loads(sector_path.read_text(encoding="utf-8"))
        from datetime import date, timedelta
        start, end = date.fromisoformat(effect["effect_start"]), date.fromisoformat(effect["effect_end"])
        expected_days = [(start + timedelta(days=i)).isoformat()
                         for i in range((end-start).days+1)]
        if (sector.get("schema") != "p013_sector_pair_v1"
                or sector.get("citizens") != effect["citizens"]
                or sector.get("days") != expected_days
                or sector.get("scoring_table_sha256") != table_hash
                or not _number(sector.get("rank_gap_percentage_points"))
                or not _number(sector.get("semidurable_relative_change_pct"))
                or not _number(sector.get("face_service_relative_change_pct"))
                or "not_external_kdi_estimand" not in str(sector.get("comparison"))):
            raise ValueError("paired sector proxy does not match the completed P013 pilot")
        ci = sector.get("citizen_bootstrap_95_interval")
        if ci is not None and (not isinstance(ci, list) or len(ci) != 2
                               or not all(_number(x) for x in ci)):
            raise ValueError("invalid paired sector interval")
        by_id["EM-4"].update(simulation=sector["rank_gap_percentage_points"],
                             simulation_unit="%p", ci=ci, n=effect["citizens"],
                             status="업종 간접 대리지표",
                             reason="준내구재 POI와 대면서비스 POI의 동일 날짜 ON−OFF 매출 변화율 차이입니다. 실측은 다른 업종 범위·전년동기 카드매출 분석이므로 순위만 참고하고 수치 오차는 계산하지 않습니다.",
                             proxy_direction_same=sector["rank_gap_percentage_points"] > 0)
    _annotate_comparability(rows, strict=scoring_path.resolve() == SCORING.resolve())
    unregistered = []
    for path in sorted((ROOT / "data/neo4j_load/policies").glob("P???.json")):
        if POLICY_ID_TO_SCORE.get(path.stem) not in policies:
            policy_file = json.loads(path.read_text(encoding="utf-8"))
            unregistered.append({"id": path.stem, "name": policy_file.get("name", path.stem),
                                 "path": _display_path(path), "status": "검증지표 미등록"})
    from collections import Counter
    sources = [{"path": _display_path(effect_path), "sha256": _sha(effect_path),
                "policy": "EMERGENCY_2020", "scoring_table_matches": "not_applicable"}]
    if sector_path:
        sources.append({"path": _display_path(sector_path), "sha256": _sha(sector_path),
                        "policy": "EMERGENCY_2020/sector", "scoring_table_matches": "not_applicable"})
    return {"experiment": experiment or effect_path.stem,
            "report_kind": "paired_pilot_proxy",
            "scoring_table": {"path": _display_path(scoring_path), "sha256": table_hash},
            "comparability_notes": {"path": _display_path(COMPARABILITY_NOTES),
                                     "sha256": _sha(COMPARABILITY_NOTES)},
            "score_files": sources,
            "indicator_count": len(rows), "policy_count": len(policies),
            "simulated_count": 3 if sector else 2, "direct_gap_count": 0,
            "tally": dict(Counter(r["status"] for r in rows)),
            "rows": rows, "unregistered_policies": unregistered,
            "paired_sector_summary": sector,
            "paired_effect_summary": {
                "citizens": effect["citizens"], "days": effect["days"],
                "effect_start": effect["effect_start"], "effect_end": effect["effect_end"],
                "grant_recipients": effect["grant_recipients"],
                "grant_issued_won": effect["grant_issued_won"],
                "grant_spent_won": effect["grant_spent_won"],
                "recorded_total_spend_difference_won": effect["recorded_total_spend_difference_won"],
                "incremental_recorded_spend_per_grant_won": effect["incremental_recorded_spend_per_grant_won"],
                "choice_repair_sensitivity": effect.get("choice_repair_sensitivity"),
                "provenance": provenance}}


def _missing_simulation_explanation(row: dict) -> str:
    """Explain a missing simulation numerator in plain Korean where evidence permits."""
    reason = row.get("reason") or ""
    if row["id"] == "DS-6":
        geo = row.get("geographic_failure_display_audit")
        if geo:
            on, off = geo["on"], geo["off"]
            return (
                "실측은 2020년 카드패널에서 관광특구 매출 변화 −8.7%, 발달상권 −4.4%의 "
                "차이 −4.3%p입니다. 시뮬에는 2019년 비교 매출과 같은 가맹점 표본이 없어 "
                "동일한 전년 대비 수치를 만들 수 없습니다. 별도 2023년 상권 경계를 "
                "2026년 POI에 씌운 탐색 대리값도 사전 기술 관문에 실패했습니다: "
                f"경계 중첩 제외 비율 ON {100 * on['overlap_rate']:.2f}% "
                f"({on['overlap_count']}/{on['positive_receipts']}건), "
                f"OFF {100 * off['overlap_rate']:.2f}% "
                f"({off['overlap_count']}/{off['positive_receipts']}건)으로 "
                "각 팔 1% 상한을 넘었습니다. 관광특구 양수 구매도 "
                f"ON {on['tourism_receipts']}건/{on['tourism_citizens']}명·"
                f"{on['tourism_won']:,}원, "
                f"OFF {off['tourism_receipts']}건/{off['tourism_citizens']}명·"
                f"{off['tourism_won']:,}원뿐입니다. 기술적 계산값은 검증 가능한 "
                "시뮬 수치로 채택하지 않아 실측과 방향·크기를 비교하지 않습니다."
            )
        return ("시뮬 원장에는 2020년 관광특구·발달상권 공식 구분과 2019년 비교 매출이 "
                "없어, 실측과 같은 상권별 전년 대비 변화율의 분자·분모를 만들 수 없습니다. "
                "2023년 경계 탐색값도 사전 기술 관문을 통과했을 때만 별도 표시합니다.")
    if row["id"] in ("C2", "C3") and "outside the mart parent" in reason:
        return ("대상 업종 POI 일부가 원장의 '마트' 상위 분류 밖에 있어, 같은 마트 안의 "
                "대상 상품 매출을 분자로 놓을 수 없습니다. 잘못된 분모로 비율을 만들지 않았습니다.")
    if row["id"] in ("P012-4", "P012-6") and "State is absent" in reason:
        return ("10월 한 달의 캐시백 누적액과 상한 도달 State가 이 거래 원장에 없어 "
                "수령자별 지급액 또는 상한 도달자의 분자를 만들 수 없습니다.")
    if row["status"] == "정책 미실행":
        return "이번 보고서 묶음에 이 정책의 완료된 v53 원장이 아직 없습니다."
    return row.get("expert_opinion") or reason or "시뮬 수치의 산출 근거가 없습니다."


def numeric_pair_view(report: dict, *, in_progress_policies: set[str] | None = None) -> dict:
    """Show measured-versus-simulated pairs without hiding policy coverage.

    This is a presentation filter.  A pair is *not* a validated effect-size
    comparison unless the original row's run-specific audit emitted a gap.
    In particular, a historical score and a P013 proxy cannot be promoted by
    merely putting their numbers next to an empirical number.
    """
    original = report["rows"]
    in_progress_policies = in_progress_policies or set()
    coverage = []
    exploratory_by_policy = {}
    for entry in report.get("exploratory_pairs", []):
        exploratory_by_policy[entry["policy"]] = exploratory_by_policy.get(entry["policy"], 0) + 1
    for policy in dict.fromkeys(row["policy"] for row in original):
        rows = [row for row in original if row["policy"] == policy]
        measured = [row for row in rows if _number(row["truth"])]
        simulated = [row for row in rows if _number(row["simulation"])]
        paired = [row for row in measured if _number(row["simulation"])]
        missing_sim = [row for row in measured if not _number(row["simulation"])]
        sample_sizes = {row.get("sample_citizens") for row in rows
                        if isinstance(row.get("sample_citizens"), int)}
        coverage.append({
            "policy": policy, "policy_name": rows[0]["policy_name"],
            "registered_count": len(rows), "empirical_numeric_count": len(measured),
            "simulation_numeric_count": len(simulated), "paired_numeric_count": len(paired),
            "exploratory_numeric_count": exploratory_by_policy.get(policy, 0),
            "sample_citizens": max(sample_sizes) if sample_sizes else None,
            "in_progress": policy in in_progress_policies,
            "unmeasured_count": len(rows) - len(measured),
            "missing_simulation_ids": [row["id"] for row in missing_sim],
            "missing_simulation_reasons": [
                {"id": row["id"], "status": row["status"],
                 "truth": row["truth"], "truth_unit": row["truth_unit"],
                 "reason": _missing_simulation_explanation(row),
                 "technical_reason": row["reason"]}
                for row in missing_sim
            ],
        })
    visible = [row for row in original
               if _number(row["truth"]) and _number(row["simulation"])]
    for row in visible:
        row["surface_comparison"] = _surface_comparison(row)
    view = dict(report)
    from collections import Counter
    view.update(report_kind="numeric_pairs", rows=visible,
                in_progress_policies=sorted(in_progress_policies),
                indicator_count=len(visible), simulated_count=len(visible),
                direct_gap_count=sum(row["gap"] is not None for row in visible),
                catalog_tally=report.get("tally"),
                tally=dict(Counter(row["status"] for row in visible)),
                policy_coverage=coverage,
                omitted_without_empirical=sum(not _number(row["truth"]) for row in original),
                omitted_without_simulation=sum(_number(row["truth"])
                                               and not _number(row["simulation"])
                                               for row in original))
    return view


def _surface_comparison(row: dict) -> dict | None:
    """Arithmetic impression only; never a policy-effect accuracy estimate."""
    # P012-5 is a difference of two log-regression coefficients. Converting
    # that difference to a single percentage does not produce a percentage-
    # point difference between the two sector growth rates.
    if (row.get("gap") is not None or row.get("id") == "P012-5"
            or row.get("binomial_zero_count_audit")
            or row.get("structural_zero_display_audit")
            or ((row.get("run_context") or {}).get("preperiod_balance") or {}).get(
                "status") == "fail"):
        return None
    truth, sim = row["truth"], row["simulation"]
    truth_unit, sim_unit = row["truth_unit"], row["simulation_unit"]
    conversion = None
    if truth_unit == sim_unit and truth_unit in ("%", "%p", "ratio", "원"):
        converted = truth
    elif truth_unit == "log-point" and sim_unit == "%":
        try:
            converted = 100.0 * math.expm1(truth)
        except OverflowError:
            return None
        conversion = "실측 로그계수는 100×(exp(계수)−1)로 퍼센트 환산"
    else:
        return None
    if not _number(converted):
        return None
    delta = sim - converted
    if converted * sim < 0:
        scale = "부호 반대"
    elif converted == 0 or sim == 0:
        scale = "한쪽 값이 0에 가까움"
    else:
        multiple = abs(sim / converted)
        if 0.5 <= multiple <= 2:
            scale = "같은 자리수 규모"
        elif multiple > 2:
            scale = f"시뮬 숫자 약 {multiple:.1f}배"
        else:
            scale = f"시뮬 숫자 약 {1/multiple:.1f}분의 1"
    ci = row.get("ci")
    return {"empirical_as": converted, "unit": sim_unit,
            "delta": delta, "delta_unit": "%p" if sim_unit == "%" else sim_unit,
            "scale": scale, "conversion": conversion,
            "simulation_interval_crosses_zero": bool(ci and ci[0] <= 0 <= ci[1])}


def _validate_geo_proxy_exploratory(entry: dict) -> None:
    """Keep a 2023 geographic proxy outside the registered 2020 DS-6 score."""
    if entry.get("simulation") is None:
        return
    audit = entry.get("geo_proxy_audit")
    components = entry.get("simulation_components")
    if (entry.get("simulation_unit") not in ("percentage points", "%p")
            or entry.get("estimand_alignment") != "different"
            or entry.get("direction_comparable") is not False
            or not isinstance(audit, dict) or not isinstance(components, dict)):
        raise ValueError("DS6 geo proxy requires explicit exploratory audit and units")
    if not all(_number(components.get(key)) for key in (
            "tourism_special_zone_pct", "developed_commercial_district_pct")):
        raise ValueError("DS6 geo proxy lacks both sector components")
    match_rate = audit.get("match_rate")
    overlaps = audit.get("overlap_count")
    overlap_by_arm = audit.get("overlap_count_by_arm")
    receipts_by_arm = audit.get("total_receipt_count_by_arm")
    overlap_rates = audit.get("overlap_rate_by_arm")
    maximum_overlap = audit.get("maximum_overlap_rate_allowed")
    on_days, off_days = audit.get("on_citizen_days"), audit.get("off_citizen_days")
    denominators = audit.get("off_denominator_won_by_type")
    overlap_audited = (
        isinstance(overlap_by_arm, dict) and set(overlap_by_arm) == {"on", "off"}
        and isinstance(receipts_by_arm, dict) and set(receipts_by_arm) == {"on", "off"}
        and isinstance(overlap_rates, dict) and set(overlap_rates) == {"on", "off"}
        and maximum_overlap == 0.01
        and audit.get("overlap_rule") ==
        "Exclude ambiguous receipts from both hub types; no category priority"
        and all(isinstance(overlap_by_arm.get(arm), int)
                and not isinstance(overlap_by_arm[arm], bool)
                and isinstance(receipts_by_arm.get(arm), int)
                and not isinstance(receipts_by_arm[arm], bool)
                and 0 <= overlap_by_arm[arm] <= receipts_by_arm[arm]
                and receipts_by_arm[arm] > 0
                and _number(overlap_rates.get(arm))
                and abs(overlap_rates[arm] - overlap_by_arm[arm] / receipts_by_arm[arm]) < 1e-9
                and 0 <= overlap_rates[arm] <= maximum_overlap
                for arm in ("on", "off"))
        and isinstance(overlaps, int) and not isinstance(overlaps, bool)
        and overlaps == sum(overlap_by_arm.values())
        and audit.get("total_receipt_count") == sum(receipts_by_arm.values())
    )
    if (not _number(match_rate) or not 0.99 <= match_rate <= 1
            or not overlap_audited
            or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                       for value in (on_days, off_days))
            or on_days != off_days
            or not isinstance(denominators, dict)
            or not all(_number(denominators.get(key)) and denominators[key] > 0
                       for key in ("tourism_special_zone", "developed_commercial_district"))
            or audit.get("source_year") != 2023
            or not re.fullmatch(r"[0-9a-fA-F]{64}", str(audit.get("source_boundary_sha256")))):
        raise ValueError("DS6 geo proxy failed coordinate, overlap, balance or OFF denominator gate")


def apply_empirical_registry(report: dict, path: Path = EMPIRICAL_REGISTRY) -> dict:
    """Use the separately audited numeric source registry, never a prompt input."""
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema") != "empirical_registry_v1":
        raise ValueError("invalid empirical registry schema")
    entries = registry.get("indicators")
    if not isinstance(entries, list):
        raise ValueError("empirical registry indicators must be a list")
    mapped = {}
    for entry in entries:
        policy = POLICY_ID_TO_SCORE.get(entry.get("policy"), entry.get("policy"))
        key = (policy, entry.get("id"))
        if key in mapped:
            raise ValueError(f"duplicate empirical indicator: {key}")
        empirical = entry.get("empirical") or {}
        value = empirical.get("value")
        if value is None:
            value = empirical.get("gap")
        if value is None:
            value = empirical.get("difference_tourism_minus_developed")
        if not _number(value):
            raise ValueError(f"empirical indicator lacks numeric value: {key}")
        mapped[key] = (value, empirical, entry)
    rows = report["rows"]
    known = {(row["policy"], row["id"]) for row in rows}
    if set(mapped) - known:
        raise ValueError(f"empirical registry has unregistered indicators: {sorted(set(mapped)-known)}")
    units = {"share": "ratio", "KRW per recipient": "원",
             "percentage points": "%p", "% and percentage points": "%p",
             "% of cashback recipients": "%"}
    for row in rows:
        data = mapped.get((row["policy"], row["id"]))
        if data is None:
            row.update(truth=None, truth_unit=None, truth_kind="", source=None,
                       gap=None, gap_unit=None, external_direction_match=None)
            continue
        value, empirical, entry = data
        variant = row.get("empirical_variant")
        if variant == "october_only" and row["id"] in ("P012-4", "P012-6"):
            field = ("october_only_from_rounded_table" if row["id"] == "P012-4"
                     else "october_only_from_rounded_table_percent")
            value = empirical.get(field)
            if not _number(value):
                raise ValueError(f"missing October-only empirical value: {row['id']}")
        unit = units.get(empirical.get("unit"), empirical.get("unit"))
        row.update(truth=value, truth_unit=unit, truth_kind="원문 수치 감사",
                   source=empirical.get("source"), source_locator=empirical.get("locator"),
                   empirical_estimand=empirical.get("estimand"),
                   empirical_period=empirical.get("period"),
                   empirical_population=empirical.get("population"),
                   empirical_components=empirical.get("components"),
                   empirical_comparison_reason=entry.get("reason"),
                   registry_direct_gap_allowed=entry.get("direct_gap_allowed") is True)
        if variant == "october_only" and row["id"] in ("P012-4", "P012-6"):
            row["truth_kind"] = "원문 반올림 표에서 10월만 재계산"
            row["empirical_period"] = "2021년 10월 한 달"
            row["empirical_comparison_reason"] = (
                "실측은 원문 표의 반올림 수치에서 재계산한 10월 단독 참고값입니다. "
                + (entry.get("reason") or "")
            )
        if entry.get("direct_gap_allowed") is not True:
            row.update(gap=None, gap_unit=None, external_direction_match=None)
    updated = dict(report)
    updated["rows"] = rows
    updated["empirical_registry"] = {"path": _display_path(path), "sha256": _sha(path)}
    updated["direct_gap_count"] = sum(row["gap"] is not None for row in rows)
    exploratory_truth = {}
    for entry in registry.get("exploratory_additional_benchmarks_not_in_registered_38", []):
        policy = POLICY_ID_TO_SCORE.get(entry.get("policy"), entry.get("policy"))
        key = (policy, entry.get("id"))
        if key in exploratory_truth:
            raise ValueError(f"duplicate exploratory benchmark: {key}")
        exploratory_truth[key] = entry
    exploratory_pairs = []
    for item in report.get("exploratory_simulations", []):
        benchmark = exploratory_truth.get((item["policy"], item["id"]))
        if not benchmark or not _number(item.get("simulation")):
            continue
        empirical = benchmark.get("empirical") or {}
        value = empirical.get("value", benchmark.get("empirical_coefficient"))
        if not _number(value):
            continue
        candidate = dict(item)
        full_unit = empirical.get("unit", benchmark.get("unit"))
        display_unit = ("%" if isinstance(full_unit, str) and full_unit.startswith("% of ")
                        else "log-point" if isinstance(full_unit, str)
                        and full_unit.startswith("log-sales coefficient") else full_unit)
        candidate.update(truth=value,
                         truth_unit=display_unit, truth_unit_full=full_unit,
                         source=empirical.get("source", benchmark.get("source")),
                         source_locator=empirical.get("locator", benchmark.get("locator")),
                         empirical_estimand=empirical.get("estimand", benchmark.get("estimand")),
                         empirical_components=empirical.get("components"),
                         empirical_comparison_reason=benchmark.get("reason"),
                         gap=None, gap_unit=None)
        exploratory_pairs.append(candidate)
    updated["exploratory_pairs"] = exploratory_pairs
    return updated


def apply_p010_funding_display_audit(report: dict, path: Path) -> dict:
    """Attach a post-run explanation without changing frozen numeric scores."""
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (audit.get("schema") != "p010_funded_subclass_display_audit_v1"
            or audit.get("status") != "post_run_display_erratum_not_rescoring"
            or audit.get("policy") != "P010"):
        raise ValueError("invalid P010 funded-subclass display audit")
    run = next((item for item in report.get("run_evidence", [])
                if item["policy"] == "P010"), None)
    if not run:
        raise ValueError("P010 display audit requires a P010 numeric run")
    if audit.get("numeric_score_sha256") not in {
            item["sha256"] for item in report.get("score_files", [])}:
        raise ValueError("P010 display audit does not match a frozen numeric score")
    if audit.get("on_sector_ledger_sha256") not in {
            item["sha256"] for item in run["evidence"]
            if item["path"].endswith("sector.ledger.jsonl")}:
        raise ValueError("P010 display audit does not match a verified sector ledger")
    start_end = str(run.get("on") or "").split(":")
    if audit.get("effect_days") != start_end:
        raise ValueError("P010 display audit effect days differ from numeric run")
    density = run["run_context"].get("policy_funding_density") or {}
    for audit_key, source_key in (("observed_citizen_days", "policy_funded_observed_citizen_days"),
                                  ("positive_policy_funded_citizen_days", "policy_funded_positive_citizen_days"),
                                  ("policy_funded_total_won", "policy_funded_total_won")):
        if audit.get(audit_key) != density.get(source_key):
            raise ValueError(f"P010 display audit {audit_key} differs from numeric run")
    amounts = audit.get("funded_by_sub_won")
    if (not isinstance(amounts, dict) or not amounts
            or any(not isinstance(key, str) or not key
                   or not isinstance(value, int) or isinstance(value, bool) or value < 0
                   for key, value in amounts.items())
            or sum(amounts.values()) != audit.get("policy_funded_total_won")
            or audit.get("funded_by_sub_total_won") != audit.get("policy_funded_total_won")):
        raise ValueError("P010 funded-subclass amounts do not reconcile")
    evidence = {"path": _display_path(path), "sha256": _sha(path)}
    augmented = []
    for row in report.get("exploratory_simulations", []):
        candidate = dict(row)
        if row["policy"] == "P010":
            candidate["funded_subclass_display_audit"] = {
                "funded_by_sub_won": amounts, "positive_citizen_days": audit["positive_policy_funded_citizen_days"],
                "observed_citizen_days": audit["observed_citizen_days"],
                "funded_total_won": audit["policy_funded_total_won"], **evidence}
        augmented.append(candidate)
    updated = dict(report)
    updated["exploratory_simulations"] = augmented
    updated["post_run_display_audits"] = [evidence]
    return updated


def apply_p010_channel_display_audit(report: dict, path: Path) -> dict:
    """Show a sourced channel decomposition without changing policy scores."""
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (audit.get("status") != "posthoc_simulator_channel_diagnostic_not_empirical_policy_effect"
            or not isinstance(audit.get("source_sha256"), dict)):
        raise ValueError("invalid P010 channel display audit")
    run = next((item for item in report.get("run_evidence", [])
                if item["policy"] == "P010"), None)
    if not run:
        raise ValueError("P010 channel audit requires a P010 numeric run")
    try:
        start_text, end_text = run["on"].split(":")
        start, end = date.fromisoformat(start_text), date.fromisoformat(end_text)
        days = [(start + timedelta(days=offset)).isoformat()
                for offset in range((end - start).days + 1)]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("P010 channel audit requires valid effect dates") from exc
    citizens = run.get("citizens")
    if (not days or not isinstance(citizens, int) or isinstance(citizens, bool)
            or citizens <= 0 or audit.get("policy_effect_days") != days
            or audit.get("paired_citizen_days") != citizens * len(days)):
        raise ValueError("P010 channel audit dates or citizen-days differ from numeric run")
    verified = {item["path"].replace("\\", "/"): item["sha256"]
                for item in run["evidence"]}
    sources = audit["source_sha256"]
    expected = {source: verified[source] for source in verified
                if any(source.endswith(f"/{arm}/{arm}/metrics/day_{day}.jsonl")
                       for arm in ("on", "off") for day in days)}
    if (len(expected) != 2 * len(days) or sources != expected):
        raise ValueError("P010 channel audit metrics do not match verified paired evidence")
    on, off, gap = (audit.get(key) for key in ("on_sum", "off_sum", "on_minus_off"))
    keys = ("cm_today_total_incl_online", "cm_online_total",
            "offline_positive_receipt_won", "cm_policy_allocated_total")
    if not all(isinstance(part, dict) and all(_number(part.get(key)) for key in keys)
               for part in (on, off, gap)):
        raise ValueError("P010 channel audit lacks numeric channel sums")
    for key in keys:
        if abs((on[key] - off[key]) - gap[key]) > 1e-6:
            raise ValueError("P010 channel audit ON-OFF sums do not reconcile")
    for part in (on, off, gap):
        if abs(part["cm_today_total_incl_online"]
               - part["cm_online_total"] - part["offline_positive_receipt_won"]) > 1e-6:
            raise ValueError("P010 channel audit online/offline sums do not reconcile")
    total = gap["cm_today_total_incl_online"]
    online = gap["cm_online_total"]
    online_fraction = audit.get("online_fraction_of_total_gap")
    offline_fraction = audit.get("offline_fraction_of_total_gap")
    if (total <= 0 or online < 0 or gap["offline_positive_receipt_won"] < 0
            or not _number(online_fraction) or not _number(offline_fraction)
            or abs(online_fraction - online / total) > 1e-9
            or abs(offline_fraction - gap["offline_positive_receipt_won"] / total) > 1e-9
            or gap["cm_policy_allocated_total"] !=
            (run["run_context"].get("policy_funding_density") or {}).get("policy_funded_total_won")):
        raise ValueError("P010 channel audit fractions or policy wallet do not reconcile")
    evidence = {"path": _display_path(path), "sha256": _sha(path)}
    displayed = {
        "status": audit["status"], "total_gap_won": total, "online_gap_won": online,
        "offline_gap_won": gap["offline_positive_receipt_won"],
        "online_fraction": online_fraction,
        "wallet_won": gap["cm_policy_allocated_total"],
        "paired_citizen_days": audit["paired_citizen_days"], **evidence,
    }
    updated = dict(report)
    updated["run_evidence"] = [
        {**item, "run_context": {**item["run_context"], "p010_channel_display_audit": displayed}}
        if item["policy"] == "P010" else item for item in report["run_evidence"]
    ]
    updated["rows"] = [
        {**row, "run_context": {**row["run_context"], "p010_channel_display_audit": displayed}}
        if row["policy"] == "P010" and row.get("run_context") else row
        for row in report["rows"]
    ]
    updated["post_run_display_audits"] = [
        *report.get("post_run_display_audits", []), evidence]
    return updated


def apply_p010_concentration_display_audit(report: dict, path: Path) -> dict:
    """Disclose how few simulated citizens dominate P010's short-window gap."""
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (audit.get("status") != "posthoc_citizen_gap_concentration_diagnostic_not_policy_effect"
            or not isinstance(audit.get("source_sha256"), dict)):
        raise ValueError("invalid P010 concentration display audit")
    run = next((item for item in report.get("run_evidence", [])
                if item["policy"] == "P010"), None)
    if not run:
        raise ValueError("P010 concentration audit requires a P010 numeric run")
    try:
        start_text, end_text = run["on"].split(":")
        start, end = date.fromisoformat(start_text), date.fromisoformat(end_text)
        days = [(start + timedelta(days=offset)).isoformat()
                for offset in range((end - start).days + 1)]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("P010 concentration audit requires valid effect dates") from exc
    verified = {item["path"].replace("\\", "/"): item["sha256"]
                for item in run["evidence"]}
    expected = {source: verified[source] for source in verified
                if any(source.endswith(f"/{arm}/{arm}/metrics/day_{day}.jsonl")
                       for arm in ("on", "off") for day in days)}
    if not days or len(expected) != 2 * len(days) or audit["source_sha256"] != expected:
        raise ValueError("P010 concentration audit metrics do not match verified paired evidence")
    n = audit.get("paired_citizens")
    counts = [audit.get(key) for key in (
        "positive_delta_citizens", "negative_delta_citizens", "zero_delta_citizens")]
    top = audit.get("top_five_positive_deltas_won")
    net, top_sum = audit.get("net_gap_won"), audit.get("top_five_sum_won")
    share = audit.get("top_five_share_of_net_gap")
    channel = run["run_context"].get("p010_channel_display_audit") or {}
    if (not isinstance(n, int) or isinstance(n, bool) or n != run.get("citizens") or n < 5
            or not all(isinstance(value, int) and not isinstance(value, bool)
                       and value >= 0 for value in counts)
            or sum(counts) != n
            or not isinstance(top, list) or len(top) != 5
            or not all(_number(value) and value > 0 for value in top)
            or top != sorted(top, reverse=True)
            or not _number(net) or net <= 0 or not _number(top_sum)
            or abs(top_sum - sum(top)) > 1e-6
            or not _number(share) or abs(share - top_sum / net) > 1e-9
            or (channel and abs(channel["total_gap_won"] - net) > 1e-6)):
        raise ValueError("P010 concentration audit counts or gap do not reconcile")
    evidence = {"path": _display_path(path), "sha256": _sha(path)}
    displayed = {"status": audit["status"], "paired_citizens": n,
                 "net_gap_won": net, "top_five_sum_won": top_sum,
                 "top_five_share": share, "remaining_gap_won": net - top_sum,
                 **evidence}
    updated = dict(report)
    updated["run_evidence"] = [
        {**item, "run_context": {**item["run_context"], "p010_concentration_display_audit": displayed}}
        if item["policy"] == "P010" else item for item in report["run_evidence"]
    ]
    updated["rows"] = [
        {**row, "run_context": {**row["run_context"], "p010_concentration_display_audit": displayed}}
        if row["policy"] == "P010" and row.get("run_context") else row
        for row in report["rows"]
    ]
    updated["post_run_display_audits"] = [
        *report.get("post_run_display_audits", []), evidence]
    return updated


def apply_p012_sector_display_audit(report: dict, path: Path) -> dict:
    """Explain P012's unstable sector proxy using source-bound ON/OFF amounts."""
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (audit.get("schema") != "p012_sector_denominator_display_audit_v1"
            or audit.get("policy") != "P012" or audit.get("indicator") != "P012-5"):
        raise ValueError("invalid P012 sector display audit")
    run = next((item for item in report.get("run_evidence", [])
                if item["policy"] == "P012"), None)
    row = next((item for item in report.get("rows", [])
                if item["policy"] == "P012" and item["id"] == "P012-5"), None)
    if not run or not row or not _number(row.get("simulation")):
        raise ValueError("P012 sector audit requires a scored P012-5 run")
    if not any(item["path"] == audit.get("numeric_path")
               and item["sha256"] == audit.get("numeric_sha256")
               for item in report.get("score_files", [])):
        raise ValueError("P012 sector audit does not match the frozen numeric score")
    evidence = {item["path"]: item["sha256"] for item in run["evidence"]}
    if (audit.get("on_sector_ledger_sha256") != evidence.get(audit.get("on_sector_ledger_path"))
            or audit.get("off_sector_ledger_sha256") != evidence.get(audit.get("off_sector_ledger_path"))
            or not str(audit.get("on_sector_ledger_path", "")).replace("\\", "/").endswith(
                "/on/on/sector.ledger.jsonl")
            or not str(audit.get("off_sector_ledger_path", "")).replace("\\", "/").endswith(
                "/off/off/sector.ledger.jsonl")):
        raise ValueError("P012 sector audit does not match verified ON/OFF ledgers")
    start_end = str(run.get("on") or "").split(":")
    if (audit.get("effect_window") != run.get("on") or run.get("on") != run.get("off")
            or audit.get("citizens") != run.get("citizens")
            or len(start_end) != 2):
        raise ValueError("P012 sector audit period or citizens differ from run")
    try:
        days = (date.fromisoformat(start_end[1]) - date.fromisoformat(start_end[0])).days + 1
    except ValueError as exc:
        raise ValueError("P012 sector audit needs valid dates") from exc
    if days <= 0 or audit.get("paired_citizen_days") != days * run["citizens"]:
        raise ValueError("P012 sector audit citizen-days do not reconcile")
    expected = (("appliance_furniture", "appliance_furniture_pct"),
                ("hair_beauty", "hair_beauty_pct"))
    components = row.get("simulation_components") or {}
    for source, score_key in expected:
        item = audit.get(source)
        if (not isinstance(item, dict)
                or not all(isinstance(item.get(key), int) and not isinstance(item[key], bool)
                           and item[key] >= 0 for key in ("on_won", "off_won"))
                or item["off_won"] <= 0 or not _number(item.get("on_off_percent"))
                or abs(item["on_off_percent"]
                       - 100 * (item["on_won"] - item["off_won"]) / item["off_won"]) > 1e-9
                or not _number(components.get(score_key))
                or abs(item["on_off_percent"] - components[score_key]) > 1e-9):
            raise ValueError("P012 sector audit amounts differ from scored components")
    if (not _number(audit.get("gap_percentage_points"))
            or abs(audit["gap_percentage_points"] - row["simulation"]) > 1e-9):
        raise ValueError("P012 sector audit gap differs from frozen numeric score")
    evidence_note = {"path": _display_path(path), "sha256": _sha(path)}
    displayed = {
        "appliance_furniture": audit["appliance_furniture"],
        "hair_beauty": audit["hair_beauty"],
        "paired_citizen_days": audit["paired_citizen_days"], **evidence_note,
    }
    updated = dict(report)
    updated["rows"] = [{**item, "sector_denominator_display_audit": displayed}
                       if item is row else item for item in report["rows"]]
    updated["post_run_display_audits"] = [
        *report.get("post_run_display_audits", []), evidence_note]
    return updated


def apply_distancing_geo_failure_display_audit(report: dict, path: Path) -> dict:
    """Explain a rejected 2023 geographic proxy without promoting its raw value."""
    run = next((item for item in report.get("run_evidence", [])
                if item["policy"] == "DISTANCING_2020"), None)
    row = next((item for item in report.get("rows", [])
                if item["policy"] == "DISTANCING_2020" and item["id"] == "DS-6"), None)
    proxy = next((item for item in report.get("exploratory_simulations", [])
                  if item["policy"] == "DISTANCING_2020"
                  and item["id"] == "DS6-2023-GEO-PROXY"), None)
    if (not run or not row or not proxy or row.get("simulation") is not None
            or proxy.get("simulation") is not None):
        raise ValueError("DIST geographic failure display needs a rejected paired geo proxy")
    evidence = {item["path"]: item["sha256"] for item in run["evidence"]}
    source = _display_path(path)
    if (evidence.get(source) != _sha(path)
            or Path(source).name != "distancing_geo_proxy_20260928.json"):
        raise ValueError("DIST geographic sidecar is not verified by the frozen score")
    audit = json.loads(path.read_text(encoding="utf-8"))
    geo = proxy.get("geo_proxy_audit") or {}
    receipt_counts = audit.get("positive_receipts") or {}
    overlaps = audit.get("ambiguous_receipts_excluded") or {}
    rates = audit.get("ambiguous_receipt_rate_excluded") or {}
    arms = {}
    for arm in ("on", "off"):
        cell = (audit.get(arm) or {}).get("관광특구") or {}
        count, ambiguous, rate = (receipt_counts.get(arm), overlaps.get(arm),
                                  rates.get(arm))
        if (not isinstance(count, int) or isinstance(count, bool) or count <= 0
                or not isinstance(ambiguous, int) or isinstance(ambiguous, bool)
                or not 0 <= ambiguous <= count or not _number(rate)
                or abs(rate - ambiguous / count) > 1e-12
                or not all(isinstance(cell.get(key), int)
                           and not isinstance(cell[key], bool) and cell[key] >= 0
                           for key in ("positive_receipts", "citizens_with_receipts", "spend_won"))):
            raise ValueError("DIST geographic sidecar has invalid receipt or tourism cells")
        if (geo.get("overlap_count_by_arm", {}).get(arm) != ambiguous
                or geo.get("total_receipt_count_by_arm", {}).get(arm) != count
                or not _number(geo.get("overlap_rate_by_arm", {}).get(arm))
                or abs(geo["overlap_rate_by_arm"][arm] - rate) > 1e-12):
            raise ValueError("DIST geographic sidecar differs from scored geo audit")
        arms[arm] = {"positive_receipts": count, "overlap_count": ambiguous,
                     "overlap_rate": rate,
                     "tourism_receipts": cell["positive_receipts"],
                     "tourism_citizens": cell["citizens_with_receipts"],
                     "tourism_won": cell["spend_won"]}
    source_hashes = audit.get("sources_sha256") or {}
    official_boundary = [digest for source_path, digest in source_hashes.items()
                         if Path(source_path).name.endswith("2023-10-23.zip")]
    if (geo.get("maximum_overlap_rate_allowed") != 0.01
            or not any(item["overlap_rate"] > 0.01 for item in arms.values())
            or geo.get("source_year") != 2023
            or official_boundary != [geo.get("source_boundary_sha256")]
            or (geo.get("off_denominator_won_by_type") or {}).get(
                "tourism_special_zone") != arms["off"]["tourism_won"]):
        raise ValueError("DIST geographic sidecar does not substantiate a 1% gate failure")
    display = {**arms, "path": source, "sha256": evidence[source],
               "status": "post_run_display_erratum_not_rescoring"}
    updated = dict(report)
    updated["rows"] = [{**item, "geographic_failure_display_audit": display}
                       if item is row else item for item in report["rows"]]
    updated["post_run_display_audits"] = [
        *report.get("post_run_display_audits", []), {"path": source, "sha256": evidence[source]}]
    return updated


def apply_p016_postfix_display_audit(report: dict, path: Path) -> dict:
    """Disclose an excluded failed arm and the patched, separately scored pair."""
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (audit.get("schema") != "p016_postfix_provenance_display_audit_v1"
            or audit.get("policy") != "P016"
            or audit.get("run_manifest_source_commit_is_not_live_code_evidence") is not True):
        raise ValueError("invalid P016 post-fix display audit")
    run = next((item for item in report.get("run_evidence", [])
                if item["policy"] == "P016"), None)
    if not run or not any(
        item["path"] == audit.get("numeric_path")
        and item["sha256"] == audit.get("numeric_sha256")
        for item in report.get("score_files", [])
    ):
        raise ValueError("P016 post-fix audit does not match frozen numeric score")
    evidence = {item["path"]: item["sha256"] for item in run["evidence"]}
    patch_sha = "bc3819adaee5c8c922309aa0dc6b62c22a89f8630023939edb874b2b8fec50c4"
    if audit.get("patched_instant_discount_sha256") != patch_sha:
        raise ValueError("P016 post-fix audit has a different account-engine patch")
    run_ids = []
    revisions = []
    for arm in ("on", "off"):
        item = audit.get(arm) or {}
        frozen_path = item.get("frozen_inputs_path")
        sector_path = next((source for source in evidence
                            if source.replace("\\", "/").endswith(
                                f"/{arm}/{arm}/sector.ledger.jsonl")), None)
        if not frozen_path or not sector_path:
            raise ValueError("P016 post-fix audit lacks scored arm evidence")
        frozen = Path(frozen_path)
        if not frozen.is_absolute():
            frozen = ROOT / frozen
        if (not frozen.is_file() or _sha(frozen) != item.get("frozen_inputs_sha256")
                or item.get("sector_ledger_sha256") != evidence[sector_path]
                or (item.get("frozen_hashes") or {}).get(
                    "scripts/sim/instant_discount.py") != patch_sha):
            raise ValueError("P016 post-fix arm differs from patched frozen inputs or score")
        manifest_source = sector_path + ".manifest.json"
        manifest = Path(manifest_source)
        if not manifest.is_absolute():
            manifest = ROOT / manifest
        if (evidence.get(manifest_source) != item.get("sector_manifest_sha256")
                or _sha(manifest) != item["sector_manifest_sha256"]
                or (json.loads(manifest.read_text(encoding="utf-8"))
                    .get("prompt_provenance") or {}).get("run_id") != item.get("run_id")):
            raise ValueError("P016 post-fix run ID differs from scored sector manifest")
        run_ids.append(item["run_id"])
        revisions.append(item.get("run_revision"))
    if (run_ids[0] == run_ids[1] or revisions != ["no_eligible_discount_fix1"] * 2
            or not all(run_id.endswith(f"-p016-no_eligible_discount_fix1-{arm}")
                       for run_id, arm in zip(run_ids, ("on", "off")))):
        raise ValueError("P016 post-fix paired run IDs or revisions differ")
    invalid = audit.get("invalidated_prepatch_arm") or {}
    preflight = audit.get("preflight") or {}
    failed_snapshot = Path(invalid.get("snapshot_path") or "")
    if not failed_snapshot.is_absolute():
        failed_snapshot = ROOT / failed_snapshot
    if (not failed_snapshot.is_file()
            or _sha(failed_snapshot) != invalid.get("snapshot_sha256")):
        raise ValueError("P016 excluded failed-arm snapshot SHA mismatch")
    snapshot = json.loads(failed_snapshot.read_text(encoding="utf-8"))
    if (invalid.get("excluded_from_score") is not True
            or snapshot.get("schema") != "failed_p016_prepatch_snapshot_v1"
            or snapshot.get("run_status") != "incomplete_failed"
            or snapshot.get("model_calls_occurred") is not True
            or snapshot.get("raw_keyerror_rows") != invalid.get("raw_keyerror_rows")
            or snapshot.get("date_of_failure") != invalid.get("failure_date")
            or not isinstance(invalid.get("raw_keyerror_rows"), int)
            or invalid["raw_keyerror_rows"] <= 0
            or preflight.get("first_failure_model_calls") != 0
            or preflight.get("passed_preflight_model_calls") != 0):
        raise ValueError("P016 excluded arm or preflight evidence is inconsistent")
    for source_key, sha_key in (("first_failure_manifest_path", "first_failure_manifest_sha256"),
                               ("passed_preflight_checksums_path", "passed_preflight_checksums_sha256")):
        source = Path(preflight.get(source_key) or "")
        if not source.is_absolute():
            source = ROOT / source
        if not source.is_file() or _sha(source) != preflight.get(sha_key):
            raise ValueError("P016 preflight evidence SHA mismatch")
    displayed = {"patch_sha256": patch_sha, "run_ids": run_ids,
                 "failed_rows": invalid["raw_keyerror_rows"],
                 "failure_date": invalid.get("failure_date"),
                 "failed_snapshot_sha256": invalid["snapshot_sha256"],
                 "first_preflight_sha256": preflight["first_failure_manifest_sha256"],
                 "passed_preflight_sha256": preflight["passed_preflight_checksums_sha256"],
                 "source_commit_not_live_code_evidence": True,
                 "path": _display_path(path), "sha256": _sha(path)}
    updated = dict(report)
    updated["run_evidence"] = [
        {**item, "run_context": {**item["run_context"], "p016_postfix_display_audit": displayed}}
        if item is run else item for item in report["run_evidence"]]
    updated["rows"] = [
        {**item, "run_context": {**item["run_context"], "p016_postfix_display_audit": displayed}}
        if item["policy"] == "P016" and item.get("run_context") else item
        for item in report["rows"]]
    updated["post_run_display_audits"] = [
        *report.get("post_run_display_audits", []),
        {"path": displayed["path"], "sha256": displayed["sha256"]}]
    return updated


def apply_p016_taxonomy_display_audit(report: dict, path: Path) -> dict:
    """Flag C2/C3 zeroes caused by equality of two POI taxonomy buckets."""
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (audit.get("schema") != "p016_taxonomy_identity_display_audit_v1"
            or audit.get("policy") != "P016"
            or audit.get("indicators") != ["C2", "C3"]
            or audit.get("target_poi_subclasses") != ["청과", "정육", "슈퍼마켓", "식료품"]
            or audit.get("mart_l1_category") != "마트"
            or audit.get("structural_identity_all_rows") is not True):
        raise ValueError("invalid P016 taxonomy identity audit")
    run = next((item for item in report.get("run_evidence", [])
                if item["policy"] == "P016"), None)
    if not run or not any(
        item["path"] == audit.get("numeric_path")
        and item["sha256"] == audit.get("numeric_sha256")
        for item in report.get("score_files", [])
    ):
        raise ValueError("P016 taxonomy audit does not match frozen numeric score")
    evidence = {item["path"]: item["sha256"] for item in run["evidence"]}
    for arm in ("on", "off"):
        item = audit.get(arm) or {}
        total, effect = item.get("citizen_days_all"), item.get("citizen_days_effect")
        if (evidence.get(item.get("sector_ledger_path")) != item.get("sector_ledger_sha256")
                or not all(isinstance(value, int) and not isinstance(value, bool)
                           for value in (total, effect, item.get("identity_count_all"),
                                         item.get("identity_count_effect"),
                                         item.get("effect_target_poi_won"),
                                         item.get("effect_mart_l1_won")))
                or total <= 0 or effect <= 0 or effect > total
                or item["identity_count_all"] != total
                or item["identity_count_effect"] != effect
                or item["effect_target_poi_won"] != item["effect_mart_l1_won"]):
            raise ValueError("P016 taxonomy identity is not bound to scored ledgers")
    marked = []
    for indicator in ("C2", "C3"):
        row = next((item for item in report["rows"]
                    if item["policy"] == "P016" and item["id"] == indicator), None)
        if (not row or row.get("simulation") != 0
                or row.get("ci") != [0, 0]
                or audit.get(f"score_{indicator.lower()}_percentage_points") != 0):
            raise ValueError("P016 structural zero differs from frozen score")
        marked.append(row)
    displayed = {"on": audit["on"], "off": audit["off"],
                 "path": _display_path(path), "sha256": _sha(path),
                 "meaning": "taxonomy_identity_not_policy_nonresponse"}
    updated = dict(report)
    updated["rows"] = [
        {**item,
         **({"structural_zero_display_audit": displayed}
            if any(item is chosen for chosen in marked) else {}),
         "run_context": {**item["run_context"], "p016_taxonomy_display_audit": displayed}}
        if item["policy"] == "P016" and item.get("run_context") else item
        for item in report["rows"]]
    updated["post_run_display_audits"] = [
        *report.get("post_run_display_audits", []),
        {"path": displayed["path"], "sha256": displayed["sha256"]}]
    return updated


def apply_distancing_input_display_audit(report: dict, path: Path) -> dict:
    """Disclose the dated ON/OFF regime, keeping its internal ID distinct from 2021."""
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (audit.get("purpose") != "read-only frozen input audit; no observed policy outcomes or target numbers"
            or not isinstance(audit.get("daily"), list)
            or not isinstance(audit.get("source_sha256"), dict)):
        raise ValueError("invalid distancing input display audit")
    run = next((item for item in report.get("run_evidence", [])
                if item["policy"] == "DISTANCING_2020"), None)
    if not run:
        raise ValueError("distancing input audit requires a scored distancing run")
    context = run["run_context"]
    if (audit.get("on_environment_id") != context.get("on_environment_id")
            or audit.get("off_environment_id") != context.get("off_environment_id")
            or not context.get("generic_prompt_sha256")
            or run.get("on") != run.get("off")):
        raise ValueError("distancing input audit differs from run environment or prompt")
    try:
        first, last = run["on"].split(":")
        start, end = date.fromisoformat(first), date.fromisoformat(last)
        days = [(start + timedelta(days=offset)).isoformat()
                for offset in range((end - start).days + 1)]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("distancing input audit needs valid simulation dates") from exc
    if not days or [entry.get("date") for entry in audit["daily"]] != days:
        raise ValueError("distancing input audit dates differ from numeric run")
    if not audit["source_sha256"]:
        raise ValueError("distancing input audit lacks source hashes")
    for source, digest in audit["source_sha256"].items():
        source_path = ROOT / source
        if (not source_path.is_file() or not re.fullmatch(r"[0-9a-fA-F]{64}", str(digest))
                or _sha(source_path).lower() != digest.lower()):
            raise ValueError(f"distancing input audit source SHA mismatch: {source}")
    case_counts = []
    for entry in audit["daily"]:
        on, off, shared = (entry.get(key) for key in ("on", "off", "shared_disease_facts"))
        if (not isinstance(on, dict) or not isinstance(off, dict)
                or not isinstance(shared, list) or len(shared) != 1
                or not isinstance(on.get("facts"), list)
                or not isinstance(off.get("facts"), list)
                or shared[0] not in on["facts"] or shared[0] not in off["facts"]
                or not any("21:00" in fact and "식당" in fact for fact in on["facts"])
                or not any("카페" in fact and "포장" in fact for fact in on["facts"])
                or not any("집합금지" in fact for fact in on["facts"])
                or not any("추가" in fact and "제한 없음" in fact for fact in off["facts"])):
            raise ValueError("distancing ON/OFF rendered facts do not support display")
        match = re.search(r"서울 신규 확진 (\d+)명", shared[0])
        if not match:
            raise ValueError("distancing shared case count missing")
        case_counts.append(int(match.group(1)))
    evidence = {"path": _display_path(path), "sha256": _sha(path)}
    displayed = {"dates": days, "seoul_case_counts": case_counts,
                 "on_environment_id": audit["on_environment_id"],
                 "off_environment_id": audit["off_environment_id"],
                 "scope": "frozen static render, not complete HTTP request capture", **evidence}
    updated = dict(report)
    updated["run_evidence"] = [
        {**item, "run_context": {**item["run_context"], "distancing_input_display_audit": displayed}}
        if item["policy"] == "DISTANCING_2020" else item for item in report["run_evidence"]
    ]
    updated["rows"] = [
        {**row, "run_context": {**row["run_context"], "distancing_input_display_audit": displayed}}
        if row["policy"] == "DISTANCING_2020" and row.get("run_context") else row
        for row in report["rows"]
    ]
    updated["post_run_display_audits"] = [
        *report.get("post_run_display_audits", []), evidence]
    return updated


def build_multi_policy_pairs(manifest_paths: Path | list[Path],
                             scoring_path: Path = SCORING,
                             suite: str = "") -> dict:
    """Render new, explicitly sourced v53 proxy values from several policy arms."""
    paths = [manifest_paths] if isinstance(manifest_paths, Path) else list(manifest_paths)
    if not paths:
        raise ValueError("at least one multi-policy manifest required")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    table_hash = _sha(scoring_path)
    source_experiments = [manifest.get("experiment") for manifest in manifests]
    experiment = suite or source_experiments[0]
    for manifest in manifests:
        if (manifest.get("schema") != "multi_policy_numeric_v1"
                or manifest.get("prompt_variant") != "v53"
                or manifest.get("scoring_table_sha256") != table_hash
                or not isinstance(manifest.get("experiment"), str)
                or not manifest["experiment"].strip()
                or (not suite and manifest["experiment"] != experiment)):
            raise ValueError("multi-policy manifest schema, v53 variant, experiment or scoring hash mismatch")
    table = json.loads(scoring_path.read_text(encoding="utf-8"))
    policies = {k: v for k, v in table.items()
                if isinstance(v, dict) and isinstance(v.get("indicators"), list)}
    rows = [_row(policy, indicator, None, None, table_hash)
            for policy, spec in policies.items() for indicator in spec["indicators"]]
    by_key = {(row["policy"], row["id"]): row for row in rows}
    runs = []
    for manifest in manifests:
        entries = manifest.get("runs")
        if not isinstance(entries, list) or not entries:
            raise ValueError("multi-policy manifest needs at least one run")
        runs.extend((entry, manifest["experiment"]) for entry in entries)
    seen_policies = set()
    run_evidence = []
    exploratory = []
    for run, source_experiment in runs:
        if not isinstance(run, dict):
            raise ValueError("run must be an object")
        policy = POLICY_ID_TO_SCORE.get(run.get("policy"), run.get("policy"))
        if policy not in policies or policy in seen_policies:
            raise ValueError(f"unknown or duplicate policy run: {policy}")
        seen_policies.add(policy)
        provenance = run.get("run_provenance") or {}
        if not isinstance(provenance, dict):
            raise ValueError(f"{policy}: run provenance must be an object")
        run_context = {
            "policy_id": run.get("policy_id") or provenance.get("policy_id") or next(
                (key for key, value in POLICY_ID_TO_SCORE.items() if value == policy), policy),
            "policy_input_file": run.get("policy_input_file") or provenance.get("policy_file_path"),
            "policy_input_sha256": run.get("policy_input_sha256") or provenance.get("policy_file_sha256"),
            "effective_from": run.get("effective_from") or provenance.get("effective_from"),
            "effective_until": run.get("effective_until") or provenance.get("effective_until"),
            "on_environment_id": provenance.get("on_environment_id"),
            "off_environment_id": provenance.get("off_environment_id"),
            "paired_environment_fingerprint": provenance.get("paired_environment_fingerprint"),
            "generic_prompt_variant": "v53",
            "generic_prompt_sha256": run.get("generic_prompt_sha256")
            or provenance.get("generic_prompt_sha256"),
            "stage2_sha256": provenance.get("stage2_sha256"),
            "quality_audit": provenance.get("quality_audit"),
            "preperiod_balance": provenance.get("preperiod_balance"),
            "policy_funding_diagnostic": provenance.get("policy_funding_diagnostic"),
            "policy_funding_density": provenance.get("policy_funding_density"),
        }
        prompt_hash = run_context["generic_prompt_sha256"]
        if prompt_hash and not re.fullmatch(r"[0-9a-fA-F]{64}", str(prompt_hash)):
            raise ValueError(f"{policy}: invalid generic prompt SHA256")
        sample_citizens = run.get("citizens")
        if (sample_citizens is not None and
                (not isinstance(sample_citizens, int) or isinstance(sample_citizens, bool)
                 or sample_citizens <= 0)):
            raise ValueError(f"{policy}: invalid run citizen count")
        evidence = run.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{policy}: missing raw evidence files")
        verified = []
        for item in evidence:
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                raise ValueError(f"{policy}: malformed evidence reference")
            source = Path(item["path"])
            if not source.is_absolute():
                source = ROOT / source
            if not source.is_file() or _sha(source).lower() != str(item["sha256"]).lower():
                raise ValueError(f"{policy}: missing or SHA256-mismatched evidence: {source}")
            verified.append({"path": _display_path(source), "sha256": _sha(source)})
        diagnostic = run_context.get("policy_funding_diagnostic")
        if isinstance(diagnostic, dict) and diagnostic.get("status") == "post_run_descriptive_quality_audit":
            diagnostic_path = diagnostic.get("path")
            diagnostic_sha = diagnostic.get("sha256")
            if (not isinstance(diagnostic_path, str)
                    or not re.fullmatch(r"[0-9a-fA-F]{64}", str(diagnostic_sha))
                    or not any(item["path"] == diagnostic_path
                               and item["sha256"].lower() == diagnostic_sha.lower()
                               for item in verified)):
                raise ValueError(f"{policy}: P010 diagnostic lacks verified SHA evidence")
        served = provenance.get("served_model_provenance")
        if served is not None:
            if not isinstance(served, dict) or not isinstance(served.get("model_id"), str):
                raise ValueError(f"{policy}: malformed served-model provenance")
            model_id = served["model_id"]
            for arm in ("on", "off"):
                digest = served.get(f"{arm}_evidence_sha256")
                matches = [item for item in verified if item["sha256"] == digest
                           and Path(item["path"]).name == "served_model_evidence.json"]
                if not matches:
                    raise ValueError(f"{policy}: unverified {arm} served-model evidence")
                source = Path(matches[0]["path"])
                if not source.is_absolute():
                    source = ROOT / source
                snapshot = json.loads(source.read_text(encoding="utf-8"))
                if (snapshot.get("served_model_ids") != [model_id]
                        or f"--model-path {model_id}" not in
                        str(snapshot.get("server_command") or "")):
                    raise ValueError(f"{policy}: inconsistent {arm} served-model evidence")
            run_context["served_model_id"] = model_id
            run_context["served_model_on_evidence_sha256"] = served["on_evidence_sha256"]
            run_context["served_model_off_evidence_sha256"] = served["off_evidence_sha256"]
        entries = run.get("indicators")
        if not isinstance(entries, list):
            raise ValueError(f"{policy}: indicators must be a list")
        seen_ids = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{policy}: indicator must be an object")
            indicator_id = entry.get("id")
            key = (policy, indicator_id)
            if indicator_id in seen_ids:
                raise ValueError(f"{policy}: unknown or duplicate indicator {indicator_id}")
            seen_ids.add(indicator_id)
            value = entry.get("simulation")
            unit = entry.get("simulation_unit")
            if value is not None and (not _number(value) or not isinstance(unit, str)
                                      or not unit.strip()):
                raise ValueError(f"{policy}/{indicator_id}: finite simulation and unit required")
            unit = {"percentage points": "%p", "share": "ratio",
                    "KRW": "원"}.get(unit, unit)
            ci = entry.get("ci")
            if ci is not None and (not isinstance(ci, list) or len(ci) != 2
                                   or not all(_number(x) for x in ci) or ci[0] > ci[1]):
                raise ValueError(f"{policy}/{indicator_id}: invalid confidence interval")
            n = entry.get("n")
            if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n <= 0):
                raise ValueError(f"{policy}/{indicator_id}: invalid sample size")
            alignment = entry.get("estimand_alignment")
            if alignment not in ("different", "unreviewed", "matched"):
                raise ValueError(f"{policy}/{indicator_id}: estimand alignment must be explicit")
            reason = entry.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{policy}/{indicator_id}: comparison reason required")
            if key not in by_key:
                if entry.get("exploratory_not_registered") is not True:
                    raise ValueError(f"{policy}: unregistered indicator {indicator_id}")
                if policy == "DISTANCING_2020" and indicator_id == "DS6-2023-GEO-PROXY":
                    _validate_geo_proxy_exploratory(entry)
                exploratory.append({"policy": policy, "policy_name": POLICY_NAMES.get(policy, policy),
                                    "id": indicator_id, "simulation": value,
                                    "simulation_unit": unit, "ci": ci, "n": n,
                                    "policy_funded_positive_citizen_days": entry.get(
                                        "policy_funded_positive_citizen_days"),
                                    "policy_funded_observed_citizen_days": entry.get(
                                        "policy_funded_observed_citizen_days"),
                                    "policy_funded_total_won": entry.get(
                                        "policy_funded_total_won"),
                                    "full_run_citizen_days": entry.get(
                                        "full_run_citizen_days"),
                                    "simulation_components": entry.get("simulation_components"),
                                    "geo_proxy_audit": entry.get("geo_proxy_audit"),
                                    "reason": reason, "simulation_method": entry.get("method"),
                                    "simulation_evidence": verified,
                                    "run_context": run_context,
                                    "score_label": f'{run.get("label") or policy} · {source_experiment}'})
                continue
            row = by_key[key]
            if value is None:
                row.update(status="시뮬 수치 없음", reason=reason,
                           score_label=f'{run.get("label") or policy} · {source_experiment}',
                           off=run.get("off"), on=run.get("on"),
                           sample_citizens=sample_citizens,
                           run_context=run_context,
                           simulation_method=entry.get("method"),
                           simulation_evidence=verified,
                           empirical_variant=entry.get("empirical_variant"))
                continue
            if policy == "P012" and indicator_id in ("P012-4", "P012-6") and not entry.get("empirical_variant"):
                raise ValueError(f"{indicator_id}: cashback reference period variant required")
            if policy == "P012" and indicator_id == "P012-5":
                parts = entry.get("simulation_components")
                if (not isinstance(parts, dict)
                        or not _number(parts.get("appliance_furniture_pct"))
                        or not _number(parts.get("hair_beauty_pct"))):
                    raise ValueError("P012-5: both sector proxy components required")
            row.update(simulation=value, simulation_unit=unit, ci=ci, n=n,
                       status={"different": "추정량 다름", "unreviewed": "정의 미감사",
                               "matched": "정의 일치 주장·별도 감사 필요"}[alignment],
                       reason=reason,
                       score_label=f'{run.get("label") or policy} · {source_experiment}',
                       off=run.get("off"), on=run.get("on"),
                       sample_citizens=sample_citizens,
                       run_context=run_context,
                       n_label="캐시백 수령자" if indicator_id in ("P012-4", "P012-6") else "시민",
                       simulation_method=entry.get("method"),
                       simulation_components=entry.get("simulation_components"),
                       simulation_evidence=verified,
                       empirical_variant=entry.get("empirical_variant"),
                       estimand_alignment=alignment,
                       gap=None, gap_unit=None, external_direction_match=None)
            if policy == "P012" and indicator_id == "P012-6" and value == 0:
                month_evidence = [item for item in verified
                                  if Path(item["path"]).name == "paired_cashback_month.json"]
                if len(month_evidence) != 1:
                    raise ValueError("P012-6 zero rate needs one verified monthly cashback score")
                month_path = Path(month_evidence[0]["path"])
                if not month_path.is_absolute():
                    month_path = ROOT / month_path
                month = json.loads(month_path.read_text(encoding="utf-8"))
                if (month.get("policy_id") != "P012" or month.get("month") != "2021-10"
                        or month.get("complete_paired_matrix") is not True
                        or month.get("recipients") != n
                        or month.get("capped_recipients") != 0
                        or (month.get("metrics") or {}).get("cap_share_recipients", {}).get("value") != 0):
                    raise ValueError("P012-6 zero-cap evidence differs from numeric score")
                row["binomial_zero_count_audit"] = {
                    "capped_recipients": 0, "recipients": n,
                    "two_sided_95_upper_pct": 100 * (1 - 0.025 ** (1 / n)),
                    "source_path": month_evidence[0]["path"],
                    "source_sha256": month_evidence[0]["sha256"],
                    "assumption": "independent-binomial-recipient-reference; not model uncertainty",
                }
            if entry.get("direction_comparable") is True and _number(row["truth"]):
                row["proxy_direction_same"] = ((value > 0 and row["truth"] > 0)
                                               or (value < 0 and row["truth"] < 0))
        run_evidence.append({"policy": policy, "label": run.get("label"),
                             "source_experiment": source_experiment,
                             "citizens": sample_citizens,
                             "run_context": run_context,
                             "off": run.get("off"), "on": run.get("on"),
                             "evidence": verified, "indicator_count": len(entries)})
    for row in rows:
        if row["status"] == "미실행":
            if row["policy"] in seen_policies:
                row.update(status="지표 미산출",
                           reason="정책 팔은 실행됐지만 이 지표의 시뮬 수치는 산출되지 않았습니다.")
            else:
                row.update(status="정책 미실행",
                           reason="이번 보고서 묶음에 이 정책의 v53 시뮬 원장이 아직 없습니다.")
    prompt_hashes = [item["run_context"].get("generic_prompt_sha256")
                     for item in run_evidence]
    known_prompt_hashes = {value for value in prompt_hashes if value}
    if len(known_prompt_hashes) > 1:
        raise ValueError("policy runs do not share the same v53 generic prompt SHA256")
    shared_prompt_verified = (len(prompt_hashes) >= 2
                              and all(prompt_hashes)
                              and len(known_prompt_hashes) == 1)
    _annotate_comparability(rows, strict=scoring_path.resolve() == SCORING.resolve())
    from collections import Counter
    unregistered = []
    for path in sorted((ROOT / "data/neo4j_load/policies").glob("P???.json")):
        if POLICY_ID_TO_SCORE.get(path.stem) not in policies:
            policy_file = json.loads(path.read_text(encoding="utf-8"))
            unregistered.append({"id": path.stem, "name": policy_file.get("name", path.stem),
                                 "path": _display_path(path), "status": "검증지표 미등록"})
    return {"experiment": experiment, "report_kind": "multi_policy_proxy",
            "prompt_variant": "v53", "scoring_table": {"path": _display_path(scoring_path),
                                                 "sha256": table_hash},
            "comparability_notes": {"path": _display_path(COMPARABILITY_NOTES),
                                    "sha256": _sha(COMPARABILITY_NOTES)},
            "score_files": [{"path": _display_path(path), "sha256": _sha(path),
                             "policy": "multi-policy-v53", "scoring_table_matches": True}
                            for path in paths],
            "indicator_count": len(rows), "policy_count": len(policies),
            "simulated_count": sum(_number(row["simulation"]) for row in rows),
            "direct_gap_count": 0, "tally": dict(Counter(row["status"] for row in rows)),
            "rows": rows, "unregistered_policies": unregistered,
            "run_evidence": run_evidence,
            "source_experiments": source_experiments,
            "shared_prompt_sha256_verified": shared_prompt_verified,
            "generic_prompt_sha256": next(iter(known_prompt_hashes), None),
            "exploratory_simulations": exploratory}


def _fmt(value, unit):
    if not _number(value):
        return "—"
    if unit in ("원",):
        return f"{value:+,.0f}{unit}"
    if unit in ("ratio", "비율차", "log-point"):
        shown_unit = "비율" if unit == "ratio" else unit
        return f"{value:+.4f} {shown_unit}"
    return f"{value:+.2f}{unit or ''}"


def _esc(value):
    return html.escape(str("" if value is None else value), quote=True)


def _track(row):
    sim = row["simulation"]
    if not _number(sim):
        return '<div class="track empty"><span class="zero"></span></div>'
    ci = row["ci"]
    truth = row["truth"] if row["gap"] is not None else None
    vals = [abs(sim)] + ([abs(v) for v in ci] if ci else []) + ([abs(truth)] if truth is not None else [])
    limit = max(1.0 if row["simulation_unit"] in ("%", "%p") else 0.01, max(vals) * 1.15)
    pos = lambda v: max(0.0, min(100.0, 50 + 50 * v / limit))
    bits = ['<span class="zero"></span>']
    if ci:
        a, b = sorted((pos(ci[0]), pos(ci[1])))
        bits.append(f'<span class="ci" style="left:{a:.2f}%;width:{b-a:.2f}%"></span>')
    bits.append(f'<span class="mk sim" style="left:{pos(sim):.2f}%"></span>')
    if truth is not None:
        bits.append(f'<span class="mk tru" style="left:{pos(truth):.2f}%"></span>')
    return '<div class="track">' + ''.join(bits) + '</div>'


def _coverage_html(coverage: list[dict], unregistered_count: int = 0) -> str:
    body = []
    included = [item for item in coverage if item["empirical_numeric_count"]]
    exploratory_only = [item for item in coverage
                        if not item["empirical_numeric_count"]
                        and item.get("exploratory_numeric_count", 0)]
    excluded_indicators = sum(item["unmeasured_count"] for item in coverage)
    for item in included:
        missing = item["missing_simulation_reasons"]
        in_progress = item.get("in_progress") is True
        if in_progress and missing:
            action = "진행 중"
        elif missing:
            action = "실측 수치는 있으나 시뮬 수치가 미산출됨. 아래 사유 확인"
        else:
            action = "이번 실험에서 실측·시뮬 수치가 모두 있는 지표 표시됨"
        details = ("<details><summary>실측값과 시뮬 미산출 사유</summary><ul>"
                   + ''.join(f'<li><strong>{_esc(entry["id"])}</strong>: '
                             f'실측 {_esc(_fmt(entry["truth"], entry["truth_unit"]))}. '
                             f'{_esc(entry["reason"])}</li>' for entry in missing)
                   + '</ul></details>') if missing and not in_progress else ''
        sample = (str(item["sample_citizens"]) + "명" if item["sample_citizens"] else
                  "진행 중" if in_progress else "미실행")
        body.append(
            '<tr>'
            f'<th scope="row">{_esc(item["policy_name"])}</th>'
            f'<td>{_esc(sample)}</td>'
            f'<td>{item["empirical_numeric_count"]}</td>'
            f'<td>{item["simulation_numeric_count"]}</td>'
            f'<td>{item["paired_numeric_count"]}</td>'
            f'<td>{_esc(action)}{details}</td>'
            '</tr>'
        )
    exploratory_note = ''
    if exploratory_only:
        labels = ', '.join(f'{_esc(item["policy_name"])} '
                           f'{item["exploratory_numeric_count"]}쌍'
                           for item in exploratory_only)
        exploratory_note = (
            '<p class="balance"><strong>주지표 밖 탐색 숫자가 있는 정책:</strong> '
            f'{labels}. 등록 검증지표의 실측 숫자는 없어서 주 비교표에는 '
            '넣지 않았습니다. 아래 별도 탐색 카드에서 실측·시뮬 값을 '
            '나란히 보되 효과 오차로 채점하지 않습니다.</p>'
        )
    return ('<section class="note"><h2>실측 수치가 있는 정책의 숫자 확보 현황</h2>'
            '<p>비교 행에는 실측과 이번 실험의 시뮬 수치가 모두 있는 지표만 표시합니다. '
            f'등록 지표 중 실측 숫자가 없는 {excluded_indicators}개는 HTML의 '
            '정책 카드·부록 목록에서 제외했습니다. '
            + (f'검증지표가 등록되지 않은 정책 파일 {unregistered_count}개도 '
               '숫자 비교 대상에서 제외했습니다. ' if unregistered_count else '')
            + '수치 쌍이 '
            '측정 정의까지 일치한다는 뜻은 아닙니다.</p>'
            '<div class="coverage-scroll"><table class="coverage"><thead><tr>'
            '<th>정책</th><th>시뮬 시민</th><th>실측 수치</th><th>시뮬 수치</th><th>나란히 표시</th><th>남은 작업</th>'
            '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'
            + exploratory_note + '</section>')


def _exploratory_html(pairs: list[dict]) -> str:
    cards_by_policy: dict[str, list[str]] = {}
    policy_names = {}
    contexts = {}
    for row in pairs:
        truth_unit = row.get("truth_unit")
        sim_unit = row.get("simulation_unit")
        truth = (_fmt(row["truth"], truth_unit)
                 if truth_unit in ("%", "%p", "ratio", "원", "log-point")
                 else f'{row["truth"]:+.4f} {truth_unit or ""}')
        sim = (_fmt(row["simulation"], sim_unit)
               if sim_unit in ("%", "%p", "ratio", "원", "log-point")
               else f'{row["simulation"]:+.4f} {sim_unit or ""}')
        if row["policy"] == "P010":
            scope_note = ("실측은 쿠폰 사용처의 카드결제 분포, 시뮬은 지급액으로 결제된 POI 분포입니다. "
                          "업종 대응과 모집단을 감사하기 전에는 정식 효과 점수로 쓰지 않습니다.")
        elif row["policy"] == "DISTANCING_2020":
            scope_note = ("실측은 2020년 서울 신한카드 가맹점 패널의 관광특구·발달상권 "
                          "전년 대비 매출 변화이고, 시뮬은 2023-10-23 상권 경계를 "
                          "2026년 3월 POI에 적용한 3일 쌍체 ON−OFF 영수증 대리값입니다. "
                          "연도·장소 표본·기간·추정량이 달라 정식 DS-6과 직접 비교하거나 "
                          "정책 효과 오차로 채점하지 않습니다.")
        elif row["policy"] == "P014":
            scope_note = (
                "실측은 지역화폐 발행 강도에 대한 지역·연도별 업종 매출 로그회귀계수입니다. "
                "시뮬은 정책 문구와 자치구 가맹점 적격 표시에 대한 시민의 짧은 ON−OFF "
                "POI 장소·소비 반응 대리값입니다. 이번 정책 구현은 price_discount 유형이며 "
                "할인 정산 경로가 활성화되지 않았고 상품권 구매·잔액·상환 지갑 원장이 "
                "없습니다. policy_hits도 실제 자치구·상호 적격 결제 건수를 세지 않습니다. "
                "원문 KSIC 47121/47129와 시뮬 POI 하위명 사이의 업종 대응도 "
                "감사되지 않았습니다. 따라서 이 숫자는 "
                "상품권 거래 효과나 사용률이 아니고, 정식 효과 오차·적중률로 채점하지 않습니다."
            )
        else:
            scope_note = ("실측은 지역화폐 발행 강도에 대한 지역·연도별 업종 매출 회귀계수, "
                          "시뮬은 시민의 단기 ON−OFF POI 지출 변화입니다. 정식 효과 점수로 쓰지 않습니다.")
        denominator_html = ""
        if row["policy"] == "P010":
            positive = row.get("policy_funded_positive_citizen_days")
            observed = row.get("policy_funded_observed_citizen_days")
            funded_won = row.get("policy_funded_total_won")
            full_run = row.get("full_run_citizen_days")
            if all(isinstance(value, int) and not isinstance(value, bool)
                   for value in (positive, observed, funded_won, full_run)):
                if not (0 <= positive <= observed <= full_run and funded_won >= 0):
                    raise ValueError("invalid P010 funded-spending denominator metadata")
                denominator_html = (
                    '<p class="balance"><strong>업종비중 분모 주의:</strong> '
                    f'정책 시행기간에 정책결제액이 양수인 시민×일 {_esc(positive)}/{_esc(observed)}, '
                    f'정책결제 총액 {funded_won:,}원. '
                    f'전체 원장은 {_esc(full_run)} 시민×일(시행 전 포함)입니다. '
                    f'시민 {_esc(row.get("n") or "미확인")}명이라는 부트스트랩 표본 수는 '
                    '실제 결제 관측 수가 아닙니다. '
                    '이 작은 분모의 업종비중으로 크기 적중을 판단할 수 없습니다. '
                    '개별 거래 건수는 시민×일 집계 원장에서 알 수 없습니다. '
                    '정책 카드의 별도 진단에서 선택 필드인 Stage1 지갑 태세는 0이고 '
                    'Stage2 양수 요청은 극소수로 관찰됐습니다. 필수 출력 위반은 아니며, '
                    '프롬프트·파서·스키마·결제선택 로직 중 원인은 미확정입니다.</p>'
                )
            else:
                denominator_html = (
                    '<p class="balance">정책결제 분모가 검증되지 않아 업종비중의 '
                    '크기를 판단할 수 없습니다. 시민 부트스트랩 표본 수는 결제 건수가 아닙니다.</p>'
                )
            subclass_audit = row.get("funded_subclass_display_audit")
            if isinstance(subclass_audit, dict):
                destinations = ', '.join(
                    f'{_esc(name)} {amount:,}원'
                    for name, amount in sorted(subclass_audit["funded_by_sub_won"].items()))
                if row["simulation"] == 0:
                    denominator_html += (
                        '<p class="balance"><strong>시뮬 0%의 의미:</strong> '
                        f'이 실행의 정책지갑 결제 총액 {subclass_audit["funded_total_won"]:,}원은 '
                        f'{destinations}에만 기록됐습니다. 이 지표의 대응 업종 분자는 '
                        '0원이라 시뮬 구성비가 0%입니다. 실측 업종 비중이 0%라는 뜻도, '
                        '정책 효과가 0이라는 뜻도 아닙니다. 정책지갑 결제 관측이 '
                        f'{subclass_audit["positive_citizen_days"]}/'
                        f'{subclass_audit["observed_citizen_days"]} 시민×일뿐이어서 '
                        '이 0%를 실측과의 크기 적중으로 채점할 수 없습니다.</p>'
                    )
        if row["policy"] == "DISTANCING_2020":
            denominator_html = _geo_proxy_exploratory_html(row)
        policy_names[row["policy"]] = row["policy_name"]
        contexts.setdefault(row["policy"], row.get("run_context"))
        cards_by_policy.setdefault(row["policy"], []).append('<div class="row">'
                     f'<div class="meta"><span class="id">{_esc(row["id"])}</span>'
                     f'<span class="desc">{_esc(row["policy_name"])} · '
                     f'{_esc(EXPLORATORY_NAMES.get(row["id"], row["id"]))}</span></div>'
                     '<div class="nums">'
                     f'<span class="tru">실측 {_esc(truth)}</span>'
                     f'<span class="sim">시뮬 {_esc(sim)}</span></div>'
                     f'<p class="reason">{_esc(scope_note)}</p>'
                     + denominator_html
                     + (f'<p class="reason source">실측 출처: {_esc(row["source"])}'
                        f' · {_esc(row.get("source_locator"))}</p>' if row.get("source") else '')
                     + _technical_details(row)
                     + '</div>')
    grouped = ''.join(
        '<div class="exploratory-group">'
        + f'<h3>{_esc(policy_names[policy])} <small>{_esc(policy)}</small></h3>'
        + (_run_context_html(contexts[policy]) if policy == "P014"
           and contexts.get(policy) else '')
        + ('<p class="balance">이 정책의 시뮬 입력은 price_discount와 지역 가맹점 '
           '적격 표시를 사용합니다. 할인 정산과 상품권 지갑·구매·잔액·상환 원장이 '
           '없으므로 '
           '아래 POI 지출은 상품권 거래 효과 또는 사용률이 아닙니다.</p>'
           if policy == "P014" else '')
        + '<div class="rows">' + ''.join(cards) + '</div></div>'
        for policy, cards in cards_by_policy.items()
    )
    return ('<section class="pol"><h2>등록 38개 지표 밖의 탐색 참고값</h2>'
            '<p class="runline">실측과 시뮬 수치는 있지만 정책 효과 채점표의 검증지표가 아닙니다. '
            '표본·기간·추정량이 달라 외부 효과 오차 또는 적중률에 포함하지 않습니다.</p>'
            + grouped + '</section>')


def _geo_proxy_exploratory_html(row: dict) -> str:
    empirical = {item.get("hub"): item.get("value")
                 for item in row.get("empirical_components") or []}
    simulation = row.get("simulation_components") or {}
    audit = row.get("geo_proxy_audit") or {}
    pairs = (("관광특구", "tourism_special_zone", "tourism_special_zone_pct"),
             ("발달상권", "developed_commercial_district", "developed_commercial_district_pct"))
    if not all(_number(empirical.get(source)) and _number(simulation.get(proxy))
               for _, source, proxy in pairs):
        raise ValueError("DS6 geo proxy requires empirical and simulation components")
    rows = ''.join('<tr>'
                   f'<th>{_esc(label)}</th>'
                   f'<td>{_esc(_fmt(empirical[source], "%"))}</td>'
                   f'<td>{_esc(_fmt(simulation[proxy], "%"))}</td>'
                   '</tr>' for label, source, proxy in pairs)
    den = audit["off_denominator_won_by_type"]
    sparse = audit.get("sparse_interpretation_blocked") is True
    n_label = _esc(row.get("n") or "미확인")
    match_label = _esc(_fmt(100 * audit["match_rate"], "%"))
    overlap_parts = []
    for arm, label in (("on", "ON"), ("off", "OFF")):
        count = audit["overlap_count_by_arm"][arm]
        total = audit["total_receipt_count_by_arm"][arm]
        rate = 100 * audit["overlap_rate_by_arm"][arm]
        overlap_parts.append(f'{label} {count:,}/{total:,} ({rate:.2f}%)')
    overlap_label = _esc(', '.join(overlap_parts))
    boundary_sha = _esc(audit["source_boundary_sha256"])
    return ('<div class="component-breakdown"><strong>상권별 원수치와 결합 관문</strong>'
            '<table><thead><tr><th>상권 유형</th><th>실측 2020 전년 대비</th>'
            '<th>시뮬 3일 ON−OFF</th></tr></thead><tbody>' + rows + '</tbody></table>'
            f'<p>시뮬 시민 n={n_label}, '
            f'영수증 좌표 결합률 {match_label}, '
            f'애매한 상권 겹침 영수증 {overlap_label}; 두 유형 모두에서 제외하고 '
            '어느 한쪽에 우선 배정하지 않았습니다(각 팔 허용 상한 1%). '
            f'관광특구 OFF 분모 {den["tourism_special_zone"]:,}원, '
            f'발달상권 OFF 분모 {den["developed_commercial_district"]:,}원.</p>'
            + ('<p>희소한 관측 때문에 이 숫자는 기술값일 뿐 방향·크기 의미를 '
               '판정하지 않습니다.</p>' if sparse else
               '<p>이 숫자는 공간 대리값의 기술통계입니다. 2020 실측과의 방향·크기 '
               '일치 판정이나 정책 효과 오차로 사용하지 않습니다.</p>')
            + f'<p>2023 경계 ZIP SHA256 {boundary_sha}.</p>'
            + '</div>')


def _technical_details(row: dict) -> str:
    fields = []
    for label, key in (("실측 원문 정의", "empirical_estimand"),
                       ("실측 원문 단위", "truth_unit_full"),
                       ("실측 원문 기간", "empirical_period"),
                       ("시뮬 계산식", "simulation_method"),
                       ("원문·시뮬 차이", "empirical_comparison_reason"),
                       ("시뮬 원자료 사유", "reason")):
        if row.get(key):
            fields.append(f'<p><strong>{_esc(label)}:</strong> {_esc(row[key])}</p>')
    evidence = row.get("simulation_evidence") or []
    if evidence:
        fields.append('<p><strong>시뮬 증거 SHA256:</strong> '
                      + _esc('; '.join(item["path"] + ' ' + item["sha256"]
                                       for item in evidence)) + '</p>')
    subclass_audit = row.get("funded_subclass_display_audit")
    if isinstance(subclass_audit, dict):
        fields.append('<p><strong>정책결제 업종 사후 표시 감사:</strong> '
                      + _esc(subclass_audit["path"] + ' SHA256 '
                             + subclass_audit["sha256"]) + '</p>')
    zero_cap = row.get("binomial_zero_count_audit")
    if isinstance(zero_cap, dict):
        fields.append('<p><strong>10월 0건 상한률 월말 원장:</strong> '
                      + _esc(zero_cap["source_path"] + ' SHA256 '
                             + zero_cap["source_sha256"]) + '</p>')
    sector_audit = row.get("sector_denominator_display_audit")
    if isinstance(sector_audit, dict):
        fields.append('<p><strong>업종 분모 사후 표시 감사:</strong> '
                      + _esc(sector_audit["path"] + ' SHA256 '
                             + sector_audit["sha256"]) + '</p>')
    return ('<details class="technical"><summary>원문 정의·산식·증거 자세히 보기</summary>'
            + ''.join(fields) + '</details>') if fields else ''


def _p012_rank_components(row: dict) -> str:
    if row.get("id") != "P012-5":
        return ""
    empirical = {part.get("sector"): part.get("value")
                 for part in row.get("empirical_components") or []}
    simulation = row.get("simulation_components") or {}
    pairs = (("가전·가구", "appliances_furniture", "appliance_furniture_pct"),
             ("이·미용", "hair_beauty", "hair_beauty_pct"))
    if not all(_number(empirical.get(source)) and _number(simulation.get(proxy))
               for _, source, proxy in pairs):
        return ""
    cells = ''.join(
        f'<tr><th scope="row">{_esc(label)}</th>'
        f'<td>{_esc(_fmt(empirical[source], "log-point"))}</td>'
        f'<td>{_esc(_fmt(simulation[proxy], "%"))}</td></tr>'
        for label, source, proxy in pairs
    )
    audit = row.get("sector_denominator_display_audit")
    denominator_note = ''
    if isinstance(audit, dict):
        appliance = audit["appliance_furniture"]
        beauty = audit["hair_beauty"]
        denominator_note = (
            '<p><strong>시뮬 원화 분모 주의:</strong> '
            f'가전·가구 POI는 ON {appliance["on_won"]:,}원 / '
            f'OFF {appliance["off_won"]:,}원, '
            f'이·미용 POI는 ON {beauty["on_won"]:,}원 / '
            f'OFF {beauty["off_won"]:,}원입니다. '
            f'{audit["paired_citizen_days"]} 시민×일에서 가전·가구 OFF 분모가 작아 '
            f'시뮬 {_esc(_fmt(row["simulation"], "%p"))}의 크기가 매우 불안정하고 '
            '재표집 구간도 넓습니다. '
            '실측의 월별 가구 삼중차분 로그계수와 같은 추정량이 아닙니다.</p>'
        )
    return ('<div class="component-breakdown"><strong>두 업종의 원수치</strong>'
            '<table><thead><tr><th>업종</th><th>실측 로그회귀 계수</th>'
            '<th>시뮬 ON−OFF 변화율</th></tr></thead><tbody>' + cells + '</tbody></table>'
            + denominator_note
            + '<p>실측 0.3336 log-point는 두 회귀계수의 차이이고, '
            '시뮬 값은 두 업종 퍼센트 변화율의 차이입니다. '
            '로그계수 차이를 퍼센트로 변환해도 퍼센트포인트 차이가 되지 않으므로 '
            '두 차이를 빼거나 적중률로 채점하지 않습니다.</p></div>')


def _run_context_html(context: dict | None) -> str:
    if not context:
        return ""
    pieces = [f'정책 입력 {_esc(context.get("policy_id") or "미확인")}']
    if context.get("policy_input_file"):
        pieces.append('정책 파일 ' + _esc(Path(context["policy_input_file"]).name))
    if context.get("effective_from") or context.get("effective_until"):
        pieces.append('시행 ' + _esc(context.get("effective_from") or "시작 미확인")
                      + ' ~ ' + _esc(context.get("effective_until") or "종료 미확인"))
    env_on, env_off = context.get("on_environment_id"), context.get("off_environment_id")
    if env_on or env_off:
        pieces.append('환경 ON ' + _esc(env_on or "미확인") + ' / OFF ' + _esc(env_off or "미확인"))
    prompt_hash = context.get("generic_prompt_sha256")
    pieces.append('범용 프롬프트 v53 ' + (_esc(prompt_hash[:12]) + '…' if prompt_hash
                                      else '(바이트 지문 미확인)'))
    if context.get("served_model_id"):
        pieces.append('실제 서빙 모델 ' + _esc(context["served_model_id"]))
    detail = []
    for label, key in (("정책 파일", "policy_input_file"),
                       ("정책 파일 SHA256", "policy_input_sha256"),
                       ("환경 쌍체 지문", "paired_environment_fingerprint"),
                       ("범용 프롬프트 SHA256", "generic_prompt_sha256"),
                       ("Stage2 SHA256", "stage2_sha256"),
                       ("ON 모델 증거 SHA256", "served_model_on_evidence_sha256"),
                       ("OFF 모델 증거 SHA256", "served_model_off_evidence_sha256")):
        if context.get(key):
            detail.append(f'<p><strong>{_esc(label)}:</strong> {_esc(context[key])}</p>')
    quality = context.get("quality_audit")
    if isinstance(quality, dict) and all(isinstance(quality.get(arm), dict)
                                         for arm in ("on", "off")):
        quality_lines = []
        quality_details = []
        for arm, label in (("on", "ON"), ("off", "OFF")):
            item = quality[arm]
            if item.get("audit_status") == "unavailable":
                quality_lines.append(f'{label} 첫 시도·Stage2 품질 감사 불가')
                if item.get("reason"):
                    quality_details.append(f'{label} 감사 불가 사유: {_esc(item["reason"])}')
                continue
            total = item.get("citizen_days")
            first = item.get("stage1_first_attempt_internal_validation_pass_count")
            final = item.get("stage1_final_ok_count")
            outer_recovered = item.get("stage1_outer_retry_recovered_count", 0)
            unknown = item.get("stage1_first_attempt_unknown_count", 0)
            fallback = item.get("stage2_fallback_only_count")
            repair = item.get("stage2_choice_repair_count")
            if not all(isinstance(value, int) and not isinstance(value, bool)
                       for value in (total, first, final, outer_recovered, unknown,
                                     fallback, repair)):
                raise ValueError("malformed run quality audit")
            if total <= 0 or any(value < 0 or value > total
                                 for value in (first, final, outer_recovered, unknown,
                                               fallback, repair)):
                raise ValueError("out-of-range run quality audit")
            if first + unknown > total:
                raise ValueError("first-attempt quality count exceeds citizen-days")
            if unknown:
                first_label = (f'첫 기록의 Stage1 내부검증 성공 확인 {first}/{total}; '
                               f'이력 불명 {unknown}, 가능 범위 '
                               f'{100 * first / total:.1f}~'
                               f'{100 * (first + unknown) / total:.1f}%')
            else:
                first_label = (f'Stage1 첫 시도 내부검증 {first}/{total} '
                               f'({100 * first / total:.1f}%)')
            quality_lines.append(
                f'{label} 시민×일 {total}: {first_label}, '
                f'외부 재시도 회복 {outer_recovered}, 최종 원장 성공 {final}/{total}; '
                f'Stage2 fallback만 사용 {fallback}, 선택 보정 {repair}'
            )
            if item.get("stage2_quality_gate_pass") is not True:
                quality_lines[-1] += ' · Stage2 품질 관문 미통과'
            optimistic = item.get("stage1_successful_invocation_first_attempt_pass_rate")
            if _number(optimistic):
                quality_details.append(f'{label} 성공한 마지막 호출 안의 첫 시도 통과율 '
                                       f'{100 * optimistic:.1f}%는 이전 외부 실패를 '
                                       '제외하므로 낙관적인 진단값입니다.')
        quality_html = ('<p class="qualityline"><strong>생성·복구 품질:</strong> '
                        + ' / '.join(_esc(line) for line in quality_lines)
                        + '. 첫 시도 내부검증률은 원시 첫응답의 엄격 형식률과 다릅니다. '
                        '최종 원장이 완결돼도 첫 시도 품질이 높았다는 뜻은 아닙니다.</p>'
                        + ('<details class="technical"><summary>품질 감사 상세</summary>'
                           + ''.join(f'<p>{text}</p>' for text in quality_details)
                           + '</details>' if quality_details else ''))
        if any(quality[arm].get("audit_status") == "unavailable"
               for arm in ("on", "off")):
            quality_html += ('<p class="balance">한쪽 팔의 생성 품질 증거가 없어 '
                             '정책 후 차이의 인과효과 해석과 프롬프트 성능 판정을 보류합니다.</p>')
        if context.get("policy_id") == "P012":
            on, off = quality["on"], quality["off"]
            on_total, off_total = on.get("citizen_days"), off.get("citizen_days")
            on_first = on.get("stage1_first_attempt_internal_validation_pass_count")
            off_first = off.get("stage1_first_attempt_internal_validation_pass_count")
            if all(isinstance(value, int) and not isinstance(value, bool)
                   for value in (on_total, off_total, on_first, off_first)) and on_total and off_total:
                quality_html += (
                    '<p class="balance"><strong>P012 양팔 생성 품질 차이:</strong> '
                    f'Stage1 첫 기록 내부검증 ON {on_first}/{on_total} '
                    f'({100 * on_first / on_total:.1f}%), '
                    f'OFF {off_first}/{off_total} ({100 * off_first / off_total:.1f}%). '
                    '최종 원장이 완결돼도 이 차이는 사라지지 않습니다. '
                    '재시도·보정 경로가 양팔 행동 기록에 다르게 작용했을 가능성이 있어 '
                    '현재 지출 차이를 정책 효과의 정확도나 범용 프롬프트의 '
                    '최적 성능으로 단정하지 않습니다.</p>'
                )
    else:
        quality_html = ('<p class="qualityline">이 팔의 첫 시도 생성 품질은 '
                        '현재 보고서에서 검증되지 않았습니다. 최종 원장 완결과 구분해야 합니다.</p>')
    balance = context.get("preperiod_balance")
    if isinstance(balance, dict):
        status = balance.get("status")
        blocked = balance.get("post_effect_causal_interpretation_blocked") is True
        if status == "pass":
            balance_message = ("시행 전 ON/OFF 균형 사전 관문 통과. "
                               "이것만으로 정책 인과효과나 실측 적중이 입증되지는 않습니다.")
        elif status == "fail":
            balance_message = "시행 전 ON/OFF 균형 관문 실패. 정책 후 차이를 인과효과로 해석할 수 없습니다."
        elif status == "historical_not_preregistered":
            balance_message = ("시행 전 ON/OFF 차이가 있었고 관문을 사전등록하지 않았습니다. "
                               "정책 후 차이를 인과효과로 해석할 수 없습니다.")
        elif blocked:
            balance_message = "시행 전 균형을 검증할 자료가 없어 정책 후 차이를 인과효과로 해석할 수 없습니다."
        else:
            balance_message = "시행 전 균형 검증 상태를 확인해야 합니다."
        comparisons = balance.get("comparisons") or {}
        balance_detail = []
        balance_names = {
            "recorded_total_spend": "기록된 총지출",
            "eligible_offline_spend": "적격 오프라인 지출",
            "offline_spend_proxy": "오프라인 지출 대리지표",
            "target_poi_spend_proxy": "대상 판매처 지출 대리지표",
            "supermarket_poi_spend_proxy": "슈퍼마켓 지출 대리지표",
            "food_store_poi_spend_proxy": "식료품점 지출 대리지표",
        }
        if isinstance(comparisons, dict):
            for metric, values in comparisons.items():
                if not isinstance(values, dict):
                    continue
                pct = values.get("difference_pct_of_off")
                if not _number(pct) and _number(values.get("relative_gap")):
                    pct = 100 * values["relative_gap"]
                if _number(pct):
                    balance_detail.append(f'{_esc(balance_names.get(metric, metric))} '
                                          '시행 전 (ON−OFF)/OFF '
                                          f'{_esc(_fmt(pct, "%"))}')
        balance_html = ('<p class="balance">' + _esc(balance_message) + '</p>'
                        + ('<details class="technical"><summary>시행 전 차이 수치 보기</summary>'
                           + '<p>' + '; '.join(balance_detail) + '</p></details>'
                           if balance_detail else ''))
    else:
        balance_html = ''
    diagnostic = context.get("policy_funding_diagnostic")
    diagnostic_html = ''
    if (isinstance(diagnostic, dict)
            and diagnostic.get("status") == "post_run_descriptive_quality_audit"):
        keys = ("citizens", "positive_purchase_events", "eligible_purchase_events",
                "positive_purchase_won", "eligible_purchase_won", "funded_purchase_events",
                "funded_won", "stage1_grant_style_present_citizen_days",
                "stage1_grant_use_present_citizen_days", "policy_request_positive_citizen_days",
                "policy_requested_won")
        if (not all(isinstance(diagnostic.get(key), int)
                    and not isinstance(diagnostic[key], bool) and diagnostic[key] >= 0
                    for key in keys)
                or not isinstance(diagnostic.get("days"), list)
                or not diagnostic["days"]):
            raise ValueError("malformed P010 policy funding diagnostic")
        days = diagnostic["citizens"] * len(diagnostic["days"])
        won = lambda key: f'{diagnostic[key]:,}원'
        diagnostic_html = (
            '<p class="balance"><strong>정책결제 경로의 사후 진단:</strong> '
            f'시행기간 Neo4j INCLUDES 기준 적격 구매 이벤트 '
            f'{_esc(diagnostic["eligible_purchase_events"])}/'
            f'{_esc(diagnostic["positive_purchase_events"])}, '
            f'적격 구매액 {_esc(won("eligible_purchase_won"))}/'
            f'전체 구매액 {_esc(won("positive_purchase_won"))}. '
            f'Stage1 지갑 태세 필드(style/use) '
            f'{_esc(diagnostic["stage1_grant_style_present_citizen_days"])}/'
            f'{_esc(diagnostic["stage1_grant_use_present_citizen_days"])} 시민×일 '
            f'(전체 {days}), Stage2 양수 정책결제 요청 '
            f'{_esc(diagnostic["policy_request_positive_citizen_days"])}/{days} 시민×일·'
            f'{_esc(won("policy_requested_won"))}, '
            f'실제 지갑결제 {_esc(diagnostic["funded_purchase_events"])} 구매 이벤트·'
            f'{_esc(won("funded_won"))}. '
            '적격 구매는 많았지만 선택 필드인 Stage1 지갑 태세와 Stage2 양수 요청은 '
            '드물거나 없었습니다. 이 필드들은 필수가 아니므로 모델의 지시 위반으로 '
            '해석하지 않습니다. 프롬프트·파서·스키마·결제선택 로직 중 '
            '어떤 구조가 지갑 결제를 적게 만들었는지는 추가 감사가 필요합니다. '
            '이 사후 진단은 정책 효과 채점에 쓰지 않습니다.</p>'
        )
        detail.append('<p><strong>정책결제 진단 원본:</strong> '
                      f'{_esc(diagnostic.get("path"))} '
                      f'SHA256 {_esc(diagnostic.get("sha256"))}</p>')
    channel = context.get("p010_channel_display_audit")
    channel_html = ''
    if isinstance(channel, dict):
        total = channel["total_gap_won"]
        online = channel["online_gap_won"]
        offline = channel["offline_gap_won"]
        channel_html = (
            '<p class="balance"><strong>P010 총지출 차이의 채널별 사후 분해:</strong> '
            f'시행기간 같은 시민×일 {channel["paired_citizen_days"]}개에서 '
            f'ON−OFF 총지출 {total:,.0f}원 중 온라인 모델 채널 '
            f'{online:,.0f}원({100 * channel["online_fraction"]:.1f}%), '
            f'오프라인 영수증 {offline:,.0f}원입니다. '
            f'온라인 채널은 쿠폰 사용 대상이 아니며 실제 지갑결제는 '
            f'{channel["wallet_won"]:,.0f}원입니다. '
            '이 산술 분해는 왜 지출 계획이 바뀌었는지 증명하지 않으며 '
            '실측 설문 MPC와의 정확도나 정책효과 크기 적중으로 해석하지 않습니다.</p>'
        )
        detail.append('<p><strong>채널 분해 원본:</strong> '
                      f'{_esc(channel["path"])} SHA256 {_esc(channel["sha256"])}</p>')
    concentration = context.get("p010_concentration_display_audit")
    concentration_html = ''
    if isinstance(concentration, dict):
        remainder = concentration["remaining_gap_won"]
        remainder_text = (f'나머지 시민의 순합은 {-remainder:,.0f}원 감소입니다. '
                          if remainder < 0 else
                          f'나머지 시민의 순합은 {remainder:,.0f}원 증가입니다. ')
        concentration_html = (
            '<p class="balance"><strong>P010 단기 수치의 시민별 집중:</strong> '
            f'{concentration["paired_citizens"]}명 중 상위 5명의 ON−OFF 지출차이 합은 '
            f'{concentration["top_five_sum_won"]:,.0f}원으로 전체 순증 '
            f'{concentration["net_gap_won"]:,.0f}원의 '
            f'{100 * concentration["top_five_share"]:.1f}%입니다. '
            + remainder_text + '이 3일 대리값의 크기는 소수 시민에 '
            '매우 민감합니다. 실제 인구의 정책효과 분포나 실측 MPC 정합을 '
            '뜻하지 않습니다.</p>'
        )
        detail.append('<p><strong>시민별 집중 감사 원본:</strong> '
                      f'{_esc(concentration["path"])} '
                      f'SHA256 {_esc(concentration["sha256"])}</p>')
    regime = context.get("distancing_input_display_audit")
    regime_html = ''
    if isinstance(regime, dict):
        dates = regime["dates"]
        counts = '→'.join(str(value) for value in regime["seoul_case_counts"])
        regime_html = (
            '<p class="balance"><strong>거리두기 입력의 실제 차이:</strong> '
            f'시뮬 날짜 {_esc(dates[0])}~{_esc(dates[-1])}에 양팔은 '
            f'같은 서울 신규 확진 배경({counts}명)을 받았습니다. '
            'ON만 수도권 2단계(식당 21시 이후 매장취식 제한, 카페 포장·배달만, '
            '집합 제한)를 받았고 OFF는 추가 방역 영업·모임 제한이 없습니다. '
            '범용 v53 프롬프트는 같고 정책 레짐을 담은 환경 입력이 다릅니다. '
            f'ON 환경 ID {_esc(regime["on_environment_id"])}의 숫자 2021은 '
            '내부 이름일 뿐 이 시뮬이나 실측의 연도가 아닙니다. '
            '이 근거는 동결 입력의 정적 렌더 감사이며 완성된 HTTP 요청 전수 '
            '캡처를 뜻하지 않습니다.</p>'
        )
        detail.append('<p><strong>거리두기 입력 렌더 감사:</strong> '
                      f'{_esc(regime["path"])} '
                      f'SHA256 {_esc(regime["sha256"])}</p>')
    p016 = context.get("p016_postfix_display_audit")
    p016_html = ''
    if isinstance(p016, dict):
        p016_html = (
            '<p class="balance"><strong>P016 실행 이력과 회계 패치:</strong> '
            '첫 사전검증에서 이전 P012 정책 노드가 남은 것을 모델 호출 0회에 '
            '발견했고, 그래프 초기화 순서를 고쳐 재검증을 통과했습니다. '
            f'이후 첫 ON 실행은 {_esc(p016["failure_date"])}에 '
            f'적격 구매가 없는 시민의 장부 키 오류가 원시 기록 '
            f'{p016["failed_rows"]}행에서 발생해 미완결·채점 제외했습니다. '
            '0원 즉시할인 키를 초기화하는 회계 코드 패치 뒤 ON/OFF를 '
            '새 run_id로 처음부터 실행했습니다. 범용 v53 프롬프트는 '
            '양팔에서 같은 바이트이며 패치 전 실패 팔은 어떤 지표에도 섞지 않았습니다. '
            '서버 저장소의 source_commit은 실행 파일 증거가 아니므로 '
            '양팔 frozen_inputs의 회계 코드 SHA와 원장·점수 SHA로 확인했습니다.</p>'
        )
        detail.append('<p><strong>P016 실패·패치 감사:</strong> '
                      f'{_esc(p016["path"])} SHA256 {_esc(p016["sha256"])}; '
                      f'회계 엔진 SHA256 {_esc(p016["patch_sha256"])}; '
                      f'실패 arm 스냅샷 SHA256 '
                      f'{_esc(p016["failed_snapshot_sha256"])}</p>')
    taxonomy = context.get("p016_taxonomy_display_audit")
    taxonomy_html = ''
    if isinstance(taxonomy, dict):
        on, off = taxonomy["on"], taxonomy["off"]
        taxonomy_html = (
            '<p class="balance"><strong>P016 C2·C3의 시뮬 0%p는 구조적 0:</strong> '
            f'양팔 {on["citizen_days_all"]}+{off["citizen_days_all"]} 시민×일 전부에서 '
            '청과·정육·슈퍼마켓·식료품 POI 지출 합이 마트 상위 분류 지출과 '
            '항상 같습니다. 시행기간에도 ON '
            f'{on["effect_target_poi_won"]:,}원, OFF '
            f'{off["effect_target_poi_won"]:,}원으로 분자와 분모가 각 팔에서 '
            '동일합니다. 따라서 두 proxy의 0은 분류 항등식의 결과이며 '
            '농축산물 상품 매출의 무반응, 실측의 0, 실측과의 불일치를 '
            '검증한 숫자가 아닙니다. C2·C3의 산술 차이와 방향 판정을 생략합니다.</p>'
        )
        detail.append('<p><strong>P016 분류 항등식 감사:</strong> '
                      f'{_esc(taxonomy["path"])} '
                      f'SHA256 {_esc(taxonomy["sha256"])}</p>')
    return ('<p class="contextline">' + ' · '.join(pieces) + '</p>'
            + quality_html + balance_html + diagnostic_html + channel_html
            + concentration_html + regime_html + p016_html + taxonomy_html
            + ('<details class="technical"><summary>정책·환경·프롬프트 지문 보기</summary>'
               + ''.join(detail) + '</details>' if detail else ''))


def render(report: dict, template_path: Path = TEMPLATE) -> str:
    sections = []
    rows = report["rows"]
    numeric_only = report.get("report_kind") == "numeric_pairs"
    for policy in dict.fromkeys(r["policy"] for r in rows):
        rs = [r for r in rows if r["policy"] == policy]
        score_line = next((f'{r["score_label"] or "무명"}'
                           + (f' · 시뮬 시민 {r["sample_citizens"]}명'
                              if r.get("sample_citizens") else '')
                           + f' · OFF {r["off"]} · ON {r["on"]}'
                           for r in rs if r["off"]), "이 실험에서 미실행")
        context_line = _run_context_html(rs[0].get("run_context")) if numeric_only else ""
        parts = [f'<section class="pol"><h2>{_esc(rs[0]["policy_name"])} '
                 f'<small>{_esc(policy)}</small></h2><p class="runline">{_esc(score_line)}</p>'
                 + context_line,
                 '<div class="rows">']
        for r in rs:
            cl = ("ok" if r["gap"] is not None else
                  "wait" if r["status"] in ("미실행", "관측 부족", "결과 누락", "참고 지표") else "sus")
            ref = _fmt(r["truth"], r["truth_unit"])
            sim = _fmt(r["simulation"], r["simulation_unit"])
            if r["truth_kind"] and r["truth"] is not None:
                ref += " · " + r["truth_kind"]
            nums = [f'<span class="tru">실측 {_esc(ref)}</span>',
                    f'<span class="sim">시뮬 {_esc(sim)}</span>']
            if r["gap"] is not None:
                nums.append(f'<span class="gap">시뮬−실측 {_esc(_fmt(r["gap"], r["gap_unit"]))}</span>')
            elif numeric_only and r.get("structural_zero_display_audit"):
                nums.append('<span class="memo">분류 항등식 0: 산술 차이·방향 판정 생략</span>')
            elif (numeric_only and ((r.get("run_context") or {}).get(
                    "preperiod_balance") or {}).get("status") == "fail"):
                nums.append('<span class="memo">시행 전 균형 실패: 크기·방향 적중 판정 보류</span>')
            elif numeric_only and r.get("binomial_zero_count_audit"):
                nums.append('<span class="memo">0건·소표본: 산술 차이 생략</span>')
            elif numeric_only and r.get("surface_comparison"):
                surface = r["surface_comparison"]
                nums.append(f'<span class="memo">탐색적 숫자상 차이 '
                            f'{_esc(_fmt(surface["delta"], surface["delta_unit"]))}'
                            f' · {_esc(surface["scale"])}</span>')
            elif numeric_only:
                nums.append('<span class="memo">서로 다른 단위·정의: 산술 차이 생략</span>')
            if r.get("binomial_zero_count_audit"):
                zero_cap = r["binomial_zero_count_audit"]
                nums.append('<span class="smalln">'
                            f'시뮬 상한 도달 {zero_cap["capped_recipients"]}/'
                            f'{zero_cap["recipients"]}명; 시민 재표집 [0, 0]은 퇴화. '
                            f'독립 이항 표본 가정의 양측 95% 상한 '
                            f'{zero_cap["two_sided_95_upper_pct"]:.2f}%'
                            '</span>')
            elif r.get("structural_zero_display_audit"):
                nums.append('<span class="smalln">재표집 [0, 0]은 같은 분류 항등식을 '
                            '반복한 값으로 효과 불확실성 구간이 아닙니다.</span>')
            elif r["ci"]:
                ci_label = ("시뮬 시민 재표집 95% 구간(모델·외부 표본 불확실성 미포함)"
                            if numeric_only else "95% 구간")
                nums.append(f'<span class="ciTxt">{_esc(ci_label)} '
                            f'[{_esc(_fmt(r["ci"][0], r["simulation_unit"]))}, '
                            f'{_esc(_fmt(r["ci"][1], r["simulation_unit"]))}]</span>')
            if r["n"] is not None:
                nums.append(f'<span class="n">{_esc(r.get("n_label") or "시민")} '
                            f'n={_esc(r["n"])}</span>')
            if numeric_only and r.get("sample_citizens") and r["sample_citizens"] <= 12:
                nums.append('<span class="smalln">표본 12명 이하: 방향·크기 매우 불확실</span>')
            if (numeric_only and r.get("ci") and not r.get("structural_zero_display_audit")
                    and r["ci"][0] <= 0 <= r["ci"][1]):
                nums.append('<span class="smalln">시뮬 방향 불확실: 재표집 구간에 0 포함</span>')
            if (r["simulation_unit"] == "%" and _number(r["raw_base"])
                    and _number(r["raw_mean"])
                    and "share" not in str(r["metric"])
                    and "ratio" not in str(r["metric"])):
                nums.append(f'<span class="memo">원값 {_esc(_fmt(r["raw_mean"], "원"))} / '
                            f'기준 {_esc(_fmt(r["raw_base"], "원"))}</span>')
            if r["internal_hit"] is not None:
                nums.append('<span class="memo">내부 등록판정: '
                            + ('일치' if r["internal_hit"] else '불일치') + '</span>')
            if r["external_direction_match"] is not None:
                nums.append('<span class="memo">외부 방향: '
                            + ('일치' if r["external_direction_match"] else '반대') + '</span>')
            if r.get("proxy_direction_same") is not None:
                interval = r.get("ci")
                if interval and interval[0] <= 0 <= interval[1]:
                    nums.append('<span class="memo">방향 불확실: 시뮬 95% 구간에 0 포함 '
                                '(정식 검증 아님)</span>')
                else:
                    nums.append('<span class="memo">방향 참고: '
                                + ('양쪽 증가' if r["proxy_direction_same"] else '부호 다름')
                                + ' (정식 검증 아님)</span>')
            visible_reason = ("실측과 시뮬의 정의가 달라 정식 정책 효과 오차를 계산하지 않았습니다. "
                              "아래 지표별 설명과 원문 상세를 확인하세요."
                              if numeric_only and r["gap"] is None else r["reason"])
            parts.append('<div class="row"><div class="meta">'
                         f'<span class="id">{_esc(r["id"])}</span>'
                         f'<span class="exp">{_esc(EXPECT.get(r["expect"], r["expect"]))}</span>'
                         f'<span class="tag {cl}">{_esc(r["status"])}</span>'
                         f'<span class="desc">{_esc(r["desc"])}</span></div>'
                         + _track(r) + '<div class="nums">' + ''.join(nums) + '</div>'
                         f'<p class="reason">{_esc(visible_reason)}</p>'
                         f'<p class="opinion"><strong>표본만 확대:</strong> {_esc(r["more_people"])}. '
                         f'{_esc(r["expert_opinion"])}</p>'
                         + (f'<p class="balance">0/{r["binomial_zero_count_audit"]["recipients"]} '
                            '수령자는 상한 도달률이 실제로 0이라는 증거가 아닙니다. '
                            f'위 {r["binomial_zero_count_audit"]["two_sided_95_upper_pct"]:.2f}%는 '
                            f'{r["binomial_zero_count_audit"]["recipients"]}명을 '
                            '독립 이항 표본으로 본 '
                            '참고 상한이며, 모델 반복 변동·외부 표본 불확실성은 '
                            f'포함하지 않습니다. 실측은 10월 원문 표를 재구성한 '
                            f'{_esc(_fmt(r["truth"], r["truth_unit"]))}이고 '
                            '시뮬은 10월 말 발생추정(익월 실제 지급 관측 아님)이므로 '
                            '이 상한에 실측이 들어온다고 해서 정책 효과 검증이 된 것은 '
                            '아닙니다.</p>' if r.get("binomial_zero_count_audit") else '')
                         + ('<p class="balance">이 지표의 시뮬 0%p는 대상 POI 네 업종 합이 '
                            '마트 상위 분류 전체와 같은 원장 구조 때문에 자동으로 나온 값입니다. '
                            '실측은 같은 마트 안의 농축산물 상품 매출을 따로 측정했으므로 '
                            '이 0%p를 정책 무반응이나 실측 효과와의 차이로 읽을 수 없습니다.</p>'
                            if r.get("structural_zero_display_audit") else '')
                         + (f'<p class="reason">탐색적 차이는 표시된 두 숫자의 산술 차이일 뿐 '
                            f'정책 효과 오차나 프롬프트 적중률이 아닙니다. '
                            f'{_esc(r["surface_comparison"].get("conversion") or "")}'
                            f'{" 시뮬 구간이 0을 포함하므로 방향도 불확실합니다." if r["surface_comparison"]["simulation_interval_crosses_zero"] else ""}'
                            f'</p>' if r.get("surface_comparison") else '')
                         + (_p012_rank_components(r) if numeric_only else '')
                         + (f'<p class="reason source">실측 출처: {_esc(r["source"])}</p>'
                            if r["source"] else '')
                         + (f'<p class="reason source">원문 위치: {_esc(r["source_locator"])}</p>'
                            if r.get("source_locator") else '')
                         + (_technical_details(r) if numeric_only else '') + '</div>')
        parts.append('</div></section>')
        sections.append(''.join(parts))
    if numeric_only:
        measured_policies = sum(item["empirical_numeric_count"] > 0
                                for item in report["policy_coverage"])
        stats = [(measured_policies, "실측 숫자 있는 정책"),
                 (report["indicator_count"], "실측·시뮬 숫자 쌍"),
                 (report["direct_gap_count"], "직접 차감 감사 통과"),
                 (report["omitted_without_simulation"], "시뮬 수치 없어 보류")]
        if report.get("exploratory_pairs"):
            stats.append((len(report["exploratory_pairs"]), "채점표 밖 탐색 참고"))
    else:
        stats = [(report["policy_count"], "등록 정책·위약"),
                 (report["indicator_count"], "등록 지표 전체"),
                 (report["simulated_count"], "시뮬 수치 있음"),
                 (sum(r["external_direction_match"] is not None for r in rows), "외부 방향 대조 가능"),
                 (sum(r.get("proxy_direction_same") is not None
                      and not (r.get("ci") and r["ci"][0] <= 0 <= r["ci"][1])
                      for r in rows), "대리 부호 참고 가능"),
                 (report["direct_gap_count"], "실측과 직접 차감 가능"),
                 (sum(r["more_people"] == "아니요" for r in rows), "표본 확대만으로 부족"),
                 (sum(r["more_people"] == "내부 정밀도만" for r in rows), "내부 정밀도만 개선"),
                 (len(report.get("unregistered_policies") or []), "지표 미등록 정책")]
    paired = report.get("paired_effect_summary")
    if paired:
        run_note = report.get("run_note")
        run_note_html = (f'<p><strong>실행 품질 기록:</strong> {_esc(run_note["text"])}</p>'
                         if run_note else '')
        sector = report.get("paired_sector_summary")
        sector_line = (f'<p>업종 대리 지표: 준내구재 {_esc(_fmt(sector["semidurable_relative_change_pct"], "%"))}, '
                       f'대면서비스 {_esc(_fmt(sector["face_service_relative_change_pct"], "%"))}; '
                       f'변화율 차이 {_esc(_fmt(sector["rank_gap_percentage_points"], "%p"))}. '
                       '이는 같은 날짜 POI 매출의 내부 비교이며 KDI 실측 업종 범위와 다릅니다.</p>'
                       if sector else '')
        sections.insert(0, '<section class="note"><h2>이번 실험에서 실제로 확인한 범위</h2>'
                        f'<p>{_esc(paired["citizens"])}명 × {_esc(paired["days"])}일 × ON/OFF 두 팔. '
                        f'정책 후 분석 {_esc(paired["effect_start"])} ~ {_esc(paired["effect_end"])}. '
                        f'지원금 수령 {_esc(paired["grant_recipients"])}명, '
                        f'지급 {_esc(_fmt(paired["grant_issued_won"], "원"))}, '
                        f'사용 {_esc(_fmt(paired["grant_spent_won"], "원"))}, '
                        f'양팔 총지출 차이 {_esc(_fmt(paired["recorded_total_spend_difference_won"], "원"))}, '
                        f'지원금 1원당 기록된 추가 지출 '
                        f'{_esc(_fmt(paired["incremental_recorded_spend_per_grant_won"], "ratio"))}.</p>'
                        + sector_line + run_note_html +
                        '<p>아래 EM-2/EM-3의 파란 수치는 쌍체 시뮬레이션의 대리 변화율입니다. '
                        '실측의 전년동기 카드매출 효과와 분모·기간·모집단이 달라 같은 방향의 참고만 가능하며, '
                        '정식 정책 효과 오차 또는 최적 프롬프트 정확도는 계산하지 않습니다. '
                        '95% 구간은 시민 재표본에 한정되며 모델 생성 결과를 다시 뽑았을 때의 변동은 포함하지 않습니다. '
                        '나머지 정책의 미실행은 표본 부족이 아니라 이번 파일럿에 정책 팔이 없다는 뜻입니다.</p></section>')
    if numeric_only:
        sections.insert(0, _coverage_html(report["policy_coverage"],
                                          len(report.get("unregistered_policies") or [])))
    if numeric_only and report.get("prompt_variant") == "v53":
        prompt_note = (
            '모든 정책 팔의 범용 v53 프롬프트 SHA256 지문이 일치합니다. '
            '정책 내용·시행 배경 입력은 정책별로 별도 기록했습니다.'
            if report.get("shared_prompt_sha256_verified") else
            '각 산출물은 범용 v53 프롬프트 사용을 선언합니다. 모든 정책 팔의 '
            '동일 바이트 지문은 아직 교차 확인되지 않았으므로 동일 프롬프트의 '
            '확정 증거로 취급하지 않습니다.'
        )
        sections.insert(0, '<section class="note"><h2>v53 기준선: 범용 프롬프트 최적화 완료 아님</h2>'
                        f'<p>{_esc(prompt_note)}</p>'
                        '<p>이 보고서는 현재 범용 프롬프트가 정책별로 어떤 방향과 대략적인 '
                        '크기의 숫자를 내는지 확인하는 출발점입니다. 실측과 시뮬의 기간·모집단·'
                        '분모·대조군이 다른 행은 정식 효과 오차로 채점하지 않습니다. '
                        '정책별 약점과 측정 공백을 확인한 뒤, 같은 평가 설계로 후속 프롬프트 '
                        '후보를 비교해야 최적화를 주장할 수 있습니다. 정책별 표본 수가 다르며 '
                        '특히 12명 월간 실험은 방향·크기 모두 매우 불확실합니다.</p></section>')
    if report.get("exploratory_pairs"):
        sections.append(_exploratory_html(report["exploratory_pairs"]))
    tally_html = ''.join(f'<div class="stat"><span class="v">{v}</span><span class="k">{_esc(k)}</span></div>'
                         for v, k in stats)
    unregistered = report.get("unregistered_policies") or []
    if unregistered and not numeric_only:
        missing = ''.join(f'<li><strong>{_esc(p["id"])}</strong> {_esc(p["name"])} — '
                          '검증지표 미등록, 프롬프트 성능 평가 대상에 아직 넣을 수 없음</li>'
                          for p in unregistered)
        sections.append('<section class="note"><h2>정책 파일은 있으나 검증지표가 없는 정책</h2>'
                        '<p>채점표에 지표·창·원문 추정량을 사전등록해야 누락 없이 비교할 수 있습니다.</p>'
                        f'<ul>{missing}</ul></section>')
    source_note = '; '.join(f'{_esc(x["policy"])} {_esc(x["sha256"][:12])}' for x in report["score_files"])
    warnings = [x["policy"] for x in report["score_files"] if not x["scoring_table_matches"]]
    if warnings:
        source_note += ' · 채점표 지문 없음/불일치: ' + ', '.join(map(_esc, warnings))
    page = template_path.read_text(encoding="utf-8")
    if numeric_only:
        if report.get("prompt_variant") == "v53":
            page = page.replace('실험별 정책 검증지표 · <!--EXPERIMENT-->',
                                'v53 기준선 · 정책별 숫자 비교 · <!--EXPERIMENT-->')
        page = page.replace(
            '채점표의 모든 정책·위약과 모든 지표를 빠짐없이 표시합니다. 주황은 실측, 파랑은 이 실험의 시뮬레이션입니다. 각 막대는 해당 지표의 단위와 범위로 그리며, 감사된 동일 추정량일 때만 실측점을 같은 막대에 얹고 차이를 계산합니다.',
            '본문에는 실측 수치와 이번 실험의 시뮬 수치가 모두 있는 지표만 표시합니다. 아래 정책별 표에서 누락 정책과 남은 측정을 확인할 수 있습니다. 단위가 맞는 경우 숫자상 차이를 탐색적으로 표시하지만, 측정 정의가 다르면 정식 정책 효과 오차로 보지 않습니다.')
        page = page.replace(
            '‘미실행’은 이 실험에 해당 정책의 채점 파일이 없다는 뜻입니다. ‘관측부족’은 채점 파일은 있지만 지표값이 없다는 뜻입니다. ‘참고 지표’도 목록에 남깁니다.',
            '본문에 없는 지표는 실측 수치 또는 이번 실험의 시뮬 수치가 없습니다. '
            '실측 숫자가 있는 정책의 미산출 시뮬 지표는 위 표에 사유를 남겼습니다. '
            '실측 숫자가 없는 지표는 HTML에서 제외하고 개수만 설명합니다. '
            '전체 감사 내역은 JSON 원장에서 확인할 수 있습니다.')
        page = page.replace('시뮬 95% 구간</span>',
                            '시뮬 시민 재표집 95% 구간</span>')
    return (page
            .replace('<!--EXPERIMENT-->', _esc(report["experiment"]))
            .replace('<!--TALLY-->', tally_html)
            .replace('<!--BODY-->', '\n'.join(sections))
            .replace('<!--SOURCE-->', source_note)
            .replace('<!--NOTES_SHA-->', _esc(report["comparability_notes"]["sha256"]))
            .replace('<!--SCORING_SHA-->', _esc(report["scoring_table"]["sha256"])))


def _atomic_write(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                     prefix=".comparison-", delete=False) as fh:
        fh.write(value)
        temporary = Path(fh.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(score_paths: list[Path], *, experiment: str = "", out: Path | None = None,
             json_out: Path | None = None, scoring_path: Path = SCORING,
             paired_effect: Path | None = None,
             paired_sector: Path | None = None,
             run_note: Path | None = None,
             numeric_only: bool = False,
             multi_policy_pairs: Path | list[Path] | None = None,
             empirical_registry: Path | None = None,
             p010_funding_audit: Path | None = None,
             p010_channel_audit: Path | None = None,
             p010_concentration_audit: Path | None = None,
             p012_sector_audit: Path | None = None,
             distancing_input_audit: Path | None = None,
             distancing_geo_failure_audit: Path | None = None,
             p016_postfix_audit: Path | None = None,
             p016_taxonomy_audit: Path | None = None,
             in_progress_policies: list[str] | None = None) -> tuple[Path, Path, dict]:
    multi_paths = ([multi_policy_pairs] if isinstance(multi_policy_pairs, Path)
                   else list(multi_policy_pairs or []))
    if sum((bool(score_paths), bool(paired_effect), bool(multi_paths))) != 1:
        raise ValueError("supply score files, one paired effect, or one multi-policy manifest")
    if paired_sector and not paired_effect:
        raise ValueError("--paired-sector requires --paired-effect")
    if run_note and not paired_effect:
        raise ValueError("--run-note requires --paired-effect")
    if p010_funding_audit and not multi_paths:
        raise ValueError("--p010-funding-audit requires --multi-policy-pairs")
    if p010_channel_audit and not multi_paths:
        raise ValueError("--p010-channel-audit requires --multi-policy-pairs")
    if p010_concentration_audit and not multi_paths:
        raise ValueError("--p010-concentration-audit requires --multi-policy-pairs")
    if p012_sector_audit and not multi_paths:
        raise ValueError("--p012-sector-audit requires --multi-policy-pairs")
    if distancing_input_audit and not multi_paths:
        raise ValueError("--distancing-input-audit requires --multi-policy-pairs")
    if distancing_geo_failure_audit and not multi_paths:
        raise ValueError("--distancing-geo-failure-audit requires --multi-policy-pairs")
    if p016_postfix_audit and not multi_paths:
        raise ValueError("--p016-postfix-audit requires --multi-policy-pairs")
    if p016_taxonomy_audit and not multi_paths:
        raise ValueError("--p016-taxonomy-audit requires --multi-policy-pairs")
    if in_progress_policies and not multi_paths:
        raise ValueError("--in-progress-policy requires --multi-policy-pairs")
    if len(score_paths) > 1 and out is None:
        raise ValueError("multiple scores require --out")
    if len(multi_paths) > 1 and out is None:
        raise ValueError("multiple multi-policy manifests require --out")
    source = multi_paths[0] if multi_paths else paired_effect or score_paths[0]
    out = out or source.with_suffix(".comparison.html")
    json_out = json_out or out.with_suffix(".json")
    if out.resolve() == json_out.resolve() or any(out.resolve() == p.resolve() or json_out.resolve() == p.resolve()
                                                 for p in [source, *score_paths, *multi_paths,
                                                           *([paired_sector] if paired_sector else []),
                                                           *([run_note] if run_note else []),
                                                           *([empirical_registry] if empirical_registry else []),
                                                           *([p010_funding_audit] if p010_funding_audit else []),
                                                           *([p010_channel_audit] if p010_channel_audit else []),
                                                           *([p010_concentration_audit] if p010_concentration_audit else []),
                                                           *([p012_sector_audit] if p012_sector_audit else []),
                                                           *([distancing_input_audit] if distancing_input_audit else []),
                                                           *([distancing_geo_failure_audit]
                                                             if distancing_geo_failure_audit else []),
                                                           *([p016_postfix_audit]
                                                             if p016_postfix_audit else []),
                                                           *([p016_taxonomy_audit]
                                                             if p016_taxonomy_audit else [])]):
        raise ValueError("report paths must not overwrite source scores or each other")
    if multi_paths:
        report = build_multi_policy_pairs(multi_paths, scoring_path, suite=experiment)
        if p010_funding_audit:
            report = apply_p010_funding_display_audit(report, p010_funding_audit)
        if p010_channel_audit:
            report = apply_p010_channel_display_audit(report, p010_channel_audit)
        if p010_concentration_audit:
            report = apply_p010_concentration_display_audit(report, p010_concentration_audit)
        if p012_sector_audit:
            report = apply_p012_sector_display_audit(report, p012_sector_audit)
        if distancing_input_audit:
            report = apply_distancing_input_display_audit(report, distancing_input_audit)
        if distancing_geo_failure_audit:
            report = apply_distancing_geo_failure_display_audit(
                report, distancing_geo_failure_audit)
        if p016_postfix_audit:
            report = apply_p016_postfix_display_audit(report, p016_postfix_audit)
        if p016_taxonomy_audit:
            report = apply_p016_taxonomy_display_audit(report, p016_taxonomy_audit)
    elif paired_effect:
        report = build_paired_effect(paired_effect, scoring_path, experiment, paired_sector)
    else:
        report = build(score_paths, scoring_path, experiment)
    if run_note:
        note_text = run_note.read_text(encoding="utf-8").strip()
        if not note_text or len(note_text) > 2000:
            raise ValueError("run note must contain 1-2000 characters")
        report["run_note"] = {"text": note_text, "path": _display_path(run_note),
                              "sha256": _sha(run_note)}
    if multi_paths and empirical_registry is None:
        empirical_registry = EMPIRICAL_REGISTRY
    if empirical_registry:
        report = apply_empirical_registry(report, empirical_registry)
    if multi_paths:
        numeric_only = True
    if numeric_only:
        progress = {POLICY_ID_TO_SCORE.get(policy, policy)
                    for policy in (in_progress_policies or [])}
        unknown = progress - {row["policy"] for row in report["rows"]}
        if unknown:
            raise ValueError(f"unknown in-progress policy: {sorted(unknown)}")
        report = numeric_pair_view(report, in_progress_policies=progress)
    _atomic_write(json_out, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(out, render(report))
    return out, json_out, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", type=Path, action="append", default=[])
    ap.add_argument("--paired-effect", type=Path,
                    help="audited P013 ON/OFF paired effect; proxies stay non-comparable")
    ap.add_argument("--paired-sector", type=Path,
                    help="optional same-calendar P013 sector proxy for EM-4")
    ap.add_argument("--run-note", type=Path,
                    help="frozen execution-quality caveat displayed in the paired report")
    ap.add_argument("--multi-policy-pairs", type=Path, action="append",
                    help="v53 multi-policy numeric manifest with SHA256-verified run evidence")
    ap.add_argument("--empirical-registry", type=Path,
                    help="audited external numeric values; required by default for multi-policy pairs")
    ap.add_argument("--p010-funding-audit", type=Path,
                    help="optional SHA-checked post-run P010 subclass display erratum; never rescores")
    ap.add_argument("--p010-channel-audit", type=Path,
                    help="optional SHA-checked post-run P010 channel decomposition; never rescores")
    ap.add_argument("--p010-concentration-audit", type=Path,
                    help="optional SHA-checked post-run P010 citizen-gap concentration; never rescores")
    ap.add_argument("--p012-sector-audit", type=Path,
                    help="optional SHA-checked post-run P012 sector denominator disclosure; never rescores")
    ap.add_argument("--distancing-input-audit", type=Path,
                    help="optional SHA-checked static ON/OFF distancing-regime render audit")
    ap.add_argument("--distancing-geo-failure-audit", type=Path,
                    help="optional score-bound 2023 geography gate-failure disclosure; never rescores")
    ap.add_argument("--p016-postfix-audit", type=Path,
                    help="optional score-bound P016 patched-arm and excluded-run provenance")
    ap.add_argument("--p016-taxonomy-audit", type=Path,
                    help="optional score-bound P016 C2/C3 structural-zero disclosure")
    ap.add_argument("--in-progress-policy", action="append", default=[],
                    help="explicit policy still running; show progress without empty numeric rows")
    ap.add_argument("--experiment", default="")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--scoring", type=Path, default=SCORING)
    ap.add_argument("--numeric-only", action="store_true",
                    help="show only rows with both empirical and run-specific simulation numbers; keep policy coverage separately")
    a = ap.parse_args()
    try:
        out, json_out, report = generate(a.score, experiment=a.experiment, out=a.out,
                                         json_out=a.json_out, scoring_path=a.scoring,
                                         paired_effect=a.paired_effect,
                                         paired_sector=a.paired_sector,
                                         run_note=a.run_note,
                                         numeric_only=a.numeric_only,
                                         multi_policy_pairs=a.multi_policy_pairs,
                                         empirical_registry=a.empirical_registry,
                                         p010_funding_audit=a.p010_funding_audit,
                                         p010_channel_audit=a.p010_channel_audit,
                                         p010_concentration_audit=a.p010_concentration_audit,
                                         p012_sector_audit=a.p012_sector_audit,
                                         distancing_input_audit=a.distancing_input_audit,
                                         distancing_geo_failure_audit=a.distancing_geo_failure_audit,
                                         p016_postfix_audit=a.p016_postfix_audit,
                                         p016_taxonomy_audit=a.p016_taxonomy_audit,
                                         in_progress_policies=a.in_progress_policy)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        ap.error(str(exc))
    print(f"{out} | {json_out} | indicators={report['indicator_count']} "
          f"simulated={report['simulated_count']} direct_gaps={report['direct_gap_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
