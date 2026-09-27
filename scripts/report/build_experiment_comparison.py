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


def numeric_pair_view(report: dict) -> dict:
    """Show measured-versus-simulated pairs without hiding policy coverage.

    This is a presentation filter.  A pair is *not* a validated effect-size
    comparison unless the original row's run-specific audit emitted a gap.
    In particular, a historical score and a P013 proxy cannot be promoted by
    merely putting their numbers next to an empirical number.
    """
    original = report["rows"]
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
            "unmeasured_count": len(rows) - len(measured),
            "missing_simulation_ids": [row["id"] for row in missing_sim],
            "missing_simulation_reasons": [
                {"id": row["id"], "status": row["status"],
                 "reason": (row["reason"] if row["status"] == "정책 미실행" else
                            "이번 원장에는 이 지표의 완결된 시뮬 수치가 없습니다. "
                            + row["expert_opinion"]),
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
    if row.get("gap") is not None or row.get("id") == "P012-5":
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
                         empirical_comparison_reason=benchmark.get("reason"),
                         gap=None, gap_unit=None)
        exploratory_pairs.append(candidate)
    updated["exploratory_pairs"] = exploratory_pairs
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


def _coverage_html(coverage: list[dict]) -> str:
    body = []
    included = [item for item in coverage if item["empirical_numeric_count"]]
    excluded = [item for item in coverage if not item["empirical_numeric_count"]]
    for item in included:
        missing = item["missing_simulation_reasons"]
        if not item["empirical_numeric_count"]:
            action = "등록 검증지표의 실측 수치가 없어 제외. 대응 실측값 확보 필요"
            if item["exploratory_numeric_count"]:
                action += (f' · 채점표 밖 탐색 참고값 {item["exploratory_numeric_count"]}건은 '
                           '아래 별도 표시')
        elif missing:
            action = "실측 수치가 있으나 이번 실험의 시뮬 수치 없음"
        else:
            action = "이번 실험에서 실측·시뮬 수치가 모두 있는 지표 표시됨"
        details = ("<details><summary>미측정 지표와 사유</summary><ul>"
                   + ''.join(f'<li><strong>{_esc(entry["id"])}</strong>: '
                             f'{_esc(entry["reason"])}</li>' for entry in missing)
                   + '</ul></details>') if missing else ''
        body.append(
            '<tr>'
            f'<th scope="row">{_esc(item["policy_name"])}</th>'
            f'<td>{_esc(str(item["sample_citizens"]) + "명" if item["sample_citizens"] else "미실행")}</td>'
            f'<td>{item["empirical_numeric_count"]}</td>'
            f'<td>{item["simulation_numeric_count"]}</td>'
            f'<td>{item["paired_numeric_count"]}</td>'
            f'<td>{_esc(action)}{details}</td>'
            '</tr>'
        )
    excluded_html = (
        '<details class="technical"><summary>실측 수치가 없어 평가에서 제외한 정책·위약 '
        f'{len(excluded)}개 보기</summary><ul>'
        + ''.join(f'<li><strong>{_esc(item["policy_name"])} '
                  f'({_esc(item["policy"])})</strong>: 등록 검증지표에 실측 숫자가 없어 '
                  '이번 실측·시뮬 대조에서 제외했습니다.'
                  + (f' 채점표 밖 탐색 참고값 {item["exploratory_numeric_count"]}건은 '
                     '아래 별도 표에 표시합니다.' if item["exploratory_numeric_count"] else '')
                  + '</li>' for item in excluded)
        + '</ul></details>' if excluded else '')
    return ('<section class="note"><h2>실측 수치가 있는 정책의 숫자 확보 현황</h2>'
            '<p>비교 행에는 실측과 이번 실험의 시뮬 수치가 모두 있는 지표만 표시합니다. '
            '실측 수치가 없는 지표는 검증 대상에서 제외했습니다. 수치 쌍이 '
            '측정 정의까지 일치한다는 뜻은 아닙니다.</p>'
            '<div class="coverage-scroll"><table class="coverage"><thead><tr>'
            '<th>정책</th><th>시뮬 시민</th><th>실측 수치</th><th>시뮬 수치</th><th>나란히 표시</th><th>남은 작업</th>'
            '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'
            + excluded_html + '</section>')


def _exploratory_html(pairs: list[dict]) -> str:
    cards = []
    for row in pairs:
        truth_unit = row.get("truth_unit")
        sim_unit = row.get("simulation_unit")
        truth = (_fmt(row["truth"], truth_unit)
                 if truth_unit in ("%", "%p", "ratio", "원", "log-point")
                 else f'{row["truth"]:+.4f} {truth_unit or ""}')
        sim = (_fmt(row["simulation"], sim_unit)
               if sim_unit in ("%", "%p", "ratio", "원", "log-point")
               else f'{row["simulation"]:+.4f} {sim_unit or ""}')
        scope_note = ("실측은 쿠폰 사용처의 카드결제 분포, 시뮬은 지급액으로 결제된 POI 분포입니다. "
                      "업종 대응과 모집단을 감사하기 전에는 정식 효과 점수로 쓰지 않습니다."
                      if row["policy"] == "P010" else
                      "실측은 지역화폐 발행 강도에 대한 지역·연도별 업종 매출 회귀계수, "
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
                    f'정책결제 총액 {_esc(f"{funded_won:,}원")}. '
                    f'전체 원장은 {_esc(full_run)} 시민×일(시행 전 포함)입니다. '
                    f'시민 {_esc(row.get("n") or "미확인")}명이라는 부트스트랩 표본 수는 '
                    '실제 결제 관측 수가 아닙니다. '
                    '이 작은 분모의 업종비중으로 크기 적중을 판단할 수 없습니다. '
                    '개별 거래 건수는 시민×일 집계 원장에서 알 수 없습니다. '
                    '정책결제가 적게 기록된 원인은 확정되지 않았습니다. '
                    'Stage2 결제 선택·요청 과정의 병목 가능성은 진단상의 추정입니다.</p>'
                )
            else:
                denominator_html = (
                    '<p class="balance">정책결제 분모가 검증되지 않아 업종비중의 '
                    '크기를 판단할 수 없습니다. 시민 부트스트랩 표본 수는 결제 건수가 아닙니다.</p>'
                )
        cards.append('<div class="row">'
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
    return ('<section class="pol"><h2>등록 38개 지표 밖의 탐색 참고값</h2>'
            '<p class="runline">실측과 시뮬 수치는 있지만 정책 효과 채점표의 검증지표가 아닙니다. '
            '표본·기간·추정량이 달라 외부 효과 오차 또는 적중률에 포함하지 않습니다.</p>'
            '<div class="rows">' + ''.join(cards) + '</div></section>')


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
    return ('<div class="component-breakdown"><strong>두 업종의 원수치</strong>'
            '<table><thead><tr><th>업종</th><th>실측 로그회귀 계수</th>'
            '<th>시뮬 ON−OFF 변화율</th></tr></thead><tbody>' + cells + '</tbody></table>'
            '<p>실측 0.3336 log-point는 두 회귀계수의 차이이고, '
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
    return ('<p class="contextline">' + ' · '.join(pieces) + '</p>'
            + quality_html + balance_html
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
            elif numeric_only and r.get("surface_comparison"):
                surface = r["surface_comparison"]
                nums.append(f'<span class="memo">탐색적 숫자상 차이 '
                            f'{_esc(_fmt(surface["delta"], surface["delta_unit"]))}'
                            f' · {_esc(surface["scale"])}</span>')
            elif numeric_only:
                nums.append('<span class="memo">서로 다른 단위·정의: 산술 차이 생략</span>')
            if r["ci"]:
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
            if numeric_only and r.get("ci") and r["ci"][0] <= 0 <= r["ci"][1]:
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
        sections.insert(0, _coverage_html(report["policy_coverage"]))
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
    if unregistered:
        missing = ''.join(f'<li><strong>{_esc(p["id"])}</strong> {_esc(p["name"])} — '
                          '검증지표 미등록, 프롬프트 성능 평가 대상에 아직 넣을 수 없음</li>'
                          for p in unregistered)
        if numeric_only:
            sections.append('<details class="technical"><summary>등록 검증지표가 없는 정책 파일 '
                            f'{len(unregistered)}개 보기</summary><p>지표·기간·원문 추정량을 '
                            '등록한 뒤에야 평가할 수 있습니다.</p>'
                            f'<ul>{missing}</ul></details>')
        else:
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
            '본문에 없는 지표는 실측 수치 또는 이번 실험의 시뮬 수치가 없습니다. 어떤 정책과 지표의 측정이 부족한지는 정책별 숫자 확보 현황과 JSON 원장에서 확인할 수 있습니다.')
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
             empirical_registry: Path | None = None) -> tuple[Path, Path, dict]:
    multi_paths = ([multi_policy_pairs] if isinstance(multi_policy_pairs, Path)
                   else list(multi_policy_pairs or []))
    if sum((bool(score_paths), bool(paired_effect), bool(multi_paths))) != 1:
        raise ValueError("supply score files, one paired effect, or one multi-policy manifest")
    if paired_sector and not paired_effect:
        raise ValueError("--paired-sector requires --paired-effect")
    if run_note and not paired_effect:
        raise ValueError("--run-note requires --paired-effect")
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
                                                           *([empirical_registry] if empirical_registry else [])]):
        raise ValueError("report paths must not overwrite source scores or each other")
    if multi_paths:
        report = build_multi_policy_pairs(multi_paths, scoring_path, suite=experiment)
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
        report = numeric_pair_view(report)
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
                                         empirical_registry=a.empirical_registry)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        ap.error(str(exc))
    print(f"{out} | {json_out} | indicators={report['indicator_count']} "
          f"simulated={report['simulated_count']} direct_gaps={report['direct_gap_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
