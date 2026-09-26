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
            "score_files": sources, "indicator_count": len(rows), "policy_count": len(policies),
            "simulated_count": sum(_number(r["simulation"]) for r in rows),
            "direct_gap_count": sum(r["gap"] is not None for r in rows),
            "tally": tally, "rows": rows, "unregistered_policies": unregistered}


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


def render(report: dict, template_path: Path = TEMPLATE) -> str:
    sections = []
    rows = report["rows"]
    for policy in dict.fromkeys(r["policy"] for r in rows):
        rs = [r for r in rows if r["policy"] == policy]
        score_line = next((f'{r["score_label"] or "무명"} · OFF {r["off"]} · ON {r["on"]}'
                           for r in rs if r["off"]), "이 실험에서 미실행")
        parts = [f'<section class="pol"><h2>{_esc(rs[0]["policy_name"])} '
                 f'<small>{_esc(policy)}</small></h2><p class="runline">{_esc(score_line)}</p>',
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
            if r["ci"]:
                nums.append(f'<span class="ciTxt">95% 구간 [{_esc(_fmt(r["ci"][0], r["simulation_unit"]))}, '
                            f'{_esc(_fmt(r["ci"][1], r["simulation_unit"]))}]</span>')
            if r["n"] is not None:
                nums.append(f'<span class="n">n={_esc(r["n"])}</span>')
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
            parts.append('<div class="row"><div class="meta">'
                         f'<span class="id">{_esc(r["id"])}</span>'
                         f'<span class="exp">{_esc(EXPECT.get(r["expect"], r["expect"]))}</span>'
                         f'<span class="tag {cl}">{_esc(r["status"])}</span>'
                         f'<span class="desc">{_esc(r["desc"])}</span></div>'
                         + _track(r) + '<div class="nums">' + ''.join(nums) + '</div>'
                         f'<p class="reason">{_esc(r["reason"])}</p>'
                         + (f'<p class="reason source">실측 출처: {_esc(r["source"])}</p>'
                            if r["source"] else '') + '</div>')
        parts.append('</div></section>')
        sections.append(''.join(parts))
    stats = [(report["policy_count"], "등록 정책·위약"),
             (report["indicator_count"], "등록 지표 전체"),
             (report["simulated_count"], "시뮬 수치 있음"),
             (sum(r["external_direction_match"] is not None for r in rows), "외부 방향 대조 가능"),
             (report["direct_gap_count"], "실측과 직접 차감 가능"),
             (len(report.get("unregistered_policies") or []), "지표 미등록 정책")]
    tally_html = ''.join(f'<div class="stat"><span class="v">{v}</span><span class="k">{_esc(k)}</span></div>'
                         for v, k in stats)
    unregistered = report.get("unregistered_policies") or []
    if unregistered:
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
    return (template_path.read_text(encoding="utf-8")
            .replace('<!--EXPERIMENT-->', _esc(report["experiment"]))
            .replace('<!--TALLY-->', tally_html)
            .replace('<!--BODY-->', '\n'.join(sections))
            .replace('<!--SOURCE-->', source_note)
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
             json_out: Path | None = None, scoring_path: Path = SCORING) -> tuple[Path, Path, dict]:
    if len(score_paths) > 1 and out is None:
        raise ValueError("multiple scores require --out")
    out = out or score_paths[0].with_suffix(".comparison.html")
    json_out = json_out or out.with_suffix(".json")
    if out.resolve() == json_out.resolve() or any(out.resolve() == p.resolve() or json_out.resolve() == p.resolve()
                                                 for p in score_paths):
        raise ValueError("report paths must not overwrite source scores or each other")
    report = build(score_paths, scoring_path, experiment)
    _atomic_write(json_out, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(out, render(report))
    return out, json_out, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", type=Path, action="append", required=True)
    ap.add_argument("--experiment", default="")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--scoring", type=Path, default=SCORING)
    a = ap.parse_args()
    try:
        out, json_out, report = generate(a.score, experiment=a.experiment, out=a.out,
                                         json_out=a.json_out, scoring_path=a.scoring)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        ap.error(str(exc))
    print(f"{out} · {json_out} — {report['indicator_count']}개 지표, "
          f"시뮬 {report['simulated_count']}개, 직접 차감 {report['direct_gap_count']}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
