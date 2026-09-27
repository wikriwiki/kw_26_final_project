"""Score numerical *simulation proxies* from frozen paired sector ledgers.

The empirical reference values are deliberately never read by this module.
These are same-calendar ON/OFF transaction summaries, not published causal
estimands. A later report joins them to the evaluation-only empirical registry.
Only complete, SHA-verified citizen-day matrices are accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

if __package__:
    from .audit_stage2_generation import inspect as inspect_stage2
else:
    from audit_stage2_generation import inspect as inspect_stage2


ROOT = Path(__file__).resolve().parents[2]
SCORING_TABLE = ROOT / "data/experiments/scoring_table.json"
TARGET_P016 = ("청과", "정육", "슈퍼마켓", "식료품")
APPLIANCE_P012 = ("가구", "가전·통신", "통신")
P010_BOK_GROUPS = {
    "P010-BOK-RESTAURANT": ("구내식당·뷔페", "기타식사", "기타외국", "분식",
                             "아시안", "양식", "일식", "중식", "치킨", "피자", "한식"),
    "P010-BOK-MART_FOOD": ("수산", "슈퍼마켓", "식료품", "음료소매", "정육",
                            "종합소매", "청과"),
    "P010-BOK-MEDICAL": ("기타보건", "병원", "의원", "치과", "한의원"),
    "P010-BOK-BEAUTY": ("네일", "미용실", "피부관리"),
    "P010-BOK-ACADEMY": ("학원",),
    "P010-BOK-PHARMACY": ("약국",),
}
DS6_GEO_PROXY_ID = "DS6-2023-GEO-PROXY"
DS6_OFFICIAL_BOUNDARY_SHA256 = "38bb8fab4e45a1171af4989cd7fa1275f68e5d644aa770f5431ce7ccc38384dd"
DS6_GEO_TYPES = {"tourism_special_zone": "관광특구",
                 "developed_commercial_district": "발달상권"}
# Fixed before the DISTANCING pair is scored. This governs interpretation,
# separately from the coordinate/denominator conditions needed for a number.
DS6_MIN_RECEIPTS_PER_CELL = 20
DS6_MIN_CITIZENS_PER_CELL = 10
POLICY_TO_SCORE_KEY = {"P010": "P010", "P012": "P012", "P013": "EMERGENCY_2020",
                       "DISTANCING_2020": "DISTANCING_2020", "P016": "P016",
                       "P014": "LOCAL_VOUCHER"}
POLICY_LABEL = {"P010": "민생회복소비쿠폰", "P012": "상생소비지원금",
                "P013": "긴급재난지원금", "DISTANCING_2020": "사회적 거리두기",
                "P016": "농축산물 할인쿠폰", "P014": "지역사랑상품권 탐색 지표"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def evidence(*paths: Path) -> list[dict]:
    return [{"path": display_path(p), "sha256": sha256(p)} for p in paths]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _money(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _bucket(row: dict, name: str) -> dict[str, int]:
    value = row.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing {name} bucket")
    return {str(key): _money(amount, f"{name}[{key}]") for key, amount in value.items()}


def _days(start: str, end: str) -> list[str]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError("invalid date interval")
    return [(first + timedelta(days=i)).isoformat()
            for i in range((last - first).days + 1)]


def _manifest_path(ledger: Path) -> Path:
    return ledger.with_name(ledger.name + ".manifest.json")


def read_pair(on_file: Path, off_file: Path, *, policy: str,
              effect_start: str | None = None, effect_end: str | None = None
              ) -> tuple[list[dict], list[dict], list[str], list[str], list[Path]]:
    on_m_file, off_m_file = _manifest_path(on_file), _manifest_path(off_file)
    on_m, off_m = read_json(on_m_file), read_json(off_m_file)
    for arm, path, manifest in (("on", on_file, on_m), ("off", off_file, off_m)):
        if manifest.get("schema") != "multi_policy_sector_ledger_v1" or manifest.get("arm") != arm:
            raise ValueError(f"wrong {arm} sector manifest")
        if manifest.get("output_sha256") != sha256(path):
            raise ValueError(f"{arm} sector ledger SHA256 mismatch")
    if any(on_m.get(key) != off_m.get(key) for key in
           ("start", "end", "citizens", "days", "rows", "roster_sha256")):
        raise ValueError("ON/OFF manifests describe different cohorts or calendars")
    prompt_on, prompt_off = on_m.get("prompt_provenance"), off_m.get("prompt_provenance")
    if not isinstance(prompt_on, dict) or not isinstance(prompt_off, dict):
        raise ValueError("sector manifests lack verified prompt provenance")
    if prompt_on.get("prompt_variant") != "v53" or prompt_off.get("prompt_variant") != "v53":
        raise ValueError("sector ledgers are not both v53")
    for key in ("system_prompt_sha256", "stage2_system_prompt_sha256",
                "baseline_income_map_sha256"):
        value = prompt_on.get(key)
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                or prompt_off.get(key) != value):
            raise ValueError(f"paired v53 provenance mismatch: {key}")
    run_ids = [prompt_on.get("run_id"), prompt_off.get("run_id")]
    if (any(not isinstance(run_id, str) or not run_id for run_id in run_ids)
            or run_ids[0] == run_ids[1]):
        raise ValueError("paired arms need distinct nonempty run IDs")
    provenance_keys = ("execution_fingerprint", "paired_environment_fingerprint",
                       "source_fingerprint", "experience_environment_id",
                       "experience_run_id")
    values = {}
    for arm, manifest, prompt in (("on", on_m, prompt_on), ("off", off_m, prompt_off)):
        provenance = manifest.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"missing {arm} metric provenance")
        for key in provenance_keys:
            found = provenance.get(key)
            if (not isinstance(found, list) or len(found) != 1
                    or (key != "experience_environment_id"
                        and (not isinstance(found[0], str) or not found[0]))
                    or (key == "experience_environment_id" and found[0] is not None
                        and (not isinstance(found[0], str) or not found[0]))):
                raise ValueError(f"invalid {arm} metric provenance: {key}")
            values[arm, key] = found[0]
        if (values[arm, "execution_fingerprint"] != prompt.get("execution_fingerprint")
                or values[arm, "experience_run_id"] != prompt.get("run_id")):
            raise ValueError(f"{arm} metric provenance differs from cohort")
    for key in ("paired_environment_fingerprint",):
        if values["on", key] != values["off", key]:
            raise ValueError(f"paired runs differ in {key}")
    # This is the *requested* model ID echoed by the API. It checks pair
    # consistency only; the serving process/weights need independent evidence.
    requested_models = [p.get("requested_model_id") for p in (prompt_on, prompt_off)]
    if (any(not isinstance(model, str) or not model for model in requested_models)
            or requested_models[0] != requested_models[1]):
        raise ValueError("paired runs have inconsistent requested model IDs")
    served_models = [m.get("served_model_provenance") for m in (on_m, off_m)]
    expected_served_model = "LGAI-EXAONE/EXAONE-4.5-33B-AWQ"
    for arm, served in zip(("on", "off"), served_models):
        digest = served.get("evidence_sha256") if isinstance(served, dict) else None
        if (not isinstance(served, dict) or served.get("model_id") != expected_served_model
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError(f"{arm} served-model evidence is missing or invalid")
    model_evidence_files = [on_file.parent / "served_model_evidence.json",
                            off_file.parent / "served_model_evidence.json"]
    for arm, path, served in zip(("on", "off"), model_evidence_files, served_models):
        if not path.is_file() or sha256(path) != served["evidence_sha256"]:
            raise ValueError(f"{arm} served-model evidence SHA256 mismatch")
    for manifest in (on_m, off_m):
        if not isinstance(manifest.get("cohort_sha256"), dict):
            raise ValueError("missing per-day cohort SHA256 manifest")
    if on_m["policy_id"] not in (policy, None) or off_m["policy_id"] is not None:
        raise ValueError("policy ID mismatch or control policy leak")
    if policy != "DISTANCING_2020" and on_m["policy_id"] != policy:
        raise ValueError("treatment ledger lacks requested policy ID")
    policy_input = on_m.get("policy_input")
    if policy != "DISTANCING_2020":
        if (not isinstance(policy_input, dict)
                or not isinstance(policy_input.get("path"), str)
                or not policy_input["path"]
                or not isinstance(policy_input.get("sha256"), str)
                or len(policy_input["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in policy_input["sha256"])):
            raise ValueError("missing verified frozen policy input")
    elif policy_input is not None:
        raise ValueError("environment-only arm unexpectedly has policy input")
    if off_m.get("policy_input") is not None:
        raise ValueError("control arm unexpectedly has policy input")
    days = _days(on_m["start"], on_m["end"])
    if len(days) != on_m["days"]:
        raise ValueError("manifest day count mismatch")
    for manifest in (on_m, off_m):
        cohorts = manifest["cohort_sha256"]
        if (set(cohorts) != set(days)
                or any(not isinstance(digest, str) or len(digest) != 64
                       or any(char not in "0123456789abcdef" for char in digest)
                       for digest in cohorts.values())):
            raise ValueError("incomplete or invalid per-day cohort hashes")
    start = effect_start or on_m.get("effective_from")
    end = effect_end or min(on_m.get("effective_until") or on_m["end"], on_m["end"])
    if not start or not end:
        raise ValueError("effect start/end required when manifest lacks policy period")
    effect_days = _days(start, end)
    if not set(effect_days).issubset(days):
        raise ValueError("effect window outside ledger dates")
    on, off = read_jsonl(on_file), read_jsonl(off_file)
    expected_count = on_m["citizens"] * len(days)
    if len(on) != expected_count or len(off) != expected_count or on_m["rows"] != expected_count:
        raise ValueError("incomplete citizen-day matrix")
    indexes = []
    for arm, rows in (("on", on), ("off", off)):
        indexed: dict[tuple[str, str], dict] = {}
        for row in rows:
            aid, day = row.get("aid"), row.get("day")
            if not isinstance(aid, str) or not aid or day not in days:
                raise ValueError("invalid citizen or date")
            key = (aid, day)
            if key in indexed or row.get("arm") != arm:
                raise ValueError(f"duplicate or wrong-arm row: {key}")
            if row.get("policy_id") != (on_m["policy_id"] if arm == "on" else None):
                raise ValueError(f"row policy ID mismatch: {key}")
            for field in ("offline_spent", "online_spent", "total_spent",
                          "sangsaeng_eligible_offline_spent", "unclassified_won", "policy_funded_won",
                          "grant_received_cumulative", "grant_remaining"):
                _money(row.get(field), field)
            sub, l1, funded = (_bucket(row, name) for name in
                               ("by_sub", "by_l1", "funded_by_sub"))
            if (row["total_spent"] != row["offline_spent"] + row["online_spent"]
                    or sum(sub.values()) + row["unclassified_won"] != row["offline_spent"]
                    or row["sangsaeng_eligible_offline_spent"] > row["offline_spent"]
                    or row["policy_funded_won"] > row["offline_spent"]
                    or sum(funded.values()) > row["policy_funded_won"]
                    or any(amount > sub.get(name, 0) for name, amount in funded.items())):
                raise ValueError(f"unreconciled money or category spending: {key}")
            if arm == "off" and any(row[field] for field in
                                    ("policy_funded_won", "grant_received_cumulative",
                                     "grant_remaining")):
                raise ValueError("policy funding in OFF arm")
            indexed[key] = row
        indexes.append(indexed)
    on_by_key, off_by_key = indexes
    if on_by_key.keys() != off_by_key.keys():
        raise ValueError("ON/OFF citizen-day keys differ")
    roster = sorted({aid for aid, _ in on_by_key})
    if len(roster) != on_m["citizens"] or len(on_by_key) != len(roster) * len(days):
        raise ValueError("citizen-day grid incomplete")
    roster_digest = hashlib.sha256(json.dumps(roster, ensure_ascii=False).encode()).hexdigest()
    if roster_digest != on_m["roster_sha256"]:
        raise ValueError("roster fingerprint mismatch")
    for aid in roster:
        prior_received = 0
        total_funded = 0
        for day in days:
            row = on_by_key[(aid, day)]
            received = row["grant_received_cumulative"]
            total_funded += row["policy_funded_won"]
            if received < prior_received:
                raise ValueError(f"decreasing grant: {aid} {day}")
            # The funding dictionary also covers non-wallet policies, for
            # which both grant fields remain zero. Reconcile only real grants.
            if policy == "P010" and (received < total_funded or
                                      received - total_funded != row["grant_remaining"]):
                raise ValueError(f"grant wallet does not reconcile: {aid} {day}")
            prior_received = received
    # Only policy-active dates enter the numerical proxy. Preperiod values remain
    # available in the source ledger for the independent balance check.
    on_rows = [on_by_key[(aid, day)] for aid in roster for day in effect_days]
    off_rows = [off_by_key[(aid, day)] for aid in roster for day in effect_days]
    return on_rows, off_rows, roster, effect_days, [on_file, on_m_file, off_file, off_m_file,
                                                    *model_evidence_files]


def audit_arm_quality(ledger_file: Path, manifest: dict) -> tuple[dict, list[Path]]:
    """Recompute technical quality from the SHA-frozen daily metric records."""
    days = _days(manifest["start"], manifest["end"])
    metrics_hashes = manifest.get("metrics_sha256")
    if not isinstance(metrics_hashes, dict) or set(metrics_hashes) != set(days):
        raise ValueError("sector manifest lacks a complete metrics SHA ledger")
    daily: dict[str, list[dict]] = {}
    files: list[Path] = []
    for day in days:
        path = ledger_file.parent / "metrics" / f"day_{day}.jsonl"
        if not path.is_file() or sha256(path) != metrics_hashes[day]:
            raise ValueError(f"metrics SHA256 mismatch for {day}")
        daily[day] = read_jsonl(path)
        files.append(path)
    audited = inspect_stage2(daily, expected_per_day=manifest["citizens"])
    stage2_path = ledger_file.parent / "stage2.json"
    if not stage2_path.is_file():
        raise ValueError("missing saved Stage2 quality audit")
    saved = read_json(stage2_path)
    for key in ("days", "expected_per_day", "quality_gate_pass",
                "unrepaired_choice_trace_pass", "totals"):
        if saved.get(key) != audited.get(key):
            raise ValueError(f"Stage2 quality audit differs from raw metrics: {key}")
    files.append(stage2_path)
    rows = [row for day in days for row in daily[day]]
    successful_invocation_first_ok = 0
    successful_invocation_errors: Counter[str] = Counter()
    for row in rows:
        attempts = (row.get("s1_timing") or {}).get("attempts")
        if (row.get("status") != "ok" or not isinstance(attempts, list) or not attempts
                or row.get("s1_attempts") != len(attempts)
                or attempts[-1].get("status") != "ok"):
            raise ValueError("missing or inconsistent Stage1 attempt trace")
        first = attempts[0]
        if first.get("status") == "ok":
            successful_invocation_first_ok += 1
        elif first.get("status") == "error":
            successful_invocation_errors[str(first.get("error_stage") or "unknown")] += 1
        else:
            raise ValueError("invalid first Stage1 attempt status")
    total = len(rows)
    # The canonical metrics file retains only the successful outer invocation.
    # A failed outer invocation can have earlier first responses. Its append-only
    # attempt snapshot is the only defensible source for a first-ever rate.
    first_recorded_ok = 0
    first_recorded_unknown = 0
    first_recorded_errors: Counter[str] = Counter()
    outer_retry_recovered = 0
    raw_attempt_rows = 0
    snapshot_files: list[Path] = []
    for day in days:
        snapshots = sorted((ledger_file.parent / "metrics" / "attempts").glob(
            f"day_{day}_*.jsonl"))
        if not snapshots:
            first_recorded_unknown += len(daily[day])
            continue
        snapshot = snapshots[-1]
        history = read_jsonl(snapshot)
        snapshot_files.append(snapshot)
        raw_attempt_rows += len(history)
        first_by_aid = {}
        for row in history:
            if isinstance(row.get("aid"), str):
                first_by_aid.setdefault(row["aid"], row)
        for committed in daily[day]:
            first = first_by_aid.get(committed["aid"])
            if first is None:
                first_recorded_unknown += 1
            elif first.get("status") == "error":
                outer_retry_recovered += 1
                if "stage1 failed" in str(first.get("error") or "").lower():
                    first_recorded_errors["outer_invocation_stage1_failure"] += 1
                else:
                    first_recorded_unknown += 1
            elif first.get("status") == "ok":
                attempts = (first.get("s1_timing") or {}).get("attempts") or []
                if attempts and attempts[0].get("status") == "ok":
                    first_recorded_ok += 1
                elif attempts and attempts[0].get("status") == "error":
                    first_recorded_errors[str(attempts[0].get("error_stage")
                                              or "unknown")] += 1
                else:
                    first_recorded_unknown += 1
            else:
                first_recorded_unknown += 1
    files.extend(snapshot_files)
    first_recorded_rate = (first_recorded_ok / total
                           if first_recorded_unknown == 0 else None)
    counts = audited["totals"]
    return {
        "audit_status": "verified",
        "citizen_days": total,
        "stage1_final_ok_count": total,
        "stage1_first_attempt_internal_validation_pass_count": first_recorded_ok,
        "stage1_first_attempt_internal_validation_pass_rate": first_recorded_rate,
        "stage1_first_attempt_internal_validation_rate_bounds": [
            first_recorded_ok / total,
            (first_recorded_ok + first_recorded_unknown) / total],
        "stage1_first_attempt_unknown_count": first_recorded_unknown,
        "stage1_retry_recovered_count": (total - first_recorded_ok
                                         if first_recorded_unknown == 0 else None),
        "stage1_outer_retry_recovered_count": outer_retry_recovered,
        "stage1_first_attempt_error_stages": dict(sorted(first_recorded_errors.items())),
        "stage1_successful_invocation_first_attempt_pass_count": successful_invocation_first_ok,
        "stage1_successful_invocation_first_attempt_pass_rate": successful_invocation_first_ok / total,
        "stage1_successful_invocation_first_attempt_error_stages": dict(sorted(successful_invocation_errors.items())),
        "stage1_outer_raw_attempt_rows": raw_attempt_rows,
        "stage1_outer_snapshot_files": len(snapshot_files),
        "stage1_first_raw_strict_format_pass_rate": None,
        "stage1_first_raw_strict_format_note": (
            "The first recorded outer invocation includes previous failures if attempt "
            "snapshots are present. Attempt status is normalized/validated, not an "
            "independent strict check of the unmodified first model response."),
        "stage2_quality_gate_pass": audited["quality_gate_pass"],
        "stage2_unrepaired_choice_trace_pass": audited["unrepaired_choice_trace_pass"],
        "stage2_fallback_only_count": counts["stage2_fallback_only_agents"],
        "stage2_fallback_only_rate": counts["stage2_fallback_only_agents"] / total,
        "stage2_choice_repair_count": counts["stage2_choice_repair_agents"],
        "stage2_choice_repair_rate": counts["stage2_choice_repair_agents"] / total,
        "stage2_output_limited_attempts": counts["stage2_output_limited_attempts"],
    }, files


def audit_preperiod_balance(on_file: Path, off_file: Path, *, policy: str,
                            on_manifest: dict) -> dict:
    """Prospective screening of same-calendar pre-policy arm drift, never correction."""
    if policy not in ("P010", "P016", "P014"):
        return {"status": "not_available", "preperiod_days": [],
                "reason": "Frozen paired run has no same-calendar pre-policy days.",
                "post_effect_causal_interpretation_blocked": True}
    start = on_manifest.get("effective_from")
    if not start:
        raise ValueError("policy run lacks a frozen effective date")
    pre_days = [day for day in _days(on_manifest["start"], on_manifest["end"])
                if day < start]
    if len(pre_days) != 2:
        return {"status": "unexpected_preperiod_window", "preperiod_days": pre_days,
                "reason": "Pre-registered balance screen requires exactly two pre-policy days.",
                "post_effect_causal_interpretation_blocked": True}
    on_rows = [row for row in read_jsonl(on_file) if row["day"] in pre_days]
    off_rows = [row for row in read_jsonl(off_file) if row["day"] in pre_days]
    if (len(on_rows) != len(off_rows)
            or len(on_rows) != on_manifest["citizens"] * 2):
        raise ValueError("preperiod ledger citizen-day matrix incomplete")
    measures: dict[str, tuple[str, tuple[str, ...], float]] = {
        "recorded_total_spend": ("total_spent", (), 10.0),
    }
    if policy == "P010":
        measures["offline_spend_proxy"] = ("offline_spent", (), 20.0)
    elif policy == "P016":
        measures["target_poi_spend_proxy"] = ("by_sub", TARGET_P016, 20.0)
    elif policy == "P014":
        measures["supermarket_poi_spend_proxy"] = ("by_sub", ("슈퍼마켓",), 20.0)
        measures["food_store_poi_spend_proxy"] = ("by_sub", ("식료품",), 20.0)
    comparisons = {}
    for name, (field, categories, threshold_pct) in measures.items():
        on_won = sum(_measure(row, field, categories) for row in on_rows)
        off_won = sum(_measure(row, field, categories) for row in off_rows)
        difference = on_won - off_won
        difference_pct = 100 * difference / off_won if off_won > 0 else None
        comparisons[name] = {"on_won": on_won, "off_won": off_won,
                             "difference_won": difference,
                             "difference_pct_of_off": difference_pct,
                             "absolute_threshold_pct": threshold_pct,
                             "pass": (difference_pct is not None
                                      and abs(difference_pct) <= threshold_pct),
                             "method": (field if not categories else
                                        f"{field}: {', '.join(categories)}")}
    passed = all(item["pass"] for item in comparisons.values())
    return {"status": "pass" if passed else "fail",
            "preperiod_days": pre_days, "citizens": on_manifest["citizens"],
            "comparisons": comparisons,
            "thresholds_frozen_before_new_result_review": True,
            "post_effect_causal_interpretation_blocked": not passed,
            "reason": ("This only screens pre-policy drift; passing it is not proof of "
                       "causal identification or external-effect accuracy. P010 offline "
                       "spending includes merchants outside exact coupon eligibility.")}


def _measure(row: dict, field: str, names: tuple[str, ...] = ()) -> int:
    if field in ("by_sub", "by_l1", "funded_by_sub"):
        return sum(row[field].get(name, 0) for name in names)
    return row[field]


def _per_citizen(rows: list[dict], roster: list[str], field: str,
                 names: tuple[str, ...] = ()) -> dict[str, int]:
    totals = dict.fromkeys(roster, 0)
    for row in rows:
        totals[row["aid"]] += _measure(row, field, names)
    return totals


def _ratio_percent(on: dict[str, int], off: dict[str, int], sample: list[str]) -> float | None:
    a, b = sum(on[aid] for aid in sample), sum(off[aid] for aid in sample)
    return 100.0 * (a - b) / b if b > 0 else None


def _ratio_grant(on: dict[str, int], off: dict[str, int], funded: dict[str, int],
                 sample: list[str]) -> float | None:
    denominator = sum(funded[aid] for aid in sample)
    return sum(on[aid] - off[aid] for aid in sample) / denominator if denominator > 0 else None


def _share_change(on_num: dict[str, int], off_num: dict[str, int],
                  on_den: dict[str, int], off_den: dict[str, int], sample: list[str]) -> float | None:
    na, nb = sum(on_num[aid] for aid in sample), sum(off_num[aid] for aid in sample)
    da, db = sum(on_den[aid] for aid in sample), sum(off_den[aid] for aid in sample)
    return 100 * (na / da - nb / db) if da > 0 and db > 0 else None


def _share(on_num: dict[str, int], on_den: dict[str, int],
           sample: list[str]) -> float | None:
    numerator, denominator = (sum(on_num[aid] for aid in sample),
                              sum(on_den[aid] for aid in sample))
    return 100 * numerator / denominator if denominator > 0 else None


def _quantiles(values: list[float]) -> list[float] | None:
    if not values:
        return None
    values.sort()
    return [values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))]]


def _estimate(fn, roster: list[str], *, draws: int, seed: int) -> tuple[float | None, list[float] | None]:
    point = fn(roster)
    if point is None or not math.isfinite(point):
        return None, None
    rng = random.Random(seed)
    boot = []
    for _ in range(draws):
        value = fn(rng.choices(roster, k=len(roster)))
        if value is not None and math.isfinite(value):
            boot.append(value)
    return point, _quantiles(boot) if len(boot) >= max(20, draws // 10) else None


def _item(id_: str, value: float | None, unit: str, ci: list[float] | None,
          n: int, method: str, reason: str, **extra) -> dict:
    return {"id": id_, "simulation": value, "simulation_unit": unit,
            "ci": ci, "n": n if value is not None else None,
            "estimand_alignment": "different", "direction_comparable": False,
            "reason": reason, "method": method, **extra}


def score_ledger(on_rows: list[dict], off_rows: list[dict], roster: list[str],
                 *, policy: str, draws: int = 1000, seed: int = 20260927) -> list[dict]:
    if not roster or draws < 0:
        raise ValueError("nonempty roster and nonnegative bootstrap count required")
    cache = {}
    def per(arm: str, field: str, *names: str) -> dict[str, int]:
        key = (arm, field, names)
        if key not in cache:
            cache[key] = _per_citizen(on_rows if arm == "on" else off_rows,
                                      roster, field, tuple(names))
        return cache[key]
    def growth(field: str, *names: str):
        on, off = per("on", field, *names), per("off", field, *names)
        return lambda ids: _ratio_percent(on, off, ids)
    def result(id_: str, fn, unit: str, method: str, reason: str, **extra):
        value, ci = _estimate(fn, roster, draws=draws, seed=seed)
        if value is None:
            reason = "OFF denominator is zero in the frozen effect window; " + reason
        return _item(id_, value, unit, ci, len(roster), method, reason, **extra)

    out = []
    if policy == "P010":
        on, off = per("on", "total_spent"), per("off", "total_spent")
        received = {aid: next(row["grant_received_cumulative"] for row in
                              reversed(on_rows) if row["aid"] == aid) for aid in roster}
        out.append(result("P010-1", lambda ids: _ratio_grant(on, off, received, ids),
                          "ratio", "(ON−OFF recorded total spending)/sum of final grant entitlement issued",
                          "Issued-grant transaction ratio is a short-run proxy, not the BOK nine-item survey MPC. It includes unspent entitlement in the denominator."))
        funded_total = per("on", "policy_funded_won")
        for id_, sub_names in P010_BOK_GROUPS.items():
            group = per("on", "funded_by_sub", *sub_names)
            out.append(result(id_, lambda ids, group=group: _share(group, funded_total, ids),
                              "%", "100×ON policy-funded spending in listed POI subclasses / all ON policy-funded spending",
                              "Exploratory descriptive benchmark, not a preregistered policy-effect score. BOK card-merchant classes, Seoul POI subclasses, recipient coverage and observation window may differ.",
                              exploratory_not_registered=True,
                              poi_subclasses=list(sub_names)))
    elif policy == "P012":
        out.append(result("P012-1", growth("sangsaeng_eligible_offline_spent"), "%",
                          "100×(ON−OFF eligible offline spending)/OFF, matched citizen-days",
                          "KDI reference is a household-month/year/recipient triple-difference log coefficient."))
        out.append(result("P012-2", growth("online_spent"), "%",
                          "100×(ON−OFF online State spending)/OFF, matched citizen-days",
                          "Online State omits excluded department-store and large-mart transactions; KDI reference is a household-month log triple difference and nonsignificance is not zero."))
        appliance, beauty = growth("by_sub", *APPLIANCE_P012), growth("by_l1", "미용")
        def sector_gap(ids):
            x, y = appliance(ids), beauty(ids)
            return x - y if x is not None and y is not None else None
        value, ci = _estimate(sector_gap, roster, draws=draws, seed=seed)
        components = {"appliance_furniture_pct": appliance(roster),
                      "hair_beauty_pct": beauty(roster)}
        out.append(_item("P012-5", value, "percentage points", ci, len(roster),
                         "Appliance/furniture by_sub (가구, 가전·통신, 통신) growth minus beauty by_l1 growth",
                         "Short POI ON/OFF percentage-gap proxy differs from two monthly household triple-difference log coefficients.",
                         simulation_components=components))
        for id_, why in (("P012-4", "Monthly cashback payout State is absent from this transaction ledger."),
                         ("P012-6", "Monthly cap-reached State is absent from this transaction ledger.")):
            out.append(_item(id_, None, "원" if id_ == "P012-4" else "%", None, len(roster),
                             "Requires full-month finalized State cashback ledger", why))
    elif policy == "DISTANCING_2020":
        out.append(result("DS-1", growth("by_sub", "한식"), "%",
                          "100×(ON−OFF 한식 POI spending)/OFF on matched dates",
                          "Seoul Institute reports year-on-year Korean-restaurant sales across a longer COVID period, not this paired-day contrast."))
        out.append(result("DS-2", growth("by_l1", "쇼핑", "마트"), "%",
                          "100×(ON−OFF shopping-plus-mart POI spending)/OFF",
                          "Retail scope and observation period differ from the Seoul Institute COVID-period average."))
        out.append(_item("DS-6", None, "%", None, len(roster),
                         "Needs official commercial-hub geometry and per-POI hub tag",
                         "Current graph does not classify POIs as tourism special zones or developed commercial districts."))
    elif policy == "P016":
        target, mart = growth("by_sub", *TARGET_P016), growth("by_l1", "마트")
        nested_mart = all(sum(row["by_sub"].get(name, 0) for name in TARGET_P016)
                          <= row["by_l1"].get("마트", 0)
                          for row in (*on_rows, *off_rows))
        out.append(result("C1", target, "%",
                          "100×(ON−OFF sales at four target POI subclasses)/OFF",
                          "KREI measures agricultural product sales at five large retailers with historical DID; POI gross sales include other goods."))
        if not nested_mart:
            reason = ("At least one target POI subclass is outside the mart parent "
                      "in the exported graph; within-mart denominator is not valid.")
            for id_, unit, method in (("C2", "percentage points",
                                        "Target-POI growth minus all-mart-POI growth"),
                                       ("C3", "percentage points",
                                        "ON/OFF target-POI share within mart")):
                out.append(_item(id_, None, unit, None, len(roster), method, reason))
            return out
        def rank_gap(ids):
            x, y = target(ids), mart(ids)
            return x - y if x is not None and y is not None else None
        value, ci = _estimate(rank_gap, roster, draws=draws, seed=seed)
        out.append(_item("C2", value, "percentage points", ci, len(roster),
                         "Target-POI growth minus all-mart-POI growth",
                         "KREI compares agricultural products and all products inside the same large marts over the program period.",
                         simulation_components={"target_poi_pct": target(roster), "all_mart_poi_pct": mart(roster)}))
        on_num, off_num = per("on", "by_sub", *TARGET_P016), per("off", "by_sub", *TARGET_P016)
        on_den, off_den = per("on", "by_l1", "마트"), per("off", "by_l1", "마트")
        out.append(result("C3", lambda ids: _share_change(on_num, off_num, on_den, off_den, ids),
                          "percentage points", "100×[(ON target-POI/mart)−(OFF target-POI/mart)]",
                          "KREI measures eligible-product share within large marts, not separate POI subclass share."))
    elif policy == "P014":
        for id_, sector in (("P014-KIPF-47121", "슈퍼마켓"),
                            ("P014-KIPF-47129", "식료품")):
            out.append(result(id_, growth("by_sub", sector), "%",
                              f"100×(ON−OFF {sector} POI spending)/OFF",
                              "Exploratory only. KIPF reports municipality-year-industry log-sales response to issuance intensity/GRDP; category and estimator do not align.",
                              exploratory_not_registered=True))
    else:
        raise ValueError(f"unsupported policy: {policy}")
    return out


def score_ds6_geographic_proxy(path: Path | None, *, on: Path, off: Path,
                               days: list[str], roster: list[str]
                               ) -> tuple[dict, list[Path]]:
    """Attach a separately registered 2023-geometry diagnostic, never DS-6.

    The sidecar is made from preserved positive purchase receipts, not the
    sector ledger. Its hashes and paired citizen-day matrix are checked here;
    a technical number and an interpretable number have distinct gates.
    """
    method = ("2023 official Seoul commercial-district polygons × 2026 POI "
              "coordinates; receipt-won ON/OFF growth in tourism special zones "
              "minus developed commercial districts")
    mismatch = ("Exploratory geography proxy only: the Seoul Institute's 2020 "
                "Shinhan-card merchant panel, boundary vintage, merchant sample, "
                "year-on-year period and estimator differ. Never subtract this "
                "from the registered DS-6 empirical percentage-point gap.")
    if path is None:
        return (_item(DS6_GEO_PROXY_ID, None, "percentage points", None, len(roster),
                      method, "No verified 2023 polygon × receipt overlay was attached. " + mismatch,
                      exploratory_not_registered=True), [])
    payload = read_json(path)
    if payload.get("status") != "exploratory_geographic_proxy_not_direct_empirical_comparison":
        raise ValueError("wrong DS-6 geographic sidecar status")
    if payload.get("days") != days or payload.get("citizen_days_each_arm") != len(roster) * len(days):
        raise ValueError("DS-6 geographic sidecar has a different citizen-day matrix")
    if payload.get("polygon_epsg") != 5181:
        raise ValueError("DS-6 official polygon CRS mismatch")
    sources = payload.get("sources_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("DS-6 geographic sidecar lacks source hashes")
    source_paths = []
    for source, expected_hash in sources.items():
        source_path = Path(source)
        if (not isinstance(expected_hash, str) or len(expected_hash) != 64
                or not source_path.is_file() or sha256(source_path) != expected_hash):
            raise ValueError(f"DS-6 geographic source SHA256 mismatch: {source}")
        source_paths.append(source_path)
    expected_metrics = {((arm.parent / "metrics" / f"day_{day}.jsonl").resolve())
                        for arm in (on, off) for day in days}
    if not expected_metrics.issubset({source.resolve() for source in source_paths}):
        raise ValueError("DS-6 geographic sidecar is not bound to these paired metrics")
    boundary = [(source, digest) for source, digest in sources.items()
                if Path(source).name.endswith("2023-10-23.zip")]
    if (len(boundary) != 1 or boundary[0][1] != DS6_OFFICIAL_BOUNDARY_SHA256):
        raise ValueError("DS-6 official 2023 boundary SHA256 mismatch")

    receipt_totals = payload.get("positive_receipts") or {}
    join_rates = payload.get("coordinate_join_rate_receipts") or {}
    ambiguous = payload.get("ambiguous_receipts_excluded") or {}
    cells = {}
    for arm in ("on", "off"):
        if (not isinstance(receipt_totals.get(arm), int)
                or isinstance(receipt_totals[arm], bool)
                or receipt_totals[arm] < 0):
            raise ValueError(f"invalid DS-6 {arm} receipt total")
        rate = join_rates.get(arm)
        if (rate is not None and (not isinstance(rate, (int, float))
                                  or isinstance(rate, bool)
                                  or not math.isfinite(rate) or not 0 <= rate <= 1)):
            raise ValueError(f"invalid DS-6 {arm} coordinate join rate")
        if (not isinstance(ambiguous.get(arm), int)
                or isinstance(ambiguous[arm], bool)
                or ambiguous[arm] < 0):
            raise ValueError(f"invalid DS-6 {arm} overlapping receipt count")
        summary = payload.get(arm)
        if not isinstance(summary, dict):
            raise ValueError(f"missing DS-6 {arm} cell summaries")
        for key, korean_name in DS6_GEO_TYPES.items():
            cell = summary.get(korean_name)
            if not isinstance(cell, dict):
                raise ValueError(f"missing DS-6 {arm}/{korean_name} cell")
            for field in ("spend_won", "positive_receipts", "citizens_with_receipts"):
                amount = cell.get(field)
                if (not isinstance(amount, int) or isinstance(amount, bool)
                        or amount < 0):
                    raise ValueError(f"invalid DS-6 {arm}/{korean_name}/{field}")
            if cell["citizens_with_receipts"] > len(roster):
                raise ValueError("DS-6 cell has more buying citizens than cohort")
            cells[arm, key] = cell

    min_rate = min(join_rates.get(arm) if join_rates.get(arm) is not None else -1
                   for arm in ("on", "off"))
    overlap_count = ambiguous["on"] + ambiguous["off"]
    off_denominators = {key: cells["off", key]["spend_won"] for key in DS6_GEO_TYPES}
    mapped_receipts = sum(round(receipt_totals[arm] * join_rates[arm])
                          for arm in ("on", "off") if join_rates.get(arm) is not None)
    total_receipts = receipt_totals["on"] + receipt_totals["off"]
    sparse_cells = [f"{arm}/{key}: receipts={cells[arm, key]['positive_receipts']}, "
                    f"citizens={cells[arm, key]['citizens_with_receipts']}"
                    for arm in ("on", "off") for key in DS6_GEO_TYPES
                    if (cells[arm, key]["positive_receipts"] < DS6_MIN_RECEIPTS_PER_CELL
                        or cells[arm, key]["citizens_with_receipts"] < DS6_MIN_CITIZENS_PER_CELL)]
    audit = {
        "source_year": 2023,
        "source_boundary_sha256": boundary[0][1],
        "match_rate": min_rate,
        "match_rate_by_arm": join_rates,
        "overlap_count": overlap_count,
        "off_denominator_won_by_type": off_denominators,
        "mapped_receipt_count": mapped_receipts,
        "total_receipt_count": total_receipts,
        "on_citizen_days": payload["citizen_days_each_arm"],
        "off_citizen_days": payload["citizen_days_each_arm"],
        "minimum_receipts_per_cell": DS6_MIN_RECEIPTS_PER_CELL,
        "minimum_citizens_per_cell": DS6_MIN_CITIZENS_PER_CELL,
        "sparse_interpretation_blocked": bool(sparse_cells),
        "sparse_reason": ("At least one of the four hub-type × arm cells has fewer "
                          "than 20 positive purchase receipts or 10 distinct buyers: "
                          + "; ".join(sparse_cells)) if sparse_cells else None,
    }
    technical_failures = []
    if min_rate < 0.99:
        technical_failures.append("positive-receipt coordinate join below 99%")
    if overlap_count:
        technical_failures.append("receipts assigned to overlapping hub types")
    if any(amount <= 0 for amount in off_denominators.values()):
        technical_failures.append("zero OFF won denominator in a hub type")
    if technical_failures:
        return (_item(DS6_GEO_PROXY_ID, None, "percentage points", None, len(roster),
                      method, "; ".join(technical_failures) + ". " + mismatch,
                      exploratory_not_registered=True, geo_proxy_audit=audit),
                [path, *source_paths])

    expected_rates = {}
    for key, korean_name in DS6_GEO_TYPES.items():
        expected_rates[key] = 100 * (cells["on", key]["spend_won"] /
                                     cells["off", key]["spend_won"] - 1)
        observed_rate = (payload.get("on_off_percent_change") or {}).get(korean_name)
        if (not isinstance(observed_rate, (int, float))
                or not math.isfinite(observed_rate)
                or not math.isclose(observed_rate, expected_rates[key], abs_tol=1e-9)):
            raise ValueError(f"DS-6 {korean_name} reported rate differs from receipt sums")
    gap = expected_rates["tourism_special_zone"] - expected_rates["developed_commercial_district"]
    reported_gap = payload.get("tourism_minus_developed_percentage_points")
    if (not isinstance(reported_gap, (int, float)) or not math.isfinite(reported_gap)
            or not math.isclose(reported_gap, gap, abs_tol=1e-9)):
        raise ValueError("DS-6 geographic sidecar gap differs from component rates")
    return (_item(DS6_GEO_PROXY_ID, gap, "percentage points", None, len(roster),
                  method, mismatch, exploratory_not_registered=True,
                  simulation_components={"tourism_special_zone_pct": expected_rates["tourism_special_zone"],
                                         "developed_commercial_district_pct": expected_rates["developed_commercial_district"]},
                  geo_proxy_audit=audit),
            [path, *source_paths])


def score_p013(paired_effect: Path, paired_sector: Path) -> tuple[list[dict], list[Path], str]:
    effect, sector = read_json(paired_effect), read_json(paired_sector)
    if (effect.get("policy_id") != "P013" or not effect.get("complete_matrix")
            or effect.get("provenance", {}).get("prompt_variant") != "v53"
            or sector.get("schema") != "p013_sector_pair_v1"
            or effect["citizens"] != sector["citizens"]
            or sector.get("scoring_table_sha256") != sha256(SCORING_TABLE)):
        raise ValueError("P013 paired evidence is incomplete or not v53")
    source_dir = paired_effect.parent
    arm_files = [source_dir / name for name in
                 ("on.ledger.jsonl", "off.ledger.jsonl", "on.sector.json", "off.sector.json")]
    expected = [effect["provenance"]["arms"]["on"]["ledger_sha256"],
                effect["provenance"]["arms"]["off"]["ledger_sha256"],
                sector["on_sha256"], sector["off_sha256"]]
    if any(not path.is_file() or sha256(path) != digest
           for path, digest in zip(arm_files, expected)):
        raise ValueError("P013 source ledger or sector SHA256 mismatch")
    n = effect["citizens"]
    def pct_interval(key):
        bounds = effect.get(key)
        return [100 * x for x in bounds] if isinstance(bounds, list) and len(bounds) == 2 else None
    items = [
        _item("EM-2", 100 * effect["eligible_offline_relative_change"], "%",
              pct_interval("eligible_offline_relative_citizen_bootstrap_95_interval"), n,
              "100×(ON−OFF eligible offline spending)/OFF for the three policy days",
              "KDI reference is a percentage-point change in year-on-year national card-sales growth; no direct subtraction."),
        _item("EM-3", 100 * effect["recorded_total_relative_change"], "%",
              pct_interval("recorded_total_relative_citizen_bootstrap_95_interval"), n,
              "100×(ON−OFF total recorded spending)/OFF for the three policy days",
              "KDI reference is 42-day national card-sales year-on-year growth, not three-day paired spending."),
        _item("EM-4", sector["rank_gap_percentage_points"], "percentage points",
              sector.get("citizen_bootstrap_95_interval"), n,
              "Semi-durable POI-group ON/OFF growth minus face-service POI-group ON/OFF growth",
              "KDI reference uses different item/industry coverage and treatment estimator; pilot uncertainty crosses zero.",
              simulation_components={"semi_durable_pct": sector["semidurable_relative_change_pct"],
                                     "face_service_pct": sector["face_service_relative_change_pct"]})
    ]
    dates_ = sector["days"]
    if not dates_ or _days(dates_[0], dates_[-1]) != dates_:
        raise ValueError("P013 sector dates invalid")
    return items, [paired_effect, paired_sector, *arm_files], f"{dates_[0]}:{dates_[-1]}"


def apply_cashback_month(indicators: list[dict], cashback_score: Path, *,
                         expected_citizens: int,
                         sector_manifests: tuple[dict, dict] | None = None) -> list[dict]:
    """Fill P012 payout/cap proxies only from an independently gated full month."""
    payload = read_json(cashback_score)
    provenance = payload.get("provenance") or {}
    if (payload.get("policy_id") != "P012" or payload.get("month") != "2021-10"
            or payload.get("days") != 31 or payload.get("citizens") != expected_citizens
            or payload.get("complete_paired_matrix") is not True
            or provenance.get("prompt_variant") != "v53"):
        raise ValueError("cashback score is not complete matched October v53 evidence")
    if sector_manifests is not None:
        on_manifest, off_manifest = sector_manifests
        for key in ("system_prompt_sha256", "stage2_system_prompt_sha256",
                    "baseline_income_map_sha256"):
            if provenance.get(key) != on_manifest["prompt_provenance"].get(key):
                raise ValueError(f"cashback score differs from sector run: {key}")
        arms = provenance.get("arms") or {}
        for arm, manifest in (("on", on_manifest), ("off", off_manifest)):
            identity = arms.get(arm) or {}
            cohort = manifest["prompt_provenance"]
            if (identity.get("run_id") != cohort.get("run_id")
                    or identity.get("execution_fingerprint") != cohort.get("execution_fingerprint")):
                raise ValueError(f"cashback score differs from sector {arm} run identity")
    metrics = payload.get("metrics") or {}
    recipients = payload.get("recipients")
    if (isinstance(recipients, bool) or not isinstance(recipients, int)
            or not 0 <= recipients <= expected_citizens):
        raise ValueError("invalid cashback recipient count")
    capped = payload.get("capped_recipients")
    total = payload.get("total_cashback_accrued_won")
    if (isinstance(capped, bool) or not isinstance(capped, int)
            or not 0 <= capped <= recipients or isinstance(total, bool)
            or not isinstance(total, (int, float)) or not math.isfinite(total)
            or total < 0):
        raise ValueError("invalid complete-month cashback totals")
    expected_mean = total / recipients if recipients else None
    expected_cap_share = capped / recipients if recipients else None
    for key, expected in (("cashback_per_recipient_won", expected_mean),
                          ("cap_share_recipients", expected_cap_share)):
        observed = (metrics.get(key) or {}).get("value")
        if (observed is None) != (expected is None) or (
            observed is not None and (isinstance(observed, bool)
                                      or not isinstance(observed, (int, float))
                                      or not math.isclose(observed, expected, rel_tol=1e-9,
                                                          abs_tol=1e-7))
        ):
            raise ValueError(f"cashback score does not reconcile: {key}")
    mapping = (("P012-4", "cashback_per_recipient_won", 1.0, "원"),
               ("P012-6", "cap_share_recipients", 100.0, "%"))
    by_id = {entry["id"]: entry for entry in indicators}
    for id_, key, scale, unit in mapping:
        metric = metrics.get(key) or {}
        value = metric.get("value")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError(f"invalid complete-month cashback value: {key}")
        interval = metric.get("citizen_bootstrap_95_interval")
        if interval is not None and (not isinstance(interval, list) or len(interval) != 2
                                     or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                            or not math.isfinite(x) for x in interval)):
            raise ValueError(f"invalid cashback interval: {key}")
        row = by_id[id_]
        row["simulation"] = value * scale if value is not None else None
        row["simulation_unit"] = unit
        row["ci"] = [x * scale for x in interval] if interval is not None else None
        row["n"] = recipients if value is not None else None
        row["method"] = ("Complete 2021-10 ON month: rule-implied cashback accrued at month end / citizens with positive accrual; no following-month payout transaction or State was observed"
                         if id_ == "P012-4" else
                         "Complete 2021-10 ON month: citizens whose rule-implied accrual reaches the monthly cap / citizens with positive accrual")
        row["reason"] = ("The KDI main reference is cashback actually paid over October–November; its rounded October table yields a separate approximate one-month comparison. "
                         "Simulation is October month-end rule-implied accrual, not observed payment, and the synthetic citizens are not the nationwide recipient frame.")
        row["empirical_variant"] = "october_only"
        row["cashback_score_evidence"] = display_path(cashback_score)
    return indicators


def audit_policy_funding_density(rows: list[dict], full_run_citizen_days: int) -> dict:
    """Display-only P010 denominator evidence; rows are citizen-days, not receipts."""
    if not rows or full_run_citizen_days < len(rows):
        raise ValueError("invalid funded citizen-day denominator")
    return {
        "policy_funded_positive_citizen_days": sum(row["policy_funded_won"] > 0
                                                    for row in rows),
        "policy_funded_observed_citizen_days": len(rows),
        "policy_funded_total_won": sum(row["policy_funded_won"] for row in rows),
        "full_run_citizen_days": full_run_citizen_days,
        "scope": "ON policy-effect days; one ledger row per citizen-day, not per transaction",
    }


def score_run(*, policy: str, experiment: str, on: Path | None = None,
              off: Path | None = None, effect_start: str | None = None,
              effect_end: str | None = None, paired_effect: Path | None = None,
              paired_sector: Path | None = None, cashback_score: Path | None = None,
              geographic_proxy: Path | None = None,
              draws: int = 1000) -> dict:
    if policy not in POLICY_TO_SCORE_KEY:
        raise ValueError(f"unsupported policy: {policy}")
    if geographic_proxy is not None and policy != "DISTANCING_2020":
        raise ValueError("--geographic-proxy only applies to DISTANCING_2020")
    if policy == "P013":
        if not paired_effect or not paired_sector:
            raise ValueError("P013 needs --paired-effect and --paired-sector")
        indicators, files, window = score_p013(paired_effect, paired_sector)
        n = indicators[0]["n"]
        frozen = read_json(paired_effect)
        historical = frozen["provenance"]
        p013_preperiod_path = paired_effect.parent / "preperiod_balance.json"
        p013_preperiod = None
        if p013_preperiod_path.is_file():
            old = read_json(p013_preperiod_path)
            arms = historical["arms"]
            if (old.get("on_ledger_sha256") != arms["on"]["ledger_sha256"]
                    or old.get("off_ledger_sha256") != arms["off"]["ledger_sha256"]):
                raise ValueError("P013 preperiod summary does not match paired ledgers")
            p013_preperiod = {
                "status": "historical_not_preregistered",
                "preperiod_days": old["outcomes"]["total"]["pre"]["days"],
                "comparisons": {
                    "recorded_total_spend": old["outcomes"]["total"]["pre"],
                    "eligible_offline_spend": old["outcomes"]["eligible_offline"]["pre"],
                },
                "post_effect_causal_interpretation_blocked": True,
                "reason": "Historical pilot showed nonzero pre-policy arm drift; no prospective balance threshold was frozen for this run.",
            }
            files.append(p013_preperiod_path)
        run_provenance = {
            "generic_prompt_variant": "v53",
            "generic_prompt_sha256": historical["system_prompt_sha256"],
            "stage2_sha256": historical["stage2_system_prompt_sha256"],
            "policy_id": "P013", "effective_from": None, "effective_until": None,
            "on_environment_id": None, "off_environment_id": None,
            "paired_environment_fingerprint": historical.get("paired_environment_fingerprint"),
            "policy_file_path": None,
            "policy_file_sha256": historical.get("policy_file_sha256"),
            "on_run_id": historical["arms"]["on"]["run_id"],
            "off_run_id": historical["arms"]["off"]["run_id"],
            "requested_model_id": None,
            "requested_model_id_is_weight_evidence": False,
            "served_model_provenance": None,
            "quality_audit": None,
            "preperiod_balance": p013_preperiod,
        }
    else:
        if not on or not off:
            raise ValueError("paired --on and --off sector ledgers required")
        on_rows, off_rows, roster, days, files = read_pair(on, off, policy=policy,
                                                          effect_start=effect_start,
                                                          effect_end=effect_end)
        indicators = score_ledger(on_rows, off_rows, roster, policy=policy, draws=draws)
        if policy == "DISTANCING_2020":
            geo_indicator, geo_files = score_ds6_geographic_proxy(
                geographic_proxy, on=on, off=off, days=days, roster=roster)
            indicators.append(geo_indicator)
            files.extend(geo_files)
        on_manifest, off_manifest = read_json(_manifest_path(on)), read_json(_manifest_path(off))
        # Display-only denominator audit added after the frozen 452cfaf formulas.
        # This counts funded citizen-days, never individual transaction receipts.
        funding_density = None
        if policy == "P010":
            funding_density = audit_policy_funding_density(on_rows, on_manifest["rows"])
            for indicator in indicators:
                if indicator["id"] in P010_BOK_GROUPS:
                    indicator.update(funding_density)
        try:
            on_quality, on_quality_files = audit_arm_quality(on, on_manifest)
        except (OSError, ValueError) as exc:
            on_quality, on_quality_files = ({"audit_status": "unavailable",
                                              "reason": str(exc),
                                              "stage1_first_attempt_internal_validation_pass_rate": None,
                                              "stage2_quality_gate_pass": None}, [])
        try:
            off_quality, off_quality_files = audit_arm_quality(off, off_manifest)
        except (OSError, ValueError) as exc:
            off_quality, off_quality_files = ({"audit_status": "unavailable",
                                                "reason": str(exc),
                                                "stage1_first_attempt_internal_validation_pass_rate": None,
                                                "stage2_quality_gate_pass": None}, [])
        files.extend([*on_quality_files, *off_quality_files])
        try:
            preperiod = audit_preperiod_balance(on, off, policy=policy,
                                                on_manifest=on_manifest)
        except (OSError, ValueError) as exc:
            preperiod = {"status": "audit_error", "reason": str(exc),
                         "post_effect_causal_interpretation_blocked": True}
        prompt = on_manifest["prompt_provenance"]
        policy_input = on_manifest.get("policy_input") or {}
        run_provenance = {
            "generic_prompt_variant": "v53",
            "generic_prompt_sha256": prompt["system_prompt_sha256"],
            "stage2_sha256": prompt["stage2_system_prompt_sha256"],
            "policy_id": on_manifest["policy_id"],
            "effective_from": on_manifest.get("effective_from"),
            "effective_until": on_manifest.get("effective_until"),
            "on_environment_id": on_manifest["provenance"]["experience_environment_id"][0],
            "off_environment_id": off_manifest["provenance"]["experience_environment_id"][0],
            "paired_environment_fingerprint": on_manifest["provenance"]["paired_environment_fingerprint"][0],
            "policy_file_path": policy_input.get("path"),
            "policy_file_sha256": policy_input.get("sha256"),
            "on_run_id": prompt["run_id"],
            "off_run_id": off_manifest["prompt_provenance"]["run_id"],
            "requested_model_id": prompt["requested_model_id"],
            "requested_model_id_is_weight_evidence": False,
            "served_model_provenance": {
                "model_id": on_manifest["served_model_provenance"]["model_id"],
                "on_evidence_sha256": on_manifest["served_model_provenance"]["evidence_sha256"],
                "off_evidence_sha256": off_manifest["served_model_provenance"]["evidence_sha256"],
                "on_evidence_path": display_path(on.parent / "served_model_evidence.json"),
                "off_evidence_path": display_path(off.parent / "served_model_evidence.json"),
            },
            "quality_audit": {"on": on_quality, "off": off_quality},
            "preperiod_balance": preperiod,
        }
        if funding_density is not None:
            run_provenance["policy_funding_density"] = funding_density
            # Post-run explanatory evidence is attached for disclosure only.
            # It never changes an indicator's value, uncertainty or gate.
            diagnosis_path = on.parent / "p010_wallet_diagnosis.json"
            if diagnosis_path.is_file():
                diagnosis = read_json(diagnosis_path)
                if (diagnosis.get("days") != days
                        or diagnosis.get("citizens") != len(roster)
                        or diagnosis.get("funded_won") !=
                        funding_density["policy_funded_total_won"]
                        or diagnosis.get("citizen_days_with_policy_payment") !=
                        funding_density["policy_funded_positive_citizen_days"]):
                    raise ValueError("P010 wallet diagnosis differs from verified sector ledger")
                diagnostic_fields = (
                    "positive_purchase_events", "eligible_purchase_events",
                    "positive_purchase_won", "eligible_purchase_won",
                    "funded_purchase_events", "funded_won",
                    "citizen_days_with_eligible_purchase",
                    "citizen_days_with_policy_payment",
                    "policy_hits_positive_citizen_days",
                    "stage1_grant_style_present_citizen_days",
                    "stage1_grant_use_present_citizen_days",
                    "choice_mode_citizen_days",
                    "policy_request_positive_citizen_days", "policy_requested_won",
                    "policy_allocated_positive_citizen_days", "policy_allocated_won",
                )
                if any(isinstance(diagnosis.get(key), bool)
                       or not isinstance(diagnosis.get(key), int)
                       or diagnosis[key] < 0 for key in diagnostic_fields):
                    raise ValueError("P010 wallet diagnosis has invalid counts or amounts")
                run_provenance["policy_funding_diagnostic"] = {
                    "status": "post_run_descriptive_quality_audit",
                    "path": display_path(diagnosis_path),
                    "sha256": sha256(diagnosis_path),
                    "days": diagnosis["days"],
                    "citizens": diagnosis["citizens"],
                    **{key: diagnosis[key] for key in diagnostic_fields},
                }
                files.append(diagnosis_path)
            else:
                run_provenance["policy_funding_diagnostic"] = {
                    "status": "not_available",
                    "reason": "No separately preserved post-run graph/metrics diagnosis beside ON ledger",
                }
        window = f"{days[0]}:{days[-1]}"
        n = len(roster)
        if cashback_score:
            if policy != "P012":
                raise ValueError("--cashback-score only applies to P012")
            if days != _days("2021-10-01", "2021-10-31"):
                raise ValueError("cashback score requires the same complete October sector window")
            indicators = apply_cashback_month(indicators, cashback_score,
                                              expected_citizens=n,
                                              sector_manifests=(read_json(_manifest_path(on)),
                                                                read_json(_manifest_path(off))))
            files.append(cashback_score)
    return {"schema": "multi_policy_numeric_v1", "experiment": experiment,
            "prompt_variant": "v53", "scoring_table_sha256": sha256(SCORING_TABLE),
            "runs": [{"policy": POLICY_TO_SCORE_KEY[policy], "policy_id": policy,
                      "label": POLICY_LABEL[policy], "off": window, "on": window,
                      "citizens": n, "evidence": evidence(*files),
                      "run_provenance": run_provenance,
                      "indicators": indicators}],
            "scope": "Simulation proxies only. Empirical reference values are not read here; external accuracy requires a separate run-specific estimand audit."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=tuple(POLICY_TO_SCORE_KEY), required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--on", type=Path)
    parser.add_argument("--off", type=Path)
    parser.add_argument("--effect-start")
    parser.add_argument("--effect-end")
    parser.add_argument("--paired-effect", type=Path)
    parser.add_argument("--paired-sector", type=Path)
    parser.add_argument("--cashback-score", type=Path,
                        help="Separately audited paired_cashback_month.py output for full October")
    parser.add_argument("--geographic-proxy", type=Path,
                        help="Separate SHA-audited 2023 commercial-district receipt overlay for DISTANCING_2020")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = score_run(policy=args.policy, experiment=args.experiment,
                        on=args.on, off=args.off, effect_start=args.effect_start,
                        effect_end=args.effect_end, paired_effect=args.paired_effect,
                        paired_sector=args.paired_sector,
                        cashback_score=args.cashback_score,
                        geographic_proxy=args.geographic_proxy, draws=args.draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{args.out}: {len(payload['runs'][0]['indicators'])} proxy indicators")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
