"""Opt-in frozen empirical/synthetic income profiles; no graph/model writes.

Only a person's assigned income and definition reach the persona. Aggregate
calibration targets and policy-effect benchmarks remain evaluator metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading

TIERS = ("하", "중하", "중", "중상", "상")
ROOT = Path(__file__).resolve().parents[2]
_CACHE = {}
_LOCK = threading.Lock()
_CONFIG_FROZEN = False
_CONFIG_SELECTION = None


class PopulationProfileError(ValueError):
    pass


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_enabled() -> bool:
    global _CONFIG_FROZEN, _CONFIG_SELECTION
    path = os.environ.get("POPULATION_PROFILE_FILE", "").strip()
    expected = os.environ.get("POPULATION_PROFILE_EXPECTED_SHA", "").strip()
    if bool(path) != bool(expected):
        raise PopulationProfileError("population profile requires both FILE and EXPECTED_SHA")
    selection = (str(Path(path).resolve()), expected.lower()) if path else None
    with _LOCK:
        if not _CONFIG_FROZEN:
            _CONFIG_SELECTION = selection
            _CONFIG_FROZEN = True
        elif selection != _CONFIG_SELECTION:
            raise PopulationProfileError("population profile configuration changed during execution")
    return bool(path)


def _sources(items, profile_path):
    if not isinstance(items, list) or not items:
        raise PopulationProfileError("profile source/mapping evidence is missing")
    result = []
    for item in items:
        if not isinstance(item, dict) or not item.get("path") or len(item.get("sha256", "")) != 64:
            raise PopulationProfileError("source path/SHA256 missing")
        raw = Path(item["path"])
        choices = [raw] if raw.is_absolute() else [profile_path.parent / raw, ROOT / raw]
        target = next((p for p in choices if p.is_file()), choices[0])
        if not target.is_file() or _sha(target) != item["sha256"].lower():
            raise PopulationProfileError("population source SHA256 mismatch")
        result.append({"path": str(target.resolve()), "sha256": item["sha256"].lower()})
    return result


def validate_profile(data: dict, path: Path) -> dict:
    if data.get("schema") != "frozen_population_profile_v1":
        raise PopulationProfileError("unsupported population profile schema")
    if (data.get("population_unit") != "resident_person" or isinstance(data.get("reference_year"), bool) or
            not isinstance(data.get("reference_year"), int) or not 1900 <= data["reference_year"] <= 2100):
        raise PopulationProfileError("profile unit/year missing")
    if data.get("assignment_kind") not in {"calibrated_synthetic_income_assignment", "verified_empirical_income_assignment"}:
        raise PopulationProfileError("spending/LLM tiers are not an empirical income profile")
    if data.get("policy_outcome_used_for_assignment") is not False:
        raise PopulationProfileError("policy outcomes cannot select income profiles")
    mapping = data.get("mapping_definition", {})
    if mapping.get("method") != "observed_income_band_mapping" or mapping.get("policy_outcome_used") is not False:
        raise PopulationProfileError("explicit observed-band income tier mapping is required")
    if mapping.get("observed_band_direct_mapping") is not True:
        raise PopulationProfileError("quintiles/LLM income cannot silently become policy tiers")
    bands = mapping.get("income_band_order", [])
    band_to_tier = mapping.get("band_to_tier", {})
    if not isinstance(bands, list) or not bands or len(set(bands)) != len(bands) or set(bands) != set(band_to_tier):
        raise PopulationProfileError("observed income band order/mapping is incomplete")
    if any(t not in TIERS for t in band_to_tier.values()):
        raise PopulationProfileError("income tier is not an existing policy key")
    ranks = [TIERS.index(band_to_tier[b]) for b in bands]
    if ranks != sorted(ranks):
        raise PopulationProfileError("income tiers reverse observed income band order")
    if any(not isinstance(b, str) or not b.strip() or any(x in b.lower() for x in ("모름", "무응답", "unknown", "missing")) for b in bands):
        raise PopulationProfileError("unknown/refused income cannot determine policy eligibility")
    sources = _sources(data.get("source_evidence"), path)
    sources.extend(_sources(mapping.get("source_evidence"), path))
    definition = data.get("household_income_definition")
    if not isinstance(definition, str) or not definition.strip() or len(definition) > 500:
        raise PopulationProfileError("household/personal income definition missing")
    if not data.get("admin_dong_code_system") or data.get("official_admin_crosswalk_verified") is not True:
        raise PopulationProfileError("administrative dong crosswalk/vintage is unverified")
    rows = data.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise PopulationProfileError("profile rows missing")
    indexed = {}
    for row in rows:
        aid = row.get("aid")
        if not isinstance(aid, str) or not aid or aid in indexed:
            raise PopulationProfileError("profile agent IDs must be distinct")
        if row.get("sex") not in {"M", "F"} or isinstance(row.get("age"), bool) or not isinstance(row.get("age"), int) or not 0 <= row["age"] <= 120:
            raise PopulationProfileError("profile exact age/sex missing")
        if not isinstance(row.get("home_dong_code"), str) or not row["home_dong_code"]:
            raise PopulationProfileError("actual residence-anchor dong code missing")
        if row.get("income_band") not in band_to_tier or row.get("income_tier") != band_to_tier[row["income_band"]]:
            raise PopulationProfileError("profile income does not follow audited observed-band mapping")
        if row.get("household_income_definition") != definition:
            raise PopulationProfileError("row and profile income definitions differ")
        indexed[aid] = {k: row[k] for k in ("aid", "sex", "age", "home_dong_code", "income_band", "income_tier", "household_income_definition")}
    return {"rows_by_aid": indexed, "roster": sorted(indexed), "source_evidence": sources,
            "reference_year": data["reference_year"], "assignment_kind": data["assignment_kind"],
            "admin_dong_code_system": data["admin_dong_code_system"], "household_income_definition": definition}


def active_profile() -> dict | None:
    if not profile_enabled():
        return None
    path = Path(os.environ["POPULATION_PROFILE_FILE"]).resolve()
    expected = os.environ["POPULATION_PROFILE_EXPECTED_SHA"].strip().lower()
    if len(expected) != 64 or not path.is_file():
        raise PopulationProfileError("population profile file/SHA256 is invalid")
    stat = path.stat()
    stamp = (stat.st_size, stat.st_mtime_ns)
    key = (str(path), expected)
    with _LOCK:
        if key in _CACHE:
            prior_stamp, value = _CACHE[key]
            if prior_stamp != stamp:
                raise PopulationProfileError("frozen population profile changed during execution")
            return value
        if _sha(path) != expected:
            raise PopulationProfileError("population profile SHA256 mismatch")
        try:
            value = validate_profile(json.loads(path.read_text(encoding="utf-8-sig")), path)
        except (OSError, json.JSONDecodeError, TypeError, KeyError, AttributeError) as exc:
            raise PopulationProfileError("invalid population profile content") from exc
        value.update({"profile_sha256": expected, "profile_path": str(path)})
        _CACHE[key] = (stamp, value)
        return value


def profile_roster() -> list[str] | None:
    profile = active_profile()
    return list(profile["roster"]) if profile else None


def preflight_profile_roster(aids: list[str]) -> dict | None:
    profile = active_profile()
    if profile is None:
        return None
    if len(aids) != len(set(aids)) or sorted(aids) != profile["roster"]:
        raise PopulationProfileError("runtime roster differs from frozen population profile")
    return {"sha256": profile["profile_sha256"], "citizens": len(aids),
            "assignment_kind": profile["assignment_kind"], "reference_year": profile["reference_year"],
            "admin_dong_code_system": profile["admin_dong_code_system"]}


def verify_graph_projection(rows: list[dict]) -> str | None:
    """Check all selected identities before any worker can call a model."""
    profile = active_profile()
    if profile is None:
        return None
    if (len(rows) != len(profile["roster"]) or len({r.get("aid") for r in rows}) != len(rows) or
            sorted(r.get("aid") for r in rows) != profile["roster"]):
        raise PopulationProfileError("graph projection is missing/duplicating frozen profile agents")
    for row in rows:
        expected = profile["rows_by_aid"][row["aid"]]
        if (row.get("sex") != expected["sex"] or row.get("age") != expected["age"] or
                str(row.get("home_dong_code") or "") != expected["home_dong_code"]):
            raise PopulationProfileError("whole-roster graph sex/age/residence identity gate failed")
    return hashlib.sha256(json.dumps(sorted(rows, key=lambda r: r["aid"]), ensure_ascii=False,
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def bind_persona(aid: str, persona: dict, exact_age: int | None) -> dict:
    profile = active_profile()
    if profile is None:
        return persona
    row = profile["rows_by_aid"].get(aid)
    if row is None or persona.get("id") != aid:
        raise PopulationProfileError("agent not present in frozen population profile")
    if (persona.get("gender") != row["sex"] or exact_age != row["age"] or
            str(persona.get("home_dong_code") or "") != row["home_dong_code"]):
        raise PopulationProfileError("frozen sex/age/actual residence anchor disagrees with graph")
    result = dict(persona)
    result["income"] = row["income_tier"]
    result["p_income_level"] = row["income_tier"]
    result["assigned_income_band"] = row["income_band"]
    result["assigned_income_definition"] = row["household_income_definition"]
    result["income_assignment_kind"] = profile["assignment_kind"]
    result["population_profile_sha256"] = profile["profile_sha256"]
    return result


def income_for_eligibility(persona: dict) -> str:
    profile = active_profile()
    if profile is None:
        return persona.get("income") or persona.get("p_income_level") or ""
    aid = persona.get("id")
    row = profile["rows_by_aid"].get(aid)
    if (row is None or persona.get("population_profile_sha256") != profile["profile_sha256"] or
            persona.get("income") != row["income_tier"] or persona.get("p_income_level") != row["income_tier"]):
        raise PopulationProfileError("policy eligibility income is not the frozen persona income")
    return row["income_tier"]
