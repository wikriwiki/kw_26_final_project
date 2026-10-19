"""Prepare frozen inputs and launch the existing engine for the smoking experiment.

No LLM calls or graph writes during prepare/preflight/smoke. Research outcomes
are read only by score, never copied into runtime.json or agent prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/experiments/no_smoking_zone"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.sim.no_smoking_context import EFFECTIVE_DATE

PRE_POLICY_DAYS = 14
POST_POLICY_DAYS = 14
DEFAULT_START = (EFFECTIVE_DATE - timedelta(days=PRE_POLICY_DAYS)).isoformat()
DEFAULT_DAYS = PRE_POLICY_DAYS + POST_POLICY_DAYS
DEFAULT_END = (EFFECTIVE_DATE + timedelta(days=POST_POLICY_DAYS - 1)).isoformat()
DEFAULT_DAY_ZERO = (date.fromisoformat(DEFAULT_START) - timedelta(days=1)).isoformat()
DISTRICTS = {"11350": "노원구", "11650": "서초구", "11710": "송파구"}
FACILITIES = {"billiard", "indoor_golf", "screen_golf"}
SETTING_PREFIXES = ("SIM_", "EXP_", "POLICY_", "CONSUMPTION_", "STAGE1_", "STAGE2_", "NIGHT_", "MOBILITY_", "HUB_")


def simulation_window(start, days):
    """Split inclusive simulation dates on legal onset, not the study's year dummy."""
    if type(days) is not int or days <= 0:
        raise ValueError("Simulation days must be a positive integer")
    first = date.fromisoformat(start)
    last = first + timedelta(days=days - 1)
    pre_days = min(days, max(0, (EFFECTIVE_DATE - first).days))
    post_days = days - pre_days
    return {"start": first.isoformat(), "end": last.isoformat(), "days": days,
            "day_zero": (first - timedelta(days=1)).isoformat(),
            "policy_effective_date": EFFECTIVE_DATE.isoformat(),
            "pre": {"start": first.isoformat() if pre_days else None,
                    "end": min(last, EFFECTIVE_DATE - timedelta(days=1)).isoformat() if pre_days else None,
                    "days": pre_days},
            "post": {"start": max(first, EFFECTIVE_DATE).isoformat() if post_days else None,
                     "end": last.isoformat() if post_days else None, "days": post_days},
            "matches_default_14_pre_14_post": first.isoformat() == DEFAULT_START and days == DEFAULT_DAYS}


def engine_settings(env):
    excluded = {"SIM_OUTPUT_DIR", "SIM_RUN_ID", "SIM_NO_SMOKING_MANIFEST", "SIM_NO_SMOKING_ARM"}
    return {k: v for k, v in sorted(env.items()) if k.startswith(SETTING_PREFIXES)
            and k not in excluded and not any(s in k.upper() for s in ("KEY", "PASSWORD", "TOKEN", "SECRET"))}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_hashes():
    files = sorted((ROOT / "output/stats").glob("*.json"))
    expected = {ROOT / "output/stats" / name for name in ("unit_price.json", "poi_menu_price.json", "dong_context.json", "hub_catalog.json", "dong_centroids.json")}
    return {str(p.relative_to(ROOT)): file_hash(p) if p.exists() else None for p in sorted(set(files) | expected)}


def load_agents(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            value = json.loads(archive.read("agents_final.json"))
    else:
        value = read_json(path)
    if isinstance(value, dict):
        value = value.get("agents", value.get("personas"))
    if not isinstance(value, list) or not value:
        raise ValueError("Agent input must be a nonempty list (or agents/personas list)")
    return value


def assign_smoking(agents, rates, seed):
    """Fixed quotas within adult age/sex strata; preserve all source personas."""
    cohort, strata = [], defaultdict(list)
    for source in agents:
        personal = source.get("personal") or {}
        aid = source.get("id") or source.get("agent_id") or source.get("uuid")
        if not isinstance(aid, str) or not aid:
            raise ValueError("Every agent needs a nonempty string ID")
        age = personal.get("age", source.get("age"))
        sex = str(personal.get("gender", source.get("sex", source.get("gender", "unknown"))))
        sex = {"M": "male", "남자": "male", "남성": "male", "F": "female", "여자": "female", "여성": "female"}.get(sex, sex)
        sex = sex if sex in {"male", "female"} else "all"
        row = {"id": aid, "age": age, "sex": sex, "smoking_status": "unknown", "assignment_basis": "age_unavailable_or_under_19"}
        cohort.append(row)
        if type(age) is not int or age < rates["minimum_age"]:
            continue
        bands = [b for b in rates["age_bands"] if b["min_age"] <= age and (b.get("max_age") is None or age <= b["max_age"])]
        if len(bands) != 1:
            raise ValueError(f"Missing or overlapping smoking age bands for age {age}")
        band = bands[0]
        rate = band[sex]
        if not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
            raise ValueError("Invalid smoking probability")
        key = f"{band['min_age']}:{band.get('max_age')}:{sex}"
        strata[key].append((row, rate))
    if len({r["id"] for r in cohort}) != len(cohort):
        raise ValueError("Duplicate agent ID")
    allocation = []
    for key, members in sorted(strata.items()):
        members.sort(key=lambda pair: pair[0]["id"])
        count = math.floor(len(members) * members[0][1] + 0.5)
        selected = set(random.Random(f"{seed}:{key}").sample([r["id"] for r, _ in members], count))
        for row, rate in members:
            row.update(smoking_status="smoker" if row["id"] in selected else "non_smoker", assignment_basis="synthetic_seoul_2017_age_sex_quota")
        allocation.append({"stratum": key, "n": len(members), "rate": members[0][1], "smokers": count})
    return sorted(cohort, key=lambda r: r["id"]), allocation


def normalize_pois(rows):
    if isinstance(rows, dict):
        rows = rows.get("pois")
    if not isinstance(rows, list):
        raise ValueError("POI file must contain a list or {pois: [...]} object")
    result = []
    for row in rows:
        # Classification must be reviewed upstream: broad leisure/name matching is unsafe.
        required = {"poi_id", "district_code", "facility_type", "classification_source"}
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Each POI requires {sorted(required)}")
        if not isinstance(row["poi_id"], str) or not row["poi_id"].strip():
            raise ValueError("POI ID must be a nonempty string")
        if str(row["district_code"]) not in DISTRICTS:
            raise ValueError("Evaluation POIs must be in Nowon, Seocho or Songpa")
        if row["facility_type"] not in FACILITIES:
            raise ValueError("Only verified billiard/indoor_golf/screen_golf POIs belong in target registry")
        if not isinstance(row["classification_source"], str) or not row["classification_source"].strip():
            raise ValueError("POI classification needs provenance")
        result.append(dict(row, district_code=str(row["district_code"])))
    if len({p["poi_id"] for p in result}) != len(result):
        raise ValueError("Duplicate POI ID")
    return sorted(result, key=lambda p: p["poi_id"])


def prepare(args):
    if args.out.exists():
        raise ValueError("Refusing to overwrite a frozen bundle; use a new output directory")
    rates = read_json(DATA / "smoking_rates.json")
    agents = load_agents(args.agents)
    cohort, allocation = assign_smoking(agents, rates, args.seed)
    population_size = len(agents)
    eligible_path = getattr(args, "eligible_ids", None)
    if eligible_path:
        eligible = read_json(eligible_path)
        if (not isinstance(eligible, list) or not eligible
                or any(not isinstance(aid, str) or not aid for aid in eligible)
                or len(set(eligible)) != len(eligible)):
            raise ValueError("eligible IDs must be a nonempty list of unique string IDs")
        ids = set(eligible)
        if ids - {p["id"] for p in cohort}:
            raise ValueError("eligible IDs include agents absent from the source population")
        cohort = [p for p in cohort if p["id"] in ids]
        agents = [p for p in agents if (p.get("id") or p.get("agent_id") or p.get("uuid")) in ids]
    eligible_size = len(cohort)
    cohort_path = getattr(args, "cohort_ids", None)
    if cohort_path:
        selected = read_json(cohort_path)
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(aid, str) or not aid for aid in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("cohort IDs must be a nonempty list of unique string IDs")
        ids = set(selected)
        if ids - {p["id"] for p in cohort}:
            raise ValueError("cohort IDs include agents absent from the eligible population")
        cohort = [p for p in cohort if p["id"] in ids]
        agents = [p for p in agents if (p.get("id") or p.get("agent_id") or p.get("uuid")) in ids]
    # Freeze the experiment roster before selecting a smaller pilot from it.
    experiment_ids = [p["id"] for p in cohort]  # assign_smoking returns sorted IDs.
    experiment_size = len(experiment_ids)
    selection = {"source_population_size": population_size,
                 "eligible_population_size": eligible_size,
                 "experiment_cohort_size": experiment_size,
                 "experiment_cohort_ids_sha256": digest(experiment_ids),
                 "cohort_source_sha256": file_hash(cohort_path) if cohort_path else None,
                 "cohort_selection": "explicit_ids" if cohort_path else "all_eligible",
                 "pilot_limit": args.limit}
    if args.limit is not None:
        if not 0 < args.limit <= experiment_size:
            raise ValueError("limit must be positive and no larger than selected experiment cohort (within eligible population)")
        ids = set(random.Random(f"pilot:{args.seed}").sample([p["id"] for p in cohort], args.limit))
        cohort = [p for p in cohort if p["id"] in ids]
        agents = [p for p in agents if (p.get("id") or p.get("agent_id") or p.get("uuid")) in ids]
    pois = normalize_pois(read_json(args.pois)) if args.pois else []
    # Keep the entire persona unchanged in a separate file for graph restoration.
    runtime = {"schema_version": 1, "experiment_id": "no_smoking_zone", "assignment_seed": args.seed,
               "simulation_seed": args.simulation_seed, "cohort": cohort, "pois": pois}
    args.out.mkdir(parents=True)
    write_json(args.out / "runtime.json", runtime)
    write_json(args.out / "personas.json", agents)
    write_json(args.out / "experiment_cohort_ids.json", experiment_ids)
    (args.out / "smoking_rates.json").write_bytes((DATA / "smoking_rates.json").read_bytes())
    write_json(args.out / "assignment_audit.json", {**selection,
               "excluded_ineligible_count": population_size - eligible_size,
               "eligibility_applied_after_smoking_assignment": True,
               "strata_full_population": allocation, "selected_cohort_size": len(cohort),
               "status_counts": dict(Counter(r["smoking_status"] for r in cohort)),
               "observed_individual_smoking_data": False, "assignment_type": "empirical_prevalence_based_synthetic_labels",
               "rate_source": rates["source"]})
    metadata = {
        **selection,
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "base_commit": "77ec86cf49397ac118fc63501f39abbb3244cab2",
        "agent_source_sha256": file_hash(args.agents), "rates_sha256": file_hash(DATA / "smoking_rates.json"),
        "runtime_sha256": digest(runtime), "personas_sha256": digest(agents),
        "poi_source_sha256": file_hash(args.pois) if args.pois else None,
        "eligibility_source_sha256": file_hash(eligible_path) if eligible_path else None,
        "population_interpretation": "Existing contemporary synthetic Seoul personas exposed to historical rules; not a reconstructed 2017 population",
        "empirical_replication_ready": False,
        "provisional_rates": not rates.get("source", {}).get("primary_source_verified", False),
    }
    write_json(args.out / "bundle.json", metadata)
    report = inspect_bundle(args.out)
    write_json(args.out / "preflight.json", report)
    return report


def inspect_bundle(bundle):
    runtime = read_json(bundle / "runtime.json")
    metadata = read_json(bundle / "bundle.json")
    blockers, warnings = [], []
    if digest(runtime) != metadata["runtime_sha256"]:
        blockers.append("Frozen runtime.json hash mismatch")
    if digest(read_json(bundle / "personas.json")) != metadata["personas_sha256"]:
        blockers.append("Frozen personas.json hash mismatch")
    if file_hash(bundle / "smoking_rates.json") != metadata["rates_sha256"]:
        blockers.append("Frozen smoking_rates.json hash mismatch")
    cohort = runtime["cohort"]
    if not cohort or len({r["id"] for r in cohort}) != len(cohort):
        blockers.append("Empty or duplicate cohort")
    if any(r["smoking_status"] not in {"smoker", "non_smoker", "unknown"} for r in cohort):
        blockers.append("Invalid smoking assignment")
    # Older bundles lack a separate experiment roster. New bundles freeze and
    # validate it so a pilot cannot silently become a different experiment.
    if "experiment_cohort_ids_sha256" in metadata:
        try:
            experiment_ids = read_json(bundle / "experiment_cohort_ids.json")
            if (not isinstance(experiment_ids, list) or not experiment_ids
                    or any(not isinstance(aid, str) or not aid for aid in experiment_ids)
                    or len(set(experiment_ids)) != len(experiment_ids)
                    or experiment_ids != sorted(experiment_ids)):
                raise ValueError("Frozen experiment cohort IDs must be sorted unique strings")
            if digest(experiment_ids) != metadata["experiment_cohort_ids_sha256"]:
                raise ValueError("Frozen experiment cohort IDs hash mismatch")
            sizes = [metadata.get(key) for key in ("source_population_size", "eligible_population_size", "experiment_cohort_size")]
            if (any(type(n) is not int or n <= 0 for n in sizes)
                    or not sizes[0] >= sizes[1] >= sizes[2] == len(experiment_ids)):
                raise ValueError("Experiment population size metadata is inconsistent")
            limit = metadata.get("pilot_limit")
            if limit is not None and (type(limit) is not int or not 0 < limit <= sizes[2]):
                raise ValueError("Invalid frozen pilot limit")
            if len(cohort) != (limit if limit is not None else sizes[2]) or not {r["id"] for r in cohort}.issubset(experiment_ids):
                raise ValueError("Runtime cohort differs from the frozen experiment/pilot roster")
            if limit is not None:
                expected_pilot = set(random.Random(f"pilot:{runtime['assignment_seed']}").sample(experiment_ids, limit))
                if {r["id"] for r in cohort} != expected_pilot:
                    raise ValueError("Runtime pilot differs from the seeded experiment subset")
            method = metadata.get("cohort_selection")
            source_hash = metadata.get("cohort_source_sha256")
            if method not in {"explicit_ids", "all_eligible"} or (method == "explicit_ids" and (not isinstance(source_hash, str) or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash))):
                raise ValueError("Missing/invalid explicit experiment cohort provenance")
            if method == "all_eligible" and (sizes[1] != sizes[2] or source_hash is not None):
                raise ValueError("All-eligible selection metadata must preserve the eligible roster size")
            audit = read_json(bundle / "assignment_audit.json")
            keys = ("source_population_size", "eligible_population_size", "experiment_cohort_size",
                    "experiment_cohort_ids_sha256", "cohort_source_sha256", "cohort_selection", "pilot_limit")
            if any(audit.get(key) != metadata.get(key) for key in keys) or audit.get("selected_cohort_size") != len(cohort):
                raise ValueError("Experiment selection audit differs from frozen bundle metadata")
        except (OSError, ValueError, TypeError) as exc:
            blockers.append(f"Invalid experiment cohort selection: {exc}")
    pois = normalize_pois(runtime["pois"])
    counts = Counter((p["district_code"], "indoor_golf" if p["facility_type"] == "screen_golf" else p["facility_type"]) for p in pois)
    for district in DISTRICTS:
        for facility in ("billiard", "indoor_golf"):
            if not counts[district, facility]:
                blockers.append(f"Missing verified {facility} POIs in {DISTRICTS[district]} ({district})")
    if metadata.get("provisional_rates"):
        warnings.append("Smoking rates transcribed from supplied HTML; primary statistics not verified")
    from scripts.sim.reference_preflight import inspect_references
    try:
        references = inspect_references(ROOT / "output/stats")
        warnings.extend(references["limitations"])
    except (OSError, ValueError) as exc:
        blockers.append(f"Missing/invalid geographical or price reference data: {exc}")
    unknown = sum(r["smoking_status"] == "unknown" for r in cohort)
    if unknown:
        warnings.append(f"{unknown} agents retain unknown smoking status; adult rates not applied")
    warnings.extend([
        "Live run additionally requires two isolated databases restored from the same verified snapshot, matching cohort/POIs and LLM health check",
        "Baseline nonzero payments and fallback audit must pass before scaling beyond a pilot",
        "ON/OFF simulated payments are not the study's adjusted monthly card-market regression estimand",
        "Rules modeled for the evaluated facility subset; this is not the full national indoor-sports ban",
    ])
    return {"preparation_ready": not blockers, "live_run_ready": False, "agents": len(cohort), "target_pois": len(pois), "blockers": blockers, "warnings": warnings}


def graph_preflight(runtime, arm, snapshot_id, start, branch=None, resume_manifest=None):
    """Read back identities/relationships before permitting any simulation writes."""
    from neo4j import GraphDatabase
    suffix = arm.upper()
    uri = os.environ.get(f"NO_SMOKING_{suffix}_NEO4J_URI")
    database = os.environ.get(f"NO_SMOKING_{suffix}_NEO4J_DATABASE", "neo4j")
    other_uri = os.environ.get(f"NO_SMOKING_{'ON' if arm == 'off' else 'OFF'}_NEO4J_URI")
    other_database = os.environ.get(f"NO_SMOKING_{'ON' if arm == 'off' else 'OFF'}_NEO4J_DATABASE", "neo4j")
    if not uri or not other_uri or (uri, database) == (other_uri, other_database):
        raise ValueError("Declare distinct NO_SMOKING_OFF/ON_NEO4J_URI/DATABASE endpoints before running")
    # Snapshot marker is deliberately an operator import prerequisite, never fabricated here.
    password = os.environ.get(f"NO_SMOKING_{suffix}_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD")
    user = os.environ.get("NEO4J_USER", "neo4j")
    if not password:
        raise ValueError("Neo4j password missing (provide environment variable, not CLI argument)")
    ids = [p["id"] for p in runtime["cohort"]]
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            marker = session.run("MATCH (s:ExperimentSnapshot {id:$id}) RETURN s.sha256 AS sha256", id=snapshot_id).single()
            if not marker or marker["sha256"] != snapshot_id:
                raise ValueError("Restored snapshot lacks matching ExperimentSnapshot {id,sha256} marker")
            found = {r["id"] for r in session.run("MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(:Dong) WHERE a.id IN $ids RETURN DISTINCT a.id AS id", ids=ids)}
            if found != set(ids):
                raise ValueError(f"Graph roster/anchors mismatch: {len(set(ids)-found)} missing agents")
            found_pois = {r["id"]: r["district"] for r in session.run("MATCH (p:POI)-[:IN_DONG]->(:Dong)<-[:HAS_DONG]-(d:District) WHERE p.id IN $ids RETURN p.id AS id, d.code AS district", ids=[p["poi_id"] for p in runtime["pois"]])}
            for p in runtime["pois"]:
                if str(found_pois.get(p["poi_id"])) != p["district_code"]:
                    raise ValueError(f"Graph POI district mismatch/missing: {p['poi_id']}")
            categorized = {r["id"] for r in session.run("MATCH (p:POI {type:'commerce'})-[:IN_CATEGORY]->(c:Category) WHERE p.id IN $ids AND c.parent='여가' RETURN DISTINCT p.id AS id", ids=list(found_pois))}
            if categorized != set(found_pois):
                raise ValueError("Target POIs require commerce type and leisure Category edges used by Stage2")
            states = list(session.run("MATCH (a:Agent)-[:HAS_STATE {day:date($day)}]->(s:State) WHERE a.id IN $ids RETURN a.id AS id, s.balance AS balance, s.experience_run_id AS run_id, s.agent_metrics_json AS metrics",
                          day=(date.fromisoformat(start)-timedelta(days=1)).isoformat(), ids=ids))
            if len(states) != len(ids) or {r["id"] for r in states} != set(ids) or any(type(r["balance"]) not in (int, float) or not math.isfinite(r["balance"]) or r["balance"] < 0 or int(r["balance"]) != r["balance"] for r in states):
                raise ValueError("Every cohort member needs a unique valid balance State on start minus one day")
            if branch is not None:
                if any(r["run_id"] != branch["source_run_id"] or not r["metrics"] for r in states):
                    raise ValueError("Branch states do not belong to the completed shared-pre run")
                future = session.run("MATCH (a:Agent)-[:HAS_STATE]->(s:State) WHERE a.id IN $ids AND s.day > date($day) RETURN count(s) AS n",
                                     ids=ids, day=branch["pre_end"]).single()["n"]
                if future and resume_manifest is None:
                    raise ValueError("Branch graph already has post-period states")
                foreign = session.run("MATCH (a:Agent)-[:HAS_STATE]->(s:State) WHERE a.id IN $ids AND s.experience_run_id IS NOT NULL AND s.experience_run_id <> $run_id RETURN count(s) AS n",
                                      ids=ids, run_id=branch["source_run_id"]).single()["n"]
                if foreign:
                    raise ValueError("Branch graph contains a foreign simulation run")
            elif resume_manifest is None:
                dirty = session.run("MATCH (n) WHERE n:Plan OR n:Memory OR n:Conversation OR (n:State AND (n.experience_run_id IS NOT NULL OR n.agent_metrics_json IS NOT NULL OR n.day IS NULL OR n.day <> date($day))) RETURN count(n) AS n",
                                    day=(date.fromisoformat(start)-timedelta(days=1)).isoformat()).single()["n"]
                if dirty:
                    raise ValueError("Graph already has simulation output; restore the frozen Day-0 snapshot")
                learned = session.run("MATCH ()-[kp:KNOWS_POI]->() WHERE kp.source IS NULL OR kp.source <> 'initial' OR kp.since IS NULL OR kp.since <> date($day) OR kp.affinity IS NULL OR kp.affinity <> 0.5 OR kp.visit_count IS NULL OR kp.visit_count <> 0 OR kp.avg_satisfaction IS NOT NULL OR kp.last_visit IS NOT NULL OR size(coalesce(kp.recent_visit_dates,[])) <> 0 RETURN count(kp) AS n",
                                      day=(date.fromisoformat(start)-timedelta(days=1)).isoformat()).single()["n"]
                if learned:
                    raise ValueError("Graph retains learned/future POI awareness; rebuild initial KNOWS_POI before freezing Day-0")
            if resume_manifest is not None:
                last = date.fromisoformat(start) + timedelta(days=resume_manifest['days'] - 1)
                bad = session.run('MATCH (a:Agent)-[:HAS_STATE]->(s:State) WHERE a.id IN $ids '
                    'AND s.day >= date($start) AND (s.day > date($last) OR s.experience_run_id IS NULL '
                    'OR s.experience_run_id <> $run OR s.agent_metrics_json IS NULL) RETURN count(s) AS n',
                    ids=ids, start=start, last=str(last), run=resume_manifest['run_id']).single()['n']
                if bad:
                    raise ValueError('Resume graph contains foreign, incomplete, or future states')
            policy_count = session.run("MATCH (p:Policy) RETURN count(p) AS n").single()["n"]
            if policy_count:
                raise ValueError("Baseline contains other policies; restore a policy-free experiment snapshot")
    return {"NEO4J_URI": uri, "NEO4J_DATABASE": database, "NEO4J_USER": user, "NEO4J_PASSWORD": password}


def validate_server_config(config):
    """Accept explicit serving-engine provenance without assuming vLLM."""
    if not isinstance(config, dict):
        raise ValueError("Model server configuration must be an object")
    engine = config.get("engine") or config.get("serving_engine")
    if config.get("engine") and config.get("serving_engine") and config["engine"] != config["serving_engine"]:
        raise ValueError("Conflicting model serving engine identities")
    if not engine:
        present = [name for name in ("sglang", "vllm") if config.get(name + "_version")]
        engine = present[0] if len(present) == 1 else None
    if engine not in {"sglang", "vllm"}:
        raise ValueError("Record the explicit model serving engine (sglang or vllm)")
    version = config.get(engine + "_version") or config.get("engine_version")
    argv = config.get("argv")
    if (not isinstance(version, str) or not version.strip() or not isinstance(argv, list)
            or not argv or not all(isinstance(arg, str) and arg for arg in argv)):
        raise ValueError("Incomplete model server version/revision record")
    return dict(config, engine=engine, engine_version=version)


def server_context_length(config):
    flag = '--context-length' if config['engine'] == 'sglang' else '--max-model-len'
    argv = config['argv']
    if argv.count(flag) != 1:
        raise ValueError('Server launch must declare exactly one context length')
    limit = int(argv[argv.index(flag) + 1])
    if limit < 16384:
        raise ValueError('Full recorded-corpus audit requires at least 16384 context tokens')
    return limit


def record_branch(pre_dir, dump, out):
    """Bind an offline Dec-2 graph dump to a fully audited shared-pre run."""
    if out.exists():
        raise ValueError("Refusing to overwrite an existing branch record")
    pre_manifest, rows = load_run(pre_dir, "off")
    if (pre_manifest.get("phase") != "shared_pre" or pre_manifest["start"] != DEFAULT_START
            or pre_manifest["days"] != PRE_POLICY_DAYS):
        raise ValueError("Branch source must be the completed 14-day shared-pre run")
    if len(rows) != PRE_POLICY_DAYS * len(pre_manifest["cohort_ids"]):
        raise ValueError("Shared-pre evidence is incomplete")
    if not dump.is_file() or dump.is_symlink():
        raise ValueError("Branch dump must be an existing regular file")
    record = {"schema_version": 1, "kind": "no_smoking_shared_pre_branch",
              "source_run_dir": str(pre_dir.resolve()), "source_run_id": pre_manifest["run_id"],
              "source_manifest_sha256": file_hash(pre_dir / "experiment_run.json"),
              "source_runtime_sha256": pre_manifest["runtime_sha256"],
              "source_runtime_file_sha256": pre_manifest["runtime_file_sha256"],
              "cohort_ids_sha256": digest(pre_manifest["cohort_ids"]),
              "pre_start": DEFAULT_START, "pre_end": (EFFECTIVE_DATE - timedelta(days=1)).isoformat(),
              "post_start": EFFECTIVE_DATE.isoformat(), "post_days": POST_POLICY_DAYS,
              "snapshot_sha256": file_hash(dump), "snapshot_bytes": dump.stat().st_size,
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(out, record)
    return record


def validate_branch_record(path, runtime, snapshot_id):
    branch = read_json(path)
    if (branch.get("schema_version") != 1 or branch.get("kind") != "no_smoking_shared_pre_branch"
            or branch.get("pre_start") != DEFAULT_START
            or branch.get("pre_end") != (EFFECTIVE_DATE - timedelta(days=1)).isoformat()
            or branch.get("post_start") != EFFECTIVE_DATE.isoformat()
            or branch.get("post_days") != POST_POLICY_DAYS
            or branch.get("snapshot_sha256") != snapshot_id
            or branch.get("source_runtime_sha256") != digest(runtime)
            or branch.get("cohort_ids_sha256") != digest([p["id"] for p in runtime["cohort"]])):
        raise ValueError("Branch record disagrees with the frozen roster, dates, runtime or restored dump")
    pre_dir = Path(branch["source_run_dir"])
    if file_hash(pre_dir / "experiment_run.json") != branch.get("source_manifest_sha256"):
        raise ValueError("Shared-pre manifest changed after branch record creation")
    pre_manifest, _ = load_run(pre_dir, "off")
    if (pre_manifest.get("phase") != "shared_pre" or pre_manifest.get("run_id") != branch.get("source_run_id")
            or pre_manifest.get("runtime_file_sha256") != branch.get("source_runtime_file_sha256")):
        raise ValueError("Branch source run identity changed")
    return branch


def run(args):
    report = inspect_bundle(args.bundle)
    if report["blockers"]:
        raise ValueError("Preparation blocked: " + "; ".join(report["blockers"]))
    if args.days <= 0 or args.workers <= 0:
        raise ValueError("days/workers must be positive")
    phase = getattr(args, "phase", "standalone")
    branch_path = getattr(args, "branch_manifest", None)
    if phase == "shared_pre":
        if args.arm != "off" or args.start != DEFAULT_START or args.days != PRE_POLICY_DAYS or branch_path:
            raise ValueError("Shared pre must run OFF for exactly 2017-11-19 through 2017-12-02")
    elif phase == "post_branch":
        if args.start != EFFECTIVE_DATE.isoformat() or args.days != POST_POLICY_DAYS or not branch_path:
            raise ValueError("Post branch needs the recorded Dec-2 snapshot and exactly 14 post days")
    elif phase != "standalone" or branch_path:
        raise ValueError("Unexpected phase or branch record")
    if phase in {"shared_pre", "post_branch"}:
        hook = Path(os.environ.get("SIM_POST_DAY_BACKUP_HOOK", ""))
        if not hook.is_file() or hook.is_symlink():
            raise ValueError("Shared-pre and post-branch runs require a regular post-day offsite backup hook")
    start = date.fromisoformat(args.start)
    window = simulation_window(args.start, args.days)
    resume = getattr(args, 'resume', False)
    previous = read_json(args.out / 'experiment_run.json') if resume else None
    if args.out.exists() and not resume:
        raise ValueError("Refusing to mix with an existing run directory")
    snapshot_id = os.environ.get("NO_SMOKING_SNAPSHOT_SHA256", "")
    if len(snapshot_id) != 64 or any(c not in "0123456789abcdef" for c in snapshot_id):
        raise ValueError("Set NO_SMOKING_SNAPSHOT_SHA256 to the verified clean snapshot file hash")
    runtime = read_json(args.bundle / "runtime.json")
    branch = validate_branch_record(branch_path, runtime, snapshot_id) if branch_path else None
    if branch and branch["source_runtime_file_sha256"] != file_hash(args.bundle / "runtime.json"):
        raise ValueError("Post branch bundle differs from the shared-pre bundle")
    db_env = graph_preflight(runtime, args.arm, snapshot_id, args.start, branch=branch,
                             **({'resume_manifest': previous} if resume else {}))
    sys.path.insert(0, str(ROOT / "scripts/sim"))
    from llm_client import healthcheck, resolve_mode
    mode = resolve_mode()
    health = healthcheck()
    if not health.get("served_match"):
        raise ValueError("LLM health/model check failed; verify explicit LLM_BASE_URL and LLM_MODE")
    server_config_path = os.environ.get("NO_SMOKING_SERVER_CONFIG")
    if not server_config_path:
        raise ValueError("NO_SMOKING_SERVER_CONFIG must reference the saved SGLang/model-server launch version/revision record")
    server_config = validate_server_config(read_json(server_config_path))
    if (os.environ.get('SIM_JSON_GRAMMAR_MODE') == 'json_object'
            and server_config.get('grammar_backend') != 'outlines'):
        raise ValueError('JSON-object mode requires the recorded Outlines server backend')
    context_length = server_context_length(server_config)
    from scripts.sim.prompt_budget import tokenizer_manifest
    if server_config.get('model_revision') != tokenizer_manifest()['revision']:
        raise ValueError('Server model revision differs from the frozen tokenizer/model revision')
    seed = runtime["simulation_seed"]
    env = dict(os.environ, **db_env, SIM_OUTPUT_DIR=str(args.out.resolve()),
               SIM_NO_SMOKING_MANIFEST=str((args.bundle / "runtime.json").resolve()),
               SIM_NO_SMOKING_ARM=args.arm, LLM_MODE=mode, PYTHONHASHSEED=str(seed % 4294967296))
    # Do not inherit another policy environment from the calling session.
    env.pop("SIM_ENVIRONMENT", None)
    env["SIM_RUN_ID"] = (branch["source_run_id"] if branch else
                         previous["run_id"] if resume else str(args.out.resolve()))
    env["SIM_FAST_MODE"] = "off"
    env["SIM_PROMPT_VARIANT"] = "no_smoking_v1"
    env["SIM_INTERVIEW_EVIDENCE"] = "required"
    env["SIM_PROMPT_TOKEN_GUARD"] = "required"
    env["SIM_MODEL_CONTEXT_LENGTH"] = str(context_length)
    tokenizer_path = Path(env.get('SIM_TOKENIZER_PATH') or ROOT / 'output/experiments/no_smoking_zone/runtime/tokenizer')
    from scripts.sim.prompt_budget import verify_tokenizer_files
    verify_tokenizer_files(tokenizer_path)
    env['SIM_TOKENIZER_PATH'] = str(tokenizer_path.resolve())
    args.out.mkdir(parents=True, exist_ok=resume)
    command = [sys.executable, str(ROOT / "scripts/sim/run_simulation.py"), "--start", args.start, "--days", str(args.days), "--workers", str(args.workers)]
    code_files = sorted((ROOT / "scripts/sim").rglob("*.py")) + [Path(__file__)]
    manifest = {"arm": args.arm, "phase": phase, "run_id": env["SIM_RUN_ID"], "runtime_sha256": digest(runtime),
                "prompt_contract": {"variant": "no_smoking_v1", "evidence_schema_version": 1,
                                    "quoted_rationale_required": True, "token_guard_required": True,
                                    "context_length": context_length, "token_margin": 128,
                                    "tokenizer_manifest_sha256": file_hash(DATA / 'tokenizer_manifest.json')},
                "runtime_file_sha256": file_hash(args.bundle / "runtime.json"), "snapshot_sha256": snapshot_id,
                "start": args.start, "days": args.days, "cohort_ids": [p["id"] for p in runtime["cohort"]],
                "simulation_window": window,
                "assignment_seed": runtime["assignment_seed"], "simulation_seed": seed,
                "model": health["active_model"], "workers": args.workers, "engine_settings": engine_settings(env),
                "reference_sha256": reference_hashes(),
                "server_config": server_config,
                "code_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in code_files},
                "interpretation": "Synthetic historical-policy scenario; not causal empirical replication", "status": "running"}
    if resume and previous.get("source_transition"):
        from scripts.sim.experience_provenance import source_fingerprint
        transition = previous["source_transition"]
        receipt = read_json(args.out / "backup_completed_2017-11-19.json")
        if (phase != "shared_pre" or args.start != "2017-11-19"
                or transition.get("effective_day") != "2017-11-20"
                or transition.get("inherited_day") != "2017-11-19"
                or transition.get("checkpoint_sha256") != receipt.get("checkpoint_sha256")
                or transition.get("previous_run_id") != env["SIM_RUN_ID"]
                or receipt.get("graph_sha256") != snapshot_id
                or transition.get("new_source_fingerprint") != source_fingerprint()
                or transition.get("new_server_config_sha256") != digest(server_config)):
            raise ValueError("Inherited first-day checkpoint does not match the source transition")
        manifest["source_transition"] = transition
    if branch:
        pre_manifest = read_json(Path(branch["source_run_dir"]) / "experiment_run.json")
        for key in ("model", "code_sha256", "reference_sha256", "engine_settings", "prompt_contract"):
            if manifest[key] != pre_manifest[key]:
                raise ValueError(f"Post branch differs from shared pre: {key}")
        manifest["branch"] = {"record_sha256": file_hash(branch_path),
                              "snapshot_sha256": snapshot_id,
                              "source_manifest_sha256": branch["source_manifest_sha256"],
                              "source_run_id": branch["source_run_id"]}
    if resume:
        for key, value in manifest.items():
            if key != 'status' and previous.get(key) != value:
                raise ValueError(f'Resume cannot change the frozen execution: {key}')
    write_json(args.out / "experiment_run.json", manifest)
    result = subprocess.run(command, env=env, cwd=ROOT, check=False)
    manifest.update(status="complete" if result.returncode == 0 else "failed", exit_code=result.returncode)
    write_json(args.out / "experiment_run.json", manifest)
    if result.returncode:
        raise RuntimeError(f"Simulation exited with code {result.returncode}; partial results cannot be scored")
    if phase in {"shared_pre", "post_branch"}:
        try:
            subprocess.run([sys.executable, str(hook), "--finalize", str(args.out.resolve())],
                           env=env, cwd=ROOT, check=True, timeout=1800)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            manifest.update(status="backup_failed", backup_stage="final_manifest")
            write_json(args.out / "experiment_run.json", manifest)
            raise RuntimeError("Final manifest backup failed; run cannot be scored") from exc
    return {"status": "complete", "output": str(args.out)}


def load_run(folder, expected_arm):
    from scripts.sim.evidence_integrity import verify
    manifest = read_json(folder / "experiment_run.json")
    if manifest["arm"] != expected_arm or manifest["status"] != "complete":
        raise ValueError("Expected a completed run of the requested arm")
    window = simulation_window(manifest["start"], manifest["days"])
    if "simulation_window" in manifest and manifest["simulation_window"] != window:
        raise ValueError("Run manifest has inconsistent simulation-window boundaries")
    expected = {(aid, (date.fromisoformat(manifest["start"]) + timedelta(days=i)).isoformat()) for aid in manifest["cohort_ids"] for i in range(manifest["days"])}
    rows = {}
    for path in sorted((folder / "metrics").glob("day_*.jsonl")):
        day = path.stem.removeprefix("day_")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            verify(row)
            key = (row["aid"], day)
            if key in rows or key not in expected or row.get("status") not in {"ok", "skipped"} or not row.get("no_smoking"):
                raise ValueError("Duplicate, unexpected or missing-arm metrics row")
            if row["status"] == "skipped" and (row.get("skip_kind") != "failed_after_retries"
                    or row.get("attempts") != 6 or row.get("observed_behavior") is not False
                    or row["no_smoking"].get("observed_behavior") is not False):
                raise ValueError("Skipped day lacks an explicit terminal failure receipt")
            if row["no_smoking"]["arm"] != expected_arm:
                raise ValueError("Mixed arm data")
            if row.get("experience_day") != day or row["no_smoking"].get("manifest_sha256") != manifest["runtime_file_sha256"]:
                raise ValueError("Metric day or runtime fingerprint mismatch")
            if row["no_smoking"].get("policy_active") != (expected_arm == "on" and day >= EFFECTIVE_DATE.isoformat()):
                raise ValueError("Metric policy activation mismatch")
            if row["status"] == "ok" and not isinstance(row["no_smoking"].get("by_poi"), list):
                raise ValueError("Successful agent day lacks purchase evidence")
            rows[key] = row
    if set(rows) != expected:
        raise ValueError(f"Incomplete run metrics: expected {len(expected)}, got {len(rows)}")
    return manifest, rows


def score_runs(off_dir, on_dir, ground_truth):
    off_manifest, off = load_run(off_dir, "off")
    on_manifest, on = load_run(on_dir, "on")
    return _score_loaded(off_manifest, on_manifest, off, on, ground_truth)


def score_shared_runs(pre_dir, off_dir, on_dir, branch_path, ground_truth):
    """Score post OFF/ON against one unchanged pre-period evidence set."""
    pre_manifest, pre = load_run(pre_dir, "off")
    off_manifest, off_post = load_run(off_dir, "off")
    on_manifest, on_post = load_run(on_dir, "on")
    branch = read_json(branch_path)
    branch_sha = file_hash(branch_path)
    if (pre_manifest.get("phase") != "shared_pre" or pre_manifest["start"] != DEFAULT_START
            or pre_manifest["days"] != PRE_POLICY_DAYS
            or any(m.get("phase") != "post_branch" or m["start"] != EFFECTIVE_DATE.isoformat()
                   or m["days"] != POST_POLICY_DAYS for m in (off_manifest, on_manifest))):
        raise ValueError("Shared score requires the exact 14+14 day phase windows")
    if (branch.get("kind") != "no_smoking_shared_pre_branch"
            or branch.get("source_manifest_sha256") != file_hash(pre_dir / "experiment_run.json")
            or branch.get("source_run_id") != pre_manifest["run_id"]
            or branch.get("source_runtime_sha256") != pre_manifest["runtime_sha256"]
            or branch.get("cohort_ids_sha256") != digest(pre_manifest["cohort_ids"])):
        raise ValueError("Shared-pre branch record is inconsistent")
    for manifest in (off_manifest, on_manifest):
        if (manifest.get("branch") != {"record_sha256": branch_sha,
                                      "snapshot_sha256": branch.get("snapshot_sha256"),
                                      "source_manifest_sha256": branch["source_manifest_sha256"],
                                      "source_run_id": pre_manifest["run_id"]}
                or manifest["snapshot_sha256"] != branch["snapshot_sha256"]
                or manifest["run_id"] != pre_manifest["run_id"]
                or manifest["cohort_ids"] != pre_manifest["cohort_ids"]):
            raise ValueError("Post run does not descend from the recorded shared-pre snapshot")
        for key in ("runtime_sha256", "runtime_file_sha256", "assignment_seed", "simulation_seed",
                    "model", "code_sha256", "reference_sha256", "engine_settings", "prompt_contract"):
            if manifest[key] != pre_manifest[key]:
                raise ValueError(f"Shared run changed {key}")
    if set(pre) & (set(off_post) | set(on_post)) or set(off_post) != set(on_post):
        raise ValueError("Shared and post periods have missing or overlapping agent-days")
    shared = {"source_run_dir": str(pre_dir.resolve()),
              "source_manifest_sha256": branch["source_manifest_sha256"],
              "branch_record_sha256": branch_sha,
              "branch_snapshot_sha256": branch["snapshot_sha256"],
              "executed_agent_days": len(pre) + len(off_post) + len(on_post)}
    if pre_manifest.get('source_transition'):
        shared['source_transition'] = pre_manifest['source_transition']
    return _score_loaded(off_manifest, on_manifest, pre | off_post, pre | on_post,
                         ground_truth, shared_pre=shared)


def _score_loaded(off_manifest, on_manifest, off, on, ground_truth, *, shared_pre=None):
    for key in ("runtime_sha256", "runtime_file_sha256", "snapshot_sha256", "start", "days", "cohort_ids", "assignment_seed", "simulation_seed", "model", "code_sha256", "reference_sha256", "server_config", "engine_settings", "workers"):
        if off_manifest[key] != on_manifest[key]:
            raise ValueError(f"Arms are not paired: {key} differs")
    window = simulation_window(DEFAULT_START, DEFAULT_DAYS) if shared_pre else simulation_window(off_manifest["start"], off_manifest["days"])
    if off_manifest.get('prompt_contract') != on_manifest.get('prompt_contract'):
        raise ValueError('Arms are not paired: prompt_contract differs')
    totals = {period: {arm: defaultdict(lambda: {"revenue_krw": 0, "payment_count": 0})
                      for arm in ("off", "on")} for period in ("pre", "post")}
    fallback = {}
    post_days = {key for key in off if key[1] >= EFFECTIVE_DATE.isoformat()}
    observed_post_pairs = {key for key in post_days
                           if off[key]["status"] == on[key]["status"] == "ok"}
    excluded_post_pairs = post_days - observed_post_pairs
    status_counts = {arm: dict(Counter(row["status"] for row in rows.values()))
                     for arm, rows in (("off", off), ("on", on))}
    for arm, rows in (("off", off), ("on", on)):
        fallback[arm] = dict(Counter({k: sum(r.get(k, 0) or 0 for r in rows.values()) for k in {k for r in rows.values() for k in r if k.startswith("fb_")}}))
        for (aid, day), row in rows.items():
            # A failed agent-day is missing data. Pairwise post comparisons
            # exclude both arms when either arm is missing, never count zero.
            if row["status"] == "skipped" or (day >= EFFECTIVE_DATE.isoformat()
                    and (aid, day) not in observed_post_pairs):
                continue
            # Both periods are reported, while the existing comparisons remain
            # post-rule paired contrasts and exclude every pre-rule receipt.
            period = "pre" if day < EFFECTIVE_DATE.isoformat() else "post"
            for p in row["no_smoking"]["by_poi"]:
                if p["district_code"] not in DISTRICTS or p["facility_type"] not in FACILITIES:
                    raise ValueError("Metrics contain a POI outside the registered study scope")
                facility = "indoor_golf" if p["facility_type"] == "screen_golf" else p["facility_type"]
                for group in (facility, facility + ":" + p["district_code"], facility + ":" + row["no_smoking"]["smoking_status"]):
                    for metric in ("revenue_krw", "payment_count"):
                        value = p[metric]
                        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 or int(value) != value:
                            raise ValueError("Invalid transaction amount/count")
                        totals[period][arm][group][metric] += value
    comparisons = []
    post = totals["post"]
    for group in sorted(set(post["off"]) | set(post["on"]) | {"billiard", "indoor_golf"}):
        for metric in ("revenue_krw", "payment_count"):
            a, b = post["off"][group][metric], post["on"][group][metric]
            comparisons.append({"group": group, "metric": metric, "off": a, "on": b, "on_minus_off": b-a,
                                "percent_change": 100*(b-a)/a if a else None, "status": "estimable" if a else "zero_baseline_not_estimable"})
    for period in totals.values():
        for arm in period.values():
            for group in ("billiard", "indoor_golf"):
                arm.setdefault(group, {"revenue_krw": 0, "payment_count": 0})
    limitations = ["No tolerance or exact-zero target is inferred from nonsignificant study estimates",
                   "The report uses a 2018 year dummy and adjusted monthly log-card-market outcomes; this is a paired synthetic ON/OFF contrast",
                   "Simulation payment counts are not unique visitors; current engine does not identify card versus cash",
                   "Skipped agent-days are missing data; post contrasts use only agent-days observed in both arms.",
                   "Single-seed output gives no seed uncertainty interval; freeze inputs and repeat seeds independently before interpretation"]
    if shared_pre and shared_pre.get('source_transition'):
        limitations.append("The inherited first shared-pre day used the previous code and serving grammar; later days use the recorded new source transition.")
    return {"status": "scenario_comparison_only", "empirical_validation_pass": None,
            "paired_agent_days": len(off),
            "observed_paired_post_agent_days": len(observed_post_pairs),
            "expected_post_agent_days": len(post_days),
            "excluded_post_agent_days": len(excluded_post_pairs),
            "agent_day_status_counts": status_counts,
            "assignment_seed": off_manifest["assignment_seed"], "simulation_seed": off_manifest["simulation_seed"],
            "simulation_window": window, "comparison_period": "post",
            "shared_pre_provenance": shared_pre,
            "period_totals": {period: {arm: dict(sorted(groups.items())) for arm, groups in arms.items()}
                              for period, arms in totals.items()},
            "model": off_manifest["model"], "comparisons": comparisons, "fallback_counts": fallback,
            "research_benchmarks": ground_truth["metrics"],
            "limitations": limitations}


def smoke(out):
    from scripts.sim.no_smoking_context import NoSmokingContext
    # Explicit test fixtures; never reused as observed POI or research outcomes.
    cohort = [{"id": "fixture", "smoking_status": "smoker"}]
    pois = [{"poi_id": "test_billiard", "district_code": "11650", "facility_type": "billiard"}]
    off = NoSmokingContext(arm="off", cohort=cohort, pois=pois, assignment_seed=20171203)
    on = NoSmokingContext(arm="on", cohort=cohort, pois=pois, assignment_seed=20171203)
    result = {"fixture_only": True, "llm_called": False, "database_written": False,
              "off_context": off.context_for("fixture", "2017-12-03"),
              "on_context": on.context_for("fixture", "2017-12-03"),
              "research_effect_claim": False}
    write_json(out, result)
    return {"status": "context_smoke_passed", "output": str(out), "live_simulation": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--agents", type=Path, default=ROOT / "agents.zip")
    p.add_argument("--pois", type=Path)
    p.add_argument("--eligible-ids", type=Path, help="Verified graph roster IDs; filter AFTER full-population smoking assignment")
    p.add_argument("--cohort-ids", type=Path, help="Explicit experiment roster IDs; select AFTER eligibility and BEFORE a pilot limit")
    p.add_argument("--seed", type=int, default=20171203)
    p.add_argument("--simulation-seed", type=int, default=17001)
    p.add_argument("--limit", type=int, help="Pilot-only subset within the selected experiment cohort; never reassigns smoking labels")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--bundle", type=Path, required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--arm", choices=("off", "on"), required=True)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=("standalone", "shared_pre", "post_branch"), default="standalone")
    p.add_argument('--resume', action='store_true', help='Resume this exact frozen run after verifying its provenance')
    p.add_argument("--branch-manifest", type=Path)
    p = sub.add_parser("record-branch")
    p.add_argument("--pre", type=Path, required=True, help="Completed shared-pre run")
    p.add_argument("--dump", type=Path, required=True, help="Offline Dec-2 Neo4j dump")
    p.add_argument("--out", type=Path, required=True, help="New immutable branch record")
    p = sub.add_parser("score")
    p.add_argument("--off", type=Path, required=True, help="OFF run root (contains experiment_run.json and metrics/)")
    p.add_argument("--on", type=Path, required=True, help="ON run root")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score-shared")
    p.add_argument("--pre", type=Path, required=True)
    p.add_argument("--off", type=Path, required=True)
    p.add_argument("--on", type=Path, required=True)
    p.add_argument("--branch-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args)
        elif args.command == "preflight":
            result = inspect_bundle(args.bundle)
        elif args.command == "run":
            result = run(args)
        elif args.command == "smoke":
            result = smoke(args.out)
        elif args.command == "record-branch":
            result = record_branch(args.pre, args.dump, args.out)
        elif args.command == "score-shared":
            result = score_shared_runs(args.pre, args.off, args.on, args.branch_manifest,
                                       read_json(DATA / "ground_truth.json"))
            write_json(args.out, result)
        else:
            result = score_runs(args.off, args.on, read_json(DATA / "ground_truth.json"))
            write_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if args.command == "preflight" and result["blockers"] else 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"No-smoking experiment: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
