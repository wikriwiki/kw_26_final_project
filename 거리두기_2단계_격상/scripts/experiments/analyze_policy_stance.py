"""Offline expressed-stance analysis. No LLM calls and no report-outcome targets.

Recorded self-reports are simulated opinions, not observed human attitudes.
Text clustering never uses smoking, demographic attributes or outcome benchmarks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.sim.stance_argument import validate_argument, public_argument_text

REPORTED_STANCES = ("support", "oppose", "mixed", "neutral", "uncertain", "unknown")
STANCES = REPORTED_STANCES + ("insufficient_evidence",)
SUBSTANTIVE = ("support", "oppose", "mixed", "neutral")
POLICY_DAY = "2017-12-03"
NOTE = "Simulated expressed stance; not verified human public opinion or a causal effect."


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def day(value):
    require(isinstance(value, str) and date.fromisoformat(value).isoformat() == value, "Invalid ISO day")
    return value


def sha_value(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def verify_seal(value):
    require(isinstance(value, dict), "Evidence packet must be an object")
    raw = dict(value)
    checksum = raw.pop("integrity_sha256", None)
    require(sha_value(checksum) and checksum == digest(raw), "Evidence packet integrity mismatch")


def validate_record(record, design=None):
    """Validate collector v1/v2 identity, quotations, public structure and time.

    Integrity is consistency, not authenticity. The collector must separately bind
    its packet to committed simulation artifacts and its LLM response to a call log.
    """
    require(isinstance(record, dict) and type(record.get("schema_version")) is int
            and record["schema_version"] in (1, 2), "Unsupported stance record")
    if "integrity_sha256" in record:
        verify_seal(record)
    for key in ("record_id", "run_id", "agent_id", "policy_id", "question_id"):
        require(isinstance(record.get(key), str) and bool(record[key].strip()), f"Missing {key}")
    require(record.get("arm") in {"off", "on"}, "Unknown arm")
    require(record.get("period") in {"pre", "post"}, "Unknown period")
    when = day(record.get("as_of_day"))
    policy_day = design["policy_effective_date"] if design else POLICY_DAY
    require(record["period"] == ("pre" if when < policy_day else "post"), "Period/date mismatch")
    context = "experienced" if record["arm"] == "on" and record["period"] == "post" else "hypothetical"
    require(record.get("measurement_context") == context, "Measurement context differs from arm/period")
    require(sha_value(record.get("question_sha256")), "Missing question hash")
    if "question_text" in record:
        require(isinstance(record["question_text"], str) and text_sha(record["question_text"]) == record["question_sha256"], "Question text hash mismatch")
    provenance = record.get("provenance")
    require(isinstance(provenance, dict), "Missing provenance")
    require(provenance.get("source") == "structured_policy_feedback", "Not an explicit policy self-report")
    for key in ("model_id", "call_id"):
        require(isinstance(provenance.get(key), str) and bool(provenance[key]), f"Missing provenance {key}")
    for key in ("prompt_sha256", "request_sha256", "response_sha256"):
        require(sha_value(provenance.get(key)), f"Missing provenance {key}")
    require(type(provenance.get("synthetic_fixture")) is bool, "Explicit fixture flag required")
    if record["schema_version"] == 2:
        require(type(provenance.get("argument_contract_version")) is int
                and provenance["argument_contract_version"] == 2, "Missing v2 argument contract provenance")
    else:
        require(provenance.get("argument_contract_version") in (None, 1), "Legacy response cannot claim a v2 argument contract")
    packet = record.get("evidence_packet")
    verify_seal(packet)
    require(packet.get("kind") == "grounded_interview_packet", "Unsupported evidence packet")
    for key in ("run_id", "arm", "agent_id"):
        require(packet.get(key) == record[key], f"Foreign packet {key}")
    for key in ("cohort_sha256", "source_sha256"):
        require(sha_value(packet.get(key)), f"Missing packet {key}")
    require(day(packet.get("through_day")) <= when, "Future evidence packet")
    require(packet["through_day"] == when, "Packet horizon differs from measurement date")
    require(isinstance(packet.get("days"), list), "Missing packet days")
    for item_day in packet.get("days", []):
        require(day(item_day) <= packet["through_day"], "Future packet day")
    evidence = {}
    require(isinstance(packet.get("evidence_items"), list), "Missing evidence items")
    for item in packet["evidence_items"]:
        require(isinstance(item, dict), "Evidence item must be an object")
        eid = item.get("evidence_id")
        require(isinstance(eid, str) and eid and eid not in evidence, "Invalid/duplicate evidence ID")
        require(day(item.get("day")) <= packet["through_day"], "Future evidence item")
        require(item["day"] in packet["days"], "Evidence item outside packet days")
        require(isinstance(item.get("text"), str), "Evidence text required")
        for key in ("run_id", "arm", "agent_id"):
            require(key not in item or item[key] == record[key], f"Foreign evidence {key}")
        evidence[eid] = item
    status = record.get("response_status")
    require(status in {"answered", "no_response", "error"}, "Unknown response status")
    require(status == "answered" or record.get("response") is None, "Failed/unanswered record must not contain an accepted response")
    if status == "answered":
        response = record.get("response")
        require(isinstance(response, dict) and response.get("stance") in REPORTED_STANCES, "Invalid self-reported stance")
        if record["schema_version"] == 2:
            quality = validate_argument(response, packet)
            if "argument_quality" in record:
                require(record["argument_quality"] == quality, "Stored argument quality differs from recomputed validation")
        else:
            require("argument" not in response, "A public argument requires schema v2; do not relabel or mix contracts")
        answer, quote = response.get("answer"), response.get("stance_quote")
        require(isinstance(answer, str) and answer.strip(), "Missing explicit response")
        require(isinstance(quote, str) and quote.strip() and quote in answer, "Stance quote must occur exactly in answer")
        confidence = response.get("confidence")
        require(type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1, "Invalid confidence")
        require(isinstance(response.get("reasons"), list), "Reasons list required")
        for reason in response["reasons"]:
            require(isinstance(reason, dict), "Reason must be an object")
            eid, quote = reason.get("evidence_id"), reason.get("quote")
            require(eid in evidence, "Unknown reason evidence ID")
            require(isinstance(quote, str) and quote.strip() and quote in evidence[eid]["text"], "Reason quote differs from evidence")
    if design:
        require(record["schema_version"] == measurement_contract(design)["record_schema_version"], "Response schema differs from frozen measurement contract; v1/v2 cannot be pooled")
        require(record["agent_id"] in design["population"], "Agent outside frozen population")
        require(record["run_id"] == design["runs"][record["arm"]], "Run differs from registered arm")
        for key in ("policy_id", "question_id", "question_sha256"):
            require(record[key] == design[key], f"Measurement {key} mismatch")
        window = design["periods"][record["period"]]
        require(window["start"] <= when <= window["end"], "Response outside period")
        require(provenance["synthetic_fixture"] == design["synthetic_fixture"], "Mixed fixture/real-run evidence")
        if "cohort_sha256" in design:
            require(packet["cohort_sha256"] == design["cohort_sha256"], "Packet cohort differs from design")
        if "source_sha256_by_arm" in design:
            require(packet["source_sha256"] == design["source_sha256_by_arm"][record["arm"]], "Packet source differs from design")
    return record


def derive_stance(record, min_confidence=0.6):
    """A declared stance with an exact quote is primary; behavior is never a vote."""
    require(0 <= min_confidence <= 1, "Invalid confidence threshold")
    if record is None:
        return {"stance": "insufficient_evidence", "reason": "no_response_record", "reported_stance": None, "confidence": None}
    validate_record(record)
    if record["response_status"] != "answered":
        return {"stance": "unknown" if record["response_status"] == "error" else "insufficient_evidence",
                "reason": record["response_status"], "reported_stance": None, "confidence": None}
    response = record["response"]
    low = response["confidence"] < min_confidence
    return {"stance": "insufficient_evidence" if low else response["stance"],
            "reason": "low_self_reported_confidence" if low else "explicit_quoted_self_report",
            "reported_stance": response["stance"], "confidence": response["confidence"]}


def validate_design(design):
    require(isinstance(design, dict) and design.get("schema_version") == 1, "Unsupported design")
    require(isinstance(design.get("population"), dict) and design["population"], "Frozen population required")
    require(all(isinstance(k, str) and k and isinstance(v, dict) for k, v in design["population"].items()), "Invalid population")
    require(isinstance(design.get("runs"), dict) and set(design["runs"]) == {"off", "on"}
            and all(isinstance(v, str) and v for v in design["runs"].values())
            and len(set(design["runs"].values())) == 2, "Distinct OFF/ON runs required")
    for key in ("policy_id", "question_id", "population_id", "population_role"):
        require(isinstance(design.get(key), str) and design[key], f"Missing design {key}")
    require(sha_value(design.get("question_sha256")), "Question hash required")
    require(type(design.get("synthetic_fixture")) is bool, "Explicit design fixture flag required")
    effective = day(design.get("policy_effective_date"))
    require(effective == POLICY_DAY, "This analyzer targets the registered 2017 smoking policy")
    require(set(design.get("periods", {})) == {"pre", "post"}, "Two periods required")
    for period, window in design["periods"].items():
        require(day(window["start"]) <= day(window["end"]), "Reversed period")
        require(window["end"] < effective if period == "pre" else window["start"] >= effective, "Period crosses policy boundary")
    require(day(design.get("group_attributes_as_of")) < design["periods"]["pre"]["start"], "Group attributes must be frozen before measurement")
    require(isinstance(design.get("group_by", []), list) and all(isinstance(k, str) and k for k in design.get("group_by", [])), "Invalid group columns")
    if "cohort_sha256" in design:
        require(design["cohort_sha256"] == digest(sorted(design["population"])), "Design cohort hash mismatch")
    if "source_sha256_by_arm" in design:
        require(set(design["source_sha256_by_arm"]) == {"off", "on"}
                and all(sha_value(v) for v in design["source_sha256_by_arm"].values()), "Invalid source hashes")
    contract = measurement_contract(design)
    require(isinstance(contract, dict) and type(contract.get("record_schema_version")) is int
            and (contract.get("argument_contract_version") is None or type(contract["argument_contract_version"]) is int)
            and contract in ({"record_schema_version": 1, "argument_contract_version": None},
                         {"record_schema_version": 2, "argument_contract_version": 2}), "Unsupported measurement contract")
    return design


def measurement_contract(design):
    # A legacy design remains v1; new v2 evidence never silently changes it.
    return design.get("measurement_contract", {"record_schema_version": 1, "argument_contract_version": None})


def select_records(records, design):
    """Latest actual answer within each arm/period; never carry post data backward."""
    validate_design(design)
    by_id, selected, dated = {}, {}, {}
    for record in records:
        validate_record(record, design)
        rid = record["record_id"]
        require(rid not in by_id or by_id[rid] == record, "Conflicting response ID")
        if rid in by_id:
            continue
        by_id[rid] = record
        key = (record["arm"], record["period"], record["agent_id"])
        day_key = key + (record["as_of_day"],)
        require(day_key not in dated, "Multiple assessments on the same agent/arm/day; resolve explicitly upstream")
        dated[day_key] = rid
        if key not in selected or record["as_of_day"] > selected[key]["as_of_day"]:
            selected[key] = record
    return selected


def response_text(record):
    if not record or record["response_status"] != "answered":
        return ""
    response = record["response"]
    if record["schema_version"] == 2:
        return public_argument_text(response)
    # Only cited as-of evidence, not the complete behavioral history, enters text features.
    return response["answer"] + " " + " ".join(reason["quote"] for reason in response["reasons"])


def tokens(text):
    words = re.findall(r"[a-z가-힣]{2,}", unicodedata.normalize("NFKC", text).casefold())
    result = ["w:" + word for word in words]
    for word in words:
        if re.search(r"[가-힣]", word):
            result.extend("c:" + word[i:i + 3] for i in range(len(word) - 2))
    return Counter(result)


def unit(vector):
    norm = math.sqrt(sum(value * value for value in vector.values()))
    return {key: value / norm for key, value in vector.items()} if norm else {}


def vectorize(text, vocabulary, idf):
    counts = tokens(text)
    return unit({vocabulary[token]: (1 + math.log(count)) * idf[vocabulary[token]]
                 for token, count in counts.items() if token in vocabulary})


def cosine(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0) for key, value in left.items())


def fit_model(records, design, *, k=4, seed=17001, max_features=1024, min_df=2,
              max_iter=40, min_similarity=0.1):
    """Unsupervised spherical k-means trained ONLY on the OFF/pre partition."""
    require(1 <= k <= 20 and 16 <= max_features <= 10000 and min_df >= 1 and 1 <= max_iter <= 200, "Invalid clustering settings")
    require(0 <= min_similarity <= 1, "Invalid minimum similarity")
    selected = select_records(records, design)
    training = sorted((r for (arm, period, _), r in selected.items()
                       if arm == "off" and period == "pre" and response_text(r)), key=lambda r: r["record_id"])
    document_frequency = Counter(token for record in training for token in tokens(response_text(record)))
    terms = sorted(document_frequency, key=lambda token: (-document_frequency[token], token))
    terms = [t for t in terms if document_frequency[t] >= min_df][:max_features]
    vocabulary = {term: i for i, term in enumerate(terms)}
    idf = [math.log((1 + len(training)) / (1 + document_frequency[t])) + 1 for t in terms]
    documents = [(r["record_id"], vectorize(response_text(r), vocabulary, idf)) for r in training]
    documents = [(rid, vector) for rid, vector in documents if vector]
    unique = {}
    for rid, vector in documents:
        unique.setdefault(canonical(sorted(vector.items())), (rid, vector))
    candidates = list(unique.values())
    effective_k = min(k, len(candidates))
    model = {
        "schema_version": 1, "algorithm": "tfidf_spherical_kmeans_v1", "note": NOTE,
        "design_sha256": digest(design), "fit_partition": {"arm": "off", "period": "pre"},
        "fit_max_day": max((r["as_of_day"] for r in training), default=None),
        "fit_record_ids": [r["record_id"] for r in training], "fit_records_sha256": digest(training),
        "fit_documents": len(documents), "seed": seed, "requested_k": k, "effective_k": effective_k,
        "settings": {"max_features": max_features, "min_df": min_df, "max_iter": max_iter,
                     "min_similarity": min_similarity, "initialization": "seed-hash first, then farthest-first cosine",
                     "features": "answer and cited evidence; v2 additionally public personal claims, considerations, relevance, weighing, conditions and uncertainty; word tokens and Korean character trigrams; numeric tokens excluded",
                     "excluded_features": ["declared stance label field", "argument direction and basis fields", "confidence field", "quality flag fields", "raw demographic metadata columns", "raw smoking-status metadata column", "ground truth", "post-period responses"],
                     "text_caveat": "Voluntary demographic/smoking mentions in answers or cited quotes remain semantic text; no metadata-to-stance mapping is applied."},
        "terms": terms, "idf": idf, "centroids": [], "converged": False, "iterations": 0,
        "status": "insufficient_training_responses" if len(documents) < 2 or effective_k == 0 else "fitted",
    }
    if model["status"] != "fitted":
        model["effective_k"] = 0
        return {**model, "model_sha256": digest(model)}
    first = min(candidates, key=lambda item: (text_sha(f"{seed}:{item[0]}"), item[0]))
    centroids = [first[1]]
    remaining = [item for item in candidates if item[0] != first[0]]
    while len(centroids) < effective_k:
        chosen = min(remaining, key=lambda item: (max(cosine(item[1], c) for c in centroids), item[0]))
        centroids.append(chosen[1])
        remaining.remove(chosen)
    prior = None
    for iteration in range(max_iter):
        labels = [max(range(len(centroids)), key=lambda c: (cosine(vector, centroids[c]), -c)) for _, vector in documents]
        model["iterations"] = iteration + 1
        if labels == prior:
            model["converged"] = True
            break
        prior = labels
        accumulators = [defaultdict(float) for _ in centroids]
        for (_, vector), label in zip(documents, labels):
            for key, value in vector.items():
                accumulators[label][key] += value
        centroids = [unit(acc) if acc else centroids[i] for i, acc in enumerate(accumulators)]
    model["centroids"] = [{str(key): value for key, value in centroid.items()} for centroid in centroids]
    labels = [max(range(len(centroids)), key=lambda c: (cosine(vector, centroids[c]), -c)) for _, vector in documents]
    model["mean_training_cosine_distance"] = sum(1 - max(cosine(v, c) for c in centroids) for _, v in documents) / len(documents)
    model["training_cluster_sizes"] = dict(sorted(Counter(str(label) for label in labels).items()))
    model["empty_cluster_ids"] = [str(i) for i in range(len(centroids)) if i not in labels]
    return {**model, "model_sha256": digest(model)}


def verify_model(model, design):
    body = {key: value for key, value in model.items() if key != "model_sha256"}
    require(model.get("model_sha256") == digest(body), "Cluster model integrity mismatch")
    require(model.get("design_sha256") == digest(design), "Cluster model belongs to another design")
    require(model.get("fit_partition") == {"arm": "off", "period": "pre"}, "Only frozen OFF/pre training is allowed")
    require(model.get("fit_max_day") is None or model["fit_max_day"] < design["policy_effective_date"], "Post-data leaked into cluster fit")


def assign_cluster(record, model):
    if model["status"] != "fitted":
        return {"cluster_id": None, "cluster_status": "unfitted_model", "cosine_similarity": None}
    text = response_text(record)
    if not text:
        return {"cluster_id": None, "cluster_status": "no_response_text", "cosine_similarity": None}
    vocabulary = {term: i for i, term in enumerate(model["terms"])}
    vector = vectorize(text, vocabulary, model["idf"])
    if not vector:
        return {"cluster_id": None, "cluster_status": "no_reference_vocabulary", "cosine_similarity": None}
    centroids = [{int(key): value for key, value in c.items()} for c in model["centroids"]]
    chosen = max(range(len(centroids)), key=lambda c: (cosine(vector, centroids[c]), -c))
    similarity = cosine(vector, centroids[chosen])
    if similarity < model["settings"]["min_similarity"]:
        return {"cluster_id": None, "cluster_status": "low_reference_similarity", "cosine_similarity": similarity}
    return {"cluster_id": str(chosen), "cluster_status": "assigned", "cosine_similarity": similarity}


def stance_summary(rows, min_group_size=5):
    counts = {stance: sum(row["stance"] == stance for row in rows) for stance in STANCES}
    n = len(rows)
    resolved = sum(counts[label] for label in SUBSTANTIVE)
    unresolved = n - resolved
    return {"agents": n, "resolved_agents": resolved, "unresolved_agents": unresolved,
            "small_group": n < min_group_size, "stance_counts": counts,
            "shares_all_expected": {label: count / n for label, count in counts.items()} if n else {},
            "shares_among_resolved": {label: counts[label] / resolved for label in SUBSTANTIVE} if resolved else {},
            "support_missingness_bounds": {"lower": counts["support"] / n, "upper": (counts["support"] + unresolved) / n} if n else None,
            "opposition_missingness_bounds": {"lower": counts["oppose"] / n, "upper": (counts["oppose"] + unresolved) / n} if n else None,
            "argument_quality": quality_summary(rows),
            "uncertainty_note": "Bounds vary unresolved stances, not sampling error or calibrated human-prediction uncertainty; self-reported confidence is uncalibrated."}


def argument_quality(record):
    if not record or record["response_status"] != "answered":
        return {"quality_status": "not_observed", "quality_flags": [], "review_required": True}
    if record["schema_version"] == 1:
        return {"quality_status": "legacy_not_assessed", "quality_flags": [], "review_required": True}
    return validate_argument(record["response"], record["evidence_packet"])


def quality_summary(rows):
    qualities = [row.get("argument_quality", {"quality_status": "not_observed", "quality_flags": []}) for row in rows]
    return {"expected_agents": len(rows),
            "status_counts": dict(sorted(Counter(q["quality_status"] for q in qualities).items())),
            "flagged_agent_counts": dict(sorted(Counter(flag for q in qualities for flag in set(q["quality_flags"])).items())),
            "stance_counts_by_quality": {status: dict(Counter(row["stance"] for row, q in zip(rows, qualities) if q["quality_status"] == status))
                                         for status in sorted({q["quality_status"] for q in qualities})},
            "limited_input_possible_agents": sum(q.get("limited_input_possible") is True for q in qualities),
            "semantic_review_completed": False,
            "note": "Public-structure flags are separate from stance; no quality-based exclusion, neutral substitution or semantic-truth claim."}


def estimand(design):
    result = {"measure": "simulated_explicit_policy_stance", "unit": "agent_period_latest_response",
            "policy_id": design["policy_id"], "question_id": design["question_id"],
            "question_sha256": design["question_sha256"], "population_id": design["population_id"],
            "population_role": design["population_role"], "periods": design["periods"],
            "weighting": "unweighted_frozen_cohort", "hypothetical_vs_experienced_kept_separate": True}
    if measurement_contract(design)["record_schema_version"] == 2:
        result["measurement_contract"] = measurement_contract(design)
    return result


def analyze(records, design, model, *, min_confidence=0.6, min_group_size=5):
    selected = select_records(records, design)
    verify_model(model, design)
    require(min_group_size >= 1, "Invalid minimum group size")
    rows, groups, clusters, transitions = [], [], [], []
    for arm in ("off", "on"):
        for period in ("pre", "post"):
            cell = []
            for aid in sorted(design["population"]):
                record = selected.get((arm, period, aid))
                row = {"agent_id": aid, "arm": arm, "period": period,
                       "run_id": design["runs"][arm], "record_id": record["record_id"] if record else None,
                       "as_of_day": record["as_of_day"] if record else None,
                       "measurement_context": "experienced" if arm == "on" and period == "post" else "hypothetical",
                       **derive_stance(record, min_confidence), **assign_cluster(record, model),
                       "argument_quality": argument_quality(record)}
                row["stance_quote"] = record["response"]["stance_quote"] if record and record["response_status"] == "answered" else None
                packet = record["evidence_packet"] if record else {}
                row["evidence_coverage"] = {"missing_days": packet.get("missing_days", []),
                                            "missing_night_days": packet.get("missing_night_days", []),
                                            "selection": packet.get("selection"),
                                            "packet_sha256": packet.get("integrity_sha256")}
                rows.append(row)
                cell.append(row)
            for group_by in ["all"] + design.get("group_by", []):
                buckets = defaultdict(list)
                for row in cell:
                    value = "all" if group_by == "all" else str(design["population"][row["agent_id"]].get(group_by, "unknown"))
                    buckets[value].append(row)
                for value, members in sorted(buckets.items()):
                    groups.append({"arm": arm, "period": period, "group_by": group_by, "group": value,
                                   "measurement_context": members[0]["measurement_context"], **stance_summary(members, min_group_size)})
            for cluster_id in [str(i) for i in range(model["effective_k"])] + [None]:
                members = [r for r in cell if r["cluster_id"] == cluster_id]
                top_terms = []
                if cluster_id is not None:
                    centroid = model["centroids"][int(cluster_id)]
                    top_terms = [{"term": model["terms"][int(key)], "weight": value}
                                 for key, value in sorted(centroid.items(), key=lambda p: (-p[1], int(p[0])))[:10]]
                exemplars = sorted((r for r in members if r["record_id"]), key=lambda r: (-(r["cosine_similarity"] or 0), r["agent_id"]))[:3]
                clusters.append({"arm": arm, "period": period, "cluster_id": cluster_id,
                                 "measurement_context": "experienced" if arm == "on" and period == "post" else "hypothetical",
                                 "top_reference_terms": top_terms, "label_source": "descriptive terms; no invented semantic cluster names",
                                 "frozen_attribute_counts": {key: dict(sorted(Counter(str(design["population"][r["agent_id"]].get(key, "unknown")) for r in members).items())) for key in design.get("group_by", [])},
                                 "exemplars": [{"record_id": r["record_id"], "stance_quote": r["stance_quote"]} for r in exemplars],
                                 "unassigned_reasons": dict(Counter(r["cluster_status"] for r in members)) if cluster_id is None else {},
                                 **stance_summary(members, min_group_size)})
    lookup = {(r["arm"], r["period"], r["agent_id"]): r for r in rows}
    for arm in ("off", "on"):
        counts = Counter((lookup[arm, "pre", aid]["stance"], lookup[arm, "post", aid]["stance"]) for aid in design["population"])
        transitions.append({"arm": arm, "counts": [{"pre": pre, "post": post, "agents": count} for (pre, post), count in sorted(counts.items())],
                            "note": "Within-simulator longitudinal response transitions; unresolved states remain separate."})
    return {"schema_version": 1, "kind": "policy_stance_analysis", "note": NOTE, "estimand": estimand(design),
            "measurement_contract": measurement_contract(design), "argument_quality": quality_summary(rows),
            "synthetic_fixture": design["synthetic_fixture"], "design_sha256": digest(design),
            "model_sha256": model["model_sha256"], "input_records_sha256": digest(sorted(records, key=lambda r: r["record_id"])),
            "classification": {"method": "exact_quoted_explicit_self_report", "min_confidence": min_confidence,
                               "threshold_empirically_validated": False, "neutral_is_not_missing": True},
            "cluster_fit": {key: model[key] for key in ("algorithm", "fit_partition", "fit_max_day", "fit_documents", "effective_k", "status", "converged")},
            "coverage": {"expected_agent_periods": len(rows), "selected_actual_records": len(selected),
                         "missing_agent_periods": len(rows) - len(selected)},
            "agents": rows, "groups": groups, "clusters": clusters, "transitions": transitions,
            "ground_truth_evaluation": {"status": "abstained", "reason": "Independent matching stance labels/statistics have not been supplied"}}


def _abstain(reason):
    return {"status": "abstained", "reason": reason, "metrics": None}


def evaluate(report, truth=None):
    """Independent label or same-estimand group evaluation; never derives PDF labels."""
    if truth is None:
        return _abstain("No independent stance reference supplied")
    if truth.get("kind") not in {"individual_stance_labels", "group_stance_statistics"}:
        return _abstain("Reference is not stance labels/statistics; revenue regression, sales perceptions and exposure measurements are different constructs")
    if truth.get("independent") is not True or truth.get("partition_role") != "holdout":
        return _abstain("Independent holdout declaration is required; fitting/calibration labels cannot serve as evaluation")
    source = truth.get("source", {})
    if not source.get("id") or not source.get("annotation_or_measurement_protocol") or not sha_value(source.get("sha256")):
        return _abstain("Reference source hash and annotation/measurement protocol are required")
    if truth.get("estimand") != report["estimand"]:
        return _abstain("Question, population/role, periods, unit, weighting or construct differs from the registered estimand")
    if truth.get("synthetic_fixture") != report["synthetic_fixture"]:
        return _abstain("Fixture and actual-run references cannot be combined")
    if truth["kind"] == "individual_stance_labels":
        if truth.get("label_basis") != "independent_annotation_of_same_responses":
            return _abstain("Individual scores require independent annotations of these exact response records; synthetic agents are not matched observed humans")
        classes = truth.get("classes")
        if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes) or any(c not in STANCES for c in classes):
            return _abstain("Explicit nonduplicate label classes required")
        actual = {(r["arm"], r["period"], r["agent_id"]): r for r in report["agents"]}
        labels = truth.get("labels")
        if not isinstance(labels, list) or not labels:
            return _abstain("No independent labels supplied")
        pairs, seen = [], set()
        for row in labels:
            key = (row.get("arm"), row.get("period"), row.get("agent_id"))
            if key in seen or key not in actual or row.get("stance") not in classes:
                return _abstain("Duplicate, foreign or unsupported label case")
            seen.add(key)
            predicted = actual[key]
            if not row.get("record_id") or row["record_id"] != predicted["record_id"]:
                return _abstain("Gold annotation does not refer to the exact selected response record")
            if row.get("measurement_context") != predicted["measurement_context"]:
                return _abstain("Hypothetical and experienced response contexts differ")
            pairs.append((row["stance"], predicted["stance"]))
        confusion = {label: {predicted: 0 for predicted in STANCES} for label in classes}
        for label, prediction in pairs:
            confusion[label][prediction] += 1
        per_class = {}
        for label in classes:
            tp = confusion[label][label]
            support = sum(confusion[label].values())
            predicted_count = sum(confusion[actual][label] for actual in classes)
            precision = tp / predicted_count if predicted_count else 0.0
            recall = tp / support if support else 0.0
            per_class[label] = {"support": support, "precision": precision, "recall": recall,
                                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
        present = [v for v in per_class.values() if v["support"]]
        covered = sum(prediction in classes for _, prediction in pairs)
        return {"status": "evaluated_independent_response_annotations", "source": source,
                "scope": "Extraction agreement on annotated simulation responses, not human-opinion prediction accuracy",
                "independence": "Declared by reference supplier; hashes do not prove annotator independence",
                "n": len(pairs), "coverage": covered / len(pairs),
                "unresolved_prediction_count": sum(pred not in SUBSTANTIVE for _, pred in pairs),
                "metrics": {"macro_f1": sum(v["f1"] for v in per_class.values()) / len(classes),
                            "balanced_accuracy": sum(v["recall"] for v in present) / len(present),
                            "accuracy": sum(a == p for a, p in pairs) / len(pairs)},
                "confusion_matrix": confusion, "per_class": per_class, "macro_classes": classes,
                "zero_division": 0, "balanced_accuracy_classes": [c for c in classes if per_class[c]["support"]],
                "abstentions_count_as_errors_when_gold_is_substantive": True}
    rows = truth.get("groups")
    if not isinstance(rows, list) or not rows:
        return _abstain("No matching group statistics supplied")
    lookup = {(r["arm"], r["period"], r["group_by"], r["group"]): r for r in report["groups"]}
    comparisons, seen = [], set()
    for row in rows:
        key = (row.get("arm"), row.get("period"), row.get("group_by"), row.get("group"))
        if key not in lookup or key in seen:
            return _abstain("Missing or duplicate matching group")
        seen.add(key)
        actual = lookup[key]
        if row.get("measurement_context") != actual["measurement_context"]:
            return _abstain("Group measurement contexts differ")
        denominator = row.get("denominator")
        labels = STANCES if denominator == "all_expected" else SUBSTANTIVE if denominator == "resolved_responses" else None
        n, counts = row.get("n"), row.get("counts")
        if labels is None or type(n) is not int or n <= 0 or not isinstance(counts, dict):
            return _abstain("Explicit positive denominator and count categories required")
        if set(counts) != set(labels) or any(type(v) is not int or v < 0 for v in counts.values()) or sum(counts.values()) != n:
            return _abstain("Reference counts and denominator disagree")
        measured = actual["agents"] if denominator == "all_expected" else actual["resolved_agents"]
        if measured == 0 or actual["small_group"]:
            return _abstain("Matching simulation group is empty, unresolved or below minimum group size")
        predicted = actual["shares_all_expected"] if denominator == "all_expected" else actual["shares_among_resolved"]
        differences = {label: predicted[label] - counts[label] / n for label in labels}
        comparisons.append({"arm": key[0], "period": key[1], "group_by": key[2], "group": key[3],
                            "denominator": denominator, "simulation_n": measured, "reference_n": n,
                            "share_difference": differences,
                            "total_variation_distance": sum(abs(v) for v in differences.values()) / 2,
                            "mean_absolute_share_error": sum(abs(v) for v in differences.values()) / len(labels)})
    return {"status": "evaluated_matching_group_statistics", "source": source, "comparisons": comparisons,
            "note": "Descriptive same-estimand differences only; simulation-vs-human construct changes require a different validated measurement design."}


def design_from_bundle(bundle, *, off_run, on_run, question_id, question_text, synthetic_fixture=False):
    """Freeze descriptive attributes before intervention; they are not clustering inputs."""
    bundle = Path(bundle)
    runtime = json.loads((bundle / "runtime.json").read_text(encoding="utf-8"))
    people = json.loads((bundle / "personas.json").read_text(encoding="utf-8"))
    if isinstance(people, dict):
        people = people.get("agents", people.get("personas", []))
    by_id = {p.get("agent_id", p.get("id")): p for p in people}
    population = {}
    for row in runtime["cohort"]:
        age = row.get("age")
        band = "unknown" if type(age) not in (int, float) or not math.isfinite(age) or age < 0 else "under19" if age < 19 else "70+" if age >= 70 else f"{int(age)//10*10}s"
        person = by_id.get(row["id"], {})
        population[row["id"]] = {"age_band": band, "sex": row.get("sex", "unknown"),
                                  "income": person.get("personal", {}).get("income_level") or "unknown",
                                  "smoking_status": row["smoking_status"]}
    design = {"schema_version": 1, "policy_id": "indoor_sports_smoking_ban_2017",
              "question_id": question_id, "question_sha256": text_sha(question_text),
              "population_id": digest(sorted(population)), "population_role": "synthetic_residents",
              "population": population, "runs": {"off": off_run, "on": on_run},
              "policy_effective_date": POLICY_DAY,
              "periods": {"pre": {"start": "2017-11-19", "end": "2017-12-02"},
                          "post": {"start": "2017-12-03", "end": "2017-12-16"}},
              "group_attributes_as_of": "2017-11-18", "group_by": ["age_band", "sex", "income", "smoking_status"],
              "synthetic_fixture": synthetic_fixture,
              "attribute_note": "Frozen synthetic baseline attributes; not observed historical personal health data"}
    return validate_design(design)


def design_from_runs(bundle, off_dir, on_dir):
    """Use registered complete 28-day runs and the collector's pinned question.

    A first-day committed packet establishes the archived run identity, so moving
    a run directory cannot silently change that identity. Full packet validation
    is still done by the collector for every interviewed agent/as-of date.
    """
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from scripts.experiments.collect_policy_stances import QUESTION_ID, QUESTION, RECORD_SCHEMA_VERSION, ARGUMENT_CONTRACT_VERSION
    from scripts.experiments.no_smoking_zone import inspect_bundle
    from scripts.sim.interview_evidence import build_packet
    bundle = Path(bundle)
    preflight = inspect_bundle(bundle)
    require(not preflight.get("blockers"), "Bundle preflight failed")
    runtime = read_json(bundle / "runtime.json")
    ids = sorted(row["id"] for row in runtime["cohort"])
    packets, manifests = {}, {}
    for arm, folder in (("off", Path(off_dir)), ("on", Path(on_dir))):
        manifest = read_json(folder / "experiment_run.json")
        require(manifest.get("arm") == arm and manifest.get("status") == "complete", "Complete OFF/ON run manifests required")
        require(manifest.get("start") == "2017-11-19" and manifest.get("days") == 28, "28-day registered analysis window required; one-day throughput pilots cannot supply pre/post attitudes")
        require(sorted(manifest.get("cohort_ids", [])) == ids, "Run cohort differs from frozen bundle")
        require(manifest.get("runtime_sha256") == digest(runtime), "Run runtime differs from frozen bundle")
        packet = build_packet(folder, ids[0], manifest["start"], days=[manifest["start"]])
        verify_seal(packet)
        require(packet["arm"] == arm and packet["cohort_sha256"] == digest(ids), "Archived packet identity mismatch")
        packets[arm], manifests[arm] = packet, manifest
    require(packets["off"]["source_sha256"] == packets["on"]["source_sha256"], "Paired run simulator sources differ")
    design = design_from_bundle(bundle, off_run=packets["off"]["run_id"], on_run=packets["on"]["run_id"],
                                question_id=QUESTION_ID, question_text=QUESTION)
    design.update(cohort_sha256=digest(ids), source_sha256_by_arm={arm: packet["source_sha256"] for arm, packet in packets.items()},
                  measurement_contract={"record_schema_version": RECORD_SCHEMA_VERSION, "argument_contract_version": ARGUMENT_CONTRACT_VERSION},
                  run_manifest_sha256={arm: digest(value) for arm, value in manifests.items()},
                  frozen_bundle_sha256={name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in ("runtime.json", "personas.json", "bundle.json")})
    return validate_design(design)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_records(paths):
    result = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            # Accept either a collector root or its records directory, without
            # accidentally ingesting request logs or manifests as responses.
            directory = path / "records" if (path / "records").is_dir() else path
            files = sorted(directory.rglob("*.json")) + sorted(directory.rglob("*.jsonl"))
            require(bool(files), "No stance records in directory")
            result.extend(load_records(files))
            continue
        if path.suffix == ".jsonl":
            data = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        else:
            data = read_json(path)
            if isinstance(data, dict):
                data = [data]
        require(isinstance(data, list), "Record JSON must contain an object or array")
        for record in data:
            validate_record(record)
            result.append(record)
    return result


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make-design")
    make.add_argument("--bundle", type=Path, required=True)
    make.add_argument("--off-run-dir", type=Path, help="Completed OFF run; uses pinned collector question")
    make.add_argument("--on-run-dir", type=Path, help="Completed ON run; uses pinned collector question")
    make.add_argument("--off-run", help="Explicit run identity for synthetic fixtures only")
    make.add_argument("--on-run", help="Explicit run identity for synthetic fixtures only")
    make.add_argument("--question-id")
    make.add_argument("--question-file", type=Path)
    make.add_argument("--synthetic-fixture", action="store_true")
    for name in ("fit", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--records", type=Path, nargs="+", required=True)
        command.add_argument("--design", type=Path, required=True)
        if name == "fit":
            command.add_argument("--k", type=int, default=4)
            command.add_argument("--seed", type=int, default=17001)
            command.add_argument("--min-df", type=int, default=2)
            command.add_argument("--max-features", type=int, default=1024)
        else:
            command.add_argument("--model", type=Path, required=True)
            command.add_argument("--min-confidence", type=float, default=0.6)
            command.add_argument("--min-group-size", type=int, default=5)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--report", type=Path, required=True)
    evaluation.add_argument("--truth", type=Path)
    for subparser in (make, *[commands.choices[n] for n in ("fit", "analyze", "evaluate")]):
        subparser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "make-design":
        if args.off_run_dir and args.on_run_dir:
            require(not any((args.off_run, args.on_run, args.question_id, args.question_file, args.synthetic_fixture)), "Do not override pinned production run/question identity")
            result = design_from_runs(args.bundle, args.off_run_dir, args.on_run_dir)
        else:
            require(args.synthetic_fixture and all((args.off_run, args.on_run, args.question_id, args.question_file))
                    and not args.off_run_dir and not args.on_run_dir, "Supply both completed run directories, or a fully explicit --synthetic-fixture design")
            result = design_from_bundle(args.bundle, off_run=args.off_run, on_run=args.on_run,
                                        question_id=args.question_id, question_text=args.question_file.read_text(encoding="utf-8"),
                                        synthetic_fixture=True)
    elif args.command == "fit":
        result = fit_model(load_records(args.records), read_json(args.design), k=args.k, seed=args.seed,
                           min_df=args.min_df, max_features=args.max_features)
    elif args.command == "analyze":
        result = analyze(load_records(args.records), read_json(args.design), read_json(args.model),
                         min_confidence=args.min_confidence, min_group_size=args.min_group_size)
    else:
        result = evaluate(read_json(args.report), read_json(args.truth) if args.truth else None)
    write_new(args.out, result)
    print(str(args.out))


if __name__ == "__main__":
    main()
