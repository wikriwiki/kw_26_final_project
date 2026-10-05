"""One-time, auditable recovery of the 2017-11-20 Night2 model calls.

Run on the original Vast host. The normal simulator still selects pairs and
performs the transactional DB write; this tool only seals existing model
evidence and prepares a cache of validated, previously returned answers.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys


PROJECT = Path("/workspace/no-smoking-project-v22-perf-final")
RUN = Path("/workspace/no-smoking-results/integration-main-v22-1154-shared-pre")
DAY = "2017-11-20"
CUTOFF = datetime(2026, 9, 27, 4, 1, tzinfo=timezone.utc).timestamp()
BAD_PAIR = ("AGT_11650651_F_40대_002", "AGT_11650651_M_40대_005")
OUT = RUN / "night2_recovery_2017-11-20.json"
RECEIPT = RUN / "night2_evidence_ref_repair_2017-11-20.json"

sys.path.insert(0, str(PROJECT / "scripts/sim"))
from evidence_integrity import digest, verify  # noqa: E402
from interview_evidence import Scope, _ACTIVE, _load_journal, archive_interaction  # noqa: E402
from night_intent_llm import IntentOutput  # noqa: E402
from prompt_grounding import validate_stated_reason  # noqa: E402
from stage1_intent import _evidence_lines, _extract_json  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp, path)


def validate_failed(rows):
    requests = [r for r in rows if r["kind"] == "llm_request"]
    responses = [r for r in rows if r["kind"] == "llm_response"]
    if len(requests) != 6 or len(responses) != 6 or len(rows) != 12:
        raise ValueError("Unexpected failed-pair call count or extra journal records")
    identity = {k: requests[0][k] for k in
                ("run_id", "arm", "day", "agent_ids", "cohort_sha256", "source_sha256")}
    if identity["day"] != DAY or tuple(identity["agent_ids"]) != BAD_PAIR:
        raise ValueError("Failed-pair identity mismatch")
    calls = []
    for req, resp in zip(requests, responses):
        verify(req)
        verify(resp)
        if any(req[k] != identity[k] or resp[k] != identity[k] for k in identity):
            raise ValueError("Foreign model call in failed journal")
        if (resp["request_id"] != req["record_id"] or
                resp["request_record_sha256"] != req["integrity_sha256"] or
                resp["status"] != "returned" or
                digest(req["request"]) != req["request_sha256"] or
                digest(req["context"]) != req["context_sha256"] or
                digest(resp["response"]) != resp["response_sha256"]):
            raise ValueError("Broken request/response link")
        calls.append({"request_id": req["record_id"],
                      "request_sha256": req["integrity_sha256"],
                      "response_id": resp["record_id"],
                      "response_sha256": resp["integrity_sha256"]})
    if len({r["context_sha256"] for r in requests}) != 1:
        raise ValueError("Failed-pair context changed between attempts")
    req, resp = requests[-1], responses[-1]
    user = req["request"]["messages"][-1]["content"]
    refs = _evidence_lines(user)
    raw = resp["response"]["choices"][0]["content"]
    parsed = json.loads(_extract_json(raw))
    old = parsed.get("evidence_ref")
    if old != "E028" or old in refs or not re.fullmatch(r"E0*[1-9][0-9]*", old):
        raise ValueError("Expected only the known leading-zero reference error")
    canonical = "E" + str(int(old[1:])).zfill(4)
    if canonical != "E0028" or canonical not in refs:
        raise ValueError("No unique canonical factual line")
    parsed["evidence_ref"] = canonical
    parsed["evidence_quote"] = refs[canonical]
    validate_stated_reason(parsed, user)
    model = IntentOutput.model_validate(parsed)
    if (model.initiator_id, model.recipient_id) != BAD_PAIR or model.intent != "기타":
        raise ValueError("Normalized response changed pair or intent")
    data = req["context"]
    result = {
        "intent": model.intent, "initiator_id": model.initiator_id,
        "recipient_id": model.recipient_id, "topic_type": model.topic_type,
        "topic_value": model.topic_value,
        "should_inject": model.plan_signal.should_inject,
        "target_day_offset": model.plan_signal.target_day_offset,
        "target_time": model.plan_signal.target_time,
        "meeting_location_hint": model.plan_signal.meeting_location_hint,
        "reasoning": model.reasoning, "evidence_ref": model.evidence_ref,
        "evidence_quote": model.evidence_quote,
        "statement_kind": "model_inferred_interaction",
        "interaction_score": data.get("score", 0.0),
        "exposure_score": data.get("exp", 0.0),
        "relationship_score": data.get("rel", 0.0),
        "urgency_score": data.get("urg", 0.0),
        "threshold_used": data.get("threshold_used"),
        "ambient_threshold_applied": bool(data.get("ambient_threshold_applied")),
        "tokens_in": resp["response"]["usage"]["prompt_tokens"],
        "tokens_out": resp["response"]["usage"]["completion_tokens"],
        "attempt": 5,
    }
    return identity, calls, result, req["context_sha256"], resp["response_sha256"], old, canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "seal"))
    args = parser.parse_args()
    if OUT.exists() or RECEIPT.exists():
        raise RuntimeError("Recovery output already exists; inspect it before any repeat")
    root = RUN / "evidence/v1" / DAY
    files = sorted(p for p in root.glob("*.jsonl") if p.stat().st_mtime >= CUTOFF)
    if len(files) != 658:
        raise ValueError(f"Expected exactly 658 fresh journals; got {len(files)}")
    entries = []
    identities = set()
    bad = None
    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        archived = [row for row in rows if row.get("kind") == "interaction"]
        if len(archived) == 1:
            record = archived[0]
            ref = {"schema_version": 1, "path": str(path.relative_to(RUN)),
                   "record_id": record["record_id"],
                   "integrity_sha256": record["integrity_sha256"]}
            loaded, records = _load_journal(RUN, ref, expected_kind="interaction")
            interaction = loaded["interaction"]
            pair = (interaction["initiator_id"], interaction["recipient_id"])
            requests = [row for row in records.values() if row["kind"] == "llm_request"]
            context_hashes = {row["context_sha256"] for row in requests}
            if len(context_hashes) != 1 or tuple(loaded["agent_ids"]) != tuple(sorted(pair)):
                raise ValueError("Archived pair context or identity inconsistent")
            context_hash = context_hashes.pop()
            entry = {"pair": pair, "reference": ref, "context_sha256": context_hash,
                     "interaction": interaction, "journal_sha256": sha(path),
                     "run_id": loaded["run_id"], "arm": loaded["arm"],
                     "source_sha256": loaded["source_sha256"],
                     "cohort_sha256": loaded["cohort_sha256"]}
            entries.append(entry)
        elif not archived:
            if bad is not None:
                raise ValueError("More than one unarchived pair")
            identity, calls, result, context_hash, raw_hash, old, canonical = validate_failed(rows)
            bad = (path, identity, calls, result, context_hash, raw_hash, old, canonical)
        else:
            raise ValueError("Duplicate interaction in a journal")
    if len(entries) != 657 or bad is None:
        raise ValueError("Expected 657 archived and one repairable pair")
    common = {(e["run_id"], e["arm"], e["source_sha256"], e["cohort_sha256"]) for e in entries}
    path, identity, calls, result, context_hash, raw_hash, old, canonical = bad
    if common != {(identity["run_id"], identity["arm"], identity["source_sha256"], identity["cohort_sha256"])}:
        raise ValueError("Recovery journals have mixed experiment identities")
    for e in entries:
        pair = tuple(e["pair"])
        if pair in identities:
            raise ValueError("Duplicate pair")
        identities.add(pair)
    if BAD_PAIR in identities:
        raise ValueError("Failed pair also archived")
    print(json.dumps({"mode": args.mode, "archived": len(entries), "unarchived": 1,
                      "failed_pair": BAD_PAIR, "repair": old + "->" + canonical,
                      "all_journals_valid": True}, ensure_ascii=False))
    if args.mode == "preflight":
        return
    before = sha(path)
    scope = Scope(RUN, str(path.relative_to(RUN)), identity,
                  stage="night_intent", calls=calls)
    token = _ACTIVE.set(scope)
    try:
        ref = archive_interaction(result)
    finally:
        _ACTIVE.reset(token)
    loaded, _ = _load_journal(RUN, ref, expected_kind="interaction")
    if loaded["interaction"] != result:
        raise ValueError("Archived normalized result differs")
    entries.append({"pair": BAD_PAIR, "reference": ref,
                    "context_sha256": context_hash, "interaction": result,
                    "journal_sha256": sha(path),
                    "run_id": identity["run_id"], "arm": identity["arm"],
                    "source_sha256": identity["source_sha256"],
                    "cohort_sha256": identity["cohort_sha256"]})
    atomic_json(RECEIPT, {"day": DAY, "pair": BAD_PAIR,
                          "journal_sha256_before": before,
                          "journal_sha256_after": sha(path),
                          "raw_response_sha256": raw_hash,
                          "original_ref": old, "canonical_ref": canonical,
                          "canonical_factual_line_sha256": digest(result["evidence_quote"]),
                          "normalized_result_sha256": digest(result),
                          "new_interaction_record": ref,
                          "reason": "unambiguous leading-zero formatting of a factual line number; model intent and text unchanged"})
    atomic_json(OUT, {"schema_version": 1, "day": DAY, "run_id": identity["run_id"],
                      "arm": identity["arm"], "count": 658,
                      "source_sha256": identity["source_sha256"],
                      "cohort_sha256": identity["cohort_sha256"],
                      "entries": sorted(entries, key=lambda e: tuple(e["pair"]))})
    print(json.dumps({"sealed": 658, "manifest": str(OUT),
                      "manifest_sha256": sha(OUT), "repair_receipt_sha256": sha(RECEIPT)}))


if __name__ == "__main__":
    main()
