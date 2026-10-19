"""Replayable public model I/O and grounded interview packets.

Only explicit scopes enable logging. Requests are persisted before dispatch;
normal answer content is retained, never a provider's hidden reasoning field.
Journal records become interview evidence only through sealed committed metrics.
Hashes detect corruption, not deliberate replacement by an authorized operator.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
from typing import Any
from uuid import uuid4

try:
    from .evidence_integrity import EvidenceError, canonical, digest, iso_day, seal, verify, verify_observation
except ImportError:
    from evidence_integrity import EvidenceError, canonical, digest, iso_day, seal, verify, verify_observation

# The simulator imports flat modules, while offline tools import the package.
# Both must share the same ContextVar rather than silently disabling capture.
sys.modules.setdefault("interview_evidence", sys.modules[__name__])
sys.modules.setdefault("scripts.sim.interview_evidence", sys.modules[__name__])


_ACTIVE = ContextVar("interview_evidence_scope", default=None)
_WRITE_LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise EvidenceError("evidence objects require string keys")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_value(value.model_dump(mode="json"))
    # Validation rejects NaN and unsupported objects rather than inventing text.
    canonical(value)
    return value


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise EvidenceError(f"{name} must be a SHA-256 hex digest")
    return value


@dataclass
class Scope:
    root: Path
    relative_path: str
    identity: dict
    stage: str = "unspecified"
    context: dict = field(default_factory=dict)
    initial_context: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    closed: bool = False


def begin_evidence(run_dir, run_id, arm, day, agent_ids, cohort_sha256, source_sha256, context=None):
    """Return a ContextVar token; caller must clear it in finally.

    Contexts are immutable JSON snapshots, not live references to agent objects.
    Each scope has a fresh journal, so retries never overwrite previous calls.
    """
    day = iso_day(str(day))
    if (not isinstance(run_id, str) or not run_id or not isinstance(arm, str) or not arm
            or not isinstance(agent_ids, (list, tuple)) or not agent_ids
            or any(not isinstance(aid, str) or not aid for aid in agent_ids)
            or len(set(agent_ids)) != len(agent_ids)):
        raise EvidenceError("missing or duplicate evidence scope identity")
    ids = sorted(agent_ids)
    path = f"evidence/v1/{day}/{digest(ids)[:16]}_{uuid4().hex}.jsonl"
    scope = Scope(Path(run_dir).resolve(), path,
                  {"run_id": run_id, "arm": arm, "day": day, "agent_ids": ids,
                   "cohort_sha256": _hash(cohort_sha256, "cohort_sha256"),
                   "source_sha256": _hash(source_sha256, "source_sha256")})
    scope.context = json.loads(canonical(json_value(context or {})))
    scope.initial_context = scope.context
    return _ACTIVE.set(scope)


def clear_evidence(token=None):
    if token is None:
        _ACTIVE.set(None)
    else:
        _ACTIVE.reset(token)


def set_evidence_stage(stage, context=None):
    scope = _ACTIVE.get()
    if scope is None:
        return
    if scope.closed or not isinstance(stage, str) or not stage:
        raise EvidenceError("invalid or closed evidence stage")
    scope.stage = stage
    if context is not None:
        scope.context = json.loads(canonical(json_value(context)))


def _append(scope, kind, payload):
    if scope.closed:
        raise EvidenceError("evidence scope already archived")
    record = seal({"schema_version": 1, "record_id": "EV_" + uuid4().hex,
                   "recorded_at": now(), "kind": kind, **scope.identity,
                   "stage": scope.stage, **json_value(payload)})
    path = scope.root / scope.relative_path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(canonical(record) + "\n")
            output.flush()
            os.fsync(output.fileno())
    except (OSError, ValueError) as exc:
        scope.failures.append(type(exc).__name__)
        raise EvidenceError("could not durably store model evidence") from exc
    return record


def _public_response(response):
    """Explicit allowlist avoids SDK internals and hidden reasoning_content."""
    choices = []
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        choices.append({"index": getattr(choice, "index", len(choices)),
                        "content": getattr(message, "content", None),
                        "refusal": getattr(message, "refusal", None),
                        "finish_reason": getattr(choice, "finish_reason", None)})
    usage = getattr(response, "usage", None)
    return {"id": getattr(response, "id", None), "model": getattr(response, "model", None),
            "created": getattr(response, "created", None), "choices": choices,
            "usage": {key: getattr(usage, key, None) for key in
                      ("prompt_tokens", "completion_tokens", "total_tokens")}}


def record_chat_call(request, send):
    """llm_client hook. Outside a scope this preserves the original behavior."""
    scope = _ACTIVE.get()
    if scope is None:
        return send()
    if scope.closed or scope.failures:
        raise EvidenceError("evidence scope is closed or has persistence failures")
    try:
        request = json.loads(canonical(request))
        context = json.loads(canonical(scope.context))
    except (TypeError, ValueError) as exc:
        scope.failures.append('request_serialization')
        raise EvidenceError('model request cannot be serialized as replayable evidence') from exc
    request_record = _append(scope, "llm_request", {
        "request": request, "request_sha256": digest(request),
        "context": context, "context_sha256": digest(context),
        "model_revision_requested": os.environ.get("MODEL_REVISION"),
        "hidden_reasoning_requested": False})
    call = {"request_id": request_record["record_id"],
            "request_sha256": request_record["integrity_sha256"], "response_id": None}
    scope.calls.append(call)
    try:
        response = send()
    except Exception as exc:
        # Exception messages may contain URLs/credentials; the class is enough
        # to distinguish a failed call from a citizen's expressed preference.
        response_record = _append(scope, "llm_response", {
            "request_id": request_record["record_id"],
            "request_record_sha256": request_record["integrity_sha256"],
            "status": "error", "error_type": type(exc).__name__,
            "response": None, "response_sha256": digest(None)})
        call["response_id"] = response_record["record_id"]
        call["response_sha256"] = response_record["integrity_sha256"]
        raise
    public = _public_response(response)
    response_record = _append(scope, "llm_response", {
        "request_id": request_record["record_id"],
        "request_record_sha256": request_record["integrity_sha256"],
        "status": "returned", "response": public, "response_sha256": digest(public)})
    call["response_id"] = response_record["record_id"]
    call["response_sha256"] = response_record["integrity_sha256"]
    return response


def archive_agent_day(result, decisions=None, executed_events=None):
    """Archive distinct channels; commit is proven later by sealed metrics.

    Add the returned reference to result['interview_evidence'] BEFORE the
    existing agent_day_store.save_result transaction commits it.
    """
    scope = _ACTIVE.get()
    if scope is None:
        return None
    if (scope.failures or any(not call["response_id"] for call in scope.calls)
            or result.get("aid") not in scope.identity["agent_ids"]
            or result.get("experience_run_id") != scope.identity["run_id"]
            or result.get("experience_day") != scope.identity["day"]
            or result.get("status") != "ok"):
        raise EvidenceError("cannot archive incomplete or foreign agent evidence")
    receipts = result.get("execution_receipts", [])
    for receipt in receipts:
        verify_observation(receipt)
        if (receipt["agent_id"] != result["aid"] or receipt["run_id"] != scope.identity["run_id"]
                or receipt["observed_at"] != scope.identity["day"]):
            raise EvidenceError("foreign receipt in agent evidence")
    public_fields = ("reasoning", "trigger", "pick_reason", "pick_factor", "intent",
                     "evidence_quote", "evidence_quote_validated", "pick_evidence_quote")
    decision_snapshot = json_value(decisions or {})
    rationales = []
    def visit(value, pointer=""):
        if isinstance(value, dict):
            statement = {key: value[key] for key in public_fields if value.get(key) is not None}
            if statement:
                stage = pointer.split('/')[1] if pointer.startswith('/') else None
                metadata = decision_snapshot.get(f"{stage}_meta", {}) if isinstance(decision_snapshot, dict) else {}
                fallback = isinstance(metadata, dict) and metadata.get('fallback_only') is True
                rationales.append({"event_order": value.get("order"), "decision_pointer": pointer,
                                   "public_statement": statement, "reason_status": "subjective_unverified",
                                   "origin": "engine_fallback" if fallback else "postprocessed_public_statement_unverified"})
            for key, child in value.items():
                visit(child, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, pointer + f"/{index}")
    visit(decision_snapshot)
    facts = ("order", "time", "poi_id", "category", "actual_spent", "actual_satisfaction",
             "purchase_status", "execution_event_id", "facility_type", "district_code",
             "policy_target", "poi_registry_sha256")
    actual = [{key: row[key] for key in facts if key in row} for row in executed_events or []]
    diagnostic_keys = {key for key in result if key.startswith(("fb_", "fallback_", "s1_", "s2_"))}
    diagnostic_keys |= {"policy_spend_corrected", "appraisal_rejections"}
    diagnostic = {key: result[key] for key in sorted(diagnostic_keys) if key in result}
    if isinstance(decision_snapshot, dict):
        diagnostic["stage_metadata"] = {key: decision_snapshot[key] for key in ("stage1_meta", "stage2_meta") if key in decision_snapshot}
    record = _append(scope, "agent_day", {
        "agent_id": result["aid"], "commit_status": "requires_committed_metrics_reference",
        "receipts": receipts, "executed_events": actual, "stated_rationales": rationales,
        "decision_snapshot": decision_snapshot,
        "fallback_diagnostics": diagnostic, "calls": scope.calls,
        "initial_context": scope.initial_context,
        "limitations": ["Public reasons are model statements, not verified causes or hidden thoughts.",
                        "Executed events are modeled experiences, not observations of real people."]})
    scope.closed = True
    return {"schema_version": 1, "path": scope.relative_path,
            "record_id": record["record_id"], "integrity_sha256": record["integrity_sha256"]}


def archive_interaction(result):
    """Retain a classified social interaction; DB completion is separate."""
    scope = _ACTIVE.get()
    if scope is None:
        return None
    if scope.failures or any(not call["response_id"] for call in scope.calls):
        raise EvidenceError("cannot archive incomplete interaction evidence")
    record = _append(scope, "interaction", {"interaction": json_value(result),
        "commit_status": "requires_completed_night_reference", "calls": scope.calls,
        "reason_status": "public_model_statement_not_verified_cause"})
    scope.closed = True
    return {"schema_version": 1, "path": scope.relative_path,
            "record_id": record["record_id"], "integrity_sha256": record["integrity_sha256"]}


def archive_interview_answer(answer):
    """Link a post-run answer to its calls without adding it to lived history."""
    scope = _ACTIVE.get()
    if scope is None:
        raise EvidenceError("interview answer requires an active evidence scope")
    verify(answer)
    if (scope.failures or any(not call['response_id'] for call in scope.calls)
            or answer.get('run_id') != scope.identity['run_id']
            or answer.get('arm') != scope.identity['arm']
            or answer.get('agent_id') not in scope.identity['agent_ids']
            or answer.get('as_of') != scope.identity['day']):
        raise EvidenceError("interview answer identity or call evidence mismatch")
    record = _append(scope, 'post_run_interview', {'answer':answer, 'calls':scope.calls,
                     'fed_back_into_simulation':False})
    scope.closed = True
    return {'schema_version':1,'path':scope.relative_path,'record_id':record['record_id'],
            'integrity_sha256':record['integrity_sha256']}


def commit_night_evidence(run_dir, run_id, arm, day, results, skipped=None):
    """Called only after successful DB writes; never mutates a graph itself."""
    root = Path(run_dir).resolve()
    day = iso_day(str(day))
    clean = json_value(results)
    if not isinstance(clean, list):
        raise EvidenceError("night results must be a list")
    conversation_ids = set()
    for result in clean:
        reference = result.get("interview_evidence")
        if not reference:
            raise EvidenceError("night result lacks interaction evidence")
        archived, _ = _load_journal(root, reference, expected_kind="interaction")
        if archived["run_id"] != run_id or archived["arm"] != arm or archived["day"] != day:
            raise EvidenceError("night result belongs to another run or day")
        cid = result.get("conversation_id")
        # A classifier may return no social action, so no Conversation is
        # necessarily written. Any written conversation must have a unique ID.
        if cid is not None:
            if not isinstance(cid, str) or not cid or cid in conversation_ids:
                raise EvidenceError("duplicate or invalid conversation identity")
            conversation_ids.add(cid)
        original = {key: value for key, value in result.items() if key not in {"interview_evidence", "conversation_id"}}
        if original != archived["interaction"]:
            raise EvidenceError("night result changed after classification archive")
        result["evidence_agent_ids"] = archived["agent_ids"]
    from night_recovery import verify_skipped
    missing = [verify_skipped(root, value) for value in (skipped or [])]
    if any((v['run_id'], v['arm'], v['day']) != (run_id, arm, day) for v in missing):
        raise EvidenceError('Foreign skipped night interaction')
    success_pairs = {tuple(sorted(v['evidence_agent_ids'])) for v in clean}
    skip_pairs = {tuple(sorted(v['pair'])) for v in missing}
    if len(skip_pairs) != len(missing) or success_pairs & skip_pairs:
        raise EvidenceError('Duplicate terminal night pair')
    record = seal({"schema_version": 1, "kind": "completed_night_evidence",
                   "record_id": "EV_" + uuid4().hex, "recorded_at": now(),
                   "run_id": run_id, "arm": arm, "day": day, "results": clean,
                   "skipped": missing})
    relative = f"evidence/night/{day}_{record['record_id']}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(canonical(record))
        output.flush()
        os.fsync(output.fileno())
    return {"schema_version": 1, "path": relative, "record_id": record["record_id"],
            "integrity_sha256": record["integrity_sha256"]}


def _load_journal(root, reference, expected_kind="agent_day"):
    relative = Path(reference["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root / "evidence") or path.is_symlink():
        raise EvidenceError("evidence path escapes run evidence directory")
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        verify(value)
        if value.get("schema_version") != 1 or value.get("record_id") in records:
            raise EvidenceError("unsupported or duplicate journal record")
        records[value["record_id"]] = value
    record = records.get(reference.get("record_id"))
    if not record or record["integrity_sha256"] != reference.get("integrity_sha256") or record["kind"] != expected_kind:
        raise EvidenceError("missing or altered agent-day journal reference")
    requests = {key: row for key, row in records.items() if row["kind"] == "llm_request"}
    responses = {key: row for key, row in records.items() if row["kind"] == "llm_response"}
    seen_requests, seen_responses = set(), set()
    for call in record["calls"]:
        req, resp = requests.get(call["request_id"]), responses.get(call["response_id"])
        if (not req or not resp or req["record_id"] in seen_requests or resp["record_id"] in seen_responses
                or req["integrity_sha256"] != call["request_sha256"]
                or resp["integrity_sha256"] != call["response_sha256"]
                or resp["request_id"] != req["record_id"]
                or resp["request_record_sha256"] != req["integrity_sha256"]
                or digest(req["request"]) != req["request_sha256"]
                or digest(req["context"]) != req["context_sha256"]
                or digest(resp["response"]) != resp["response_sha256"]):
            raise EvidenceError("incomplete or inconsistent model-call ledger")
        for key in ("run_id", "arm", "day", "agent_ids", "cohort_sha256", "source_sha256"):
            if req[key] != record[key] or resp[key] != record[key]:
                raise EvidenceError("foreign call in agent-day journal")
        seen_requests.add(req["record_id"])
        seen_responses.add(resp["record_id"])
    if seen_requests != set(requests) or seen_responses != set(responses):
        raise EvidenceError("orphan request/response in completed journal")
    return record, records


def _quote_validation(statement, records, stage=None):
    quote = statement.get('evidence_quote')
    if quote is None or quote == '':
        return {'status':'missing', 'request_ids':[]}
    if not isinstance(quote, str):
        raise EvidenceError('invalid rationale evidence quote type')
    sources = []
    for record in records.values():
        if record['kind'] != 'llm_request' or (stage and record['stage'] != stage):
            continue
        if any(message.get('role') == 'user' and isinstance(message.get('content'), str)
               and quote in message['content'] for message in record['request'].get('messages', [])):
            sources.append(record['record_id'])
    if not sources:
        raise EvidenceError('rationale evidence quote does not match exposed input')
    return {'status':'exact_input_quote', 'request_ids':sources, 'semantic_support_verified':False}


def _read_night_index(root, day):
    marker_path = root / f"night2_completed_{day}.json"
    if not marker_path.is_file():
        return None
    marker = verify(json.loads(marker_path.read_text(encoding="utf-8")))
    reference = marker.get("evidence_ref")
    if not isinstance(reference, dict):
        raise EvidenceError("night completion marker lacks replayable evidence")
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root / "evidence/night"):
        raise EvidenceError("night index escapes evidence directory")
    index = verify(json.loads(path.read_text(encoding="utf-8")))
    if (index.get("integrity_sha256") != reference.get("integrity_sha256")
            or index.get("record_id") != reference.get("record_id")
            or index.get("kind") != "completed_night_evidence"
            or (index.get("run_id"), index.get("arm"), index.get("day")) != (marker.get("run_id"), marker.get("arm"), day)):
        raise EvidenceError("invalid night index reference")
    return marker, index


def _night_items(root, agent_id, day, identity, night_reader=None):
    loaded = night_reader(day, agent_id) if night_reader else _read_night_index(root, day)
    if loaded is None:
        return [], False
    marker, index = loaded
    run_id, arm, cohort_sha, source_sha = identity
    if (marker.get("run_id") != run_id or marker.get("arm") != arm
            or marker.get("day") != day or marker.get("status") != "complete"):
        raise EvidenceError("night completion marker identity/status mismatch")
    items = []
    for result in index["results"]:
        if agent_id not in result.get("evidence_agent_ids", []):
            continue
        archived, records = _load_journal(root, result["interview_evidence"], expected_kind="interaction")
        if (archived["run_id"], archived["arm"], archived["cohort_sha256"], archived["source_sha256"]) != identity or archived["day"] != day:
            raise EvidenceError("foreign interaction in night index")
        if sorted(result["evidence_agent_ids"]) != archived["agent_ids"] or agent_id not in archived["agent_ids"]:
            raise EvidenceError("night participant index differs from journal")
        value = {"conversation_id": result.get("conversation_id"),
                 "participants": archived["agent_ids"], "interaction": archived["interaction"],
                 "reason_status": "public_model_statement_not_verified_cause",
                 "experience_status": "recorded_social_interaction" if result.get("conversation_id") else "no_recorded_conversation"}
        value['quote_validation'] = _quote_validation(archived['interaction'], records)
        items.append({"evidence_id": archived["record_id"], "agent_id": agent_id, "day": day,
                      "kind": "social_interaction", "text": canonical(value), "value": value,
                      "source_ref": result["interview_evidence"]})
    from night_recovery import verify_skipped
    for skipped in index.get('skipped', []):
        if agent_id not in skipped['pair']:
            continue
        value = verify_skipped(root, skipped)
        if (value['run_id'], value['arm'], value['cohort_sha256'], value['source_sha256']) != identity or value['day'] != day:
            raise EvidenceError('Foreign skipped night receipt')
        items.append({'evidence_id': 'NS_' + value['integrity_sha256'],
                      'agent_id': agent_id, 'day': day, 'kind': 'skipped_social_interaction',
                      'text': canonical(value), 'value': value,
                      'source_ref': {'journal_refs': value['journals']}})
    return items, True


def build_packet(run_dir, agent_id, through_day, *, days=None, allow_incomplete=False,
                 _snapshot_reader=None, _night_reader=None):
    """Read-only export from committed snapshots; future days never enter it."""
    root = Path(run_dir).resolve()
    through_day = iso_day(str(through_day))
    manifest = json.loads((root / "experiment_run.json").read_text(encoding="utf-8"))
    cohort = manifest.get("cohort_ids", [])
    if agent_id not in cohort or len(cohort) != len(set(cohort)):
        raise EvidenceError("agent is not in the frozen run cohort")
    start = date.fromisoformat(iso_day(manifest["start"]))
    end = start + timedelta(days=manifest["days"] - 1)
    if not start <= date.fromisoformat(through_day) <= end:
        raise EvidenceError("interview date is outside the simulation window")
    expected_days = [(start + timedelta(days=i)).isoformat()
                     for i in range((date.fromisoformat(through_day) - start).days + 1)]
    if days is not None:
        chosen = list(days)
        if not chosen or len(chosen) != len(set(chosen)) or not set(chosen) <= set(expected_days):
            raise EvidenceError("interview days include future, duplicate, or out-of-window dates")
        expected_days = sorted(chosen)
    items, missing, identities, snapshot_refs, missing_nights, skipped = [], [], [], [], [], []
    identity_by_day = {}
    for day in expected_days:
        path = root / "metrics" / f"day_{day}.jsonl"
        matches = []
        if _snapshot_reader:
            matches = _snapshot_reader(day, agent_id)
        elif path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("aid") == agent_id and row.get("status") in {"ok", "skipped"}:
                        if not matches or matches[0] != row:
                            matches.append(row)
        for row in matches:
            verify(row)
            if row.get("experience_day") != day:
                raise EvidenceError("committed snapshot has the wrong day")
        if not matches:
            missing.append(day)
            continue
        if len(matches) != 1:
            raise EvidenceError("conflicting completed snapshots")
        row = matches[0]
        if row.get('status') == 'skipped':
            if (row.get('attempts') != 6 or row.get('skip_kind') != 'failed_after_retries'
                    or row.get('observed_behavior') is not False
                    or row.get('experience_run_id') != manifest.get('run_id')
                    or row.get('no_smoking', {}).get('arm') != manifest['arm']):
                raise EvidenceError('invalid skipped-day receipt')
            skipped.append(day)
            identity = (row['experience_run_id'], manifest['arm'],
                        digest(sorted(cohort)), row['source_fingerprint'])
            identities.append(identity)
            identity_by_day[day] = identity
            snapshot_refs.append({'day': day, 'status': 'skipped',
                                  'metrics_sha256': row['integrity_sha256']})
            continue
        reference = row.get("interview_evidence")
        if not reference:
            raise EvidenceError("legacy snapshot lacks replayable interview evidence")
        archived, records = _load_journal(root, reference)
        if (archived["agent_id"] != agent_id or archived["day"] != day
                or archived["run_id"] != row.get("experience_run_id")
                or (manifest.get("run_id") is not None and archived["run_id"] != manifest["run_id"])
                or archived["arm"] != manifest["arm"]
                or archived["source_sha256"] != row.get("source_fingerprint")
                or archived["cohort_sha256"] != digest(sorted(cohort))
                or archived["receipts"] != row.get("execution_receipts", [])):
            raise EvidenceError("journal does not match committed snapshot")
        identity = tuple(archived[key] for key in ("run_id", "arm", "cohort_sha256", "source_sha256"))
        identities.append(identity)
        identity_by_day[day] = identity
        snapshot_refs.append({"day": day, "metrics_sha256": row["integrity_sha256"], **reference})
        def add(kind, value, suffix, evidence_id=None):
            items.append({"evidence_id": evidence_id or f"{archived['record_id']}:{suffix}",
                          "day": day, "kind": kind, "agent_id": agent_id,
                          "text": canonical(value), "value": value,
                          "source_ref": {**reference, "field": suffix}})
        add("context", archived["initial_context"], "initial_context")
        for i, receipt in enumerate(archived["receipts"]):
            verify_observation(receipt)
            add("executed_receipt", receipt, f"receipts/{i}", receipt["event_id"])
        for i, event in enumerate(archived["executed_events"]):
            add("executed_event", event, f"executed_events/{i}")
        for i, rationale in enumerate(archived["stated_rationales"]):
            pointer = rationale.get('decision_pointer', '')
            stage = pointer.split('/')[1] if pointer.startswith('/') else None
            checked = {**rationale, 'quote_validation':_quote_validation(rationale['public_statement'],records,stage)}
            add("stated_rationale", checked, f"stated_rationales/{i}")
        add("fallback_diagnostic", archived["fallback_diagnostics"], "fallback_diagnostics")
        for call in archived["calls"]:
            request, response = records[call["request_id"]], records[call["response_id"]]
            # Context is included explicitly; full prompt/output remain replayable
            # in the referenced journal without filling the interview token budget.
            add("context", {"stage": request["stage"], "context": request["context"],
                            "context_sha256": request["context_sha256"]}, f"calls/{request['record_id']}/context")
            add("model_call_reference", {"stage": request["stage"], "request_id": request["record_id"],
                "request_sha256": request["request_sha256"], "response_id": response["record_id"],
                "response_sha256": response["response_sha256"], "status": response["status"],
                "requested_model": request["request"].get("model"),
                "returned_model": (response.get("response") or {}).get("model")}, f"calls/{request['record_id']}")
    if not identities:
        raise EvidenceError("no committed evidence available for this agent and interval")
    if len({identity[:3] for identity in identities}) != 1:
        raise EvidenceError("mixed run, arm, or cohort identities")
    transition = manifest.get('source_transition')
    source_versions = {day: identity_by_day[day][3] for day in identity_by_day}
    if len(set(source_versions.values())) != 1:
        if (not isinstance(transition, dict)
                or transition.get('effective_day') != '2017-11-20'
                or transition.get('inherited_day') != '2017-11-19'
                or any(source != (transition.get('previous_source_fingerprint')
                                  if day < transition['effective_day']
                                  else transition.get('new_source_fingerprint'))
                       for day, source in source_versions.items())):
            raise EvidenceError("unrecorded mixed source versions")
    for day in expected_days:
        if day in missing:
            continue
        nightly, present = _night_items(root, agent_id, day,
                                        identity_by_day[day], _night_reader)
        items.extend(nightly)
        if not present:
            missing_nights.append(day)
    if len({item["evidence_id"] for item in items}) != len(items):
        raise EvidenceError("duplicate evidence identity")
    if not allow_incomplete and (missing or missing_nights):
        raise EvidenceError(f"incomplete interview history: missing days={missing}, missing completed nights={missing_nights}")
    run_id, arm, cohort_sha, _ = identities[0]
    source_sha = source_versions[max(source_versions)]
    return seal({"schema_version": 1, "kind": "grounded_interview_packet",
                 "created_at": now(), "run_id": run_id, "arm": arm, "agent_id": agent_id,
                 "through_day": through_day, "days": expected_days, "missing_days": missing,
                 "skipped_days": skipped,
                 "missing_night_days": missing_nights,
                 "cohort_sha256": cohort_sha, "source_sha256": source_sha,
                 "source_versions_by_day": source_versions,
                 "evidence_items": items, "snapshot_refs": snapshot_refs,
                 "limitations": ["Only committed modeled experiences are included; future days are excluded.",
                     "Model statements and fallback diagnostics are separate from executed receipts.",
                     "A valid citation proves an exact source match, not subjective causal truth.",
                     "Missing days are unavailable, never neutral or evidence of no events."]})


@contextmanager
def packet_session(run_dir, through_day, agent_ids=None):
    """Bounded-memory census reader: scan source metrics/night indexes once.

    Temporary SQLite storage is separate from the immutable evidence tree and
    removed when the reader closes. Journals are read only for each respondent.
    """
    root = Path(run_dir).resolve()
    through_day = iso_day(str(through_day))
    manifest = json.loads((root / "experiment_run.json").read_text(encoding="utf-8"))
    wanted = set(agent_ids if agent_ids is not None else manifest["cohort_ids"])
    if not wanted or not wanted <= set(manifest["cohort_ids"]):
        raise EvidenceError("indexed interview cohort must be a nonempty run subset")
    start = date.fromisoformat(iso_day(manifest["start"]))
    end = date.fromisoformat(through_day)
    if not start <= end < start + timedelta(days=manifest["days"]):
        raise EvidenceError("interview date is outside the simulation window")
    with tempfile.TemporaryDirectory(prefix="interview-evidence-index-") as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "index.sqlite"))
        try:
            connection.executescript("CREATE TABLE snapshots(aid TEXT, day TEXT, payload TEXT, PRIMARY KEY(aid,day));"
                                     "CREATE TABLE night(aid TEXT, day TEXT, payload TEXT);"
                                     "CREATE INDEX night_person ON night(aid,day);"
                                     "CREATE TABLE night_meta(day TEXT PRIMARY KEY,payload TEXT);")
            for offset in range((end - start).days + 1):
                day = (start + timedelta(days=offset)).isoformat()
                path = root / "metrics" / f"day_{day}.jsonl"
                if path.is_file():
                    with path.open(encoding="utf-8") as stream:
                        for line in stream:
                            row = json.loads(line)
                            if row.get("status") not in {"ok", "skipped"}:
                                continue
                            verify(row)
                            if row.get("experience_day") != day:
                                raise EvidenceError("committed snapshot has the wrong day")
                            if row.get("aid") not in wanted:
                                continue
                            payload = canonical(row)
                            old = connection.execute("SELECT payload FROM snapshots WHERE aid=? AND day=?", (row['aid'],day)).fetchone()
                            if old and old[0] != payload:
                                raise EvidenceError("conflicting completed snapshots")
                            connection.execute("INSERT OR IGNORE INTO snapshots VALUES (?,?,?)", (row['aid'],day,payload))
                loaded = _read_night_index(root, day)
                if loaded:
                    marker, index = loaded
                    results = index.pop("results")
                    connection.execute("INSERT INTO night_meta VALUES (?,?)", (day,canonical([marker,index])))
                    for result in results:
                        participants = result.get("evidence_agent_ids")
                        if not isinstance(participants,list) or not participants or len(set(participants)) != len(participants):
                            raise EvidenceError("missing/duplicate night participant index")
                        for aid in wanted.intersection(participants):
                            connection.execute("INSERT INTO night VALUES (?,?,?)", (aid,day,canonical(result)))
            connection.commit()
            class Reader:
                def snapshots(self, day, aid):
                    return [json.loads(row[0]) for row in connection.execute("SELECT payload FROM snapshots WHERE aid=? AND day=?", (aid,day))]

                def night(self, day, aid):
                    row = connection.execute("SELECT payload FROM night_meta WHERE day=?", (day,)).fetchone()
                    if row is None:
                        return None
                    marker, index = json.loads(row[0])
                    index['results'] = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM night WHERE aid=? AND day=?", (aid,day))]
                    return marker, index

                def build_packet(self, agent_id, *, days=None, allow_incomplete=False):
                    if agent_id not in wanted:
                        raise EvidenceError("agent was not selected for this interview reader")
                    return build_packet(root, agent_id, through_day, days=days, allow_incomplete=allow_incomplete,
                                        _snapshot_reader=self.snapshots, _night_reader=self.night)
            yield Reader()
        finally:
            connection.close()


def export_packet(run_dir, agent_id, through_day, out, *, days=None):
    packet = build_packet(run_dir, agent_id, through_day, days=days)
    target = Path(out).resolve()
    root = Path(run_dir).resolve()
    if target.is_relative_to(root / "metrics") or target.is_relative_to(root / "evidence") or target == root / "experiment_run.json":
        raise EvidenceError("export must not overwrite source evidence")
    try:
        from .experience_provenance import atomic_json
    except ImportError:
        from experience_provenance import atomic_json
    atomic_json(target, packet)
    return packet


def audit_run_evidence(run_dir, through_day):
    """Audit the frozen cohort without model calls, DB access, or source writes."""
    root = Path(run_dir).resolve()
    through_day = iso_day(str(through_day))
    manifest = json.loads((root/'experiment_run.json').read_text(encoding='utf-8'))
    ids = manifest.get('cohort_ids')
    if not isinstance(ids,list) or not ids or len(set(ids)) != len(ids):
        raise EvidenceError('audit requires the complete distinct frozen cohort')
    day_count = (date.fromisoformat(through_day)-date.fromisoformat(manifest['start'])).days+1
    if type(manifest.get('days')) is not int or not 1 <= day_count <= manifest['days']:
        raise EvidenceError('audit date is outside the simulation window')
    report = {'schema_version':1, 'kind':'interview_evidence_census_audit', 'created_at':now(),
              'through_day':through_day, 'arm':manifest['arm'], 'expected_agents':len(ids),
              'expected_agent_days':len(ids)*day_count, 'verified_packets':0,
              'verified_agent_days':0, 'skipped_agent_days':0,
              'missing_agent_days':0, 'missing_agent_nights':0,
              'invalid_journal_or_snapshot_agents':0, 'invalid_quote_agents':0,
              'rationale_coverage':{'statements':0,'exact_input_quotes':0,'missing_quotes':0,
                                    'engine_fallback_statements':0,'unquoted_nonfallback_statements':0},
              'failures':[], 'gpu_or_llm_called':False, 'source_modified':False,
              'limitations':['Quoted source matches do not establish causal or semantic truth.',
                             'Invalid-agent counts are not counts of every corrupt field.',
                             'Failure never creates a replacement reason or a neutral attitude.']}
    try:
        with packet_session(root,through_day) as reader:
            for aid in ids:
                try:
                    packet = reader.build_packet(aid,allow_incomplete=True)
                    report['verified_agent_days'] += (len(packet['days'])-len(packet['missing_days'])
                                                      -len(packet.get('skipped_days', [])))
                    report['skipped_agent_days'] += len(packet.get('skipped_days', []))
                    report['missing_agent_days'] += len(packet['missing_days'])
                    report['missing_agent_nights'] += len(packet['missing_night_days'])
                    if packet['missing_days'] or packet['missing_night_days']:
                        report['failures'].append({'agent_id':aid,'kind':'incomplete_history',
                            'missing_days':packet['missing_days'],'missing_night_days':packet['missing_night_days']})
                    else:
                        report['verified_packets'] += 1
                    for item in packet['evidence_items']:
                        if item['kind'] not in {'stated_rationale','social_interaction'}:
                            continue
                        coverage = report['rationale_coverage']
                        coverage['statements'] += 1
                        status = item['value']['quote_validation']['status']
                        coverage['exact_input_quotes' if status=='exact_input_quote' else 'missing_quotes'] += 1
                        if item['value'].get('origin')=='engine_fallback':
                            coverage['engine_fallback_statements'] += 1
                        elif status!='exact_input_quote':
                            coverage['unquoted_nonfallback_statements'] += 1
                except (EvidenceError,OSError,ValueError,TypeError,KeyError) as exc:
                    kind = 'invalid_quote' if 'quote' in str(exc) else 'invalid_journal_or_snapshot'
                    if 'no committed evidence' in str(exc):
                        report['missing_agent_days'] += day_count
                        report['missing_agent_nights'] += day_count
                        kind = 'missing_all_committed_days'
                    else:
                        report[kind+'_agents'] += 1
                    report['failures'].append({'agent_id':aid,'kind':kind,'error':str(exc)})
    except (EvidenceError,OSError,ValueError,TypeError,KeyError) as exc:
        # Corrupt shared indexes/metrics stop the census rather than letting an
        # incomplete scan masquerade as validation of the remaining cohort.
        report['global_error'] = str(exc)
    report['status'] = ('passed' if report['verified_packets']==len(ids) and not report['failures']
                        and 'global_error' not in report
                        and not report['rationale_coverage']['unquoted_nonfallback_statements'] else 'blocked')
    return seal(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--aid")
    parser.add_argument("--audit-all", action='store_true', help='Validate every agent in the frozen run cohort')
    parser.add_argument("--through-day", required=True)
    parser.add_argument("--out", help="Optional packet export; omitted means read-only audit")
    args = parser.parse_args()
    if args.audit_all == bool(args.aid):
        parser.error('choose exactly one of --aid or --audit-all')
    try:
        if args.audit_all:
            report = audit_run_evidence(args.run_dir,args.through_day)
            if args.out:
                target = Path(args.out).resolve()
                root = Path(args.run_dir).resolve()
                if target.is_relative_to(root/'evidence') or target.is_relative_to(root/'metrics') or target==root/'experiment_run.json':
                    raise EvidenceError('audit output must not overwrite source evidence')
                try:
                    from .experience_provenance import atomic_json
                except ImportError:
                    from experience_provenance import atomic_json
                atomic_json(target,report)
            print(canonical(report))
            return 0 if report['status']=='passed' else 2
        packet = (export_packet(args.run_dir, args.aid, args.through_day, args.out) if args.out
                  else build_packet(args.run_dir, args.aid, args.through_day))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Interview evidence blocked: {exc}\n")
    print(canonical({"status": "complete" if not packet["missing_days"] else "incomplete",
                     "agent_id": packet["agent_id"], "evidence_items": len(packet["evidence_items"]),
                     "missing_days": packet["missing_days"], "packet_sha256": packet["integrity_sha256"]}))
    return 0 if not packet["missing_days"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
