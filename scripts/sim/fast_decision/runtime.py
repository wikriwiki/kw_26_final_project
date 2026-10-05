"""Opt-in capture and shadow execution. Never replaces a simulator decision."""
from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
from pathlib import Path

from .contracts import SCHEMA_VERSION, DEFAULT_MODEL, canonical, fingerprint, plain, validate_output
from .planner import propose

LOG = logging.getLogger(__name__)
_WRITE_LOCK = threading.Lock()
_BACKEND_LOCK = threading.Lock()
_BACKENDS = {}
# Missing local weights must not trigger another expensive load for every agent.
# Only load failures are cached; successful backends retain their usual lifetime.
_BACKEND_FAILURES = {}
_BACKEND_FAILURE_RETRY_SECONDS = 60.0
_BACKEND_FAILURE_LIMIT = 16
_CONTEXT_FIELDS = ("memory", "appointment", "social", "knows_poi_summary", "zone_candidates")
_CORRECTIONS = ("fallback_only", "hallucinations_corrected", "hallucinations_dropped",
                "order_mismatch", "missing_picks_filled", "review_lookup_count",
                "review_skipped_no_call_budget", "spend_imputed")


class BackendLoadBackoff(RuntimeError):
    """A recent failure loading the same local model is still in its retry window."""


def _capture_context(context):
    # Select before plain()/dataclasses.asdict(): DawnContext also contains large
    # duplicate persona/state and telemetry fields that never enter the record.
    # plain() still recursively copies every retained field into the snapshot.
    if isinstance(context, dict):
        return {key: context[key] for key in _CONTEXT_FIELDS if key in context}
    if dataclasses.is_dataclass(context) and not isinstance(context, type):
        return {key: getattr(context, key) for key in _CONTEXT_FIELDS if hasattr(context, key)}
    return context


def append_record(path: Path, record: dict) -> None:
    # Serialization happens before opening: no partial line on invalid data.
    line = canonical(record) + "\n"
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line)


def _backend():
    from .backend import ExaoneChoiceBackend
    settings = (
        os.getenv("SIM_FAST_MODEL", DEFAULT_MODEL), os.getenv("SIM_FAST_REVISION"),
        os.getenv("SIM_FAST_DEVICE", "cpu"), os.getenv("SIM_FAST_ADAPTER"),
        int(os.getenv("SIM_FAST_MAX_TOKENS", "4096")),
    )
    with _BACKEND_LOCK:
        if settings not in _BACKENDS:
            now = time.monotonic()
            failure = _BACKEND_FAILURES.get(settings)
            if failure and now < failure[0]:
                raise BackendLoadBackoff(
                    f"Local model load retry delayed after {failure[1]}: {failure[2]}")
            _BACKEND_FAILURES.pop(settings, None)
            try:
                _BACKENDS[settings] = ExaoneChoiceBackend(
                    model_id=settings[0], revision=settings[1], device=settings[2],
                    adapter_path=settings[3], max_tokens=settings[4], allow_download=False)
            except Exception as exc:
                if len(_BACKEND_FAILURES) >= _BACKEND_FAILURE_LIMIT:
                    _BACKEND_FAILURES.pop(next(iter(_BACKEND_FAILURES)))
                # Retain scalar diagnostics, never exceptions/tracebacks that can
                # keep a partially loaded model or large tensors alive.
                _BACKEND_FAILURES[settings] = (
                    time.monotonic() + _BACKEND_FAILURE_RETRY_SECONDS,
                    type(exc).__name__, str(exc)[:500])
                raise
        return _BACKENDS[settings]


class Capture:
    def __init__(self, snapshot: dict, path: Path, mode: str, model_id: str,
                 model_mode: str, synthetic: bool = False):
        self.snapshot, self.path, self.mode = snapshot, path, mode
        self.model_id, self.model_mode = model_id, model_mode
        self.synthetic, self.finished = synthetic, False

    def finish(self, output, meta) -> dict:
        if self.finished:
            return {"mode": self.mode, "status": "already_finished", "applied": False}
        self.finished = True
        started = time.perf_counter()
        result = {"mode": self.mode, "applied": False, "snapshot_id": self.snapshot["snapshot_id"]}
        try:
            teacher = plain(output)
            teacher_meta = plain(meta)
            errors = validate_output(self.snapshot, teacher)
            corrections = [k for k in _CORRECTIONS if teacher_meta.get(k)]
            if teacher_meta.get("review_lookup_used") or teacher.get("review_lookup_requests"):
                corrections.append("review_context")
            teacher_meta["validated"] = not errors and not corrections
            teacher_meta["validation_errors"] = errors + corrections
            row = {
                "schema_version": SCHEMA_VERSION, "snapshot": self.snapshot,
                "teacher": {"output": teacher, "meta": teacher_meta,
                            "model_id": self.model_id, "model_mode": self.model_mode,
                            "latency_seconds": teacher_meta.get("s2_timing", {}).get("t_llm"),
                            "latency_scope": "teacher_llm_only",
                            "stage_seconds": teacher_meta.get("s2_timing", {}).get("t_total")},
                "provenance": {"synthetic": self.synthetic, "source": "stage2_pre_decision_capture"},
            }
            if self.mode == "shadow":
                student_started = time.perf_counter()
                try:
                    # preflight may defer without loading any weights at all.
                    from .planner import preflight
                    if preflight(self.snapshot):
                        row["student"] = propose(self.snapshot, None)
                    else:
                        row["student"] = propose(self.snapshot, _backend(),
                                                  selection=os.getenv("SIM_FAST_SELECTION", "argmax"),
                                                  seed=int(os.getenv("SIM_FAST_SEED", "0")))
                except Exception as exc:
                    row["student"] = {"status": "error", "error_type": type(exc).__name__,
                                      "error": str(exc)[:500], "eligible_for_live": False}
                row["student"]["latency_seconds"] = time.perf_counter() - student_started
                row["student"]["model_id"] = os.getenv("SIM_FAST_MODEL", DEFAULT_MODEL)
                row["student"]["revision"] = os.getenv("SIM_FAST_REVISION")
                result["student_status"] = row["student"]["status"]
            append_record(self.path, row)
            result.update(status="recorded", teacher_validated=teacher_meta["validated"])
        except Exception as exc:
            LOG.warning("Fast decision capture failed (%s); teacher output retained", type(exc).__name__)
            result.update(status="capture_error", error_type=type(exc).__name__)
        result["elapsed_seconds"] = time.perf_counter() - started
        return result


def start_capture(*, aid, today, stage1, persona, state, candidates,
                  system_prompt, user_prompt, recent_poi_ids, active_policies,
                  grant_remaining, context=None):
    mode = os.getenv("SIM_FAST_MODE", "off").strip().lower()
    if mode == "off":
        return None
    if mode not in ("record", "shadow"):
        LOG.warning("Unsupported SIM_FAST_MODE=%r: only off/record/shadow; teacher retained", mode)
        return None
    try:
        snapshot = plain({
            "aid": aid, "today": today, "stage1": stage1, "persona": persona,
            "state": state, "candidates": candidates, "system_prompt": system_prompt,
            "user_prompt": user_prompt, "recent_poi_ids": recent_poi_ids,
            "active_policies": active_policies, "grant_remaining": grant_remaining,
            "context": _capture_context(context),
        })
        if isinstance(snapshot.get("context"), dict):
            # Keep evidence, not duplicate persona/state or timing telemetry.
            source_context = snapshot["context"]
            snapshot["context"] = {
                key: source_context[key] for key in
                _CONTEXT_FIELDS
                if key in source_context
            }
        if not aid:
            raise ValueError("Agent ID is required for leak-free group splitting")
        snapshot["snapshot_id"] = fingerprint(snapshot)
        if __package__.startswith("scripts."):
            from .. import llm_client
        else:
            import llm_client
        spec = llm_client.get_spec()
        path = Path(os.getenv("SIM_FAST_CAPTURE_PATH", "sim_output/fast_decision/captures.jsonl"))
        # Durable pre-teacher record proves the label was unavailable at capture.
        append_record(path.with_suffix(".pending.jsonl"), {
            "schema_version": SCHEMA_VERSION, "snapshot": snapshot,
            "provenance": {"synthetic": False, "source": "before_teacher"},
            "teacher_model_id": spec.hf_id,
        })
        return Capture(snapshot, path, mode, spec.hf_id, spec.key)
    except Exception as exc:
        LOG.warning("Unable to begin decision capture (%s); teacher retained", type(exc).__name__)
        return None
