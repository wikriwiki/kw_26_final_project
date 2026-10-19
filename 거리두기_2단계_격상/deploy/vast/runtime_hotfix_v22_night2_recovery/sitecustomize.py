"""Reuse the 658 verified Day2 Night2 model answers after a format-only repair.

This runtime layer leaves the frozen v22 source and completed agent days intact.
For all later days it only canonicalizes zero padding in a factual line number;
the normal model and factual-line validator still decide the interaction.
"""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


if os.environ.get("NO_SMOKING_V22_SKIP_PERSISTENCE_HOTFIX") == "1":
    previous = Path("/workspace/no-smoking-runtime-hotfix-v22-night2/sitecustomize.py")
    expected = "c8680ced3d27fbeaaf92a89f6e27b65fee44050280ffcddb832c2c094a4c4545"
    if hashlib.sha256(previous.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Existing v22 Night2 retry layer changed")
    spec = importlib.util.spec_from_file_location("_v22_night2_retry", previous)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import night_intent_llm
    from evidence_integrity import digest
    from experience_provenance import source_fingerprint
    from interview_evidence import _load_journal
    from no_smoking_context import configured_context

    original_extract = night_intent_llm._extract_first_json
    original_classify = night_intent_llm.classify_intent
    run_dir = Path("/workspace/no-smoking-results/integration-main-v22-1154-shared-pre")
    manifest_path = run_dir / "night2_recovery_2017-11-20.json"
    cache = None

    def extract_with_canonical_ref(raw):
        parsed = json.loads(original_extract(raw))
        ref = parsed.get("evidence_ref")
        if isinstance(ref, str) and re.fullmatch(r"E[0-9]+", ref):
            canonical = "E" + str(int(ref[1:])).zfill(4)
            if canonical != ref:
                parsed["evidence_ref"] = canonical
        return json.dumps(parsed, ensure_ascii=False)

    night_intent_llm._extract_first_json = extract_with_canonical_ref

    def classify_with_verified_recovery(pair_key, data, max_retry=2):
        global cache
        runtime = configured_context()
        day = str(data.get("simulation_day"))
        output = Path(os.environ.get("SIM_OUTPUT_DIR", "")).resolve()
        if day != "2017-11-20" or output != run_dir or runtime is None:
            return original_classify(pair_key, data, max_retry=max_retry)
        if cache is None:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("count") != 658 or manifest.get("day") != day or
                    manifest.get("run_id") != os.environ.get("SIM_RUN_ID") or
                    manifest.get("arm") != runtime.arm or
                    manifest.get("cohort_sha256") != digest(runtime.agent_ids) or
                    manifest.get("source_sha256") != source_fingerprint()):
                raise RuntimeError("Night2 recovery cache identity mismatch")
            cache = {tuple(entry["pair"]): entry for entry in manifest["entries"]}
            if len(cache) != 658:
                raise RuntimeError("Night2 recovery cache has duplicate pairs")
        entry = cache.get(tuple(pair_key))
        if entry is None:
            raise RuntimeError("Current Night2 pair missing from recovery cache")
        context = json.loads(json.dumps(data, default=str))
        if digest(context) != entry["context_sha256"]:
            raise RuntimeError("Night2 pair data differs from archived model input")
        relative = Path(entry["reference"]["path"])
        journal = (run_dir / relative).resolve()
        if (journal.is_symlink() or not journal.is_relative_to(run_dir / "evidence") or
                hashlib.sha256(journal.read_bytes()).hexdigest() != entry["journal_sha256"]):
            raise RuntimeError("Night2 journal changed after recovery preflight")
        archived, _ = _load_journal(run_dir, entry["reference"], expected_kind="interaction")
        if (archived["interaction"] != entry["interaction"] or
                archived["run_id"] != entry["run_id"] or
                archived["arm"] != entry["arm"] or
                archived["cohort_sha256"] != entry["cohort_sha256"] or
                archived["source_sha256"] != entry["source_sha256"] or
                tuple(archived["agent_ids"]) != tuple(sorted(pair_key))):
            raise RuntimeError("Night2 archived interaction differs from cache")
        result = copy.deepcopy(entry["interaction"])
        result["interview_evidence"] = entry["reference"]
        return result

    night_intent_llm.classify_intent = classify_with_verified_recovery
