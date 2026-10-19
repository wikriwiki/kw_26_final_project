"""Offline source/dispatch audit; never calls an LLM or opens a database.

Static inventories cover every Python source under scripts/sim. Rendered cases
exercise the actual no-smoking Stage 1 renderer and factual context, not model
responses. This is a reproducibility check, not an empirical model validation.
"""
from __future__ import annotations

import argparse
import ast
from datetime import date, datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "scripts" / "sim"
ACTIVE_VARIANT = "no_smoking_v1"
TOKENIZER_MODEL = "LGAI-EXAONE/EXAONE-4.5-33B-AWQ"
TOKENIZER_REVISION = "31e6a965d0661bbe4a8b895e22a77f8271772ba0"
# Independently downloaded from the pinned official repository, tokenizer-only.
TOKENIZER_HASHES = {
    "tokenizer.json": "0bd798efa30739e209d51f36cfc2f0a636711e37ad9d69b77e4e5c8ca5f09fab",
    "tokenizer_config.json": "032e35c1b75c9299e23a687b81c1e989f9b9d52c1e73640f29567b2d037b6945",
    "chat_template.jinja": "e4ece7acc79ba82121d4d57791fb7ecb796e39185087197377da5d2286bec0e5",
}
ACTIVE_PATHS = {
    "stage1_intent.py": "active_daily_stage1",
    "stage2_poi.py": "active_daily_stage2",
    "night_intent_llm.py": "active_nightly_interaction",
    "llm_client.py": "shared_llm_transport",
    "interview_agent.py": "posthoc_interview_not_daily_loop",
    "backfill_night_reasoning.py": "legacy_posthoc_reconstruction_not_original_evidence",
    "lookahead_probe.py": "manual_memorization_probe_not_daily_loop",
    "bounded_reasoning.py": "validation_transport_not_daily_loop",
    "collect_policy_stances.py": "posthoc_structured_policy_stance_not_daily_loop",
    "probe_stance_reasoning.py": "optional_synthetic_reasoning_probe_not_daily_loop",
}
MODEL_MARKER = re.compile(r"\b(?:qwen(?:[0-9._-]|\b)|deepseek|internlm|baichuan|chatglm|thudm|01-ai/yi)", re.I)


def sha(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def qualified(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return qualified(node.value) + "." + node.attr
    return ""


def literal_assignments(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    try:
                        result[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError, SyntaxError):
                        pass
    return result


def classify(path):
    name = Path(path).name
    if name in ACTIVE_PATHS:
        return ACTIVE_PATHS[name]
    if name.startswith("validate_"):
        return "validation_not_daily_loop"
    return "unclassified_requires_review"


def scan_source(source: str, path: str) -> dict:
    """Locate SDK, llm_client aliases and HTTP sinks, including unknown new files.

    HTTP calls without an explicit generation URL in the enclosing function are
    retained as unresolved/non-generation candidates rather than silently lost.
    This intentionally does not pretend to resolve arbitrary dynamic Python.
    """
    tree = ast.parse(source, filename=path)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = (node.module or "") + "." + item.name
        elif isinstance(node, ast.Import):
            for item in node.names:
                # `import urllib.request` binds urllib, while `import ... as x`
                # binds the complete module to x. Do not duplicate '.request'.
                aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    calls, network = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        original = qualified(node.func)
        head, _, rest = original.partition(".")
        name = aliases.get(head, head) + ("." + rest if rest else "")
        parent = min((f for f in functions if f.lineno <= node.lineno <= f.end_lineno),
                     key=lambda f: f.end_lineno - f.lineno, default=None)
        segment = ast.get_source_segment(source, parent) if parent else source
        kind = None
        if name in {"llm_client.call_chat", "llm_client.call_chat_async"}:
            kind = "shared_chat_wrapper"
        elif (parent and original in {arg.arg for arg in parent.args.args} and len(node.args) >= 3
              and re.search(r"SYSTEM|PROMPT", qualified(node.args[1]))):
            kind = "injected_chat_callable_requires_caller_provenance"
        elif name.endswith(".chat.completions.create") or name.endswith(".responses.create"):
            kind = "sdk_generation_transport"
        elif name in {"urllib.request.urlopen", "requests.post", "httpx.post"} or name.endswith(".post"):
            generation = bool(re.search(r"/chat/completions|/generate(?:['\"]|\b)|/responses", segment or ""))
            item = {"path": path, "line": node.lineno, "function": parent.name if parent else "<module>",
                    "call": original, "resolved_call": name,
                    "generation_url_in_function": generation, "source_sha256": sha(source)}
            network.append(item)
            if generation:
                kind = "http_generation_transport"
        if kind:
            calls.append({"path": path, "line": node.lineno,
                          "function": parent.name if parent else "<module>", "call": original,
                          "resolved_call": name, "kind": kind, "activity": classify(path),
                          "arguments": [ast.unparse(a) for a in node.args[:3]],
                          "source_sha256": sha(source)})
    constants = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            name = qualified(target)
            if not re.search(r"PROMPT|SYSTEM|^QUESTION|^Q_(?:ID|OUT)$", name, re.I):
                continue
            value = node.value
            if value is None:  # Type-only declarations do not define a prompt.
                continue
            strings = [n.value for n in ast.walk(value) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            constants.append({"name": name, "line": node.lineno, "expression": ast.unparse(value),
                              "literal_chars": sum(map(len, strings)),
                              "literal_sha256": sha("\n".join(strings)),
                              "legacy_no_think_marker": any("/no_think" in s for s in strings)})
    markers = [{"line": node.lineno, "match": match.group(0)}
               for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
               for match in MODEL_MARKER.finditer(node.value)]
    renderers = [{"name": node.name, "line": node.lineno,
                  "source_sha256": sha(ast.get_source_segment(source, node) or "")}
                 for node in functions if re.search(r"prompt|format|user_block|policy_facts", node.name, re.I)]
    return {"callsites": sorted(calls, key=lambda r: r["line"]),
            "network_candidates": sorted(network, key=lambda r: r["line"]),
            "prompt_expressions": constants, "chinese_model_markers": markers,
            "prompt_renderer_candidates": renderers,
            "source_sha256": sha(source), "syntax_valid": True}


def contract_family(name, text):
    if name == "v14":
        return "relative_time_start_activities_finish_requires_adapter"
    if name.startswith("asset_transaction"):
        return "asset_actions_not_stage1_events"
    if name.startswith("transaction"):
        return "purchase_picks_not_stage1_events"
    if "activity_id" in text:
        return "typed_activity_events_requires_adapter"
    return "stage1_event_text_contract_review_schema_separately"


def prompt_catalogue():
    # Prompt modules only contain definitions and imports of other prompt modules.
    # Do not import simulator, validation scripts, or lookahead_probe (top-level I/O).
    if str(SIM) not in sys.path:
        sys.path.insert(0, str(SIM))
    registry = importlib.import_module("prompts")
    registered = registry.list_variants()
    rows = []
    for path in sorted((SIM / "prompts").glob("*.py")):
        name = path.stem
        source = path.read_text(encoding="utf-8-sig")
        row = {"name": name, "path": path.relative_to(ROOT).as_posix(),
               "file_sha256": sha(path.read_bytes()), "registered": name in registered,
               "active_no_smoking": name == ACTIVE_VARIANT,
               "imports": sorted({node.module or "" for node in ast.walk(ast.parse(source))
                                  if isinstance(node, ast.ImportFrom)})}
        if name not in {"__init__", "candidates"}:
            module = importlib.import_module("prompts." + name)
            system = getattr(module, "SYSTEM_PROMPT", "")
            row.update(system_sha256=sha(system), system_chars=len(system),
                       has_formatter=callable(getattr(module, "format_dawn_blocks", None)),
                       contract_family=contract_family(name, system),
                       legacy_no_think_marker="/no_think" in system,
                       chinese_model_markers=MODEL_MARKER.findall(system))
        else:
            row["contract_family"] = "registry_or_shared_prompt_builder"
        rows.append(row)
    return rows, {"legacy_default": registry.DEFAULT, "no_smoking_required": ACTIVE_VARIANT,
                  "registered": registered,
                  "note": "Registration is not proof of output-schema compatibility; v14 needs an adapter."}


def representative_fixtures():
    if str(SIM) not in sys.path:
        sys.path.insert(0, str(SIM))
    from no_smoking_context import NoSmokingContext
    from prompts.no_smoking_v1 import SYSTEM_PROMPT, format_dawn_blocks
    cohort = [{"id": key, "smoking_status": status} for key, status in
              [("SYN_A", "smoker"), ("SYN_B", "non_smoker"), ("SYN_C", "unknown")]]
    poison = "FORBIDDEN_EVALUATION_TARGET_983741"
    pois = [{"poi_id": "SYN_BILLIARD", "district_code": "11650", "facility_type": "billiard",
             "evaluation_target": poison}]
    cases = []
    for arm in ("off", "on"):
        context = NoSmokingContext(arm=arm, cohort=cohort, pois=pois, assignment_seed=20171203)
        for day in (date(2017, 11, 19), date(2017, 12, 2), date(2017, 12, 3), date(2017, 12, 16)):
            for person in cohort:
                facts = context.context_for(person["id"], day)
                blocks = {"policy_facts": "(없음)", "persona": "가상 시민. 오늘 확정 약속 없음.\n" + facts["prompt"],
                          "policy": "현금 지원 정책 없음", "zones": "[거주지] 11650101 가상 생활권",
                          "state": "직전까지 확인된 추가 경험 없음", "memory": "(없음)",
                          "appointment": "(없음)", "social": "(없음)", "knows_poi": "SYN_BILLIARD: 당구장",
                          "environment": "", "ground_truth": poison, "future_policy_schedule": poison}
                user = format_dawn_blocks(blocks, day, "weekend" if day.weekday() >= 5 else "weekday",
                                          "월화수목금토일"[day.weekday()])
                cases.append({"id": f"{arm}_{day}_{person['id']}", "arm": arm, "day": str(day),
                              "smoking_status": person["smoking_status"], "policy_active": facts["policy_active"],
                              "system": SYSTEM_PROMPT, "user": user,
                              "stage": "stage1", "reserved_output_tokens": 2200,
                              "system_sha256": sha(SYSTEM_PROMPT), "user_sha256": sha(user),
                              "evaluation_canary_absent": poison not in user,
                              "no_future_effective_date_disclosed": day >= date(2017, 12, 3) or "2017-12-03" not in user})
    return cases


def interview_fixtures():
    """Render free, citation-structured and stance-structured interview inputs."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from evidence_integrity import canonical, seal
    from evidence_contract import interview_prompt
    from interview_agent import INTERVIEW_SYSTEM, build_user_block, select_grounded_interview_context
    from scripts.experiments import collect_policy_stances as collector
    cases = []
    for arm in ("off", "on"):
        for day in ("2017-12-02", "2017-12-03"):
            packet = seal({"schema_version": 1, "run_id": "SYN_RUN", "arm": arm, "agent_id": "SYN_A",
                           "through_day": day, "evidence_items": [], "missing_days": [day],
                           "missing_night_days": [day]})
            user = collector.user_message(packet)
            cases.append({"id": f"stance_{arm}_{day}_no_evidence", "stage": "stance_interview", "arm": arm,
                          "day": day, "system": collector.SYSTEM, "user": user,
                          "reserved_output_tokens": collector.MAX_OUTPUT_TOKENS,
                          "measurement_context": collector.measurement_context(arm, day),
                          "question_sha256": collector.QUESTION_SHA256})
    value = {"amount": 1000, "purchase_status": "purchased"}
    packet = seal({"schema_version": 1, "run_id": "SYN_RUN", "arm": "on", "agent_id": "SYN_A",
                   "through_day": "2017-12-03", "evidence_items": [
                       {"evidence_id": "SYN_RECEIPT", "day": "2017-12-03", "kind": "executed_receipt",
                        "value": value, "text": canonical(value)}]})
    question = "기록에 있는 구매와 기록만으로 알 수 없는 이유를 구분해 주세요."
    cases.append({"id": "grounded_interview_receipt", "stage": "grounded_interview", "system": INTERVIEW_SYSTEM,
                  "user": interview_prompt(select_grounded_interview_context(packet), question), "reserved_output_tokens": 800})
    data = {"persona": {"id": "SYN_A"}, "state": [], "plans": [], "memories": [],
            "conversations": [], "knows_poi": [], "through_day": "2017-12-03"}
    cases.append({"id": "legacy_free_interview_missing_evidence", "stage": "legacy_free_interview",
                  "system": INTERVIEW_SYSTEM, "user": build_user_block(data, question), "reserved_output_tokens": 400})
    for case in cases:
        case.update(system_sha256=sha(case["system"]), user_sha256=sha(case["user"]))
    return cases


def check_fixtures(cases):
    checks = []
    for case in cases:
        checks.extend([
            {"id": case["id"] + ":evaluation_allowlist", "passed": case["evaluation_canary_absent"]},
            {"id": case["id"] + ":no_advance_onset", "passed": case["no_future_effective_date_disclosed"]},
            {"id": case["id"] + ":onset", "passed": case["policy_active"] ==
             (case["arm"] == "on" and case["day"] >= "2017-12-03")},
        ])
        opposite = next(row for row in cases if row["arm"] != case["arm"] and
                        (row["day"], row["smoking_status"]) == (case["day"], case["smoking_status"]))
        equal = case["user"] == opposite["user"]
        checks.append({"id": case["id"] + ":paired_pre_equal_post_different",
                       "passed": equal == (case["day"] < "2017-12-03")})
    return checks


def other_stage_fixtures(persona_overrides=None):
    """Call the actual Stage 2/night text builders with synthetic, isolated data.

    Any accidental network/DB connection fails immediately. These are rendering
    fixtures, not full 7500-person production context-length measurements.
    """
    if str(SIM) not in sys.path:
        sys.path.insert(0, str(SIM))
    from no_smoking_context import NoSmokingContext, _load
    from no_smoking_prompts import SYSTEM_STAGE2, SYSTEM_NIGHT
    cases = []
    previous_modules = set(sys.modules)
    with tempfile.TemporaryDirectory(prefix="prompt_audit_") as temp, \
            patch("socket.socket.connect", side_effect=RuntimeError("Offline audit forbids network")), \
            patch("socket.create_connection", side_effect=RuntimeError("Offline audit forbids network")), \
            patch.dict(os.environ, {"SIM_PROMPT_VARIANT": ACTIVE_VARIANT, "SIM_OUTPUT_DIR": temp,
                                    "SIM_NO_SMOKING_MANIFEST": str(Path(temp) / "runtime.json")}):
        cohort = [{"id": "SYN_A", "smoking_status": "smoker"}, {"id": "SYN_B", "smoking_status": "non_smoker"}]
        pois = [{"poi_id": "SYN_BILLIARD", "district_code": "11650", "facility_type": "billiard"}]
        (Path(temp) / "runtime.json").write_text(json.dumps({"experiment_id": "no_smoking_zone", "schema_version": 1,
            "cohort": cohort, "pois": pois, "assignment_seed": 20171203}), encoding="utf-8")
        from stage2_poi import build_stage2_prompt
        from night_intent_llm import build_user_block
        for arm in ("off", "on"):
            os.environ["SIM_NO_SMOKING_ARM"] = arm
            context = NoSmokingContext(arm=arm, cohort=cohort, pois=pois, assignment_seed=20171203)
            for day in ("2017-12-02", "2017-12-03"):
                persona = {"id": "SYN_A", "lifestyle": "오늘 저녁에 여가 시간을 쓸 수 있다.",
                           "daily_wd": 20000, "daily_we": 20000,
                           "_no_smoking_prompt": context.context_for("SYN_A", day)["prompt"]}
                persona.update(persona_overrides or {})
                event = SimpleNamespace(time="18:00", anchor="zone:11650101", category="여가",
                                        sub_category="당구장", intent="여가 이용을 검토한다")
                candidate = {"poi_id": "SYN_BILLIARD", "name": "가상 당구장", "known": False,
                             "km": 0.2, "price_band": 1, "unit_anchor": 20000}
                user = build_stage2_prompt([event], {0: [candidate]}, persona=persona, state={"balance": 50000})
                cases.append({"id": f"stage2_{arm}_{day}", "stage": "stage2", "arm": arm, "day": day,
                              "system": SYSTEM_STAGE2, "user": user, "reserved_output_tokens": 2400})
                person = {"job": "가상 시민", "life": "생활 정보\n저녁에 시간이 있다.", "mood": None, "fatigue": None}
                data = {"a": dict(person), "b": dict(person), "events_a": [], "events_b": [],
                        "score": 0.1, "exp": 0.0, "rel": 0.1, "urg": 0.0, "simulation_day": day}
                user = build_user_block(("SYN_A", "SYN_B"), data)
                cases.append({"id": f"night_{arm}_{day}", "stage": "night", "arm": arm, "day": day,
                              "system": SYSTEM_NIGHT, "user": user, "reserved_output_tokens": 900})
        _load.cache_clear()
    # Stage 1 has import-time prompt/log-path globals. Do not leave a newly
    # imported module bound to this synthetic environment in the test process.
    for name in ("stage1_intent", "stage2_poi", "night_intent_llm"):
        if name not in previous_modules:
            sys.modules.pop(name, None)
    for case in cases:
        case.update(system_sha256=sha(case["system"]), user_sha256=sha(case["user"]))
    return cases


def personal_decision_fixtures():
    """Same smoking status/rule, distinct supplied needs and time resources.

    No desired activity or policy stance is assigned. These checks test exposure
    of differing situations, not whether an LLM will reason from them well.
    """
    from prompts.no_smoking_v1 import SYSTEM_PROMPT, format_dawn_blocks
    from no_smoking_context import NoSmokingContext
    from dawn_context import _format_persona
    context = NoSmokingContext(arm="on", cohort=[{"id": "SYN_A", "smoking_status": "smoker"}],
        pois=[{"poi_id": "SYN_BILLIARD", "district_code": "11650", "facility_type": "billiard"}], assignment_seed=1)
    profiles = [
        {"case_id": "short_time", "job": "야간 물류 근무", "income": "중하", "daily_wd": 12000,
         "daily_we": 12000, "nv_hobbies": "당구", "commute_min": 55,
         "lifestyle": "취미는 당구이며 퇴근 뒤 여유 시간은 30분이다."},
        {"case_id": "flexible_time", "job": "유연 근무", "income": "중상", "daily_wd": 40000,
         "daily_we": 40000, "nv_hobbies": "실내 운동 체험", "commute_min": 10,
         "lifestyle": "취미는 실내 운동 체험이며 저녁에 두 시간 여유가 있다."},
    ]
    cases = []
    for profile in profiles:
        marker = profile["lifestyle"]
        persona = {**profile, "id": "SYN_A", "smoking_status": "smoker",
                   "_no_smoking_prompt": context.context_for("SYN_A", "2017-12-03")["prompt"]}
        blocks = {"persona": _format_persona(persona),
                  "zones": "[거주지] 11650101 가상 생활권", "state": "현재 잔액 50,000원", "policy_facts": "(없음)"}
        cases.append({"id": "personal_stage1_" + profile["case_id"], "stage": "stage1",
                      "system": SYSTEM_PROMPT, "user": format_dawn_blocks(blocks, date(2017, 12, 3), "weekend", "일"),
                      "reserved_output_tokens": 2200, "personal_marker": marker,
                      "comparison_group": "same_smoking_distinct_situation", "smoking_status": "smoker"})
        stage2 = next(case for case in other_stage_fixtures(profile) if case["stage"] == "stage2"
                      and case["arm"] == "on" and case["day"] == "2017-12-03")
        cases.append({**stage2, "id": "personal_stage2_" + profile["case_id"], "personal_marker": marker,
                      "comparison_group": "same_smoking_distinct_situation", "smoking_status": "smoker"})
    for case in cases:
        case.update(system_sha256=sha(case["system"]), user_sha256=sha(case["user"]))
    return cases


def stance_probe_fixtures(tokenizer=None):
    """Render the shared synthetic diagnostic packets, never execute a model."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.experiments import collect_policy_stances as collector
    from scripts.experiments.probe_stance_reasoning import probe_cases
    cases = []
    for probe in probe_cases():
        packet = probe["packet"]
        selected = collector.bounded_packet(packet, tokenizer, 8192 - collector.MAX_OUTPUT_TOKENS - 128) if tokenizer else packet
        user = collector.user_message(selected)
        cases.append({"id": "stance_probe_" + probe["case_id"], "stage": "stance_interview",
                      "system": collector.SYSTEM, "user": user, "reserved_output_tokens": collector.MAX_OUTPUT_TOKENS,
                      "system_sha256": sha(collector.SYSTEM), "user_sha256": sha(user),
                      "review_focus": probe.get("review_focus"), "selection_exercised": tokenizer is not None,
                      "selection": selected.get("selection"), "expected_stance": None,
                      "semantic_argument_quality_verified": False})
    return cases


def token_budget(cases, tokenizer_path=None, context_limit=8192):
    if tokenizer_path is None:
        return {"verified": False, "reason": "No local exact tokenizer supplied; character counts are not token estimates.",
                "context_limit": context_limit, "cases": []}
    folder = Path(tokenizer_path).resolve()
    file_hashes = {name: sha((folder / name).read_bytes()) for name in TOKENIZER_HASHES}
    if file_hashes != TOKENIZER_HASHES:
        raise ValueError("Tokenizer files do not match the pinned official EXAONE revision")
    # The official serialized tokenizer and chat template suffice for text-only
    # input. No AutoModel, remote code, weights, or network access are used.
    from prompt_budget import load_tokenizer
    tokenizer = load_tokenizer(folder)
    from transformers import __version__ as transformers_version
    rows = []
    for case in cases:
        tokens = tokenizer.apply_chat_template([{"role": "system", "content": case["system"]},
                                               {"role": "user", "content": case["user"]}],
                                              tokenize=True, add_generation_prompt=True, enable_thinking=False)
        total = len(tokens) + case["reserved_output_tokens"] + 128
        rows.append({"id": case["id"], "stage": case["stage"], "input_tokens": len(tokens),
                     "reserved_output_tokens": case["reserved_output_tokens"], "total_reserved": total,
                     "margin_tokens": 128,
                     "remaining": context_limit - total, "overflow": total > context_limit})
    return {"verified": True, "method": "official_local_tokenizer_json_and_chat_template_text_only",
            "model_id": TOKENIZER_MODEL, "model_revision": TOKENIZER_REVISION,
            "transformers_version": transformers_version, "enable_thinking": False,
            "tokenizer_path": str(folder), "context_limit": context_limit,
            "tokenizer_files": file_hashes,
            "all_fixture_budgets_fit": not any(row["overflow"] for row in rows), "cases": rows,
            "limitation": "Synthetic fixtures only; actual daily memories/candidate lists and retries must be token-checked at dispatch."}


def verify_snapshot(manifest, root=ROOT):
    changes = []
    for path, expected in manifest["source_hashes"].items():
        full = root / path
        if not full.is_file() or sha(full.read_bytes()) != expected:
            changes.append(path)
    current = {p.relative_to(root).as_posix() for p in (root / "scripts/sim").rglob("*.py")}
    changes.extend(sorted(current - set(manifest["source_hashes"])))
    return sorted(set(changes))


def build_audit(tokenizer_path=None):
    sources = sorted([*SIM.rglob("*.py"), ROOT / "scripts/experiments/collect_policy_stances.py",
                      ROOT / "scripts/experiments/probe_stance_reasoning.py"])
    inventory, hashes, errors = [], {}, []
    for path in sources:
        relative = path.relative_to(ROOT).as_posix()
        hashes[relative] = sha(path.read_bytes())
        try:
            scanned = scan_source(path.read_text(encoding="utf-8-sig"), relative)
            # AST text decoding may normalize Windows newlines. Source identity
            # in persisted inventories always refers to the original file bytes.
            scanned["source_sha256"] = hashes[relative]
            for record in scanned["callsites"] + scanned["network_candidates"]:
                record["source_sha256"] = hashes[relative]
            inventory.append({"path": relative, **scanned})
        except SyntaxError as error:
            errors.append({"path": relative, "line": error.lineno, "error": str(error)})
    for relative in ("scripts/experiments/no_smoking_zone.py", "scripts/experiments/audit_simulation_prompts.py",
                     "data/experiments/no_smoking_zone/tokenizer_manifest.json"):
        hashes[relative] = sha((ROOT / relative).read_bytes())
    catalogue, dispatch = prompt_catalogue()
    from prompts.no_smoking_v1 import SYSTEM_PROMPT as stage1_system
    from no_smoking_prompts import SYSTEM_STAGE2, SYSTEM_NIGHT
    surfaces = [
        {"stage": "stage1", "system_source": "scripts/sim/prompts/no_smoking_v1.py",
         "system_sha256": sha(stage1_system), "selector": "SIM_PROMPT_VARIANT=no_smoking_v1 pinned by experiment runner",
         "context_sources": ["dawn_context.py", "no_smoking_context.py", "experience.py"],
         "evidence": "Input substring quote and public rationale; proposed plan is not an executed receipt."},
        {"stage": "stage2", "system_source": "scripts/sim/no_smoking_prompts.py:SYSTEM_STAGE2",
         "system_sha256": sha(SYSTEM_STAGE2), "selector": "persona._no_smoking_prompt present",
         "context_sources": ["stage2_poi.py", "no_smoking_context.py", "poi_price.py"],
         "evidence": "Input substring quote and public choice rationale; amount/satisfaction are model proposals."},
        {"stage": "night", "system_source": "scripts/sim/no_smoking_prompts.py:SYSTEM_NIGHT",
         "system_sha256": sha(SYSTEM_NIGHT), "selector": "configured no-smoking runtime",
         "context_sources": ["night_intent_llm.py", "no_smoking_context.py"],
         "evidence": "Proposed synthetic interaction, not an observed conversation or expressed real-person stance."},
        {"stage": "interview", "system_source": "scripts/sim/interview_agent.py:INTERVIEW_SYSTEM",
         "selector": "manual posthoc request; outside daily loop", "rendered_fixture": True,
         "evidence": "Archived decision/context/receipt evidence must retain source and as-of boundaries."},
        {"stage": "stance_interview", "system_source": "scripts/experiments/collect_policy_stances.py:SYSTEM+QUESTION",
         "selector": "explicit completed-run posthoc collection; no writeback to simulation", "rendered_fixture": True,
         "evidence": "Same question, explicit hypothetical/experienced measurement context; missing evidence is not neutral stance."},
    ]
    fixtures = representative_fixtures()
    checks = check_fixtures(fixtures)
    other_cases = other_stage_fixtures()
    for stage in ("stage2", "night"):
        for day in ("2017-12-02", "2017-12-03"):
            pair = [c for c in other_cases if c["stage"] == stage and c["day"] == day]
            checks.append({"id": f"{stage}_{day}:paired_pre_equal_post_different",
                           "passed": (pair[0]["user"] == pair[1]["user"]) == (day < "2017-12-03")})
    fixtures += other_cases
    interviews = interview_fixtures()
    for case in interviews:
        if case["stage"] == "stance_interview":
            expected = "experienced" if case["arm"] == "on" and case["day"] >= "2017-12-03" else "hypothetical"
            checks.append({"id": case["id"] + ":measurement_context", "passed": case["measurement_context"] == expected})
    fixtures += interviews
    personal = personal_decision_fixtures()
    checks.extend({"id": case["id"] + ":personal_situation_retained",
                   "passed": case["personal_marker"] in case["user"]} for case in personal)
    fixtures += personal
    tokenizer = None
    if tokenizer_path:
        from prompt_budget import load_tokenizer
        tokenizer = load_tokenizer(tokenizer_path)
    probes = stance_probe_fixtures(tokenizer)
    fixtures += probes
    checks.append({"id": "active_rendered_prompts_do_not_use_legacy_no_think_marker",
                   "passed": all("/no_think" not in c["system"] + c["user"] for c in fixtures)})
    budget = token_budget(fixtures, tokenizer_path)
    if budget["verified"]:
        checks.append({"id": "representative_fixture_token_budgets_fit", "passed": budget["all_fixture_budgets_fit"]})
    runner = (ROOT / "scripts/experiments/no_smoking_zone.py").read_text(encoding="utf-8-sig")
    pin = bool(re.search(r'env\[[\'"]SIM_PROMPT_VARIANT[\'"]\]\s*=\s*[\'"]no_smoking_v1[\'"]', runner))
    checks.append({"id": "experiment_runner_pins_variant", "passed": pin})
    model_source = (SIM / "llm_client.py").read_text(encoding="utf-8-sig")
    model_defaults = literal_assignments(ast.parse(model_source))
    checks.append({"id": "default_model_is_exaone_4_5", "passed": model_defaults.get("DEFAULT_MODE") == "exaone_4_5"})
    model_markers = next(row for row in inventory if row["path"].endswith("/llm_client.py"))["chinese_model_markers"]
    checks.append({"id": "no_chinese_model_markers_in_shared_client", "passed": not model_markers})
    calls = [call for row in inventory for call in row["callsites"]]
    unknown = [call for call in calls if call["activity"] == "unclassified_requires_review"]
    checks.append({"id": "all_generation_sinks_classified", "passed": not unknown})
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "all_scripts_sim_python_plus_stance_collector_and_probe_static_and_representative_rendering",
        "real_model_called": False, "semantic_reasoning_quality_verified": False,
        "source_hashes": hashes, "dispatch": dispatch,
        "simulation_window": {"start": "2017-11-19", "end": "2017-12-16", "days": 28,
                              "policy_effective": "2017-12-03", "arms": ["off", "on"]},
        "counts": {"python_sources": len(sources), "prompt_modules": len(catalogue),
                   "registered_variants": len(dispatch["registered"]), "generation_callsites": len(calls),
                   "stage1_rendered_cases": sum(c["stage"] == "stage1" for c in fixtures),
                   "stage2_rendered_cases": sum(c["stage"] == "stage2" for c in fixtures),
                   "night_rendered_cases": sum(c["stage"] == "night" for c in fixtures),
                   "interview_rendered_cases": len(interviews) + len(probes),
                   "personal_decision_comparison_cases": len(personal),
                   "stance_reasoning_probe_cases": len(probes)},
        "all_static_checks_passed": not errors and all(c["passed"] for c in checks),
        "syntax_errors": errors,
        "limitations": [
            "No network, database, or LLM inference; EXAONE behavior and GPU output quality remain untested.",
            "Static AST traversal is not a full dynamic call graph; HTTP candidates without explicit generation URLs are retained for review.",
            "Rendered fixtures cover Stage 1, Stage 2, night and three interview interfaces with synthetic inputs; not all production contexts.",
            "Stage 2 retains the existing paid-visit/positive-spend modeling constraint; neutral wording does not establish absence of structural behavior bias. Execution corrections remain diagnostics.",
            "Allowlist canaries prove rejection of extra block keys, not arbitrary factual correctness or absence of pretrained-model memorization.",
            "Literal evidence quotes prove input provenance, not truth of a model's subjective explanation or causal policy effect.",
            "Personal-context exposure and exact quotes do not verify semantic relevance, logical consistency, priority weighing or argument quality; these require actual responses and independent review.",
            "Synthetic same-smoking/different-situation cases have no expected stance. They check available context, not population realism or a required diversity of answers.",
            "Legacy/validation prompts are preserved and are not approved for the pinned no-smoking runtime.",
        ],
    }
    return {"manifest": manifest, "checks": checks, "callsites": calls,
            "prompt_catalogue": catalogue, "active_surfaces": surfaces,
            "source_inventory": inventory, "fixtures": fixtures, "token_budget": budget}


def markdown_report(audit):
    m = audit["manifest"]
    lines = ["# Simulation prompt audit (offline)", "", f"Created: {m['created_at']}", "",
             f"Static checks passed: {m['all_static_checks_passed']}. This is not a model-behavior certification.", "",
             f"Sources: {m['counts']['python_sources']}; prompt modules: {m['counts']['prompt_modules']}; "
             f"registered variants: {m['counts']['registered_variants']}; Stage 1 rendered cases: {m['counts']['stage1_rendered_cases']}.", "",
             "## Call sites", "", "| File:line | Function | Status |", "|---|---|---|"]
    lines += [f"| {r['path']}:{r['line']} | {r['function']} | {r['activity']} |" for r in audit["callsites"]]
    lines += ["", "## Prompt versions", "", "| Name | Registered | Active study | Contract family |", "|---|---|---|---|"]
    lines += [f"| {r['name']} | {r['registered']} | {r['active_no_smoking']} | {r['contract_family']} |"
              for r in audit["prompt_catalogue"]]
    lines += ["", "## Limits", ""] + ["- " + text for text in m["limitations"]]
    budget = audit["token_budget"]
    lines += ["", "## Exact tokenizer check", "", f"Verified: {budget['verified']} (synthetic fixtures only)."]
    if budget["verified"]:
        lines += ["", "| Stage | Maximum input tokens | Output reserve | Maximum combined |", "|---|---:|---:|---:|"]
        for stage in sorted({row["stage"] for row in budget["cases"]}):
            rows = [row for row in budget["cases"] if row["stage"] == stage]
            lines.append(f"| {stage} | {max(r['input_tokens'] for r in rows)} | {max(r['reserved_output_tokens'] for r in rows)} | {max(r['total_reserved'] for r in rows)} |")
    failed = [r["id"] for r in audit["checks"] if not r["passed"]]
    lines += ["", "## Failed checks", "", *(failed or ["None."])]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/no_smoking_zone/prompt_audit_v1")
    parser.add_argument("--verify", type=Path, help="Verify a prior manifest against current source bytes; do not regenerate")
    parser.add_argument("--tokenizer-path", type=Path, help="Local pinned EXAONE tokenizer-only directory; never downloads weights")
    args = parser.parse_args(argv)
    if args.verify:
        changes = verify_snapshot(json.loads(args.verify.read_text(encoding="utf-8")))
        print(json.dumps({"source_snapshot_matches": not changes, "changed_or_added_sources": changes}, ensure_ascii=False))
        return int(bool(changes))
    audit = build_audit(args.tokenizer_path)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in audit.items():
        (args.out / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(markdown_report(audit), encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), **audit["manifest"]["counts"],
                      "all_static_checks_passed": audit["manifest"]["all_static_checks_passed"]}, ensure_ascii=False))
    return int(not audit["manifest"]["all_static_checks_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
