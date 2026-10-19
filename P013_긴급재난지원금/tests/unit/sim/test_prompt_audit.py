"""Offline regressions for audit coverage and no-smoking prompt provenance."""
from datetime import date
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/sim"))
SPEC = importlib.util.spec_from_file_location("simulation_prompt_audit", ROOT / "scripts/experiments/audit_simulation_prompts.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_unregistered_generation_endpoint_is_not_silently_ignored():
    source = '''from urllib.request import Request, urlopen as send
def new_model_call(base):
    req = Request(base + '/chat/completions', data=b'{}')
    return send(req)
'''
    found = audit.scan_source(source, "scripts/sim/new_module.py")
    assert len(found["callsites"]) == 1
    assert found["callsites"][0]["activity"] == "unclassified_requires_review"
    assert found["callsites"][0]["resolved_call"] == "urllib.request.urlopen"


def test_alias_sdk_and_generation_helper_are_all_captured():
    source = '''from llm_client import call_chat as ask
def first():
    return ask(None, SYSTEM, user)
def second(client):
    return client.chat.completions.create(messages=[])
'''
    found = audit.scan_source(source, "scripts/sim/stage1_intent.py")
    assert {r["kind"] for r in found["callsites"]} == {"shared_chat_wrapper", "sdk_generation_transport"}


def test_plain_dotted_import_detects_actual_lookahead_probe():
    source = '''import urllib.request
def ask(base):
    req = urllib.request.Request(base + '/chat/completions')
    return urllib.request.urlopen(req)
'''
    found = audit.scan_source(source, "scripts/sim/lookahead_probe.py")
    assert found["callsites"][0]["resolved_call"] == "urllib.request.urlopen"
    assert found["callsites"][0]["activity"] == "manual_memorization_probe_not_daily_loop"


def test_injected_collector_chat_callable_and_question_are_inventoried():
    source = '''QUESTION = "가상 시민에게 묻는 질문"
def collect_one(packet, call):
    return call(None, SYSTEM, user_message(packet), max_tokens=700)
'''
    found = audit.scan_source(source, "scripts/experiments/collect_policy_stances.py")
    assert found["callsites"][0]["kind"] == "injected_chat_callable_requires_caller_provenance"
    assert found["callsites"][0]["activity"] == "posthoc_structured_policy_stance_not_daily_loop"
    assert found["prompt_expressions"][0]["name"] == "QUESTION"


def test_type_only_prompt_annotations_are_not_prompt_definitions():
    found = audit.scan_source('class State:\n    system_prompt: str\n', 'state.py')
    assert found['prompt_expressions'] == []


def test_metrics_and_model_discovery_are_not_mislabelled_as_generation():
    source = '''from urllib.request import urlopen
def metrics(base):
    return urlopen(base + '/metrics')
def models(base):
    return urlopen(base + '/models')
'''
    found = audit.scan_source(source, "scripts/sim/monitor.py")
    assert found["callsites"] == []
    assert len(found["network_candidates"]) == 2


def test_registered_legacy_contract_is_not_advertised_as_current():
    catalogue, dispatch = audit.prompt_catalogue()
    by_name = {r["name"]: r for r in catalogue}
    assert dispatch["no_smoking_required"] == "no_smoking_v1"
    assert by_name["no_smoking_v1"]["registered"] and by_name["no_smoking_v1"]["active_no_smoking"]
    assert by_name["v14"]["contract_family"].endswith("requires_adapter")
    assert by_name["asset_transaction_v7"]["active_no_smoking"] is False
    assert by_name["v39"]["registered"] is False


def test_policy_timing_all_statuses_and_no_evaluation_extra_keys():
    cases = audit.representative_fixtures()
    assert len(cases) == 24
    assert all(c["passed"] for c in audit.check_fixtures(cases))
    assert {c["smoking_status"] for c in cases} == {"smoker", "non_smoker", "unknown"}
    for case in cases:
        if case["smoking_status"] == "unknown":
            assert "흡연 여부 정보 없음" in case["user"]


def test_renderer_cannot_silently_add_evaluation_and_future_schedule_blocks():
    from prompts.no_smoking_v1 import format_dawn_blocks
    visible = {"persona": "가상 시민은 오늘 자유 시간이 있다.", "zones": "11650101"}
    poisoned = dict(visible, ground_truth="TARGET_937111", future_policy_schedule="2017-12-03 TARGET_937111")
    assert format_dawn_blocks(visible, date(2017, 11, 19), "weekend", "일") == \
        format_dawn_blocks(poisoned, date(2017, 11, 19), "weekend", "일")


def test_exact_input_quote_passes_and_invented_past_is_rejected():
    from prompt_grounding import validate_stated_reason
    from prompts.no_smoking_v1 import format_dawn_blocks
    user = format_dawn_blocks({"persona": "오늘 확정 약속 없음", "zones": "11650101"}, date(2017, 12, 2), "weekend", "토")
    record = {"reasoning": "확정 약속이 없어 집에서 쉬기로 선택했다.", "evidence_quote": "오늘 확정 약속 없음"}
    result = validate_stated_reason(record, user)
    assert result["quote_span_verified"] is True and result["semantic_support_verified"] is False
    record["evidence_quote"] = "어제 친구와 당구장에 다녀왔다"
    with pytest.raises(ValueError, match="does not occur"):
        validate_stated_reason(record, user)


def test_source_snapshot_detects_changed_missing_and_new_files(tmp_path):
    source = tmp_path / "scripts/sim"
    source.mkdir(parents=True)
    (source / "a.py").write_text("a=1", encoding="utf-8")
    manifest = {"source_hashes": {"scripts/sim/a.py": audit.sha(b"a=1")}}
    assert audit.verify_snapshot(manifest, tmp_path) == []
    (source / "a.py").write_text("a=2", encoding="utf-8")
    (source / "b.py").write_text("new=1", encoding="utf-8")
    assert audit.verify_snapshot(manifest, tmp_path) == ["scripts/sim/a.py", "scripts/sim/b.py"]
    (source / "a.py").unlink()
    assert "scripts/sim/a.py" in audit.verify_snapshot(manifest, tmp_path)


def test_model_registry_marker_scan_is_not_an_ethnicity_inference():
    clean = audit.scan_source('MODEL="LGAI-EXAONE/EXAONE-4.5-33B-AWQ"\nTEXT="한국과 중국의 음식"', "model.py")
    changed = audit.scan_source('MODEL="Qwen/Qwen3-8B"', "model.py")
    assert not clean["chinese_model_markers"]
    assert changed["chinese_model_markers"]


def test_token_budget_remains_unverified_without_exact_local_tokenizer():
    report = audit.token_budget(audit.representative_fixtures())
    assert report["verified"] is False
    assert report["cases"] == []


def test_another_or_modified_tokenizer_cannot_pass_as_pinned_lg(tmp_path):
    for name in audit.TOKENIZER_HASHES:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned official EXAONE"):
        audit.token_budget([], tmp_path)


def test_stage2_and_night_actual_builders_keep_preperiod_equal():
    cases = audit.other_stage_fixtures()
    assert len(cases) == 8
    for stage in ("stage2", "night"):
        for day in ("2017-12-02", "2017-12-03"):
            pair = [row for row in cases if row["stage"] == stage and row["day"] == day]
            assert (pair[0]["user"] == pair[1]["user"]) == (day == "2017-12-02")
            assert all("evidence_quote" in row["system"] for row in pair)


def test_interview_fixtures_distinguish_new_hypothetical_question_from_experience():
    import json
    cases = audit.interview_fixtures()
    assert len(cases) == 6
    structured = [case for case in cases if case["stage"] == "stance_interview"]
    assert len({case["question_sha256"] for case in structured}) == 1
    assert sum(case["measurement_context"] == "experienced" for case in structured) == 1
    assert all(json.loads(case["user"])["evidence"] == [] for case in structured)
    assert {case["stage"] for case in cases} == {"stance_interview", "grounded_interview", "legacy_free_interview"}
    assert all("/no_think" not in case["system"] + case["user"] for case in cases)


def test_all_current_generation_sinks_are_classified():
    calls = []
    for path in audit.SIM.rglob("*.py"):
        scanned = audit.scan_source(path.read_text(encoding="utf-8-sig"), path.relative_to(ROOT).as_posix())
        calls.extend(scanned["callsites"])
    assert {r["activity"] for r in calls}.issuperset({"active_daily_stage1", "active_daily_stage2", "active_nightly_interaction"})
    assert not [r for r in calls if r["activity"] == "unclassified_requires_review"]


def test_same_smoking_label_does_not_erase_different_personal_situations():
    cases = audit.personal_decision_fixtures()
    assert len(cases) == 4
    assert {case["smoking_status"] for case in cases} == {"smoker"}
    for stage in ("stage1", "stage2"):
        pair = [case for case in cases if case["stage"] == stage]
        assert pair[0]["user"] != pair[1]["user"]
        for case in pair:
            assert case["personal_marker"] in case["user"]
            other = next(row for row in pair if row is not case)
            assert other["personal_marker"] not in case["user"]


def _interview_packet(context_text="퇴근 후 쓸 수 있는 시간은 30분이다."):
    from evidence_integrity import canonical, seal
    values = [
        {"evidence_id": "PERSONAL", "day": "2017-12-02", "kind": "context",
         "value": {"persona": {"job": "야간 물류 근무", "lifestyle": context_text}}},
        {"evidence_id": "RECEIPT_1", "day": "2017-12-03", "kind": "executed_receipt", "value": {"amount": 1000}},
        {"evidence_id": "RECEIPT_2", "day": "2017-12-03", "kind": "executed_receipt", "value": {"amount": 2000}},
    ]
    for item in values:
        item["text"] = canonical(item["value"])
    return seal({"run_id": "synthetic", "agent_id": "A", "arm": "on", "through_day": "2017-12-03", "evidence_items": values})


def test_generic_interview_context_selection_preserves_person_before_recent_receipts():
    import json
    from interview_agent import select_grounded_interview_context
    from evidence_contract import interview_prompt
    packet = _interview_packet()
    context_size = len(json.dumps(packet["evidence_items"][0], ensure_ascii=False))
    selected = select_grounded_interview_context(packet, context_size)
    assert [item["evidence_id"] for item in selected["evidence_items"]] == ["PERSONAL"]
    assert selected["omitted_evidence_count"] == 2
    assert selected["personal_context_missing"] is False
    rendered = interview_prompt(selected, "내 생활 조건과 관련된 판단을 설명해 주세요.")
    assert "야간 물류 근무" in rendered and "30분" in rendered
    assert "RECEIPT_1" not in rendered and "RECEIPT_2" not in rendered


def test_generic_interview_will_not_drop_oversized_personal_context_silently():
    from interview_agent import select_grounded_interview_context
    from evidence_integrity import EvidenceError
    with pytest.raises(EvidenceError, match="personal context does not fit"):
        select_grounded_interview_context(_interview_packet("긴 기록 " * 1000), 1000)


def test_generic_interview_records_missing_personal_context_explicitly():
    from interview_agent import select_grounded_interview_context
    from evidence_integrity import seal
    packet = _interview_packet()
    packet = seal({**packet, "evidence_items": packet["evidence_items"][1:]})
    selected = select_grounded_interview_context(packet)
    assert selected["personal_context_missing"] is True
    assert selected["selection"]["selected_personal_context_id"] is None


def test_generic_interview_same_day_context_uses_packet_order_not_random_id():
    from interview_agent import select_grounded_interview_context
    from evidence_integrity import canonical, seal
    packet = _interview_packet()
    older = dict(packet["evidence_items"][0], evidence_id="ZZZ_RANDOM")
    value = {"persona": {"job": "현재 제공된 직업"}}
    newer = dict(older, evidence_id="AAA_RANDOM", value=value, text=canonical(value))
    packet = seal({**packet, "evidence_items": [older, newer]})
    selected = select_grounded_interview_context(packet)
    assert selected["selection"]["selected_personal_context_id"] == "AAA_RANDOM"


def test_exact_quote_validation_is_not_a_logical_quality_judge():
    from prompt_grounding import validate_stated_reason
    result = validate_stated_reason({"reasoning": "이 구절만으로 판단에 필요한 모든 사실을 알 수 있다고 여긴다.",
                                     "evidence_quote": "오늘 확정 약속 없음"}, "오늘 확정 약속 없음")
    assert result["quote_span_verified"] is True
    assert result["semantic_support_verified"] is False


def test_probe_renderer_uses_selected_personal_fields_and_no_target_stance():
    import json
    from scripts.experiments import collect_policy_stances as collector
    # A permissive unit-test tokenizer isolates projection/rendering behavior.
    # Exact LG token budgeting is checked separately by the offline integration.
    class PermissiveTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [0]
    cases = audit.stance_probe_fixtures(PermissiveTokenizer())
    assert len(cases) == 4
    views = {}
    for case in cases:
        assert case["selection_exercised"] is True
        assert case["expected_stance"] is None
        assert case["semantic_argument_quality_verified"] is False
        assert case["reserved_output_tokens"] == collector.MAX_OUTPUT_TOKENS
        evidence = json.loads(case["user"])["evidence"]
        fields = {}
        for item in evidence:
            if item["kind"] == "persona_snapshot":
                fields.update(json.loads(item["text"])["values"])
        views[case["id"]] = fields
    player = views["stance_probe_smoker_frequent_evening_player"]
    walker = views["stance_probe_smoker_budget_walking"]
    assert player["smoking_status"] == walker["smoking_status"] == "smoker"
    assert player["job"] != walker["job"]
    assert player["nv_hobbies"] != walker["nv_hobbies"]
    assert player["commute_min"] != walker["commute_min"]
    assert player["daily_wd"] != walker["daily_wd"]
    sparse = views["stance_probe_unknown_sparse_history"]
    assert sparse["smoking_status"] == "unknown" and sparse["age"] is None
    assert "job" not in sparse and "nv_hobbies" not in sparse
