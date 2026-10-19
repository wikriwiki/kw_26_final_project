"""Synthetic public arguments only; no model, human attitude or semantic accuracy.

These test reproducible contracts and missing-link flags, not hidden reasoning or
whether an agent's argument is logically true. Semantic review remains required.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.sim.stance_argument import LIMITS, validate_argument, public_argument_text
from scripts.experiments import analyze_policy_stance as analysis


@pytest.fixture
def packet():
    return {"evidence_items": [{"evidence_id": "self", "text": "저는 흡연자이며 평일 저녁에 여가시간을 냅니다."},
                               {"evidence_id": "visit", "text": "당구장을 1회 방문했습니다. 공개 설명: 냄새가 불편했습니다."}]}


@pytest.fixture
def response():
    return {
        "stance": "support", "answer": "저는 흡연자이지만 조건부로 찬성합니다. 실내 공기가 쾌적해지는 가치를 중요하게 생각합니다. 흡연을 위해 밖으로 나가는 불편은 있을 수 있습니다. 별도 흡연실이 있는지는 모르므로 실제 편의는 확인해야 합니다.",
        "stance_quote": "조건부로 찬성합니다", "confidence": 0.85,
        "reasons": [{"evidence_id": "self", "quote": "저는 흡연자"}],
        "argument": {
            "personal_situation": [{"claim": "평일 저녁 여가시간에 흡연 편의도 고려하는 사람입니다.",
                                    "evidence": [{"evidence_id": "self", "quote": "저는 흡연자이며 평일 저녁에 여가시간을 냅니다."}]}],
            "considerations": [
                {"claim": "공기가 쾌적한 실내에서 시간을 보내고 싶습니다.", "basis": "value_judgment", "direction": "for",
                 "personal_relevance": "제 저녁 여가시간의 편안함을 중요하게 생각합니다.", "evidence": []},
                {"claim": "실내에서 흡연하지 못하면 이동이 번거로울 수 있습니다.", "basis": "inference", "direction": "against",
                 "personal_relevance": "저는 흡연자여서 이런 불편 가능성이 제 선택에 관련됩니다.",
                 "evidence": [{"evidence_id": "self", "quote": "저는 흡연자"}]}],
            "weighing": "흡연 이동의 불편 가능성보다 실내에서 편안하게 쉬는 가치를 더 중요하게 생각해 조건부로 찬성합니다.",
            "conditions": [{"condition": "실제로 이동이 크게 어렵다는 경험을 하게 된다면", "possible_change": "시행 방법에 대한 입장을 다시 검토할 수 있습니다."}],
            "uncertainties": ["제가 이용할 시설에 별도 흡연실이 있는지는 모릅니다."]}}


def make_record(response, packet):
    fixture = json.loads((ROOT / "tests/fixtures/policy_stance/synthetic_demo_inputs.json").read_text(encoding="utf-8"))
    row = fixture["training_records"][0]
    row["schema_version"] = 2
    row["question_id"] = "synthetic_public_argument_v2"
    row["question_sha256"] = analysis.text_sha("Synthetic v2 public-argument question")
    row["provenance"]["argument_contract_version"] = 2
    row["response"] = deepcopy(response)
    packet_body = {k: v for k, v in row["evidence_packet"].items() if k != "integrity_sha256"}
    packet_body["evidence_items"] = [{**item, "day": row["as_of_day"]} for item in packet["evidence_items"]]
    row["evidence_packet"] = dict(packet_body, integrity_sha256=analysis.digest(packet_body))
    design = fixture["design"]
    design.update(question_id=row["question_id"], question_sha256=row["question_sha256"],
                  measurement_contract={"record_schema_version": 2, "argument_contract_version": 2})
    return row, design


def test_public_structure_and_citations_do_not_claim_semantic_truth(response, packet):
    quality = validate_argument(response, packet)
    assert quality["structural_validity"] == "valid"
    assert quality["quality_status"] == "sufficient_for_review" and quality["quality_flags"] == []
    assert quality["counts"]["consideration_direction"] == {"for": 1, "against": 1, "uncertain": 0}
    assert quality["review_required"] is True
    assert any("entailment" in item for item in quality["review_scope"])
    # The source quotation is exact but does not entail this deliberately false claim.
    response["argument"]["personal_situation"][0]["claim"] = "저는 비흡연자입니다."
    assert validate_argument(response, packet)["review_required"] is True


def test_exact_quote_with_unsupported_logical_jump_still_requires_semantic_review(response, packet):
    response["argument"]["considerations"][0].update(
        claim="방문 기록이 있으므로 모든 시민은 무조건 정책에 반대한다는 결론입니다.", basis="inference", direction="against",
        evidence=[{"evidence_id": "visit", "quote": "당구장을 1회 방문했습니다."}])
    quality = validate_argument(response, packet)
    assert quality["structural_validity"] == "valid" and quality["review_required"] is True
    assert "logical_score" not in quality and "semantic_pass" not in quality
    assert any("consistency" in scope for scope in quality["review_scope"])


def test_sparse_opinion_stays_support_and_flags_do_not_remove_denominator(response, packet):
    response["argument"] = {"personal_situation": [], "considerations": [], "weighing": "", "conditions": [], "uncertainties": []}
    row, design = make_record(response, packet)
    row["argument_quality"] = validate_argument(row["response"], row["evidence_packet"])
    assert row["argument_quality"]["quality_status"] == "needs_enrichment"
    assert analysis.validate_record(row, design) == row
    assert analysis.derive_stance(row)["stance"] == "support"
    model = analysis.fit_model([row], design)
    report = analysis.analyze([row], design, model)
    assert report["coverage"]["expected_agent_periods"] == 16
    assert report["argument_quality"]["stance_counts_by_quality"]["needs_enrichment"]["support"] == 1
    assert report["argument_quality"]["semantic_review_completed"] is False
    group = next(g for g in report["groups"] if g["arm"] == "off" and g["period"] == "pre" and g["group_by"] == "all")
    assert group["stance_counts"]["support"] == 1 and group["stance_counts"]["neutral"] == 0
    assert group["argument_quality"]["expected_agents"] == 4


def test_no_minimum_length_forced_opposing_case_or_condition(response, packet):
    response["answer"] = response["stance_quote"] = "찬성"
    response["argument"]["personal_situation"][0]["claim"] = "흡연자"
    response["argument"]["considerations"] = [{"claim": "공기", "basis": "value_judgment", "direction": "for", "personal_relevance": "휴식", "evidence": []}]
    response["argument"].update(weighing="공기 우선", conditions=[], uncertainties=[])
    assert validate_argument(response, packet)["quality_flags"] == []


def test_recorded_fact_without_citation_is_flagged_not_rewritten(response, packet):
    response["argument"]["considerations"][0]["basis"] = "recorded_fact"
    quality = validate_argument(response, packet)
    assert "recorded_fact_without_citation" in quality["quality_flags"]
    assert response["stance"] == "support"
    assert response["argument"]["considerations"][0]["basis"] == "recorded_fact"


@pytest.mark.parametrize("edit", [
    lambda r: r.update(hidden_reasoning="unsupported private field"),
    lambda r: r["argument"].update(internal_steps=[]),
    lambda r: r["argument"]["considerations"][0].update(basis="proved_truth"),
    lambda r: r["argument"]["considerations"][0].update(direction="neutral"),
    lambda r: r["argument"]["personal_situation"][0]["evidence"][0].update(quote="없는 인용문"),
    lambda r: r["argument"]["personal_situation"][0]["evidence"][0].update(evidence_id="unknown"),
    lambda r: r["argument"]["personal_situation"].append(deepcopy(r["argument"]["personal_situation"][0])),
    lambda r: r["argument"]["considerations"][0].pop("personal_relevance"),
    lambda r: r["argument"].update(weighing=123),
    lambda r: r["argument"].update(conditions=[{"condition": "x", "possible_change": "y", "extra": "z"}]),
    lambda r: r.update(confidence=float("nan")),
    lambda r: r.update(answer="x" * (LIMITS["answer_chars"] + 1)),
])
def test_schema_and_exact_citation_errors_raise(response, packet, edit):
    edit(response)
    with pytest.raises(ValueError):
        validate_argument(response, packet)


@pytest.mark.parametrize("field", ["personal_situation", "considerations", "conditions", "uncertainties"])
def test_excess_items_rejected_without_truncating(response, packet, field):
    item = response["argument"][field][0]
    response["argument"][field] = [deepcopy(item) for _ in range(LIMITS[field] + 1)]
    with pytest.raises(ValueError, match="at most"):
        validate_argument(response, packet)


def test_response_text_contains_public_content_not_declared_direction_or_quality(response, packet):
    row, design = make_record(response, packet)
    text = analysis.response_text(row)
    assert response["argument"]["weighing"] in text
    assert response["argument"]["considerations"][0]["personal_relevance"] in text
    assert response["argument"]["conditions"][0]["possible_change"] in text
    assert "value_judgment" not in text and "inference" not in text
    mutated = deepcopy(row)
    mutated["response"]["stance"] = "oppose"
    mutated["response"]["confidence"] = 0.01
    for item in mutated["response"]["argument"]["considerations"]:
        item["direction"] = "uncertain"
        item["basis"] = "value_judgment"
    assert analysis.response_text(mutated) == text
    assert public_argument_text(row["response"]) == text
    # The same cited substring in two places is not weighted twice in v2.
    assert text.count("저는 흡연자이며 평일 저녁에 여가시간을 냅니다.") == 1


def test_v2_cluster_geometry_ignores_label_direction_and_quality_but_uses_argument_text(response, packet):
    row, design = make_record(response, packet)
    second = deepcopy(row)
    second.update(agent_id="b", record_id="synthetic_v2_b")
    body = {k: v for k, v in second["evidence_packet"].items() if k != "integrity_sha256"}
    body["agent_id"] = "b"
    second["evidence_packet"] = dict(body, integrity_sha256=analysis.digest(body))
    second["response"]["argument"]["weighing"] = "이동 경로의 편의와 시간을 가장 중요하게 생각합니다."
    original = analysis.fit_model([row, second], design, min_df=1, k=2)
    changed = deepcopy([row, second])
    for item in changed:
        item["response"]["stance"] = "oppose"
        item["response"]["confidence"] = 0.1
        for c in item["response"]["argument"]["considerations"]:
            c["direction"] = "uncertain"
    after = analysis.fit_model(changed, design, min_df=1, k=2)
    for key in ("terms", "idf", "centroids", "training_cluster_sizes"):
        assert original[key] == after[key]
    assert any("이동" in term for term in original["terms"])


def test_v1_v2_measurement_contracts_and_saved_quality_cannot_mix(response, packet):
    row, design = make_record(response, packet)
    assert analysis.validate_record(row, design)
    legacy_design = deepcopy(design)
    legacy_design.pop("measurement_contract")
    with pytest.raises(ValueError, match="cannot be pooled"):
        analysis.validate_record(row, legacy_design)
    legacy_row = deepcopy(row)
    legacy_row["schema_version"] = 1
    legacy_row["provenance"].pop("argument_contract_version")
    legacy_row["response"].pop("argument")
    with pytest.raises(ValueError, match="cannot be pooled"):
        analysis.select_records([row, legacy_row], design)
    row["argument_quality"] = validate_argument(row["response"], row["evidence_packet"])
    row["argument_quality"]["review_required"] = False
    with pytest.raises(ValueError, match="recomputed"):
        analysis.validate_record(row, design)


def test_quality_flags_are_reproducible_and_inputs_remain_untouched(response, packet):
    old_response, old_packet = deepcopy(response), deepcopy(packet)
    assert validate_argument(response, packet) == validate_argument(response, packet)
    assert response == old_response and packet == old_packet


def test_no_personal_experience_still_allows_conditional_value_judgment(response, packet):
    response["argument"]["personal_situation"] = []
    response["argument"]["considerations"] = [{"claim": "쾌적한 공간이라는 가치에는 찬성합니다.", "basis": "value_judgment", "direction": "for",
        "personal_relevance": "제 실제 이용 경험은 없으므로 조건부 판단입니다.", "evidence": []}]
    response["argument"]["uncertainties"] = ["방문 경험이 제공되지 않았습니다."]
    result = validate_argument(response, packet)
    assert result["structural_validity"] == "valid"
    row, design = make_record(response, packet)
    assert analysis.derive_stance(analysis.validate_record(row, design))["stance"] == "support"


def test_empty_evidence_and_short_uncertainty_are_valid_not_regeneration_requests(response):
    response.update(stance="uncertain", answer="경험이 없어 판단을 유보합니다.", stance_quote="판단을 유보합니다", reasons=[])
    response["argument"] = {"personal_situation": [], "considerations": [], "weighing": "자료가 없어 유보합니다.", "conditions": [], "uncertainties": ["이용 경험 자료가 없습니다."]}
    result = validate_argument(response, {"evidence_items": []})
    assert result["structural_validity"] == "valid" and result["input_evidence_count"] == 0
    assert result["limited_input_possible"] is True
    assert "not a failure verdict" in result["quality_interpretation"]
    assert response["stance"] == "uncertain"


def test_v2_post_argument_is_never_fitted_into_pre_clusters(response, packet):
    row, design = make_record(response, packet)
    before = analysis.fit_model([row], design, min_df=1)
    post = deepcopy(row)
    post.update(record_id="synthetic_post_v2", arm="on", run_id=design["runs"]["on"],
                period="post", as_of_day="2017-12-16", measurement_context="experienced")
    body = {k: v for k, v in post["evidence_packet"].items() if k != "integrity_sha256"}
    body.update(arm="on", run_id=design["runs"]["on"], through_day="2017-12-16", days=["2017-12-16"])
    for item in body["evidence_items"]:
        item["day"] = "2017-12-16"
    post["evidence_packet"] = dict(body, integrity_sha256=analysis.digest(body))
    post["response"]["argument"]["weighing"] = "시행 후에만 존재하는 별도의 구체적인 논점입니다."
    assert analysis.fit_model([row, post], design, min_df=1) == before


def test_machine_schema_v1_v2_matches_structural_contract(response, packet):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "data/experiments/no_smoking_zone/stance_record.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    row, _ = make_record(response, packet)
    row["argument_quality"] = validate_argument(row["response"], row["evidence_packet"])
    validator.validate(row)
    legacy = json.loads((ROOT / "tests/fixtures/policy_stance/synthetic_demo_inputs.json").read_text(encoding="utf-8"))["training_records"][0]
    validator.validate(legacy)
    row["response"]["argument"]["considerations"][0]["hidden_reasoning"] = "unexpected field"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(row)
