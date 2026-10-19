"""Held-out synthetic fixtures only: these scores are not empirical validation.

Training responses and evaluation annotations are distinct literal cases. No
real participants, GPU run, or regression-paper stance labels are represented.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.experiments import analyze_policy_stance as stance


def sealed(value):
    return dict(value, integrity_sha256=stance.digest(value))


@pytest.fixture
def design():
    return {
        "schema_version": 1, "policy_id": "indoor_sports_smoking_ban_2017",
        "question_id": "synthetic_heldout_question", "question_sha256": stance.text_sha("Fixture policy question"),
        "population_id": "synthetic_fixture_population", "population_role": "synthetic_residents",
        "population": {"a": {"smoking_status": "smoker", "sex": "female"},
                       "b": {"smoking_status": "non_smoker", "sex": "male"},
                       "c": {"smoking_status": "smoker", "sex": "male"},
                       "d": {"smoking_status": "unknown", "sex": "female"}},
        "runs": {"off": "fixture_off", "on": "fixture_on"},
        "policy_effective_date": "2017-12-03",
        "periods": {"pre": {"start": "2017-11-19", "end": "2017-12-02"},
                    "post": {"start": "2017-12-03", "end": "2017-12-16"}},
        "group_attributes_as_of": "2017-11-18", "group_by": ["smoking_status", "sex"],
        "synthetic_fixture": True,
    }


def record(design, agent="a", *, arm="off", period="pre", label="support", answer="Clean air protects health", confidence=0.9, status="answered", as_of=None):
    when = as_of or design["periods"][period]["end"]
    rid = f"fixture_{arm}_{period}_{agent}_{when}"
    evidence = {"evidence_id": rid + "_e", "day": when, "kind": "stated_rationale",
                "text": "Public modeled statement: the room air was noticeable.",
                "value": {"subjective_unverified": True}, "source_ref": "synthetic_fixture"}
    packet = sealed({"schema_version": 1, "kind": "grounded_interview_packet", "run_id": design["runs"][arm],
                     "arm": arm, "agent_id": agent, "through_day": when, "days": [when],
                     "cohort_sha256": stance.digest(sorted(design["population"])), "source_sha256": stance.text_sha("synthetic fixture simulator"),
                     "evidence_items": [evidence], "missing_days": [], "limitations": ["Synthetic test fixture only"]})
    return {"schema_version": 1, "record_id": rid, "run_id": design["runs"][arm], "arm": arm, "agent_id": agent,
            "policy_id": design["policy_id"], "as_of_day": when, "period": period,
            "question_id": design["question_id"], "question_sha256": design["question_sha256"],
            "measurement_context": "experienced" if arm == "on" and period == "post" else "hypothetical",
            "response_status": status,
            "response": {"stance": label, "answer": answer, "stance_quote": answer, "confidence": confidence, "reasons": []} if status == "answered" else None,
            "evidence_packet": packet,
            "provenance": {"model_id": "synthetic_fixture_no_model_executed", "call_id": rid,
                           "prompt_sha256": stance.text_sha("fixture prompt"), "request_sha256": stance.text_sha("fixture request"),
                           "response_sha256": stance.text_sha("fixture response"), "source": "structured_policy_feedback", "synthetic_fixture": True}}


@pytest.fixture
def training(design):
    return [record(design, "a", answer="Clean air health protection breathing", label="support"),
            record(design, "b", answer="Clean air health protection comfort", label="uncertain"),
            record(design, "c", answer="Autonomy choice freedom restriction", label="oppose"),
            record(design, "d", answer="Autonomy choice freedom inconvenience", label="mixed")]


def fitted(design, records):
    return stance.fit_model(records, design, k=2, min_df=1)


def truth_base(report):
    return {"kind": "individual_stance_labels", "independent": True, "partition_role": "holdout", "synthetic_fixture": True,
            "source": {"id": "literal_synthetic_holdout_annotations_v1", "sha256": stance.text_sha("independent test annotations"),
                       "annotation_or_measurement_protocol": "Literal held-out synthetic assertions; not measured empirical accuracy"},
            "estimand": deepcopy(report["estimand"]), "label_basis": "independent_annotation_of_same_responses"}


def test_validation_exact_citations_identity_and_self_report(design):
    row = record(design)
    item = row["evidence_packet"]["evidence_items"][0]
    row["response"]["reasons"] = [{"evidence_id": item["evidence_id"], "quote": "room air"}]
    assert stance.validate_record(row, design) is row
    assert stance.derive_stance(row)["stance"] == "support"  # smoker is allowed to support
    row["response"]["reasons"][0]["quote"] = "clean air made me support the policy"
    with pytest.raises(ValueError, match="quote differs"):
        stance.validate_record(row, design)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["response"].update(stance_quote="absent words"), "quote"),
    (lambda r: r.update(measurement_context="experienced"), "context"),
    (lambda r: r.update(question_sha256="a" * 64), "question"),
    (lambda r: r["provenance"].update(synthetic_fixture=False), "fixture"),
    (lambda r: r["response"].update(confidence=float("nan")), "confidence"),
    (lambda r: r["response"].update(reasons=["invented"]), "Reason"),
])
def test_bad_records_fail(design, mutation, match):
    row = record(design)
    mutation(row)
    with pytest.raises(ValueError, match=match):
        stance.validate_record(row, design)


@pytest.mark.parametrize("change,match", [
    ({"agent_id": "b"}, "Foreign"),
    ({"through_day": "2017-12-16"}, "Future"),
    ({"days": ["2017-12-16"]}, "Future"),
    ({"cohort_sha256": "bad"}, "cohort"),
    ({"evidence_items": [None]}, "object"),
])
def test_even_resealed_foreign_future_malformed_packets_fail(design, change, match):
    row = record(design)
    packet = {k: v for k, v in row["evidence_packet"].items() if k != "integrity_sha256"}
    packet.update(change)
    row["evidence_packet"] = sealed(packet)
    with pytest.raises(ValueError, match=match):
        stance.validate_record(row, design)


def test_integrity_and_missing_stances_are_not_neutral(design):
    row = record(design, label="neutral")
    assert stance.derive_stance(row)["stance"] == "neutral"
    assert stance.derive_stance(None)["stance"] == "insufficient_evidence"
    assert stance.derive_stance(record(design, status="no_response"))["stance"] == "insufficient_evidence"
    assert stance.derive_stance(record(design, status="error"))["stance"] == "unknown"
    low = stance.derive_stance(record(design, confidence=0.2, label="oppose"))
    assert low["stance"] == "insufficient_evidence" and low["reported_stance"] == "oppose"
    wrapped = sealed(row)
    wrapped["response"]["answer"] = "Changed"
    with pytest.raises(ValueError, match="integrity"):
        stance.validate_record(wrapped)


def test_actual_unsupervised_clusters_and_unresolved_text(design, training):
    model = fitted(design, training)
    assert model["status"] == "fitted" and model["effective_k"] == 2 and model["converged"]
    ids = [stance.assign_cluster(row, model)["cluster_id"] for row in training]
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]
    assert stance.derive_stance(training[1])["stance"] == "uncertain"
    assert sum(model["training_cluster_sizes"].values()) == 4
    assert model == fitted(design, list(reversed(training)))


def test_post_heldout_data_never_affects_frozen_clusters(design, training):
    before = fitted(design, training)
    heldout = record(design, "a", arm="on", period="post", answer="Unseen extraterrestrial zygotes", label="neutral")
    assert fitted(design, training + [heldout]) == before
    assert stance.assign_cluster(heldout, before)["cluster_status"] == "no_reference_vocabulary"
    familiar = record(design, "b", arm="on", period="post", answer="Clean air health protection", label="support")
    assert stance.assign_cluster(familiar, before)["cluster_id"] == stance.assign_cluster(training[0], before)["cluster_id"]
    tampered = deepcopy(before)
    tampered["seed"] += 1
    with pytest.raises(ValueError, match="integrity"):
        stance.analyze(training, design, tampered)


def test_demographics_do_not_change_stance_or_text_clustering(design, training):
    original = fitted(design, training)
    changed = deepcopy(design)
    for attrs in changed["population"].values():
        attrs["smoking_status"] = "opposite arbitrary fixture label"
    altered = fitted(changed, training)
    assert original["centroids"] == altered["centroids"]
    assert original["terms"] == altered["terms"]
    assert stance.analyze(training, changed, altered)["agents"][0]["stance"] == "support"


def test_declared_labels_and_confidence_never_fit_clusters(design, training):
    before = fitted(design, training)
    permuted = deepcopy(training)
    for index, row in enumerate(permuted):
        row["response"]["stance"] = ("neutral", "support", "unknown", "uncertain")[index]
        row["response"]["confidence"] = 0.01
    after = fitted(design, permuted)
    # Provenance hashes change, but the learned text representation does not.
    for key in ("terms", "idf", "centroids", "training_cluster_sizes"):
        assert before[key] == after[key]


def test_missing_denominators_and_longitudinal_cells(design, training):
    post = record(design, "a", arm="on", period="post", label="oppose", answer="Autonomy choice freedom")
    report = stance.analyze(training + [post], design, fitted(design, training), min_group_size=1)
    assert report["coverage"] == {"expected_agent_periods": 16, "selected_actual_records": 5, "missing_agent_periods": 11}
    on_pre = next(g for g in report["groups"] if g["arm"] == "on" and g["period"] == "pre" and g["group_by"] == "all")
    assert on_pre["stance_counts"]["insufficient_evidence"] == 4 and on_pre["stance_counts"]["oppose"] == 0
    assert on_pre["support_missingness_bounds"] == {"lower": 0, "upper": 1}
    on_post = next(g for g in report["groups"] if g["arm"] == "on" and g["period"] == "post" and g["group_by"] == "all")
    assert on_post["shares_all_expected"]["oppose"] == 0.25
    assert on_post["shares_among_resolved"]["oppose"] == 1
    assert any(c["top_reference_terms"] and c["frozen_attribute_counts"] for c in report["clusters"])
    assert next(t for t in report["transitions"] if t["arm"] == "on")["counts"] == [
        {"pre": "insufficient_evidence", "post": "insufficient_evidence", "agents": 3},
        {"pre": "insufficient_evidence", "post": "oppose", "agents": 1}]


def test_latest_within_period_and_duplicate_disagreement(design):
    early = record(design, as_of="2017-11-20", label="oppose")
    late = record(design, label="support")
    selected = stance.select_records([late, early, early], design)
    assert selected["off", "pre", "a"] == late
    other = deepcopy(late)
    other["record_id"] = "ambiguous_retry"
    with pytest.raises(ValueError, match="Multiple assessments"):
        stance.select_records([late, other], design)


def test_independent_heldout_annotation_metrics_count_abstention(design, training):
    heldout = [record(design, "a", arm="on", period="post", label="support", answer="Clean air health"),
               record(design, "b", arm="on", period="post", label="oppose", answer="Autonomy choice", confidence=0.1)]
    report = stance.analyze(training + heldout, design, fitted(design, training))
    truth = truth_base(report)
    truth.update(classes=["support", "oppose"], labels=[
        {"agent_id": "a", "arm": "on", "period": "post", "stance": "support", "record_id": "fixture_on_post_a_2017-12-16", "measurement_context": "experienced"},
        {"agent_id": "b", "arm": "on", "period": "post", "stance": "oppose", "record_id": "fixture_on_post_b_2017-12-16", "measurement_context": "experienced"}])
    result = stance.evaluate(report, truth)
    assert result["metrics"] == {"macro_f1": 0.5, "balanced_accuracy": 0.5, "accuracy": 0.5}
    assert result["coverage"] == 0.5 and result["confusion_matrix"]["oppose"]["insufficient_evidence"] == 1
    truth["labels"][1]["record_id"] = "foreign_response"
    assert stance.evaluate(report, truth)["status"] == "abstained"


def test_pdf_missing_independence_estimand_mismatch_abstain(design, training):
    report = stance.analyze(training, design, fitted(design, training))
    assert stance.evaluate(report)["metrics"] is None
    pdf = json.loads((ROOT / "data/experiments/no_smoking_zone/ground_truth.json").read_text(encoding="utf-8"))
    assert stance.evaluate(report, pdf)["status"] == "abstained"
    truth = truth_base(report)
    truth["independent"] = False
    assert "Independent" in stance.evaluate(report, truth)["reason"]
    truth["independent"] = True
    truth["estimand"]["population_role"] = "observed_facility_owners"
    assert "population" in stance.evaluate(report, truth)["reason"]


def test_group_same_estimand_denominator_gate(design, training):
    report = stance.analyze(training, design, fitted(design, training), min_group_size=1)
    truth = truth_base(report)
    truth.update(kind="group_stance_statistics", groups=[{
        "arm": "off", "period": "pre", "group_by": "all", "group": "all", "measurement_context": "hypothetical",
        "denominator": "all_expected", "n": 4,
        "counts": {"support": 1, "oppose": 1, "mixed": 1, "neutral": 0, "uncertain": 1, "unknown": 0, "insufficient_evidence": 0}}])
    result = stance.evaluate(report, truth)
    assert result["comparisons"][0]["total_variation_distance"] == 0
    truth["groups"][0]["n"] = 300
    assert stance.evaluate(report, truth)["status"] == "abstained"


def test_directory_records_are_sealed_and_invalid_input_not_silently_skipped(tmp_path, design):
    root = tmp_path / "collection"
    (root / "records").mkdir(parents=True)
    row = sealed(record(design))
    (root / "records/a.json").write_text(json.dumps(row), encoding="utf-8")
    (root / "collection_manifest.json").write_text('{"not":"a response"}', encoding="utf-8")
    assert stance.load_records([root]) == [row]
    row["response"]["confidence"] = 0.4
    (root / "records/a.json").write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        stance.load_records([root])


def test_fewer_than_two_training_responses_abstains_clustering(design):
    model = fitted(design, [record(design)])
    assert model["effective_k"] == 0 and model["status"] == "insufficient_training_responses"
    assert stance.assign_cluster(record(design), model)["cluster_id"] is None


def test_cli_immutable_fit_analyze_evaluate(tmp_path, design, training):
    design_file, records_file = tmp_path / "design.json", tmp_path / "records.jsonl"
    design_file.write_text(json.dumps(design), encoding="utf-8")
    records_file.write_text("\n".join(json.dumps(row) for row in training), encoding="utf-8")
    model_file, report_file, eval_file = [tmp_path / n for n in ("model.json", "report.json", "evaluation.json")]
    stance.main(["fit", "--design", str(design_file), "--records", str(records_file), "--k", "2", "--out", str(model_file)])
    stance.main(["analyze", "--design", str(design_file), "--records", str(records_file), "--model", str(model_file), "--out", str(report_file)])
    stance.main(["evaluate", "--report", str(report_file), "--out", str(eval_file)])
    assert json.loads(eval_file.read_text())["status"] == "abstained"
    with pytest.raises(FileExistsError):
        stance.write_new(eval_file, {})


def test_archived_synthetic_demo_golden_expectations():
    base = ROOT / "tests/fixtures/policy_stance"
    data = json.loads((base / "synthetic_demo_inputs.json").read_text(encoding="utf-8"))
    expected = json.loads((base / "synthetic_demo_expected.json").read_text(encoding="utf-8"))
    assert data["synthetic_fixture"] is expected["synthetic_fixture"] is True
    assert stance.digest(data) == expected["input_sha256"]
    design, training = data["design"], data["training_records"]
    model = fitted(design, training)
    assert model["fit_partition"] == expected["fit_partition"]
    assert model["effective_k"] == expected["effective_k"]
    clusters = {row["agent_id"]: stance.assign_cluster(row, model)["cluster_id"] for row in training}
    for a, b in expected["same_cluster_groups"]:
        assert clusters[a] == clusters[b]
    a, b = expected["different_clusters"]
    assert clusters[a] != clusters[b]
    report = stance.analyze(training + data["heldout_records"], design, model)
    for key in ("expected_agent_periods", "selected_actual_records"):
        assert report["coverage"][key] == expected[key]
    evaluated = stance.evaluate(report, data["independent_test_annotations"])
    assert evaluated["metrics"] == expected["heldout_metrics"]
    assert evaluated["coverage"] == expected["heldout_coverage"]
    assert evaluated["confusion_matrix"]["oppose"]["insufficient_evidence"] == expected["heldout_oppose_to_insufficient_evidence"]


def test_design_from_verified_run_identities_and_frozen_bundle(tmp_path, monkeypatch):
    from scripts.experiments import no_smoking_zone, collect_policy_stances
    from scripts.sim import interview_evidence
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    runtime = {"cohort": [{"id": "a", "age": 34, "sex": "female", "smoking_status": "smoker"}]}
    people = [{"agent_id": "a", "personal": {"income_level": "middle"}}]
    for name, value in (("runtime.json", runtime), ("personas.json", people), ("bundle.json", {})):
        (bundle / name).write_text(json.dumps(value), encoding="utf-8")
    for arm in ("off", "on"):
        folder = tmp_path / arm
        folder.mkdir()
        (folder / "experiment_run.json").write_text(json.dumps({"arm": arm, "status": "complete", "start": "2017-11-19", "days": 28,
            "cohort_ids": ["a"], "runtime_sha256": stance.digest(runtime)}), encoding="utf-8")
    monkeypatch.setattr(no_smoking_zone, "inspect_bundle", lambda _: {"blockers": []})
    def fake_packet(folder, aid, through_day, *, days):
        assert through_day == "2017-11-19" and days == [through_day]
        return sealed({"run_id": "original_archived_" + folder.name, "arm": folder.name,
                       "cohort_sha256": stance.digest(["a"]), "source_sha256": "a" * 64})
    monkeypatch.setattr(interview_evidence, "build_packet", fake_packet)
    result = stance.design_from_runs(bundle, tmp_path / "off", tmp_path / "on")
    assert result["runs"] == {"off": "original_archived_off", "on": "original_archived_on"}
    assert result["question_sha256"] == collect_policy_stances.QUESTION_SHA256
    assert result["population"]["a"]["income"] == "middle"
    manifest = tmp_path / "on/experiment_run.json"
    raw = json.loads(manifest.read_text())
    raw["days"] = 1
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="28-day"):
        stance.design_from_runs(bundle, tmp_path / "off", tmp_path / "on")
