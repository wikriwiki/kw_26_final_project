"""Synthetic fixtures only; no empirical frame is fabricated for production."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("population_cohort", Path(__file__).resolve().parents[2] / "tools/freeze_population_matched_cohort.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / "synthetic_test_evidence.txt"
    source.write_text("synthetic only", encoding="utf-8")
    evidence = {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    values = {"sex": ("M", "F"), "age_band": ("20", "30"), "admin_dong": ("A", "B"), "income_band": ("low", "high")}
    rows = []
    for i in range(32):
        row = {"aid": f"synthetic_{i:02d}"}
        for bit, field in enumerate(MODULE.FIELDS):
            row[field] = values[field][(i >> bit) & 1]
        rows.append(row)
    frame = {"schema": "population_calibration_frame_v1", "population_unit": "resident_person", "reference_year": 2025,
             "targets": {f: {"proportions": {k: .5 for k in cats}, "source": evidence, "population_unit": "resident_person",
                              "reference_year": 2025, "source_population_unit": "resident_person", "source_reference_year": 2025,
                              "field_definition": "synthetic test definition"} for f, cats in values.items()}}
    candidates = {"schema": "population_candidates_v1", "population_unit": "resident_person", "reference_year": 2025,
                  "admin_dong_code_system": "synthetic_admin_2025",
                  "admin_dong_assignment_audit": {"uses_actual_residence_anchor": True,
                                                  "official_crosswalk_verified": True, "source_evidence": [evidence]},
                  "income_candidate_origin": MODULE.INCOME_ORIGIN, "income_assignment_method": "synthetic tests only",
                  "income_assignment_evidence": [evidence], "source_evidence": [evidence], "rows": rows,
                  "income_assignment_audit": {"official_target_sha256": evidence["sha256"], "policy_outcome_used": False,
                                              "uses_spending_decile_as_income": False, "distribution_validated": True,
                                              "income_definition": "synthetic person-month income"}}
    frame["targets"]["admin_dong"]["code_system"] = "synthetic_admin_2025"
    return tmp_path, frame, candidates


def test_balanced_four_fields_and_deterministic(fixture):
    root, frame, candidates = fixture
    a = MODULE.calibrate(frame, candidates, root, sample_size=32)
    b = MODULE.calibrate(frame, candidates, root, sample_size=32)
    assert a == b and a["model_calls_allowed"]
    assert a["unweighted_max_absolute_error"] == 0
    assert a["weighted_max_absolute_error"] == 0
    assert a["effective_sample_size"] == 32
    assert not a["joint_distribution_verified"]


@pytest.mark.parametrize("origin", ["spending_anchors", "LLM", "stable_persona_spending_anchors", None])
def test_non_empirical_income_blocked(fixture, origin):
    root, frame, candidates = fixture
    candidates["income_candidate_origin"] = origin
    with pytest.raises(MODULE.CalibrationError, match="income"):
        MODULE.calibrate(frame, candidates, root, sample_size=32)


def test_missing_field_and_zero_support_fail(fixture):
    root, frame, candidates = fixture
    del frame["targets"]["income_band"]
    with pytest.raises(MODULE.CalibrationError, match="exactly"):
        MODULE.calibrate(frame, candidates, root, sample_size=32)


def test_source_tamper_fail(fixture):
    root, frame, candidates = fixture
    (root / "synthetic_test_evidence.txt").write_text("changed")
    with pytest.raises(MODULE.CalibrationError, match="SHA256"):
        MODULE.calibrate(frame, candidates, root, sample_size=32)


def test_population_unit_and_year_mismatch_fail(fixture):
    root, frame, candidates = fixture
    candidates["population_unit"] = "household"
    with pytest.raises(MODULE.CalibrationError, match="unit"):
        MODULE.calibrate(frame, candidates, root, sample_size=32)


def test_missing_positive_cell_fail(fixture):
    root, frame, candidates = fixture
    candidates["rows"] = [r for r in candidates["rows"] if r["sex"] == "M"]
    with pytest.raises(MODULE.CalibrationError, match="missing positive"):
        MODULE.calibrate(frame, candidates, root, sample_size=16)


def test_small_sample_support_fail(fixture):
    root, frame, candidates = fixture
    with pytest.raises(MODULE.CalibrationError, match="cannot support"):
        MODULE.calibrate(frame, candidates, root, sample_size=1)


def test_raking_does_not_hide_raw_roster_mismatch(fixture):
    root, frame, candidates = fixture
    frame["targets"]["sex"]["proportions"] = {"M": .8, "F": .2}
    result = MODULE.calibrate(frame, candidates, root, sample_size=32)
    assert result["weighted_max_absolute_error"] < 1e-8
    assert result["unweighted_max_absolute_error"] == pytest.approx(.3)
    assert not result["model_calls_allowed"]


def test_conflicting_joint_support_does_not_converge(fixture):
    root, frame, candidates = fixture
    candidates["rows"] = [r for r in candidates["rows"] if (r["sex"] == "M") == (r["age_band"] == "20")]
    frame["targets"]["sex"]["proportions"] = {"M": .8, "F": .2}
    with pytest.raises(MODULE.CalibrationError, match="did not converge"):
        MODULE.calibrate(frame, candidates, root, sample_size=16)


def test_absent_frame_fail(fixture):
    root, _, candidates = fixture
    with pytest.raises(MODULE.CalibrationError, match="frame"):
        MODULE.calibrate({}, candidates, root, sample_size=32)


def test_calibrated_synthetic_income_explicitly_supported(fixture):
    root, frame, candidates = fixture
    candidates["income_candidate_origin"] = "calibrated_synthetic_income_assignment"
    result = MODULE.calibrate(frame, candidates, root, sample_size=32)
    assert result["model_calls_allowed"]
    assert result["income_candidate_origin"] == "calibrated_synthetic_income_assignment"


def test_household_income_requires_conversion_audit(fixture):
    root, frame, candidates = fixture
    frame["targets"]["income_band"]["source_population_unit"] = "household"
    with pytest.raises(MODULE.CalibrationError, match="conversion audit"):
        MODULE.calibrate(frame, candidates, root, sample_size=32)


def test_income_masquerading_spending_decile_rejected(fixture):
    root, frame, candidates = fixture
    candidates["income_assignment_audit"]["uses_spending_decile_as_income"] = True
    with pytest.raises(MODULE.CalibrationError, match="income donor"):
        MODULE.calibrate(frame, candidates, root, sample_size=32)


def test_id_dong_cannot_replace_residence_anchor(fixture):
    root, frame, candidates = fixture
    candidates["admin_dong_assignment_audit"]["uses_actual_residence_anchor"] = False
    with pytest.raises(MODULE.CalibrationError, match="residence-anchor"):
        MODULE.calibrate(frame, candidates, root, sample_size=32)


def test_wrong_legal_admin_code_system_fail(fixture):
    root, frame, candidates = fixture
    candidates["admin_dong_code_system"] = "legal_dong_wrong_vintage"
    with pytest.raises(MODULE.CalibrationError, match="code system"):
        MODULE.calibrate(frame, candidates, root, sample_size=32)


def test_reference_year_transport_requires_evidence(fixture):
    root, frame, candidates = fixture
    frame["targets"]["income_band"]["source_reference_year"] = 2020
    with pytest.raises(MODULE.CalibrationError, match="conversion audit"):
        MODULE.calibrate(frame, candidates, root, sample_size=32)


def test_duplicate_ids_are_not_sample_size(fixture):
    root, frame, candidates = fixture
    candidates["rows"][1]["aid"] = candidates["rows"][0]["aid"]
    with pytest.raises(MODULE.CalibrationError, match="unique"):
        MODULE.calibrate(frame, candidates, root, sample_size=32)


def test_nonfinite_target_rejected(fixture):
    root, frame, candidates = fixture
    frame["targets"]["sex"]["proportions"]["M"] = float("nan")
    with pytest.raises(MODULE.CalibrationError, match="invalid proportions"):
        MODULE.calibrate(frame, candidates, root, sample_size=32)


def test_cli_missing_real_frame_produces_blocked_result(tmp_path):
    import json
    import subprocess
    import sys
    output = tmp_path / "blocked.json"
    command = [sys.executable, str(SPEC.origin), "--frame", str(tmp_path / "missing_frame.json"),
               "--candidates", str(tmp_path / "missing_candidates.json"), "--citizens", "40", "--out", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 2
    result = json.loads(output.read_text(encoding="utf8"))
    assert result["model_calls_allowed"] is False and result["model_calls"] == 0
    assert result["graph_modified"] is False and result["status"] == "blocked"
