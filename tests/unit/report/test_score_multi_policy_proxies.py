import hashlib
import json

import pytest

from scripts.report.score_multi_policy_proxies import (
    apply_cashback_month, audit_arm_quality, audit_preperiod_balance,
    read_pair, score_ledger,
)
from scripts.report.audit_stage2_generation import inspect as inspect_stage2


def _row(arm, *, policy_id, aid="a", day="2020-08-01", on=False):
    sub = ({"가구": 20, "미용실": 15, "슈퍼마켓": 50, "식료품": 20,
            "청과": 15, "정육": 10} if on else
           {"가구": 10, "미용실": 10, "슈퍼마켓": 40, "식료품": 20,
            "청과": 10, "정육": 10})
    offline = sum(sub.values())
    return {"aid": aid, "day": day, "arm": arm, "policy_id": policy_id,
            "offline_spent": offline, "online_spent": 25 if on else 20,
            "total_spent": offline + (25 if on else 20),
            "sangsaeng_eligible_offline_spent": 100 if on else 80,
            "unclassified_won": 0,
            "policy_funded_won": 30 if on and policy_id == "P010" else 0,
            "grant_received_cumulative": 200 if on and policy_id == "P010" else 0,
            "grant_remaining": 170 if on and policy_id == "P010" else 0,
            "by_sub": sub,
            "by_l1": {"쇼핑": sub["가구"], "미용": sub["미용실"],
                      "마트": offline - sub["가구"] - sub["미용실"]},
            "funded_by_sub": {"슈퍼마켓": 30} if on and policy_id == "P010" else {}}


def test_proxy_arithmetic_keeps_units_and_missing_values():
    on, off = [_row("on", policy_id="P012", on=True)], [_row("off", policy_id=None)]
    scored = {row["id"]: row for row in score_ledger(on, off, ["a"], policy="P012", draws=0)}
    assert scored["P012-1"]["simulation"] == pytest.approx(25.0)
    assert scored["P012-2"]["simulation"] == pytest.approx(25.0)
    assert scored["P012-5"]["simulation"] == pytest.approx(50.0)
    assert scored["P012-5"]["simulation_unit"] == "percentage points"
    assert scored["P012-4"]["simulation"] is None
    assert scored["P012-6"]["simulation"] is None
    assert all(row["estimand_alignment"] == "different" for row in scored.values())

    on, off = [_row("on", policy_id="P010", on=True)], [_row("off", policy_id=None)]
    p010 = {row["id"]: row for row in score_ledger(on, off, ["a"], policy="P010", draws=0)}
    mpc = p010["P010-1"]
    assert mpc["simulation"] == pytest.approx((130 + 25 - 100 - 20) / 200)
    assert mpc["simulation_unit"] == "ratio"
    assert p010["P010-BOK-MART_FOOD"]["simulation"] == pytest.approx(100)
    assert p010["P010-BOK-RESTAURANT"]["simulation"] == pytest.approx(0)
    assert p010["P010-BOK-MART_FOOD"]["exploratory_not_registered"] is True

    on, off = [_row("on", policy_id="P016", on=True)], [_row("off", policy_id=None)]
    p016 = {row["id"]: row for row in score_ledger(on, off, ["a"], policy="P016", draws=0)}
    assert p016["C1"]["simulation"] == pytest.approx(18.75)
    assert p016["C2"]["simulation"] == pytest.approx(0)
    assert p016["C3"]["simulation"] == pytest.approx(0)
    bad_mart = _row("on", policy_id="P016", on=True)
    bad_mart["by_l1"]["마트"] -= 1
    invalid_pair = {row["id"]: row for row in score_ledger(
        [bad_mart], off, ["a"], policy="P016", draws=0)}
    assert invalid_pair["C1"]["simulation"] is not None
    assert invalid_pair["C2"]["simulation"] is None
    assert invalid_pair["C3"]["simulation"] is None

    on, off = [_row("on", policy_id=None, on=True)], [_row("off", policy_id=None)]
    distancing = {row["id"]: row for row in score_ledger(on, off, ["a"],
                                                           policy="DISTANCING_2020", draws=0)}
    assert distancing["DS-1"]["simulation"] is None  # No 한식 spending in OFF.
    assert distancing["DS-2"]["simulation"] == pytest.approx(100 * 25 / 90)
    assert distancing["DS-6"]["simulation"] is None  # Missing spatial hub type.


def test_pair_reader_rejects_changed_evidence(tmp_path):
    on_file, off_file = tmp_path / "on.sector.ledger.jsonl", tmp_path / "off.sector.ledger.jsonl"
    model_evidence = tmp_path / "served_model_evidence.json"
    model_evidence.write_text('{"model":"EXAONE"}', encoding="utf-8")
    model_evidence_sha = hashlib.sha256(model_evidence.read_bytes()).hexdigest()
    roster_digest = hashlib.sha256(json.dumps(["a"], ensure_ascii=False).encode()).hexdigest()
    for arm, path in (("on", on_file), ("off", off_file)):
        row = _row(arm, policy_id="P010" if arm == "on" else None, on=arm == "on")
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = {"schema": "multi_policy_sector_ledger_v1", "arm": arm,
                    "policy_id": "P010" if arm == "on" else None,
                    "policy_input": {"path": "frozen/p010.json", "sha256": "e" * 64}
                                    if arm == "on" else None,
                    "served_model_provenance": {
                        "model_id": "LGAI-EXAONE/EXAONE-4.5-33B-AWQ",
                        "evidence_sha256": model_evidence_sha},
                    "start": row["day"], "end": row["day"],
                    "effective_from": row["day"] if arm == "on" else None,
                    "effective_until": row["day"] if arm == "on" else None,
                    "citizens": 1, "days": 1, "rows": 1,
                    "roster_sha256": roster_digest,
                    "cohort_sha256": {row["day"]: "c" * 64},
                    "prompt_provenance": {"prompt_variant": "v53",
                                          "run_id": f"{arm}-run",
                                          "execution_fingerprint": f"{arm}-execution",
                                          "requested_model_id": "Qwen/Qwen3-8B",
                                          "system_prompt_sha256": "a" * 64,
                                          "stage2_system_prompt_sha256": "b" * 64,
                                          "baseline_income_map_sha256": "d" * 64},
                    "provenance": {"execution_fingerprint": [f"{arm}-execution"],
                                   "paired_environment_fingerprint": ["same-environment"],
                                   "source_fingerprint": ["same-source"],
                                   "experience_environment_id": ["same-env"],
                                   "experience_run_id": [f"{arm}-run"]},
                    "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        path.with_name(path.name + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    read_pair(on_file, off_file, policy="P010")
    off_manifest = off_file.with_name(off_file.name + ".manifest.json")
    payload = json.loads(off_manifest.read_text(encoding="utf-8"))
    payload["prompt_provenance"]["requested_model_id"] = "another-model"
    off_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="requested model IDs"):
        read_pair(on_file, off_file, policy="P010")
    payload["prompt_provenance"]["requested_model_id"] = "Qwen/Qwen3-8B"
    off_manifest.write_text(json.dumps(payload), encoding="utf-8")
    on_file.write_text(on_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        read_pair(on_file, off_file, policy="P010")


def test_monthly_cashback_values_require_v53_complete_month(tmp_path):
    on, off = [_row("on", policy_id="P012", on=True)], [_row("off", policy_id=None)]
    indicators = score_ledger(on, off, ["a"], policy="P012", draws=0)
    payload = {"policy_id": "P012", "month": "2021-10", "days": 31,
               "citizens": 4, "recipients": 4, "capped_recipients": 1,
               "total_cashback_accrued_won": 4936,
               "complete_paired_matrix": True,
               "provenance": {"prompt_variant": "v53"},
               "metrics": {
                   "cashback_per_recipient_won": {"value": 1234.0,
                                                   "citizen_bootstrap_95_interval": [1000, 1500]},
                   "cap_share_recipients": {"value": 0.25,
                                            "citizen_bootstrap_95_interval": [0.1, 0.4]},
               }}
    path = tmp_path / "monthly.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    updated = {row["id"]: row for row in apply_cashback_month(indicators, path,
                                                                expected_citizens=4)}
    assert updated["P012-4"]["simulation"] == 1234
    assert updated["P012-6"]["simulation"] == 25
    assert updated["P012-6"]["ci"] == [10, 40]
    assert updated["P012-6"]["n"] == 4
    assert updated["P012-6"]["empirical_variant"] == "october_only"
    payload["provenance"]["prompt_variant"] = "v5"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="v53"):
        apply_cashback_month(indicators, path, expected_citizens=4)
    payload["provenance"] = {"prompt_variant": "v53", "system_prompt_sha256": "a" * 64,
                             "stage2_system_prompt_sha256": "b" * 64,
                             "baseline_income_map_sha256": "d" * 64,
                             "arms": {"on": {"run_id": "on-run",
                                             "execution_fingerprint": "on-execution"},
                                      "off": {"run_id": "off-run",
                                              "execution_fingerprint": "off-execution"}}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    on_manifest = {"prompt_provenance": {"system_prompt_sha256": "a" * 64,
                                          "stage2_system_prompt_sha256": "b" * 64,
                                          "baseline_income_map_sha256": "d" * 64,
                                          "run_id": "on-run",
                                          "execution_fingerprint": "on-execution"}}
    off_manifest = {"prompt_provenance": {**on_manifest["prompt_provenance"],
                                           "run_id": "off-run",
                                           "execution_fingerprint": "off-execution"}}
    apply_cashback_month(indicators, path, expected_citizens=4,
                         sector_manifests=(on_manifest, off_manifest))
    payload["provenance"]["arms"]["off"]["run_id"] = "another-run"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sector off run identity"):
        apply_cashback_month(indicators, path, expected_citizens=4,
                             sector_manifests=(on_manifest, off_manifest))


def test_quality_and_preperiod_audits_do_not_hide_retries(tmp_path):
    day = "2025-07-19"
    on_file = tmp_path / "on.sector.ledger.jsonl"
    off_file = tmp_path / "off.sector.ledger.jsonl"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    metric = {"aid": "a", "status": "ok", "s1_attempts": 2,
              "s1_timing": {"attempts": [{"status": "error", "error_stage": "rule_validate"},
                                        {"status": "ok"}]},
              "s2_timing": {"n_llm_calls": 1, "attempts": [{"status": "ok"}]},
              "s2_skipped": False}
    metrics_file = metrics_dir / f"day_{day}.jsonl"
    metrics_file.write_text(json.dumps(metric) + "\n", encoding="utf-8")
    attempts_dir = metrics_dir / "attempts"
    attempts_dir.mkdir()
    raw_snapshot = attempts_dir / f"day_{day}_1.jsonl"
    raw_snapshot.write_text(json.dumps({"aid": "a", "status": "error",
                                        "error": "Stage1 failed after 3 attempts"}) + "\n"
                            + json.dumps(metric) + "\n", encoding="utf-8")
    stage2 = inspect_stage2({day: [metric]}, expected_per_day=1)
    (tmp_path / "stage2.json").write_text(json.dumps(stage2), encoding="utf-8")
    manifest = {"start": day, "end": day, "citizens": 1,
                "metrics_sha256": {day: hashlib.sha256(metrics_file.read_bytes()).hexdigest()}}
    quality, proof = audit_arm_quality(on_file, manifest)
    assert quality["stage1_first_attempt_internal_validation_pass_rate"] == 0
    assert quality["stage1_final_ok_count"] == 1
    assert quality["stage1_outer_retry_recovered_count"] == 1
    assert quality["stage1_successful_invocation_first_attempt_pass_rate"] == 0
    assert quality["stage1_first_raw_strict_format_pass_rate"] is None
    assert quality["stage2_choice_repair_rate"] == 0
    assert len(proof) == 3

    pre_days = [day, "2025-07-20"]
    for file, amount, offline in ((on_file, 110, 100), (off_file, 100, 90)):
        file.write_text("".join(json.dumps({"day": d, "total_spent": amount,
                                              "offline_spent": offline}) + "\n"
                                for d in pre_days), encoding="utf-8")
    balance = audit_preperiod_balance(on_file, off_file, policy="P010",
                                      on_manifest={"start": day, "end": "2025-07-20",
                                                   "effective_from": "2025-07-21",
                                                   "citizens": 1})
    assert balance["status"] == "pass"
    on_file.write_text("".join(json.dumps({"day": d, "total_spent": 150,
                                            "offline_spent": 140}) + "\n"
                              for d in pre_days), encoding="utf-8")
    imbalance = audit_preperiod_balance(on_file, off_file, policy="P010",
                                        on_manifest={"start": day, "end": "2025-07-20",
                                                     "effective_from": "2025-07-21",
                                                     "citizens": 1})
    assert imbalance["status"] == "fail"
    assert imbalance["post_effect_causal_interpretation_blocked"] is True
