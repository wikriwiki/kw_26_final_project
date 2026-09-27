"""Per-experiment reports cover the catalog without fabricating empirical gaps."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.report import build_experiment_comparison as report


ROOT = Path(__file__).resolve().parents[3]


def test_completed_suite_metadata_uses_october_denominators_without_rescoring():
    rows = []
    for key, policy in (("P010-1", "P010"), ("P012-1", "P012"),
                        ("P012-2", "P012"), ("P012-4", "P012"),
                        ("P012-5", "P012"), ("P012-6", "P012"),
                        ("DS-1", "DISTANCING_2020"), ("DS-2", "DISTANCING_2020"),
                        ("C2", "P016"), ("C3", "P016")):
        rows.append({"id": key, "policy": policy, "truth": 2.0, "simulation": 3.0,
                     "ci": [1.0, 4.0], "gap": None, "desc": "historical short pilot",
                     "expert_opinion": "historical opinion", "empirical_population": "16.8m",
                     "empirical_variant": "october_only" if key in ("P012-4", "P012-6") else None,
                     "run_context": {"policy_id": policy, "preperiod_balance": {
                         "status": "fail", "reason": "P010 offline spending"}}})
    original = {"experiment": "multi_policy_v53_20260928", "rows": rows,
                "score_files": [{"sha256": "a" * 64}], "run_evidence": [],
                "exploratory_pairs": [{"id": "P014-KIPF-47121", "policy": "LOCAL_VOUCHER",
                                        "truth": .141, "simulation": 2.224,
                                        "run_context": {"policy_id": "P014"}}]}
    untouched = deepcopy(original)
    updated = report.apply_completed_20260928_display_opinions(original)
    assert original == untouched
    by_id = {row["id"]: row for row in updated["rows"]}
    for before, after in zip(rows, updated["rows"]):
        for key in ("truth", "simulation", "ci", "gap"):
            assert before[key] == after[key]
        assert after["original_display_metadata"]["desc"] == "historical short pilot"
    assert "0.2082 로그포인트" in by_id["P012-1"]["desc"]
    assert "10월 전체" in by_id["P012-1"]["expert_opinion"]
    assert "0.3623" in by_id["P012-5"]["desc"]
    assert "8,102,000명" in by_id["P012-4"]["empirical_population"]
    assert "3,875억원" in by_id["P012-4"]["empirical_estimand"]
    assert "1,691,000명" in by_id["P012-6"]["empirical_estimand"]
    assert "고유 인원 수가 아님" in by_id["P012-6"]["empirical_population"]
    assert "11월 실제 지급을 관측하지 않음" in by_id["P012-4"]["empirical_period"]
    assert "28.49%" in by_id["P012-6"]["expert_opinion"]
    assert "한식 POI" in by_id["DS-1"]["expert_opinion"]
    assert "같은 3일" in by_id["DS-2"]["expert_opinion"]
    assert "분자=분모" in by_id["C3"]["expert_opinion"]
    assert "P010 offline" not in by_id["C2"]["run_context"]["preperiod_balance"]["reason"]
    assert "2010년 지역 GRDP" in updated["exploratory_pairs"][0]["empirical_estimand"]


def _ksic_failed_join_fixture(tmp_path):
    numeric = tmp_path / "numeric.json"
    numeric.write_text('{"frozen": true}', encoding="utf-8")
    source = tmp_path / "catalog.csv"
    source.write_text("id,code\na,G47121\nb,G47129\n", encoding="utf-8")
    tool = tmp_path / "join_tool.py"
    tool.write_text("# frozen grouping tool", encoding="utf-8")
    plan = {"schema": "p014_ksic2026_posthoc_receipt_proxy_plan_v1",
            "status": "frozen_before_recovered_catalog_outcome_join", "posthoc_exploratory": True,
            "source": {"sha256": report._sha(source), "bytes": source.stat().st_size,
                       "strict_utf8_rows": 2},
            "input_gate": {"numeric_sha256": report._sha(numeric),
                           "effect_days": ["2020-09-21"], "citizens": 40,
                           "citizen_days_each_arm": 40,
                           "minimum_join_count_fraction_each_arm": .99,
                           "minimum_join_won_fraction_each_arm": .99}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    evidence, arms = [], {}
    for arm in ("on", "off"):
        folder = tmp_path / arm
        folder.mkdir()
        ledger = folder / "sector.ledger.jsonl"
        ledger.write_text('{}\n', encoding="utf-8")
        metric = folder / "day_2020-09-21.jsonl"
        metric.write_text('{}\n', encoding="utf-8")
        for item in (ledger, metric):
            evidence.append({"path": report._display_path(item), "sha256": report._sha(item)})
        arms[arm] = {"run_id": "p014-" + arm, "citizen_days": 40,
                     "sector_ledger_path": report._display_path(ledger),
                     "sector_ledger_sha256": report._sha(ledger),
                     "metrics_evidence": [{"path": report._display_path(metric), "sha256": report._sha(metric)}],
                     "positive_receipts": 100, "positive_receipt_won": 10000,
                     "matched_receipts": 98, "matched_receipt_won": 9800,
                     "unmatched_receipts": 2, "unmatched_receipt_won": 200,
                     "join_count_fraction": .98, "join_won_fraction": .98,
                     "join_gate_pass": False, "ambiguous_join_receipts": 0,
                     "groups": {"47121": {"won": 1000 if arm == "on" else 2000,
                                          "positive_receipts": 17, "unique_citizens": 15,
                                          "unique_pois": 16},
                                "47129": {"won": 3000 if arm == "on" else 4000,
                                          "positive_receipts": 36, "unique_citizens": 23,
                                          "unique_pois": 33}}}
    audit = {"schema": "p014_ksic2026_posthoc_receipt_proxy_v1", "policy": "P014",
             "posthoc_exploratory": True, "numeric_path": report._display_path(numeric),
             "numeric_sha256": report._sha(numeric), "plan_path": report._display_path(plan_path),
             "plan_sha256": report._sha(plan_path), "tool_path": report._display_path(tool),
             "tool_sha256": report._sha(tool), "effect_days": ["2020-09-21"], "citizens": 40,
             "source": {**plan["source"], "path": report._display_path(source), "nul_bytes": 0,
                        "duplicate_merchant_ids": 0, "original_graph_ingest_byte_identity_confirmed": False},
             "technical_gate_pass": False,
             "join_gate": {"minimum_count_fraction_each_arm": .99, "minimum_won_fraction_each_arm": .99},
             **arms, "indicators": []}
    empirical = []
    for code, truth in (("47121", .141), ("47129", .082)):
        audit["indicators"].append({"id": "P014-KSIC2026-" + code, "ksic": code,
                                    "empirical_reference_id": "P014-KIPF-" + code,
                                    "simulation": None, "ci": None, "simulation_unit": "%", "n": 40,
                                    "on": arms["on"]["groups"][code], "off": arms["off"]["groups"][code],
                                    "direct_gap_allowed": False, "direction_comparable": False,
                                    "sparse_interpretation_blocked": code == "47121"})
        empirical.append({"id": "P014-KIPF-" + code, "policy": "LOCAL_VOUCHER",
                          "policy_name": "지역상품권 P014", "truth": truth, "truth_unit": "log-point",
                          "simulation": 2.224 if code == "47121" else None,
                          "simulation_unit": "%", "source": "KIPF", "source_locator": "VI-6",
                          "run_context": {"policy_id": "P014"}})
    original = {"score_files": [{"path": report._display_path(numeric), "sha256": report._sha(numeric)}],
                "run_evidence": [{"policy": "LOCAL_VOUCHER", "citizens": 40,
                                  "on": "2020-09-21:2020-09-21", "off": "2020-09-21:2020-09-21",
                                  "evidence": evidence,
                                  "run_context": {"on_run_id": "p014-on", "off_run_id": "p014-off"}}],
                "exploratory_pairs": empirical[:1], "exploratory_uncomputed": empirical[1:]}
    sidecar = tmp_path / "ksic.json"
    sidecar.write_text(json.dumps(audit), encoding="utf-8")
    return original, sidecar, audit


def test_p014_posthoc_failed_join_shows_raw_money_and_preserves_reference_count(tmp_path):
    original, path, audit = _ksic_failed_join_fixture(tmp_path)
    untouched = deepcopy(original)
    result = report.apply_p014_ksic_posthoc_proxy(original, path)
    assert original == untouched
    assert result["score_files"] == original["score_files"]
    assert result["exploratory_reference_count"] == 2
    assert len(result["exploratory_pairs"]) == 1
    observed = result["p014_ksic_posthoc"]
    markup = report._p014_ksic_posthoc_html(observed)
    assert "실측 +0.1410 log-point" in markup and "실측 +0.0820 log-point" in markup
    assert "시뮬 ON 1,000원 / OFF 2,000원" in markup
    assert "시뮬 ON 3,000원 / OFF 4,000원" in markup
    assert "영수증 17건·구매 시민 15명" in markup
    assert "고정 99% 관문 실패" in markup
    assert "시뮬 −50" not in markup and "시뮬 -50" not in markup
    grouped = report._exploratory_html(original["exploratory_pairs"], [], observed)
    assert 'id="p014-exploratory"' in grouped
    assert "원래 동결 by_sub 집계 2항목(재채점 아님)" in grouped
    assert grouped.index("원업종 코드 영수증 연결") < grouped.index("원래 동결 by_sub 집계")
    audit["indicators"][0]["simulation"] = -50
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="failed gate cannot produce"):
        report.apply_p014_ksic_posthoc_proxy(original, path)


def test_p014_posthoc_verifies_source_hash_and_join_fraction_arithmetic(tmp_path):
    original, path, audit = _ksic_failed_join_fixture(tmp_path)
    audit["on"]["join_count_fraction"] = .999
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="join fractions"):
        report.apply_p014_ksic_posthoc_proxy(original, path)
    audit["on"]["join_count_fraction"] = .98
    path.write_text(json.dumps(audit), encoding="utf-8")
    Path(audit["source"]["path"]).write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA mismatch"):
        report.apply_p014_ksic_posthoc_proxy(original, path)


def test_consumption_architecture_display_distinguishes_runtime_from_raw_model_and_env():
    markup = report._consumption_architecture_html({
        "p012_on_outside_0_12": 116, "p012_on_rows": 372,
        "p012_on_maximum_postclamp_difference": .30,
        "p012_off_outside_0_12": 0, "p012_off_rows": 372,
        "p012_off_maximum_postclamp_difference": .12,
        "other_latest_arms_outside_0_12": 0, "online_allocation_share_observed": .7465,
        "path": "audit.json", "sha256": "a" * 64,
        "document_path": "audit.md", "document_sha256": "b" * 64})
    assert "116/372 시민×일" in markup and "OFF는 0/372" in markup
    assert "원시 모델 소비의향이 아니라" in markup and "처리 후 관측" in markup
    assert "온라인 배분몫 74.65%" in markup
    assert "환경변수 실제 값은 보존되지 않았으므로" in markup
    assert "실행 설정의 확정 증거라고 부르지 않습니다" in markup
    assert "최신 5정책 10팔과 과거 P013 2팔" in markup


def test_consumption_architecture_audit_refuses_environment_or_prompt_only_overclaim(tmp_path):
    sidecar = tmp_path / "architecture.json"
    sidecar.write_text(json.dumps({
        "schema": "consumption_architecture_readonly_audit_v1",
        "settings_capture_limits": {"completed_process_environment_snapshot_found": False,
                                    "exact_EXP_values_recoverable_from_hash": False},
        "attribution": {"final_policy_spend_prompt_only_effect": True,
                        "direction_magnitude_external_accuracy_validated": False}}), encoding="utf-8")
    with pytest.raises(ValueError, match="overstates settings or attribution"):
        report.apply_consumption_architecture_audit({}, sidecar)


def test_historical_score_covers_every_registered_indicator_without_false_gap():
    result = report.build([ROOT / "output/answerkey_round2/score_r2_v5.json"])
    assert result["indicator_count"] == 38
    assert result["simulated_count"] == 5
    assert result["direct_gap_count"] == 0
    rows = {r["id"]: r for r in result["rows"]}
    assert rows["P012-1"]["truth_unit"] == "log-point"
    assert rows["P012-1"]["gap"] is None
    assert rows["P012-2"]["status"] == "정의 변경"
    assert rows["P010-1"]["status"] == "미실행"
    assert {p["id"] for p in result["unregistered_policies"]} == {"P008", "P011"}
    assert all(r["more_people"] and r["expert_opinion"] for r in result["rows"])


def _synthetic(tmp_path, *, score_on="2020-01-04:2020-01-05", audit_on="2020-01-04:2020-01-05"):
    audit = {
        "comparison": "matched_estimand", "source": "source/table 1",
        "reported_estimand": "paired spending change", "simulation_estimand": "paired spending change",
        "reported_window": "2020-01-01 to 2020-01-05",
        "simulation_window": "2020-01-01 to 2020-01-05",
        "reported_population": "same sample", "simulation_population": "same sample",
        "reported_denominator": "baseline spending", "simulation_denominator": "baseline spending",
        "reported_unit": "%", "simulation_unit": "%", "reported_value": 5.0,
        "simulation_off": "2020-01-01:2020-01-02", "simulation_on": audit_on,
    }
    scoring = tmp_path / "scoring.json"
    scoring.write_text(json.dumps({"P010": {"indicators": [
        {"id": "T-1", "metric": "total_spend_paired", "expect": "+",
         "desc": "test", "empirical_audit": audit}]}}), encoding="utf-8")
    score = tmp_path / "score.json"
    score.write_text(json.dumps({
        "policy": "P010", "label": "candidate", "off": "2020-01-01:2020-01-02",
        "on": score_on, "scoring_table_sha256": hashlib.sha256(scoring.read_bytes()).hexdigest(),
        "results": [{"id": "T-1", "metric": "total_spend_paired", "expect": "+",
                     "mean": 10, "base": 100, "ci": [5, 15], "n": 100, "hit": True}],
    }), encoding="utf-8")
    return scoring, score


def test_run_specific_matching_audit_allows_signed_gap_and_truth_marker(tmp_path):
    scoring, score = _synthetic(tmp_path)
    result = report.build([score], scoring)
    row = result["rows"][0]
    assert row["simulation"] == 10
    assert row["truth"] == 5
    assert row["gap"] == 5
    assert row["gap_unit"] == "%"
    assert row["external_direction_match"] is True
    markup = report.render(result)
    assert "시뮬−실측" in markup
    assert 'class="mk tru"' in markup


def test_changed_run_window_blocks_gap(tmp_path):
    scoring, score = _synthetic(tmp_path, score_on="2020-02-04:2020-02-05")
    row = report.build([score], scoring)["rows"][0]
    assert row["status"] == "실험 창 미감사"
    assert row["gap"] is None
    assert row["external_direction_match"] is None


def test_share_change_is_in_percentage_points_without_relative_denominator():
    ind = {"metric": "home_dong_spend_share", "expect": "+"}
    value, unit, ci = report._simulation(
        ind, {"mean": .05, "base": .50, "ci": [.02, .08]})
    assert (value, unit, ci) == (5.0, "%p", [2.0, 8.0])


def test_duplicate_policy_scores_refused(tmp_path):
    scoring, score = _synthetic(tmp_path)
    with pytest.raises(ValueError, match="duplicate policy"):
        report.build([score, score], scoring, experiment="bad_merge")


def test_generated_report_keeps_sources_and_machine_readable_coverage(tmp_path):
    scoring, score = _synthetic(tmp_path)
    out, json_out, result = report.generate([score], out=tmp_path / "comparison.html",
                                            scoring_path=scoring)
    saved = json.loads(json_out.read_text(encoding="utf-8"))
    assert saved["indicator_count"] == result["indicator_count"]
    assert saved["score_files"][0]["sha256"] == hashlib.sha256(score.read_bytes()).hexdigest()
    assert "T-1" in out.read_text(encoding="utf-8")


def test_paired_pilot_proxies_never_become_empirical_gaps(tmp_path):
    effect = tmp_path / "paired.json"
    effect.write_text(json.dumps({
        "policy_id": "P013", "complete_matrix": True, "funding_reconciled": True,
        "citizens": 80, "days": 5, "effect_start": "2020-05-11",
        "effect_end": "2020-05-13", "grant_recipients": 80,
        "grant_issued_won": 22400000, "grant_spent_won": 1000000,
        "recorded_total_spend_difference_won": 500000,
        "incremental_recorded_spend_per_grant_won": 0.02,
        "eligible_offline_relative_change": 0.05,
        "eligible_offline_relative_citizen_bootstrap_95_interval": [0.01, 0.08],
        "recorded_total_relative_change": 0.03,
        "recorded_total_relative_citizen_bootstrap_95_interval": [-0.01, 0.06],
        "provenance": {"prompt_variant": "v53", "arms": {"on": {}, "off": {}}},
    }), encoding="utf-8")
    note = tmp_path / "quality.txt"
    note.write_text("중간 Stage1 실패 1건을 기록하고 재개했습니다.", encoding="utf-8")
    out, json_out, built = report.generate([], paired_effect=effect,
                                           run_note=note, out=tmp_path / "pilot.html")
    assert built["indicator_count"] == 38
    assert built["simulated_count"] == 2
    assert built["direct_gap_count"] == 0
    rows = {r["id"]: r for r in built["rows"]}
    assert rows["EM-2"]["simulation"] == 5
    assert rows["EM-2"]["simulation_unit"] == "%"
    assert rows["EM-2"]["truth_unit"] == "%p"
    assert rows["EM-3"]["gap"] is None
    assert rows["P012-1"]["status"] == "미실행"
    markup = out.read_text(encoding="utf-8")
    assert markup.count('class="opinion"') == 38
    assert "중간 Stage1 실패 1건" in markup
    assert built["run_note"]["sha256"] == hashlib.sha256(note.read_bytes()).hexdigest()
    assert "표본만 확대" in markup
    assert json.loads(json_out.read_text(encoding="utf-8"))["direct_gap_count"] == 0
    sector = tmp_path / "sector.json"
    sector.write_text(json.dumps({
        "schema": "p013_sector_pair_v1", "citizens": 80,
        "days": ["2020-05-11", "2020-05-12", "2020-05-13"],
        "semidurable_relative_change_pct": 5.0,
        "face_service_relative_change_pct": 2.0,
        "rank_gap_percentage_points": 3.0,
        "citizen_bootstrap_95_interval": [-1.0, 7.0],
        "scoring_table_sha256": hashlib.sha256(
            (ROOT / "data/experiments/scoring_table.json").read_bytes()).hexdigest(),
        "comparison": "internal_same_calendar_poi_sector_proxy; not_external_kdi_estimand",
    }), encoding="utf-8")
    sector_report = report.build_paired_effect(effect, sector_path=sector)
    assert sector_report["simulated_count"] == 3
    assert sector_report["direct_gap_count"] == 0
    assert {r["id"]: r for r in sector_report["rows"]}["EM-4"]["simulation"] == 3
    assert "준내구재" in report.render(sector_report)
    assert "방향 불확실: 시뮬 95% 구간에 0 포함" in report.render(sector_report)


def test_numeric_only_view_keeps_policy_coverage_but_hides_unpaired_rows(tmp_path):
    scoring, score = _synthetic(tmp_path)
    original = report.build([score], scoring)
    measured = original["rows"][0]
    no_truth = dict(measured, id="T-2", truth=None, truth_unit=None, gap=None)
    no_simulation = dict(measured, id="T-3", simulation=None,
                         simulation_unit=None, gap=None, status="미실행")
    original["rows"].extend([no_truth, no_simulation])
    view = report.numeric_pair_view(original)
    assert [row["id"] for row in view["rows"]] == ["T-1"]
    assert view["omitted_without_empirical"] == 1
    assert view["omitted_without_simulation"] == 1
    assert view["policy_coverage"][0]["missing_simulation_ids"] == ["T-3"]
    markup = report.render(view)
    assert "실측 수치가 있는 정책의 숫자 확보 현황" in markup
    assert 'class="id">T-2<' not in markup
    assert 'class="id">T-3<' not in markup
    assert "실측 —" not in markup and "시뮬 —" not in markup
    assert "시뮬−실측" in markup  # only the audited T-1 has a direct gap


def test_numeric_only_cli_output_preserves_source_and_direct_gap(tmp_path):
    scoring, score = _synthetic(tmp_path)
    out, json_out, result = report.generate([score], scoring_path=scoring,
                                            numeric_only=True,
                                            out=tmp_path / "numeric.html")
    assert result["report_kind"] == "numeric_pairs"
    assert result["indicator_count"] == 1
    assert result["direct_gap_count"] == 1
    assert result["policy_coverage"][0]["empirical_numeric_count"] == 1
    assert "실측 +5.00%" in out.read_text(encoding="utf-8")
    assert json.loads(json_out.read_text(encoding="utf-8"))["score_files"][0]["sha256"] == hashlib.sha256(score.read_bytes()).hexdigest()


def test_multi_policy_proxy_needs_verified_evidence_and_never_self_certifies_gap(tmp_path):
    scoring, evidence = _synthetic(tmp_path)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema": "empirical_registry_v1", "indicators": [{
        "policy": "P010", "id": "T-1",
        "empirical": {"value": 5, "unit": "%", "source": "example/table 1",
                      "estimand": "reported 2020 effect"},
        "direct_gap_allowed": False, "reason": "proxy has another denominator",
    }], "exploratory_additional_benchmarks_not_in_registered_38": [{
        "policy": "P010", "id": "P010-BOK-X",
        "empirical": {"value": 46.0, "unit": "%", "source": "example/table 3"},
        "direct_gap_allowed": False, "reason": "survey and payments differ",
    }]}), encoding="utf-8")
    manifest = tmp_path / "multi.json"
    payload = {
        "schema": "multi_policy_numeric_v1", "experiment": "v53_test",
        "prompt_variant": "v53",
        "scoring_table_sha256": hashlib.sha256(scoring.read_bytes()).hexdigest(),
        "runs": [{"policy": "P010", "label": "v53_test", "citizens": 12,
                  "run_provenance": {"generic_prompt_sha256": "a" * 64,
                                     "on_environment_id": "env-on",
                                     "off_environment_id": "env-off"},
                  "off": "2020-01-01:2020-01-02",
                  "on": "2020-01-04:2020-01-05",
                  "evidence": [{"path": str(evidence),
                                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}],
                  "indicators": [{"id": "T-1", "simulation": 10, "simulation_unit": "%",
                                  "ci": [5, 15], "n": 80, "estimand_alignment": "matched",
                                  "direction_comparable": False,
                                  "reason": "same unit alone does not audit the estimand",
                                  "method": "paired change"},
                                 {"id": "P010-BOK-X", "simulation": 45,
                                  "simulation_unit": "%", "ci": None, "n": 80,
                                  "estimand_alignment": "different",
                                  "direction_comparable": False,
                                  "exploratory_not_registered": True,
                                  "policy_funded_positive_citizen_days": 2,
                                  "policy_funded_observed_citizen_days": 240,
                                  "policy_funded_total_won": 45672,
                                  "full_run_citizen_days": 400,
                                  "reason": "funded transaction shares are not survey responses",
                                  "method": "funded share"}]}],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    out, _, result = report.generate([], multi_policy_pairs=manifest,
                                     empirical_registry=registry, scoring_path=scoring,
                                     out=tmp_path / "numeric_report.html")
    assert result["indicator_count"] == 1
    assert result["rows"][0]["truth"] == 5
    assert result["rows"][0]["simulation"] == 10
    assert result["rows"][0]["gap"] is None
    assert result["exploratory_pairs"][0]["truth"] == 46.0
    assert "등록 38개 지표 밖의 탐색 참고값" in out.read_text(encoding="utf-8")
    assert "v53 기준선: 범용 프롬프트 최적화 완료 아님" in out.read_text(encoding="utf-8")
    assert "표본 12명 이하: 방향·크기 매우 불확실" in out.read_text(encoding="utf-8")
    assert "시뮬 시민 재표집 95% 구간(모델·외부 표본 불확실성 미포함)" in out.read_text(encoding="utf-8")
    assert "환경 ON env-on / OFF env-off" in out.read_text(encoding="utf-8")
    assert "원문 정의·산식·증거 자세히 보기" in out.read_text(encoding="utf-8")
    assert "탐색적 숫자상 차이" in out.read_text(encoding="utf-8")
    assert "정책결제액이 양수인 시민×일 2/240" in out.read_text(encoding="utf-8")
    assert "정책결제 총액 45,672원" in out.read_text(encoding="utf-8")
    assert "전체 원장은 400 시민×일(시행 전 포함)" in out.read_text(encoding="utf-8")
    model_id = "LGAI-EXAONE/EXAONE-4.5-33B-AWQ"
    model_snapshot = {"served_model_ids": [model_id],
                      "server_command": f"python -m sglang.launch_server --model-path {model_id}"}
    snapshots = []
    for arm in ("on", "off"):
        snapshot = tmp_path / arm / "served_model_evidence.json"
        snapshot.parent.mkdir()
        snapshot.write_text(json.dumps(model_snapshot), encoding="utf-8")
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        snapshots.append(digest)
        payload["runs"][0]["evidence"].append({"path": str(snapshot), "sha256": digest})
    payload["runs"][0]["run_provenance"].update({
        "requested_model_id": "Qwen/Qwen3-8B",
        "served_model_provenance": {
            "model_id": model_id,
            "on_evidence_sha256": snapshots[0],
            "off_evidence_sha256": snapshots[1],
        },
    })
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    verified_html, _, _ = report.generate([], multi_policy_pairs=manifest,
                                          empirical_registry=registry, scoring_path=scoring,
                                          out=tmp_path / "verified_model.html")
    model_markup = verified_html.read_text(encoding="utf-8")
    assert "실제 서빙 모델 " + model_id in model_markup
    assert "Qwen/Qwen3-8B" not in model_markup
    payload["runs"][0]["run_provenance"]["served_model_provenance"]["model_id"] = "other/model"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent on served-model evidence"):
        report.build_multi_policy_pairs(manifest, scoring)
    payload["runs"][0]["run_provenance"]["served_model_provenance"]["model_id"] = model_id
    payload["runs"][0]["evidence"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256-mismatched evidence"):
        report.build_multi_policy_pairs(manifest, scoring)


def test_coverage_omits_policies_and_rows_without_numeric_truth_from_html():
    coverage = [
        {"policy": "P010", "policy_name": "쿠폰", "empirical_numeric_count": 1,
         "simulation_numeric_count": 1, "paired_numeric_count": 1,
         "sample_citizens": 80, "exploratory_numeric_count": 0,
         "unmeasured_count": 1,
         "missing_simulation_reasons": []},
        {"policy": "P015", "policy_name": "위약", "empirical_numeric_count": 0,
         "simulation_numeric_count": 0, "paired_numeric_count": 0,
         "sample_citizens": None, "exploratory_numeric_count": 0,
         "unmeasured_count": 2,
         "missing_simulation_reasons": []},
    ]
    markup = report._coverage_html(coverage, unregistered_count=2)
    assert "쿠폰" in markup and "위약" not in markup
    assert "실측 숫자가 없는 3개" in markup
    assert "등록되지 않은 정책 파일 2개" in markup


def test_p014_exploratory_only_policy_is_discoverable_without_fake_primary_pair():
    coverage = [
        {"policy": "P010", "policy_name": "쿠폰", "empirical_numeric_count": 1,
         "simulation_numeric_count": 1, "paired_numeric_count": 1,
         "sample_citizens": 80, "exploratory_numeric_count": 6,
         "unmeasured_count": 0, "missing_simulation_reasons": []},
        {"policy": "P014", "policy_name": "지역사랑상품권 P014",
         "empirical_numeric_count": 0, "simulation_numeric_count": 0,
         "paired_numeric_count": 0, "sample_citizens": 40,
         "exploratory_numeric_count": 2, "unmeasured_count": 3,
         "missing_simulation_reasons": []},
    ]
    markup = report._coverage_html(coverage)
    assert "지역사랑상품권 P014 2쌍" in markup
    assert "주지표 밖 탐색 숫자가 있는 정책" in markup
    assert "<th scope=\"row\">지역사랑상품권 P014</th>" not in markup


def test_in_progress_policy_shows_progress_without_blank_number_or_unfinished_details():
    coverage = [{"policy": "P012", "policy_name": "상생소비지원금 P012",
                 "empirical_numeric_count": 5, "simulation_numeric_count": 0,
                 "paired_numeric_count": 0, "sample_citizens": None,
                 "unmeasured_count": 1, "in_progress": True,
                 "missing_simulation_reasons": [{"id": "P012-4", "truth": 47880,
                                                 "truth_unit": "원", "reason": "월말 정산 필요"}]}]
    markup = report._coverage_html(coverage)
    assert "진행 중" in markup
    assert "P012-4" not in markup and "실측 —" not in markup
    coverage[0]["in_progress"] = False
    markup = report._coverage_html(coverage)
    assert "실측 +47,880원" in markup
    assert "월말 정산 필요" in markup
    assert "실측 —" not in markup and "시뮬 —" not in markup


def test_missing_simulation_explanations_identify_absent_denominator():
    assert "2020년 관광특구" in report._missing_simulation_explanation({
        "id": "DS-6", "status": "시뮬 수치 없음", "reason": "missing geometry"})
    assert "마트' 상위 분류 밖" in report._missing_simulation_explanation({
        "id": "C2", "status": "시뮬 수치 없음",
        "reason": "At least one target POI subclass is outside the mart parent"})
    assert "캐시백 누적액" in report._missing_simulation_explanation({
        "id": "P012-4", "status": "시뮬 수치 없음",
        "reason": "Monthly cashback payout State is absent from this transaction ledger."})


def test_distancing_geo_failure_discloses_counts_without_promoting_proxy(tmp_path):
    sidecar = tmp_path / "distancing_geo_proxy_20260928.json"
    payload = {
        "sources_sha256": {"official-2023-10-23.zip": "a" * 64},
        "positive_receipts": {"on": 448, "off": 477},
        "ambiguous_receipts_excluded": {"on": 22, "off": 23},
        "ambiguous_receipt_rate_excluded": {"on": 22 / 448, "off": 23 / 477},
        "on": {"관광특구": {"positive_receipts": 1,
                           "citizens_with_receipts": 1, "spend_won": 1833}},
        "off": {"관광특구": {"positive_receipts": 2,
                            "citizens_with_receipts": 1, "spend_won": 3537}},
    }
    sidecar.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    source = report._display_path(sidecar)
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    geo = {"overlap_count_by_arm": payload["ambiguous_receipts_excluded"],
           "total_receipt_count_by_arm": payload["positive_receipts"],
           "overlap_rate_by_arm": payload["ambiguous_receipt_rate_excluded"],
           "maximum_overlap_rate_allowed": 0.01,
           "source_year": 2023, "source_boundary_sha256": "a" * 64,
           "off_denominator_won_by_type": {"tourism_special_zone": 3537}}
    original = {
        "run_evidence": [{"policy": "DISTANCING_2020",
                          "evidence": [{"path": source, "sha256": digest}]}],
        "exploratory_simulations": [{"policy": "DISTANCING_2020",
                                     "id": "DS6-2023-GEO-PROXY",
                                     "simulation": None, "geo_proxy_audit": geo}],
        "rows": [{"policy": "DISTANCING_2020", "id": "DS-6",
                  "simulation": None, "reason": "no comparable 2020 panel",
                  "status": "시뮬 수치 없음"}],
    }
    updated = report.apply_distancing_geo_failure_display_audit(original, sidecar)
    explanation = report._missing_simulation_explanation(updated["rows"][0])
    assert "실측은 2020년 카드패널" in explanation
    assert "ON 4.91% (22/448건), OFF 4.82% (23/477건)" in explanation
    assert "ON 1건/1명·1,833원, OFF 2건/1명·3,537원" in explanation
    assert "−48.58" not in explanation
    assert updated["rows"][0]["simulation"] is None
    assert updated["post_run_display_audits"][0]["sha256"] == digest
    payload["ambiguous_receipts_excluded"]["on"] = 21
    sidecar.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen score"):
        report.apply_distancing_geo_failure_display_audit(original, sidecar)


def test_p016_patch_history_is_bound_to_new_score_not_failed_arm(tmp_path):
    patch_sha = "bc3819adaee5c8c922309aa0dc6b62c22a89f8630023939edb874b2b8fec50c4"
    numeric = tmp_path / "numeric.json"
    numeric.write_text('{"runs": []}', encoding="utf-8")
    provenance = {}
    evidence = []
    for arm in ("on", "off"):
        folder = tmp_path / "p016" / arm / arm
        folder.mkdir(parents=True)
        run_id = f"multipolicy-v53-20260928-p016-no_eligible_discount_fix1-{arm}"
        frozen = folder / "frozen_inputs.sha256"
        frozen.write_text(f"{patch_sha}  scripts/sim/instant_discount.py\n", encoding="utf-8")
        sector = folder / "sector.ledger.jsonl"
        sector.write_text('{}\n', encoding="utf-8")
        manifest = folder / "sector.ledger.jsonl.manifest.json"
        manifest.write_text(json.dumps({"prompt_provenance": {"run_id": run_id}}), encoding="utf-8")
        for source in (sector, manifest):
            evidence.append({"path": report._display_path(source),
                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        provenance[arm] = {
            "run_id": run_id, "run_revision": "no_eligible_discount_fix1",
            "frozen_inputs_path": report._display_path(frozen),
            "frozen_inputs_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
            "frozen_hashes": {"scripts/sim/instant_discount.py": patch_sha},
            "sector_ledger_sha256": hashlib.sha256(sector.read_bytes()).hexdigest(),
            "sector_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
    failed = tmp_path / "failed_prepatch_snapshot.json"
    failed.write_text(json.dumps({"schema": "failed_p016_prepatch_snapshot_v1",
                                  "run_status": "incomplete_failed",
                                  "model_calls_occurred": True,
                                  "raw_keyerror_rows": 11,
                                  "date_of_failure": "2020-07-30"}), encoding="utf-8")
    first = tmp_path / "preflight_first_failure.manifest.json"
    first.write_text('{"model_calls": 0}', encoding="utf-8")
    passed = tmp_path / "preflight_success.sha256"
    passed.write_text('verified\n', encoding="utf-8")
    audit = {
        "schema": "p016_postfix_provenance_display_audit_v1", "policy": "P016",
        "run_manifest_source_commit_is_not_live_code_evidence": True,
        "numeric_path": report._display_path(numeric),
        "numeric_sha256": hashlib.sha256(numeric.read_bytes()).hexdigest(),
        "patched_instant_discount_sha256": patch_sha, **provenance,
        "invalidated_prepatch_arm": {
            "excluded_from_score": True, "snapshot_path": report._display_path(failed),
            "snapshot_sha256": hashlib.sha256(failed.read_bytes()).hexdigest(),
            "raw_keyerror_rows": 11, "failure_date": "2020-07-30"},
        "preflight": {"first_failure_model_calls": 0,
                      "passed_preflight_model_calls": 0,
                      "first_failure_manifest_path": report._display_path(first),
                      "first_failure_manifest_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
                      "passed_preflight_checksums_path": report._display_path(passed),
                      "passed_preflight_checksums_sha256": hashlib.sha256(passed.read_bytes()).hexdigest()},
    }
    sidecar = tmp_path / "p016_postfix.json"
    sidecar.write_text(json.dumps(audit), encoding="utf-8")
    context = {"policy_id": "P016", "generic_prompt_sha256": "a" * 64}
    original = {"score_files": [{"path": report._display_path(numeric),
                                  "sha256": audit["numeric_sha256"]}],
                "run_evidence": [{"policy": "P016", "evidence": evidence,
                                  "run_context": context}],
                "rows": [{"policy": "P016", "id": "C1", "run_context": context}]}
    updated = report.apply_p016_postfix_display_audit(original, sidecar)
    markup = report._run_context_html(updated["rows"][0]["run_context"])
    assert "미완결·채점 제외" in markup and "원시 기록 11행" in markup
    assert "서버 저장소의 source_commit은 실행 파일 증거가 아니므로" in markup
    audit["on"]["frozen_hashes"]["scripts/sim/instant_discount.py"] = "0" * 64
    sidecar.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="patched frozen inputs"):
        report.apply_p016_postfix_display_audit(original, sidecar)


def test_p014_exploratory_card_keeps_source_coefficient_separate_from_poi_proxy():
    markup = report._exploratory_html([{
        "policy": "LOCAL_VOUCHER", "policy_name": "지역사랑상품권 탐색 지표",
        "id": "P014-KIPF-47121", "truth": 0.141, "truth_unit": "log-point",
        "simulation": 12.5, "simulation_unit": "%", "ci": [1.0, 25.0], "n": 40,
        "source": "KIPF table VI-6", "source_locator": "column 3",
        "simulation_evidence": [],
        "run_context": {"policy_id": "P014", "generic_prompt_sha256": "a" * 64,
                        "policy_input_file": "data/neo4j_load/policies/P014.json"},
    }])
    assert "실측 +0.1410 log-point" in markup
    assert "시뮬 +12.50%" in markup
    assert "지역·연도별 업종 매출 로그회귀계수" in markup
    assert "매장 면적 165㎡ 이상/미만" in markup
    assert "곡물·반찬·건어물·사료" in markup
    assert "시뮬 시민 40명" in markup
    assert "시뮬 시민 재표집 95% 구간 +1.00% ~ +25.00%" in markup
    assert "상품권 구매·잔액·상환 지갑 원장이" in markup
    assert "policy_hits도 실제 자치구·상호 적격 결제 건수를 세지 않습니다" in markup
    assert "상품권 거래 효과나 사용률이 아니고" in markup
    assert "범용 프롬프트 v53" in markup
    assert "시뮬−실측" not in markup


def test_p014_mechanism_audit_shows_repairs_without_claiming_voucher_usage(tmp_path):
    numeric = tmp_path / "numeric.json"
    numeric.write_text('{"runs": []}', encoding="utf-8")
    policy = ROOT / "data/neo4j_load/policies/P014.json"
    quality_arm = {"citizen_days": 200, "stage1_final_ok_count": 200,
                   "stage1_first_attempt_internal_validation_pass_count": 160,
                   "stage2_fallback_only_count": 16,
                   "stage2_choice_repair_count": 8,
                   "stage2_quality_gate_pass": True}
    context = {"policy_id": "P014", "policy_input_sha256": report._sha(policy),
               "quality_audit": {arm: quality_arm for arm in ("on", "off")}}
    evidence = []
    arms = {}
    for arm in ("on", "off"):
        folder = tmp_path / arm
        folder.mkdir()
        ledger = folder / "sector.ledger.jsonl"
        ledger.write_text('{}\n', encoding="utf-8")
        stage2 = folder / "stage2.json"
        stage2.write_text(json.dumps({
            "quality_gate_pass": True, "unrepaired_choice_trace_pass": False,
            "totals": {"metrics_rows": 200, "stage2_choice_repair_agents": 8,
                       "stage2_hallucinations_corrected": 9,
                       "stage2_spend_amount_fallbacks": 16}}), encoding="utf-8")
        for source in (ledger, stage2):
            evidence.append({"path": report._display_path(source),
                             "sha256": report._sha(source)})
        arms[arm] = {"citizen_days": 120,
                     "sector_ledger_path": report._display_path(ledger),
                     "sector_ledger_sha256": report._sha(ledger),
                     "stage2_audit_path": report._display_path(stage2),
                     "stage2_audit_sha256": report._sha(stage2),
                     "stage2_quality_gate_pass": True,
                     "stage2_unrepaired_choice_trace_pass": False,
                     "stage2_choice_repair_agents": 8,
                     "stage2_hallucinations_corrected": 9,
                     "stage2_spend_amount_fallbacks": 16,
                     "positive_purchase_receipts": 634,
                     "positive_purchase_won": 5991515,
                     "policy_hits_total": 634}
    audit = {"schema": "p014_voucher_mechanism_postscore_audit_v1",
             "policy": "P014", "policy_type": "price_discount",
             "policy_file_path": report._display_path(policy),
             "policy_file_sha256": report._sha(policy),
             "numeric_path": report._display_path(numeric),
             "numeric_sha256": report._sha(numeric),
             "voucher_purchase_count": None, "voucher_redemption_count": None,
             "voucher_usage_rate_among_eligible_purchases": None, **arms}
    sidecar = tmp_path / "mechanism.json"
    sidecar.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    original = {"score_files": [{"path": report._display_path(numeric),
                                  "sha256": report._sha(numeric)}],
                "run_evidence": [{"policy": "LOCAL_VOUCHER", "run_context": context,
                                  "evidence": evidence}],
                "exploratory_simulations": [{"policy": "LOCAL_VOUCHER", "run_context": context}]}
    augmented = report.apply_p014_mechanism_display_audit(original, sidecar)
    markup = report._run_context_html(
        augmented["exploratory_simulations"][0]["run_context"])
    assert "전체 200 시민×일 중 장소 선택 보정 8 시민×일" in markup
    assert "존재하지 않는 장소 보정 9건" in markup
    assert "금액 대체 16건" in markup
    assert "상품권 사용률이 아닙니다" in markup
    audit["on"]["stage2_hallucinations_corrected"] = 10
    sidecar.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen Stage2 evidence"):
        report.apply_p014_mechanism_display_audit(original, sidecar)


def test_p014_food_zero_displays_raw_amounts_without_fabricating_percentage(tmp_path):
    numeric = tmp_path / "numeric.json"
    numeric.write_text('{"runs": []}', encoding="utf-8")
    arms, evidence = {}, []
    for arm in ("on", "off"):
        ledger = tmp_path / arm / "sector.ledger.jsonl"
        ledger.parent.mkdir()
        ledger.write_text(json.dumps({"aid": "a", "day": "2020-09-21",
                                      "by_sub": {"식료품": 0}}, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        arms[arm] = {"sector_ledger_path": report._display_path(ledger),
                     "sector_ledger_sha256": report._sha(ledger),
                     "won": 0, "citizen_days": 1}
        evidence.append({"path": report._display_path(ledger), "sha256": report._sha(ledger)})
    audit = {"schema": "p014_food_zero_display_audit_v1", "policy": "P014",
             "indicator": "P014-KIPF-47129", "subclass": "식료품",
             "numeric_path": report._display_path(numeric),
             "numeric_sha256": report._sha(numeric), "effect_days": ["2020-09-21"], **arms}
    path = tmp_path / "foodstore_zero_audit.json"
    path.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    row = {"policy": "LOCAL_VOUCHER", "id": "P014-KIPF-47129",
           "simulation": None, "simulation_unit": "%", "policy_name": "지역상품권 P014",
           "truth": .082, "truth_unit": "log-point", "source": "KIPF", "source_locator": "VI-6"}
    original = {"score_files": [{"path": report._display_path(numeric),
                                  "sha256": report._sha(numeric)}],
                "run_evidence": [{"policy": "LOCAL_VOUCHER", "citizens": 1,
                                  "on": "2020-09-21:2020-09-21",
                                  "off": "2020-09-21:2020-09-21", "evidence": evidence}],
                "exploratory_simulations": [row]}
    augmented = report.apply_p014_food_zero_display_audit(original, path)
    marked = augmented["exploratory_simulations"][0]
    markup = report._exploratory_html([], [marked])
    assert "실측 +0.0820 log-point" in markup
    assert "시뮬 원금액 ON 0원 / OFF 0원" in markup
    assert "비율 미산출" in markup and "분모 OFF가 0" in markup
    assert "시뮬 +0.00%" not in markup
    audit["off"]["won"] = 1
    path.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="source or population differs"):
        report.apply_p014_food_zero_display_audit(original, path)


def test_p014_zero_preperiod_support_is_distinguished_from_large_imbalance():
    markup = report._run_context_html({
        "policy_id": "P014", "preperiod_balance": {
            "status": "fail", "post_effect_causal_interpretation_blocked": True,
            "comparisons": {"food_store_poi_spend_proxy": {"on_won": 0, "off_won": 0}},
            "reason": "P010 offline spending includes merchants outside exact eligibility"}})
    assert "변화율 기준 분모가 없어 사전 관문에 실패" in markup
    assert "총지출·슈퍼마켓의 차이 폭이 문턱을 넘었다는 뜻이 아닙니다" in markup
    assert "P010 offline" not in markup


def test_p014_industry_scope_audit_retains_verified_definition_sources(tmp_path):
    numeric = tmp_path / "numeric.json"
    numeric.write_text('{}', encoding="utf-8")
    audit = {"schema": "p014_industry_scope_postscore_audit_v1", "policy": "P014",
             "numeric_path": report._display_path(numeric), "numeric_sha256": report._sha(numeric),
             "original_ksic_or_floor_area_crosswalk_in_subclass_mapping": False,
             "source_locator": "KIPF table VI-6 footnotes 62-63"}
    for name, path_key, hash_key in (("source.pdf", "source_pdf_path", "source_pdf_sha256"),
                                   ("source.txt", "local_extracted_text_path", "local_extracted_text_sha256"),
                                   ("mapping.json", "mapping_path", "mapping_sha256")):
        source = tmp_path / name
        source.write_text('definition snapshot', encoding="utf-8")
        audit[path_key], audit[hash_key] = report._display_path(source), report._sha(source)
    sidecar = tmp_path / "industry.json"
    sidecar.write_text(json.dumps(audit), encoding="utf-8")
    original = {"score_files": [{"path": report._display_path(numeric),
                                  "sha256": report._sha(numeric)}],
                "exploratory_simulations": [{"policy": "LOCAL_VOUCHER",
                                             "run_context": {"policy_id": "P014"}}]}
    result = report.apply_p014_industry_scope_display_audit(original, sidecar)
    context = result["exploratory_simulations"][0]["run_context"]
    assert "원문 면적·업종 대응 감사" in report._run_context_html(context)
    audit["mapping_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA mismatch"):
        report.apply_p014_industry_scope_display_audit(original, sidecar)


def test_p016_taxonomy_identity_suppresses_false_zero_gap(tmp_path):
    numeric = tmp_path / "numeric.json"
    numeric.write_text('{"runs": []}', encoding="utf-8")
    arms, evidence = {}, []
    for arm in ("on", "off"):
        ledger = tmp_path / arm / arm / "sector.ledger.jsonl"
        ledger.parent.mkdir(parents=True)
        ledger.write_text('{}\n', encoding="utf-8")
        digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
        evidence.append({"path": report._display_path(ledger), "sha256": digest})
        arms[arm] = {"sector_ledger_path": report._display_path(ledger),
                     "sector_ledger_sha256": digest, "citizen_days_all": 200,
                     "identity_count_all": 200, "citizen_days_effect": 120,
                     "identity_count_effect": 120,
                     "effect_target_poi_won": 968643 if arm == "on" else 626734,
                     "effect_mart_l1_won": 968643 if arm == "on" else 626734}
    audit = {"schema": "p016_taxonomy_identity_display_audit_v1",
             "policy": "P016", "indicators": ["C2", "C3"],
             "target_poi_subclasses": ["청과", "정육", "슈퍼마켓", "식료품"],
             "mart_l1_category": "마트", "structural_identity_all_rows": True,
             "numeric_path": report._display_path(numeric),
             "numeric_sha256": hashlib.sha256(numeric.read_bytes()).hexdigest(),
             "score_c2_percentage_points": 0.0, "score_c3_percentage_points": 0.0,
             **arms}
    sidecar = tmp_path / "taxonomy_identity_audit.json"
    sidecar.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    context = {"policy_id": "P016", "generic_prompt_sha256": "a" * 64,
               "preperiod_balance": {"status": "fail",
                                     "post_effect_causal_interpretation_blocked": True}}
    rows = [{"policy": "P016", "id": key, "simulation": 0.0,
             "ci": [0.0, 0.0], "truth": 7.0 if key == "C2" else 1.9,
             "truth_unit": "%p", "simulation_unit": "%p", "gap": None,
             "run_context": context} for key in ("C2", "C3")]
    original = {"score_files": [{"path": report._display_path(numeric),
                                  "sha256": audit["numeric_sha256"]}],
                "run_evidence": [{"policy": "P016", "evidence": evidence}],
                "rows": rows}
    updated = report.apply_p016_taxonomy_display_audit(original, sidecar)
    assert all(row["structural_zero_display_audit"]["meaning"] ==
               "taxonomy_identity_not_policy_nonresponse" for row in updated["rows"])
    assert all(report._surface_comparison(row) is None for row in updated["rows"])
    assert "분류 항등식" in report._run_context_html(updated["rows"][0]["run_context"])
    audit["on"]["identity_count_all"] = 199
    sidecar.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound to scored ledgers"):
        report.apply_p016_taxonomy_display_audit(original, sidecar)


def test_policy_card_separates_first_attempt_quality_from_recovered_ledger():
    arm = {"citizen_days": 80, "stage1_final_ok_count": 80,
           "stage1_first_attempt_internal_validation_pass_count": 60,
           "stage1_retry_recovered_count": 20, "stage2_fallback_only_count": 2,
           "stage2_choice_repair_count": 4, "stage2_quality_gate_pass": True}
    markup = report._run_context_html({
        "policy_id": "P010", "generic_prompt_sha256": "a" * 64,
        "quality_audit": {"on": arm, "off": arm},
        "preperiod_balance": {"status": "fail",
                              "post_effect_causal_interpretation_blocked": True,
                              "comparisons": {"recorded_total_spend": {
                                  "difference_pct_of_off": 15.0}}},
    })
    assert "첫 시도 내부검증 60/80 (75.0%)" in markup
    assert "최종 원장 성공 80/80" in markup
    assert "원시 첫응답의 엄격 형식률과 다릅니다" in markup
    assert "정책 후 차이를 인과효과로 해석할 수 없습니다" in markup
    assert "기록된 총지출" in markup


def test_policy_card_does_not_overstate_first_pass_with_missing_attempt_history():
    markup = report._run_context_html({
        "policy_id": "P010", "generic_prompt_sha256": "a" * 64,
        "quality_audit": {
            "on": {"audit_status": "verified", "citizen_days": 80,
                   "stage1_final_ok_count": 80,
                   "stage1_first_attempt_internal_validation_pass_count": 60,
                   "stage1_first_attempt_unknown_count": 4,
                   "stage1_outer_retry_recovered_count": 4,
                   "stage1_successful_invocation_first_attempt_pass_rate": 0.9,
                   "stage2_fallback_only_count": 2,
                   "stage2_choice_repair_count": 4,
                   "stage2_quality_gate_pass": True},
            "off": {"audit_status": "unavailable", "reason": "missing raw metrics"},
        },
    })
    assert "이력 불명 4, 가능 범위 75.0~80.0%" in markup
    assert "성공한 마지막 호출 안의 첫 시도 통과율 90.0%" in markup
    assert "OFF 첫 시도·Stage2 품질 감사 불가" in markup
    assert "80/80" in markup


def test_p012_card_calls_out_arm_asymmetry_after_full_recovery():
    arm = {"citizen_days": 372, "stage1_final_ok_count": 372,
           "stage1_first_attempt_internal_validation_pass_count": 244,
           "stage1_outer_retry_recovered_count": 4,
           "stage2_fallback_only_count": 0, "stage2_choice_repair_count": 21,
           "stage2_quality_gate_pass": True}
    off = {**arm, "stage1_first_attempt_internal_validation_pass_count": 344,
           "stage1_outer_retry_recovered_count": 0,
           "stage2_choice_repair_count": 25}
    markup = report._run_context_html({"policy_id": "P012",
                                       "quality_audit": {"on": arm, "off": off}})
    assert "ON 244/372 (65.6%), OFF 344/372 (92.5%)" in markup
    assert "최종 원장이 완결돼도 이 차이는 사라지지 않습니다" in markup
    assert "재시도·보정 경로가 양팔 행동 기록에 다르게 작용" in markup


def test_p010_wallet_diagnosis_is_descriptive_and_distinguishes_events_from_citizen_days():
    diagnostic = {
        "status": "post_run_descriptive_quality_audit", "path": "verified/p010_wallet_diagnosis.json",
        "sha256": "b" * 64, "days": ["2025-07-21", "2025-07-22", "2025-07-23"],
        "citizens": 80, "positive_purchase_events": 1288,
        "eligible_purchase_events": 1280, "positive_purchase_won": 9427617,
        "eligible_purchase_won": 9368052, "funded_purchase_events": 2,
        "funded_won": 45672, "stage1_grant_style_present_citizen_days": 0,
        "stage1_grant_use_present_citizen_days": 0,
        "policy_request_positive_citizen_days": 2,
        "policy_requested_won": 165000,
    }
    markup = report._run_context_html({"policy_id": "P010",
                                       "policy_funding_diagnostic": diagnostic})
    assert "적격 구매 이벤트 1280/1288" in markup
    assert "Stage2 양수 정책결제 요청 2/240 시민×일·165,000원" in markup
    assert "실제 지갑결제 2 구매 이벤트·45,672원" in markup
    assert "필수가 아니므로 모델의 지시 위반으로 해석하지 않습니다" in markup
    assert "프롬프트·파서·스키마·결제선택 로직" in markup
    assert "SHA256 " + "b" * 64 in markup


def test_p010_subclass_display_erratum_is_evidence_bound_and_does_not_rescore(tmp_path):
    numeric = tmp_path / "numeric.json"
    numeric.write_text('{"frozen": true}', encoding="utf-8")
    ledger = tmp_path / "sector.ledger.jsonl"
    ledger.write_text('{"citizen_id": "sample"}\n', encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    audit = {
        "schema": "p010_funded_subclass_display_audit_v1",
        "status": "post_run_display_erratum_not_rescoring", "policy": "P010",
        "numeric_score_sha256": digest(numeric),
        "on_sector_ledger_sha256": digest(ledger),
        "effect_days": ["2025-07-21", "2025-07-23"],
        "observed_citizen_days": 240,
        "positive_policy_funded_citizen_days": 2,
        "policy_funded_total_won": 45672,
        "funded_by_sub_won": {"의류": 3694, "가전·통신": 41978},
        "funded_by_sub_total_won": 45672,
    }
    sidecar = tmp_path / "funded_by_sub_audit.json"
    sidecar.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    original = {
        "score_files": [{"path": str(numeric), "sha256": digest(numeric)}],
        "run_evidence": [{"policy": "P010", "on": "2025-07-21:2025-07-23",
                          "evidence": [{"path": str(ledger), "sha256": digest(ledger)}],
                          "run_context": {"policy_funding_density": {
                              "policy_funded_observed_citizen_days": 240,
                              "policy_funded_positive_citizen_days": 2,
                              "policy_funded_total_won": 45672}}}],
        "exploratory_simulations": [{
            "policy": "P010", "policy_name": "민생회복 소비쿠폰",
            "id": "P010-BOK-X", "truth": 46.0, "truth_unit": "%",
            "simulation": 0.0, "simulation_unit": "%", "n": 80,
            "policy_funded_positive_citizen_days": 2,
            "policy_funded_observed_citizen_days": 240,
            "policy_funded_total_won": 45672, "full_run_citizen_days": 400,
            "source": "BOK", "simulation_evidence": [],
        }],
    }
    augmented = report.apply_p010_funding_display_audit(original, sidecar)
    assert augmented["score_files"] == original["score_files"]
    assert "funded_subclass_display_audit" not in original["exploratory_simulations"][0]
    assert augmented["post_run_display_audits"][0]["sha256"] == digest(sidecar)
    markup = report._exploratory_html(augmented["exploratory_simulations"])
    assert "시뮬 0%의 의미" in markup
    assert "가전·통신 41,978원" in markup and "의류 3,694원" in markup
    assert "실측 업종 비중이 0%라는 뜻도" in markup
    assert "2/240 시민×일" in markup
    audit["numeric_score_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match a frozen numeric score"):
        report.apply_p010_funding_display_audit(original, sidecar)


def test_p010_channel_display_audit_reconciles_verified_metrics_without_rescoring(tmp_path):
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = []
    for arm in ("on", "off"):
        metric = tmp_path / "p010" / arm / arm / "metrics" / "day_2025-07-21.jsonl"
        metric.parent.mkdir(parents=True)
        metric.write_text('{"verified": true}\n', encoding="utf-8")
        evidence.append({"path": metric.as_posix(), "sha256": digest(metric)})
    context = {"policy_funding_density": {"policy_funded_total_won": 5}}
    original = {
        "score_files": [{"sha256": "a" * 64}],
        "run_evidence": [{"policy": "P010", "on": "2025-07-21:2025-07-21",
                          "citizens": 6, "evidence": evidence, "run_context": context}],
        "rows": [{"policy": "P010", "simulation": 0.375, "run_context": context}],
    }
    audit = {
        "status": "posthoc_simulator_channel_diagnostic_not_empirical_policy_effect",
        "policy_effect_days": ["2025-07-21"], "paired_citizen_days": 6,
        "source_sha256": {entry["path"]: entry["sha256"] for entry in evidence},
        "on_sum": {"cm_today_total_incl_online": 120, "cm_online_total": 80,
                   "offline_positive_receipt_won": 40, "cm_policy_allocated_total": 5},
        "off_sum": {"cm_today_total_incl_online": 100, "cm_online_total": 65,
                    "offline_positive_receipt_won": 35, "cm_policy_allocated_total": 0},
        "on_minus_off": {"cm_today_total_incl_online": 20, "cm_online_total": 15,
                         "offline_positive_receipt_won": 5, "cm_policy_allocated_total": 5},
        "online_fraction_of_total_gap": .75, "offline_fraction_of_total_gap": .25,
    }
    sidecar = tmp_path / "p010_channel_gap_audit.json"
    sidecar.write_text(json.dumps(audit), encoding="utf-8")
    augmented = report.apply_p010_channel_display_audit(original, sidecar)
    assert augmented["rows"][0]["simulation"] == original["rows"][0]["simulation"]
    assert "p010_channel_display_audit" not in original["rows"][0]["run_context"]
    assert augmented["post_run_display_audits"][0]["sha256"] == digest(sidecar)
    markup = report._run_context_html(augmented["rows"][0]["run_context"])
    assert "ON−OFF 총지출 20원 중 온라인 모델 채널 15원(75.0%)" in markup
    assert "실측 설문 MPC와의 정확도" in markup
    concentration = {
        "status": "posthoc_citizen_gap_concentration_diagnostic_not_policy_effect",
        "source_sha256": {entry["path"]: entry["sha256"] for entry in evidence},
        "paired_citizens": 6, "positive_delta_citizens": 5,
        "negative_delta_citizens": 1, "zero_delta_citizens": 0,
        "net_gap_won": 20, "top_five_positive_deltas_won": [10, 5, 4, 3, 2],
        "top_five_sum_won": 24, "top_five_share_of_net_gap": 1.2,
    }
    concentration_sidecar = tmp_path / "p010_citizen_gap_concentration_audit.json"
    concentration_sidecar.write_text(json.dumps(concentration), encoding="utf-8")
    concentrated = report.apply_p010_concentration_display_audit(augmented,
                                                                  concentration_sidecar)
    markup = report._run_context_html(concentrated["rows"][0]["run_context"])
    assert "상위 5명의 ON−OFF 지출차이 합은 24원" in markup
    assert "나머지 시민의 순합은 4원 감소" in markup
    assert concentrated["rows"][0]["simulation"] == original["rows"][0]["simulation"]
    audit["source_sha256"][evidence[0]["path"]] = "0" * 64
    sidecar.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics do not match verified paired evidence"):
        report.apply_p010_channel_display_audit(original, sidecar)
    concentration["top_five_sum_won"] = 25
    concentration_sidecar.write_text(json.dumps(concentration), encoding="utf-8")
    with pytest.raises(ValueError, match="counts or gap do not reconcile"):
        report.apply_p010_concentration_display_audit(augmented, concentration_sidecar)


def test_ds6_2023_geo_proxy_remains_exploratory_with_mapping_and_sparse_gate():
    audit = {"match_rate": 0.995, "overlap_count": 3,
             "overlap_count_by_arm": {"on": 2, "off": 1},
             "total_receipt_count_by_arm": {"on": 403, "off": 500},
             "total_receipt_count": 903,
             "overlap_rate_by_arm": {"on": 2 / 403, "off": 1 / 500},
             "maximum_overlap_rate_allowed": 0.01,
             "overlap_rule": "Exclude ambiguous receipts from both hub types; no category priority",
             "on_citizen_days": 240, "off_citizen_days": 240,
             "off_denominator_won_by_type": {"tourism_special_zone": 5000,
                                             "developed_commercial_district": 9000},
             "source_year": 2023, "source_boundary_sha256": "a" * 64,
             "sparse_interpretation_blocked": True}
    entry = {"id": "DS6-2023-GEO-PROXY", "simulation": -2.0,
             "simulation_unit": "percentage points", "estimand_alignment": "different",
             "direction_comparable": False,
             "simulation_components": {"tourism_special_zone_pct": -5.0,
                                       "developed_commercial_district_pct": -3.0},
             "geo_proxy_audit": audit}
    report._validate_geo_proxy_exploratory(entry)
    markup = report._exploratory_html([{
        "policy": "DISTANCING_2020", "policy_name": "사회적 거리두기",
        "id": entry["id"], "truth": -4.3, "truth_unit": "%p",
        "simulation": -2.0, "simulation_unit": "%p", "n": 40,
        "empirical_components": [{"hub": "tourism_special_zone", "value": -8.7},
                                 {"hub": "developed_commercial_district", "value": -4.4}],
        "simulation_components": entry["simulation_components"],
        "geo_proxy_audit": audit, "source": "2020 Seoul card panel",
        "source_locator": "publisher summary", "simulation_evidence": [],
    }])
    assert "2023-10-23 상권 경계" in markup
    assert "2026년 3월 POI" in markup
    assert "실측 2020 전년 대비" in markup and "시뮬 3일 ON−OFF" in markup
    assert "ON 2/403 (0.50%), OFF 1/500 (0.20%)" in markup
    assert "두 유형 모두에서 제외" in markup and "허용 상한 1%" in markup
    assert "희소한 관측 때문에 이 숫자는 기술값" in markup
    audit["match_rate"] = 0.98
    with pytest.raises(ValueError, match="coordinate, overlap, balance or OFF denominator"):
        report._validate_geo_proxy_exploratory(entry)
    audit["match_rate"] = 0.995
    audit["overlap_rate_by_arm"]["on"] = 0.02
    with pytest.raises(ValueError, match="coordinate, overlap, balance or OFF denominator"):
        report._validate_geo_proxy_exploratory(entry)


def test_distancing_input_display_distinguishes_internal_environment_id_from_year(tmp_path):
    source = tmp_path / "frozen_input.json"
    source.write_text('{"frozen": true}', encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    shared = "서울 신규 확진 112명 (2020-11-23 기준)"
    audit = {
        "purpose": "read-only frozen input audit; no observed policy outcomes or target numbers",
        "on_environment_id": "covid_2021", "off_environment_id": "covid_no_distancing",
        "daily": [{"date": "2020-11-24", "shared_disease_facts": [shared],
                   "on": {"facts": [shared, "식당 21:00 이후 매장취식 제한",
                                    "카페 포장·배달만", "유흥업소 집합금지"]},
                   "off": {"facts": [shared, "추가 방역 영업시간 제한 없음"]}}],
        "source_sha256": {source.relative_to(report.ROOT).as_posix(): digest}
        if source.is_relative_to(report.ROOT) else {str(source): digest},
    }
    sidecar = tmp_path / "distancing_render_audit.json"
    sidecar.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    context = {"policy_id": "DISTANCING_2020", "generic_prompt_sha256": "a" * 64,
               "on_environment_id": "covid_2021",
               "off_environment_id": "covid_no_distancing"}
    original = {"run_evidence": [{"policy": "DISTANCING_2020",
                                  "on": "2020-11-24:2020-11-24",
                                  "off": "2020-11-24:2020-11-24",
                                  "run_context": context}],
                "rows": [{"policy": "DISTANCING_2020", "id": "DS-1",
                          "run_context": context}]}
    updated = report.apply_distancing_input_display_audit(original, sidecar)
    markup = report._run_context_html(updated["rows"][0]["run_context"])
    assert "같은 서울 신규 확진 배경(112명)" in markup
    assert "covid_2021" in markup and "내부 이름일 뿐" in markup
    assert "정적 렌더 감사" in markup and "HTTP 요청 전수" in markup
    assert updated["post_run_display_audits"][0]["sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    audit["daily"][0]["date"] = "2021-11-24"
    sidecar.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="dates differ"):
        report.apply_distancing_input_display_audit(original, sidecar)


def test_log_point_is_converted_only_for_exploratory_number_impression():
    surface = report._surface_comparison({
        "truth": 0.2082, "truth_unit": "log-point",
        "simulation": 20.0, "simulation_unit": "%", "gap": None,
        "ci": [5.0, 35.0],
    })
    assert surface is not None
    assert 23 < surface["empirical_as"] < 24
    assert surface["delta_unit"] == "%p"
    assert surface["delta"] < 0
    assert report._surface_comparison({
        "truth": 11.1, "truth_unit": "%p", "simulation": 13.5,
        "simulation_unit": "%", "gap": None, "ci": None,
    }) is None


def test_october_cashback_uses_october_reference_instead_of_two_month_mean(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema": "empirical_registry_v1", "indicators": [
        {"policy": "P012", "id": "P012-4", "empirical": {
            "value": 47880, "unit": "KRW per recipient",
            "october_only_from_rounded_table": 47827.7},
         "direct_gap_allowed": False, "reason": "different panel"},
        {"policy": "P012", "id": "P012-6", "empirical": {
            "value": 21.0, "unit": "% of cashback recipients",
            "october_only_from_rounded_table_percent": 20.8714},
         "direct_gap_allowed": False, "reason": "different panel"},
    ]}), encoding="utf-8")
    source = {"rows": [
        {"policy": "P012", "id": "P012-4", "empirical_variant": "october_only",
         "gap": None},
        {"policy": "P012", "id": "P012-6", "empirical_variant": "october_only",
         "gap": None},
    ]}
    updated = report.apply_empirical_registry(source, registry)
    assert updated["rows"][0]["truth"] == 47827.7
    assert updated["rows"][0]["truth_unit"] == "원"
    assert updated["rows"][1]["truth"] == 20.8714
    assert updated["rows"][1]["truth_unit"] == "%"
    assert all("10월" in row["truth_kind"] for row in updated["rows"])


def test_zero_of_eleven_cap_recipients_gets_exact_reference_not_accuracy_gap(tmp_path):
    scoring = tmp_path / "scoring.json"
    scoring.write_text(json.dumps({"P012": {"indicators": [{
        "id": "P012-6", "metric": "cap_share_recipients", "expect": "0",
        "desc": "10월 캐시백 상한 도달률"}]}}), encoding="utf-8")
    month = tmp_path / "paired_cashback_month.json"
    month.write_text(json.dumps({
        "policy_id": "P012", "month": "2021-10", "complete_paired_matrix": True,
        "recipients": 11, "capped_recipients": 0,
        "metrics": {"cap_share_recipients": {"value": 0.0}},
    }), encoding="utf-8")
    manifest = tmp_path / "numeric.json"
    manifest.write_text(json.dumps({
        "schema": "multi_policy_numeric_v1", "experiment": "cap_test",
        "prompt_variant": "v53",
        "scoring_table_sha256": hashlib.sha256(scoring.read_bytes()).hexdigest(),
        "runs": [{"policy": "P012", "citizens": 12,
                  "on": "2021-10-01:2021-10-31", "off": "2021-10-01:2021-10-31",
                  "run_provenance": {"generic_prompt_sha256": "a" * 64},
                  "evidence": [{"path": str(month),
                                "sha256": hashlib.sha256(month.read_bytes()).hexdigest()}],
                  "indicators": [{"id": "P012-6", "simulation": 0.0,
                                  "simulation_unit": "%", "ci": [0.0, 0.0], "n": 11,
                                  "estimand_alignment": "different",
                                  "direction_comparable": False,
                                  "empirical_variant": "october_only",
                                  "reason": "monthly accrual differs from paid survey",
                                  "method": "capped recipients / recipients"}]}],
    }), encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema": "empirical_registry_v1", "indicators": [{
        "policy": "P012", "id": "P012-6", "empirical": {
            "value": 21.0, "unit": "% of cashback recipients",
            "october_only_from_rounded_table_percent": 20.8714,
            "source": "test table"},
        "direct_gap_allowed": False, "reason": "accrual differs from paid benefits",
    }]}), encoding="utf-8")
    _, _, built = report.generate([], multi_policy_pairs=manifest,
                                  scoring_path=scoring, empirical_registry=registry,
                                  out=tmp_path / "cap.html")
    row = built["rows"][0]
    assert row["truth"] == 20.8714 and row["simulation"] == 0.0
    assert row["gap"] is None and row["surface_comparison"] is None
    assert row["binomial_zero_count_audit"]["two_sided_95_upper_pct"] == pytest.approx(28.49, abs=.01)
    markup = (tmp_path / "cap.html").read_text(encoding="utf-8")
    assert "0/11" in markup and "28.49%" in markup
    assert "시민 재표집 [0, 0]은 퇴화" in markup
    assert "탐색적 숫자상 차이" not in markup


def test_p012_sector_denominator_display_audit_binds_frozen_score_and_ledgers(tmp_path):
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    numeric = tmp_path / "numeric.json"
    numeric.write_text('{"frozen": true}', encoding="utf-8")
    ledgers = []
    for arm in ("on", "off"):
        source = tmp_path / arm / arm / "sector.ledger.jsonl"
        source.parent.mkdir(parents=True)
        source.write_text('{"verified": true}\n', encoding="utf-8")
        ledgers.append({"path": source.as_posix(), "sha256": digest(source)})
    row = {"policy": "P012", "id": "P012-5", "simulation": 125.0,
           "simulation_components": {"appliance_furniture_pct": 100.0,
                                     "hair_beauty_pct": -25.0},
           "truth": .3336, "truth_unit": "log-point",
           "simulation_unit": "%p", "gap": None, "ci": [-10, 500],
           "empirical_components": [{"sector": "appliances_furniture", "value": .3623},
                                    {"sector": "hair_beauty", "value": .0287}]}
    original = {
        "score_files": [{"path": numeric.as_posix(), "sha256": digest(numeric)}],
        "run_evidence": [{"policy": "P012", "on": "2021-10-01:2021-10-01",
                          "off": "2021-10-01:2021-10-01", "citizens": 2,
                          "evidence": ledgers}],
        "rows": [row],
    }
    audit = {
        "schema": "p012_sector_denominator_display_audit_v1",
        "policy": "P012", "indicator": "P012-5",
        "numeric_path": numeric.as_posix(), "numeric_sha256": digest(numeric),
        "on_sector_ledger_path": ledgers[0]["path"],
        "on_sector_ledger_sha256": ledgers[0]["sha256"],
        "off_sector_ledger_path": ledgers[1]["path"],
        "off_sector_ledger_sha256": ledgers[1]["sha256"],
        "effect_window": "2021-10-01:2021-10-01", "citizens": 2,
        "paired_citizen_days": 2,
        "appliance_furniture": {"on_won": 100, "off_won": 50,
                                "on_off_percent": 100.0},
        "hair_beauty": {"on_won": 75, "off_won": 100,
                        "on_off_percent": -25.0},
        "gap_percentage_points": 125.0,
    }
    sidecar = tmp_path / "sector_denominator_display_audit.json"
    sidecar.write_text(json.dumps(audit), encoding="utf-8")
    updated = report.apply_p012_sector_display_audit(original, sidecar)
    assert updated["rows"][0]["simulation"] == row["simulation"]
    assert updated["post_run_display_audits"][0]["sha256"] == digest(sidecar)
    markup = report._p012_rank_components(updated["rows"][0])
    assert "ON 100원 / OFF 50원" in markup
    assert "시뮬 +125.00%p의 크기가 매우 불안정" in markup
    audit["off_sector_ledger_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="verified ON/OFF ledgers"):
        report.apply_p012_sector_display_audit(original, sidecar)


def test_p012_sector_log_coefficient_gap_is_never_treated_as_percent_point_gap():
    row = {
        "id": "P012-5", "truth": 0.3336, "truth_unit": "log-point",
        "simulation": 18.0, "simulation_unit": "%p", "gap": None, "ci": [-3.0, 40.0],
        "empirical_components": [
            {"sector": "appliances_furniture", "value": 0.3623},
            {"sector": "hair_beauty", "value": 0.0287},
        ],
        "simulation_components": {"appliance_furniture_pct": 25.0,
                                  "hair_beauty_pct": 7.0},
    }
    assert report._surface_comparison(row) is None
    markup = report._p012_rank_components(row)
    assert "0.3623" in markup and "0.0287" in markup
    assert "+25.00%" in markup and "+7.00%" in markup
    assert "두 차이를 빼거나 적중률로 채점하지 않습니다" in markup
