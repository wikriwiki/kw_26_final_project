"""Distancing effects use realized POI sectors and matched citizen-days."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import export_distancing_daily_ledger as exporter  # noqa: E402
import paired_distancing_effect as paired  # noqa: E402


MAPPING = {
    "I20101": {"cat": "식사", "sub": "한식", "_industry_l1": "음식"},
    "G20404": {"cat": "마트", "sub": "슈퍼마켓", "_industry_l1": "소매"},
}


def spend(aid, amount, *, code=None, cat=None, sub=None):
    return {"aid": aid, "amount": amount, "upjong_l3": code,
            "category_names": [sub] if sub else [],
            "category_parents": [cat] if cat else []}


def test_actual_poi_category_and_unclassified_money_are_separate():
    rows = exporter.aggregate_day(
        [{"aid": "a", "online_spent": 20, "self_month_cumulative": 180},
         {"aid": "b", "online_spent": 0, "self_month_cumulative": 0}],
        [spend("a", 100, code="I20101", cat="식사", sub="한식"),
         spend("a", 50, cat="카페", sub="카페"),
         spend("a", 10)],
        roster=["a", "b"], day="2020-11-24", arm="restricted",
        mapping=MAPPING, previous={})
    assert rows[0]["offline_spent"] == 160
    assert rows[0]["korean_restaurant_won"] == 100
    assert rows[0]["cafe_won"] == 50
    assert rows[0]["classified_by_code_won"] == 100
    assert rows[0]["classified_by_category_won"] == 50
    assert rows[0]["unclassified_won"] == 10
    assert rows[1]["offline_spent"] == 0


def test_poi_conflict_or_missing_amount_blocks_export():
    with pytest.raises(ValueError, match="code/Category conflict"):
        exporter.classify_poi(spend("a", 10, code="I20101", cat="카페", sub="카페"),
                              MAPPING)
    with pytest.raises(ValueError, match="invalid transaction"):
        exporter.aggregate_day(
            [{"aid": "a", "online_spent": 0, "self_month_cumulative": 0}],
            [spend("a", None, cat="카페", sub="카페")], roster=["a"],
            day="2020-11-24", arm="restricted", mapping=MAPPING, previous={})


def ledger_row(aid, day, arm, *, restaurant=0, retail=0, unknown=0):
    offline = restaurant + retail + unknown
    return {"aid": aid, "day": day, "arm": arm,
            "offline_spent": offline, "online_spent": 0,
            "self_month_cumulative": offline,
            "restaurant_won": restaurant, "korean_restaurant_won": restaurant,
            "retail_won": retail, "cafe_won": 0,
            "unclassified_won": unknown,
            "classified_by_code_won": restaurant + retail,
            "classified_by_category_won": 0}


def test_sector_effect_and_unknown_allocation_bounds():
    day = "2020-11-24"
    on = [ledger_row("a", day, "restricted", restaurant=80, retail=30, unknown=10)]
    off = [ledger_row("a", day, "control", restaurant=100, retail=20)]
    result = paired.score(on, off, roster=["a"], days=[day], draws=20)
    dining = result["sectors"]["restaurant_won"]
    assert dining["relative_change"] == pytest.approx(-.2)
    assert dining["unknown_allocation_difference_bounds_per_citizen_won"] == [-20, -10]
    assert dining["direction_robust_to_unclassified"] == "negative"
    assert result["sectors"]["retail_won"]["direction_robust_to_unclassified"] == "positive"
    assert result["classification_coverage"]["restricted_unclassified_share"] == pytest.approx(
        10 / 120)
    assert result["sectors"]["cafe_won"]["relative_change"] is None
    on[0]["self_month_cumulative"] += 1
    with pytest.raises(ValueError, match="monthly State ledger mismatch"):
        paired.score(on, off, roster=["a"], days=[day], draws=0)


def test_distancing_manifest_requires_same_apparatus_and_distinct_runs(tmp_path):
    roster = ["a"]
    paths = {}
    for arm, env in exporter.ENVIRONMENTS.items():
        path = tmp_path / f"{arm}.jsonl"
        path.write_text('{}\n', encoding="utf-8")
        (tmp_path / f"{arm}.jsonl.manifest.json").write_text(json.dumps({
            "arm": arm, "environment_id": env,
            "start": "2020-11-24", "end": "2020-11-24", "days": 1,
            "citizens": 1, "rows": 1,
            "roster_sha256": hashlib.sha256(json.dumps(roster).encode()).hexdigest(),
            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "quality_gate_pass": True, "mapping_sha256": "same-map",
            "paired_environment_fingerprint": "same-settings",
            "execution_fingerprint": arm + "-settings",
            "baseline_income_map_sha256": "same-income", "run_id": "reused",
        }), encoding="utf-8")
        paths[arm] = path
    with pytest.raises(ValueError, match="distinct run IDs"):
        paired.verify_manifests(paths["restricted"], paths["control"],
                                roster=roster, days=["2020-11-24"])


def test_exporter_checks_environment_and_writes_manifest(tmp_path, monkeypatch):
    day = "2020-11-24"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    metric = {"aid": "a", "status": "ok", "experience_environment_id": "covid_2021",
              "experience_policy_ids": [], "execution_fingerprint": "restricted-settings",
              "experience_run_id": "restricted-run",
              "paired_environment_fingerprint": "shared-settings",
              "s2_timing": {"n_llm_calls": 1,
              "attempts": [{"status": "ok"}]}}
    (metrics_dir / f"day_{day}.jsonl").write_text(json.dumps(metric) + "\n",
                                                      encoding="utf-8")
    (tmp_path / f"cohort_{day}.json").write_text(json.dumps({
        "agent_ids": ["a"], "environment_id": "covid_2021",
        "execution_fingerprint": "restricted-settings",
        "paired_environment_fingerprint": "shared-settings",
        "baseline_income_map_sha256": "shared-income", "run_id": "restricted-run",
    }), encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(MAPPING), encoding="utf-8")

    class PolicyResult:
        def single(self):
            return {"n": 0}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def run(self, query, **_kwargs):
            if query == exporter.POLICY_QUERY:
                return PolicyResult()
            if query == exporter.STATE_QUERY:
                return [{"aid": "a", "online_spent": 20,
                         "self_month_cumulative": 120}]
            if query == exporter.SPEND_QUERY:
                return [spend("a", 100, code="I20101", cat="식사", sub="한식")]
            raise AssertionError("unexpected query")

    monkeypatch.setattr(exporter, "driver_session", lambda: Session())
    out = tmp_path / "restricted.jsonl"
    assert exporter.export(roster=["a"], days=[day], arm="restricted",
                           metrics_dir=metrics_dir, mapping_path=mapping_path, out=out) == 1
    manifest = json.loads((tmp_path / "restricted.jsonl.manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["environment_id"] == "covid_2021"
    assert manifest["paired_environment_fingerprint"] == "shared-settings"
    assert manifest["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_environment_exposure_and_stage2_fallback_block_export(tmp_path):
    day = "2020-11-24"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    path = metrics_dir / f"day_{day}.jsonl"
    row = {"aid": "a", "status": "ok", "experience_environment_id": "covid_2021",
           "experience_policy_ids": [], "s2_timing": {"n_llm_calls": 1,
           "attempts": [{"status": "ok"}]}}
    path.write_text(json.dumps({**row,
                                "experience_environment_id": "covid_no_distancing"}) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="wrong environment exposure"):
        exporter.verify_metrics(metrics_dir, [day], ["a"], "covid_2021")
    path.write_text(json.dumps({**row, "s2_fallback_only": True}) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="quality gate failed"):
        exporter.verify_metrics(metrics_dir, [day], ["a"], "covid_2021")
