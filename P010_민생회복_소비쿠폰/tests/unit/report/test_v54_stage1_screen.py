"""No network/model calls: frozen-input, raw-contract and co-primary checks."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/multi_policy_v53_20260927"))
import run_v54_stage1_screen as screen  # noqa: E402
from audit_v54_stage1_screen import audit  # noqa: E402

inspect_raw = screen.inspect_raw
prepare = screen.prepare
summarize = screen.summarize
verify_freeze = screen.verify_freeze


def test_prepare_freezes_original_96_without_model_call(tmp_path):
    folder = tmp_path / "screen"
    manifest = prepare(folder)
    verified, inputs, systems = verify_freeze(folder)
    assert verified == manifest
    assert len(inputs["cells"]) == 96
    assert len(systems) == 2
    assert not (folder / "responses.jsonl").exists()
    with pytest.raises(FileExistsError):
        prepare(folder)
    with (folder / "system_v54.txt").open("a", encoding="utf-8") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="frozen asset changed"):
        verify_freeze(folder)


def test_old_v3_pass_can_hide_event_count_max():
    events = [{"time": f"{8 + i // 2:02d}:{30 * (i % 2):02d}",
               "anchor": "residence", "category": "집", "intent": "휴식",
               "reasoning": "집에서 쉼", "trigger": "none"}
              for i in range(11)]
    raw = json.dumps({"events": events, "daily_propensity": 0.5}, ensure_ascii=False)
    result = inspect_raw(raw, {"date": "2020-05-14", "zones": ["11680103"],
                               "arm": "off", "case": "grant"},
                         {"work_poi_id": None})
    assert result["v3_valid"] is True
    assert result["full_structural_valid"] is False
    assert result["full_structural_errors"] == ["event_count_max"]
    assert result["event_count_max_exceeded"] is True


def test_co_primary_requires_both_contracts():
    cells = [{"aid": f"a{i}", "case": "grant", "arm": "off"} for i in range(96)]
    rows = []
    for variant in ("v53", "v54-example-contract"):
        for i, cell in enumerate(cells):
            full = i < (91 if variant == "v54-example-contract" else 80)
            rows.append({**cell, "variant": variant, "v3_valid": True,
                         "full_structural_valid": full,
                         "v3_errors": [],
                         "full_structural_errors": [] if full else ["event_count_max"],
                         "fiscal_off_attribution_screen": False,
                         "event_count_max_exceeded": not full})
    summary = summarize(rows, {"expected_responses": 192, "co_primary_thresholds": {
        "old_v3_min_pass": 92, "full_structural_min_pass": 92,
        "max_request_failures": 0, "max_fiscal_off_screen_flags": 0,
        "max_excess_paired_failures_vs_v53": 0}}, cells)
    assert summary["complete_unique_matrix"]
    assert summary["co_primary"]["old_v3_contract_pass"] is True
    assert summary["co_primary"]["full_structural_contract_pass"] is False
    assert summary["co_primary"]["overall_pass"] is False


def test_mock_192_first_response_round_trip_and_audit(tmp_path, monkeypatch):
    folder = tmp_path / "screen"
    prepare(folder)
    model = "LGAI-EXAONE/EXAONE-4.5-33B-AWQ"
    evidence = tmp_path / "model.json"
    evidence.write_text(json.dumps({
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "served_model_ids": [model], "server_pid": 123,
        "server_command": f"sglang.launch_server --model-path {model} --port 8000",
    }), encoding="utf-8")
    events = [{"time": f"{8 + i:02d}:00", "anchor": "residence",
               "category": "집", "intent": "휴식", "reasoning": "집에서 쉼",
               "trigger": "none"} for i in range(6)]
    raw = json.dumps({"events": events, "daily_propensity": 0.5}, ensure_ascii=False)

    def fake_urlopen(request, timeout):
        if isinstance(request, str):
            assert request.endswith("/models")
            body = {"data": [{"id": model}]}
        else:
            body = {"model": "request-alias", "choices": [{"message": {"content": raw},
                                                                "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 120}}
        return BytesIO(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(screen, "urlopen", fake_urlopen)
    screen.execute(folder, evidence, "http://127.0.0.1:8000/v1", workers=2)
    result = audit(folder)
    assert result["co_primary"]["overall_pass"] is True
    assert result["variants"]["v53"]["responses"] == 96
    assert result["variants"]["v54-example-contract"]["responses"] == 96
    assert len((folder / "responses.jsonl").read_text(encoding="utf-8").splitlines()) == 192
    with pytest.raises(ValueError, match="responses exist"):
        screen.execute(folder, evidence, "http://127.0.0.1:8000/v1", workers=2)
