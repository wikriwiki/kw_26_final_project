"""A missing review DB or truncated JSON must not waste whole simulation days."""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

SIM = Path(__file__).resolve().parents[3] / "scripts" / "sim"
sys.path.insert(0, str(SIM))
import poi_review_lookup as reviews  # noqa: E402
import stage2_poi as stage2  # noqa: E402
from neo4j_load import _common as graph  # noqa: E402


def response(content: str, *, finish: str = "stop", out: int = 300):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                 finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=out),
    )


def setup_stage2(monkeypatch):
    def no_graph():
        raise RuntimeError("no graph needed in this isolated test")
    monkeypatch.setattr(graph, "driver_session", no_graph)
    candidate = {"poi_id": "C_test", "price_band": 2, "price_factor": 1.0,
                 "coupon_eligible": True, "unit_anchor": 10000}
    monkeypatch.setattr(stage2, "fetch_candidates_for_events",
                        lambda *args, **kwargs: {0: [candidate]})
    monkeypatch.setattr(stage2, "build_stage2_prompt", lambda *args, **kwargs: "prompt")
    event = SimpleNamespace(category="식사", pinned_poi=None)
    return SimpleNamespace(events=[event])


def valid_pick(review_ids):
    return json.dumps({"picks": [{"order": 0, "poi_id": "C_test",
                                  "actual_spent": 10000,
                                  "actual_satisfaction": 0.7}],
                       "review_lookup_requests": review_ids})


def test_absent_review_db_is_not_created(tmp_path, monkeypatch):
    missing = tmp_path / "no-review-database.db"
    monkeypatch.setattr(reviews, "DB_PATH", missing)
    reviews._conn.cache_clear()
    try:
        assert reviews.lookup_review("C_test") is None
        assert not missing.exists()
    finally:
        reviews._conn.cache_clear()


def test_optional_review_sqlite_failure_keeps_valid_initial_picks(monkeypatch):
    stage1 = setup_stage2(monkeypatch)
    caps = []
    def llm(*args, **kwargs):
        caps.append(kwargs["max_tokens"])
        return response(valid_pick(["C_test"]))
    monkeypatch.setattr(stage2, "_llm_call", llm)
    def broken_review(*args, **kwargs):
        raise sqlite3.OperationalError("review database unavailable")
    monkeypatch.setattr(stage2, "lookup_reviews_batch", broken_review)
    result, _, meta = stage2.call_stage2(
        "a", stage1, {"daily_wd": 30000}, date(2020, 5, 11))
    assert [p.poi_id for p in result.picks] == ["C_test"]
    assert caps == [2200]
    assert meta["review_lookup_error"] == 1
    assert meta["s2_timing"]["n_llm_calls"] == 1
    assert (meta["tokens_in"], meta["tokens_out"]) == (100, 300)


def test_length_retry_increases_cap_and_retains_llm_choice(monkeypatch):
    stage1 = setup_stage2(monkeypatch)
    caps = []
    def llm(*args, **kwargs):
        caps.append(kwargs["max_tokens"])
        if len(caps) == 1:
            return response('{"picks": [{', finish="length", out=2200)
        return response(valid_pick([]), out=420)
    monkeypatch.setattr(stage2, "_llm_call", llm)
    result, _, meta = stage2.call_stage2(
        "a", stage1, {"daily_wd": 30000}, date(2020, 5, 11))
    assert [p.poi_id for p in result.picks] == ["C_test"]
    assert caps == [2200, 3200]
    assert meta["s2_timing"]["attempts"][0]["output_limited"] is True
    assert meta["s2_timing"]["n_llm_calls"] == 2
    assert (meta["tokens_in"], meta["tokens_out"]) == (200, 2620)


def test_all_failed_llm_attempts_cannot_be_reported_as_citizen_choice(monkeypatch):
    stage1 = setup_stage2(monkeypatch)
    monkeypatch.delenv("SIM_ALLOW_STAGE2_FALLBACK", raising=False)
    monkeypatch.setattr(stage2, "_llm_call",
                        lambda *args, **kwargs: response('{"picks": [{',
                                                         finish="length", out=2200))
    with pytest.raises(RuntimeError, match="Stage2 failed"):
        stage2.call_stage2("a", stage1, {"daily_wd": 30000},
                           date(2020, 5, 11), max_retry=0)
