"""The evidence copier must not read a day while the runner can still write it."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import snapshot_completed_day as snapshot  # noqa: E402


def write_day(output: Path, day: str, rows: list[dict]) -> None:
    path = output / "metrics" / f"day_{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_requires_next_day_and_preserves_failed_rows(tmp_path):
    write_day(tmp_path, "2021-10-16", [{"aid": "a", "status": "ok"},
                                      {"aid": "b", "status": "error"}])
    checkpoint = tmp_path / "checkpoints" / "done_2021-10-16.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text('["a"]', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Next day has not started"):
        snapshot.verify_finished_metrics(tmp_path, date(2021, 10, 16), 2)

    write_day(tmp_path, "2021-10-17", [{"aid": "a", "status": "ok"}])
    _, rows, counts = snapshot.verify_finished_metrics(tmp_path, date(2021, 10, 16), 2)
    assert len(rows) == 2
    assert counts["metrics_status"] == {"ok": 1, "error": 1}
    assert counts["checkpoint_done"] == 1


def test_duplicate_or_unrelated_checkpoint_is_rejected(tmp_path):
    write_day(tmp_path, "2021-10-16", [{"aid": "a", "status": "ok"},
                                      {"aid": "a", "status": "ok"}])
    write_day(tmp_path, "2021-10-17", [{"aid": "a", "status": "ok"}])
    checkpoint = tmp_path / "checkpoints" / "done_2021-10-16.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text('["a"]', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Duplicate citizen IDs"):
        snapshot.verify_finished_metrics(tmp_path, date(2021, 10, 16), 2)

    write_day(tmp_path, "2021-10-16", [{"aid": "a", "status": "ok"},
                                      {"aid": "b", "status": "ok"}])
    checkpoint.write_text('["c"]', encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        snapshot.verify_finished_metrics(tmp_path, date(2021, 10, 16), 2)


def test_final_day_requires_completed_run_marker(tmp_path):
    write_day(tmp_path, "2021-10-28", [{"aid": "a", "status": "ok"}])
    checkpoint = tmp_path / "checkpoints" / "done_2021-10-28.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text('["a"]', encoding="utf-8")
    with pytest.raises(RuntimeError, match="summary missing"):
        snapshot.verify_finished_metrics(tmp_path, date(2021, 10, 28), 1, final_day=True)
    summary = {"args": {"start": "2021-10-28", "days": 1},
               "summary": [{"day": "2021-10-28", "ok": 1}]}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="has not completed"):
        snapshot.verify_finished_metrics(tmp_path, date(2021, 10, 28), 1, final_day=True)
    summary["completed_at"] = "2026-09-26T04:00:00"
    path.write_text(json.dumps(summary), encoding="utf-8")
    _, rows, counts = snapshot.verify_finished_metrics(
        tmp_path, date(2021, 10, 28), 1, final_day=True)
    assert len(rows) == counts["checkpoint_done"] == 1


def test_script_credentials_are_parsed_without_running_script(tmp_path):
    path = tmp_path / "runner.sh"
    path.write_text("export NEO4J_URI='bolt://localhost:7687' "
                    "NEO4J_USER=reader NEO4J_PASSWORD='test value'\n"
                    "echo must-not-run\n", encoding="utf-8")
    assert snapshot.script_credentials(path) == {
        "NEO4J_URI": "bolt://localhost:7687", "NEO4J_USER": "reader",
        "NEO4J_PASSWORD": "test value"}
