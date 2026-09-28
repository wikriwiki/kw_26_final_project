import json

import pytest

from deploy.vast import backup_checkpoint as backup
from scripts.sim.evidence_integrity import seal


def fixture_day(tmp_path):
    day = "2017-11-19"
    run = tmp_path / "run"
    (run / "metrics").mkdir(parents=True)
    (run / "evidence" / "v1" / day).mkdir(parents=True)
    (run / "evidence" / "v1" / day / "agent.jsonl").write_text("sealed evidence\n")
    (run / "experiment_run.json").write_text("{}")
    (run / "summary.json").write_text("{}")
    row = {"aid": "A", "status": "ok", "experience_day": day,
           "experience_run_id": "run-1", "no_smoking": {"arm": "off"}}
    (run / "metrics" / f"day_{day}.jsonl").write_text(json.dumps(seal(row)) + "\n")
    marker = {"run_id": "run-1", "arm": "off", "day": day, "status": "complete"}
    (run / f"night2_completed_{day}.json").write_text(json.dumps(seal(marker)))
    return run, day, {"cohort_ids": ["A"], "run_id": "run-1", "arm": "off"}


def test_checkpoint_selects_only_day_files_and_verifies_seals(tmp_path):
    run, day, manifest = fixture_day(tmp_path)
    (run / "metrics" / "day_2017-11-20.jsonl").write_text("future\n")
    backup.validate_day(run, day, manifest)
    files = backup.select_day_files(run, day)
    names = {p.relative_to(run).as_posix() for p in files}
    assert f"metrics/day_{day}.jsonl" in names
    assert f"evidence/v1/{day}/agent.jsonl" in names
    assert "experiment_run.json" in names
    assert "metrics/day_2017-11-20.jsonl" not in names
    archive = tmp_path / "day.tar.gz"
    assert backup.make_archive(run, day, archive) == len(files)
    assert archive.stat().st_size > 0


def test_checkpoint_accepts_explicit_skipped_agent_day_but_rejects_fake_zero(tmp_path):
    run, day, manifest = fixture_day(tmp_path)
    path = run / "metrics" / f"day_{day}.jsonl"
    skipped = {"aid": "A", "status": "skipped", "experience_day": day,
               "experience_run_id": "run-1", "skip_kind": "failed_after_retries",
               "attempts": 6, "observed_behavior": False,
               "no_smoking": {"arm": "off", "observed_behavior": False}}
    path.write_text(json.dumps(seal(skipped)) + "\n")
    backup.validate_day(run, day, manifest)
    skipped["no_smoking"]["by_poi"] = []
    skipped["observed_behavior"] = True
    path.write_text(json.dumps(seal(skipped)) + "\n")
    with pytest.raises(ValueError, match="Skipped day"):
        backup.validate_day(run, day, manifest)


@pytest.mark.parametrize("change", ["duplicate", "wrong_run", "missing_night"])
def test_checkpoint_rejects_incomplete_or_foreign_day(tmp_path, change):
    run, day, manifest = fixture_day(tmp_path)
    path = run / "metrics" / f"day_{day}.jsonl"
    if change == "duplicate":
        path.write_text(path.read_text() * 2)
    elif change == "wrong_run":
        row = json.loads(path.read_text())
        row["experience_run_id"] = "someone-else"
        path.write_text(json.dumps(seal(row)))
    else:
        (run / f"night2_completed_{day}.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        backup.validate_day(run, day, manifest)
