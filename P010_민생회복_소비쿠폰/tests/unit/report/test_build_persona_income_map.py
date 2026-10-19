"""A fixed budget requires the same pre-existing persona anchors in two snapshots."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "build_persona_income_map", ROOT / "scripts/report/build_persona_income_map.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def archive(path: Path, values: dict[str, tuple[int, int]]) -> None:
    rows = [json.dumps({"aid": aid, "agent": {"id": aid, "s_daily_wd": wd,
                                               "s_daily_we": we}})
            for aid, (wd, we) in values.items()]
    data = gzip.compress(("\n".join(rows) + "\n").encode())
    with tarfile.open(path, "w:gz") as tar:
        member = tarfile.TarInfo("graph/test_agent.jsonl.gz")
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    path.with_name(path.name.removesuffix(".tar.gz") + ".manifest.json").write_text(
        json.dumps({"archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size}), encoding="utf-8")


def test_frozen_budget_uses_weekly_average_and_rejects_mutated_agent_inputs(tmp_path):
    a, b = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    archive(a, {"a": (70, 35), "b": (14, 14)})
    archive(b, {"a": (70, 35), "b": (14, 14)})
    result = module.build(a, b, roster)
    assert result["daily_income_by_aid"] == {"a": 60, "b": 14}
    assert result["total_daily_budget_won"] == 74
    assert result["policy_outcome_used"] is False
    archive(b, {"a": (71, 35), "b": (14, 14)})
    with pytest.raises(ValueError, match="differ across snapshots"):
        module.build(a, b, roster)
