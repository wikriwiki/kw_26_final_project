"""Expected payments come from frozen citizens and runtime grant rules."""
from __future__ import annotations

import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
from freeze_grant_entitlements import build  # noqa: E402


def _snapshot(path, rows):
    member = "graph/p012_completed_oct18_agent.jsonl.gz"
    payload = gzip.compress("".join(json.dumps(row) + "\n" for row in rows).encode())
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return member


def test_p010_entitlements_are_independent_of_simulated_receipts(tmp_path):
    archive = tmp_path / "agents.tar.gz"
    member = _snapshot(archive, [
        {"aid": "b", "agent": {"id": "b", "spending_level_wd": 3,
                                 "p_income_level": "중"}},
        {"aid": "a", "agent": {"id": "a", "spending_level_wd": 1,
                                 "p_income_level": "하"}},
    ])
    frozen = build(archive, member, ROOT / "data/neo4j_load/policies/P010.json",
                   expected_agents=2, require_all_recipients=True)
    assert frozen["amount_by_aid"] == {"a": 400_000, "b": 150_000}
    assert frozen["expected_issued_won"] == 550_000
    assert frozen["recipient_count"] == 2
    assert len(frozen["policy_file_sha256"]) == 64
    assert len(frozen["source_archive_sha256"]) == 64


def test_incomplete_or_duplicate_agent_snapshot_is_rejected(tmp_path):
    archive = tmp_path / "agents.tar.gz"
    row = {"aid": "a", "agent": {"id": "a", "spending_level_wd": 1}}
    member = _snapshot(archive, [row, row])
    policy = ROOT / "data/neo4j_load/policies/P010.json"
    with pytest.raises(ValueError, match="duplicate"):
        build(archive, member, policy, expected_agents=2)
    with pytest.raises(ValueError, match="count"):
        build(archive, member, policy, expected_agents=3)
    member = _snapshot(archive, [{"aid": "a", "agent": {"id": "a"}}])
    with pytest.raises(ValueError, match="without an entitlement"):
        build(archive, member, policy, expected_agents=1,
              require_all_recipients=True)
