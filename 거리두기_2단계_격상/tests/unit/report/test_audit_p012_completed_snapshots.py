"""Saved P012 evidence must independently support its State cashback ledger."""

import gzip
import hashlib
import io
import json
import tarfile

import pytest

from scripts.report.audit_p012_completed_snapshots import _transaction, audit


def _make_archive(tmp_path, day2_spend):
    path = tmp_path / "sample.tar.gz"
    states = [
        {"day": "2021-10-01", "aid": "a", "state": {"sangsaeng_month_spent": 100}},
        {"day": "2021-10-02", "aid": "a", "state": {"sangsaeng_month_spent": 110}},
    ]
    spends = [{"day": "2021-10-02", "aid": "a", "poi": {
        "type": "commerce", "sangsaeng_eligible": True}, "spend": {
            "actual_spent": day2_spend}}]
    with tarfile.open(path, "w:gz") as archive:
        for name, rows in (("sample_state.jsonl.gz", states),
                           ("sample_spend.jsonl.gz", spends)):
            payload = gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode())
            member = tarfile.TarInfo("graph/" + name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    manifest = {"archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}
    (tmp_path / "sample.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_snapshot_audit_detects_amount_mismatch(tmp_path):
    good = audit([_make_archive(tmp_path, 10)], expected_per_day=1, strict=True)
    assert good["paired_agent_days"] == 1
    assert good["mismatch_agents"] == 0

    bad = audit([_make_archive(tmp_path, 9)], expected_per_day=1)
    assert bad["mismatch_agents"] == 1
    with pytest.raises(ValueError, match="strict P012 snapshot audit failed"):
        audit([tmp_path / "sample.tar.gz"], expected_per_day=1, strict=True)


def test_missing_commerce_eligibility_cannot_be_treated_as_ineligible():
    with pytest.raises(ValueError, match="eligibility"):
        _transaction({"poi": {"type": "commerce"},
                      "spend": {"actual_spent": 10}})
    assert _transaction({"poi": {"type": "residence"},
                         "spend": {"actual_spent": 0}}) == (False, 0)
