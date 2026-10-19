"""Independent critic checks for report snapshot immutability."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from scripts.report.snapshot import (
    SnapshotError,
    build_manifest,
    snapshot_readiness,
    verify_manifest,
    write_manifest,
)


class ReportSnapshotTests(unittest.TestCase):
    def _run_tree(self, root: Path, days: int = 2) -> dict:
        start = date(2025, 7, 21)
        available = [(start + timedelta(days=index)).isoformat() for index in range(days)]
        for relative, content in {
            "summary.json": json.dumps({"args": {"start": available[0], "days": days}, "completed_at": "2026-08-02T00:00:00Z"}),
            "events.jsonl": '{"day":"2025-07-27","amt":100}\n',
            "poi_summary.json": '{}',
        }.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        for day in available:
            for relative in (
                f"metrics/day_{day}.jsonl",
                f"timing/day_{day}.json",
                f"checkpoints/done_{day}.json",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
        return {
            "run_id": "SEOUL7500",
            "root": str(root),
            "status": "completed",
            "days_present": available,
            "days_with_timing": available,
            "days_with_done_checkpoint": available,
            "plan": {"start_day": available[0], "planned_days": days, "agents_target": 2},
            "completed_at": "2026-08-02T00:00:00Z",
        }

    def test_manifest_hashes_sources_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="snapshot-critic-") as temp:
            data_root = Path(temp)
            run_root = data_root / "out_BASE"
            run = self._run_tree(run_root)
            manifest = build_manifest(
                run_id="SEOUL7500",
                run=run,
                data_root=data_root,
                requested_start=date(2025, 7, 21),
                requested_days=1,
            )
            self.assertTrue(snapshot_readiness(run_id="SEOUL7500", run=run, data_root=data_root)["ready"])
            manifest_path = write_manifest(data_root / "report.json", manifest)
            verified = verify_manifest(
                manifest_path,
                expected_run_id="SEOUL7500",
                requested_start=date(2025, 7, 21),
                requested_days=1,
                data_root=data_root,
            )
            self.assertEqual(verified["snapshot_id"], manifest["snapshot_id"])

            partial = dict(manifest)
            partial["files"] = manifest["files"][:-1]
            partial_path = write_manifest(data_root / "partial-report.json", partial)
            with self.assertRaises(SnapshotError):
                verify_manifest(
                    partial_path,
                    expected_run_id="SEOUL7500",
                    requested_start=date(2025, 7, 21),
                    requested_days=1,
                    data_root=data_root,
                )

            (run_root / "events.jsonl").write_text('{"day":"2025-07-27","amt":999}\n', encoding="utf-8")
            with self.assertRaises(SnapshotError):
                verify_manifest(
                    manifest_path,
                    expected_run_id="SEOUL7500",
                    requested_start=date(2025, 7, 21),
                    requested_days=1,
                    data_root=data_root,
                )

    def test_manifest_rejects_report_range_outside_completed_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="snapshot-range-") as temp:
            data_root = Path(temp)
            run = self._run_tree(data_root / "out_BASE")
            with self.assertRaises(SnapshotError):
                build_manifest(
                    run_id="SEOUL7500",
                    run=run,
                    data_root=data_root,
                    requested_start=date(2025, 7, 22),
                    requested_days=2,
                )


if __name__ == "__main__":
    unittest.main()
