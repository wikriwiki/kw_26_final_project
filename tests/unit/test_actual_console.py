"""Actual-data integration smoke tests for the console API."""
from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from web.api.app import create_app
from web.api.store import ArtifactStore


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SIM_DATA_ROOT", r"C:\Users\srdyh\gpu_exp_data\20260802"))


@unittest.skipUnless(DATA_ROOT.is_dir(), "documented actual data root is not mounted")
class ActualConsoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(create_app(store=ArtifactStore(repo_root=ROOT, data_root=DATA_ROOT)))

    def test_first_screen_uses_lightweight_actual_counts(self) -> None:
        started = time.perf_counter()
        response = self.client.get("/api/runs/BASE7500/days")
        elapsed = time.perf_counter() - started
        self.assertEqual(response.status_code, 200)
        self.assertLess(elapsed, 2.0)
        item = response.json()["items"][0]
        self.assertEqual(item["counts_source"], "status_scan")
        self.assertIsNone(item["progress_ratio"])
        self.assertIn("agents_target", item["unknown"])

    def test_large_day_is_aggregated_server_side(self) -> None:
        response = self.client.get("/api/runs/BASE7500/days/2025-07-14")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["aggregated_server_side"])
        self.assertGreater(payload["source_bytes"], 10 * 1024 * 1024)
        self.assertLess(len(response.content), payload["source_bytes"])
        self.assertEqual(sum(row["agents"] for row in payload["by_spend_decile"]), payload["agents_ok"])

    def test_actual_artifacts_and_incomplete_events_are_distinct(self) -> None:
        artifacts = self.client.get("/api/artifacts")
        self.assertEqual(artifacts.status_code, 200)
        paths = {item["path"] for item in artifacts.json()["items"]}
        self.assertIn("visualization/index.html", paths)
        self.assertIn("report/FINAL_REPORT_5D_FULL.html", paths)

        events = self.client.get("/api/runs/BASE7500/events/summary")
        self.assertEqual(events.status_code, 200)
        self.assertFalse(events.json()["available"])
        self.assertIsNone(events.json()["totals"])

    def test_actual_preflight_result_is_exposed(self) -> None:
        response = self.client.get("/api/policies/P010/validate")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["exit_code"], 0)
        self.assertTrue(payload["ok"])
        self.assertIn("배경:", payload["prompt_preview"])


if __name__ == "__main__":
    unittest.main()
