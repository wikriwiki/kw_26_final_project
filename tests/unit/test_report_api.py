"""DASOL report catalog/job contract tests.

The tests use the explicit S1 fixture provider and verify that the API exposes
real applicability/lock/snapshot decisions without starting a fake report
process.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from web.api.app import create_app
from web.api.report_jobs import _neo4j_source_status
from web.api.runner import RunLock, Runner
from web.api.store import ArtifactStore


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "web" / "fixtures"


class ReportApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="report-api-")
        temp_path = Path(self.temp.name)
        store = ArtifactStore(
            repo_root=ROOT,
            data_root=temp_path / "data",
            fixture_dir=FIXTURES,
        )
        self.runner = Runner(repo_root=ROOT, lock=RunLock(temp_path / "run.lock"))
        self.client = TestClient(create_app(store=store, runner=self.runner))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_catalog_separates_applicable_menu_items_and_reports_lock(self) -> None:
        response = self.client.get("/api/reports/catalog", params={"run_id": "BASE", "policy_id": "P010"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["run"]["status"], "completed")
        self.assertFalse(payload["report_lock"]["locked"])
        self.assertIn("configured", payload["engine"])
        self.assertIn("uri", payload["engine"])
        self.assertFalse(payload["engine"]["snapshot_bound"])
        self.assertFalse(payload["engine"]["binding_verified"])
        items = {item["id"]: item for item in payload["analyses"]}
        self.assertFalse(items["sales"]["applicable"])
        self.assertIn("income_grants", items["sales"]["disabled_reason"])
        self.assertFalse(items["spillover"]["applicable"])
        self.assertTrue(items["spillover"]["disabled_reason"])
        self.assertTrue(items["triggers"]["applicable"])

    def test_environment_binding_claim_is_not_treated_as_verified_provenance(self) -> None:
        temp_path = Path(self.temp.name)
        env_dir = temp_path / "data" / "neo4j_load"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / ".env").write_text(
            "NEO4J_PASSWORD=test-only\nDASOL_NEO4J_RUN_ID=BASE\n",
            encoding="utf-8",
        )
        with mock.patch.dict("os.environ", {}, clear=True):
            status = _neo4j_source_status(temp_path, run_id="BASE")
        self.assertTrue(status["configured"])
        self.assertTrue(status["binding_declared"])
        self.assertFalse(status["binding_verified"])
        self.assertFalse(status["snapshot_bound"])
        self.assertEqual(status["verification_level"], "environment_only")
        self.assertIn("hash", status["reason"])

    def test_incomplete_snapshot_is_rejected_before_job_creation(self) -> None:
        response = self.client.post(
            "/api/reports/jobs",
            json={
                "run_id": "BASE7500",
                "policy_id": "P010",
                "start": "2025-07-14",
                "days": 1,
                "analyses": ["triggers"],
            },
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("완료된 run", response.json()["error"])

    def test_runner_lock_is_checked_before_report_job(self) -> None:
        self.runner.lock.acquire(run_id="BASE", policy_id="P010")
        try:
            response = self.client.post(
                "/api/reports/jobs",
                json={
                    "run_id": "BASE",
                    "policy_id": "P010",
                    "start": "2025-07-21",
                    "days": 7,
                    "analyses": ["triggers"],
                },
            )
        finally:
            self.runner.lock.release()
        self.assertEqual(response.status_code, 409)
        self.assertIn("실행 lock", response.json()["error"])

    def test_fixture_mode_refuses_to_claim_a_real_report_job(self) -> None:
        response = self.client.post(
            "/api/reports/jobs",
            json={
                "run_id": "BASE",
                "policy_id": "P010",
                "start": "2025-07-21",
                "days": 7,
                "analyses": ["triggers"],
            },
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("fixture", response.json()["error"])
        self.assertEqual(self.client.get("/api/reports/jobs", params={"run_id": "BASE"}).json()["total"], 0)


if __name__ == "__main__":
    unittest.main()
