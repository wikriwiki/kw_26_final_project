"""보고서 v2 job 의 **끝에서 끝까지** 검사.

목업 없이 진짜로 돌린다. 합성 run 산출물을 만들고, API 에 job 을 요청하고,
실제 자식 프로세스가 HTML·Markdown·계산결과 JSON 을 낼 때까지 기다린 뒤
파일 내용을 확인한다. "완료로 표시되었다"만으로는 통과시키지 않는다.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from web.api.app import create_app
from web.api.runner import RunLock, Runner
from web.api.store import ArtifactStore

from . import _demo_run

ROOT = Path(__file__).resolve().parents[3]
TIMEOUT_SEC = 180


class ReportJobV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="report-v2-job-")
        base = Path(self.temp.name)
        self.data_root = base / "data"
        # 스토어는 BASE 를 data_root/out_BASE 로 찾는다
        _demo_run.build(self.data_root / "out_BASE")
        (self.data_root / "logs_scripts").mkdir(parents=True, exist_ok=True)

        self.policy_dir = base / "policies"
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        (self.policy_dir / "P777.json").write_text(
            json.dumps(_demo_run.policy(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self.output_root = base / "output"
        (self.output_root / "report").mkdir(parents=True, exist_ok=True)

        store = ArtifactStore(
            repo_root=ROOT,
            data_root=self.data_root,
            policy_dir=self.policy_dir,
            output_root=self.output_root,
        )
        runner = Runner(repo_root=ROOT, lock=RunLock(base / "run.lock"))
        self.app = create_app(store=store, runner=runner)
        # report lock 을 저장소 밖으로 돌린다 — 테스트가 작업 트리를 건드리지 않게
        self.app.state.report_jobs.lock.path = base / "report.lock"
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _wait(self, job_id: str) -> dict:
        deadline = time.time() + TIMEOUT_SEC
        while time.time() < deadline:
            payload = self.client.get(f"/api/reports/jobs/{job_id}").json()
            if payload["state"] in {"completed", "failed"}:
                return payload
            time.sleep(0.4)
        self.fail(f"보고서 job 이 {TIMEOUT_SEC}초 안에 끝나지 않았습니다")

    def test_catalog_binds_the_policy_through_actual_payment_records(self) -> None:
        response = self.client.get(
            "/api/reports/catalog", params={"run_id": "BASE", "policy_id": "P777"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["engine_v2"]["available"], payload["engine_v2"])
        self.assertTrue(payload["policy_binding"]["bound"])
        self.assertEqual(payload["policy_binding"]["source"], "events.jsonl policy payments")
        engines = {item["id"]: item for item in payload["engines"]}
        self.assertTrue(engines["v2"]["available"])
        self.assertFalse(engines["dasol"]["available"])

    def test_job_produces_a_real_self_contained_report(self) -> None:
        response = self.client.post(
            "/api/reports/jobs",
            json={
                "run_id": "BASE",
                "policy_id": "P777",
                "start": _demo_run.START.isoformat(),
                "days": _demo_run.PRE_DAYS + _demo_run.POST_DAYS,
                "policy_from": _demo_run.POLICY_FROM.isoformat(),
                "analyses": [],
                "engine": "v2",
                "use_llm": False,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        job = self._wait(response.json()["job_id"])
        self.assertEqual(job["state"], "completed", job.get("logs"))
        self.assertTrue(job["consistent"], job.get("error"))

        html_path = self.output_root / job["output_path"]
        self.assertTrue(html_path.is_file())
        html = html_path.read_text(encoding="utf-8")
        self.assertIn("이중차분", html)
        self.assertIn("시행 전후 겹쳐보기", html)
        self.assertIn("일관성 검증", html)
        self.assertGreater(html.count("<svg"), 10)
        self.assertNotIn("gradient", html.lower())

        # 계산 결과 원본이 함께 나와야 재현·검증이 가능하다
        data_path = html_path.with_suffix(".data.json")
        self.assertTrue(data_path.is_file())
        data = json.loads(data_path.read_text(encoding="utf-8"))
        self.assertTrue(data["consistency"]["consistent"])
        self.assertAlmostEqual(
            data["bundle"]["did"]["did_absolute"], _demo_run.expected_did_absolute(), delta=1.0
        )
        self.assertFalse(data["narration"]["used_llm"])

        self.assertIn(html_path.name, " ".join(job["artifacts"]))
        self.assertIn(data_path.name, " ".join(job["artifacts"]))

    def test_section_selection_reaches_the_generated_file(self) -> None:
        response = self.client.post(
            "/api/reports/jobs",
            json={
                "run_id": "BASE",
                "policy_id": "P777",
                "start": _demo_run.START.isoformat(),
                "days": _demo_run.PRE_DAYS + _demo_run.POST_DAYS,
                "policy_from": _demo_run.POLICY_FROM.isoformat(),
                "analyses": ["s6", "s7"],
                "engine": "v2",
                "use_llm": False,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        job = self._wait(response.json()["job_id"])
        self.assertEqual(job["state"], "completed", job.get("logs"))
        # 서버가 필수 절을 되돌려 넣는다
        self.assertEqual(job["analyses"], ["s1", "s6", "s7", "s10", "s11"])
        html = (self.output_root / job["output_path"]).read_text(encoding="utf-8")
        self.assertIn('id="s6"', html)
        self.assertIn('id="s10"', html)
        self.assertNotIn('id="s3"', html)

    def test_wrong_policy_for_the_run_is_refused(self) -> None:
        other = {**_demo_run.policy(), "id": "P778"}
        (self.policy_dir / "P778.json").write_text(
            json.dumps(other, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        response = self.client.post(
            "/api/reports/jobs",
            json={
                "run_id": "BASE",
                "policy_id": "P778",
                "start": _demo_run.START.isoformat(),
                "days": 8,
                "analyses": [],
                "engine": "v2",
            },
        )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("결제 기록", response.json()["error"])

    def test_simulation_lock_blocks_report_generation(self) -> None:
        self.app.state.runner.lock.acquire(run_id="BASE", policy_id="P777")
        try:
            response = self.client.post(
                "/api/reports/jobs",
                json={
                    "run_id": "BASE",
                    "policy_id": "P777",
                    "start": _demo_run.START.isoformat(),
                    "days": 8,
                    "analyses": [],
                    "engine": "v2",
                },
            )
        finally:
            self.app.state.runner.lock.release()
        self.assertEqual(response.status_code, 409)
        self.assertIn("실행 lock", response.json()["error"])

    def test_report_lock_allows_only_one_job_at_a_time(self) -> None:
        owner = self.app.state.report_jobs.lock.acquire(
            job_id="rpt-manual", run_id="BASE", policy_id="P777"
        )
        try:
            response = self.client.post(
                "/api/reports/jobs",
                json={
                    "run_id": "BASE",
                    "policy_id": "P777",
                    "start": _demo_run.START.isoformat(),
                    "days": 8,
                    "analyses": [],
                    "engine": "v2",
                },
            )
        finally:
            self.app.state.report_jobs.lock.release(owner["job_id"])
        self.assertEqual(response.status_code, 409)
        self.assertIn("보고서", response.json()["error"])


if __name__ == "__main__":
    unittest.main()
