"""S2 API tests using only the explicit S1 fixture provider."""
from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from web.api.app import create_app
from web.api.runner import RunLock, Runner
from web.api.store import ArtifactStore, StoreError


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "web" / "fixtures"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class S2ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="s2-api-")
        self.temp_path = Path(self.temp.name)
        store = ArtifactStore(
            repo_root=ROOT,
            data_root=self.temp_path / "data",
            fixture_dir=FIXTURES,
        )
        self.runner = Runner(repo_root=ROOT, lock=RunLock(self.temp_path / "run.lock"))
        self.client = TestClient(create_app(store=store, runner=self.runner))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_health_and_contract_version(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "contract_version": "s1.0.0", "unknown": []})

    def test_run_index_and_first_screen_use_s1_shapes(self) -> None:
        self.assertEqual(self.client.get("/api/runs").json(), fixture("runs.index.json"))
        self.assertEqual(self.client.get("/api/runs/BASE/days").json(), fixture("run.BASE.days.json"))
        rescue = self.client.get("/api/runs/BASE7500/days").json()
        self.assertIsNone(rescue["items"][0]["progress_ratio"])
        self.assertIn("agents_target", rescue["unknown"])

    def test_day_and_incomplete_subresources_are_server_contract_responses(self) -> None:
        routes = (
            ("/api/runs/BASE/days/2025-07-21", "run.BASE.day.2025-07-21.json"),
            ("/api/runs/BASE7500/days/2025-07-14/bottlenecks", "run.BASE7500.day.2025-07-14.bottlenecks.json"),
            ("/api/runs/BASE7500/days/2025-07-14/failed", "run.BASE7500.day.2025-07-14.failed.json"),
            ("/api/runs/BASE7500/events/summary", "run.BASE7500.events.summary.json"),
        )
        for route, fixture_name in routes:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, route)
            self.assertEqual(response.json(), fixture(fixture_name), route)

    def test_preflight_accepts_direct_policy_json_body(self) -> None:
        policy = fixture("policy.P010.detail.json")["policy"]
        response = self.client.post("/api/policies/P010/validate", json=policy)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["policy_id"], "P010")
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("배경:", result["prompt_preview"])

    def test_sse_emits_a_single_completed_snapshot_without_polling_loop(self) -> None:
        response = self.client.get("/api/runs/BASE/events")
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: run", response.text)
        self.assertIn('"run_id":"BASE"', response.text)

    def test_lock_is_physical_and_second_acquire_is_rejected(self) -> None:
        lock = self.runner.lock
        first = lock.acquire(run_id="BASE", policy_id="P010")
        self.assertEqual(first["state"], "starting")
        with self.assertRaises(StoreError) as ctx:
            lock.acquire(run_id="FINAL", policy_id="P011")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertTrue(lock.status()["locked"])
        self.assertTrue(lock.release()["released"])
        self.assertFalse(lock.status()["locked"])

    def test_start_does_not_fallback_to_a_mock_command(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SIM_RUN_COMMAND", None)
            os.environ.pop("SIM_RUN_COMMAND_JSON", None)
            response = self.client.post("/api/runner/start", json={"run_id": "BASE", "policy_id": "P010"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("SIM_RUN_COMMAND_JSON", response.json()["error"])

    def test_fixture_mode_never_serves_arbitrary_paths(self) -> None:
        # `../` 를 그대로 쓰면 HTTP 클라이언트가 보내기 전에 주소를 접어버려
        # (`/api/artifacts/../../x` → `/x`) 산출물 엔드포인트까지 가지도 못한다.
        # 엔드포인트의 방어를 검사하려면 접히지 않는 인코딩된 형태로 보내야 한다.
        response = self.client.get("/api/artifacts/..%2F..%2Fsecrets.txt")
        self.assertEqual(response.status_code, 404)
        response = self.client.get("/api/artifacts/report%2F..%2F..%2F..%2F.env")
        self.assertEqual(response.status_code, 404)

    def test_collapsed_traversal_falls_back_to_the_app_shell_not_the_file(self) -> None:
        """접힌 주소는 그냥 모르는 화면 주소다 — 앱 껍데기가 나오고 파일은 나오지 않는다."""
        secret = ROOT / "web" / "ui" / "dist" / ".." / ".." / ".." / "conftest.py"
        response = self.client.get("/api/artifacts/../../conftest.py")
        self.assertEqual(response.request.url.path, "/conftest.py")  # 클라이언트가 접었다
        if secret.is_file():
            self.assertNotIn(secret.read_text(encoding="utf-8")[:40], response.text)
        self.assertNotIn("import ", response.text)

    def test_invalid_day_namespace_is_rejected_before_file_access(self) -> None:
        response = self.client.get("/api/runs/BASE/days/not-a-day")
        self.assertEqual(response.status_code, 400)

    def test_policy_write_is_blocked_while_run_lock_is_held(self) -> None:
        policy = fixture("policy.P010.detail.json")["policy"]
        self.runner.lock.acquire(run_id="BASE", policy_id="P010")
        response = self.client.put("/api/policies/P010", json=policy)
        self.assertEqual(response.status_code, 409)
        self.assertIn("실행 중", response.json()["error"])
        self.runner.lock.release()


class SpaFallbackTests(unittest.TestCase):
    """`BrowserRouter` 주소를 새로고침해도 살아 있어야 한다 (스펙 §9 deep-linking).

    콘솔 화면 주소(`/runs/FINAL/report`)는 서버에 파일로 존재하지 않는다.
    클릭해서 들어가면 되지만 새로고침하거나 링크로 열면 404 가 나던 자리다.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not (ROOT / "web" / "ui" / "dist" / "index.html").is_file():
            raise unittest.SkipTest("web/ui/dist 가 없습니다 — `npm run build` 후에 검사합니다")

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="spa-")
        store = ArtifactStore(
            repo_root=ROOT, data_root=Path(self.temp.name) / "data", fixture_dir=FIXTURES
        )
        runner = Runner(repo_root=ROOT, lock=RunLock(Path(self.temp.name) / "run.lock"))
        self.client = TestClient(create_app(store=store, runner=runner))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_deep_link_serves_the_app_shell(self) -> None:
        for path in ("/runs/FINAL/report", "/runs/BASE/visualize", "/new"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.headers["content-type"])
                self.assertIn("<div id=\"root\">", response.text)

    def test_unknown_api_path_stays_a_404_not_the_app_shell(self) -> None:
        # API 가 HTML 을 돌려주면 JSON 을 기대한 클라이언트가 엉뚱한 곳에서 터진다
        response = self.client.get("/api/definitely-not-a-route")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("<div id=\"root\">", response.text)

    def test_real_assets_are_still_served_as_themselves(self) -> None:
        index = (ROOT / "web" / "ui" / "dist" / "index.html").read_text(encoding="utf-8")
        asset = re.search(r'src="(/assets/[^"]+\.js)"', index)
        self.assertIsNotNone(asset, "index.html 에 번들 script 태그가 없습니다")
        response = self.client.get(asset.group(1))
        self.assertEqual(response.status_code, 200)
        self.assertIn("javascript", response.headers["content-type"])


if __name__ == "__main__":
    unittest.main()
