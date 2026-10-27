"""공개 시연 모드: 변경 요청은 막고, 인터뷰는 횟수를 센다."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from web.api import demo_guard
from web.api.app import create_app
from web.api.runner import RunLock, Runner
from web.api.store import ArtifactStore


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "web" / "fixtures"


class DemoModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="demo-")
        temp = Path(self.temp.name)
        store = ArtifactStore(repo_root=ROOT, data_root=temp / "data", fixture_dir=FIXTURES)
        runner = Runner(repo_root=ROOT, lock=RunLock(temp / "run.lock"))
        with patch.dict(os.environ, {"WEB_PUBLIC_DEMO": "1", "WEB_DEMO_PER_IP_HOUR": "2"}):
            self.app = create_app(store=store, runner=runner)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_changes_are_blocked(self) -> None:
        for method, path in [
            ("post", "/api/runner/start"),
            ("post", "/api/runner/stop"),
            ("post", "/api/reports/jobs"),
            ("post", "/api/llm/ping"),
            ("post", "/api/policies"),
            ("delete", "/api/policies/P017"),
            ("post", "/api/experiments/p017/sync"),
        ]:
            response = self.client.request(method.upper(), path, json={})
            self.assertEqual(response.status_code, 403, path)

    def test_policy_validation_is_open(self) -> None:
        # 새 시뮬레이션 만들기 흐름: 검증은 되고(403 아님), 실행 시작·정책 저장은 막힌다
        response = self.client.post("/api/policies/draft/validate", json={"policy": {}})
        self.assertNotIn(response.status_code, (403, 429))

    def test_docs_are_hidden(self) -> None:
        self.assertEqual(self.client.get("/openapi.json").status_code, 404)

    def test_effect_is_not_counted_but_interviews_are(self) -> None:
        # 집단 효과는 모델을 부르지 않으므로 몇 번이든 문지기를 지난다(실험이 없어 404 는 괜찮다)
        for _ in range(5):
            self.assertNotIn(self.client.post("/api/experiments/x/query/effect", json={}).status_code, (403, 429))
        path = "/api/experiments/x/agents/a/interview"
        headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1"}
        codes = [self.client.post(path, json={}, headers=headers).status_code for _ in range(3)]
        self.assertNotIn(codes[0], (403, 429))
        self.assertNotIn(codes[1], (403, 429))
        self.assertEqual(codes[2], 429)
        # 다른 방문자는 따로 센다
        other = self.client.post(path, json={}, headers={"x-forwarded-for": "198.51.100.2"})
        self.assertNotEqual(other.status_code, 429)


class LimiterTests(unittest.TestCase):
    def test_hour_window_slides(self) -> None:
        limiter = demo_guard.InterviewLimiter(per_ip_hour=1, per_day=10)
        self.assertIsNone(limiter.allow("a", now=0))
        self.assertIsNotNone(limiter.allow("a", now=10))
        self.assertIsNone(limiter.allow("a", now=3700))

    def test_day_cap_is_shared(self) -> None:
        limiter = demo_guard.InterviewLimiter(per_ip_hour=10, per_day=2)
        self.assertIsNone(limiter.allow("a", now=0))
        self.assertIsNone(limiter.allow("b", now=1))
        self.assertIsNotNone(limiter.allow("c", now=2))


if __name__ == "__main__":
    unittest.main()
