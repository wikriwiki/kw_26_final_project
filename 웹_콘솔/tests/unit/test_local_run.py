"""시연 노트북용 로컬 실행 — 지도 저장본(web/api/map_cache.py)·켤 때 미리 계산(web/api/warmup.py)."""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from web.api import map_cache
from web.api.app import create_app
from web.api.runner import RunLock, Runner
from web.api.store import ArtifactStore

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "web" / "fixtures"


class MapCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="map-cache-")
        self.cache = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"SIM_MAP_CACHE_DIR": str(self.cache)})
        self.env.start()
        store = ArtifactStore(repo_root=ROOT, data_root=self.cache / "data", fixture_dir=FIXTURES)
        runner = Runner(repo_root=ROOT, lock=RunLock(self.cache / "run.lock"))
        self.client = TestClient(create_app(store=store, runner=runner))

    def tearDown(self) -> None:
        self.env.stop()
        self.temp.cleanup()

    def test_path_outside_cache_is_refused(self) -> None:
        for bad in ("../secret", "planet/../../x.pbf", "a\\b", ""):
            with self.assertRaises(ValueError):
                map_cache.upstream_path(bad)
        self.assertEqual(map_cache.upstream_path("fonts/Noto%20Sans%20Regular/0-255.pbf"), "fonts/Noto Sans Regular/0-255.pbf")

    def test_shell_uses_local_library_only_when_saved(self) -> None:
        html = self.client.get("/viz/shell.html").text
        self.assertIn('window.SIM3D_MAP_BASE = location.origin + "/viz/map"', html)
        self.assertIn("https://unpkg.com/deck.gl", html)  # 저장본이 없으면 원래 주소
        (self.cache / "vendor").mkdir()
        for name in map_cache.VENDOR:
            (self.cache / "vendor" / name).write_text("/* saved */", encoding="utf-8")
        html = self.client.get("/viz/shell.html").text
        self.assertNotIn("unpkg.com", html)
        self.assertIn('src="/viz/vendor/deck.gl.min.js"', html)
        self.assertEqual(self.client.get("/viz/vendor/deck.gl.min.js").text, "/* saved */")

    def test_saved_style_points_back_to_this_server(self) -> None:
        style = self.cache / "ofm" / "styles" / "bright.json"
        style.parent.mkdir(parents=True)
        style.write_text('{"sprite":"https://tiles.openfreemap.org/sprites/x/ofm"}', encoding="utf-8")
        body = self.client.get("/viz/map/styles/bright").json()
        self.assertEqual(body["sprite"], "http://testserver/viz/map/sprites/x/ofm")

    def test_missing_tile_without_internet_is_an_empty_tile(self) -> None:
        with patch.object(map_cache, "download", side_effect=OSError("offline")):
            response = self.client.get("/viz/map/planet/v1/14/1/2.pbf")
        self.assertEqual(response.status_code, 204)

    def test_fetched_tile_is_saved_for_next_time(self) -> None:
        with patch.object(map_cache, "download", return_value=b"tile") as fetch:
            self.assertEqual(self.client.get("/viz/map/planet/v1/14/1/3.pbf").content, b"tile")
            self.assertEqual(self.client.get("/viz/map/planet/v1/14/1/3.pbf").content, b"tile")
        self.assertEqual(fetch.call_count, 1)

    def test_map_routes_are_off_without_cache_setting(self) -> None:
        with patch.dict(os.environ, {"SIM_MAP_CACHE_DIR": ""}):
            self.assertEqual(self.client.get("/viz/map/styles/bright").status_code, 404)
            self.assertIn("https://unpkg.com/maplibre-gl", self.client.get("/viz/shell.html").text)


class WarmupTests(unittest.TestCase):
    def test_warmup_reports_done(self) -> None:
        with tempfile.TemporaryDirectory(prefix="warmup-") as temp:
            root = Path(temp)
            store = ArtifactStore(repo_root=ROOT, data_root=root / "data", fixture_dir=FIXTURES)
            runner = Runner(repo_root=ROOT, lock=RunLock(root / "run.lock"))
            with patch.dict(os.environ, {"WEB_WARMUP": "1", "SIM_EXPERIMENTS_ROOT": str(root / "experiments")}):
                app = create_app(store=store, runner=runner)
                with TestClient(app) as client:
                    for _ in range(100):
                        state = client.get("/api/warmup").json()
                        if state["state"] == "done":
                            break
                        time.sleep(0.05)
            self.assertEqual(state["state"], "done")
            self.assertEqual(state["done"], state["total"])
            # 상태 확인 응답은 계약 그대로다
            self.assertNotIn("warmup", TestClient(create_app(store=store, runner=runner)).get("/api/health").json())


if __name__ == "__main__":
    unittest.main()
