"""3D 지도 서빙(web/api/viz.py) — 껍데기, 정적 파일, 데이터 팩."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from scripts.sim.build_standalone_html import STATIC_JS_FILES
from scripts.sim.visualization_3d import pack
from web.api.app import create_app
from web.api.runner import RunLock, Runner
from web.api.store import ArtifactStore

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "web" / "fixtures"


def _small_pack(out_dir: Path) -> dict:
    agents = [{"id": "AGT_A", "dist_code": "11110", "home_lon": 126.97, "home_lat": 37.57}]
    timeline = [
        {"day": "2026-05-01", "hour": 9, "label": "D1 09:00", "agents": [{"id": "AGT_A", "lon": 126.98, "lat": 37.56}]},
        {"day": "2026-05-02", "hour": 9, "label": "D2 09:00", "agents": []},
    ]
    manifest, files = pack.build_pack(agents, timeline, {}, {}, policy_dongs=[], pack_id=out_dir.name, label="테스트 팩")
    pack.write_pack(out_dir, manifest, files)
    return manifest


class VizApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="viz-api-")
        self.root = Path(self.temp.name)
        self.packs = self.root / "packs"
        self.manifest = _small_pack(self.packs / "DEMO1")
        (self.packs / "not-a-pack").mkdir()
        store = ArtifactStore(repo_root=ROOT, data_root=self.root / "data", fixture_dir=FIXTURES)
        runner = Runner(repo_root=ROOT, lock=RunLock(self.root / "run.lock"))
        self.client = TestClient(create_app(store=store, runner=runner))
        self.env = patch.dict(os.environ, {"SIM_VIZ_PACKS_DIR": str(self.packs)})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.temp.cleanup()

    def test_pack_index_lists_only_real_packs_without_paths(self) -> None:
        body = self.client.get("/api/viz/packs").json()
        self.assertEqual(body["total"], 1)
        item = body["items"][0]
        self.assertEqual(item["pack_id"], "DEMO1")
        self.assertEqual(item["label"], "테스트 팩")
        self.assertEqual((item["first_day"], item["last_day"], item["day_count"]), ("2026-05-01", "2026-05-02", 2))
        self.assertEqual(item["shell_url"], "/viz/shell.html?pack=/viz/packs/DEMO1/")
        self.assertNotIn(str(self.packs), json.dumps(body, ensure_ascii=False))

    def test_pack_index_reports_unknown_root(self) -> None:
        with patch.dict(os.environ, {"SIM_VIZ_PACKS_DIR": str(self.root / "missing")}):
            body = self.client.get("/api/viz/packs").json()
        self.assertEqual(body, {"total": 0, "items": [], "unknown": ["viz_packs_root"]})

    def test_manifest_revalidates_and_hashed_chunks_are_immutable(self) -> None:
        manifest = self.client.get("/viz/packs/DEMO1/manifest.json")
        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(manifest.headers["cache-control"], "no-cache")
        self.assertEqual(manifest.json()["agent_count"], 1)

        chunk = self.manifest["chunks"][0]
        hashed = self.client.get(f"/viz/packs/DEMO1/{chunk['file']}?v={chunk['sha256'][:16]}")
        self.assertEqual(hashed.status_code, 200)
        self.assertEqual(hashed.headers["cache-control"], "public, max-age=31536000, immutable")
        self.assertEqual(len(hashed.content), chunk["bytes"])
        plain = self.client.get(f"/viz/packs/DEMO1/{chunk['file']}")
        self.assertEqual(plain.headers["cache-control"], "no-cache")

    def test_pack_files_cannot_escape_the_pack(self) -> None:
        (self.root / "secret.json").write_text("{}", encoding="utf-8")
        for url in (
            "/viz/packs/DEMO1/..%2F..%2Fsecret.json",
            "/viz/packs/DEMO1/../../secret.json",
            "/viz/packs/DEMO1/missing.json",
            "/viz/packs/DEMO1/manifest.txt",
        ):
            self.assertEqual(self.client.get(url).status_code, 404, url)
        self.assertEqual(self.client.get("/viz/packs/.hidden/manifest.json").status_code, 400)

    def test_shell_loads_runtime_from_static_in_order(self) -> None:
        response = self.client.get("/viz/shell.html")
        self.assertEqual(response.status_code, 200)
        html = response.text
        positions = [html.index(f"/viz/static/{name}?v=") for name in STATIC_JS_FILES]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("deck.gl@", html)
        self.assertIn("maplibre-gl@", html)
        self.assertNotIn("<!-- __SIM_", html)
        self.assertNotIn("/viz/theme.css", html)
        self.assertIn("/viz/theme.css", self.client.get("/viz/shell.html?theme=console").text)

    def test_static_files_are_whitelisted(self) -> None:
        script = self.client.get("/viz/static/pack_loader.js?v=1")
        self.assertEqual(script.status_code, 200)
        self.assertIn("text/javascript", script.headers["content-type"])
        self.assertIn("Sim3D.openPack", script.text)
        self.assertEqual(self.client.get("/viz/static/styles.css").status_code, 200)
        self.assertEqual(self.client.get("/viz/static/derive.py").status_code, 404)
        self.assertEqual(self.client.get("/viz/static/..%2Fpack.py").status_code, 404)


if __name__ == "__main__":
    unittest.main()
