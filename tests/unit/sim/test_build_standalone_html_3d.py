import json

import pytest

from scripts.sim import build_standalone_html
from scripts.sim.visualization_3d import assets


def _write_fixture(viz_dir, unsafe_text="Home A"):
    agents = [
        {
            "id": "AGT_A",
            "dist_code": "11110",
            "home_lon": 126.97,
            "home_lat": 37.57,
            "home_poi_name": unsafe_text,
        }
    ]
    timeline = [
        {
            "day": "2026-05-01",
            "hour": 12,
            "label": "Day 1 12:00",
            "agents": [
                {
                    "id": "AGT_A",
                    "lon": 126.981,
                    "lat": 37.566,
                    "cat": "Cafe",
                    "l1": "Cafe",
                    "spent": 12000,
                    "sat": 0.8,
                    "anchor": "lunch",
                }
            ],
        }
    ]
    memories = {
        "AGT_A": {
            "appointments": [],
            "memories": [],
            "visited": [],
            "knows_poi": [],
            "state": {},
        }
    }
    events = {
        "AGT_A": [
            {
                "day": "2026-05-01",
                "time": "12:00",
                "poi_id": "C_1",
                "poi_name": unsafe_text,
                "lon": 126.981,
                "lat": 37.566,
                "cat": "Cafe",
                "l1": "Cafe",
                "spent": 12000,
                "sat": 0.8,
            }
        ]
    }

    for filename, payload in [
        ("agents.json", agents),
        ("timeline.json", timeline),
        ("memories.json", memories),
        ("events.json", events),
    ]:
        (viz_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )


def test_build_3d_standalone_embeds_data_and_runtime_assets(tmp_path, monkeypatch):
    _write_fixture(tmp_path)
    monkeypatch.setattr(
        build_standalone_html,
        "fetch_url",
        lambda url: f"/* asset: {url} */",
    )

    out_path = build_standalone_html.build_3d_standalone(tmp_path)

    assert out_path == tmp_path / "sim_standalone.html"
    html = out_path.read_text(encoding="utf-8")
    assert "window.__AGENTS__" in html
    assert "window.__VIZ_META__" in html
    assert "AGT_A" in html
    assert "deck.gl" in html
    assert "maplibre-gl" in html
    assert "chart.js" in html
    assert "__SIM_DATA__" in html
    assert "__VIZ_META__" in html
    assert "fetch('agents.json')" not in html


def test_build_3d_standalone_escapes_script_terminators_in_embedded_json(
    tmp_path,
    monkeypatch,
):
    unsafe_text = "</script><script>alert(1)</script>"
    _write_fixture(tmp_path, unsafe_text=unsafe_text)
    monkeypatch.setattr(
        build_standalone_html,
        "fetch_url",
        lambda url: f"/* asset: {url} */",
    )

    out_path = build_standalone_html.build_3d_standalone(tmp_path)

    html = out_path.read_text(encoding="utf-8")
    assert "window.__AGENTS__" in html
    assert unsafe_text not in html
    assert "</script><script" not in html.lower()
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)\\u003c/script\\u003e" in html


def test_script_tag_escapes_closing_script_sequences():
    wrapped = assets.script_tag("console.log('</ScRiPt><script>alert(1)</script>')")

    assert "</script><script" not in wrapped.lower()
    assert "<\\/ScRiPt>" in wrapped


def test_style_tag_escapes_closing_style_sequences():
    wrapped = assets.style_tag("body::before { content: '</StYlE><script>alert(1)</script>'; }")

    assert "</style><script" not in wrapped.lower()
    assert "<\\/StYlE>" in wrapped


def test_build_3d_standalone_resolves_viz_out_dir_at_call_time(
    tmp_path,
    monkeypatch,
):
    env_dir = tmp_path / "env-output"
    env_dir.mkdir()
    _write_fixture(env_dir)
    monkeypatch.setenv("VIZ_OUT_DIR", str(env_dir))
    monkeypatch.setattr(
        build_standalone_html,
        "fetch_url",
        lambda url: f"/* asset: {url} */",
    )

    out_path = build_standalone_html.build_3d_standalone()

    assert out_path == env_dir / "sim_standalone.html"
    assert out_path.exists()


def test_fetch_url_uses_finite_timeout(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"body"

    def fake_urlopen(request, timeout):
        seen["request"] = request
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(build_standalone_html.urllib.request, "urlopen", fake_urlopen)

    assert build_standalone_html.fetch_url("https://example.test/asset.js") == "body"
    assert seen["timeout"] == 30


def test_asset_urls_are_pinned_to_concrete_versions():
    urls = [
        assets.DECK_JS_URL,
        assets.MAPLIBRE_JS_URL,
        assets.MAPLIBRE_CSS_URL,
        assets.CHART_JS_URL,
    ]

    assert all("@^" not in url for url in urls)
    assert all("@latest" not in url for url in urls)
    assert "deck.gl@" in assets.DECK_JS_URL
    assert "maplibre-gl@" in assets.MAPLIBRE_JS_URL
    assert "maplibre-gl@" in assets.MAPLIBRE_CSS_URL
    assert "chart.js@" in assets.CHART_JS_URL


def test_render_template_accepts_logical_keys_and_replaces_all_markers():
    rendered = build_standalone_html.render_template(
        (
            "<html><!-- __SIM_STYLES__ --><!-- __SIM_STYLES__ -->"
            "<!-- __SIM_DATA__ --><!-- __SIM_SCRIPTS__ --></html>"
        ),
        {
            "styles": "<style></style>",
            "data": "<script id=\"__SIM_DATA__\"></script>",
            "scripts": "<script></script>",
        },
    )

    assert rendered.count("<style></style>") == 2
    assert "<!-- __SIM_STYLES__ -->" not in rendered
    assert "<!-- __SIM_DATA__ -->" not in rendered
    assert "<!-- __SIM_SCRIPTS__ -->" not in rendered


def test_render_template_rejects_missing_required_marker():
    with pytest.raises(RuntimeError, match="missing template marker"):
        build_standalone_html.render_template(
            "<html><!-- __SIM_STYLES__ --><!-- __SIM_SCRIPTS__ --></html>",
            {
                "styles": "<style></style>",
                "data": "<script></script>",
                "scripts": "<script></script>",
            },
        )
