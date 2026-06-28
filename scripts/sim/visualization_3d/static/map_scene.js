(function () {
  const Sim3D = (window.Sim3D = window.Sim3D || {});

  // OpenFreeMap vector tiles (no API key, free). OpenMapTiles schema.
  // Replicates https://maplibre.org/maplibre-gl-js/docs/examples/display-buildings-in-3d/
  const OPENFREEMAP_TILES = "https://tiles.openfreemap.org/planet";
  const BRIGHT_STYLE_URL = "https://tiles.openfreemap.org/styles/bright";

  // Add the 3D buildings layer exactly as the MapLibre reference example:
  // find the first text label layer and insert the extrusion beneath it.
  function add3dBuildings(map) {
    if (!map || typeof map.getStyle !== "function" || !map.getStyle()) return;
    if (map.getLayer("3d-buildings")) return;

    const layers = map.getStyle().layers || [];
    let labelLayerId;
    for (let i = 0; i < layers.length; i++) {
      if (layers[i].type === "symbol" && layers[i].layout && layers[i].layout["text-field"]) {
        labelLayerId = layers[i].id;
        break;
      }
    }

    if (!map.getSource("openfreemap")) {
      map.addSource("openfreemap", {
        url: OPENFREEMAP_TILES,
        type: "vector"
      });
    }

    map.addLayer(
      {
        id: "3d-buildings",
        source: "openfreemap",
        "source-layer": "building",
        type: "fill-extrusion",
        minzoom: 15,
        filter: ["!=", ["get", "hide_3d"], true],
        paint: {
          "fill-extrusion-color": [
            "interpolate",
            ["linear"],
            ["get", "render_height"],
            0,
            "#d8dce2",
            200,
            "#aeb6c2",
            400,
            "#9aa3ad"
          ],
          "fill-extrusion-height": [
            "interpolate",
            ["linear"],
            ["zoom"],
            15,
            0,
            16,
            ["get", "render_height"]
          ],
          "fill-extrusion-base": [
            "case",
            [">=", ["get", "zoom"], 16],
            ["get", "render_min_height"],
            0
          ]
        }
      },
      labelLayerId
    );
  }

  // Best-effort Korean labels: OpenFreeMap (OpenMapTiles schema) carries
  // name:ko on many features, but not all (rural roads/minor POIs often lack
  // it) - coalesce falls back to the style's original text-field per layer.
  function applyKoreanLabels(map) {
    if (!map || typeof map.getStyle !== "function" || !map.getStyle()) return;
    const layers = map.getStyle().layers || [];
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      if (layer.type !== "symbol" || !layer.layout || layer.layout["text-field"] == null) continue;
      try {
        const original = layer.layout["text-field"];
        map.setLayoutProperty(layer.id, "text-field", ["coalesce", ["get", "name:ko"], original]);
      } catch (error) {
        console.warn("Korean label override skipped for layer", layer.id, error);
      }
    }
  }

  // Desaturate the basemap (roads/parks/water/buildings) so data layers
  // (agent dots, heatmap, arcs) read as the visual focus - same idea as
  // MapTiler's "Data Visualization" style, applied as paint overrides on
  // top of the existing OpenFreeMap Bright style rather than swapping tile
  // providers.
  const MUTE_SOURCE_LAYERS = { water: 0.55, waterway: 0.5, park: 0.5, landcover: 0.45, landuse: 0.45 };
  const ROAD_MUTE_AMOUNT = 0.3;

  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  function parseColor(value) {
    if (typeof value !== "string") return null;
    let match = value.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (match) {
      let hex = match[1];
      if (hex.length === 3) hex = hex.split("").map(function (c) { return c + c; }).join("");
      const num = parseInt(hex, 16);
      return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
    }
    match = value.match(/^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/i);
    if (match) return [Number(match[1]), Number(match[2]), Number(match[3])];
    match = value.match(/^hsla?\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%/i);
    if (match) return hslToRgb(Number(match[1]) / 360, Number(match[2]) / 100, Number(match[3]) / 100);
    return null;
  }

  function rgbToHsl(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const l = (max + min) / 2;
    const d = max - min;
    let h = 0, s = 0;
    if (d !== 0) {
      s = d / (1 - Math.abs(2 * l - 1));
      if (max === r) h = ((g - b) / d) % 6;
      else if (max === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h *= 60;
      if (h < 0) h += 360;
    }
    return [h, s, l];
  }

  function hslToRgb(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = l - c / 2;
    let r, g, b;
    if (h < 60) { r = c; g = x; b = 0; }
    else if (h < 120) { r = x; g = c; b = 0; }
    else if (h < 180) { r = 0; g = c; b = x; }
    else if (h < 240) { r = 0; g = x; b = c; }
    else if (h < 300) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }
    return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
  }

  function desaturate(value, amount) {
    const rgb = parseColor(value);
    if (!rgb) return null;
    const hsl = rgbToHsl(rgb[0], rgb[1], rgb[2]);
    const muted = hslToRgb(hsl[0], hsl[1] * (1 - amount), clamp01(hsl[2] + (1 - hsl[2]) * 0.1));
    const toHex = function (n) { return n.toString(16).padStart(2, "0"); };
    return "#" + toHex(muted[0]) + toHex(muted[1]) + toHex(muted[2]);
  }

  function applyAnalysisPalette(map) {
    if (!map || typeof map.getStyle !== "function" || !map.getStyle()) return;
    const layers = map.getStyle().layers || [];

    layers.forEach(function (layer) {
      if (!layer.paint) return;
      const sourceLayer = layer["source-layer"];
      try {
        if (Object.prototype.hasOwnProperty.call(MUTE_SOURCE_LAYERS, sourceLayer)) {
          const amount = MUTE_SOURCE_LAYERS[sourceLayer];
          ["fill-color", "line-color"].forEach(function (prop) {
            const muted = desaturate(layer.paint[prop], amount);
            if (muted) map.setPaintProperty(layer.id, prop, muted);
          });
        } else if (sourceLayer === "transportation" && layer.paint["line-color"]) {
          const muted = desaturate(layer.paint["line-color"], ROAD_MUTE_AMOUNT);
          if (muted) map.setPaintProperty(layer.id, "line-color", muted);
        }
      } catch (error) {
        console.warn("Analysis palette override skipped for layer", layer.id, error);
      }
    });

    layers.forEach(function (layer) {
      if (layer.type !== "symbol" || !layer.layout || layer.layout["text-field"] == null) return;
      try {
        map.setPaintProperty(layer.id, "text-halo-color", "#0a0d12");
        map.setPaintProperty(layer.id, "text-halo-width", 1.4);
      } catch (error) {
        console.warn("Label halo override skipped for layer", layer.id, error);
      }
    });
  }

  function notify(message) {
    if (typeof Sim3D.showNews === "function") {
      Sim3D.showNews(message);
    } else {
      console.warn(message);
    }
  }

  function ensureMapDependencies() {
    if (!window.maplibregl || !window.maplibregl.Map) {
      throw new Error("MapLibre GL JS is not loaded");
    }
    if (!window.deck || !window.deck.MapboxOverlay) {
      throw new Error("deck.gl MapboxOverlay is not loaded");
    }
  }

  Sim3D.initMapScene = async function initMapScene() {
    ensureMapDependencies();
    const state = Sim3D.state;
    state.map = new maplibregl.Map({
      container: "map",
      style: BRIGHT_STYLE_URL,
      center: [126.978, 37.566],
      zoom: 15.5,
      pitch: 45,
      bearing: -17.6,
      antialias: true,
      attributionControl: false
    });

    // Tiles/glyphs come from OpenFreeMap over the network. Treat per-resource
    // errors as non-fatal so a slow/blocked tile never blocks the agent overlay;
    // a safety timeout resolves even if "load" is delayed.
    await new Promise(function (resolve) {
      let settled = false;
      function done() {
        if (settled) return;
        settled = true;
        resolve();
      }
      state.map.once("load", done);
      state.map.on("error", function (event) {
        const error = event && event.error ? event.error : new Error("map resource failed");
        console.warn("MapLibre resource error:", error.message || error);
        if (!settled) notify("지도 타일을 불러오지 못했습니다. 인터넷 연결을 확인해주세요.");
      });
      window.setTimeout(done, 15000);
    });

    add3dBuildings(state.map);
    applyKoreanLabels(state.map);
    applyAnalysisPalette(state.map);

    state.overlay = new deck.MapboxOverlay({
      interleaved: true,
      layers: []
    });
    state.map.addControl(state.overlay);
  };

  Sim3D.setCamera = function setCamera(view, options) {
    const state = Sim3D.state || {};
    if (!state.map || typeof state.map.easeTo !== "function" || !view) return;
    state.map.easeTo({
      center: [view.longitude, view.latitude],
      zoom: view.zoom,
      pitch: view.pitch,
      bearing: view.bearing,
      duration: options && options.duration != null ? options.duration : 900,
      easing: function (t) {
        return t * (2 - t);
      }
    });
  };

  Sim3D.cameraForChapter = function cameraForChapter(name) {
    const cameras = {
      opening: { longitude: 126.978, latitude: 37.566, zoom: 10.8, pitch: 54, bearing: -20 },
      rhythm: { longitude: 127.01, latitude: 37.54, zoom: 12.8, pitch: 62, bearing: -32 },
      policy: { longitude: 126.988, latitude: 37.568, zoom: 14.2, pitch: 68, bearing: -44 },
      interaction: { longitude: 126.985, latitude: 37.565, zoom: 15.2, pitch: 70, bearing: -36 },
      finale: { longitude: 126.995, latitude: 37.558, zoom: 11.4, pitch: 50, bearing: 12 }
    };
    return cameras[name] || cameras.opening;
  };

  Sim3D.switchBaseMode = function switchBaseMode(mode) {
    const state = Sim3D.state || {};
    if (!state.map || typeof state.map.setStyle !== "function") return;

    state.baseMode = mode;
    if (mode === "style") {
      state.map.setStyle(BRIGHT_STYLE_URL);
      state.map.once("styledata", function () {
        add3dBuildings(state.map);
        applyKoreanLabels(state.map);
        applyAnalysisPalette(state.map);
        Sim3D.refreshLayers();
      });
      return;
    }

    if (!state.googleMapsApiKey) {
      state.baseMode = "style";
      notify("Photo 3D에는 Google Maps Tile API 키가 필요합니다. 3D City 모드를 유지합니다.");
      return;
    }

    state.map.setStyle({
      version: 8,
      sources: {},
      layers: [
        {
          id: "background",
          type: "background",
          paint: { "background-color": "#05070d" }
        }
      ]
    });
    state.map.once("styledata", function () {
      Sim3D.refreshLayers();
    });
  };

  Sim3D.refreshLayers = function refreshLayers() {
    const state = Sim3D.state || {};
    if (!state.overlay || typeof state.overlay.setProps !== "function") return;
    if (typeof Sim3D.makeLayers !== "function") return;
    state.overlay.setProps({ layers: Sim3D.makeLayers() });
  };

  Sim3D.add3dBuildings = add3dBuildings;
})();
