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
            "lightgray",
            200,
            "royalblue",
            400,
            "lightblue"
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
