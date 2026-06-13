(function () {
  const Sim3D = (window.Sim3D = window.Sim3D || {});

  function emptyFeatureCollection() {
    return { type: "FeatureCollection", features: [] };
  }

  function buildingFeatures() {
    const state = Sim3D.state || {};
    const features = state.meta && state.meta.building_features;
    return features && features.type === "FeatureCollection" ? features : emptyFeatureCollection();
  }

  function styleModeMap() {
    return {
      version: 8,
      sources: {
        cityBuildings: {
          type: "geojson",
          data: buildingFeatures()
        }
      },
      layers: [
        {
          id: "background",
          type: "background",
          paint: { "background-color": "#05070d" }
        },
        {
          id: "city-buildings",
          type: "fill-extrusion",
          source: "cityBuildings",
          minzoom: 10,
          paint: {
            "fill-extrusion-color": [
              "interpolate",
              ["linear"],
              ["get", "spent_total"],
              0,
              "#101826",
              20000,
              "#153046",
              150000,
              "#1d5b72"
            ],
            "fill-extrusion-height": ["coalesce", ["get", "height"], 18],
            "fill-extrusion-base": 0,
            "fill-extrusion-opacity": 0.72,
            "fill-extrusion-vertical-gradient": true
          }
        }
      ]
    };
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
      style: styleModeMap(),
      center: [126.978, 37.566],
      zoom: 11.2,
      pitch: 58,
      bearing: -18,
      antialias: true,
      attributionControl: false
    });

    await new Promise(function (resolve, reject) {
      state.map.once("load", resolve);
      state.map.once("error", function (event) {
        reject(event && event.error ? event.error : new Error("MapLibre failed to load"));
      });
    });

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
      state.map.setStyle(styleModeMap());
      state.map.once("styledata", function () {
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

  Sim3D.styleModeMap = styleModeMap;
})();
