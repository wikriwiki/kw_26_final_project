(function () {
  const Sim3D = (window.Sim3D = window.Sim3D || {});

  function zoom() {
    const state = Sim3D.state || {};
    return state.map && typeof state.map.getZoom === "function" ? state.map.getZoom() : 11;
  }

  function currentMap() {
    const state = Sim3D.state || {};
    return Array.isArray(state.frameMaps) && state.frameMaps[state.frameIndex] instanceof Map
      ? state.frameMaps[state.frameIndex]
      : new Map();
  }

  function activeTrips(agents) {
    const limit = zoom() >= 12 ? 900 : 220;
    return agents
      .filter(function (item) {
        return item.frame && item.frame.anchor !== "residence";
      })
      .slice(0, limit)
      .map(function (item) {
        const home = [
          item.agent && item.agent.home_lon != null ? item.agent.home_lon : item.position[0],
          item.agent && item.agent.home_lat != null ? item.agent.home_lat : item.position[1],
          6
        ];
        return { id: item.id, path: [home, item.position], color: item.color };
      });
  }

  function framePoint(frameItem) {
    if (!frameItem || frameItem.lon == null || frameItem.lat == null) return null;
    return [Number(frameItem.lon), Number(frameItem.lat), 24];
  }

  function rumorPath(edge) {
    const map = currentMap();
    const source = framePoint(map.get(edge.source_id) || map.get(String(edge.source_id)));
    const target = framePoint(map.get(edge.target_id) || map.get(String(edge.target_id)));
    if (!source || !target) return null;
    return {
      id: edge.source_id + "-" + edge.target_id + "-" + edge.frame_index,
      path: [
        source,
        [(source[0] + target[0]) / 2, (source[1] + target[1]) / 2, 130],
        target
      ],
      topic: edge.topic_value || edge.topic_type || ""
    };
  }

  function googleTilesLayer() {
    const state = Sim3D.state || {};
    if (state.baseMode !== "google" || !state.googleMapsApiKey) return null;
    return new deck.Tile3DLayer({
      id: "google-photorealistic-tiles",
      data: "https://tile.googleapis.com/v1/3dtiles/root.json",
      loadOptions: {
        fetch: {
          headers: { "X-GOOG-API-KEY": state.googleMapsApiKey }
        }
      },
      onTileError: function (_tile, _url, message) {
        console.warn("Google 3D tile load failed", message);
        if (typeof Sim3D.showNews === "function") {
          Sim3D.showNews("Photo 3D 타일 로드 실패로 3D City 모드로 돌아갑니다.");
        }
        if (typeof Sim3D.switchBaseMode === "function") {
          Sim3D.switchBaseMode("style");
        }
      }
    });
  }

  function safeColor(color, fallback) {
    return Array.isArray(color) ? color : fallback;
  }

  function makeCityDotLayer(agents, z) {
    return new deck.ScatterplotLayer({
      id: "agent-city-dots",
      data: agents,
      pickable: true,
      opacity: z < 12 ? 0.9 : 0.35,
      radiusUnits: "meters",
      getPosition: function (item) {
        return item.position;
      },
      getRadius: function (item) {
        return item.frame && Number(item.frame.spent || 0) > 0 ? 26 : 15;
      },
      getFillColor: function (item) {
        return safeColor(item.color, [154, 169, 190, 180]);
      },
      stroked: true,
      getLineColor: [230, 248, 255, 90],
      lineWidthUnits: "pixels",
      getLineWidth: 1,
      onClick: function (info) {
        if (info.object && typeof Sim3D.selectAgent === "function") {
          Sim3D.selectAgent(info.object.id);
        }
      }
    });
  }

  function makeTripLayer(agents) {
    return new deck.PathLayer({
      id: "commute-light-trips",
      data: activeTrips(agents),
      widthUnits: "pixels",
      getPath: function (item) {
        return item.path;
      },
      getColor: function (item) {
        const color = safeColor(item.color, [57, 215, 255, 120]);
        return [color[0], color[1], color[2], 95];
      },
      getWidth: 1.4,
      rounded: true
    });
  }

  function makeSpendLayers(bursts) {
    return [
      new deck.ColumnLayer({
        id: "spend-columns",
        data: bursts,
        diskResolution: 48,
        radius: 24,
        extruded: true,
        elevationScale: 0.018,
        getPosition: function (item) {
          return [item.lon, item.lat];
        },
        getElevation: function (item) {
          return Math.max(120, Number(item.amount || 0));
        },
        getFillColor: function (item) {
          return Number(item.sat || 0) >= 0.75 ? [63, 242, 184, 180] : [255, 191, 90, 160];
        },
        pickable: true
      }),
      new deck.ScatterplotLayer({
        id: "spend-rings",
        data: bursts,
        radiusUnits: "meters",
        getPosition: function (item) {
          return [item.lon, item.lat, 4];
        },
        getRadius: function (item) {
          return 28 + Math.min(80, Number(item.amount || 0) / 450);
        },
        getFillColor: [0, 0, 0, 0],
        stroked: true,
        getLineColor: [57, 215, 255, 210],
        lineWidthUnits: "pixels",
        getLineWidth: 2
      })
    ];
  }

  function makeInteractionLayers(meetups, rumors) {
    return [
      new deck.ArcLayer({
        id: "meetup-arcs",
        data: meetups,
        getSourcePosition: function (item) {
          return [item.lon - 0.002, item.lat, 50];
        },
        getTargetPosition: function (item) {
          return [item.lon + 0.002, item.lat, 50];
        },
        getSourceColor: [169, 139, 255, 210],
        getTargetColor: [57, 215, 255, 210],
        getWidth: 3,
        pickable: true
      }),
      new deck.PathLayer({
        id: "rumor-pulses",
        data: rumors,
        getPath: function (item) {
          return item.path;
        },
        getColor: [255, 191, 90, 180],
        getWidth: 2,
        widthUnits: "pixels",
        rounded: true
      })
    ];
  }

  Sim3D.makeLayers = function makeLayers() {
    if (!window.deck) return [];
    const z = zoom();
    const agents = typeof Sim3D.getInterpolatedAgents === "function" ? Sim3D.getInterpolatedAgents() : [];
    const bursts = typeof Sim3D.getCurrentBursts === "function" ? Sim3D.getCurrentBursts() : [];
    const meetups = typeof Sim3D.getCurrentMeetups === "function" ? Sim3D.getCurrentMeetups() : [];
    const rumorPaths = typeof Sim3D.getCurrentRumors === "function"
      ? Sim3D.getCurrentRumors().map(rumorPath).filter(Boolean)
      : [];
    const layers = [];
    const google = googleTilesLayer();
    if (google) layers.push(google);

    layers.push(makeCityDotLayer(agents, z));

    if (z >= 12) {
      layers.push(makeTripLayer(agents));
    }

    if (z >= 14) {
      layers.push.apply(layers, makeSpendLayers(bursts));
    }

    if (z >= 13) {
      layers.push.apply(layers, makeInteractionLayers(meetups, rumorPaths));
    }

    return layers;
  };
})();
