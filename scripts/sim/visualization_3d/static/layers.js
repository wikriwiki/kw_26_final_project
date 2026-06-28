(function () {
  const Sim3D = (window.Sim3D = window.Sim3D || {});

  function zoom() {
    const state = Sim3D.state || {};
    return state.map && typeof state.map.getZoom === "function" ? state.map.getZoom() : 11;
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

  function currentDay() {
    const state = Sim3D.state || {};
    const timeline = Array.isArray(state.timeline) ? state.timeline : [];
    const frame = timeline[state.frameIndex];
    return frame ? String(frame.day || "") : "";
  }

  function makePolicyZoneLayers() {
    const state = Sim3D.state || {};
    const zones = (state.meta && state.meta.policy_zones) || [];
    const day = currentDay();
    const activeZones = zones.filter(function (zone) {
      if (!day) return true;
      if (zone.effective_from && day < zone.effective_from) return false;
      if (zone.effective_until && day > zone.effective_until) return false;
      return true;
    });

    // Thin translucent amber outline (no fill) so policy zones read as
    // boundaries without obscuring the spending heatmap underneath.
    return [
      new deck.ScatterplotLayer({
        id: "policy-zone-markers",
        data: activeZones,
        radiusUnits: "meters",
        getPosition: function (item) {
          return [item.lon, item.lat, 2];
        },
        getRadius: 220,
        getFillColor: [0, 0, 0, 0],
        stroked: true,
        getLineColor: [255, 191, 90, 150],
        lineWidthUnits: "pixels",
        getLineWidth: 1,
        pickable: true,
        onClick: function (info) {
          if (info.object && typeof Sim3D.showNews === "function") {
            Sim3D.showNews(info.object.policy_name + ": " + info.object.dong_name);
          }
        }
      })
    ];
  }

  function makeTripsLayer() {
    const z = zoom();
    const windowSize = z >= 12 ? 8 : 4;
    const trails = typeof Sim3D.getAgentTrails === "function" ? Sim3D.getAgentTrails(windowSize) : [];
    return new deck.TripsLayer({
      id: "agent-move-trails",
      data: trails,
      getPath: function (item) {
        return item.path;
      },
      getTimestamps: function (item) {
        return item.timestamps;
      },
      trailLength: windowSize,
      currentTime: windowSize,
      opacity: 0.6,
      widthMinPixels: 2,
      getColor: [57, 215, 255],
      rounded: true
    });
  }

  function makeAgentArcLayer() {
    const arcs = typeof Sim3D.getAgentMoveArcs === "function" ? Sim3D.getAgentMoveArcs() : [];
    return new deck.ArcLayer({
      id: "agent-move-arcs",
      data: arcs,
      getSourcePosition: function (item) {
        return [item.from[0], item.from[1], 4];
      },
      getTargetPosition: function (item) {
        return [item.to[0], item.to[1], 4];
      },
      getHeight: 0.3,
      getWidth: 2,
      getSourceColor: [57, 215, 255, 160],
      getTargetColor: [169, 139, 255, 200],
      pickable: true,
      onClick: function (info) {
        if (info.object && typeof Sim3D.selectAgent === "function") {
          Sim3D.selectAgent(info.object.id);
        }
      }
    });
  }

  // Yellow -> orange -> red ramp so the spending heatmap stands out against
  // the muted basemap. First stop keeps a faint alpha (not fully transparent)
  // so isolated points still read as a soft glow once kernels stop overlapping.
  const HEATMAP_COLOR_RANGE = [
    [255, 255, 178, 40],
    [255, 237, 160, 150],
    [254, 178, 76, 200],
    [253, 141, 60, 225],
    [240, 59, 32, 245],
    [189, 0, 38, 255]
  ];

  // HeatmapLayer's radiusPixels is a fixed screen-space radius, so at higher
  // zoom the same world-space points drift further apart on screen and their
  // kernels stop overlapping -- the SUM aggregation that made them look hot
  // when zoomed out collapses to a single low weight and fades away. Growing
  // the radius with zoom keeps a lone point's kernel visible.
  function heatmapRadiusForZoom(z) {
    return Math.min(110, Math.max(45, 45 + (z - 12) * 6));
  }

  function makeHeatmapLayer(agents, visible) {
    const state = Sim3D.state || {};
    const heatMode = state.heatMode || "spending";
    const z = zoom();
    return new deck.HeatmapLayer({
      id: "agent-density-heatmap",
      data: agents,
      visible: !!visible,
      getPosition: function (item) {
        return item.position;
      },
      getWeight: function (item) {
        if (heatMode === "spending") {
          return Math.max(0.18, Number((item.frame && item.frame.spent) || 0) / 12000);
        }
        return 1;
      },
      updateTriggers: { getWeight: heatMode },
      colorRange: HEATMAP_COLOR_RANGE,
      radiusPixels: heatmapRadiusForZoom(z),
      intensity: 1.6,
      threshold: 0.01
    });
  }

  Sim3D.makeLayers = function makeLayers() {
    if (!window.deck) return [];
    const z = zoom();
    const toggles = (Sim3D.state && Sim3D.state.layerToggles) || {};
    const agents = typeof Sim3D.getInterpolatedAgents === "function" ? Sim3D.getInterpolatedAgents() : [];
    const layers = [];

    layers.push(makeCityDotLayer(agents, z));

    if (toggles.trails !== false) {
      layers.push(makeTripsLayer());
    } else if (z >= 12) {
      layers.push(makeTripLayer(agents));
    }

    layers.push(makeHeatmapLayer(agents, toggles.heatmap));

    if (toggles.odArcs && z >= 12) {
      layers.push(makeAgentArcLayer());
    }

    if (toggles.policyZones !== false) {
      layers.push.apply(layers, makePolicyZoneLayers());
    }

    return layers;
  };
})();
