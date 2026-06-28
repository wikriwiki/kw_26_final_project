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
    const pulsePeriod = 900;
    const pulsePhase = (performance.now() % pulsePeriod) / pulsePeriod;
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
        getSourceColor: [255, 93, 122, 210],
        getTargetColor: [255, 191, 90, 210],
        getWidth: 3,
        pickable: true
      }),
      new deck.ScatterplotLayer({
        id: "meetup-pulses",
        data: meetups,
        radiusUnits: "meters",
        getPosition: function (item) {
          return [item.lon, item.lat, 4];
        },
        getRadius: 18 + pulsePhase * 70,
        getFillColor: [0, 0, 0, 0],
        stroked: true,
        getLineColor: [255, 93, 122, Math.round((1 - pulsePhase) * 220)],
        lineWidthUnits: "pixels",
        getLineWidth: 1.5,
        updateTriggers: { getRadius: pulsePhase, getLineColor: pulsePhase }
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

    const glowPeriod = 2200;
    const glowPhase = (performance.now() % glowPeriod) / glowPeriod;
    const glowAlpha = Math.round((1 - glowPhase) * 200);

    return [
      new deck.ScatterplotLayer({
        id: "policy-zone-markers",
        data: activeZones,
        radiusUnits: "meters",
        getPosition: function (item) {
          return [item.lon, item.lat, 2];
        },
        getRadius: 220,
        getFillColor: [255, 158, 68, 70],
        stroked: true,
        getLineColor: [255, 191, 90, 220],
        lineWidthUnits: "pixels",
        getLineWidth: 2,
        pickable: true,
        onClick: function (info) {
          if (info.object && typeof Sim3D.showNews === "function") {
            Sim3D.showNews(info.object.policy_name + ": " + info.object.dong_name);
          }
        }
      }),
      new deck.ScatterplotLayer({
        id: "policy-zone-glow",
        data: activeZones,
        radiusUnits: "meters",
        getPosition: function (item) {
          return [item.lon, item.lat, 2];
        },
        getRadius: 220 + glowPhase * 140,
        getFillColor: [0, 0, 0, 0],
        stroked: true,
        getLineColor: [255, 191, 90, glowAlpha],
        lineWidthUnits: "pixels",
        getLineWidth: 1.2,
        updateTriggers: { getRadius: glowPhase, getLineColor: glowPhase }
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

  const HEATMAP_COLOR_RANGE = [
    [12, 30, 45, 0],
    [22, 86, 110, 110],
    [57, 215, 255, 160],
    [255, 191, 90, 200],
    [255, 93, 122, 230],
    [255, 93, 122, 255]
  ];

  function makeHeatmapLayer(agents, visible) {
    const state = Sim3D.state || {};
    const heatMode = state.heatMode || "density";
    return new deck.HeatmapLayer({
      id: "agent-density-heatmap",
      data: agents,
      visible: !!visible,
      getPosition: function (item) {
        return item.position;
      },
      getWeight: function (item) {
        if (heatMode === "spending") {
          return Math.max(0.05, Number((item.frame && item.frame.spent) || 0) / 30000);
        }
        return 1;
      },
      updateTriggers: { getWeight: heatMode },
      colorRange: HEATMAP_COLOR_RANGE,
      radiusPixels: 28,
      intensity: 1.4,
      threshold: 0.04
    });
  }

  Sim3D.makeLayers = function makeLayers() {
    if (!window.deck) return [];
    const z = zoom();
    const toggles = (Sim3D.state && Sim3D.state.layerToggles) || {};
    const agents = typeof Sim3D.getInterpolatedAgents === "function" ? Sim3D.getInterpolatedAgents() : [];
    const bursts = typeof Sim3D.getCurrentBursts === "function" ? Sim3D.getCurrentBursts() : [];
    const meetups = typeof Sim3D.getCurrentMeetups === "function" ? Sim3D.getCurrentMeetups() : [];
    const rumorPaths = typeof Sim3D.getCurrentRumors === "function"
      ? Sim3D.getCurrentRumors().map(rumorPath).filter(Boolean)
      : [];
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

    if (z >= 14) {
      layers.push.apply(layers, makeSpendLayers(bursts));
    }

    if (z >= 13) {
      layers.push.apply(layers, makeInteractionLayers(meetups, rumorPaths));
    }

    return layers;
  };
})();
