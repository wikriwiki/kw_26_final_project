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

  // Classic dense-to-sparse heatmap ramp (like the reference web heatmap):
  // a continuous violet -> blue -> teal -> green -> yellow -> orange -> red
  // gradient with many intermediate stops so density reads as a SMOOTH band of
  // colour, never jumping straight from a pale tint to a red core. Alpha climbs
  // with the ramp: sub-threshold cells are fully transparent, the violet/blue
  // low end is faint, and only the hot core is fully opaque red.
  const HEATMAP_COLOR_RANGE = [
    [50, 20, 80, 0],
    [92, 55, 160, 70],
    [60, 80, 200, 110],
    [40, 150, 215, 150],
    [40, 200, 188, 182],
    [120, 218, 118, 206],
    [228, 222, 85, 226],
    [246, 145, 45, 242],
    [218, 28, 28, 255]
  ];

  // HeatmapLayer's radiusPixels is a fixed screen-space radius, so at higher
  // zoom the same world-space points drift further apart on screen and their
  // kernels stop overlapping -- the SUM aggregation that made them look hot
  // when zoomed out collapses to a single low weight and fades away. Growing
  // the radius aggressively with zoom keeps lone points' kernels overlapping,
  // and a wider radius also spreads the colour ramp into a broader, smoother
  // gradient band instead of a tight red dot.
  function heatmapRadiusForZoom(z) {
    return Math.min(175, Math.max(58, 58 + (z - 11) * 11));
  }

  // The screen-space radius shrinks to its 55px floor when zoomed OUT, so
  // kernels barely overlap and the SUM peak is low; zoomed IN the radius grows
  // to 170px and overlapping kernels push the peak much higher. The colour
  // ceiling has to follow that: a low ceiling when zoomed out so sparse
  // clusters still reach red, a higher ceiling when zoomed in so dense cores
  // don't smear the whole neighbourhood red. Anchored at z=12.5 (~4.25), which
  // reads cleanest as a paper-style blue->red cloud.
  function heatColorMaxForZoom(z) {
    return Math.max(2.8, Math.min(6, 4.2 + (z - 12.5) * 0.7));
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
          return Math.max(0.22, Number((item.frame && item.frame.spent) || 0) / 9000);
        }
        return 1;
      },
      updateTriggers: { getWeight: heatMode },
      aggregation: "SUM",
      colorRange: HEATMAP_COLOR_RANGE,
      // Pin the colour scale (instead of HeatmapLayer's default per-frame
      // re-normalisation to the on-screen max, which made the heatmap fade out
      // on zoom-in) but float the ceiling with zoom so the same cluster keeps a
      // consistent hot/cool reading across zoom levels.
      colorDomain: [0.3, heatColorMaxForZoom(z)],
      radiusPixels: heatmapRadiusForZoom(z),
      intensity: 1.05,
      threshold: 0.055
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
