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

  // Classic dense-to-sparse heatmap ramp (like the reference web heatmap):
  // a continuous violet -> blue -> teal -> green -> yellow -> orange -> red
  // gradient with many intermediate stops so density reads as a SMOOTH band of
  // colour, never jumping straight from a pale tint to a red core. Alpha climbs
  // with the ramp: sub-threshold cells are fully transparent, the violet/blue
  // low end is faint, and only the hot core is fully opaque red.
  const HEATMAP_COLOR_RANGE = [
    [55, 30, 110, 0],
    [70, 85, 200, 80],
    [45, 150, 220, 130],
    [40, 200, 188, 170],
    [95, 216, 128, 196],
    [165, 224, 92, 212],
    [236, 226, 80, 226],
    [247, 150, 45, 242],
    [216, 28, 28, 255]
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

  // The screen-space radius shrinks to its floor when zoomed OUT, so kernels
  // barely overlap and the SUM peak is low; zoomed IN the radius grows and
  // overlapping kernels push the peak much higher. The colour ceiling has to
  // follow that: a low ceiling when zoomed out so sparse clusters still reach
  // red, a higher ceiling when zoomed in so dense cores don't smear the whole
  // neighbourhood red. Anchored at z=12.5 (3.6), which reads cleanest as a
  // paper-style blue->red cloud.
  function heatColorMaxForZoom(z) {
    return Math.max(2.4, Math.min(5.2, 3.6 + (z - 12.5) * 0.6));
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
      // Raise the floor (0.75) so the single-point / pass-through cells that
      // dominate the map -- the ones that used to flood it violet -- fall below
      // the ramp and stay nearly transparent, leaving violet only as a thin
      // rim around real clusters. The remaining 0.75..max range then spreads
      // across the cyan/green/yellow mids instead of jumping violet -> red.
      colorDomain: [0.75, heatColorMaxForZoom(z)],
      radiusPixels: heatmapRadiusForZoom(z),
      intensity: 1.05,
      threshold: 0.04
    });
  }

  Sim3D.makeLayers = function makeLayers() {
    if (!window.deck) return [];
    const z = zoom();
    const toggles = (Sim3D.state && Sim3D.state.layerToggles) || {};
    const agents = typeof Sim3D.getInterpolatedAgents === "function" ? Sim3D.getInterpolatedAgents() : [];
    const layers = [];

    layers.push(makeCityDotLayer(agents, z));

    // 잔상(연하늘색 이동 궤적) 토글. 켜져 있을 때만 애니메이션 궤적을 그리고,
    // 끄면 완전히 사라진다. (기존 고배율 fallback인 commute-light-trips는
    // trails 키가 항상 undefined라 실제로는 렌더된 적이 없어 제거했다.)
    if (toggles.trails !== false) {
      layers.push(makeTripsLayer());
    }

    layers.push(makeHeatmapLayer(agents, toggles.heatmap));

    if (toggles.policyZones !== false) {
      layers.push.apply(layers, makePolicyZoneLayers());
    }

    return layers;
  };

  // Exposed so the map-click hotspot handler can size its aggregation circle to
  // the same kernel radius the heatmap renders at the current zoom.
  Sim3D.heatmapRadiusForZoom = heatmapRadiusForZoom;
})();
