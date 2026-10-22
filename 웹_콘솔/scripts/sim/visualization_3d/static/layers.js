(function () {
  const Sim3D = (window.Sim3D = window.Sim3D || {});

  // 레이어 객체는 매 프레임 새로 만들어도 싸다(deck.gl 이 props 를 얕게 비교한다).
  // 비싼 것은 data 가 바뀌어 속성을 다시 계산·업로드하는 일이다. 그래서
  //   · 점 위치만 매 프레임 새 버퍼 래퍼로 넘기고
  //   · 색·반지름·히트맵·궤적은 시(hour)·필터·모드가 바뀔 때만 새 data 를 넘기며
  //   · 줌에 따라 달라지는 값은 카메라가 멈췄을 때(settledZoom)만 바꾼다.

  function liveZoom() {
    const state = Sim3D.state || {};
    return state.map && typeof state.map.getZoom === "function" ? state.map.getZoom() : 11;
  }

  function settledZoom() {
    const state = Sim3D.state || {};
    return Number.isFinite(state.settledZoom) ? state.settledZoom : liveZoom();
  }

  const DOT_LINE_COLOR = [230, 248, 255, 90];
  const TRAIL_COLOR = [57, 215, 255];
  const ZONE_FILL = [0, 0, 0, 0];
  const ZONE_LINE = [255, 191, 90, 150];
  const ROUTE_COLOR = [255, 196, 64, 110];
  const ROUTE_PROGRESS_COLOR = [255, 176, 32, 235];
  const ROUTE_STOP_COLOR = [255, 214, 120, 230];
  const HEAD_FILL = [255, 255, 255, 40];
  const HEAD_LINE = [255, 176, 32, 255];

  function onDotClick(info) {
    if (!info || !(info.index >= 0) || typeof Sim3D.selectAgent !== "function") return;
    const agentId = Sim3D.agentIdAtVisible(info.index);
    if (agentId != null) Sim3D.selectAgent(agentId);
  }

  function makeCityDotLayer(z) {
    return new deck.ScatterplotLayer({
      id: "agent-city-dots",
      data: Sim3D.dotsData(),
      pickable: true,
      opacity: z < 12 ? 0.9 : 0.35,
      radiusUnits: "meters",
      // 반지름은 미터라 서울 전체를 볼 때는 1픽셀도 안 된다. 최소 크기를 둔다
      radiusMinPixels: 2,
      stroked: true,
      getLineColor: DOT_LINE_COLOR,
      lineWidthUnits: "pixels",
      getLineWidth: 1,
      onClick: onDotClick
    });
  }

  function currentDay() {
    const frame = typeof Sim3D.frameInfo === "function" ? Sim3D.frameInfo((Sim3D.state || {}).frameIndex) : null;
    return frame ? String(frame.day || "") : "";
  }

  const zoneCache = { day: null, source: null, data: [] };

  function zonePosition(item) {
    return [item.lon, item.lat, 2];
  }

  function onZoneClick(info) {
    if (info.object && typeof Sim3D.showNews === "function") {
      Sim3D.showNews(info.object.policy_name + ": " + info.object.dong_name);
    }
  }

  function makePolicyZoneLayers() {
    const state = Sim3D.state || {};
    const zones = (state.meta && state.meta.policy_zones) || [];
    const day = currentDay();
    if (zoneCache.day !== day || zoneCache.source !== zones) {
      zoneCache.day = day;
      zoneCache.source = zones;
      zoneCache.data = zones.filter(function (zone) {
        if (!day) return true;
        if (zone.effective_from && day < zone.effective_from) return false;
        if (zone.effective_until && day > zone.effective_until) return false;
        return true;
      });
    }

    // Thin translucent amber outline (no fill) so policy zones read as
    // boundaries without obscuring the spending heatmap underneath.
    return [
      new deck.ScatterplotLayer({
        id: "policy-zone-markers",
        data: zoneCache.data,
        radiusUnits: "meters",
        getPosition: zonePosition,
        getRadius: 220,
        getFillColor: ZONE_FILL,
        stroked: true,
        getLineColor: ZONE_LINE,
        lineWidthUnits: "pixels",
        getLineWidth: 1,
        pickable: true,
        onClick: onZoneClick
      })
    ];
  }

  function makeTripsLayer(zq, t) {
    const windowSize = zq >= 12 ? 8 : 4;
    const trails = Sim3D.trailsData(windowSize);
    return new deck.TripsLayer({
      id: "agent-move-trails",
      data: trails.data,
      _pathType: "open",
      trailLength: windowSize,
      currentTime: trails.baseTime + t,
      // 꼬리 끝이 움직이는 점에 붙으면서 한 칸만큼 더 진해졌다. 예전 평균 밝기에 맞춘다.
      opacity: 0.45,
      widthMinPixels: 2,
      getColor: TRAIL_COLOR,
      capRounded: true,
      jointRounded: true
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

  const colorDomains = new Map();

  function colorDomainForZoom(z) {
    let domain = colorDomains.get(z);
    if (!domain) {
      // Raise the floor (0.75) so the single-point / pass-through cells that
      // dominate the map -- the ones that used to flood it violet -- fall below
      // the ramp and stay nearly transparent, leaving violet only as a thin
      // rim around real clusters.
      domain = [0.75, heatColorMaxForZoom(z)];
      colorDomains.set(z, domain);
    }
    return domain;
  }

  // 가중치 텍스처 한 변의 크기. 기본값 2048 이면 갱신할 때마다 최대값을 찾느라 4M 개의 점을
  // 그린다. 커널 반경이 58px 이상이라 1024 로 줄여도 화면 차이는 보이지 않는다.
  const HEAT_TEXTURE_SIZE = 1024;

  let hiddenHeatData = null;

  function heatmapLayerAt(id, data, opacity, zq) {
    return new deck.HeatmapLayer({
      id: id,
      data: data,
      opacity: opacity,
      aggregation: "SUM",
      colorRange: HEATMAP_COLOR_RANGE,
      colorDomain: colorDomainForZoom(zq),
      radiusPixels: heatmapRadiusForZoom(zq),
      intensity: 1.05,
      threshold: 0.04,
      weightsTextureSize: HEAT_TEXTURE_SIZE
    });
  }

  /**
   * 열지도 교차 페이드 — 지금 시각과 다음 시각을 겹쳐 그리고, 시간대 안에서 나아간 만큼
   * 다음 쪽으로 넘긴다. 예전에는 1시간(재생 1.2초)마다 열지도가 통째로 바뀌어 화면이 툭툭
   * 끊겨 보였다. 레이어 id 에 시각을 넣어 두면 경계를 넘을 때 '다음' 레이어가 그대로 '지금'이
   * 되어 텍스처를 다시 그리지 않는다. 새로 계산하는 것은 시간대마다 한 장뿐이다.
   */
  function makeHeatmapCrossfade(visible, zq, t) {
    const state = Sim3D.state || {};
    if (!visible || typeof Sim3D.heatDataAt !== "function") return [makeHeatmapLayer(visible, zq)];
    const index = state.frameIndex || 0;
    const mode = state.heatMode || "spending";
    const total = (state.timeline && state.timeline.length) || 0;
    const loaded = typeof Sim3D.loadedFrameCount === "function" ? Sim3D.loadedFrameCount() : total;
    const layers = [heatmapLayerAt("agent-heat-" + index, Sim3D.heatDataAt(index, mode), 1 - t, zq)];
    if (index + 1 < Math.min(total, loaded)) {
      layers.push(heatmapLayerAt("agent-heat-" + (index + 1), Sim3D.heatDataAt(index + 1, mode), t, zq));
    }
    return layers;
  }

  function makeHeatmapLayer(visible, zq) {
    // 꺼져 있을 때는 마지막 data 를 그대로 둔다 — 보이지 않는 텍스처를 다시 그리지 않는다.
    const data = visible || !hiddenHeatData ? Sim3D.heatData() : hiddenHeatData;
    hiddenHeatData = data;
    return new deck.HeatmapLayer({
      id: "agent-density-heatmap",
      data: data,
      visible: !!visible,
      aggregation: "SUM",
      colorRange: HEATMAP_COLOR_RANGE,
      // Pin the colour scale (instead of HeatmapLayer's default per-frame
      // re-normalisation to the on-screen max, which made the heatmap fade out
      // on zoom-in) but float the ceiling with zoom so the same cluster keeps a
      // consistent hot/cool reading across zoom levels.
      colorDomain: colorDomainForZoom(zq),
      radiusPixels: heatmapRadiusForZoom(zq),
      intensity: 1.05,
      threshold: 0.04,
      weightsTextureSize: HEAT_TEXTURE_SIZE
    });
  }

  function routePath(item) {
    return item.path;
  }

  function stopPosition(item) {
    return item;
  }

  function makeSelectionLayers() {
    const selection = typeof Sim3D.selectionData === "function" ? Sim3D.selectionData() : null;
    if (!selection) return [];
    return [
      new deck.PathLayer({
        id: "selected-agent-route",
        data: selection.route,
        getPath: routePath,
        getColor: ROUTE_COLOR,
        widthUnits: "pixels",
        getWidth: 3,
        capRounded: true,
        jointRounded: true
      }),
      new deck.ScatterplotLayer({
        id: "selected-agent-stops",
        data: selection.stops,
        getPosition: stopPosition,
        radiusUnits: "pixels",
        getRadius: 3.5,
        getFillColor: ROUTE_STOP_COLOR
      }),
      new deck.PathLayer({
        id: "selected-agent-progress",
        data: selection.progress,
        getPath: routePath,
        getColor: ROUTE_PROGRESS_COLOR,
        widthUnits: "pixels",
        getWidth: 4,
        capRounded: true,
        jointRounded: true
      }),
      new deck.ScatterplotLayer({
        id: "selected-agent-head",
        data: { length: 1, attributes: { getPosition: { value: new Float64Array([selection.head[0], selection.head[1], 14]), size: 3 } } },
        radiusUnits: "pixels",
        getRadius: 11,
        getFillColor: HEAD_FILL,
        stroked: true,
        getLineColor: HEAD_LINE,
        lineWidthUnits: "pixels",
        getLineWidth: 2.5
      })
    ];
  }

  Sim3D.makeLayers = function makeLayers() {
    if (!window.deck) return [];
    const state = Sim3D.state || {};
    const toggles = state.layerToggles || {};
    const t = Math.max(0, Math.min(1, Number(state.frameT) || 0));
    const zq = settledZoom();
    const layers = [];

    layers.push(makeCityDotLayer(liveZoom()));

    // 잔상(연하늘색 이동 궤적) 토글. 켜져 있을 때만 애니메이션 궤적을 그리고,
    // 끄면 완전히 사라진다.
    if (toggles.trails !== false) {
      layers.push(makeTripsLayer(zq, t));
    }

    layers.push.apply(layers, makeHeatmapCrossfade(toggles.heatmap, zq, t));

    if (toggles.policyZones !== false) {
      layers.push.apply(layers, makePolicyZoneLayers());
    }

    layers.push.apply(layers, makeSelectionLayers());
    return layers;
  };

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // 점에 마우스를 올리면 그 사람이 지금 무엇을 하는지 바로 보인다.
  Sim3D.tooltipFor = function tooltipFor(info) {
    if (!info || !info.layer || info.layer.id !== "agent-city-dots" || !(info.index >= 0)) return null;
    const agentIndex = Sim3D.agentIndexAtVisible(info.index);
    const agent = Sim3D.state && Sim3D.state.pack ? Sim3D.state.pack.agents[agentIndex] : null;
    if (!agent) return null;
    const frame = Sim3D.agentFrame(agentIndex);
    const district = (Sim3D.DIST_NAMES || {})[String(agent.dist_code)] || agent.district || "";
    let doing = "집에 있음";
    if (frame) {
      const what = [frame.l1 || frame.cat, frame.intent].filter(Boolean).join(" · ");
      doing = (what || "활동 중") + (frame.dong ? " · " + frame.dong : "");
    }
    const spent = frame && frame.spent > 0 ? "<div>결제 " + esc(Sim3D.formatWon(frame.spent)) + "</div>" : "";
    const sex = agent.gender === "M" ? "남" : agent.gender === "F" ? "여" : agent.gender;
    return {
      html:
        '<div class="sim-tip"><b>' + esc(agent.id) + "</b>" +
        "<div>" + esc([agent.age, sex, district].filter(Boolean).join(" · ")) + "</div>" +
        "<div>" + esc(doing) + "</div>" + spent + "</div>",
      className: "sim-tip-wrap"
    };
  };

  // Exposed so the map-click hotspot handler can size its aggregation circle to
  // the same kernel radius the heatmap renders at the current zoom.
  Sim3D.heatmapRadiusForZoom = heatmapRadiusForZoom;
})();
