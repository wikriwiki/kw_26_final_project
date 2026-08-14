(function () {
  const Sim3D = (window.Sim3D = window.Sim3D || {});

  function byId(id) {
    return document.getElementById(id);
  }

  function bindLayerToggle(buttonId, key, defaultOn) {
    const state = Sim3D.state;
    state.layerToggles = state.layerToggles || {};
    if (state.layerToggles[key] == null) state.layerToggles[key] = defaultOn;

    const button = byId(buttonId);
    if (!button) return;
    button.classList.toggle("active", !!state.layerToggles[key]);
    button.addEventListener("click", function () {
      state.layerToggles[key] = !state.layerToggles[key];
      button.classList.toggle("active", state.layerToggles[key]);
      if (typeof Sim3D.refreshLayers === "function") Sim3D.refreshLayers();
    });
  }

  // The 구 경계 line is a native MapLibre style layer (not a deck.gl layer), so
  // it toggles via setLayoutProperty instead of refreshLayers(). Tracked in
  // state.layerToggles.guBoundary purely so the button's active class survives.
  function bindGuBoundaryToggle(buttonId) {
    const state = Sim3D.state;
    state.layerToggles = state.layerToggles || {};
    if (state.layerToggles.guBoundary == null) state.layerToggles.guBoundary = false;

    const button = byId(buttonId);
    if (!button) return;
    button.classList.toggle("active", !!state.layerToggles.guBoundary);
    button.addEventListener("click", function () {
      state.layerToggles.guBoundary = !state.layerToggles.guBoundary;
      button.classList.toggle("active", state.layerToggles.guBoundary);
      const map = state.map;
      const visibility = state.layerToggles.guBoundary ? "visible" : "none";
      if (map && typeof map.getLayer === "function") {
        // The 구 경계 line and the 자치구 name labels toggle together.
        ["gu-boundary-emphasis", "gu-label-text"].forEach(function (layerId) {
          if (map.getLayer(layerId)) {
            map.setLayoutProperty(layerId, "visibility", visibility);
          }
        });
      }
    });
  }

  function rgbCss(color) {
    return "rgb(" + color[0] + "," + color[1] + "," + color[2] + ")";
  }

  function renderLegendItems() {
    const el = byId("legend-items");
    if (!el) return;
    const mode = Sim3D.state.colorMode || "dist";
    const distNames = Sim3D.DIST_NAMES || {};
    const distColors = Sim3D.DIST_COLORS || {};
    const catColors = Sim3D.CAT_COLORS || {};
    const catOrder = ["식사", "카페", "디저트", "쇼핑", "문화", "생활", "교통", "교육", "의료", "여가", "집", "직장"];

    if (mode === "dist") {
      el.innerHTML = '<div class="legacy-legend-grid">' +
        Object.keys(distColors).map(function (code) {
          return '<div class="legacy-item"><span class="legacy-dot" style="background:' +
            rgbCss(distColors[code]) + '"></span>' + (distNames[code] || code) + "</div>";
        }).join("") + "</div>";
    } else if (mode === "cat") {
      el.innerHTML = catOrder
        .filter(function (key) { return catColors[key]; })
        .map(function (key) {
          return '<div class="legacy-item"><span class="legacy-dot" style="background:' +
            rgbCss(catColors[key]) + '"></span>' + key + "</div>";
        }).join("");
    } else if (mode === "appointment") {
      el.innerHTML =
        '<div class="legacy-item"><span class="legacy-dot" style="background:rgb(157,78,221)"></span>약속 있는 agent</div>' +
        '<div class="legacy-item"><span class="legacy-dot" style="background:#888"></span>약속 없음</div>';
    }
  }

  Sim3D.initLegend = function initLegend() {
    const state = Sim3D.state;
    state.colorMode = state.colorMode || "dist";
    state.heatMode = state.heatMode || "spending";
    state.distFilter = state.distFilter || "all";

    const colorSelect = byId("legend-color-mode");
    if (colorSelect) {
      colorSelect.value = state.colorMode;
      colorSelect.addEventListener("change", function (event) {
        state.colorMode = event.target.value;
        renderLegendItems();
        if (typeof Sim3D.refreshLayers === "function") Sim3D.refreshLayers();
      });
    }

    const heatSelect = byId("legend-heat-mode");
    if (heatSelect) {
      heatSelect.value = state.heatMode;
      heatSelect.addEventListener("change", function (event) {
        state.heatMode = event.target.value;
        state.layerToggles = state.layerToggles || {};
        state.layerToggles.heatmap = state.heatMode !== "off";
        const heatmapButton = byId("toggle-heatmap-btn");
        if (heatmapButton) heatmapButton.classList.toggle("active", state.layerToggles.heatmap);
        if (typeof Sim3D.refreshLayers === "function") Sim3D.refreshLayers();
      });
    }

    const distSelect = byId("legend-dist-filter");
    if (distSelect) {
      distSelect.value = state.distFilter;
      distSelect.addEventListener("change", function (event) {
        state.distFilter = event.target.value;
        if (typeof Sim3D.refreshLayers === "function") Sim3D.refreshLayers();
      });
    }

    renderLegendItems();
  };

  function policyFrameIndex() {
    const timeline = Sim3D.state.timeline || [];
    const idx = timeline.findIndex(function (frame) {
      return frame && frame.day === "2026-05-02" && Number(frame.hour) === 12;
    });
    return idx >= 0 ? idx : Math.min(36, Math.max(0, timeline.length - 1));
  }

  function defaultChapters() {
    const timeline = Sim3D.state.timeline || [];
    const last = Math.max(0, timeline.length - 1);
    const meetup = (Sim3D.state.meta && Sim3D.state.meta.meetups && Sim3D.state.meta.meetups[0]) || null;
    return [
      { label: "오프닝", frame: 0, camera: "opening" },
      { label: "일상 리듬", frame: Math.min(8, last), camera: "rhythm" },
      { label: "정책 발효", frame: policyFrameIndex(), camera: "policy" },
      { label: "상호작용", frame: meetup ? meetup.frame_index : Math.min(20, last), camera: "interaction" },
      { label: "피날레", frame: last, camera: "finale" }
    ];
  }

  function ensureChapters() {
    const rail = byId("chapter-rail");
    if (!rail || rail.querySelector(".chapter")) return;
    defaultChapters().forEach(function (chapter, index) {
      const button = document.createElement("button");
      button.className = "chapter" + (index === 0 ? " active" : "");
      button.type = "button";
      button.dataset.frame = String(chapter.frame);
      button.dataset.camera = chapter.camera;
      button.textContent = chapter.label;
      rail.appendChild(button);
    });
  }

  function setChapterActive(frameIndex) {
    const chapters = Array.from(document.querySelectorAll(".chapter"));
    let best = null;
    chapters.forEach(function (button) {
      const frame = Number(button.dataset.frame || 0);
      if (frame <= frameIndex && (!best || frame >= Number(best.dataset.frame || 0))) {
        best = button;
      }
    });
    chapters.forEach(function (button) {
      button.classList.toggle("active", button === best);
    });
  }

  Sim3D.setFrame = function setFrame(frameIndex, frameT) {
    const state = Sim3D.state;
    const maxFrame = Math.max(0, (state.timeline || []).length - 1);
    state.frameIndex = Math.max(0, Math.min(maxFrame, Number(frameIndex) || 0));
    state.frameT = Math.max(0, Math.min(0.999, Number(frameT) || 0));

    const frame = state.timeline[state.frameIndex];
    const summary = state.meta && state.meta.frame_summaries ? state.meta.frame_summaries[state.frameIndex] : null;
    const label = frame && frame.label ? frame.label : "Frame " + state.frameIndex;

    const frameSlider = byId("frame-slider");
    if (frameSlider) frameSlider.value = String(state.frameIndex);
    const frameLabel = byId("frame-label");
    if (frameLabel) frameLabel.textContent = label;
    const sceneClock = byId("scene-clock");
    if (sceneClock) sceneClock.textContent = label;
    const activeValue = String(summary ? summary.active_agents : (frame && frame.agents ? frame.agents.length : 0));
    const infoFrameLabel = byId("info-frame-label");
    if (infoFrameLabel) infoFrameLabel.textContent = label;
    const infoActive = byId("info-active-cnt");
    if (infoActive) infoActive.textContent = activeValue;

    setChapterActive(state.frameIndex);
    Sim3D.checkStoryNews(frame);
    if (typeof Sim3D.refreshLayers === "function") Sim3D.refreshLayers();
    if (state.selectedAgentId && typeof Sim3D.renderSelectedAgent === "function") {
      Sim3D.renderSelectedAgent();
    }
  };

  Sim3D.showNews = function showNews(message) {
    const banner = byId("news-banner");
    if (!banner) return;
    banner.innerHTML = "";
    const item = document.createElement("div");
    item.className = "news-item";
    item.textContent = message;
    banner.appendChild(item);
    window.clearTimeout(Sim3D.state.newsTimer);
    Sim3D.state.newsTimer = window.setTimeout(function () {
      banner.innerHTML = "";
    }, 5200);
  };

  Sim3D.checkStoryNews = function checkStoryNews(frame) {
    const state = Sim3D.state;
    if (!frame) return;
    state.newsShown = state.newsShown || new Set();
    const key = frame.day + "-" + frame.hour;
    if (frame.day === "2026-05-02" && Number(frame.hour) === 12 && !state.newsShown.has(key)) {
      state.newsShown.add(key);
      Sim3D.showNews("정책 발효: 종로구·중구 외식/카페 지원이 시작되었습니다.");
      if (typeof Sim3D.setCamera === "function" && typeof Sim3D.cameraForChapter === "function") {
        Sim3D.setCamera(Sim3D.cameraForChapter("policy"), { duration: 1200 });
      }
    }
  };

  Sim3D.initHud = function initHud() {
    const state = Sim3D.state;
    ensureChapters();
    Sim3D.initLegend();

    const playButton = byId("play-btn");
    if (playButton) {
      playButton.addEventListener("click", function () {
        state.playing = !state.playing;
        playButton.textContent = state.playing ? "Ⅱ" : "▶";
      });
    }

    const frameSlider = byId("frame-slider");
    if (frameSlider) {
      frameSlider.addEventListener("input", function (event) {
        state.playing = false;
        if (playButton) playButton.textContent = "▶";
        Sim3D.setFrame(Number(event.target.value), 0);
      });
    }

    const speedSelect = byId("speed-select");
    if (speedSelect) {
      speedSelect.addEventListener("change", function (event) {
        state.speed = Number(event.target.value) || 1;
      });
    }

    bindLayerToggle("toggle-heatmap-btn", "heatmap", true);
    bindLayerToggle("toggle-trails-btn", "trails", true);
    bindGuBoundaryToggle("toggle-gu-boundary-btn");

    const chapterToggleButton = byId("chapter-toggle-btn");
    const chapterRail = byId("chapter-rail");
    if (chapterToggleButton && chapterRail) {
      chapterToggleButton.addEventListener("click", function () {
        const expanded = !chapterRail.hidden;
        chapterRail.hidden = expanded;
        chapterToggleButton.textContent = expanded ? "챕터 ▾" : "챕터 ▴";
      });
    }

    document.querySelectorAll(".chapter").forEach(function (button) {
      button.addEventListener("click", function () {
        state.playing = false;
        if (playButton) playButton.textContent = "▶";
        Sim3D.setFrame(Number(button.dataset.frame || 0), 0);
        if (typeof Sim3D.setCamera === "function" && typeof Sim3D.cameraForChapter === "function") {
          Sim3D.setCamera(Sim3D.cameraForChapter(button.dataset.camera || "rhythm"), { duration: 1100 });
        }
      });
    });
  };
})();
