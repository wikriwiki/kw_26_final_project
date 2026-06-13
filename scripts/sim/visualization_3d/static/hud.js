(function () {
  const Sim3D = (window.Sim3D = window.Sim3D || {});

  function byId(id) {
    return document.getElementById(id);
  }

  function setActiveButton(activeId, ids) {
    ids.forEach(function (id) {
      const element = byId(id);
      if (element) element.classList.toggle("active", id === activeId);
    });
  }

  function chartAvailable() {
    return typeof window.Chart === "function";
  }

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

  function createLineChart(canvas, labels, datasets) {
    if (!canvas || !chartAvailable()) return null;
    return new Chart(canvas, {
      type: "line",
      data: { labels: labels, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: {
            labels: { color: "#edf7ff", boxWidth: 9, font: { size: 10 } }
          },
          tooltip: { mode: "index", intersect: false }
        },
        scales: {
          x: {
            ticks: { color: "#9fb2c5", maxTicksLimit: 5, font: { size: 10 } },
            grid: { color: "rgba(255,255,255,0.05)" }
          },
          y: {
            ticks: {
              color: "#9fb2c5",
              font: { size: 10 },
              callback: function (value) {
                return Math.floor(Number(value) / 10000) + "만";
              }
            },
            grid: { color: "rgba(255,255,255,0.05)" }
          }
        },
        interaction: { mode: "index", intersect: false }
      }
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
    const active = byId("active-cnt");
    if (active) active.textContent = String(summary ? summary.active_agents : (frame && frame.agents ? frame.agents.length : 0));
    const spend = byId("spend-total");
    if (spend) spend.textContent = Sim3D.formatWon ? Sim3D.formatWon(summary ? summary.spend_total : 0) : String(summary ? summary.spend_total : 0);
    const bursts = byId("burst-cnt");
    if (bursts) bursts.textContent = String(typeof Sim3D.getCurrentBursts === "function" ? Sim3D.getCurrentBursts().length : 0);

    setChapterActive(state.frameIndex);
    Sim3D.updateCharts();
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

  Sim3D.initCharts = function initCharts() {
    const state = Sim3D.state;
    const series = state.macroSeries || { labels: [], targetCats: [], catData: [], distData: [] };
    state.charts = state.charts || {};
    if (state.charts.cat || state.charts.dist) return;
    state.charts.cat = createLineChart(byId("chart-cat"), series.labels || [], (series.targetCats || []).map(function (label, index) {
      return {
        label: label,
        data: series.catData[index] || [],
        borderColor: ["#e76f51", "#f4a261", "#f4d35e"][index] || "#39d7ff",
        pointRadius: 0,
        tension: 0.22
      };
    }));
    state.charts.dist = createLineChart(byId("chart-dist"), series.labels || [], ["종로구", "중구"].map(function (label, index) {
      return {
        label: label,
        data: series.distData[index] || [],
        borderColor: ["#39d7ff", "#a98bff"][index] || "#39d7ff",
        pointRadius: 0,
        tension: 0.22
      };
    }));
  };

  Sim3D.updateCharts = function updateCharts() {
    const charts = Sim3D.state.charts || {};
    Object.keys(charts).forEach(function (key) {
      if (charts[key] && typeof charts[key].update === "function") {
        charts[key].update("none");
      }
    });
  };

  Sim3D.initHud = function initHud() {
    const state = Sim3D.state;
    ensureChapters();
    Sim3D.initCharts();

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

    const story = byId("story-mode-btn");
    if (story) {
      story.addEventListener("click", function () {
        state.mode = "story";
        setActiveButton("story-mode-btn", ["story-mode-btn", "explore-mode-btn"]);
        if (typeof Sim3D.setCamera === "function" && typeof Sim3D.cameraForChapter === "function") {
          Sim3D.setCamera(Sim3D.cameraForChapter("opening"), { duration: 1000 });
        }
      });
    }

    const explore = byId("explore-mode-btn");
    if (explore) {
      explore.addEventListener("click", function () {
        state.mode = "explore";
        setActiveButton("explore-mode-btn", ["story-mode-btn", "explore-mode-btn"]);
      });
    }

    const styleButton = byId("base-style-btn");
    if (styleButton) {
      styleButton.addEventListener("click", function () {
        if (typeof Sim3D.switchBaseMode === "function") Sim3D.switchBaseMode("style");
        setActiveButton("base-style-btn", ["base-style-btn", "base-google-btn"]);
      });
    }

    const googleButton = byId("base-google-btn");
    if (googleButton) {
      googleButton.addEventListener("click", function () {
        if (typeof Sim3D.switchBaseMode === "function") Sim3D.switchBaseMode("google");
        setActiveButton(Sim3D.state.baseMode === "google" ? "base-google-btn" : "base-style-btn", ["base-style-btn", "base-google-btn"]);
      });
    }

    const keyInput = byId("google-key-input");
    if (keyInput) {
      keyInput.addEventListener("change", function (event) {
        state.googleMapsApiKey = event.target.value.trim();
        try {
          window.localStorage.setItem("sim3d.googleMapsApiKey", state.googleMapsApiKey);
        } catch (error) {
          console.warn("Could not persist Google Maps key", error);
        }
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
