(function () {
  const Sim3D = (window.Sim3D = window.Sim3D || {});
  const GOOGLE_KEY_STORAGE = "sim3d.googleMapsApiKey";

  function storedGoogleMapsApiKey() {
    try {
      return window.localStorage.getItem(GOOGLE_KEY_STORAGE) || "";
    } catch (error) {
      return "";
    }
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function isFunction(name) {
    return typeof Sim3D[name] === "function";
  }

  function callOptional(name) {
    if (!isFunction(name)) return undefined;
    return Sim3D[name]();
  }

  function requestFrame(callback) {
    const request = window.requestAnimationFrame || function fallbackRequestFrame(fn) {
      return window.setTimeout(function () {
        fn(window.performance && window.performance.now ? window.performance.now() : Date.now());
      }, 16);
    };
    return request.call(window, callback);
  }

  function cancelFrame(handle) {
    const cancel = window.cancelAnimationFrame || window.clearTimeout;
    if (handle != null) cancel.call(window, handle);
  }

  function refreshOrSetFrame(frameIndex, frameT) {
    const state = Sim3D.state;
    if (isFunction("setFrame")) {
      Sim3D.setFrame(frameIndex, frameT);
      return;
    }

    state.frameIndex = frameIndex;
    state.frameT = frameT;
    if (isFunction("refreshLayers")) {
      Sim3D.refreshLayers();
    }
  }

  Sim3D.state = {
    agents: [],
    timeline: [],
    memories: {},
    events: {},
    meta: {},
    agentById: new Map(),
    frameIndex: 0,
    frameT: 0,
    playing: true,
    speed: 1,
    mode: "story",
    baseMode: "style",
    selectedAgentId: null,
    layerToggles: { heatmap: false, policyZones: true, odArcs: false },
    colorMode: "dist",
    heatMode: "off",
    distFilter: "all",
    showAllMem: false,
    googleMapsApiKey: storedGoogleMapsApiKey(),
    map: null,
    overlay: null,
    charts: {},
    animationHandle: null,
    lastTick: 0
  };

  Sim3D.fail = function fail(error) {
    const message = error && error.stack ? error.stack : error;
    document.body.innerHTML =
      '<main style="min-height:100vh;padding:40px;background:#05070d;color:#edf7ff;font-family:monospace">' +
      '<h1 style="margin:0 0 18px;color:#ff5d7a">3D visualization failed to load</h1>' +
      '<pre style="white-space:pre-wrap;line-height:1.5">' +
      escapeHtml(message) +
      "</pre>" +
      "</main>";
  };

  Sim3D.boot = async function boot() {
    const state = Sim3D.state;
    state.agents = Array.isArray(window.__AGENTS__) ? window.__AGENTS__ : [];
    state.timeline = Array.isArray(window.__TIMELINE__) ? window.__TIMELINE__ : [];
    state.memories = window.__MEMORIES__ || {};
    state.events = window.__EVENTS__ || {};
    state.meta = window.__VIZ_META__ || {};
    state.agentById = new Map(
      state.agents
        .filter(function (agent) {
          return agent && agent.id != null;
        })
        .map(function (agent) {
          return [agent.id, agent];
        })
    );

    if (!state.timeline.length) {
      throw new Error("timeline data is empty");
    }

    const totals = isFunction("computeMemoryTotals")
      ? Sim3D.computeMemoryTotals()
      : { totMem: 0, totAppt: 0, totRumor: 0 };
    const totalAgentsEl = document.getElementById("info-total-agents");
    if (totalAgentsEl) totalAgentsEl.textContent = state.agents.length.toLocaleString();
    const totalMemEl = document.getElementById("info-total-mem");
    if (totalMemEl) totalMemEl.textContent = totals.totMem.toLocaleString();
    const totalApptEl = document.getElementById("info-total-appt");
    if (totalApptEl) totalApptEl.textContent = totals.totAppt.toLocaleString();
    const totalRumorEl = document.getElementById("info-total-rumor");
    if (totalRumorEl) totalRumorEl.textContent = totals.totRumor.toLocaleString();

    const keyInput = document.getElementById("google-key-input");
    if (keyInput) {
      keyInput.value = state.googleMapsApiKey;
    }

    const frameSlider = document.getElementById("frame-slider");
    if (frameSlider) {
      frameSlider.max = String(Math.max(0, state.timeline.length - 1));
      frameSlider.value = "0";
    }

    const detailCard = document.getElementById("detail-card");
    const detailContent = document.getElementById("detail-content");
    if (detailCard && detailContent && !detailContent.textContent.trim()) {
      detailCard.hidden = true;
    }

    callOptional("initDataModel");
    if (isFunction("initMapScene")) {
      await Sim3D.initMapScene();
    }
    callOptional("initHud");
    callOptional("initDetail");
    refreshOrSetFrame(0, 0);
    Sim3D.startRenderLoop();
  };

  Sim3D.startRenderLoop = function startRenderLoop() {
    const state = Sim3D.state;
    if (state.animationHandle != null) {
      cancelFrame(state.animationHandle);
    }
    state.lastTick = 0;

    function tick(now) {
      if (!state.lastTick) {
        state.lastTick = now;
      }
      const delta = Math.min(120, now - state.lastTick);
      state.lastTick = now;

      if (state.playing && state.timeline.length) {
        const speed = Number.isFinite(Number(state.speed)) ? Number(state.speed) : 1;
        const lastFrameIndex = Math.max(0, state.timeline.length - 1);
        const nextT = state.frameT + (delta / 1200) * speed;
        const step = Math.floor(nextT);
        const frameT = nextT - step;
        let frameIndex = state.frameIndex + step;

        if (lastFrameIndex === 0 || frameIndex >= lastFrameIndex) {
          frameIndex = 0;
        }

        refreshOrSetFrame(frameIndex, lastFrameIndex === 0 ? 0 : frameT);
      } else if (isFunction("refreshLayers")) {
        Sim3D.refreshLayers();
      }

      state.animationHandle = requestFrame(tick);
    }

    state.animationHandle = requestFrame(tick);
  };

  document.addEventListener("DOMContentLoaded", function () {
    Sim3D.boot().catch(Sim3D.fail);
  });
})();
