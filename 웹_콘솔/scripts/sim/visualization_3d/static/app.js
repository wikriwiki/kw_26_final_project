(function () {
  const Sim3D = (window.Sim3D = window.Sim3D || {});

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
    pack: null,
    agents: [],
    timeline: [],
    meta: {},
    agentById: new Map(),
    frameIndex: 0,
    frameT: 0,
    playing: true,
    speed: 1,
    selectedAgentId: null,
    layerToggles: { heatmap: true, trails: true, policyZones: true, guBoundary: false },
    colorMode: "dist",
    heatMode: "spending",
    distFilter: "all",
    showAllMem: false,
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

  function now() {
    return window.performance && window.performance.now ? window.performance.now() : Date.now();
  }

  function describeLoad(pack) {
    const days = pack.chunks.length;
    const loadedDays = pack.chunkLoaded.reduce(function (sum, value) {
      return sum + value;
    }, 0);
    return loadedDays >= days ? "준비됨" : "불러오는 중 " + loadedDays + "/" + days + "일";
  }

  /**
   * 첫 화면 위치 — 그 정책 실행에서 가장 활발해 보이는 곳. 원본 콘솔과 같은 거리 수준(줌 15.5)으로 본다.
   * 첫날 시작 시각부터 낮 동안(최대 15시간) 집이 아닌 곳에 있는 사람-시간을 약 500m 격자로 세고,
   * 그 시각에 결제한 사람은 더 무겁게 센다(히트맵과 점이 함께 보이는 곳). 가장 높은 칸의 무게 중심으로 간다.
   */
  function focusMostActiveArea(startIndex) {
    const state = Sim3D.state;
    if (!state.map || !isFunction("agentFrame")) return;
    const agents = (state.pack && state.pack.agents) || state.agents || [];
    const frames = state.timeline || [];
    const day = frames[startIndex] && frames[startIndex].day;
    const loaded = isFunction("loadedFrameCount") ? Sim3D.loadedFrameCount() : frames.length;
    const cells = new Map();
    for (let f = startIndex; f < Math.min(frames.length, loaded, startIndex + 15); f++) {
      if (day && frames[f].day !== day) break;
      for (let a = 0; a < agents.length; a++) {
        const at = Sim3D.agentFrame(a, f);
        if (!at || at.anchor === "residence" || !Number.isFinite(at.lon) || !Number.isFinite(at.lat)) continue;
        const weight = at.spent > 0 ? 3 : 1;
        const key = Math.round(at.lon / 0.006) + ":" + Math.round(at.lat / 0.0045);
        const cell = cells.get(key) || { w: 0, lon: 0, lat: 0 };
        cell.w += weight;
        cell.lon += at.lon * weight;
        cell.lat += at.lat * weight;
        cells.set(key, cell);
      }
    }
    let best = null;
    cells.forEach(function (cell) {
      if (!best || cell.w > best.w) best = cell;
    });
    if (best && best.w > 0) {
      state.map.jumpTo({ center: [best.lon / best.w, best.lat / best.w], zoom: 15.5 });
    }
  }

  Sim3D.boot = async function boot() {
    const state = Sim3D.state;
    const started = now();
    const marks = (Sim3D.bootMarks = { started: started });

    const pack = await Sim3D.openPack();
    marks.manifest = now() - started;
    state.pack = pack;
    state.agents = pack.agents;
    // 예전 코드가 기대하던 프레임 목록(일자·시각·라벨). 대상자 위치는 pack.columns 에 있다.
    state.timeline = pack.frames;
    state.meta = pack.meta || {};
    state.agentById = new Map(
      pack.agents.map(function (agent) {
        return [agent.id, agent];
      })
    );

    if (!state.timeline.length) {
      throw new Error("timeline data is empty");
    }

    callOptional("initDataModel");

    // 타임라인은 하루치씩 받는다. 첫날이 오면 바로 그리고, 나머지는 재생하는 동안 채운다.
    let firstDayReady;
    const firstDay = new Promise(function (resolve) {
      firstDayReady = resolve;
    });
    const timelineDone = pack.loadTimeline(function (loadedPack) {
      if (isFunction("showLoadState")) Sim3D.showLoadState(describeLoad(loadedPack));
      if (loadedPack.loadedFrames > 0) firstDayReady();
    });
    timelineDone.then(
      function () {
        marks.timeline = now() - started;
      },
      function (error) {
        console.error(error);
        if (isFunction("showNews")) Sim3D.showNews("타임라인 일부를 불러오지 못했습니다: " + (error && error.message ? error.message : error));
        if (isFunction("showLoadState")) Sim3D.showLoadState("일부만 불러옴");
      }
    );

    const totals = isFunction("computeMemoryTotals")
      ? Sim3D.computeMemoryTotals()
      : { totMem: 0, totAppt: 0, totRumor: 0 };
    const totalAgentsEl = document.getElementById("info-total-agents");
    if (totalAgentsEl) totalAgentsEl.textContent = state.agents.length.toLocaleString();
    const totalMemEl = document.getElementById("info-total-mem");
    if (totalMemEl) totalMemEl.textContent = totals.totMem.toLocaleString();
    const totalApptEl = document.getElementById("info-total-appt");
    if (totalApptEl) totalApptEl.textContent = totals.totAppt.toLocaleString();
    // 약속(대화) 기록이 없는 데이터에서 0 을 보이면 '약속이 없었다'로 읽힌다. 줄을 숨긴다
    const apptRow = document.getElementById("info-appt-row");
    if (apptRow) apptRow.hidden = !totals.totAppt;
    const totalRumorEl = document.getElementById("info-total-rumor");
    if (totalRumorEl) totalRumorEl.textContent = totals.totRumor.toLocaleString();

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

    // 지도 타일과 타임라인 첫날을 동시에 받는다.
    const mapReady = isFunction("initMapScene") ? Sim3D.initMapScene() : Promise.resolve();
    await Promise.all([
      mapReady.then(function () {
        marks.map = now() - started;
      }),
      Promise.race([firstDay, timelineDone]).then(function () {
        marks.firstDay = now() - started;
      })
    ]);
    if (!pack.loadedFrames) {
      throw new Error("타임라인을 불러오지 못했습니다");
    }

    callOptional("initHud");
    callOptional("initDetail");
    // 자정 프레임은 모두 집에 있어 지도가 비어 보인다. 첫날 아침 7시(사람들이 나서는 때)에서 시작하고,
    // 7시 프레임이 없으면 처음 활동이 있는 시각에서 시작한다.
    const firstActive = Number(pack.manifest.first_active_frame) || 0;
    const seven = (state.timeline || []).findIndex(function (frame) { return Number(frame && frame.hour) === 7; });
    const start = seven >= 0 ? seven : firstActive;
    refreshOrSetFrame(Math.min(start, pack.loadedFrames - 1), 0);
    focusMostActiveArea(Math.min(start, pack.loadedFrames - 1));
    Sim3D.startRenderLoop();
    marks.interactive = now() - started;
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
        const loaded = isFunction("loadedFrameCount") ? Sim3D.loadedFrameCount() : state.timeline.length;
        const lastLoaded = Math.max(0, Math.min(lastFrameIndex, loaded - 1));
        const nextT = state.frameT + (delta / 1200) * speed;
        const step = Math.floor(nextT);
        let frameT = nextT - step;
        let frameIndex = state.frameIndex + step;

        if (lastLoaded < lastFrameIndex && frameIndex >= lastLoaded) {
          // 다음 날 기록이 아직 오는 중이다. 처음으로 되감지 않고 그 자리에서 기다린다.
          frameIndex = lastLoaded;
          frameT = 0;
        } else if (lastFrameIndex === 0 || frameIndex >= lastFrameIndex) {
          frameIndex = 0;
        }

        refreshOrSetFrame(frameIndex, lastFrameIndex === 0 ? 0 : frameT);
      }
      // When paused we do NOT rebuild layers every frame: regenerating the
      // 5k-point heatmap + trail layers on every rAF was the main cost behind
      // sluggish zoom/pan. Map "move" events (bound in initMapScene) refresh
      // the layers while the camera is changing; otherwise nothing changed.

      state.animationHandle = requestFrame(tick);
    }

    state.animationHandle = requestFrame(tick);
  };

  document.addEventListener("DOMContentLoaded", function () {
    Sim3D.boot().catch(Sim3D.fail);
  });
})();
