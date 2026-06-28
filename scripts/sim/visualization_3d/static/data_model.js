(function () {
  const Sim3D = (window.Sim3D = window.Sim3D || {});
  const SEOUL_CENTER = [126.978, 37.566];

  const DIST_NAMES = {
    "11110": "종로구",
    "11140": "중구",
    "11170": "용산구",
    "11200": "성동구",
    "11215": "광진구",
    "11230": "동대문구",
    "11260": "중랑구",
    "11290": "성북구",
    "11305": "강북구",
    "11320": "도봉구",
    "11350": "노원구",
    "11380": "은평구",
    "11410": "서대문구",
    "11440": "마포구",
    "11470": "양천구",
    "11500": "강서구",
    "11530": "구로구",
    "11545": "금천구",
    "11560": "영등포구",
    "11590": "동작구",
    "11620": "관악구",
    "11650": "서초구",
    "11680": "강남구",
    "11710": "송파구",
    "11740": "강동구"
  };

  const CAT_COLORS = {
    "식사": [255, 120, 84],
    "카페": [72, 202, 228],
    "디저트": [255, 198, 109],
    "쇼핑": [197, 148, 255],
    "문화": [126, 217, 87],
    "생활": [255, 148, 194],
    "교통": [122, 162, 247],
    "교육": [104, 211, 145],
    "의료": [255, 93, 122],
    "여가": [82, 197, 160],
    "집": [166, 185, 205],
    "직장": [113, 146, 255],
    "Cafe": [72, 202, 228],
    "Food": [255, 120, 84],
    "Dessert": [255, 198, 109],
    "기타": [154, 169, 190]
  };

  const DIST_COLORS = {
    "11110": [250, 204, 21],
    "11140": [244, 114, 182],
    "11170": [56, 189, 248],
    "11200": [34, 197, 94],
    "11215": [129, 140, 248],
    "11230": [251, 146, 60],
    "11260": [45, 212, 191],
    "11290": [248, 113, 113],
    "11305": [163, 230, 53],
    "11320": [192, 132, 252],
    "11350": [125, 211, 252],
    "11380": [253, 186, 116],
    "11410": [134, 239, 172],
    "11440": [251, 113, 133],
    "11470": [147, 197, 253],
    "11500": [252, 211, 77],
    "11530": [74, 222, 128],
    "11545": [216, 180, 254],
    "11560": [103, 232, 249],
    "11590": [253, 164, 175],
    "11620": [190, 242, 100],
    "11650": [196, 181, 253],
    "11680": [96, 165, 250],
    "11710": [52, 211, 153],
    "11740": [251, 191, 36]
  };

  function frameMap(frame) {
    const map = new Map();
    const agents = frame && Array.isArray(frame.agents) ? frame.agents : [];
    agents.forEach(function (agentFrame) {
      if (agentFrame && agentFrame.id != null) {
        map.set(agentFrame.id, agentFrame);
        if (typeof agentFrame.id !== "string") {
          map.set(String(agentFrame.id), agentFrame);
        }
      }
    });
    return map;
  }

  function clamp01(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return 0;
    return Math.max(0, Math.min(1, number));
  }

  function lerp(start, end, t) {
    const from = finiteNumber(start, 0);
    const to = finiteNumber(end, from);
    return from + (to - from) * clamp01(t);
  }

  function formatWon(value) {
    return Math.round(Math.max(0, finiteNumber(value, 0))).toLocaleString("ko-KR") + "원";
  }

  function finiteNumber(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function intOrZero(value) {
    return Math.trunc(finiteNumber(value, 0));
  }

  function getState() {
    Sim3D.state = Sim3D.state || {};
    return Sim3D.state;
  }

  function timelineFrames(state) {
    return Array.isArray(state.timeline) ? state.timeline : [];
  }

  function ensureFrameMaps(state) {
    const timeline = timelineFrames(state);
    if (!Array.isArray(state.frameMaps) || state.frameMaps.length !== timeline.length) {
      state.frameMaps = timeline.map(frameMap);
    }
    return state.frameMaps;
  }

  function safeFrameIndex(state) {
    const timeline = timelineFrames(state);
    if (!timeline.length) return 0;
    const index = intOrZero(state.frameIndex);
    return Math.max(0, Math.min(timeline.length - 1, index));
  }

  function getFrameFromMap(map, id) {
    if (!(map instanceof Map) || id == null) return null;
    if (map.has(id)) return map.get(id);
    const stringId = String(id);
    return map.has(stringId) ? map.get(stringId) : null;
  }

  function agentLookup(state) {
    if (state.agentById instanceof Map) return state.agentById;

    const map = new Map();
    if (state.agentById && typeof state.agentById === "object") {
      Object.keys(state.agentById).forEach(function (id) {
        const agent = state.agentById[id];
        if (agent) {
          map.set(agent.id != null ? agent.id : id, agent);
          map.set(id, agent);
        }
      });
    }

    const agents = Array.isArray(state.agents) ? state.agents : [];
    agents.forEach(function (agent) {
      if (agent && agent.id != null) {
        map.set(agent.id, agent);
        if (typeof agent.id !== "string") {
          map.set(String(agent.id), agent);
        }
      }
    });
    return map;
  }

  function getAgentFromLookup(lookup, id) {
    if (!(lookup instanceof Map) || id == null) return null;
    if (lookup.has(id)) return lookup.get(id);
    const stringId = String(id);
    return lookup.has(stringId) ? lookup.get(stringId) : null;
  }

  function collectAgents(state, currentMap, nextMap) {
    const agents = [];
    const seen = new Set();
    const lookup = agentLookup(state);

    function add(agent, id) {
      const agentId = agent && agent.id != null ? agent.id : id;
      const seenKey = agentId == null ? null : String(agentId);
      if (seenKey == null || seen.has(seenKey)) return;
      seen.add(seenKey);
      agents.push(agent || { id: agentId });
    }

    if (Array.isArray(state.agents)) {
      state.agents.forEach(function (agent) {
        add(agent, agent && agent.id);
      });
    }

    if (lookup instanceof Map) {
      lookup.forEach(function (agent, id) {
        add(agent, id);
      });
    }

    [currentMap, nextMap].forEach(function (map) {
      if (!(map instanceof Map)) return;
      map.forEach(function (agentFrame, id) {
        add(getAgentFromLookup(lookup, id), id);
      });
    });

    return agents;
  }

  function framePosition(agent, agentFrame) {
    const homeLon = finiteNumber(agent && agent.home_lon, SEOUL_CENTER[0]);
    const homeLat = finiteNumber(agent && agent.home_lat, SEOUL_CENTER[1]);
    return [
      finiteNumber(agentFrame && agentFrame.lon, homeLon),
      finiteNumber(agentFrame && agentFrame.lat, homeLat)
    ];
  }

  Sim3D.initDataModel = function initDataModel() {
    const state = getState();
    state.frameMaps = timelineFrames(state).map(frameMap);
  };

  Sim3D.hasAppointment = function hasAppointment(agentId) {
    const state = getState();
    const mem = state.memories && state.memories[agentId];
    return !!(mem && Array.isArray(mem.appointments) && mem.appointments.length > 0);
  };

  Sim3D.computeMemoryTotals = function computeMemoryTotals() {
    const state = getState();
    const memories = state.memories || {};
    let totMem = 0;
    let totAppt = 0;
    let totRumor = 0;
    Object.keys(memories).forEach(function (agentId) {
      const mem = memories[agentId] || {};
      totMem += (mem.visited || []).length;
      totAppt += (mem.appointments || []).length;
      totRumor += (mem.memories || []).filter(function (item) {
        return item.type === "rumor";
      }).length;
    });
    return { totMem: totMem, totAppt: totAppt, totRumor: totRumor };
  };

  Sim3D.getColorForAgentFrame = function getColorForAgentFrame(agent, agentFrame) {
    const state = getState();
    const mode = state.colorMode || "cat";
    const distCode = agent && agent.dist_code != null ? String(agent.dist_code) : "";

    if (mode === "appointment") {
      const hasAppt = Sim3D.hasAppointment(agent && agent.id);
      return hasAppt ? [157, 78, 221, 220] : [136, 136, 136, 150];
    }

    if (!agentFrame) {
      if (mode === "dist") {
        const base = DIST_COLORS[distCode] || [136, 136, 136];
        return [base[0], base[1], base[2], 70];
      }
      return [98, 115, 134, 70];
    }

    const alpha = finiteNumber(agentFrame.spent, 0) > 0 ? 245 : 185;
    if (mode === "dist") {
      const base = DIST_COLORS[distCode] || CAT_COLORS["기타"];
      return [base[0], base[1], base[2], alpha];
    }

    const category = agentFrame.l1 || agentFrame.cat;
    const baseColor = CAT_COLORS[category] || DIST_COLORS[distCode] || CAT_COLORS["기타"];
    return [baseColor[0], baseColor[1], baseColor[2], alpha];
  };

  Sim3D.getInterpolatedAgents = function getInterpolatedAgents() {
    const state = getState();
    const timeline = timelineFrames(state);
    if (!timeline.length) return [];

    const frameMaps = ensureFrameMaps(state);
    const index = safeFrameIndex(state);
    const nextIndex = Math.min(index + 1, timeline.length - 1);
    const currentMap = frameMaps[index] instanceof Map ? frameMaps[index] : frameMap(timeline[index]);
    const nextMap = frameMaps[nextIndex] instanceof Map ? frameMaps[nextIndex] : frameMap(timeline[nextIndex]);
    const t = clamp01(state.frameT);

    const interpolated = collectAgents(state, currentMap, nextMap).map(function (agent) {
      const id = agent.id;
      const currentFrame = getFrameFromMap(currentMap, id);
      const nextFrame = getFrameFromMap(nextMap, id);
      const currentPosition = framePosition(agent, currentFrame);
      const nextPosition = framePosition(agent, nextFrame || currentFrame);
      const frame = currentFrame || nextFrame || null;

      return {
        id: id,
        agent: agent,
        frame: frame,
        position: [
          lerp(currentPosition[0], nextPosition[0], t),
          lerp(currentPosition[1], nextPosition[1], t),
          10
        ],
        color: Sim3D.getColorForAgentFrame(agent, frame)
      };
    });

    const distFilter = state.distFilter || "all";
    if (distFilter === "all") return interpolated;
    return interpolated.filter(function (item) {
      return item.agent && String(item.agent.dist_code) === distFilter;
    });
  };

  let lastTrailFrameIndex = -1;
  let lastTrailWindowSize = -1;
  let cachedTrails = [];

  Sim3D.getAgentTrails = function getAgentTrails(windowSize) {
    const state = getState();
    const timeline = timelineFrames(state);
    if (!timeline.length) return [];

    const size = Math.max(2, intOrZero(windowSize) || 6);
    const index = safeFrameIndex(state);
    if (index === lastTrailFrameIndex && size === lastTrailWindowSize) {
      return cachedTrails;
    }

    const frameMaps = ensureFrameMaps(state);
    const start = Math.max(0, index - size + 1);
    const byAgent = new Map();

    for (let frameIndex = start; frameIndex <= index; frameIndex++) {
      const map = frameMaps[frameIndex];
      if (!(map instanceof Map)) continue;
      map.forEach(function (agentFrame, id) {
        if (agentFrame.lon == null || agentFrame.lat == null) return;
        const key = String(id);
        if (!byAgent.has(key)) byAgent.set(key, { id: id, path: [], timestamps: [] });
        const entry = byAgent.get(key);
        entry.path.push([Number(agentFrame.lon), Number(agentFrame.lat), 8]);
        entry.timestamps.push(frameIndex - start);
      });
    }

    cachedTrails = Array.from(byAgent.values()).filter(function (entry) {
      return entry.path.length > 1;
    });
    lastTrailFrameIndex = index;
    lastTrailWindowSize = size;
    return cachedTrails;
  };

  let lastArcFrameIndex = -1;
  let cachedArcs = [];

  Sim3D.getAgentMoveArcs = function getAgentMoveArcs() {
    const state = getState();
    const timeline = timelineFrames(state);
    if (!timeline.length) return [];

    const index = safeFrameIndex(state);
    if (index === lastArcFrameIndex) return cachedArcs;

    const frameMaps = ensureFrameMaps(state);
    const prevMap = frameMaps[Math.max(0, index - 1)];
    const currentMap = frameMaps[index];
    const arcs = [];

    if (currentMap instanceof Map && prevMap instanceof Map && prevMap !== currentMap) {
      currentMap.forEach(function (agentFrame, id) {
        const prevFrame = getFrameFromMap(prevMap, id);
        if (!prevFrame || agentFrame.lon == null || agentFrame.lat == null) return;
        if (prevFrame.lon == null || prevFrame.lat == null) return;
        const dLon = Number(agentFrame.lon) - Number(prevFrame.lon);
        const dLat = Number(agentFrame.lat) - Number(prevFrame.lat);
        if (Math.abs(dLon) < 0.0003 && Math.abs(dLat) < 0.0003) return;
        arcs.push({
          id: id,
          from: [Number(prevFrame.lon), Number(prevFrame.lat)],
          to: [Number(agentFrame.lon), Number(agentFrame.lat)]
        });
      });
    }

    cachedArcs = arcs;
    lastArcFrameIndex = index;
    return cachedArcs;
  };

  Sim3D.formatWon = formatWon;
  Sim3D.DIST_NAMES = DIST_NAMES;
  Sim3D.CAT_COLORS = CAT_COLORS;
  Sim3D.DIST_COLORS = DIST_COLORS;
})();
