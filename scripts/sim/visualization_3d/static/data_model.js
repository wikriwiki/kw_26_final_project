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

  function currentEffects(name) {
    const state = getState();
    const items = state.meta && Array.isArray(state.meta[name]) ? state.meta[name] : [];
    const index = safeFrameIndex(state);
    return items.filter(function (item) {
      const frameIndex = Number(item && item.frame_index);
      return Number.isFinite(frameIndex) && Math.abs(frameIndex - index) <= 1;
    });
  }

  function macroLabel(frame, index) {
    const day = frame && frame.day != null ? String(frame.day) : "";
    const dayPart = day ? day.slice(5) : "Frame " + index;
    const hour = frame && frame.hour != null ? frame.hour : 0;
    return dayPart + " " + hour + "h";
  }

  Sim3D.initDataModel = function initDataModel() {
    const state = getState();
    state.frameMaps = timelineFrames(state).map(frameMap);
    state.macroSeries = Sim3D.buildMacroSeries();
  };

  Sim3D.getColorForAgentFrame = function getColorForAgentFrame(agent, agentFrame) {
    if (!agentFrame) return [98, 115, 134, 70];

    const category = agentFrame.l1 || agentFrame.cat;
    const distCode = agent && agent.dist_code != null ? String(agent.dist_code) : "";
    const baseColor = CAT_COLORS[category] || DIST_COLORS[distCode] || CAT_COLORS["기타"];
    const alpha = finiteNumber(agentFrame.spent, 0) > 0 ? 245 : 185;
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

    return collectAgents(state, currentMap, nextMap).map(function (agent) {
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
  };

  Sim3D.getCurrentBursts = function getCurrentBursts() {
    return currentEffects("spend_bursts");
  };

  Sim3D.getCurrentMeetups = function getCurrentMeetups() {
    return currentEffects("meetups");
  };

  Sim3D.getCurrentRumors = function getCurrentRumors() {
    return currentEffects("rumor_edges");
  };

  Sim3D.buildMacroSeries = function buildMacroSeries() {
    const state = getState();
    const timeline = timelineFrames(state);
    const targetCats = ["식사", "카페", "디저트"];
    const targetDists = ["11110", "11140"];
    const catTotals = [0, 0, 0];
    const distTotals = [0, 0];
    const labels = [];
    const catData = [[], [], []];
    const distData = [[], []];
    const lookup = agentLookup(state);

    timeline.forEach(function (frame, index) {
      labels.push(macroLabel(frame, index));

      (frame && Array.isArray(frame.agents) ? frame.agents : []).forEach(function (agentFrame) {
        if (!agentFrame) return;
        const amount = Math.max(0, finiteNumber(agentFrame.spent, 0));
        if (!amount) return;

        const category = agentFrame.l1 || agentFrame.cat || "기타";
        const catIndex = targetCats.indexOf(category);
        if (catIndex >= 0) {
          catTotals[catIndex] += amount;
        }

        const agent = getAgentFromLookup(lookup, agentFrame.id);
        const distCode = agent && agent.dist_code != null ? String(agent.dist_code) : "";
        const distIndex = targetDists.indexOf(distCode);
        if (distIndex >= 0) {
          distTotals[distIndex] += amount;
        }
      });

      targetCats.forEach(function (_category, index) {
        catData[index].push(catTotals[index]);
      });
      targetDists.forEach(function (_distCode, index) {
        distData[index].push(distTotals[index]);
      });
    });

    return {
      labels: labels,
      catData: catData,
      distData: distData,
      targetCats: targetCats,
      targetDists: targetDists
    };
  };

  Sim3D.formatWon = formatWon;
  Sim3D.DIST_NAMES = DIST_NAMES;
})();
