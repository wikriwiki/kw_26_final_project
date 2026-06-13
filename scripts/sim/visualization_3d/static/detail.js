(function () {
  const Sim3D = (window.Sim3D = window.Sim3D || {});

  function byId(id) {
    return document.getElementById(id);
  }

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function row(label, value) {
    return '<div class="detail-row"><span>' + esc(label) + "</span><b>" + esc(value || "-") + "</b></div>";
  }

  function currentFrame() {
    const state = Sim3D.state || {};
    return state.timeline && state.timeline[state.frameIndex] ? state.timeline[state.frameIndex] : null;
  }

  function currentFrameMap() {
    const state = Sim3D.state || {};
    if (Array.isArray(state.frameMaps) && state.frameMaps[state.frameIndex] instanceof Map) {
      return state.frameMaps[state.frameIndex];
    }
    return new Map();
  }

  function currentAgentFrame(agentId) {
    const map = currentFrameMap();
    return map.get(agentId) || map.get(String(agentId)) || null;
  }

  function eventsFor(agentId) {
    const state = Sim3D.state || {};
    const events = state.events && state.events[agentId];
    return Array.isArray(events) ? events : [];
  }

  function lastEventFor(agentId) {
    const frame = currentFrame();
    if (!frame) return null;
    const currentDay = String(frame.day || "");
    const currentHour = Number(frame.hour || 0);
    return eventsFor(agentId)
      .filter(function (event) {
        const eventDay = String(event.day || "");
        const eventHour = Number.parseInt(String(event.time || "00:00").slice(0, 2), 10);
        return eventDay < currentDay || (eventDay === currentDay && eventHour <= currentHour);
      })
      .slice(-1)[0] || null;
  }

  function memoryBundle(agentId) {
    const state = Sim3D.state || {};
    return state.memories && state.memories[agentId] ? state.memories[agentId] : {};
  }

  function summarizeAppointments(mem) {
    return (mem.appointments || [])
      .slice(0, 4)
      .map(function (item) {
        return (item.with_agent || "?") + " @ " + (item.meeting_poi_name || item.hint || "미정");
      })
      .join(" / ") || "없음";
  }

  function summarizeRumors(mem) {
    return (mem.memories || [])
      .filter(function (item) {
        return item.type === "rumor";
      })
      .slice(0, 4)
      .map(function (item) {
        return (item.source || "?") + ": " + (item.topic_value || item.topic_type || "rumor");
      })
      .join(" / ") || "없음";
  }

  function agentById(agentId) {
    const state = Sim3D.state || {};
    if (state.agentById instanceof Map) {
      return state.agentById.get(agentId) || state.agentById.get(String(agentId));
    }
    return null;
  }

  function positionForFollow(agentId, agent) {
    const frame = currentAgentFrame(agentId);
    if (frame && frame.lon != null && frame.lat != null) {
      return [Number(frame.lon), Number(frame.lat)];
    }
    if (agent && agent.home_lon != null && agent.home_lat != null) {
      return [Number(agent.home_lon), Number(agent.home_lat)];
    }
    return null;
  }

  Sim3D.renderSelectedAgent = function renderSelectedAgent() {
    const state = Sim3D.state || {};
    if (!state.selectedAgentId) return;
    const agent = agentById(state.selectedAgentId);
    if (!agent) return;
    const mem = memoryBundle(state.selectedAgentId);
    const lastEvent = lastEventFor(state.selectedAgentId);
    const frame = currentAgentFrame(state.selectedAgentId);
    const content = byId("detail-content");
    const card = byId("detail-card");
    if (!content || !card) return;

    const district = (Sim3D.DIST_NAMES && Sim3D.DIST_NAMES[String(agent.dist_code)]) || agent.district || "";
    content.innerHTML =
      "<h2>" + esc(state.selectedAgentId) + "</h2>" +
      '<div class="detail-grid">' +
      row("거주", [district, agent.home_dong].filter(Boolean).join(" ")) +
      row("직업", agent.job || "-") +
      row("소비 성향", agent.tendency || "-") +
      row("현재 장소", lastEvent ? lastEvent.poi_name || lastEvent.poi_id : frame ? frame.anchor || frame.cat : "집") +
      row("현재 의도", lastEvent ? lastEvent.intent || lastEvent.anchor : frame ? frame.intent || frame.anchor : "-") +
      row("현재 소비", Sim3D.formatWon ? Sim3D.formatWon((frame && frame.spent) || (lastEvent && lastEvent.spent) || 0) : "0원") +
      row("약속", summarizeAppointments(mem)) +
      row("소문", summarizeRumors(mem)) +
      "</div>";
    card.hidden = false;
  };

  Sim3D.selectAgent = function selectAgent(agentId) {
    const agent = agentById(agentId);
    if (!agent) return;
    Sim3D.state.selectedAgentId = agentId;
    Sim3D.renderSelectedAgent();
    Sim3D.followAgent(agentId);
  };

  Sim3D.followAgent = function followAgent(agentId) {
    const agent = agentById(agentId);
    const position = positionForFollow(agentId, agent);
    if (!position || typeof Sim3D.setCamera !== "function") return;
    const bearing = Sim3D.state && Sim3D.state.map && typeof Sim3D.state.map.getBearing === "function"
      ? Sim3D.state.map.getBearing() + 18
      : -24;
    Sim3D.setCamera(
      { longitude: position[0], latitude: position[1], zoom: 15.5, pitch: 70, bearing: bearing },
      { duration: 900 }
    );
  };

  Sim3D.initDetail = function initDetail() {
    const close = byId("detail-close");
    if (close) {
      close.addEventListener("click", function () {
        Sim3D.state.selectedAgentId = null;
        const card = byId("detail-card");
        if (card) card.hidden = true;
      });
    }

    const search = byId("agent-search");
    if (search) {
      search.addEventListener("keydown", function (event) {
        if (event.key !== "Enter") return;
        const id = event.target.value.trim();
        if (!id) return;
        if (!agentById(id)) {
          if (typeof Sim3D.showNews === "function") {
            Sim3D.showNews("에이전트를 찾을 수 없습니다: " + id);
          }
          return;
        }
        Sim3D.selectAgent(id);
      });
    }
  };
})();
