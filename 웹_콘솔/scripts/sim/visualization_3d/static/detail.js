(function () {
  const Sim3D = (window.Sim3D = window.Sim3D || {});

  const COLORS_L1 = {
    "식사": "#e76f51", "카페": "#f4a261", "디저트": "#f4d35e",
    "주점": "#c9184a", "편의점": "#83c5be", "마트": "#52b788",
    "미용": "#ff70a6", "쇼핑": "#9d4edd", "여가": "#7b2cbf",
    "건강": "#06d6a0", "교육": "#118ab2", "기타": "#888",
    "집": "#555", "직장": "#777"
  };

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

  function currentFrame() {
    const state = Sim3D.state || {};
    return typeof Sim3D.frameInfo === "function" ? Sim3D.frameInfo(state.frameIndex) : null;
  }

  // 기억·일정은 지도를 열 때 받지 않는다. 사람을 고르면 그 사람이 든 묶음만 받아 둔다.
  const detail = { agentId: null, status: "idle", data: null, token: 0 };

  // 같은 사람에 대해서는 한 번만 요청한다. 실패해도 여기서 다시 부르지 않는다 —
  // 실패 → 다시 그리기 → 다시 요청이 끝없이 돌기 때문이다. 다시 고르면(selectAgent) 재시도한다.
  function ensureDetail(agentId) {
    if (detail.agentId === agentId) return;
    const token = ++detail.token;
    detail.agentId = agentId;
    detail.status = "loading";
    detail.data = null;
    Sim3D.loadAgentDetail(agentId).then(
      function (data) {
        if (token !== detail.token) return;
        detail.status = "ready";
        detail.data = data || {};
        if (Sim3D.state.selectedAgentId === agentId) Sim3D.renderSelectedAgent();
      },
      function (error) {
        if (token !== detail.token) return;
        console.warn("agent detail load failed", error);
        detail.status = "error";
        detail.error = error && error.message ? error.message : String(error);
        if (Sim3D.state.selectedAgentId === agentId) Sim3D.renderSelectedAgent();
      }
    );
  }

  function eventsFor(agentId) {
    const events = detail.agentId === agentId && detail.data ? detail.data.events : null;
    return Array.isArray(events) ? events : [];
  }

  function memoryBundle(agentId) {
    const memories = detail.agentId === agentId && detail.data ? detail.data.memories : null;
    return memories || {};
  }

  function agentById(agentId) {
    const state = Sim3D.state || {};
    if (state.agentById instanceof Map) {
      return state.agentById.get(agentId) || state.agentById.get(String(agentId));
    }
    return null;
  }

  function positionForFollow(agentId, agent) {
    const position = typeof Sim3D.agentPosition === "function" ? Sim3D.agentPosition(agentId) : null;
    if (position) return position;
    if (agent && agent.home_lon != null && agent.home_lat != null) {
      return [Number(agent.home_lon), Number(agent.home_lat)];
    }
    return null;
  }

  function targetDay(appointment) {
    if (!appointment.day || appointment.offset_days == null) return appointment.day;
    const date = new Date(appointment.day + "T00:00:00Z");
    date.setUTCDate(date.getUTCDate() + Number(appointment.offset_days));
    return date.toISOString().slice(0, 10);
  }

  function placeLabel(event) {
    const poiId = event.poi_id || "";
    if (poiId.startsWith("R_")) {
      return event.cat && event.cat !== "집" ? "집 · " + event.cat : "집";
    }
    if (poiId.startsWith("W_")) {
      return event.cat && event.cat !== "직장" ? "직장 · " + event.cat : "직장";
    }
    return event.poi_name || poiId || "?";
  }

  function empty(text) {
    return '<div class="vz-empty">' + esc(text) + "</div>";
  }

  function sat(value) {
    return value != null ? "만족 " + Number(value).toFixed(2) : "";
  }

  function won(value) {
    return value && value > 0 ? Number(value).toLocaleString("ko-KR") + "원" : "";
  }

  // 글자색은 카드에서 물려받는다 — 콘솔(밝은 바탕)과 단독 보기(어두운 바탕) 모두에서 읽힌다.
  function renderPersona(agent) {
    let topCats = "";
    try {
      const parsed = JSON.parse(agent.top_wd || "{}");
      topCats = Object.entries(parsed)
        .slice(0, 3)
        .map(function (entry) {
          return entry[0] + " " + Math.round(entry[1] * 100) + "%";
        })
        .join(", ");
    } catch (error) {
      topCats = "";
    }
    const sex = agent.gender === "M" ? "남" : agent.gender === "F" ? "여" : agent.gender || "";
    // age 는 연령대("40대")다. 정확한 나이가 있으면 그것을 쓴다
    const age = agent.age_years ? agent.age_years + "세" : agent.age || "";
    const tags = [[age, sex].filter(Boolean).join(" "), agent.life_stage, agent.income ? "소득 " + agent.income : "", agent.tendency]
      .filter(Boolean)
      .map(function (tag) { return '<span class="legacy-tag">' + esc(tag) + "</span>"; })
      .join("");
    const lines = [
      agent.job ? "직업 " + esc(agent.job) : "",
      "평일 하루 " + (agent.daily_wd || 0).toLocaleString("ko-KR") + "원" + (topCats ? " · " + esc(topCats) : ""),
      "사는 곳 " + esc(agent.home_dong || "") + " " + esc(agent.home_poi_name || ""),
      agent.work_poi_name ? "직장 " + esc(agent.work_poi_name) + (agent.commute ? " · 통근 " + agent.commute + "분" : "") : "",
    ].filter(Boolean);
    return '<div class="vz-persona">' + tags + "<br/>" + lines.join("<br/>") + "</div>";
  }

  function renderCurrent(lastEvent) {
    if (!lastEvent) return '<div class="vz-now">집에 있음</div>';
    const what = [lastEvent.cat, lastEvent.sub && lastEvent.sub !== lastEvent.cat ? lastEvent.sub : ""].filter(Boolean).join(" · ");
    const facts = [lastEvent.poi_name || lastEvent.poi_id, sat(lastEvent.sat), won(lastEvent.spent)].filter(Boolean).join(" · ");
    return (
      '<div class="vz-now">' +
      "<b>지금</b> " + esc(lastEvent.time) + " " + esc(what) + (lastEvent.intent ? " — " + esc(lastEvent.intent) : "") + "<br/>" +
      esc(facts) +
      "</div>"
    );
  }

  function renderSchedule(todayEvents, currentHour, lastEvent) {
    if (!todayEvents.length) return empty("일정 없음");
    return todayEvents
      .map(function (event) {
        const eventHour = Number.parseInt(String(event.time || "00:00").slice(0, 2), 10);
        const past = eventHour <= currentHour;
        const isCurrent = lastEvent && event.ord === lastEvent.ord;
        const catLine = event.cat ? (event.sub && event.sub !== event.cat ? event.cat + " · " + event.sub : event.cat) : "";
        const metaParts = [catLine, sat(event.sat), won(event.spent)].filter(Boolean).join(" · ");
        return (
          '<div class="legacy-mem-item' + (isCurrent ? " vz-current" : "") + (past ? "" : " vz-later") +
          '" style="border-left:3px solid ' + (COLORS_L1[event.cat] || "#888") + ';">' +
          "<div><b>" + esc(event.time) + "</b> " + esc(placeLabel(event)) + "</div>" +
          (metaParts ? '<div class="meta">' + esc(metaParts) + "</div>" : "") +
          (event.intent ? '<div class="meta">' + esc(event.intent) + "</div>" : "") +
          "</div>"
        );
      })
      .join("");
  }

  function renderMemories(visibleVisited) {
    if (!visibleVisited.length) return empty("아직 없음");
    return visibleVisited
      .slice(0, 40)
      .map(function (item) {
        return (
          '<div class="legacy-mem-item legacy-visited">' +
          "<div>" + esc(item.poi_name || item.poi_id || "?") + "</div>" +
          '<div class="meta">' + esc([item.day, sat(item.sat)].filter(Boolean).join(" · ")) + "</div>" +
          "</div>"
        );
      })
      .join("");
  }

  function renderRumors(visibleRumor) {
    if (!visibleRumor.length) return empty("없음");
    return visibleRumor
      .slice(0, 30)
      .map(function (item) {
        const topic = item.topic_value ? (item.topic_type ? item.topic_type + " " : "") + item.topic_value : "";
        return (
          '<div class="legacy-mem-item" style="border-left:3px solid #f4a261;">' +
          "<div><b>" + esc(item.source || "?") + "</b>에게 들음" + (topic ? " · " + esc(topic) : "") + "</div>" +
          (item.poi_name ? '<div class="meta">' + esc(item.poi_name) + "</div>" : "") +
          (item.summary ? '<div class="meta">' + esc(item.summary) + "</div>" : "") +
          '<div class="meta">' + esc(item.day) + "</div>" +
          "</div>"
        );
      })
      .join("");
  }

  function renderAppointments(visibleAppts) {
    if (!visibleAppts.length) return empty("없음");
    return visibleAppts
      .slice(0, 20)
      .map(function (item) {
        const target = targetDay(item);
        const time = item.target_time ? " " + item.target_time : "";
        const where = item.meeting_poi_name || item.hint || "장소 미정";
        const role = item.role === "initiator" ? "내가 잡음" : "상대가 잡음";
        return (
          '<div class="legacy-mem-item" style="border-left:3px solid #4cc9f0;">' +
          "<div><b>" + esc(item.with_agent || "?") + "</b> · " + esc(role) + "</div>" +
          '<div class="meta">' + esc(target) + esc(time) + " · " + esc(where) + "</div>" +
          (item.summary ? '<div class="meta">' + esc(item.summary) + "</div>" : "") +
          '<div class="meta">잡은 날 ' + esc(item.day) + "</div>" +
          "</div>"
        );
      })
      .join("");
  }

  function renderKnownPois(visitedMems, mem, currentDay) {
    const visitsByPoi = {};
    visitedMems.forEach(function (item) {
      if (!item.day || item.day > currentDay) return;
      const poiId = item.poi_id;
      if (!poiId || !poiId.startsWith("C_")) return;
      if (!visitsByPoi[poiId]) {
        visitsByPoi[poiId] = { poi_id: poiId, poi_name: item.poi_name || poiId, count: 0, satSum: 0, satN: 0, lastDay: item.day };
      }
      const entry = visitsByPoi[poiId];
      entry.count += 1;
      if (item.sat != null) {
        entry.satSum += item.sat;
        entry.satN += 1;
      }
      if (item.day > entry.lastDay) entry.lastDay = item.day;
    });

    const kpMap = {};
    (mem.knows_poi || []).forEach(function (item) {
      kpMap[item.poi_id] = item;
    });

    const knownTop = Object.values(visitsByPoi)
      .map(function (entry) {
        const known = kpMap[entry.poi_id] || {};
        return Object.assign({}, entry, {
          cat: known.cat || "",
          sub: known.sub || "",
          avgSat: entry.satN > 0 ? entry.satSum / entry.satN : null
        });
      })
      .sort(function (a, b) {
        return b.count - a.count || (b.avgSat || 0) - (a.avgSat || 0);
      })
      .slice(0, 10);

    if (!knownTop.length) return empty("아직 없음");
    return knownTop
      .map(function (item) {
        return (
          '<div class="legacy-mem-item legacy-known">' +
          "<div>" + esc(item.poi_name) +
          (item.cat ? '<span class="legacy-tag">' + esc(item.cat) + (item.sub ? " · " + esc(item.sub) : "") + "</span>" : "") +
          "</div>" +
          '<div class="meta">' + item.count + "회 · 마지막 " + esc(item.lastDay) +
          (item.avgSat != null ? " · 평균 만족 " + item.avgSat.toFixed(2) : "") +
          "</div>" +
          "</div>"
        );
      })
      .join("");
  }

  function renderState(mem, currentDay) {
    const stateDays = mem.state ? Object.keys(mem.state).sort() : [];
    const todayState = mem.state ? mem.state[currentDay] : null;
    const yesterdayKey = stateDays.filter(function (day) { return day < currentDay; }).pop() || null;
    const yesterdayState = yesterdayKey ? mem.state[yesterdayKey] : null;

    function block(label, day, st) {
      return (
        '<div class="legacy-state-block">' +
        "<b>" + esc(label) + "</b> " + esc(day) + "<br/>" +
        "잔액 " + (st.balance || 0).toLocaleString("ko-KR") + "원 · " +
        "기분 " + (st.mood || 0).toFixed(2) + " · " +
        "피로 " + (st.fatigue || 0).toFixed(2) +
        (st.yest_sat != null ? "<br/>전날 평균 만족 " + st.yest_sat.toFixed(2) : "") +
        "</div>"
      );
    }

    let html = "";
    if (todayState) html += block("오늘", currentDay, todayState);
    if (yesterdayState) html += block("어제", yesterdayKey, yesterdayState);
    return html || empty("기록 없음");
  }

  Sim3D.renderSelectedAgent = function renderSelectedAgent() {
    const state = Sim3D.state || {};
    const agentId = state.selectedAgentId;
    if (!agentId) return;
    const agent = agentById(agentId);
    if (!agent) return;
    const content = byId("detail-content");
    const card = byId("detail-card");
    if (!content || !card) return;

    const frame = currentFrame();
    if (!frame) return;
    ensureDetail(agentId);
    if (detail.status !== "ready") {
      const message = detail.status === "error"
        ? "불러오지 못했습니다. 다시 누르면 한 번 더 시도합니다."
        : "불러오는 중…";
      content.innerHTML =
        '<h3 id="d-id">' + esc(agentId) + "</h3>" +
        renderPersona(agent) +
        '<div class="vz-empty">' + esc(message) + "</div>";
      card.hidden = false;
      return;
    }

    const mem = memoryBundle(agentId);
    const events = eventsFor(agentId);
    const currentDay = String(frame.day || "");
    const currentHour = Number(frame.hour || 0);

    const pastEvents = events.filter(function (event) {
      const eventDay = String(event.day || "");
      const eventHour = Number.parseInt(String(event.time || "00:00").slice(0, 2), 10);
      return eventDay < currentDay || (eventDay === currentDay && eventHour <= currentHour);
    });
    const lastEvent = pastEvents[pastEvents.length - 1] || null;
    const todayEvents = events.filter(function (event) {
      return String(event.day || "") === currentDay;
    });

    const visitedMems = (mem.memories || []).filter(function (item) {
      return item.type === "visited" && (item.poi_id || "").startsWith("C_");
    });
    const rumorMems = (mem.memories || []).filter(function (item) {
      return item.type === "rumor";
    });

    const filterByFrame = !state.showAllMem;
    const visibleVisited = filterByFrame ? visitedMems.filter(function (item) { return item.day < currentDay; }) : visitedMems;
    const visibleRumor = filterByFrame ? rumorMems.filter(function (item) { return item.day < currentDay; }) : rumorMems;
    const appts = mem.appointments || [];
    const visibleAppts = filterByFrame ? appts.filter(function (item) { return item.day <= currentDay; }) : appts;

    content.innerHTML =
      '<h3 id="d-id">' + esc(agentId) + "</h3>" +
      renderPersona(agent) +
      renderCurrent(lastEvent) +
      "<h4>오늘 일정</h4>" +
      renderSchedule(todayEvents, currentHour, lastEvent) +
      '<h4>다녀온 곳 <span id="d-mem-toggle" class="legacy-mem-toggle">' +
      (filterByFrame ? "전체" : "지금까지") + "</span></h4>" +
      renderMemories(visibleVisited) +
      "<h4>들은 이야기</h4>" +
      renderRumors(visibleRumor) +
      "<h4>약속</h4>" +
      renderAppointments(visibleAppts) +
      "<h4>자주 가는 곳</h4>" +
      renderKnownPois(visitedMems, mem, currentDay) +
      "<h4>상태</h4>" +
      renderState(mem, currentDay);

    card.hidden = false;
  };

  function formatWon(value) {
    if (typeof Sim3D.formatWon === "function") return Sim3D.formatWon(value);
    return Math.round(Math.max(0, Number(value) || 0)).toLocaleString("ko-KR") + "원";
  }

  // Renders the "why is this spot red?" panel from a Sim3D.computeLocalStats()
  // result: the real agent count, total spending, and category breakdown within
  // the clicked radius -- the actual composition behind the heat colour, not the
  // renderer's internal thresholds.
  Sim3D.showLocalStats = function showLocalStats(stats, radiusMeters) {
    const content = byId("hotspot-content");
    const card = byId("hotspot-card");
    if (!content || !card || !stats) return;

    // Mutually exclusive with the per-agent detail card.
    const detailCard = byId("detail-card");
    if (detailCard) detailCard.hidden = true;
    if (Sim3D.state.selectedAgentId) {
      Sim3D.state.selectedAgentId = null;
      if (typeof Sim3D.refreshLayers === "function") Sim3D.refreshLayers();
    }

    const radius = Math.round(Number(radiusMeters) || stats.radiusMeters || 0);
    const present = Number(stats.totalPresent) || 0;
    let html = "<h3>반경 " + radius + "m</h3>";

    if (!present) {
      html += empty("지금 이 안에 있는 시민이 없습니다.");
      content.innerHTML = html;
      card.hidden = false;
      return;
    }

    html +=
      '<div class="vz-now">' +
      "<b>" + present + "명</b> 중 <b>" + stats.activeCount + "명</b> 외출<br/>" +
      "결제 <b>" + formatWon(stats.totalSpent) + "</b>" +
      "</div>";

    if (!stats.activeCount) {
      // Heat here is driven by residential density, not commercial activity.
      html += empty("모두 집에 있습니다. 사는 사람이 많아 색이 진합니다.");
      content.innerHTML = html;
      card.hidden = false;
      return;
    }

    html += "<h4>업종별 결제</h4>";
    const top = (stats.categories || []).slice(0, 5);
    html += top
      .map(function (entry) {
        return (
          '<div class="legacy-mem-item" style="border-left:3px solid ' + (COLORS_L1[entry.cat] || "#888") + ';">' +
          "<div>" + esc(entry.cat) + "</div>" +
          '<div class="meta">' + entry.count + "건 · " + formatWon(entry.spent) + "</div>" +
          "</div>"
        );
      })
      .join("");

    content.innerHTML = html;
    card.hidden = false;
  };

  Sim3D.selectAgent = function selectAgent(agentId) {
    const agent = agentById(agentId);
    if (!agent) return;
    // Mutually exclusive with the hotspot panel.
    const hotspotCard = byId("hotspot-card");
    if (hotspotCard) hotspotCard.hidden = true;
    if (detail.status === "error") detail.agentId = null;
    Sim3D.state.selectedAgentId = agent.id;
    Sim3D.renderSelectedAgent();
    Sim3D.followAgent(agent.id);
    if (typeof Sim3D.refreshLayers === "function") Sim3D.refreshLayers();
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

  Sim3D.toggleMemView = function toggleMemView() {
    Sim3D.state.showAllMem = !Sim3D.state.showAllMem;
    if (Sim3D.state.selectedAgentId) Sim3D.renderSelectedAgent();
  };

  Sim3D.initDetail = function initDetail() {
    const close = byId("detail-close");
    if (close) {
      close.addEventListener("click", function () {
        Sim3D.state.selectedAgentId = null;
        const card = byId("detail-card");
        if (card) card.hidden = true;
        if (typeof Sim3D.refreshLayers === "function") Sim3D.refreshLayers();
      });
    }

    const hotspotClose = byId("hotspot-close");
    if (hotspotClose) {
      hotspotClose.addEventListener("click", function () {
        const card = byId("hotspot-card");
        if (card) card.hidden = true;
      });
    }

    document.addEventListener("click", function (event) {
      if (event.target && event.target.id === "d-mem-toggle") Sim3D.toggleMemView();
    });

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
