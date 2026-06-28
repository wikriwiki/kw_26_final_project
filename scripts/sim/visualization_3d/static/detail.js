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
    return state.timeline && state.timeline[state.frameIndex] ? state.timeline[state.frameIndex] : null;
  }

  function eventsFor(agentId) {
    const state = Sim3D.state || {};
    const events = state.events && state.events[agentId];
    return Array.isArray(events) ? events : [];
  }

  function memoryBundle(agentId) {
    const state = Sim3D.state || {};
    return state.memories && state.memories[agentId] ? state.memories[agentId] : {};
  }

  function agentById(agentId) {
    const state = Sim3D.state || {};
    if (state.agentById instanceof Map) {
      return state.agentById.get(agentId) || state.agentById.get(String(agentId));
    }
    return null;
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

  function targetDay(appointment) {
    if (!appointment.day || appointment.offset_days == null) return appointment.day;
    const date = new Date(appointment.day + "T00:00:00Z");
    date.setUTCDate(date.getUTCDate() + Number(appointment.offset_days));
    return date.toISOString().slice(0, 10);
  }

  function placeLabel(event) {
    const poiId = event.poi_id || "";
    if (poiId.startsWith("C_")) {
      return "📍 " + (event.poi_name || poiId);
    }
    if (poiId.startsWith("R_")) {
      return event.cat && event.cat !== "집" ? "🏠 집 — " + event.cat : "🏠 집";
    }
    if (poiId.startsWith("W_")) {
      return event.cat && event.cat !== "직장" ? "💼 직장 — " + event.cat : "💼 직장";
    }
    return "📍 " + (event.poi_name || poiId || "?");
  }

  function renderPersona(agent) {
    let topCats = "";
    try {
      const parsed = JSON.parse(agent.top_wd || "{}");
      topCats = Object.entries(parsed)
        .slice(0, 3)
        .map(function (entry) {
          return entry[0] + "(" + Math.round(entry[1] * 100) + "%)";
        })
        .join(", ");
    } catch (error) {
      topCats = "";
    }
    return (
      '<div style="font-size:11px; line-height:1.7;">' +
      '<span class="legacy-tag">' + esc(agent.age) + " " + esc(agent.gender) + "</span>" +
      '<span class="legacy-tag">' + esc(agent.life_stage || "") + "</span>" +
      '<span class="legacy-tag">소득 ' + esc(agent.income || "") + "</span>" +
      '<span class="legacy-tag">' + esc(agent.tendency || "") + "</span><br/>" +
      "직업: " + esc(agent.job || "-") + " · 평일 " + (agent.daily_wd || 0).toLocaleString() + "원<br/>" +
      "평일 Top: " + esc(topCats) + "<br/>" +
      "거주: " + esc(agent.home_dong) + " " + esc(agent.home_poi_name || "") + "<br/>" +
      (agent.work_poi_name
        ? "직장: " + esc(agent.work_poi_name) + " (" + (agent.commute || 0) + "분)"
        : "직장: 없음") +
      "</div>"
    );
  }

  function renderCurrent(lastEvent) {
    if (!lastEvent) return '<div style="font-size:11px; color:#888;">아직 활동 없음 (집)</div>';
    return (
      '<div style="background:rgba(76,201,240,0.1); padding:6px 10px; border-radius:6px; margin-top:6px; font-size:11px;">' +
      "<b>📍 현재 활동</b> " + esc(lastEvent.time) + " " + esc(lastEvent.anchor) + "<br/>" +
      esc(lastEvent.cat || "") + (lastEvent.sub ? " · " + esc(lastEvent.sub) : "") + " — " + esc(lastEvent.intent || "") + "<br/>" +
      "장소: " + esc(lastEvent.poi_name || lastEvent.poi_id) +
      (lastEvent.sat != null ? " / 만족도 " + lastEvent.sat.toFixed(2) : "") +
      (lastEvent.spent && lastEvent.spent > 0 ? " / 💰 " + Number(lastEvent.spent).toLocaleString() + "원" : "") +
      "</div>"
    );
  }

  function renderSchedule(todayEvents, currentHour, lastEvent) {
    if (!todayEvents.length) return '<div style="font-size:11px; color:#888;">(오늘 스케줄 없음)</div>';
    return todayEvents
      .map(function (event) {
        const eventHour = Number.parseInt(String(event.time || "00:00").slice(0, 2), 10);
        const past = eventHour <= currentHour;
        const isCurrent = lastEvent && event.ord === lastEvent.ord;
        const bg = isCurrent ? "rgba(231,111,81,0.25)" : past ? "rgba(255,255,255,0.04)" : "transparent";
        const opacity = past ? 1 : 0.55;
        const catLine = event.cat ? (event.sub && event.sub !== event.cat ? event.cat + " · " + event.sub : event.cat) : "";
        const satLine = event.sat != null ? "만족도 " + event.sat.toFixed(2) : "";
        const spentLine = event.spent && event.spent > 0 ? "💰 " + Number(event.spent).toLocaleString() + "원" : "";
        const metaParts = [catLine, satLine, spentLine].filter(Boolean).join(" · ");
        return (
          '<div class="legacy-mem-item" style="background:' + bg + "; opacity:" + opacity +
          "; border-left:3px solid " + (COLORS_L1[event.cat] || "#888") + ';">' +
          "<div><b>" + esc(event.time) + '</b> <span style="color:#fff;">' + esc(placeLabel(event)) + "</span></div>" +
          (metaParts ? '<div class="meta">' + esc(metaParts) + "</div>" : "") +
          (event.intent ? '<div class="meta" style="color:#aaa;">' + esc(event.intent) + "</div>" : "") +
          "</div>"
        );
      })
      .join("");
  }

  function renderMemories(visibleVisited) {
    if (!visibleVisited.length) return '<div style="font-size:11px; color:#888;">(누적된 방문 기억 없음)</div>';
    return visibleVisited
      .slice(0, 40)
      .map(function (item) {
        return (
          '<div class="legacy-mem-item legacy-visited">' +
          '<div><span style="color:#fff;">📍 ' + esc(item.poi_name || item.poi_id || "?") + "</span></div>" +
          '<div class="meta">' + esc(item.day) + " · 만족도 " + (item.sat != null ? item.sat.toFixed(2) : "?") +
          " · 중요도 " + (item.imp || 0).toFixed(2) + "</div>" +
          "</div>"
        );
      })
      .join("");
  }

  function renderRumors(visibleRumor) {
    if (!visibleRumor.length) return '<div style="font-size:11px; color:#888;">(들은 소문 없음)</div>';
    return visibleRumor
      .slice(0, 30)
      .map(function (item) {
        const topic = item.topic_value ? (item.topic_type || "") + ": " + item.topic_value : "";
        return (
          '<div class="legacy-mem-item" style="border-left:3px solid #f4a261;">' +
          '<div><span style="background:rgba(244,162,97,0.2); color:#f4a261; font-size:10px; padding:1px 6px; border-radius:3px;">💬 ' +
          esc(item.source || "?") + " 한테 들음</span></div>" +
          (topic ? '<div class="meta" style="color:#fff;">' + esc(topic) + "</div>" : "") +
          (item.poi_name ? '<div class="meta">→ ' + esc(item.poi_name) + "</div>" : "") +
          '<div class="meta">' + esc(item.day) + " · 중요도 " + (item.imp || 0).toFixed(2) + "</div>" +
          (item.summary ? '<div class="meta" style="color:#cbd5e0; font-style:italic;">💬 ' + esc(item.summary) + "</div>" : "") +
          "</div>"
        );
      })
      .join("");
  }

  function renderAppointments(visibleAppts) {
    if (!visibleAppts.length) return '<div style="font-size:11px; color:#888;">(잡은 약속 없음)</div>';
    return visibleAppts
      .slice(0, 20)
      .map(function (item) {
        const target = targetDay(item);
        const time = item.target_time ? " " + item.target_time : "";
        const where = item.meeting_poi_name || item.hint || "미정";
        const role = item.role === "initiator" ? "내가 제안" : "상대가 제안";
        return (
          '<div class="legacy-mem-item" style="border-left:3px solid #4cc9f0;">' +
          '<div><span style="background:rgba(76,201,240,0.2); color:#4cc9f0; font-size:10px; padding:1px 6px; border-radius:3px;">🤝 ' +
          esc(role) + '</span><span style="color:#fff; margin-left:4px;">' + esc(item.with_agent || "?") + "</span></div>" +
          '<div class="meta" style="color:#fff;">📅 만남일: <b>' + esc(target) + esc(time) + "</b> @ " + esc(where) + "</div>" +
          '<div class="meta">잡은 날: ' + esc(item.day) + "</div>" +
          (item.summary ? '<div class="meta" style="color:#aaa;">' + esc(item.summary) + "</div>" : "") +
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
          cat: known.cat || "?",
          sub: known.sub || "",
          avgSat: entry.satN > 0 ? entry.satSum / entry.satN : null,
          affinity: known.affinity
        });
      })
      .sort(function (a, b) {
        return b.count - a.count || (b.avgSat || 0) - (a.avgSat || 0);
      })
      .slice(0, 10);

    if (!knownTop.length) return '<div style="font-size:11px; color:#888;">(아직 단골 POI 없음 — 외출 기록 누적 후 표시)</div>';
    return knownTop
      .map(function (item) {
        return (
          '<div class="legacy-mem-item legacy-known">' +
          "<div>" + esc(item.poi_name) +
          (item.cat && item.cat !== "?" ? '<span class="legacy-tag">' + esc(item.cat) + (item.sub ? " · " + esc(item.sub) : "") + "</span>" : "") +
          "</div>" +
          '<div class="meta">방문 <b>' + item.count + "회</b> (~" + esc(item.lastDay) + ")" +
          (item.avgSat != null ? " · 평균 만족도 " + item.avgSat.toFixed(2) : "") +
          (item.affinity != null ? " · affinity " + item.affinity.toFixed(2) : "") +
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
        "<b>" + esc(label) + " (" + esc(day) + ")</b><br/>" +
        "💰 잔액 " + (st.balance || 0).toLocaleString() + "원 · " +
        "😊 mood " + (st.mood || 0).toFixed(2) + " · " +
        "😴 fatigue " + (st.fatigue || 0).toFixed(2) +
        (st.yest_sat != null ? "<br/>어제 평균 만족도 " + st.yest_sat.toFixed(2) : "") +
        "</div>"
      );
    }

    let html = "";
    if (todayState) html += block("오늘", currentDay, todayState);
    if (yesterdayState) html += block("어제", yesterdayKey, yesterdayState);
    return html || '<div style="font-size:11px; color:#888;">(State 없음)</div>';
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

    const mem = memoryBundle(agentId);
    const events = eventsFor(agentId);
    const frame = currentFrame();
    if (!frame) return;
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
      "<h4>🗓️ 오늘 스케줄</h4>" +
      renderSchedule(todayEvents, currentHour, lastEvent) +
      '<h4>🧠 기억 메모리 (Memory Stream) <span id="d-mem-toggle" class="legacy-mem-toggle">' +
      (filterByFrame ? "전체 보기" : "현재 시점까지만") + "</span></h4>" +
      renderMemories(visibleVisited) +
      "<h4>💬 들은 소문 (rumor)</h4>" +
      renderRumors(visibleRumor) +
      "<h4>🤝 잡은 약속 (appointment)</h4>" +
      renderAppointments(visibleAppts) +
      "<h4>⭐ 단골 POI (KNOWS_POI Top)</h4>" +
      renderKnownPois(visitedMems, mem, currentDay) +
      "<h4>📊 어제 State</h4>" +
      renderState(mem, currentDay);

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
