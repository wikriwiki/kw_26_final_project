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

  // The base palettes above are Tailwind 300/400-level tints: pretty, but very
  // light + medium saturation, so on the muted basemap the agent dots read as
  // washed-out pastels. vivify() deepens every swatch IN PLACE -- it keeps the
  // hue (so legend identity and dot colour stay matched) and only boosts
  // saturation and pulls high-lightness tints down toward a richer mid-tone.
  function clampUnit(value) {
    return Math.max(0, Math.min(1, value));
  }

  function rgbToHslLocal(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const l = (max + min) / 2;
    const d = max - min;
    let h = 0, s = 0;
    if (d !== 0) {
      s = d / (1 - Math.abs(2 * l - 1));
      if (max === r) h = ((g - b) / d) % 6;
      else if (max === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h *= 60;
      if (h < 0) h += 360;
    }
    return [h, s, l];
  }

  function hslToRgbLocal(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = l - c / 2;
    let r, g, b;
    if (h < 60) { r = c; g = x; b = 0; }
    else if (h < 120) { r = x; g = c; b = 0; }
    else if (h < 180) { r = 0; g = c; b = x; }
    else if (h < 240) { r = 0; g = x; b = c; }
    else if (h < 300) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }
    return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
  }

  function vivify(rgb) {
    const hsl = rgbToHslLocal(rgb[0], rgb[1], rgb[2]);
    const s = clampUnit(hsl[1] * 1.3 + 0.18);
    const l = clampUnit(Math.min(hsl[2], 0.56));
    return hslToRgbLocal(hsl[0], s, l);
  }

  function vivifyPalette(palette) {
    Object.keys(palette).forEach(function (key) {
      palette[key] = vivify(palette[key]);
    });
  }

  vivifyPalette(CAT_COLORS);
  vivifyPalette(DIST_COLORS);

  const NO_FRAME = -1;
  const COLOR_APPT_ON = [157, 78, 221, 220];
  const COLOR_APPT_OFF = [136, 136, 136, 150];
  const COLOR_ABSENT = [98, 115, 134, 120];
  const COLOR_GREY = [136, 136, 136];

  function finiteNumber(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function formatWon(value) {
    return Math.round(Math.max(0, finiteNumber(value, 0))).toLocaleString("ko-KR") + "원";
  }

  function getState() {
    Sim3D.state = Sim3D.state || {};
    return Sim3D.state;
  }

  // 모든 렌더 입력을 대상자 수만큼 미리 잡아 둔 typed array 에 담는다. 재생 중 매 프레임은
  // 보간(interpolate)만 하고 객체를 만들지 않는다 — 예전에는 매 프레임 대상자마다 객체·배열을
  // 새로 만들고 히트맵까지 다시 집계해서 GC 와 GPU 가 번갈아 멈췄다.
  let model = null;

  function paletteForDict(values) {
    const palette = new Array(values.length + 1).fill(null);
    values.forEach(function (value, index) {
      palette[index + 1] = CAT_COLORS[value] || null;
    });
    return palette;
  }

  function clampFrame(index) {
    if (!model) return 0;
    const last = Math.max(0, model.F - 1);
    const value = Math.trunc(finiteNumber(index, 0));
    return Math.max(0, Math.min(last, value));
  }

  function loadedFrames() {
    return model ? model.pack.loadedFrames : 0;
  }

  function isPresent(column, slot) {
    const value = column[slot];
    return value === value; // NaN 이면 그 시각에 기록이 없다
  }

  Sim3D.initDataModel = function initDataModel() {
    const state = getState();
    const pack = state.pack;
    if (!pack) throw new Error("data pack is not loaded");
    const N = pack.N;
    const homeX = new Float64Array(N);
    const homeY = new Float64Array(N);
    const distKey = new Array(N);
    const appt = new Uint8Array(N);
    const indexById = new Map();
    pack.agents.forEach(function (agent, index) {
      homeX[index] = finiteNumber(agent && agent.home_lon, SEOUL_CENTER[0]);
      homeY[index] = finiteNumber(agent && agent.home_lat, SEOUL_CENTER[1]);
      distKey[index] = agent && agent.dist_code != null ? String(agent.dist_code) : "";
      indexById.set(String(agent.id), index);
    });
    pack.appointmentAgents.forEach(function (index) {
      if (index >= 0 && index < N) appt[index] = 1;
    });
    const anchors = (pack.dicts && pack.dicts.anchor) || [];

    model = {
      pack: pack,
      N: N,
      F: pack.F,
      columns: pack.columns,
      homeX: homeX,
      homeY: homeY,
      distKey: distKey,
      appt: appt,
      indexById: indexById,
      l1Palette: paletteForDict((pack.dicts && pack.dicts.l1) || []),
      catPalette: paletteForDict((pack.dicts && pack.dicts.cat) || []),
      residenceCode: anchors.indexOf("residence") + 1,
      visible: new Uint32Array(N),
      visibleCount: 0,
      visibleKey: null,
      sx: new Float64Array(N),
      sy: new Float64Array(N),
      ex: new Float64Array(N),
      ey: new Float64Array(N),
      ref: new Int32Array(N),
      colors: new Uint8Array(N * 4),
      colorFrom: new Uint8Array(N * 4),
      colorTo: new Uint8Array(N * 4),
      refNext: new Int32Array(N),
      radii: new Float32Array(N),
      radiusFrom: new Float32Array(N),
      radiusTo: new Float32Array(N),
      heatCache: new Map(),
      heatPos: new Float64Array(N * 3),
      heatW: new Float32Array(N),
      pos: new Float64Array(N * 3),
      geomKey: null,
      colorKey: null,
      heatKey: null,
      colorVersion: 0,
      heatVersion: 0,
      colorAttr: null,
      radiusAttr: null,
      attrVersion: -1,
      heatData: null,
      heatDataVersion: -1,
      trailKey: null,
      trails: null,
      routeKey: null,
      route: null
    };
    state.agentIndexById = indexById;
  };

  function updateVisible(state) {
    const filter = state.distFilter || "all";
    if (model.visibleKey === filter) return;
    let count = 0;
    for (let a = 0; a < model.N; a++) {
      if (filter === "all" || model.distKey[a] === filter) model.visible[count++] = a;
    }
    model.visibleCount = count;
    model.visibleKey = filter;
  }

  function writeGeometry(index, hasNext) {
    const N = model.N;
    const x = model.columns.x;
    const y = model.columns.y;
    const spent = model.columns.spent;
    const base = index * N;
    const nextBase = (index + 1) * N;
    for (let k = 0; k < model.visibleCount; k++) {
      const a = model.visible[k];
      const ci = base + a;
      const present = isPresent(x, ci);
      const cx = present ? x[ci] : model.homeX[a];
      const cy = present ? y[ci] : model.homeY[a];
      let nx = cx;
      let ny = cy;
      let ref = present ? ci : NO_FRAME;
      if (hasNext) {
        const ni = nextBase + a;
        if (isPresent(x, ni)) {
          nx = x[ni];
          ny = y[ni];
          if (!present) ref = ni; // 예전 규칙: 현재 기록이 없으면 다음 시각 기록으로 색을 정한다
        }
      }
      model.sx[k] = cx;
      model.sy[k] = cy;
      model.ex[k] = nx;
      model.ey[k] = ny;
      model.ref[k] = ref;
      let refNext = NO_FRAME;
      if (hasNext) {
        const ni = nextBase + a;
        const nni = ni + N;
        if (isPresent(x, ni)) refNext = ni;
        else if (index + 2 < Math.min(model.F, loadedFrames()) && isPresent(x, nni)) refNext = nni;
      } else {
        refNext = ref;
      }
      model.refNext[k] = refNext;
      // 크기는 그 시각에 결제했는지로 정한다. 시간대 경계에서 15 ↔ 26 으로 뚝 바뀌지 않도록
      // 다음 시각의 크기를 같이 적어 두고 위치처럼 보간한다(interpolate)
      model.radiusFrom[k] = present && spent[ci] > 0 ? 26 : 15;
      model.radiusTo[k] = hasNext && isPresent(x, nextBase + a) ? (spent[nextBase + a] > 0 ? 26 : 15) : model.radiusFrom[k];
      model.radii[k] = model.radiusFrom[k];
      const o = k * 3;
      model.heatPos[o] = cx;
      model.heatPos[o + 1] = cy;
      model.heatPos[o + 2] = 10;
    }
  }

  function writeColors(colorMode) {
    writeColorSet(colorMode, model.ref, model.colorFrom);
    writeColorSet(colorMode, model.refNext, model.colorTo);
    model.colors.set(model.colorFrom);
  }

  function writeColorSet(colorMode, refs, colors) {
    const spent = model.columns.spent;
    const l1 = model.columns.l1;
    const cat = model.columns.cat;
    for (let k = 0; k < model.visibleCount; k++) {
      const a = model.visible[k];
      const ref = refs[k];
      const o = k * 4;
      let rgb;
      let alpha;
      if (colorMode === "appointment") {
        rgb = model.appt[a] ? COLOR_APPT_ON : COLOR_APPT_OFF;
        alpha = rgb[3];
      } else if (ref < 0) {
        rgb = colorMode === "dist" ? DIST_COLORS[model.distKey[a]] || COLOR_GREY : COLOR_ABSENT;
        alpha = 120;
      } else {
        alpha = spent[ref] > 0 ? 255 : 215;
        if (colorMode === "dist") {
          rgb = DIST_COLORS[model.distKey[a]] || CAT_COLORS["기타"];
        } else {
          // 예전 규칙: category = l1 || cat, 색은 CAT → 자치구 → 기타 순
          const l1Code = l1 ? l1[ref] : 0;
          const catCode = cat ? cat[ref] : 0;
          const named = l1Code ? model.l1Palette[l1Code] : catCode ? model.catPalette[catCode] : null;
          rgb = named || DIST_COLORS[model.distKey[a]] || CAT_COLORS["기타"];
        }
      }
      colors[o] = rgb[0];
      colors[o + 1] = rgb[1];
      colors[o + 2] = rgb[2];
      colors[o + 3] = alpha;
    }
  }

  // 열지도 색 눈금은 5,000명 표본에 맞춰 잡혀 있다. 시민 수가 다른 팩도 같은 진하기로 읽히게
  // 사람 수에 반비례해 무게를 키운다(5,000명보다 많으면 그대로).
  const HEAT_REFERENCE_AGENTS = 5000;
  function heatScale() {
    return model && model.N > 0 ? Math.max(1, HEAT_REFERENCE_AGENTS / model.N) : 1;
  }

  function writeHeatWeights(heatMode) {
    const spent = model.columns.spent;
    for (let k = 0; k < model.visibleCount; k++) {
      const ref = model.ref[k];
      const amount = ref >= 0 ? spent[ref] : 0;
      model.heatW[k] = (heatMode === "spending" ? Math.max(0.22, amount / 9000) : 1) * heatScale();
    }
  }

  // 프레임 번호·필터·색상 모드가 바뀔 때만 다시 계산한다. 같은 키면 바로 끝난다.
  Sim3D.prepareFrame = function prepareFrame() {
    if (!model) return;
    const state = getState();
    updateVisible(state);
    const index = clampFrame(state.frameIndex);
    const hasNext = index + 1 < Math.min(model.F, loadedFrames());
    const geomKey = index + "|" + model.visibleKey + "|" + (hasNext ? 1 : 0);
    const colorKey = geomKey + "|" + (state.colorMode || "cat");
    const heatKey = geomKey + "|" + (state.heatMode || "spending");
    if (geomKey !== model.geomKey) {
      writeGeometry(index, hasNext);
      model.geomKey = geomKey;
    }
    if (colorKey !== model.colorKey) {
      writeColors(state.colorMode || "cat");
      model.colorBlend = 0;
      model.colorKey = colorKey;
      model.colorVersion++;
    }
    if (heatKey !== model.heatKey) {
      writeHeatWeights(state.heatMode || "spending");
      model.heatKey = heatKey;
      model.heatVersion++;
    }
  };

  Sim3D.interpolate = function interpolate(t) {
    if (!model) return;
    const tt = t > 0 ? (t < 1 ? t : 1) : 0;
    const pos = model.pos;
    for (let k = 0; k < model.visibleCount; k++) {
      const o = k * 3;
      pos[o] = model.sx[k] + (model.ex[k] - model.sx[k]) * tt;
      pos[o + 1] = model.sy[k] + (model.ey[k] - model.sy[k]) * tt;
      pos[o + 2] = 10;
      model.radii[k] = model.radiusFrom[k] + (model.radiusTo[k] - model.radiusFrom[k]) * tt;
    }
    // 색은 장소 종류를 뜻하므로 시간대 대부분은 그대로 두고, 마지막 25% 동안만 다음 색으로 넘긴다
    const u = tt <= 0.75 ? 0 : (tt - 0.75) / 0.25;
    const w = u * u * (3 - 2 * u);
    if (w !== model.colorBlend) {
      const from = model.colorFrom;
      const to = model.colorTo;
      const out = model.colors;
      const m = model.visibleCount * 4;
      for (let i = 0; i < m; i++) out[i] = from[i] + (to[i] - from[i]) * w;
      model.colorBlend = w;
      model.colorVersion++;
    }
  };

  // deck.gl 은 binary 속성 객체의 참조가 같으면 다시 올리지 않는다. 그래서 위치는 매번 새
  // 래퍼로, 색·반지름은 내용이 바뀐 때만 새 래퍼로 넘긴다.
  Sim3D.dotsData = function dotsData() {
    if (!model) return { length: 0 };
    const n = model.visibleCount;
    if (model.attrVersion !== model.colorVersion) {
      model.colorAttr = { value: model.colors.subarray(0, n * 4), size: 4 };
      model.attrVersion = model.colorVersion;
    }
    return {
      length: n,
      attributes: {
        getPosition: { value: model.pos.subarray(0, n * 3), size: 3 },
        getFillColor: model.colorAttr,
        getRadius: { value: model.radii.subarray(0, n), size: 1 }
      }
    };
  };

  /**
   * 시각 index 의 열지도 데이터. 위치·가중치를 그 시각 기록으로 만든다(없으면 다음 시각, 그것도
   * 없으면 집 — writeGeometry 와 같은 규칙). 같은 시각·필터·모드면 같은 객체를 돌려줘야
   * deck.gl 이 가중치 텍스처를 다시 그리지 않는다. 최근 4개만 남긴다.
   */
  Sim3D.heatDataAt = function heatDataAt(index, heatMode) {
    if (!model) return { length: 0 };
    const i = clampFrame(index);
    const mode = heatMode || "spending";
    const hasNext = i + 1 < Math.min(model.F, loadedFrames());
    const key = i + "|" + model.visibleKey + "|" + mode + "|" + (hasNext ? 1 : 0);
    const hit = model.heatCache.get(key);
    if (hit) return hit;
    const N = model.N;
    const x = model.columns.x;
    const y = model.columns.y;
    const spent = model.columns.spent;
    const n = model.visibleCount;
    const positions = new Float64Array(n * 3);
    const weights = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      const a = model.visible[k];
      const ci = i * N + a;
      const present = isPresent(x, ci);
      let ref = present ? ci : NO_FRAME;
      if (!present && hasNext && isPresent(x, (i + 1) * N + a)) ref = (i + 1) * N + a;
      positions[k * 3] = present ? x[ci] : model.homeX[a];
      positions[k * 3 + 1] = present ? y[ci] : model.homeY[a];
      positions[k * 3 + 2] = 10;
      const amount = ref >= 0 ? spent[ref] : 0;
      weights[k] = (mode === "spending" ? Math.max(0.22, amount / 9000) : 1) * heatScale();
    }
    const data = { length: n, attributes: { getPosition: { value: positions, size: 3 }, getWeight: { value: weights, size: 1 } } };
    model.heatCache.set(key, data);
    while (model.heatCache.size > 4) model.heatCache.delete(model.heatCache.keys().next().value);
    return data;
  };

  Sim3D.heatData = function heatData() {
    if (!model) return { length: 0 };
    if (model.heatDataVersion !== model.heatVersion) {
      const n = model.visibleCount;
      // 히트맵은 데이터가 바뀌면 가중치 변환을 새로 만들고 텍스처 전체를 다시 그린다.
      // 시(hour) 단위로만 갱신해 그 비용을 프레임마다 치르지 않는다.
      model.heatData = {
        length: n,
        attributes: {
          getPosition: { value: model.heatPos.slice(0, n * 3), size: 3 },
          getWeight: { value: model.heatW.slice(0, n), size: 1 }
        }
      };
      model.heatDataVersion = model.heatVersion;
    }
    return model.heatData;
  };

  // 지난 windowSize 시간의 이동 궤적. 다음 시각 위치까지 넣어 두면 TripsLayer 의
  // currentTime 만 움직여도 꼬리 끝이 보간 중인 점을 따라간다.
  Sim3D.trailsData = function trailsData(windowSize) {
    if (!model) return { data: { length: 0 }, baseTime: 0 };
    const state = getState();
    updateVisible(state);
    const N = model.N;
    const size = Math.max(2, Math.trunc(finiteNumber(windowSize, 6)) || 6);
    const index = clampFrame(state.frameIndex);
    const hasNext = index + 1 < Math.min(model.F, loadedFrames());
    const key = index + "|" + size + "|" + model.visibleKey + "|" + (hasNext ? 1 : 0);
    if (key === model.trailKey && model.trails) return model.trails;

    const x = model.columns.x;
    const y = model.columns.y;
    const start = Math.max(0, index - size + 1);
    const span = index - start + 1 + (hasNext ? 1 : 0);
    const path = new Float64Array(model.visibleCount * span * 3);
    const times = new Float32Array(model.visibleCount * span);
    const starts = new Uint32Array(model.visibleCount);
    let vertex = 0;
    let paths = 0;
    for (let k = 0; k < model.visibleCount; k++) {
      const a = model.visible[k];
      const first = vertex;
      for (let f = start; f <= index; f++) {
        const slot = f * N + a;
        if (!isPresent(x, slot)) continue;
        path[vertex * 3] = x[slot];
        path[vertex * 3 + 1] = y[slot];
        path[vertex * 3 + 2] = 8;
        times[vertex] = f - start;
        vertex++;
      }
      if (hasNext && isPresent(x, index * N + a) && isPresent(x, (index + 1) * N + a)) {
        const slot = (index + 1) * N + a;
        path[vertex * 3] = x[slot];
        path[vertex * 3 + 1] = y[slot];
        path[vertex * 3 + 2] = 8;
        times[vertex] = index + 1 - start;
        vertex++;
      }
      if (vertex - first >= 2) {
        starts[paths++] = first;
      } else {
        vertex = first;
      }
    }
    model.trails = {
      data: {
        length: paths,
        startIndices: starts.slice(0, paths),
        attributes: {
          getPath: { value: path.slice(0, vertex * 3), size: 3 },
          getTimestamps: { value: times.slice(0, vertex), size: 1 }
        }
      },
      baseTime: index - start
    };
    model.trailKey = key;
    return model.trails;
  };

  function positionAt(agentIndex, index, t) {
    const N = model.N;
    const x = model.columns.x;
    const y = model.columns.y;
    const ci = index * N + agentIndex;
    const present = isPresent(x, ci);
    let cx = present ? x[ci] : model.homeX[agentIndex];
    let cy = present ? y[ci] : model.homeY[agentIndex];
    if (index + 1 < Math.min(model.F, loadedFrames())) {
      const ni = (index + 1) * N + agentIndex;
      if (isPresent(x, ni)) {
        cx += (x[ni] - cx) * t;
        cy += (y[ni] - cy) * t;
      }
    }
    return [cx, cy];
  }

  // 고른 대상자의 오늘 하루 경로(전체는 옅게, 지금까지 온 길은 진하게)와 현재 위치.
  Sim3D.selectionData = function selectionData() {
    const state = getState();
    if (!model || !state.selectedAgentId) return null;
    const agentIndex = model.indexById.get(String(state.selectedAgentId));
    if (agentIndex == null) return null;
    const index = clampFrame(state.frameIndex);
    const t = Math.max(0, Math.min(1, finiteNumber(state.frameT, 0)));
    const frames = model.pack.frames;
    const day = frames[index] ? frames[index].day : null;
    const routeKey = agentIndex + "|" + day + "|" + loadedFrames();
    if (routeKey !== model.routeKey) {
      const N = model.N;
      const x = model.columns.x;
      const y = model.columns.y;
      const points = [];
      const frameOfPoint = [];
      const limit = loadedFrames();
      for (let f = 0; f < limit; f++) {
        if (!frames[f] || frames[f].day !== day) continue;
        const slot = f * N + agentIndex;
        if (!isPresent(x, slot)) continue;
        points.push([x[slot], y[slot], 12]);
        frameOfPoint.push(f);
      }
      model.route = { points: points, frames: frameOfPoint, full: points.length > 1 ? [{ path: points }] : [] };
      model.routeKey = routeKey;
    }
    const head = positionAt(agentIndex, index, t);
    const travelled = [];
    for (let i = 0; i < model.route.points.length; i++) {
      if (model.route.frames[i] > index) break;
      travelled.push(model.route.points[i]);
    }
    travelled.push([head[0], head[1], 12]);
    return {
      agentIndex: agentIndex,
      route: model.route.full,
      stops: model.route.points,
      progress: travelled.length > 1 ? [{ path: travelled }] : [],
      head: head
    };
  };

  Sim3D.agentIndexOf = function agentIndexOf(agentId) {
    if (!model || agentId == null) return undefined;
    return model.indexById.get(String(agentId));
  };

  Sim3D.agentIdAtVisible = function agentIdAtVisible(visibleIndex) {
    if (!model || visibleIndex < 0 || visibleIndex >= model.visibleCount) return null;
    const agent = model.pack.agents[model.visible[visibleIndex]];
    return agent ? agent.id : null;
  };

  Sim3D.agentIndexAtVisible = function agentIndexAtVisible(visibleIndex) {
    if (!model || visibleIndex < 0 || visibleIndex >= model.visibleCount) return -1;
    return model.visible[visibleIndex];
  };

  Sim3D.agentPosition = function agentPosition(agentId) {
    const state = getState();
    const index = Sim3D.agentIndexOf(agentId);
    if (index == null) return null;
    return positionAt(index, clampFrame(state.frameIndex), Math.max(0, Math.min(1, finiteNumber(state.frameT, 0))));
  };

  // 한 대상자의 한 시각 기록을 원래 JSON 모양으로 되살린다 (툴팁·상세 카드용).
  Sim3D.agentFrame = function agentFrame(agentIndex, frameIndex) {
    if (!model || agentIndex == null || agentIndex < 0 || agentIndex >= model.N) return null;
    const f = frameIndex == null ? clampFrame(getState().frameIndex) : clampFrame(frameIndex);
    const slot = f * model.N + agentIndex;
    const cols = model.columns;
    if (!isPresent(cols.x, slot)) return null;
    function word(name) {
      const column = cols[name];
      const code = column ? column[slot] : 0;
      return code ? model.pack.dicts[name][code - 1] : null;
    }
    const sat = cols.sat ? cols.sat[slot] : NaN;
    return {
      lon: cols.x[slot],
      lat: cols.y[slot],
      cat: word("cat"),
      l1: word("l1"),
      dong: word("dong"),
      intent: word("intent"),
      anchor: word("anchor"),
      sat: sat === sat ? sat : null,
      spent: cols.spent ? cols.spent[slot] : 0
    };
  };

  Sim3D.frameInfo = function frameInfo(index) {
    if (!model) return null;
    return model.pack.frames[clampFrame(index)] || null;
  };

  Sim3D.frameCount = function frameCount() {
    return model ? model.F : 0;
  };

  Sim3D.loadedFrameCount = function loadedFrameCount() {
    return loadedFrames();
  };

  Sim3D.loadAgentDetail = function loadAgentDetail(agentId) {
    const index = Sim3D.agentIndexOf(agentId);
    if (!model || index == null) return Promise.resolve({});
    return model.pack.loadDetail(index);
  };

  Sim3D.hasAppointment = function hasAppointment(agentId) {
    const index = Sim3D.agentIndexOf(agentId);
    return index != null && model.appt[index] === 1;
  };

  Sim3D.computeMemoryTotals = function computeMemoryTotals() {
    const totals = (model && model.pack.totals) || {};
    return {
      totMem: finiteNumber(totals.visited, 0),
      totAppt: finiteNumber(totals.appointments, 0),
      totRumor: finiteNumber(totals.rumors, 0)
    };
  };

  // Aggregate the agents currently within `radiusMeters` of a clicked map point
  // and break them down by activity category. This is what powers the "why is
  // this spot red?" hotspot panel: rather than exposing the heatmap's internal
  // render thresholds (meaningless to a viewer), it re-counts the real agents +
  // spending right under the cursor, so two different hotspots inside the same
  // district report genuinely different compositions.
  //
  // The heatmap weights every agent (residents at home contribute a 0.22 floor,
  // out-and-about agents contribute their spending), so a blob can be red from
  // either residential density OR commercial spending. We therefore split the
  // tally: totalPresent = everyone in range, activeCount = those NOT anchored at
  // home, and the category breakdown covers only active agents -- letting the
  // panel say whether the heat is "사람들이 모여 소비 중" or just "주거 밀집".
  Sim3D.computeLocalStats = function computeLocalStats(lon, lat, radiusMeters) {
    const radius = finiteNumber(radiusMeters, 250);
    const empty = { totalPresent: 0, activeCount: 0, totalSpent: 0, categories: [], radiusMeters: radius };
    if (!model) return empty;
    Sim3D.prepareFrame();
    Sim3D.interpolate(getState().frameT);
    const cosLat = Math.cos((finiteNumber(lat, 37.55) * Math.PI) / 180);
    const cols = model.columns;
    const byCat = new Map();
    let totalPresent = 0;
    let activeCount = 0;
    let totalSpent = 0;

    for (let k = 0; k < model.visibleCount; k++) {
      const dx = (model.pos[k * 3] - lon) * 111320 * cosLat;
      const dy = (model.pos[k * 3 + 1] - lat) * 110540;
      if (Math.hypot(dx, dy) > radius) continue;

      totalPresent += 1;
      const ref = model.ref[k];
      const atHome = ref < 0 || (model.residenceCode > 0 && cols.anchor[ref] === model.residenceCode);
      if (atHome) continue;

      activeCount += 1;
      const spent = cols.spent[ref];
      totalSpent += spent;
      const l1Code = cols.l1 ? cols.l1[ref] : 0;
      const catCode = cols.cat ? cols.cat[ref] : 0;
      const category = l1Code ? model.pack.dicts.l1[l1Code - 1] : catCode ? model.pack.dicts.cat[catCode - 1] : "기타";
      const entry = byCat.get(category) || { cat: category, count: 0, spent: 0 };
      entry.count += 1;
      entry.spent += spent;
      byCat.set(category, entry);
    }

    const categories = Array.from(byCat.values()).sort(function (a, b) {
      return b.spent - a.spent || b.count - a.count;
    });

    return {
      totalPresent: totalPresent,
      activeCount: activeCount,
      totalSpent: totalSpent,
      categories: categories,
      radiusMeters: radius
    };
  };

  Sim3D.formatWon = formatWon;
  Sim3D.DIST_NAMES = DIST_NAMES;
  Sim3D.CAT_COLORS = CAT_COLORS;
  Sim3D.DIST_COLORS = DIST_COLORS;
})();
