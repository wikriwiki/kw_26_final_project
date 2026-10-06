(function () {
  const Sim3D = (window.Sim3D = window.Sim3D || {});

  // 시각화 팩(scripts/sim/visualization_3d/pack.py)을 읽어 열 단위 typed array 로 펼친다.
  // 입력은 세 가지다.
  //   1) 인라인 팩  — 단일 HTML 에 manifest(window.__SIM_PACK__)와 base64 gzip 블롭을 담은 경우
  //   2) URL 팩     — 서버가 manifest.json / *.gz 를 내려 주는 경우 (?pack=<경로> 또는 기본 pack/)
  //   3) 예전 전역  — window.__TIMELINE__ 등 JSON 전역을 채운 경우 (같은 구조로 변환)
  // 어느 쪽이든 화면 코드는 같은 pack 객체만 본다.

  const PACK_FORMAT = "sim-viz-pack";
  const PACK_VERSION = 1;
  const TYPED_ARRAYS = {
    float32: Float32Array,
    int32: Int32Array,
    uint8: Uint8Array,
    uint16: Uint16Array,
    uint32: Uint32Array
  };
  const CHUNK_CONCURRENCY = 3;
  const DETAIL_CACHE_SHARDS = 8;

  function assertLittleEndian() {
    if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
      throw new Error("big-endian 환경은 지원하지 않습니다");
    }
  }

  function isGzip(bytes) {
    return bytes.length > 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
  }

  async function gunzip(bytes) {
    // 서버가 Content-Encoding 으로 이미 풀어 보냈으면 그대로 쓴다.
    if (!isGzip(bytes)) return bytes;
    if (typeof DecompressionStream !== "function") {
      throw new Error("이 브라우저는 압축 해제(DecompressionStream)를 지원하지 않습니다. 최신 Chrome·Edge·Firefox·Safari 로 열어 주세요.");
    }
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  async function base64ToBytes(text) {
    const clean = String(text || "").replace(/\s+/g, "");
    if (typeof Uint8Array.fromBase64 === "function") return Uint8Array.fromBase64(clean);
    try {
      const response = await fetch("data:application/octet-stream;base64," + clean);
      return new Uint8Array(await response.arrayBuffer());
    } catch (error) {
      const binary = atob(clean);
      const out = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
      return out;
    }
  }

  async function sha256Hex(bytes) {
    if (!window.crypto || !window.crypto.subtle) return null;
    const digest = await window.crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest), function (b) {
      return b.toString(16).padStart(2, "0");
    }).join("");
  }

  async function verifiedBytes(entry, raw) {
    // 해시는 gzip 파일 그대로에 대해 기록돼 있다. 서버가 풀어서 보낸 경우는 비교할 수 없다.
    if (entry.sha256 && isGzip(raw)) {
      const actual = await sha256Hex(raw);
      if (actual && actual !== entry.sha256) {
        throw new Error("팩 파일 해시가 manifest 와 다릅니다: " + entry.file + " (캐시를 비우거나 팩을 다시 만드세요)");
      }
    }
    return gunzip(raw);
  }

  function decodeText(bytes) {
    return new TextDecoder("utf-8").decode(bytes);
  }

  function resolvePackUrl() {
    const params = new URLSearchParams(window.location.search);
    const candidate = window.__SIM_PACK_URL__ || params.get("pack") || "pack/";
    const url = new URL(candidate, window.location.href);
    if (window.location.protocol !== "file:" && url.origin !== window.location.origin) {
      throw new Error("다른 출처의 데이터 팩은 열지 않습니다: " + candidate);
    }
    if (!url.pathname.endsWith("/")) url.pathname += "/";
    url.search = "";
    return url;
  }

  function inlineSource() {
    const blobs = new Map();
    document.querySelectorAll('script[type="application/octet-stream"][data-pack-file]').forEach(function (tag) {
      blobs.set(tag.getAttribute("data-pack-file"), tag);
    });
    return {
      kind: "inline",
      bytes: async function bytes(entry) {
        const tag = blobs.get(entry.file);
        if (!tag) throw new Error("HTML 안에 팩 파일이 없습니다: " + entry.file);
        return verifiedBytes(entry, await base64ToBytes(tag.textContent));
      }
    };
  }

  function urlSource(baseUrl) {
    return {
      kind: "url",
      baseUrl: baseUrl.href,
      bytes: async function bytes(entry) {
        const url = new URL(entry.file, baseUrl);
        // 파일 해시를 주소에 붙여 팩을 다시 만들면 캐시가 저절로 갈리게 한다.
        if (entry.sha256) url.searchParams.set("v", entry.sha256.slice(0, 16));
        const response = await fetch(url.href, { credentials: "same-origin" });
        if (!response.ok) throw new Error("팩 파일을 받지 못했습니다 (" + response.status + "): " + entry.file);
        return verifiedBytes(entry, new Uint8Array(await response.arrayBuffer()));
      }
    };
  }

  function checkManifest(manifest) {
    if (!manifest || manifest.format !== PACK_FORMAT) throw new Error("시각화 팩 manifest 가 아닙니다");
    if (manifest.version !== PACK_VERSION) {
      throw new Error("지원하지 않는 팩 버전입니다: " + manifest.version + " (뷰어는 " + PACK_VERSION + ")");
    }
    if (!(manifest.agent_count > 0) || !(manifest.frame_count > 0)) throw new Error("팩에 대상자나 프레임이 없습니다");
  }

  function allocateColumns(columnSpecs, size) {
    const columns = {};
    Object.keys(columnSpecs).forEach(function (name) {
      const Typed = TYPED_ARRAYS[columnSpecs[name].dtype];
      if (!Typed) throw new Error("알 수 없는 열 형식: " + columnSpecs[name].dtype);
      columns[name] = new Typed(size);
    });
    if (columns.x) columns.x.fill(NaN);
    if (columns.y) columns.y.fill(NaN);
    return columns;
  }

  function decodeChunkInto(pack, chunk, bytes) {
    const N = pack.N;
    const expected = chunk.frame_count * N;
    Object.keys(chunk.offsets).forEach(function (name) {
      const target = pack.columns[name];
      if (!target) return;
      const spec = chunk.offsets[name];
      const Typed = TYPED_ARRAYS[spec.dtype];
      if (spec.length !== expected) throw new Error("청크 열 길이가 맞지 않습니다: " + chunk.file + " " + name);
      const byteOffset = bytes.byteOffset + spec.offset;
      const byteLength = spec.length * Typed.BYTES_PER_ELEMENT;
      if (spec.offset + byteLength > bytes.byteLength) throw new Error("청크가 잘렸습니다: " + chunk.file);
      const view = byteOffset % Typed.BYTES_PER_ELEMENT === 0
        ? new Typed(bytes.buffer, byteOffset, spec.length)
        : new Typed(bytes.buffer.slice(byteOffset, byteOffset + byteLength));
      target.set(view, chunk.frame_start * N);
    });
  }

  function contiguousLoadedFrames(pack) {
    let frames = 0;
    for (let i = 0; i < pack.chunks.length; i++) {
      if (!pack.chunkLoaded[i]) break;
      frames = pack.chunks[i].frame_start + pack.chunks[i].frame_count;
    }
    return frames;
  }

  function createPack(source, manifest, agents) {
    const N = manifest.agent_count;
    const F = manifest.frame_count;
    if (!Array.isArray(agents) || agents.length !== N) {
      throw new Error("대상자 수가 manifest 와 다릅니다 (" + (agents ? agents.length : 0) + " / " + N + ")");
    }
    const pack = {
      source: source,
      manifest: manifest,
      agents: agents,
      N: N,
      F: F,
      frames: manifest.frames,
      dicts: manifest.dicts,
      columns: allocateColumns(manifest.columns, N * F),
      meta: manifest.meta || {},
      totals: manifest.totals || {},
      appointmentAgents: manifest.appointment_agents || [],
      chunks: manifest.chunks,
      chunkLoaded: new Uint8Array(manifest.chunks.length),
      loadedFrames: 0,
      detailCache: new Map()
    };

    pack.loadTimeline = function loadTimeline(onProgress) {
      let next = 0;
      async function worker() {
        while (next < pack.chunks.length) {
          const index = next++;
          const chunk = pack.chunks[index];
          const bytes = await source.bytes(chunk);
          decodeChunkInto(pack, chunk, bytes);
          pack.chunkLoaded[index] = 1;
          pack.loadedFrames = contiguousLoadedFrames(pack);
          if (typeof onProgress === "function") onProgress(pack, index);
        }
      }
      const workers = [];
      for (let i = 0; i < Math.min(CHUNK_CONCURRENCY, pack.chunks.length); i++) workers.push(worker());
      return Promise.all(workers).then(function () {
        return pack;
      });
    };

    pack.loadDetail = function loadDetail(agentIndex) {
      const detail = manifest.detail || {};
      if (detail.omitted) {
        return Promise.reject(new Error("이 파일에는 대상자별 기억·일정 기록을 넣지 않았습니다"));
      }
      const shardSize = detail.shard_size || 1;
      const shardIndex = Math.floor(agentIndex / shardSize);
      const shard = (detail.shards || [])[shardIndex];
      if (!shard) return Promise.resolve({});
      let promise = pack.detailCache.get(shardIndex);
      if (promise) {
        // 최근에 쓴 묶음을 뒤로 옮겨 LRU 로 유지한다.
        pack.detailCache.delete(shardIndex);
      } else {
        promise = source.bytes(shard).then(function (bytes) {
          return JSON.parse(decodeText(bytes));
        });
        promise.catch(function () {
          pack.detailCache.delete(shardIndex);
        });
      }
      pack.detailCache.set(shardIndex, promise);
      while (pack.detailCache.size > DETAIL_CACHE_SHARDS) {
        pack.detailCache.delete(pack.detailCache.keys().next().value);
      }
      const agentId = agents[agentIndex] && agents[agentIndex].id;
      return promise.then(function (payload) {
        return (payload && payload.agents && payload.agents[agentId]) || {};
      });
    };

    return pack;
  }

  Sim3D.openPack = async function openPack() {
    assertLittleEndian();
    if (window.__SIM_PACK__ && window.__SIM_PACK__.format === PACK_FORMAT) {
      const manifest = window.__SIM_PACK__;
      checkManifest(manifest);
      const source = inlineSource();
      const agents = Array.isArray(window.__AGENTS__)
        ? window.__AGENTS__
        : JSON.parse(decodeText(await source.bytes(manifest.agents)));
      return createPack(source, manifest, agents);
    }
    if (Array.isArray(window.__TIMELINE__)) {
      return Sim3D.packFromLegacy({
        agents: window.__AGENTS__,
        timeline: window.__TIMELINE__,
        memories: window.__MEMORIES__,
        events: window.__EVENTS__,
        meta: window.__VIZ_META__
      });
    }
    const base = resolvePackUrl();
    const response = await fetch(new URL("manifest.json", base).href, { cache: "no-cache", credentials: "same-origin" });
    if (!response.ok) throw new Error("데이터 팩 manifest 를 받지 못했습니다 (" + response.status + "): " + base.href);
    const manifest = await response.json();
    checkManifest(manifest);
    const source = urlSource(base);
    const agents = JSON.parse(decodeText(await source.bytes(manifest.agents)));
    return createPack(source, manifest, agents);
  };

  // 예전 JSON 전역(window.__AGENTS__ 등)을 같은 팩 구조로 바꾼다. 규칙은 pack.py 와 같다.
  Sim3D.packFromLegacy = function packFromLegacy(input) {
    const timeline = Array.isArray(input.timeline) ? input.timeline : [];
    if (!timeline.length) throw new Error("timeline data is empty");
    const memories = input.memories || {};
    const events = input.events || {};
    const meta = input.meta || {};

    const agents = [];
    const indexById = new Map();
    function addAgent(agent) {
      const id = String(agent.id);
      if (indexById.has(id)) return;
      indexById.set(id, agents.length);
      agents.push(Object.assign({}, agent, { id: id }));
    }
    (Array.isArray(input.agents) ? input.agents : []).forEach(function (agent) {
      if (agent && agent.id != null) addAgent(agent);
    });
    timeline.forEach(function (frame) {
      (frame && frame.agents ? frame.agents : []).forEach(function (item) {
        if (item && item.id != null) addAgent({ id: item.id });
      });
    });

    const N = agents.length;
    const F = timeline.length;
    const dictNames = ["cat", "l1", "dong", "intent", "anchor"];
    const dicts = {};
    const codes = {};
    dictNames.forEach(function (name) {
      dicts[name] = [];
      codes[name] = new Map();
    });
    const columnSpecs = {
      x: { dtype: "float32" }, y: { dtype: "float32" }, sat: { dtype: "float32" }, spent: { dtype: "int32" }
    };
    dictNames.forEach(function (name) {
      columnSpecs[name] = { dtype: "uint32", dict: name };
    });
    const columns = allocateColumns(columnSpecs, N * F);

    function codeFor(name, value) {
      if (value == null) return 0;
      const text = String(value);
      if (text === "") return 0;
      let code = codes[name].get(text);
      if (code == null) {
        dicts[name].push(text);
        code = dicts[name].length;
        codes[name].set(text, code);
      }
      return code;
    }
    function f32(value) {
      if (value == null || typeof value === "boolean") return NaN;
      const number = Number(value);
      return Number.isFinite(number) ? number : NaN;
    }

    const summaries = [];
    timeline.forEach(function (frame, f) {
      const items = frame && Array.isArray(frame.agents) ? frame.agents : [];
      let spendTotal = 0;
      let spenders = 0;
      items.forEach(function (item) {
        if (!item || item.id == null) return;
        const slot = f * N + indexById.get(String(item.id));
        columns.x[slot] = f32(item.lon);
        columns.y[slot] = f32(item.lat);
        columns.sat[slot] = f32(item.sat);
        const spent = Number(item.spent);
        columns.spent[slot] = Number.isFinite(spent) ? Math.round(spent) : 0;
        if (columns.spent[slot] > 0) {
          spendTotal += columns.spent[slot];
          spenders += 1;
        }
        dictNames.forEach(function (name) {
          columns[name][slot] = codeFor(name, item[name]);
        });
      });
      summaries.push({
        day: frame.day, hour: frame.hour, label: frame.label, active_agents: items.length,
        spend_total: spendTotal, spenders: spenders
      });
    });

    let visited = 0;
    let appointments = 0;
    let rumors = 0;
    const appointmentAgents = [];
    agents.forEach(function (agent, index) {
      const bundle = memories[agent.id];
      if (!bundle) return;
      visited += (bundle.visited || []).length;
      appointments += (bundle.appointments || []).length;
      rumors += (bundle.memories || []).filter(function (item) {
        return item && item.type === "rumor";
      }).length;
      if ((bundle.appointments || []).length) appointmentAgents.push(index);
    });

    const frames = timeline.map(function (frame, index) {
      return {
        day: frame.day == null ? null : String(frame.day),
        hour: frame.hour,
        label: frame.label || "Frame " + index,
        day_idx: frame.day_idx
      };
    });
    const frameSummaries = Array.isArray(meta.frame_summaries) && meta.frame_summaries.length === F
      ? meta.frame_summaries
      : summaries;
    let firstActive = 0;
    for (let i = 0; i < frameSummaries.length; i++) {
      if (frameSummaries[i] && frameSummaries[i].active_agents) {
        firstActive = i;
        break;
      }
    }

    const manifest = {
      format: PACK_FORMAT, version: PACK_VERSION, pack_id: null, label: null, run_id: null,
      agent_count: N, frame_count: F, frames: frames, first_active_frame: firstActive,
      columns: columnSpecs, dicts: dicts, chunks: [{ file: "(legacy)", frame_start: 0, frame_count: F, offsets: {} }],
      detail: { shard_size: N, shards: [] },
      totals: { visited: visited, appointments: appointments, rumors: rumors },
      appointment_agents: appointmentAgents,
      meta: Object.assign({}, meta, { frame_summaries: frameSummaries })
    };

    const pack = {
      source: { kind: "legacy" },
      manifest: manifest,
      agents: agents,
      N: N,
      F: F,
      frames: frames,
      dicts: dicts,
      columns: columns,
      meta: manifest.meta,
      totals: manifest.totals,
      appointmentAgents: appointmentAgents,
      chunks: manifest.chunks,
      chunkLoaded: new Uint8Array([1]),
      loadedFrames: F,
      detailCache: new Map()
    };
    pack.loadTimeline = function loadTimeline(onProgress) {
      if (typeof onProgress === "function") onProgress(pack, 0);
      return Promise.resolve(pack);
    };
    pack.loadDetail = function loadDetail(agentIndex) {
      const id = agents[agentIndex] && agents[agentIndex].id;
      const bundle = {};
      if (memories[id]) bundle.memories = memories[id];
      if (Array.isArray(events[id])) bundle.events = events[id];
      return Promise.resolve(bundle);
    };
    return pack;
  };
})();
