# 설계도 — 3D 시각화 대시보드 연동

> 목표: **콘솔에서 선택한 시뮬레이션 run이 3D 지도에 그대로 반영된다.**
> 관련: `docs/DESIGN_SPEC_WEB_CONSOLE.md`(디자인 기준), `web/CONTRACT.md`(데이터 계약)

---

## 1. 기존 파이프라인 (실측)

```
Neo4j
  └─ export_visualization.py ──> agents.json / timeline.json / memories.json / events.json
                                      │
                                      ├─ derive.py::build_viz_meta ──> 파생 레이어
                                      │    (정책구역, 소비버스트, 만남, 소문엣지, 프레임요약)
                                      ▼
                              build_standalone_html.py
                                      │  template.html 의 3개 토큰을 치환
                                      │   __SIM_STYLES__  ← static/styles.css
                                      │   __SIM_DATA__    ← window.__AGENTS__ 등 5개 전역
                                      │   __SIM_SCRIPTS__ ← CDN 3종 + static JS 6개 인라인
                                      ▼
                              sim_standalone.html  (135~147MB)
```

**전역 데이터 규약** — 정적 JS가 기대하는 계약. 이것이 통합의 접합면이다.

| 전역 | 출처 |
|---|---|
| `window.__AGENTS__` | `agents.json` |
| `window.__TIMELINE__` | `timeline.json` |
| `window.__MEMORIES__` | `memories.json` |
| `window.__EVENTS__` | `events.json` |
| `window.__VIZ_META__` | `derive.build_viz_meta()` |

**스크립트 로드 순서 (고정)**: `data_model.js → map_scene.js → layers.js → detail.js → hud.js → app.js`
전부 `window.Sim3D` 네임스페이스에 등록하는 IIFE. `app.js`가 마지막에 부팅한다.

**외부 의존**: maplibre-gl 5.24, deck.gl 9.3.3, chart.js 4.5.1 — 현재는 빌드 시점에 CDN에서 받아 인라인.

### 지금 구조가 웹 연동에 안 맞는 이유

1. **135~147MB 단일 파일.** run마다 이걸 만들어 서빙하는 건 불가능하다. 브라우저가 파싱만으로 수 초를 쓴다
2. **데이터가 빌드 타임에 구워진다.** run을 바꾸려면 HTML을 다시 만들어야 한다
3. **run 개념이 없다.** `VIZ_OUT_DIR` 하나에 최신 export만 덮어쓴다. 어떤 run의 결과인지 파일에 안 적혀 있다

---

## 2. 설계 원칙

- **`scripts/sim/visualization_3d/` 를 수정하지 않는다.** 정적 JS 6개와 template.html은 그대로 둔다.
  통합은 **전역 5개를 채워주는 방식**으로만 한다. 시뮬레이션 엔진 코드에 손대지 않는다는 제약과 같은 선상이다
- **데이터를 빌드에서 분리한다.** HTML 껍데기는 고정, 데이터는 런타임에 fetch
- **iframe으로 격리한다.** 정적 JS는 전역(`window.Sim3D`, maplibre, deck.gl)을 점유하고 자체 CSS를 가진다.
  React 앱과 같은 문서에 넣으면 스타일·전역이 충돌한다. iframe이면 서로 모른다
- **선택 상태는 콘솔이 소유한다.** iframe은 "어떤 run/일자를 그릴지" 지시받는 쪽이다

---

## 3. 목표 구조

```
콘솔 (React)                          API (FastAPI)                 데이터
────────────                          ─────────────                 ──────
ResultsScreen
  run 선택 ──┐
             │
VisualizationScreen
  ├ 상단바: run·일자 표시, "결과로"
  ├ iframe  ──── postMessage ────┐
  │                              │
  └ 하단: 일자 스크러버          │
                                 ▼
                    /viz/shell.html  (고정 껍데기, ~200KB)
                      ├ static JS 6개 <script src>
                      ├ CDN 3종 <script src>
                      └ 부트로더
                           │
                           │ fetch
                           ▼
                    GET /api/runs/{runId}/viz/manifest
                    GET /api/runs/{runId}/viz/{day}/agents
                    GET /api/runs/{runId}/viz/{day}/timeline
                    GET /api/runs/{runId}/viz/{day}/events
                    GET /api/runs/{runId}/viz/{day}/memories
                    GET /api/runs/{runId}/viz/{day}/meta
                                 │
                                 ▼
                    viz_store/{runId}/{day}/*.json.gz
```

### 3.1 껍데기 `/viz/shell.html`

`template.html`을 복사해 만든다(원본 수정 금지). 차이는 세 곳뿐이다.

- `__SIM_STYLES__` → `<link rel="stylesheet" href="/viz/static/styles.css">`
- `__SIM_SCRIPTS__` → CDN 3종 + static JS 6개를 **`<script src>` 로**. 순서는 기존 그대로
- `__SIM_DATA__` → **부트로더 스크립트**로 교체

부트로더가 하는 일:

```js
// 1) 부모가 준 지시를 받는다
window.addEventListener('message', onCommand);   // {type:'load', runId, day}
// 2) 5개 전역을 채운다
window.__AGENTS__ = ...; window.__TIMELINE__ = ...;
window.__MEMORIES__ = ...; window.__EVENTS__ = ...; window.__VIZ_META__ = ...;
// 3) 그 다음에야 static JS 6개를 순서대로 주입한다
//    (app.js 가 부팅 시점에 전역을 읽으므로 순서가 뒤집히면 안 된다)
// 4) 준비되면 부모에게 알린다
parent.postMessage({type:'ready', runId, day, counts:{...}}, origin);
```

**핵심 제약**: `app.js`는 로드되는 순간 전역을 읽는다. 따라서 **데이터 fetch 완료 → 전역 할당 → 스크립트 주입** 순서를 반드시 지켜야 한다. 일자 전환 시에는 스크립트를 다시 주입하지 말고 iframe을 재생성하거나, `Sim3D`가 재초기화 API를 노출하는지 확인해 그걸 쓴다. **(미확인 — §7 참조)**

### 3.2 데이터 준비 `viz_store`

`export_visualization.py`는 run 개념이 없고 Neo4j 접속이 필요하다. 콘솔이 직접 호출하지 않는다.
대신 **오프라인 준비 단계**를 하나 둔다.

```
python web/api/tools/prepare_viz.py --run BASE --src <sim_output_dir> --out viz_store/
```

하는 일:
1. `VIZ_OUT_DIR`를 `viz_store/{runId}/`로 지정해 `export_visualization.py`를 **그대로 호출** (수정 없이 환경변수만 주입)
2. `derive.build_viz_meta()`를 호출해 `meta.json` 생성
3. **일자별로 쪼갠다** — `timeline.json`은 24h × N일이라 통째로 보내면 안 된다
4. gzip 압축 후 `viz_store/{runId}/{day}/` 에 배치
5. `manifest.json` 기록 — run id, 일자 목록, agent 수, 각 파일 크기, 생성 시각

**미확인**: 실제 JSON 용량. 현재 저장소에 `agents.json` 등이 없다(standalone HTML만 존재).
147MB HTML에서 CDN 라이브러리(~3MB)와 정적 JS를 빼면 데이터가 대부분이므로 **일자당 수십 MB 규모로 추정**된다.
prepare 단계에서 실측한 뒤 §5의 예산을 초과하면 다운샘플링을 건다. `scripts/sim/downsample_standalone.py`가 이미 있으니 그 로직을 참고한다(호출만, 수정 금지).

### 3.3 API

| 엔드포인트 | 응답 | 비고 |
|---|---|---|
| `GET /api/runs/{runId}/viz/manifest` | 일자 목록, agent 수, 준비 여부 | 없으면 `available:false` + `unknown:["viz_store"]` |
| `GET /api/runs/{runId}/viz/{day}/{kind}` | 해당 JSON (gzip) | kind ∈ agents/timeline/events/memories/meta |
| `GET /viz/shell.html` | 껍데기 | 캐시 가능 |
| `GET /viz/static/*` | 정적 JS·CSS | `scripts/sim/visualization_3d/static/` 을 **읽기 전용 마운트** |

정적 파일은 복사하지 않고 원본 디렉터리를 그대로 서빙한다. 원본이 갱신되면 자동 반영되고, 사본이 어긋날 일이 없다.

### 3.4 콘솔 ↔ iframe 프로토콜

부모 → iframe:

| type | payload | 의미 |
|---|---|---|
| `load` | `{runId, day}` | 이 run/일자를 그려라 |
| `setDay` | `{day}` | 일자만 바꿔라 |
| `setLayer` | `{heatmap, trails, boundary}` | 레이어 토글 |

iframe → 부모:

| type | payload | 의미 |
|---|---|---|
| `ready` | `{runId, day, counts}` | 렌더 완료 |
| `progress` | `{phase, pct}` | 로딩 진행 (스켈레톤 갱신용) |
| `error` | `{code, message}` | 실패 — 부모가 에러 상태 표시 |
| `select` | `{agentId}` | 사용자가 agent를 골랐다 |

`postMessage`의 `targetOrigin`을 같은 출처로 고정하고, 수신 측에서 `event.origin`을 검증한다.

---

## 4. 화면 동작

`docs/DESIGN_SPEC_WEB_CONSOLE.md` §7 "시각화 페이지 진입 흐름"을 그대로 따른다. 데이터 연동만 얹는다.

1. 결과 화면에서 run 선택 → "지도에서 열기"
2. `/visualize?run={runId}&day={day}` 로 이동 (딥링크)
3. manifest 조회 → **준비 안 됐으면 지도를 그리지 않고 안내**:
   "이 실행은 아직 지도 데이터가 준비되지 않았습니다" + 준비 방법 안내.
   빈 지도를 보여주는 것보다 낫다 (§8 `empty-states`)
4. 준비됐으면 iframe에 `load` 전송, 스켈레톤 표시
5. `ready` 수신 → 스켈레톤 해제
6. 하단 스크러버로 일자 이동 → `setDay`. **URL도 함께 갱신**해 공유·뒤로가기가 맞게 동작
7. `error` 수신 → 지도 자리에 에러 + 재시도 (§8 `error-recovery`)

run이 불완전한 경우(`rescue` 같은 중단 run)는 **있는 일자만 스크러버에 노출**하고, 없는 구간은 비활성 + 이유를 표시한다. 계약의 "모르는 값에 0을 넣지 않는다" 원칙과 같다.

---

## 5. 성능 예산

| 항목 | 예산 | 근거 |
|---|---|---|
| 껍데기 + 정적 JS | 300KB | 캐시됨. 최초 1회 |
| CDN 3종 | ~3MB | 캐시됨 |
| 일자 1개 데이터 (gzip 후) | **8MB 이하** | 초과 시 다운샘플링 |
| 첫 지도 렌더 | 3초 이내 | B4(2초)는 콘솔 화면 기준. 지도는 별도 예산 |
| 일자 전환 | 1.5초 이내 | 인접 일자 프리페치로 흡수 |

- 일자별 응답은 `Cache-Control: immutable` (완료된 run은 변하지 않는다)
- 현재 일자 로드 후 **다음 일자를 백그라운드 프리페치**
- 데이터가 예산을 넘으면 prepare 단계에서 agent 표본을 줄인다. 런타임에 줄이지 않는다

---

## 6. 단계

| 단계 | 산출물 | 검증 |
|---|---|---|
| V1 | `prepare_viz.py` + `viz_store` 생성, 실제 용량 실측 | BASE run으로 manifest·일자별 파일 생성 확인 |
| V2 | API 4종 + 정적 마운트 | curl로 manifest·일자 데이터 응답 확인 |
| V3 | `shell.html` + 부트로더 | 브라우저에서 단독으로 열어 지도가 뜨는지 |
| V4 | 콘솔 iframe 연동 + postMessage | run 선택 → 지도 반영, 일자 전환, 딥링크 |
| V5 | 미준비·중단 run·에러 상태 | `rescue` run으로 안내 문구 확인 |

**V1이 선행이다.** 실제 용량을 모른 채 V2~V4를 설계하면 예산이 틀어져 재작업이 난다.

---

## 7. 미확인 / 결정 필요

1. **JSON 실제 용량** — 저장소에 없다. V1에서 실측
2. **`Sim3D` 재초기화 가능 여부** — 일자 전환 때 iframe을 통째로 다시 만들지, 내부 API로 갱신할지가 갈린다.
   `app.js`를 읽어 부팅 함수가 노출되는지 확인해야 한다. 없으면 iframe 재생성(간단하지만 깜빡임)
3. **CDN 접근성** — 배포 대상(EC2 `43.201.218.176`)에서 unpkg·jsdelivr가 열리는지. 막히면 라이브러리를 로컬에 벤더링
4. **Neo4j 의존** — `export_visualization.py`는 Neo4j가 필요하다. prepare를 **GPU 서버에서 돌리고 결과만 가져올지**, 콘솔 호스트에 Neo4j를 두는지 결정 필요
5. **좌표 데이터** — `events.jsonl`의 `dong`이 100% null임이 이미 확인됐다. 지도에 동 단위 레이어를 그릴 수 없다. 구 단위까지만 가능
