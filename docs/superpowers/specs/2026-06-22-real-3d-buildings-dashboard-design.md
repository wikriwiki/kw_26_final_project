# 실제 3D 도시 건물 위 에이전트 시각화 (sim_standalone)

**날짜:** 2026-06-22
**대상:** `output/sim/visualization/sim_standalone.html` (빌더: `scripts/sim/build_standalone_html.py`)
**참조:** MapLibre 예제 — https://maplibre.org/maplibre-gl-js/docs/examples/display-buildings-in-3d/

## 목표

MapLibre "Display buildings in 3D" 예제 방식을 적용해, `sim_standalone.html`의
지도 배경을 **실제 도시 건물(OpenFreeMap 벡터 타일)** 로 바꾼다. 그 위에서 기존
deck.gl 에이전트/지출/상호작용 레이어가 그대로 움직인다. 합성 POI 박스는 제거하고,
전체 톤은 기존 야경 네온 컨셉에 맞춰 **어두운 커스텀 스타일**로 간다.

## 핵심 결정 (확정)

- **건물 소스:** OpenFreeMap (API 키 불필요, 무료, 온라인). 예제와 동일한
  `openfreemap` 벡터 소스(`https://tiles.openfreemap.org/planet`)의 `building`
  source-layer를 `fill-extrusion`으로 렌더.
- **베이스맵 톤:** 어두운 커스텀 스타일. OpenFreeMap의 기성 밝은 스타일 대신,
  같은 벡터 소스를 참조하는 **최소 다크 스타일을 직접 작성**해 야경 톤을 통제한다.
- **합성 POI 박스(`building_features`) 제거:** 실제 건물이 도시 맥락을 채우므로
  중복/시각적 충돌을 피하기 위해 지도 렌더에서 제거. 지출/방문 데이터는 기존
  deck.gl 지출 컬럼·점과 detail 카드에서 계속 표현.
- **트레이드오프 수용:** 지도 타일은 온라인 의존(인터넷 필요). 임베드된 JS
  (maplibre/deck/chart)와 시뮬레이션 데이터는 오프라인 유지. 타일 로드 실패 시
  어두운 빈 배경으로 graceful fallback.

## 아키텍처 (변경 범위)

기존 빌드 파이프라인(`build_standalone_html.py` → `template.html` + static JS +
임베드 JSON)은 **그대로 유지**. 실질 변경은 두 파일에 집중된다.

### 1. `scripts/sim/visualization_3d/static/map_scene.js` (주 변경)

- `buildingFeatures()` / 합성 `cityBuildings` 소스·`city-buildings` 레이어 제거.
- `styleModeMap()` → `darkCityStyle()`로 교체. 다음을 포함하는 다크 스타일 반환:
  - `sources.openfreemap`: `{ type: "vector", url: "https://tiles.openfreemap.org/planet" }`
  - `glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf"`
  - 레이어 순서(아래→위):
    1. `background` — 다크 (`#0a0e16`)
    2. `water` — 어두운 청록 (`openfreemap` source-layer `water`)
    3. (선택) `landuse`/`road` 최소 표현, 낮은 명도 — 야경 거리 느낌
    4. `building-3d` — `fill-extrusion`, source-layer `building`:
       - filter: `["!=", ["get", "hide_3d"], true]`
       - color: `render_height` 보간, 다크 네온 톤
         (예: `0 → #14202e`, `120 → #1f4a5e`, `300 → #2e7da0`)
       - height: zoom 보간으로 `render_height` (뷰 줌 범위에 맞게 임계값
         조정, 예제의 15→16 대신 **13.5→14.5** 권장: 더 많은 챕터에서 건물 보임)
       - base: zoom ≥ 14 일 때 `render_min_height`, 아니면 0
       - opacity ~0.85, `fill-extrusion-vertical-gradient: true`
- 다크 스타일에는 심볼/라벨 레이어가 (거의) 없으므로 deck 오버레이는 건물 위에
  자연스럽게 올라간다. 라벨을 추가할 경우 building-3d를 첫 심볼 레이어 아래에
  삽입(예제의 labelLayerId 로직).
- `switchBaseMode("style")`가 `darkCityStyle()`을 사용하도록 업데이트.
  Google 3D 타일 모드 분기는 기존대로 유지.
- 카메라 기본값/챕터 줌은 유지하되, 거리 단위 챕터(`policy`/`interaction`)에서
  건물이 확실히 보이도록 줌이 13.5 이상인지 점검(필요 시 소폭 상향).

### 2. `scripts/sim/visualization_3d/derive.py`

- `build_viz_meta`의 `"building_features": _building_features(...)` 항목 제거.
- `_building_features()` 및 그 전용 헬퍼(`_stable_angle` 등 다른 곳에서
  안 쓰이면) 제거하여 임베드 데이터 용량 축소.

### 3. (영향 없음 확인) 나머지

- `layers.js`의 deck.gl 에이전트/지출/상호작용 레이어 변경 없음.
- `hud.js` / `detail.js` / `app.js` / `data_model.js`는 `building_features`를
  참조하지 않으므로 변경 없음(grep로 확인됨).
- `assets.py`: 신규 외부 CDN 없음(OpenFreeMap은 런타임 fetch). 변경 불필요.

## 데이터 흐름

빌드 시: 시뮬레이션 JSON → `derive.py`(building_features 제외) → 임베드.
런타임: 브라우저가 OpenFreeMap 스타일/타일/폰트를 fetch → MapLibre가 실제
건물을 `fill-extrusion`으로 렌더 → deck.gl 오버레이(interleaved)가 그 위에
에이전트 점/경로/지출/상호작용을 프레임마다 갱신.

## 알려진 고려사항 / 검증 포인트

- **오클루전(z-order):** `interleaved: true`에서 낮은 고도의 에이전트 점이 높은
  건물에 가려질 수 있음. 빌드 후 스크린샷으로 확인하고, 필요 시 (a) 점 고도
  상향, (b) 건물 opacity 하향, (c) interleaved 조정 중 택1.
- **줌-건물 가시성:** 도시 개요 줌에서는 건물 미표시(정상). 거리 줌에서 표시.
- **오프라인:** 타일 fetch 실패 시 다크 배경만 남고 에이전트는 계속 동작해야 함
  (map `error` 핸들링 + 사용자 안내 배너).

## 검증

1. `python -m scripts.sim.build_standalone_html` 로 재빌드, 파일 생성 확인.
2. 브라우저로 열어: (a) 실제 건물 3D 표시, (b) 어두운 톤, (c) 에이전트 이동,
   (d) Play/슬라이더/검색/detail 동작, (e) 합성 박스 사라짐 확인.
3. 기존 `verify_visualization_3d.py` / playwright probe로 회귀 점검.
