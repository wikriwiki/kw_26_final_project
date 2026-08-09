# Continuous Builder Record — round 2

기준 입력은 `docs/GAUNTLET_WEB_CONSOLE.md` 하나로 고정했다. 구현은 S1 계약을
깨지 않는 범위에서 S5/S6 UI와 DASOL 연결을 실제 코드로 확장했다.

## 구현 단위

- 결과 화면을 `보고서 허브 / 보고서 생성 / 보고서 열람 / 3D 결과 / 근거·부록`의
  다섯 workspace로 분리하고, run·policy·workspace breadcrumb를 유지했다.
- Monitor를 `개요 / 일자·단계 / 실패·병목 / 원본 근거`로 분리했다. 선택한
  workspace에 필요한 API만 요청하고, 나머지 판넬은 CSS로 숨겨 mega-dashboard를
  피했다.
- `data-theme="light|dark"`와 localStorage 저장, 명시적 상단 토글을 추가했다.
- `scripts/report/catalog.py`, `menu.py`, `narrate.py`, `build_html.py`를 추가해
  보호된 `scripts/sim/generate_final_report.py`를 수정하지 않고 기존 계산/렌더러를
  구조화된 DASOL job으로 연결했다.
- `web/api/report_jobs.py`와 `/api/reports/*`를 추가했다. 완료 run만 허용하고,
  runner lock/report lock/출력 경로/분석 applicability를 서버에서 검증한다.
- Neo4j 원본 설정이 없을 때는 생성 job을 만들지 않고 정확한 이유를 표시한다.
  fixture로 실제 보고서를 가장하지 않는다.
- 레거시 `visualization/index.html`이 현재 snapshot에 없는 네 개 JSON을 요청하는
  사실을 확인하고, 산출물 목록은 보존하되 기본 3D viewer는
  self-contained `visualization/sim_standalone_fast.html`로 선택했다.

## 검증 입력과 산출물

- `tests/unit/test_report_api.py` — catalog/applicability, incomplete snapshot,
  runner lock, fixture refusal.
- `web/ui/e2e/gauntlet-screen.spec.ts` — 1280×768·768×768, theme reload,
  Monitor/Results workspace 클릭, iframe 내부 Leaflet DOM, console/network/
  overflow 검사.
- `docs/gauntlet/evidence/s6-round2/manifest.json`
- `docs/gauntlet/evidence/s6-round2/interaction-manifest.json`
- `docs/gauntlet/evidence/s6-round2/interaction-wide-results-map.png`
- `docs/gauntlet/evidence/s6-round2/interaction-wide-results-generate.png`
- `docs/gauntlet/evidence/s6-round2/interaction-compact-results-generate.png`

## 보호 범위

`scripts/sim` 및 `scripts/neo4j_load`의 기존 엔진은 수정하지 않았다. 보고서
어댑터는 기존 함수 호출만 감싼다.
