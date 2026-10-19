# CLI feedback — active desktop round 4

## R004 PC 전용 구현 상태

- 기준 화면: 1440×900, 최소 화면: 1280×800. 모바일은 사용자 지시에 따라 이번 게이트에서 제외했다.
- 목록·4단계 설정·run 상세 5개 workspace를 새 `StudioShell`과 3계층 디자인 토큰으로 재구성했다.
- run 상세의 과도한 hero 여백을 운용형 객체 헤더로 압축하고 첫 판단 정보가 PC 첫 화면 안에 더 많이 들어오도록 조정했다.
- 에이전트 조사는 선택·대화/결정·기억/근거의 PC 3열 구조로 분리했다. API 계약이 없어 입력과 목록은 명시적으로 비활성화했고 가짜 데이터는 만들지 않았다.
- 시각화는 모든 날이 100%인 무의미한 진행률 차트를 없애고 실제 `elapsed_sec`, `agent_elapsed_sec`, `night2_elapsed_sec`를 사용한다. 표 대체도 함께 제공한다.
- DASOL 리포트는 분석 정의 → immutable snapshot/engine/run binding 검증 → job·로그·HTML의 PC 3열 작업공간으로 연결했다.
- report catalog의 HTML 목록이 서버 공용이라는 점을 확인했다. 공용 파일은 열지 않고 해당 run의 report job 산출물 경로와 정확히 일치할 때만 링크한다.
- HashRouter에서 skip link가 404 route를 만들던 문제를 수정했고, 필수 입력 오류·포커스 이동·semantic progressbar를 추가했다.

## R004 검증 증거

- 캡처: `docs/gauntlet/evidence/s7-redesign/wide-*.png`, `desktop-*.png`
- 시각 manifest: `docs/gauntlet/evidence/s7-redesign/visual-manifest.json`
- 접근성: `docs/gauntlet/evidence/s7-redesign/axe-wcag-report.json`
- `npm run build`: 통과, 2,221 modules transformed.
- 1440/1280 9개 주요 화면: 가로 오버플로 0, 중첩 세로 스크롤 0, 브라우저 console/page error 0.
- 설정 마법사: URL 전환, 자동 저장, P010 정책 선택, 검토 화면까지 실제 조작 통과.
- PC light/dark, reduced-motion, skip link 키보드 이동: 통과.
- light/dark 핵심 화면 Axe WCAG A·AA·AAA 태그: 위반 0. 에이전트·시각화·리포트 route도 포함했다.

## R004 다음 Critic이 계속 비판할 항목

- 1280px에서 리포트 3열의 텍스트 밀도와 열 너비가 판단 속도를 떨어뜨리지 않는지 다시 본다.
- run 정책 ID가 저장 설계에는 있지만 서버 산출물에는 없을 때 `연결 정보 없음`이 충분히 설명적인지 개선한다.
- 공용 report catalog와 run job의 명시적 서버측 관계 필드가 없으므로, 현재 클라이언트의 정확 경로 교집합보다 강한 provenance 계약이 필요하다.
- 에이전트 기억·대화 API가 구현되기 전 조사 workspace는 구조 검증만 가능하다. 실제 데이터가 연결된 뒤 검색·시간축·근거 이동의 조작 비용을 다시 비판한다.
- 시각화는 timing만 다루며 소비·이동·정책 효과 탐색은 아직 별도 분석 계약과 시각 체계가 필요하다.
- 현재 구현은 자동 테스트 통과를 종료로 간주하지 않는다. 독립 PC UX Critic과 Data-truth Critic 결과를 다음 수정 입력으로 반영한다.

---

# 이전 기록 — continuous round 3

## 이번 라운드

- Report boundary: 완료 run의 `summary/events/poi/metrics/timing/checkpoint`를
  SHA-256 immutable manifest로 고정하고 CLI 직전 재검증.
- DASOL adapter: P009 income DID dispatch와 origin/dasol modern
  `load_policy_ctx/run_section2/run_section3` bridge를 분리.
- Applicability: P010처럼 `income_grants`/`benefit_categories`가 없는 정책의
  sales DID 메뉴를 disabled로 전환.
- Results: report 시작일·기간이 실제 snapshot day 범위를 벗어나면 UI에서 먼저 차단.
- Browser evidence: `docs/gauntlet/evidence/s6-round3/`.

## 실제 클릭 경로

`#/monitor` → theme toggle(light) → reload(light 유지) → dark →
`일자·단계` → `실패·병목` → `원본 근거` → `#/results` → run을 `BASE`로 선택 →
policy를 `P010`으로 선택 → `보고서 생성` → 원본 엔진 미설정/버튼 disabled 확인 →
`3D 결과` → iframe Leaflet 렌더 확인 → `근거·부록`.

실제 source 비교:

- `BASE`: 7/7 completed, snapshot source 24개, `snapshot_id=2b14e04827229e6402c16dba`.
- `FINAL`: 28/28 completed.
- `BASE7500`: incomplete 1일, report 생성 거부.
- `docs/gauntlet/evidence/s6-round3/base-run-snapshot.json`은 실제 `out_BASE`
  원본을 검증한 manifest다.
- 기존 기준 산출물 `output/sim/report/FINAL_REPORT_5D_FULL.html/.md`는 참조로
  유지하고, 현재 CLI는 Neo4j 원본 연결이 없어 `exit 1`/HTML 미생성으로 끝났다.
  숫자 일치라고 주장하지 않았다.

## 다음 Critic이 다시 볼 항목

- `NEO4J_PASSWORD`와 `DASOL_NEO4J_RUN_ID=BASE`가 있는 환경에서 완료 `BASE` run의
  DASOL job을 실제 생성하고 queued/running/completed 로그·lock 해제·HTML/MD/
  snapshot artifact 열람을 확인할 것.
- 생성된 HTML의 provenance note/source links와 기존 report 숫자가 원본 계산
  함수·참조 자료와 일치하는지 비교할 것.
- in-app Browser가 제공되는 환경에서 동일 경로를 재검증할 것.

현재 라운드는 위 외부 조건을 충족했다고 가장하지 않고 조건부로 열어 둔다.
