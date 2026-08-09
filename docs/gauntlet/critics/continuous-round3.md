# Independent Critic Record — continuous round 3

계약/보안, report engine, UI 정보 구조, 실제 Chromium을 별도 평가 관점으로
재검증했다. 기준 미달 결과는 통과로 기록하지 않았다.

## 발견하고 수정한 결함

1. `run_id`가 이전에는 provenance 문자열에만 있었고 계산 원본 범위와 연결되지
   않았다. immutable source manifest를 추가하고 CLI 직전 hash 재검증으로 수정했다.
2. 현재 protected generator와 `origin/dasol`의 함수 시그니처가 달랐다. legacy
   direct-section path와 modern context-dispatch path를 명시적으로 분기하고,
   P009는 income DID 전용 함수로 dispatch하도록 수정했다.
3. P010은 effective date는 있지만 `income_grants`/`benefit_categories`가 없어
   origin 기준 sales DID 대상이 아니다. 메뉴를 disabled로 바꾸고 테스트를
   업데이트했다.
4. 완료 BASE에서 P008의 2026 날짜를 선택하면 2025 snapshot 밖이었다. UI가
   먼저 범위 경고와 disabled 상태를 표시하도록 고쳤다.
5. 초기 브라우저 시나리오가 rescue `BASE7500`만 보고 끝날 수 있었다. 실제
   completed `BASE`를 선택하고 P010을 선택하는 검증 경로를 추가했다.

## 독립 검증 결과

- `python -m unittest discover -s tests -p 'test_*.py' -v`: 28 passed.
- `npm run typecheck`, `npm run build`: exit 0.
- `npm run gauntlet:screen`: 2 passed, 1280×768/768×768, workspace 클릭,
  light→reload→dark, Leaflet iframe, overflow/console/page/network 검사 통과.
- `docs/gauntlet/evidence/s6-round3/interaction-manifest.json`: errors,
  failures, bad responses 모두 빈 배열.
- 실제 API: BASE 7/7, FINAL 28/28 source-ready; BASE7500 incomplete.
- 실제 CLI: manifest 검증 뒤 Neo4j password 부재로 exit 1, report HTML 미생성.
  이를 성공이나 숫자 일치로 주장하지 않았다.
- style audit: forbidden gradient/backdrop-filter/box-shadow/radius>4px 및 금지
  폰트 0건.

## 아직 거절/보류

- `NEO4J_PASSWORD`와 동일 snapshot임을 선언하는 `DASOL_NEO4J_RUN_ID`가 없는
  현재 환경에서는 DASOL job 생성이 503이어야 한다.
- in-app Browser 목록이 비어 있어 browser skill 연결 검증은 blocked다. 실제
  Chromium Playwright 증거로 대체했지만 같은 것으로 간주하지 않는다.
- Neo4j binding이 준비되기 전에는 생성 HTML의 숫자와 기존 reference report의
  숫자 일치 판정을 닫지 않는다.

판정: **S1/S2/S5/S6 contract·UI는 유지 통과, DASOL runtime은 conditional open.**
