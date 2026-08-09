# 병렬 디자인 건틀렛 루프 핸드오프

이 문서는 CLI와 Codex Desktop이 같은 웹 콘솔을 동시에 발전시킬 때 사용하는 현재 상태 포인터다. 자동 테스트 통과나 점수를 완료 조건으로 사용하지 않는다. 매 라운드는 기존 화면에서 다음 미흡점을 찾아 다음 라운드 입력으로 넘긴다.

```yaml
loop_id: DESIGN-001
owner: codex-desktop
mode: persistent-resumable-rounds
current_round: R004
status: active
target_surface: pc-only
primary_viewport: 1440x900
minimum_viewport: 1280x800
mobile_gate: excluded-by-user
implementation_worktree: C:\Users\srdyh\OneDrive\사진\바탕 화면\agent_simulate\kw_26_final_project\.claude\worktrees\silly-gagarin-0b181a
implementation_branch: feat/web-console
previous_design_worktree: C:\Users\srdyh\OneDrive\사진\바탕 화면\agent_simulate\kw_26_final_project\.claude\worktrees\codex-gauntlet-design
latest_evidence: docs/gauntlet/evidence/s7-redesign
latest_accessibility_report: docs/gauntlet/evidence/s7-redesign/axe-wcag-report.json
latest_cli_feedback: docs/gauntlet/CLI_FEEDBACK.md
latest_critique: docs/gauntlet/critics/desktop-round4.md
design_system: design-system/seoul-simulation-console/MASTER.md
next_focus: pc-object-header-density-data-provenance-and-report-workspace-critique
terminal_completion: forbidden
```

## R004 동시 작업 경계

R004 구현은 사용자의 지시에 따라 현재 `feat/web-console` worktree의 최신 웹 소스를 기준으로 진행 중이다. 과거 문서에 적힌 별도 `codex-gauntlet-design` worktree가 이번 구현의 소스 오브 트루스라고 가정하면 안 된다.

CLI는 다음 R004 활성 파일을 덮어쓰거나 이전 버전으로 되돌리지 말고, 수정 전 이 문서와 `CLI_FEEDBACK.md`를 다시 읽는다.

- `web/ui/src/routes/SimulationLibraryPage.tsx`
- `web/ui/src/routes/SimulationSetupPage.tsx`
- `web/ui/src/routes/SimulationDetailPage.tsx`
- `web/ui/src/routes/agents/AgentInvestigationWorkspace.tsx`
- `web/ui/src/routes/report/SimulationReportWorkspace.tsx`
- `web/ui/src/shell/StudioShell.tsx`
- `web/ui/src/styles/studio.css`
- `web/ui/src/styles/agent-workspace.css`
- `web/ui/src/styles/report-workspace.css`
- `web/ui/e2e/gauntlet-screen.spec.ts`

시뮬레이션 엔진 `scripts/sim`, `scripts/neo4j_load`는 디자인 루프의 수정 대상이 아니다. 보고서 계산도 웹에서 재구현하지 않고 기존 DASOL report API를 호출한다.

## 이번 PC 전용 방향

- 1440px를 주 화면, 1280px를 최소 화면으로 검증한다.
- 모바일 레이아웃은 이번 라운드의 설계·증거·게이트에서 제외한다.
- 한 화면에 모든 정보를 쌓지 않는다. 시뮬레이션 목록 → 설정 → run 개요 → 진행 기록 → 에이전트 조사 → 시각화 → 리포트 순으로 판단 단위를 분리한다.
- run 상세의 상단은 홍보용 hero가 아니라 객체 식별·상태·기간·에이전트·정책을 빠르게 판독하는 운용 헤더로 취급한다.
- 공용 artifact의 파일명이나 부분 문자열로 run 귀속을 추정하지 않는다. report job 또는 명시적 run 디렉터리 계약과 정확히 일치할 때만 링크한다.
- 에이전트 대화 API 계약이 없으면 가짜 대화·가짜 에이전트 목록을 표시하지 않는다.
- light와 dark 모두 일급 테마로 유지하되 현재 기본 테마는 light다.

## 건틀렛 규칙

1. Builder는 현재 미흡점을 개선한다.
2. 독립 Critic은 결함뿐 아니라 이미 작동하는 부분이 목표 수준에 못 미치는 이유를 찾는다.
3. Lead는 Critic 지적을 그대로 수용하지 않고 실제 데이터 계약·화면 증거·사용자 목표와 대조한다.
4. 수정 뒤 1440/1280 캡처, 실제 클릭, 콘솔 오류, 오버플로, 키보드, light/dark 접근성을 다시 검증한다.
5. 검증 통과는 다음 비판 라운드의 기준선일 뿐 종료 판정이 아니다.

각 라운드 종료 시 `docs/gauntlet/CLI_FEEDBACK.md`와 `docs/gauntlet/critics/desktop-round4.md`에 남은 미흡점을 반드시 기록한다.
