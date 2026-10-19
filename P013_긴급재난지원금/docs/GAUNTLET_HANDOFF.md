# Gauntlet Loop 인수인계 — 2026-08-03 05:50 갱신

> **최신 상태 — 2026-08-03 재개 후 갱신.** 아래의 초기 재작업 메모는
> 이 표와 `docs/gauntlet/`의 stage별 기록으로 대체한다.

## Current continuous audit — R003

현재 구현·재검증의 정본은 `docs/gauntlet/builders/continuous-round3.md`,
`docs/gauntlet/critics/continuous-round3.md`, `docs/gauntlet/CLI_FEEDBACK.md`다.
완료 run source를 SHA-256 manifest로 묶고 report 날짜 범위와 policy 분석 적용
여부를 UI/API에서 분리했다. 실제 `BASE`/`FINAL` source scan과 Chromium 검증은
통과했다. 다만 이 환경에는 Neo4j password와 `DASOL_NEO4J_RUN_ID`가 없어 실제
DASOL HTML/MD 생성은 의도적으로 보류되며, UI/API는 생성 버튼을 차단하고
이유를 표시한다. in-app Browser도 가용 브라우저가 없어 Playwright 결과로
대체했음을 기록한다. 다음 라운드는 외부 source binding과 reference 숫자 비교가
가능해질 때까지 닫지 않는다.

## 최신 게이트 상태

| 조각 | 계약/구현 | Critic | 게이트 |
|---|---|---|---|
| S1 | `s1.0.0` 잠금, 실제 3종 픽스처 36개 | `docs/gauntlet/critics/S1-1.md` | 통과 |
| S2 | FastAPI API·서버 집계·SSE·물리 lock | `docs/gauntlet/critics/S2-1.md` | 통과 |
| S3 | 실제 정책 편집·preflight·JSON preview | `docs/gauntlet/critics/S3-1.md` | 통과 |
| S4 | 실제 run monitor·rescue·lock control | `docs/gauntlet/critics/S4-1.md` | 통과 |
| S5 | 기존 3D/리포트 iframe 통합 | `docs/gauntlet/critics/S5-1.md` | 통과 |
| S6 | workspace shell·data contract·system | `docs/gauntlet/critics/S6-1.md` | 통과 |

최종 검증 명령과 파일 목록은 `docs/gauntlet/FINAL_REVIEW.md`에 기록한다.

---

## 디자인 방향 전환 (사용자 반려, 05:50)

사용자가 `web/ui/` 초안을 **"너무 AI스럽다"**고 반려. 팔란티어(Palantir) 대시보드를 레퍼런스로 지정.

원인: `tokens.css`의 `Plus Jakarta Sans`(둥근 지오메트릭 산세리프), `components.css`/`shell.css`의 그라디언트 버튼·`backdrop-filter: blur` 글래스모피즘 — 전형적 "AI가 만든 제네릭 SaaS" 시그니처.

**`docs/GAUNTLET_WEB_CONSOLE.md` §2.1에 팔란티어 방향성과 명시적 금지 목록을 추가함 (05:50).** 이후 모든 S6 작업과 S3~S5의 화면 구현은 이 갱신된 §2.1을 기준으로 판정한다. 핵심 금지: 그라디언트, `backdrop-filter: blur`, 둥근 지오메트릭 폰트, 큰 box-shadow, 12px 이상 border-radius 남발, 히어로+카드그리드 구조. 목표: 어두운 배경·고밀도 그리드·각진 모서리(≤4px)·1px 헤어라인 보더·accent색은 상태표시 전용·모노스페이스 숫자.

S6 재작업 빌더 투입함 (`web/ui/src/styles/` 전면 재작성). 다음 세션은 **먼저 `grep -n "gradient\|backdrop-filter" web/ui/src/styles/*.css`로 0건인지 확인**하고, 크리틱을 붙여 §2.1 신규 기준으로 재검증할 것.

---


> 다음 세션(예약 작업 `gauntlet-web-console-resume`)이 읽는 문서.
> 기준서는 `docs/GAUNTLET_WEB_CONSOLE.md`. 이 문서는 **진행 상태와 미결 사항**만 담는다.

## 브랜치

`feat/web-console` — **커밋 0건.** 모든 산출물이 작업 트리에만 있다.

## 조각별 상태

| 조각 | 빌드 | 크리틱 | 상태 |
|---|---|---|---|
| S1 데이터 계약 + 픽스처 | 완료 → `web/CONTRACT.md`, `web/fixtures/` | **미달 판정** | 재작업 지시 전달, 작업 중이었음 |
| S6 앱 셸 + 디자인 시스템 | 완료 → `web/ui/` | 검증 중 | **판정 결과 소실** — 재검증 필요 |
| S2 백엔드 API | 미착수 | — | S1 확정 + 인증 결정 후 |
| S3 정책 설정 화면 | 미착수 | — | |
| S4 실행 모니터 | 미착수 | — | |
| S5 시각화 통합 | 미착수 | — | |

### 다음 세션이 먼저 할 일

1. `git status`와 `web/` 실제 내용으로 상태 확인. **S1 재작업이 반영됐는지 파일로 판단할 것** (아래 8건이 고쳐졌는지)
2. S1 재검증 크리틱을 새로 띄울 것 (이전 판정은 재작업 전 기준)
3. S6 크리틱을 다시 띄울 것 (판정 소실)

## S1 크리틱이 지적한 재작업 8건 (재검증 체크리스트)

크리틱은 실측 기반 자체는 견고하다고 평가했다. B5 압축비(2,876:1, peak 1.5MB)와 B2 preflight 일치(4개 정책 diff 0)를 **직접 재실행해 재현**했고, 픽스처 값을 원본에서 독립 재계산해 하드코딩 0건을 확인했다. 이 부분은 재작업 대상이 아니다.

미달 사유는 **계약이 소비자에게 주는 지시가 틀린 지점**이다.

1. **`prompt_preview` 파싱 (치명, B2)** — `-`로 시작하는 줄만 줍는 규칙이 2칸 들여쓴 `배경:` 줄을 버림. 실측 누락 P008 127자 / P009 143자 / P010 286자 / P011 84자. 이 줄은 `scripts/sim/dawn_context.py:507`이 만드는 **실제 Dawn 프롬프트 원문**. 계약대로면 정책 `description`을 고쳐도 미리보기가 안 변함
2. **`spend_decile: null` (치명, §4.1.2)** — `rescue` ok 4,533행 중 **5행이 null**. `_build_fixtures.py:279`의 `isinstance(d, int)`가 조용히 버려 `sum(by_spend_decile) = 4528 ≠ agents_ok 4533`. 불완전 run 테스트 케이스에서 자기 규칙을 위반
3. **`unknown: string[]` (치명, §4.1.3)** — "모든 리소스가 갖는다"고 선언했으나 8종 중 3종에만 존재. 없는 것: `*.day.*`, `*.bottlenecks`, `*.slow`, `*.failures`, `*.events.summary`, `*.failed`
4. **`available:false` 키 집합 불일치** — `slow`·`events.summary`가 `true`일 때만 있는 필드를 무조건부로 선언. `bottlenecks`가 올바른 본보기
5. **실측 오기 8건** — `:74`(첫날에만 없음, 7일 전부 아님) `:177`(`ValueError`·`length` 누락) `:27`(24개) `:86`(33개) `:60`(83/82) `:188`(BASE 15/FINAL 18) `:556`(34개) `README:16`(390KB)
6. **`_fields_not_aggregated`** — 50행 표본 제한 제거, 응답에 반영된 7필드 제외 또는 개명
7. **B4 설계 공백** — `aggregate_day(19.6MB)` = **3.74초** 실측. 진행 중 일자는 캐시 불가라 매 요청 재집계 → 2초 기준을 구현 단계에서 반드시 깸. 대안을 §3.3 계약에 명시해야 함
8. **누락 픽스처 2건** — BASE7500 `failed` 응답, `grant_key` 기본값 불일치(P009·P011이 null)

## 시뮬레이션 데이터에 대해 확인된 사실 (웹 작업 외 중요)

- **`out_BASE`는 이름과 달리 P010 정책이 적용된 run.** `events.jsonl`에 P010 결제 4,857,655원 실측. 무정책 대조군 아님
- **`events.jsonl`의 `dong` 필드가 100% null** (7,785/7,785). 행정동 단위 지도 조인 불가
- `stage1_failures.jsonl`(LLM 파싱 실패, 재시도 복구분 포함, BASE 32건) ≠ `failed_*.json`(최종 agent 실패, BASE 0건). **합산 금지**
- `run_BASE7500.log`는 `workers: 48`인데 `chain_p2.sh`는 `--workers 128`. **`summary.json`이 정본**, 로그는 `log_hint`로 출처 분리
- `NEO4J_URI` 미설정 시 preflight의 DB 배선 점검이 통째로 스킵됨. "정책이 아무에게도 적용 안 됨"이라는 치명 결함을 로컬에서 못 잡음. 이 warn을 초록 체크로 묶지 말 것

## 사용자 결정 대기 2건 (재개 시 반드시 확인)

1. **보안 그룹 22번 개방** — EC2 `i-02a20c6a1b974f0b1` (43.201.218.176)의 `sg-0a45d2b0c60286a61`에 인바운드 SSH 없음. 22/80/443/8000 전부 타임아웃. 마지막 확인 시 접속 PC 공인 IP는 `211.109.170.94`(변동 가능 — `curl -s https://checkip.amazonaws.com`로 재확인)
2. **웹 공개 시 인증 여부** — 실행 트리거를 넣기로 했는데 인증은 비목표. 인터넷 공개 시 누구나 실험 기동·중단 가능. 선택지 (a) 소스 IP 제한 (b) 인증을 조각 S7로 추가 (c) 조회 전용 회귀.
   **이 결정 전에는 S2의 실행 lock 설계를 확정하지 말 것.**
