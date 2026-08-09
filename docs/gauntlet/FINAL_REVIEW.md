# 최종 개발·건틀렛 기록 — 2026-08-03

> 이 문서는 초기 결과의 역사 기록이다. 현재 round-3 판정은
> `docs/gauntlet/critics/continuous-round3.md`와
> `docs/gauntlet/gates/continuous-round3.json`을 우선한다. UI/API와 실제
> Chromium 화면은 통과했지만 immutable snapshot 이후에도 Neo4j password와
> `DASOL_NEO4J_RUN_ID`가 없어 DASOL 런타임 생성 게이트는 조건부 보류이며,
> in-app Browser는 가용 브라우저 부재로 실행하지 못했다.

## 판정

S1~S6와 최종 스무딩 Critic을 모두 통과했다. 계약 버전은 `s1.0.0`이다.

| 게이트 | 결과 | Critic |
|---|---|---|
| S1 데이터 계약/픽스처 | 통과 | `critics/S1-1.md` |
| S2 FastAPI/API/lock | 통과 | `critics/S2-1.md` |
| S3 정책 설정 | 통과 | `critics/S3-1.md` |
| S4 실행 모니터 | 통과 | `critics/S4-1.md` |
| S5 시각화 통합 | 통과 | `critics/S5-1.md` |
| S6 앱 셸/디자인 시스템 | 통과 | `critics/S6-1.md` |
| 최종 일관성 스무딩 | 통과 | `critics/final-smoothing-1.md` |

각 게이트의 재현 명령과 변경 범위는 `gates/S*.json` 및 `builders/`에 있다.

## 구현 파일 기록

### S1

- `web/CONTRACT.md`
- `web/fixtures/_build_fixtures.py`
- `web/fixtures/README.md`
- `web/fixtures/*.json` — 실제 BASE/FINAL/BASE7500·P008~P011 응답 36개
- `tests/unit/test_s1_contract.py`

### S2

- `web/api/__init__.py`
- `web/api/app.py`
- `web/api/store.py`
- `web/api/runner.py`
- `web/api/requirements.txt`
- `web/README.md`
- `tests/unit/test_s2_api.py`

### S3~S5

- `web/ui/src/lib/api.ts`
- `web/ui/src/routes/PolicyPage.tsx`
- `web/ui/src/routes/MonitorPage.tsx`
- `web/ui/src/routes/ResultsPage.tsx`
- `web/ui/src/styles/workspaces.css`
- `web/ui/src/main.tsx`
- S5의 `web/api/app.py`/`store.py` artifact 목록·서빙 확장

### S6

- `web/ui/src/App.tsx`
- `web/ui/src/shell/nav.ts`
- `web/ui/src/shell/NavRail.tsx`
- `web/ui/src/routes/DataContractPage.tsx`
- `web/ui/src/styles/tokens.css`
- `web/ui/src/styles/workspaces.css`
- `web/ui/src/main.tsx`

### 검증/기록

- `tests/unit/test_actual_console.py`
- `docs/gauntlet/contracts/S1.md` … `S6.md`
- `docs/gauntlet/builders/S1-round1.md` … `S6-round1.md`
- `docs/gauntlet/critics/S1-1.md` … `S6-1.md`, `final-smoothing-1.md`
- `docs/gauntlet/gates/S1.json` … `S6.json`, `FINAL.json`
- `docs/GAUNTLET_HANDOFF.md` 최신 게이트 상태 갱신

## 검증 결과

```text
python web/fixtures/_build_fixtures.py                         exit 0
python -m unittest discover -s tests -p 'test_*.py' -v         20 passed, 0 failed
cd web/ui && npm run typecheck                                 exit 0
cd web/ui && npm run build                                     exit 0
```

실제 데이터 smoke:

- `BASE7500 /days`: 2초 미만, `counts_source: status_scan`, 진행률 null 보존
- `BASE7500 /days/2025-07-14`: 19,599,953 bytes 원본을 서버 aggregate로 축약
- `P010 validate`: preflight exit 0, pass 16, warn 1, `배경:` 원문 보존
- `/api/artifacts`: HTML 목록만 반환, 기존 3D/리포트는 iframe으로 lazy open
- rescue events: `available:false`, `totals:null`, unknown 보존

스타일 감사 결과: `gradient=0`, `backdrop-filter=0`, `box-shadow=0`, 숫자
`border-radius > 4px=0`, `Plus Jakarta Sans/Poppins=0`.

## 보호 범위

이번 구현은 `scripts/sim`과 `scripts/neo4j_load`를 수정하지 않았다. 해당
경로에 보이는 기존 untracked 보조 파일은 작업 시작 전부터 있던 사용자
변경으로 보존했다. 실제 배포·실행 명령 설정은 기준서대로 사람 확인 전까지
수행하지 않았다.
