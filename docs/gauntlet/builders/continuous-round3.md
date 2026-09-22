# Continuous Builder Record — round 3

기준서는 `docs/GAUNTLET_WEB_CONSOLE.md` 하나로 고정했다. 이번 라운드는
보고서 숫자를 만들기 전에 run snapshot과 Neo4j source 경계를 다시 만들었다.

## 구현

- `scripts/report/snapshot.py`를 추가했다. 완료 run의 계획 일수와 실제
  `summary.json`, `events.jsonl`, `poi_summary.json`, 일자별 `metrics/`, `timing/`,
  `checkpoints/done_*.json`을 검사하고 SHA-256 manifest를 원자적으로 기록한다.
  CLI는 계산 직전 모든 hash를 다시 검사하며, 변경된 파일은 거부한다.
- `web/api/report_jobs.py`에 snapshot readiness, physical report lock 이후 manifest
  생성, snapshot artifact, `DASOL_NEO4J_RUN_ID` binding gate를 추가했다.
  password만 있고 run binding이 없는 live DB는 성공으로 표시하지 않는다.
- `scripts/report/menu.py`는 verified manifest를 필수 입력으로 받고
  `SIM_OUTPUT_DIR`를 선택 snapshot root에 묶는다. P009 income DID dispatch를
  복원했고, origin/dasol의 modern context-dispatch 함수가 들어오면
  `scripts/report/engine.py` bridge를 사용한다.
- `scripts/report/catalog.py`는 origin/dasol 기준으로 income grant 또는
  benefit category가 없는 sales DID를 비활성화한다. UI는 report 범위를 실제
  snapshot day 목록과 비교해 API 409 전에 생성 버튼을 막는다.
- provenance HTML/Markdown에는 snapshot ID와 API source links를 포함시켰다.

## 보호 범위

`scripts/sim/*`와 `scripts/neo4j_load/*`는 변경하지 않았다. 기존 report 계산과
renderer만 adapter가 호출한다.
