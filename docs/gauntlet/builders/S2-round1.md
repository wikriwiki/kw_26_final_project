# S2 Builder 전달물 — round 1

역할: S2 전담 빌더
선행 계약: `s1.0.0`

## 변경 파일

- `web/api/__init__.py`
- `web/api/store.py`
- `web/api/runner.py`
- `web/api/app.py`
- `web/api/requirements.txt`
- `tests/unit/test_s2_api.py`
- `docs/gauntlet/contracts/S2.md`

## 검증 명령과 결과

```text
python -m compileall -q web/api
python -m unittest tests.unit.test_s1_contract tests.unit.test_s2_api -v
```

결과: **16 passed**, exit 0.

실제 모드 smoke test:

- `/api/runs`: 200, 약 0.046s
- `/api/runs/BASE/days`: 200, 약 0.046s
- `/api/runs/BASE7500/days`: 200, 약 0.059s
- `/api/runs/BASE7500/days/2025-07-14`: 200, 원본 19,599,953 bytes를 약 4.9KB 응답으로 집계
- `/api/policies/P010/validate`: 200, 실제 preflight `pass:16`, `warn:1`

## 알려진 운영 설정

실제 실행 버튼은 `SIM_RUN_COMMAND_JSON`이 없으면 503을 반환한다. 명령을
지어내거나 엔진을 건드리는 fallback은 없다. 운영 배포 전에 실행 명령과
lock 경로를 사람이 설정해야 한다.

