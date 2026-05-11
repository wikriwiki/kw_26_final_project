# `output/` — 시뮬레이션 산출물

**런타임 생성물.** 폴더 구조만 추적되고, 내부 파일은 `.gitignore`로 제외됩니다.

## 폴더 맵

| 폴더 | 내용 |
|------|------|
| `logs/` | 시뮬레이션 실행 로그 (`orchestrator`, `policy_pipeline`) |
| `plans/` | Day별 Plan 덤프 (디버깅용) |

## 규칙

- **이 폴더에 사람이 직접 파일을 두지 말 것.** 모두 코드가 생성하는 것.
- 보관이 필요한 산출물(리포트, 그래프 익스포트)이라도 여기에 — 사람이 만든 데이터는 `data/`로
- 디스크 부담 시 `scripts/reset_simulation.py` 또는 단순 `rm -rf output/*` 으로 정리
