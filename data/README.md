# `data/` — 시뮬레이션 입력 데이터

코드가 아닌 **데이터 파일**만 두는 곳. CSV/JSON/Markdown/PDF.

## 폴더 맵

| 폴더 | 내용 |
|------|------|
| `policies/inbox/` | **Watchdog 감시 대상.** 새 정책 파일(`*.md`, `*.pdf`, `*.txt`)을 여기 드랍 → `src/policy_pipeline`이 자동 처리 |
| `policies/processed/` | 처리 완료된 정책 (감사용 보관, 삭제 금지) |
| `seed/` | 초기 시드 데이터 — 행정동 좌표, POI 카탈로그, Persona 템플릿 |

## 규칙

- **합성 데이터/생성 데이터는 여기 두지 말 것** — `output/`으로
- 큰 파일(>10MB)은 Git LFS 또는 외부 스토리지 사용 검토
- 정책 파일명 컨벤션: `YYYYMMDD_<제목-슬러그>.md` (예: `20260301_소상공인_쿠폰.md`)
- `seed/`의 파일은 `scripts/seed_data.py`로만 로딩 — 직접 다른 코드가 읽지 말 것
