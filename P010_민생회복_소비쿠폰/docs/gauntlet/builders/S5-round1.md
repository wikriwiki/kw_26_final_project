# S5 Builder 전달물 — round 1

역할: S5 전담 빌더
선행 계약: `s1.0.0`, S2 API

## 변경 파일

- `web/ui/src/routes/ResultsPage.tsx` — run 선택, 기존 3D/리포트 iframe, events summary 집계·미확인 상태
- `web/ui/src/lib/api.ts` — artifact 목록/이벤트 summary 타입 및 호출
- `web/api/store.py` — output HTML 목록 endpoint와 안전한 root 제한
- `web/api/app.py` — `GET /api/artifacts` endpoint
- `web/ui/src/styles/workspaces.css` — iframe/결과 inspector 레이아웃
- `docs/gauntlet/contracts/S5.md`

## 검증

```text
npm run typecheck
npm run build
python -m unittest tests.unit.test_s1_contract tests.unit.test_s2_api -v
```

결과: typecheck/build exit 0, Python 16/16 통과.
실제 smoke: artifact 목록 769 bytes, BASE events summary 9.5KB, rescue events
summary 408 bytes(`available:false`).

