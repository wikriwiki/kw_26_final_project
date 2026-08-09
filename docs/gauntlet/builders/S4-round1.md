# S4 Builder 전달물 — round 1

역할: S4 전담 빌더
선행 계약: `s1.0.0`, S2 API

## 변경 파일

- `web/ui/src/routes/MonitorPage.tsx` — run 선택, days/SSE, 진행·실패율·ETA 원문, 병목, lock 실행 제어
- `web/ui/src/lib/api.ts` — monitor 응답 타입/API
- `web/ui/src/styles/workspaces.css` — monitor table/inspector/lock/bottleneck 레이아웃
- `docs/gauntlet/contracts/S4.md`

## 검증

```text
npm run typecheck
npm run build
python -m unittest tests.unit.test_s1_contract tests.unit.test_s2_api -v
```

결과: typecheck/build exit 0, Python 16/16 통과.

실제 API smoke에서 BASE7500 days는 약 0.059초였고, 상세 집계는 선택 후에만
호출된다. lock duplicate는 S2 테스트에서 409를 확인했다.

