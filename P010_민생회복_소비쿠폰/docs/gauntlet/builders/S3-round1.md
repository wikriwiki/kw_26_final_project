# S3 Builder 전달물 — round 1

역할: S3 전담 빌더
선행 계약: `s1.0.0`, S2 API

## 변경 파일

- `web/ui/src/lib/api.ts` — S1 정책/run 타입과 실제 API 호출 클라이언트
- `web/ui/src/routes/PolicyPage.tsx` — 정책 목록, 입력, preflight, JSON 미리보기, 저장
- `web/ui/src/styles/workspaces.css` — 정책 3열 workspace와 반응형 규칙
- `web/ui/src/main.tsx` — workspace stylesheet 로드
- `docs/gauntlet/contracts/S3.md`

## 검증

```text
npm run typecheck
npm run build
python -m unittest tests.unit.test_s1_contract tests.unit.test_s2_api -v
```

결과: TypeScript typecheck exit 0, Vite build exit 0, Python API/S1 회귀 16/16
통과. 화면 수치는 fixture 상수 대신 `/api/policies`와 실제 preflight 응답만
사용한다.

