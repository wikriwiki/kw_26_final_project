# S6 Builder 전달물 — round 1

역할: S6 전담 빌더

## 변경 파일

- `web/ui/src/shell/nav.ts`, `NavRail.tsx`, `App.tsx` — 정책/실행/결과/데이터 계약/시스템 navigation
- `web/ui/src/routes/DataContractPage.tsx` — 계약·API health inspector
- `web/ui/src/styles/tokens.css` — muted 상태색과 224/48px 셸 토큰
- `web/ui/src/styles/workspaces.css` — 고정 workspace, inspector, responsive layout, 상태 블록
- `web/ui/src/main.tsx` — 디자인 시스템 stylesheet 로드
- `docs/gauntlet/contracts/S6.md`

## 검증

```text
npm run typecheck
npm run build
Get-ChildItem web/ui/src/styles/*.css | Select-String -Pattern "gradient|backdrop-filter|box-shadow|Plus Jakarta Sans|Poppins"
```

결과: typecheck/build exit 0. 금지 CSS 토큰은 전부 0건이고, 4px 초과
숫자 `border-radius`도 0건이다.

