# 정책 시뮬레이션 웹 콘솔 — UI 셸 + 디자인 시스템 (S6)

정책 설정(S3) · 실행 모니터(S4) · 결과 시각화(S5) 세 화면이 **하나의 제품처럼** 보이게 하는
껍데기와 공통 시각 언어. 세 화면의 내용은 각 조각의 빌더가 채운다.

```
web/ui/
  index.html
  vite.config.ts          /api, /artifacts 를 FastAPI 로 프록시
  src/
    main.tsx              CSS import 순서 고정 (tokens → base → components → shell)
    App.tsx               라우팅 (HashRouter)
    styles/
      tokens.css          ★ 디자인 토큰. 값의 출처가 전부 주석에 있다
      base.css            리셋 · 포커스 링 · 스크롤바
      components.css      패널 · 버튼 · 배지 · 수치행 · 표 · 상태 블록
      shell.css           상단바 · 레일 · 본문 · 상태바 · 반응형
    shell/                AppShell, TopBar, NavRail, StatusBar, Page, ShellContext, nav.ts
    components/           공통 컴포넌트 (배럴: components/index.ts)
    routes/               화면 4개 (정책/모니터/결과는 플레이스홀더) + 디자인 킷
    lib/                  async.ts(상태 모델) · format.ts(표시 포맷)
```

## 실행

```bash
npm install          # web/ui 에서
npm run dev          # http://localhost:5173  (API 는 127.0.0.1:8000 으로 프록시)
npm run build        # tsc -b && vite build → web/ui/dist
npm run typecheck
```

`VITE_API_ORIGIN` 으로 프록시 대상을 바꿀 수 있다. 빌드 산출물은 `web/ui/dist` 에만 쓴다
(`web/api`, `web/fixtures` 는 다른 조각 소유이므로 여기서 건드리지 않는다).

**디자인 킷은 `#/kit`.** 토큰과 컴포넌트 실물이 다 있다. 새 화면을 만들기 전에 여기부터 연다.

---

## 1. 토큰은 어디서 왔나

임의로 지어낸 값은 없다. 세 갈래다.

| 표기 | 출처 |
|---|---|
| `[3D]` | `scripts/sim/visualization_3d/static/styles.css` — 값 그대로 |
| `[RPT]` | `output/sim/report/FINAL_REPORT_5D_FULL.html` `<style>` — 값 그대로 |
| `[계산]` | 위 두 곳의 값을 알파 합성하거나 스케일로 정리한 것. 계산식이 주석에 있다 |

`tokens.css` 의 모든 선언에 이 표기가 붙어 있다. 새 토큰을 추가할 때도 반드시 붙인다.

### 그대로 가져온 것

- 색: `--cyan #39d7ff` `--mint #3ff2b8` `--amber #ffbf5a` `--rose #ff5d7a` `--violet #a98bff`
  `--text #edf7ff` `--muted #9fb2c5` `--dim #627386` `--aqua #72efdd`(legacy-val) `--coral #e76f51`(legacy-detail)
- 선: `--line rgba(180,216,255,.16)` `--line-bright rgba(88,214,255,.42)` `--line-soft rgba(255,255,255,.08)`
- 유리: `--glass-panel rgba(6,10,18,.55)` `--glass-strong rgba(6,10,18,.92)` `--glass-legacy rgba(15,15,30,.55)`,
  블러 18px / 10px
- 상호작용: `--hover rgba(255,255,255,.06)` `--field-bg rgba(0,0,0,.34)` `--selected rgba(57,215,255,.08)`
- 그림자: `--shadow 0 12px 32px rgba(0,0,0,.22)` `--shadow-panel 0 2px 14px rgba(0,0,0,.25)`
- 타이포 스케일: 10 / 11 / 12 / 13 / 14 / 15 / 18px, 굵기 500·600·700·750·850·900, 행간 1.2·1.4·1.45
- 여백: 2 / 4 / 6 / 8 / 10 / 12 / 14 / 16 / 18px (styles.css 에 실제로 등장하는 값만),
  24·32 는 `[RPT]` 카드 패딩에서
- 라디우스: 4 / 5 / 6 / 8 / 12 / 999px
- 컨트롤 높이: 28 / 30 / 32 / 34 / 36px, 상단바 52px, 레일 212px(좁으면 188px)
- 모션: 0.15s (`[RPT]` 사이드바 트랜지션)

### 의도적으로 바꾼 것과 그 이유

| 바꾼 것 | 원본 | 콘솔 | 이유 |
|---|---|---|---|
| 표면 | 반투명 패널이 지도 위에 뜸 | `--surface-1 #0b0b16` 불투명 | 콘솔에는 뒤에 지도가 없다. 같은 rgba 를 쓰면 배경과 붙어 패널이 사라진다. **`rgba(15,15,30,.55)` 를 `#05070d` 위에 합성한 값**을 고정색으로 썼다. 즉 3D 뷰에서 눈에 보이던 그 색 그대로다 |
| `--surface-2 #191a24` | — | 호버·표 헤더 | `rgba(255,255,255,.06)` 를 `--surface-1` 에 합성한 값 |
| 폰트 | `"Plus Jakarta Sans", system-ui …` | 뒤에 `Pretendard, "Noto Sans KR", "Apple SD Gothic Neo", "Malgun Gothic"` 추가 | 3D 뷰 스택에는 한글 페이스가 없다. 콘솔은 레이블이 전부 한글이라 폴백이 없으면 굵기·자간이 튄다. **네트워크 폰트는 로드하지 않는다** (리포트는 Google Fonts 를 쓰지만, 콘솔은 사내망/오프라인에서도 같아야 한다) |
| 의미색 | 리포트의 `#38a169 / #d69e2e / #e53e3e` | 3D 팔레트의 mint / amber / rose | 밝은 문서용 색을 어두운 면에 올리면 채도가 죽는다. 의미만 가져오고 색은 3D 것을 썼다 |
| 패널 제목 | `.legacy-panel h3` 가 제목 전체를 시안으로 칠함 | 제목은 `--text`, 왼쪽에 3px 시안 마커 | 3D 뷰는 패널이 3~4개지만 콘솔은 한 화면에 6개 넘게 뜬다. 제목이 전부 시안이면 강조가 강조가 아니게 된다. 시안 신호는 마커로만 남겼다 |
| 페이지 최대 제목 | `[RPT]` cover h1 34px | 22px(`--fs-2xl`) | 콘솔은 문서가 아니라 작업 화면이다. 밀도를 리포트보다 한 단 높게 잡았다 |
| 브레이크포인트 | 980 / 760 / 420 | 1100 / 900 / 560 추가 | 검사 폭 **768px** 이 760px 규칙 바깥이라 태블릿 레이아웃으로 안 들어간다. 900px 을 둬서 768px 에서 레일→탭 전환이 일어나게 했다 |
| 스크롤 | `html,body{overflow:hidden}` | `body{overflow-x:hidden}`, 세로 스크롤은 `.main` 안에서만 | 3D 뷰는 전체화면 지도라 스크롤이 없지만 콘솔은 문서형 화면이 섞인다 |

### 접근성 주의

`--dim (#627386)` 은 `--surface-1` 대비 약 **3.9:1** 로 본문 기준(4.5:1) 미달이다.
**placeholder·비활성·장식에만** 쓴다. 읽어야 하는 텍스트는 `--muted (약 8.5:1)` 이상을 쓴다.

---

## 2. 다른 빌더가 지켜야 할 규약

### R1. 리터럴 값 금지

CSS 에 `padding: 12px`, `color: #39d7ff` 를 직접 쓰지 않는다. 항상 `var(--sp-6)`, `var(--cyan)`.
JSX `style={{}}` 에도 마찬가지다 — `style={{ gap: 'var(--sp-5)' }}`.
필요한 토큰이 없으면 `tokens.css` 에 출처 주석과 함께 추가하고 이 문서에 적는다.

### R2. 색-의미 대응은 고정

| 색 | 의미 | 쓰는 곳 |
|---|---|---|
| cyan `accent` | 진행 중 · 선택됨 · 강조 | 활성 탭, 실행 중 배지, 포커스 |
| mint `ok` | 정상 · 완료 | 완료 run, 임계 이하 |
| amber `warn` | 주의 · **부분 데이터** | 불완전 run, 커버리지 |
| rose `danger` | 실패 · 차단 | 에러, lock 충돌, 중단 |
| violet `info` | 참고 | 부가 정보 |

다른 의미로 쓰면 세 화면 사이의 신호가 깨진다. `Badge` 의 `tone` 이 이 매핑을 강제한다.

### R3. 숫자는 `lib/format.ts` 로만 찍는다

`int` `dec` `percent` `krw` `duration` `shortTime` `dateTime` `bytes`.
값이 없으면 **`0` 을 만들지 말고** `EMPTY('—')` 를 돌려준다. `Stat` / `Metric` 은 `null` 을 받으면 알아서 `—` 를 찍는다.
숫자를 담는 요소에는 `.num` 또는 `.mono` 를 붙인다 (등폭 + tabular-nums). 표의 숫자 열은 `<td className="num">`.

**"값 없음"과 "값 모름"은 다르다** (CONTRACT §4.1-2/3). API 가 `unknown: []` 배열로 미확인을 선언하므로:

```tsx
<Stat label="목표 agent 수"
      value={int(run.agents_target)}
      unknown={isUnknown(run.unknown, 'agents_target')} />   // → '알 수 없음' 배지
```

`null` 인데 `unknown` 에 없으면 `—`, `unknown` 에 있으면 앰버 배지다. 0 으로 채우지 않는다.

### R4. 상태 4종은 직접 만들지 않는다

`if (loading) return <Spinner/>` 같은 분기를 화면마다 쓰면 바로 이질감이 생긴다.
`lib/async.ts` 의 `AsyncState<T>` 를 쓰고 `AsyncBoundary` 에 넘긴다.

```tsx
import { AsyncBoundary } from '../components';
import { loading, ready, failed, empty, type AsyncState } from '../lib/async';

const [runs, setRuns] = useState<AsyncState<Run[]>>(loading());

<AsyncBoundary
  state={runs}
  loadingShape="table"          // 채워질 모양과 같게 (레이아웃이 안 튄다 → B4 체감)
  emptyTitle="완료된 run 이 없습니다"
  emptyDescription="정책을 저장한 뒤 실행하면 여기에 결과가 쌓입니다."
  onRetry={reload}
  subject="BASE run"
>
  {(data) => <RunTable rows={data} />}
</AsyncBoundary>
```

- **부분 데이터는 에러가 아니다.** `ready(data, coverage)` 로 넘기면 `AsyncBoundary` 가 본문 위에
  `PartialDataNotice` 를 자동으로 얹는다. 본문은 그대로 보여준다.
  `rescue/out_BASE7500` 처럼 Day 0 만 있는 run 이 실물로 존재한다 — 막으면 있는 데이터도 못 보고,
  그냥 보여주면 전체 기간의 결론으로 오독한다.
- 커버리지는 CONTRACT §3.1 필드에서 바로 만든다:
  ```ts
  ready(days, coverageFromRun(run));   // days_present / days_planned / status 를 그대로 읽는다
  ```
  `days_planned` 가 `null` 이면 배너가 **비율을 그리지 않는다.** 모르는 것을 %로 지어내지 않기 위해서다.
  `status: "incomplete"` 는 `partial: true` 로만 들어가며, 문구도 "중단"이지 "실패"가 아니다
  (실패는 rose, 중단·부분은 amber — R2).
- `coverage` 의 `available` / `expected` / `reason` 은 **서버가 준 값 그대로** 넣는다. UI 가 계산하거나 추측하지 않는다.
- 에러의 `detail` 에는 서버 응답 원문을 그대로 넣는다. `ErrorState` 가 접어서 보여준다. 삼키지 않는다.
- CONTRACT §3.5 의 `degraded: true` 는 `coverage.reason` 에 `degraded_note` 를 그대로 넣어 표현한다.

### R5. 화면 골격은 `Page` 로 시작한다

```tsx
<Page eyebrow="MONITOR" title="실행 모니터" description="…" actions={<Button …/>}>
  <Panel title="진행 요약" subtitle="Day 3">…</Panel>
  <div className="split">
    <div className="stack">…</div>   {/* 본문 */}
    <div className="stack">…</div>   {/* 우측 사이드 (900px 이하에서 아래로 떨어짐) */}
  </div>
</Page>
```

- 레이아웃 클래스: `.split`(본문+사이드) `.stack`(세로 묶음) `.toolbar`(가로 묶음) `.metric-grid`
- 문서형 화면은 `<Page reading>` (1024px 제한), 전체 높이를 쓰는 임베드는 `<Page fill>`
- 담는 그릇은 항상 `Panel`. div 에 직접 border/padding 을 주지 않는다.
- `Button variant="primary"` 는 **화면당 하나**. 되돌리기 어려운 행동(실행·저장) 전용이다.

### R6. 셸에 상태를 알릴 때

```tsx
useShellBadge({ label: '실행 중', tone: 'accent', dot: 'pulse' });   // 상단바 우측 배지
useStatusItems([{ label: 'Day', value: '3/7' }], [day]);             // 하단 상태바
const { runId, setRunId } = useShell();                             // 세 화면이 공유하는 run
```

`runId` 는 localStorage 에 저장되어 새로고침해도 유지된다(B6 의 UI 쪽 절반).
셸은 **데이터를 갖지 않는다** — 표시할 문자열만 받는다. 즉 셸이 수치를 지어낼 여지가 없다(B1).

### R7. 반응형 (기준 B7)

1280px / 768px 에서 가로 스크롤이 없어야 한다. 지키면 자동으로 통과하는 규칙:

- 모든 그리드 트랙에 `minmax(0, …)`, 유연 자식에 `min-width: 0`
- 넓은 표는 `<div className="table-wrap">` 안에 넣는다 (화면이 아니라 표가 스크롤한다)
- 긴 문자열(run id, 경로)은 `text-overflow: ellipsis` 또는 `overflow-wrap: anywhere`
- iframe 임베드는 `.embed` 컨테이너 안에
- 900px 이하에서 좌측 레일이 사라지고 상단 탭이 나온다 — 레일에 기능을 숨기지 않는다

검증 방법(빌더가 직접 돌릴 것):

```js
// 개발자 콘솔에서
document.documentElement.scrollWidth === innerWidth   // true 여야 한다
```

### R8. 소유 경계

- `web/ui/**` 만 고친다. `scripts/`, `data/`, `output/` 은 **읽기 전용**이다.
- `web/CONTRACT.md`, `web/fixtures/`, `web/api/` 는 다른 조각 소유다.
- 화면을 추가하려면 `src/shell/nav.ts` 의 `NAV_ITEMS` 에 넣고 `App.tsx` 에 라우트를 건다.
  레일·탭·페이지 헤더가 자동으로 따라온다.
- 새 공통 컴포넌트는 `src/components/` 에 넣고 `index.ts` 에 export 한 뒤 이 문서에 적는다.
  화면 폴더 안에 프리미티브를 만들지 않는다 — 그게 드리프트의 시작이다.

### R9. 라우팅은 HashRouter

정적 서빙에서 새로고침 404 가 나지 않게 해시 라우팅을 쓴다(B6 가 서버 설정에 인질로 잡히지 않도록).
S2 가 SPA fallback 을 보장하면 `App.tsx` 의 한 줄만 `BrowserRouter` 로 바꾸면 된다.

---

## 3. 현재 상태

- 셸(상단바·레일/탭·본문·상태바), 토큰, 공통 컴포넌트, 상태 4종 — 완료
- 정책/모니터/결과 세 화면 — **의도적으로 빈 플레이스홀더.** 각 화면 파일 상단 주석에
  "셸이 제공하는 것 / 담당 조각이 채울 것"이 적혀 있다
- `npm run build` 통과, 1280 / 768 / 375px 에서 가로 스크롤 0
