# 웹 콘솔 디자인 스펙 v2 — 전면 재설계

> **이 문서가 디자인의 유일한 기준이다.** `docs/GAUNTLET_WEB_CONSOLE.md` §2.1의 시각 기준을 이 문서가 대체한다.
> 근거: `.agents/skills/ui-ux-pro-max/SKILL.md` (67 styles / 99 UX guidelines) + `design-system/policy-simulation-console/MASTER.md`
> 목표 등급: **WCAG AAA (본문 7:1)**. AA는 미달로 간주한다.

## 0. v1이 반려된 이유 (반복 금지)

사용자 지적 → 위반 규칙 매핑. 크리틱은 이 표를 체크리스트로 쓴다.

| 사용자 지적 | 위반 규칙 | 재발 방지 기준 |
|---|---|---|
| 위젯끼리 맞지 않음 | §5 `visual-hierarchy`, §6 `whitespace-balance` | 12열 그리드에 스냅. 임의 width/height 금지 |
| 정책 객체에 내부 좌우 스크롤 | §5 `horizontal-scroll` (명시적 안티패턴), §5 `scroll-behavior` | **어떤 컨테이너에도 `overflow-x: auto` 금지.** 긴 내용은 wrap 또는 접기 |
| 테마가 까매서 안 보임 | §6 `color-accessible-pairs`, `color-dark-mode` | **라이트 테마 기본.** 본문 7:1 이상 |
| 너무 불친절 | §8 `empty-states`, `input-helper-text`, `error-clarity` | 모든 화면에 목적 1문장 + 다음 행동 |
| 한 화면에 다 때려박아 정보량 과다 | §8 `progressive-disclosure`, §5 `content-priority` | **1화면 = 1작업.** 기본 노출은 상위 지표만, 상세는 펼치기 |
| 데이터 계약·시스템 섹션을 왜 만들었는지 모르겠음 | §5 `content-priority` | **내부 개발 산출물을 사용자 화면에 노출 금지.** KitPage는 `/__kit` 개발 전용 |
| AI스러움 | §4 `style-match` | 그라디언트·글래스모피즘·둥근 지오메트릭 폰트 금지 (§4 참조) |

## 1. 제품 정의

정책 담당자가 **매일 쓰는 업무 도구**다. 마케팅 페이지가 아니고, 데이터 엔지니어용 내부 콘솔도 아니다.
읽는 사람은 시뮬레이션 내부 구조를 모른다. **용어는 정책 담당자의 언어로 쓴다.**

- `stage1_failures` → "응답 오류"
- `by_spend_decile` → "소비 분위별"
- `t_s2` → "장소·지출 결정 소요"

## 2. 색 (라이트 기본, AAA)

```css
--bg:            #F8FAFC;  /* 페이지 배경 */
--surface:       #FFFFFF;  /* 카드·패널 */
--surface-sunken:#F1F5F9;  /* 표 헤더, 입력 배경 */
--fg:            #0F172A;  /* 본문 — bg 대비 16.9:1 (AAA) */
--fg-muted:      #475569;  /* 보조 — bg 대비 7.5:1 (AAA) */
--fg-subtle:     #64748B;  /* 캡션 전용 — 5.2:1, 본문 금지 */
--border:        #E2E8F0;
--border-strong: #CBD5E1;
--primary:       #1E3A5F;  /* 네이비. 흰 글자 대비 11.4:1 */
--primary-hover: #16304F;
--ring:          #2563EB;  /* 포커스 링 */

/* 상태 — 반드시 아이콘/텍스트와 함께 (§1 color-not-only) */
--ok:      #166534;  /* 7.4:1 */
--warn:    #854D0E;  /* 7.1:1 */
--danger:  #991B1B;  /* 8.2:1 */
--info:    #1E40AF;  /* 8.6:1 */
--ok-bg:   #F0FDF4;  --warn-bg: #FEFCE8;  --danger-bg: #FEF2F2;  --info-bg: #EFF6FF;
```

**금지**: 다크 배경 기본, 채도 높은 네온, 색만으로 의미 전달, 컴포넌트에 raw hex(§6 `color-semantic`).

## 3. 타이포

```css
--font-sans: Inter, "Pretendard", "Noto Sans KR", system-ui, sans-serif;
--font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, monospace;
```

- 본문 **16px / line-height 1.6** (§5 `readable-font-size`, §6 `line-height`)
- 스케일: 12 / 13 / 14 / 16 / 20 / 24 / 32 — 그 외 금지
- **12px는 캡션·레이블 전용.** 본문 금지 (§6)
- 굵기 400 / 500 / 600 세 단계만. 700 이상 금지
- **모든 수치는 `--font-mono` + `font-variant-numeric: tabular-nums`** (§6 `number-tabular`)
- 금지: Plus Jakarta Sans, Poppins 등 둥근 지오메트릭 산세리프

## 4. 형태

- `border-radius`: **6px** (카드·입력·버튼), **4px**(뱃지), **0**(표 셀). 그 외 금지
- 그림자: `0 1px 2px rgba(15,23,42,.06)` **한 단계만**. 카드를 띄우는 큰 그림자 금지
- **`linear-gradient` / `radial-gradient` 전면 금지**
- **`backdrop-filter: blur` 금지** — 단, 모달 스크림은 예외 (§4 `blur-purpose`)
- 1px 헤어라인 `--border`로 영역 구분

## 5. 간격 — 4/8 리듬 (§5 `spacing-scale`)

```css
--sp-1:4px; --sp-2:8px; --sp-3:12px; --sp-4:16px; --sp-5:24px; --sp-6:32px; --sp-7:48px;
```

밀도 8/10 — 카드 패딩 `--sp-4`, 섹션 간격 `--sp-6`, 표 셀 `--sp-2 --sp-3`.

## 6. 레이아웃

### 사이드바 (사용자 지정 요구사항)

**기본 숨김 → 클릭하면 나타나며 선택 가능.**

- 기본 상태: 좌측 **56px 아이콘 레일** 상주. 완전 숨김이 아니라 레일로 남긴다 (§9 `persistent-nav` — 핵심 내비는 항상 도달 가능해야 함)
- 상단 햄버거 클릭 → **240px로 확장**. 아이콘 + 텍스트 레이블 동시 표시 (§9 `nav-label-icon`)
- 확장은 **push**(1024px 이상) / **overlay + 스크림**(1024px 미만)
- 현재 위치는 좌측 3px 인디케이터 + `--primary` 텍스트 + `aria-current="page"` (§9 `nav-state-active`)
- 확장 상태는 `localStorage` 유지 (§9 `state-preservation`)
- 애니메이션 **200ms `ease-out`**, `transform`만 사용. `prefers-reduced-motion`에서 0ms (§7)
- 레일 아이콘은 44×44 히트 영역, `aria-label` 필수 (§1, §2)

### 본문 그리드

- 12열, gutter `--sp-4`, 최대 폭 **1440px**
- **모든 위젯은 열 경계에 스냅.** 임의 px width 금지 → 위젯 정렬 문제의 근본 해결
- 같은 행의 카드는 `align-items: stretch`로 높이 일치
- 브레이크포인트 **375 / 768 / 1024 / 1440** (§5 `breakpoint-consistency`)

### 가로 스크롤 절대 금지

- 어떤 컨테이너에도 `overflow-x: auto` **금지**
- 긴 표는 **열 우선순위 기반으로 축소** — 좁은 폭에서 부차 열을 숨기고 행 펼치기로 제공
- 긴 텍스트(정책 설명 등)는 wrap. 잘라야 하면 3줄 클램프 + "더 보기" (§6 `truncation-strategy`)
- 검증: 375 / 768 / 1024 / 1440 전 폭에서 `documentElement.scrollWidth === innerWidth`

## 7. 정보 구조 — 1화면 1작업

세 화면만. 각 화면 진입 시 **한 화면에 보이는 1차 정보는 최대 7개 블록**.

| 화면 | 단일 목적 | 기본 노출 | 접어두는 것 |
|---|---|---|---|
| 정책 설정 | 정책을 만들고 검증한다 | 정책명·기간·분위별 지급액·사용처 제한, 검증 결과 요약 | 원문 JSON, 프롬프트 미리보기, 검증 상세 로그 |
| 실행 모니터 | 지금 잘 돌고 있는지 본다 | 진행률·경과/예상·오류 건수·일자별 추이 | 단계별 소요 분해, 느린 케이스 목록, 원본 로그 |
| 결과 | 정책 효과를 해석한다 | 핵심 지표 4개, 분위별 비교, **지도 미리보기 1개** | 원자료 표, 내보내기 |
| 시각화 | 지도 위에서 시뮬레이션을 관찰한다 | 3D 지도 전체화면 + 최소 컨트롤 | 레이어·필터 패널(기본 닫힘) |

### 시각화 페이지 진입 흐름 (§9 `deep-linking`, `back-behavior`)

시각화는 **결과 화면의 자연스러운 심화**다. 별도 기능이 아니라 "더 자세히 보기"의 종착지다.

1. **주 진입 — 결과 화면의 지도 미리보기.** 결과 화면에 정적 미리보기 카드를 두고, 우상단에 "지도에서 열기" 액션.
   미리보기를 클릭해도 같은 곳으로 간다 (§2 `no-precision-required`)
2. **보조 진입 — 사이드바 항목.** 이미 어떤 run을 보는지 아는 사용자용 직접 경로
3. 진입 시 **결과 화면에서 보던 run·기간·필터를 그대로 이어받는다** (§9 `state-preservation`)
4. 시각화 화면 좌상단에 **"← 결과로"** 브레드크럼. 뒤로 가면 원래 스크롤 위치로 복귀 (§9 `back-behavior`)
5. URL에 run/일자를 담아 공유 가능하게 (§9 `deep-linking`)

시각화 화면은 **지도가 주인공**이다. 지도를 전체 폭으로 두고 컨트롤은 최소한만 얹는다.
사이드바는 이 화면에서 기본 접힘(레일)으로 진입한다.

- 접힌 내용은 `<details>` 또는 "상세 보기" 토글. **기본은 닫힘** (§8 `progressive-disclosure`)
- **데이터 계약·픽스처·시스템 진단은 사용자 화면에서 제거.** 개발용은 `/__kit` 라우트로 격리
- 각 화면 최상단에 **목적 1문장**. 예: "지급 대상과 금액을 정하고, 실행 전에 오류를 확인합니다."
- 빈 상태는 안내 + 다음 행동 버튼 (§8 `empty-states`)

## 7b. 절제 (미니멀·세련됨)

"미니멀"은 **요소를 줄이는 것**이지 여백을 늘리는 것이 아니다. 밀도 8은 유지한다.

- **선을 줄인다.** 카드 테두리와 내부 구분선을 동시에 쓰지 않는다. 표는 세로 괘선 없이 가로선만, 그것도 `--border` 한 톤
- **면을 줄인다.** 배경색을 가진 블록은 화면당 최대 2종. 카드 안에 또 카드를 넣지 않는다
- **색을 줄인다.** 기본 화면은 무채색 + `--fg` 계열만. 유채색은 상태 표시와 primary 버튼에만 등장. 화면 하나에 유채색 3개 이상 금지
- **뱃지를 줄인다.** 상태가 정상일 때는 뱃지를 그리지 않는다. 이상할 때만 표시
- **제목을 줄인다.** 카드마다 제목을 달지 않는다. 내용으로 자명하면 생략
- **정렬로 말한다.** 구분선·배경 대신 **정렬과 간격**으로 그룹을 만든다 (§6 `whitespace-balance`)
- 숫자는 크게, 레이블은 작게. 지표 카드는 값 24px / 레이블 12px `--fg-muted`
- 아이콘은 내비게이션과 상태에만. 장식용 아이콘 금지

## 8. 컴포넌트 필수 기준

- 클릭 요소 최소 **44×44** 히트 영역, 간격 8px 이상 (§2)
- **포커스 링 필수** — `outline: 2px solid var(--ring); outline-offset: 2px`. 제거 금지 (§1 `focus-states`)
- 입력에 **보이는 레이블**. placeholder만으로 대체 금지 (§8 `input-labels`)
- 오류는 **해당 필드 바로 아래**, 원인 + 해결 방법 함께 (§8 `error-placement`, `error-clarity`)
- 비동기 버튼은 처리 중 비활성 + 스피너 (§2 `loading-buttons`)
- 300ms 초과 로딩은 **스켈레톤** (§7 `loading-states`)
- 아이콘은 **Lucide 또는 Phosphor 한 세트**, stroke 1.5px 통일. **이모지 금지** (§4 `no-emoji-icons`)
- 트랜지션 **150–200ms**, `transform`/`opacity`만 (§7)
- 화면당 **primary 버튼 1개** (§4 `primary-action`)

## 9. 통과 조건 (크리틱 체크리스트)

기계 검증 — 하나라도 걸리면 미달:

```bash
grep -rn "gradient\|backdrop-filter" web/ui/src/styles/   # 스크림 외 0건
grep -rn "overflow-x" web/ui/src/                          # 0건
grep -rn "Plus Jakarta\|Poppins" web/ui/src/               # 0건
```

- 375 / 768 / 1024 / 1440 전 폭에서 `scrollWidth === innerWidth`
- 본문 텍스트 대비 **7:1 이상** 실측 (AAA)
- 포커스 링이 모든 인터랙티브 요소에 보임 — 키보드 Tab으로 확인
- 각 화면 1차 블록 7개 이하
- 사이드바: 레일 상주 → 클릭 확장 → 현재 위치 표시 → 새로고침 후 상태 유지
- `prefers-reduced-motion: reduce`에서 애니메이션 정지
- 이모지 아이콘 0건
