# Gauntlet Loop — 정책 시뮬레이션 웹 콘솔

> 이 문서는 Gauntlet Loop의 **단일 기준서**다. 리드·빌더·크리틱 에이전트 전원이 이 문서를 읽고 시작한다.
> 방법론 출처: https://somethingbig.ai/gauntlet-loop

---

## 1. 목표 (destination)

> 정책 담당자가 **정책을 설계하고 → 시뮬레이션이 돌아가는 것을 지켜보고 → 결과를 해석하는** 전 과정을,
> 터미널이나 파일 탐색기를 거치지 않고 하나의 웹에서 끊김 없이 수행할 수 있게 한다.

현재 디자인 범위는 **PC 전용**이다. 1440×900을 주 기준, 1280×800을 최소 기준으로 삼는다.
모바일·태블릿 대응은 사용자 지시에 따라 이번 건틀렛의 구현 범위와 통과 기준에서 제외한다.

경로는 지정하지 않는다. 아래 세 화면이 **하나의 제품처럼** 느껴져야 한다는 것이 유일한 형태 제약이다.

| 화면 | 사용자가 하는 일 | 현재 상태 |
|---|---|---|
| 정책 설정 | 지급 분위·금액·기간·사용처 제한을 정하고 검증받는다 | `data/neo4j_load/policies/*.json` 수기 편집 |
| 실행 모니터 | 시뮬레이션 진행·병목·실패를 실시간으로 본다 | `tail -f run_*.log` |
| 결과 시각화 | 3D 지도와 기존 산출물을 탐색한다 | 147MB 단독 HTML 파일 수동 열기 |
| 보고서 생성 | DASOL 분석 카탈로그에서 분석을 선택하고 최종 HTML 보고서를 생성·열람한다 | `scripts/report/menu.py` 수동 실행 |

### 명시적 비목표

- **시뮬레이션 엔진(`scripts/sim/*`, `scripts/neo4j_load/*`) 코드 변경 금지.** 이번 작업은 웹 프로그래밍만 한다.
  콘솔은 기존 산출물을 **읽고, 띄우고, 프로세스를 기동**할 뿐이다. 엔진 로직에 손대야 할 것 같으면 멈추고 보고한다.
- 인증·다중 사용자·권한 관리. 단일 연구팀 내부 도구다.
- 기존 3D 시각화의 재작성. **감싸서 통합**하되 렌더링 코드는 건드리지 않는다.

---

## 2. 기준 (the bar)

Gauntlet Loop의 핵심은 "훌륭함"을 **에이전트가 직접 열어볼 수 있는 실물**로 두는 것이다.
추상적 형용사("깔끔하게", "직관적으로")는 기준이 아니다.

### 2.1 시각 품질 기준 — 운영 콘솔형 디자인으로 전면 재정의

- `scripts/sim/visualization_3d/static/styles.css` (12KB)
- `scripts/sim/visualization_3d/template.html`
- `output/sim/report/FINAL_REPORT_5D_FULL.html` (601KB)

크리틱은 새 화면을 위의 저장소 내부 실물과 나란히 열어 비교한다.
단, 기존의 "어두운 AI 대시보드" 방향은 폐기한다. 새 콘솔은 Palantir Foundry/Gotham의
운영 소프트웨어 철학을 참고한 **객체·상태·행동 중심의 연구 운영 도구**여야 한다.
Palantir 공식 자료에서 참고하는 요소는 workspace sidebar, 상단 작업 도구, 중앙 작업영역,
우측 상세 패널, 저장·이력·승인 상태, 데이터/온톨로지 객체를 중심으로 한 탐색 구조다.
Palantir의 시각 자산을 복제하거나 브랜드를 모방하지 않고 정보 구조와 운영 원칙만 참고한다.

참고 자료:

- [Palantir Foundry](https://www.palantir.com/platforms/foundry/)
- [Palantir Gotham](https://www.palantir.com/platforms/gotham/)
- [Foundry workspace navigation](https://www.palantir.com/docs/foundry/getting-started/orientation-and-nav)
- [Solution Designer navigation](https://www.palantir.com/docs/foundry/solution-designer/navigation)
- [Integrated platforms and workspace](https://www.palantir.com/docs/foundry/architecture-center/platforms)

#### 새 디자인 규칙

1. **운영 화면 우선**

   첫 화면은 제목·히어로·마케팅 카피가 아니라 현재 run, 정책 상태, 데이터 신선도,
   실패·경고·진행률을 바로 보여준다. 모든 패널은 사용자가 다음 행동을 결정하도록 돕는
   실제 데이터나 상태를 가져야 한다.

2. **고정 workspace 구조**

   - 좌측 216~240px 고정 navigation rail: `정책`, `실행`, `결과`, `데이터 계약`, `시스템`
   - 상단 44~52px context bar: 현재 run·정책·데이터 시점·저장 상태·lock 상태
   - 중앙 primary workspace: 표·타임라인·지도·검증 결과가 주 콘텐츠
   - 우측 320~380px inspector: 선택한 정책·일자·에이전트·오류의 상세와 원본 근거
   - 화면 전환보다 동일 셸 안의 context 유지가 우선이며, 사용자가 어디에서 무엇을 보고 있는지 항상 표시한다.

3. **객체와 상태를 시각적 중심으로 둔다**

   정책, run, day, agent, POI, metric을 모두 명확한 object label로 표현한다.
   숫자는 단독 KPI 카드로 부풀리지 않고 `값 → 기준일 → 출처 → 상태`가 한 행 또는 한 패널에서
   함께 보이게 한다. 수치 옆에 `metrics/day_*.jsonl`, `summary.json`, `checkpoint` 등
   데이터 출처를 열 수 있는 링크를 둔다.

4. **색은 의미가 있을 때만 사용한다**

   바탕은 중성 흑색/차콜, 표면은 한 단계 밝은 회색, 보더는 저채도 헤어라인으로 둔다.
   accent는 상태 의미에만 배정한다: 정상=muted green, 진행=muted cyan, 경고=amber,
   실패/중단=muted red, 선택 상태=차분한 blue. 네온 cyan, 보라색 AI 강조, 무지개 팔레트,
   장식용 색칠 버튼은 사용하지 않는다.

5. **기하와 타이포그래피**

   - 패널·버튼·입력은 0~4px radius만 허용한다.
   - 1px border와 구분선으로 계층을 만들고 큰 shadow는 사용하지 않는다.
   - 배경·버튼·카드 어디에도 `linear-gradient`, `radial-gradient`, `backdrop-filter`를 사용하지 않는다.
   - 기술적인 sans-serif와 monospace 숫자 조합을 사용한다. 둥근 SaaS 폰트와 굵은 Inter/Poppins 계열은 금지한다.
   - 제목은 짧고 작게, 데이터 행과 표의 정보 밀도를 우선한다.

6. **행동은 명시적이고 추적 가능해야 한다**

   `정책 검증`, `JSON 미리보기`, `실행 시작`, `중단`, `결과 열기` 같은 동작은 이름이 분명한
   outline/neutral control로 표현한다. 실행·중단·정책 저장은 확인과 결과 상태를 남긴다.
    사람이 승인해야 하는 지점은 자동으로 숨기거나 AI 아이콘으로 대체하지 않는다.

7. **고담형 정보작전 UI/UX와 탑다운 탐색**

   목표 수준은 일반적인 AI 대시보드가 아니라, Gotham과 같은 정보작전 플랫폼의 임무 수행형
   UI/UX다. 여기서 “CIA 수준”은 특정 기관의 내부 화면을 복제한다는 뜻이 아니라, 정보기관용
   분석 도구에 요구되는 높은 수준의 맥락 보존·근거 추적·다중 단계 탐색·판단 지원을 뜻한다.
   브랜드 자산, 내부 화면, 로고, 비공개 디자인을 모방하지 않는다.

   - 탐색 순서는 `임무/전체 현황 → 정책·run/작전 → stage·day/국면 → agent·POI·metric/객체 → 원본 로그·checkpoint·근거`의 탑다운 계층을 따른다.
   - 상위 화면은 결론을 단정하지 않고 현재 상태·우선순위·이상 징후·다음 판단 지점을 제시한다. 사용자가 하위 객체를 선택할 때만 세부 정보와 원본 근거를 점진적으로 펼친다.
   - 지도·타임라인·객체 목록·관계/이벤트·근거 패널은 서로 다른 분석 관점이다. 모든 것을 한 화면에 욱여넣지 말고, 사용자의 작업 흐름에 따라 별도 workspace/route/panel로 분리한다.
   - 화면 이동 후에도 현재 임무·run·정책·기준일·필터·선택 객체를 context bar와 breadcrumb로 보존한다. 사용자는 언제든 상위 수준으로 올라가거나 하위 근거로 내려갈 수 있어야 한다.
   - “예쁜 요약 카드”보다 상황 파악→의심 지점 선택→비교→근거 확인→행동 결정의 흐름을 우선한다. 고밀도 표, 시간축, 객체 상태, provenance가 핵심 구성요소다.
   - Critic은 각 화면에 대해 “현재 수준에서 무엇이 미흡한가”뿐 아니라 “다음 분석 수준으로 올라가기 위해 어떤 정보 구조·상호작용·근거 연결이 필요한가”를 계속 제안한다.

8. **정보 밀도와 점진적 공개**

   현재 실행 모니터는 context bar, run 상태 카드, 일자별 표, 단계 병목, 실행 제어,
   선택 일자 inspector를 한 화면에 동시에 밀어 넣어 정보 우선순위가 무너진다. 이것은
   디자인 취향이 아니라 **과밀 상태라는 명시적 결함**으로 취급한다.

   - 한 workspace는 한 가지 주된 판단을 지원한다. 첫 화면은 전체 상황·우선순위·다음 행동만 보여준다.
   - 상세 정보는 다음 계층으로 분리한다: `L0 임무 현황 → L1 정책/run → L2 stage/day → L3 객체/지표 → L4 원본 근거`.
   - 실행 모니터는 `개요`, `일자·단계 상세`, `실패·병목 분석`, `원본 로그·metrics` workspace로 나눈다. 모든 패널을 첫 viewport에 고정하지 않는다.
   - 결과는 `보고서 허브`, `보고서 생성`, `보고서 열람`, `3D 결과`, `근거·부록`을 별도 route/workspace로 나눈다.
   - 우측 inspector는 선택한 한 객체의 상세만 보여준다. 여러 객체의 상세를 한 번에 나열해 화면을 채우지 않는다.
   - 화면에는 현재 위치와 상위로 올라가는 breadcrumb를 유지하되, 하위 정보는 사용자가 선택했을 때만 펼친다.

9. **라이트·다크 테마를 동등한 시스템으로 설계한다**

   라이트 테마를 임시 색상 반전으로 만들지 않는다. 모든 색은 CSS 토큰으로 정의하고
   `data-theme="light"`와 `data-theme="dark"` 양쪽에 값을 제공한다.

   - 라이트 기본값: 백색에 가까운 표면, 따뜻한 회색 canvas, 진한 navy/charcoal 텍스트, 1px hairline border.
   - 다크 선택값: charcoal canvas, 한 단계 밝은 surface, 동일한 정보 위계와 상태색.
   - 테마 전환은 context bar의 명시적 컨트롤로 제공하고 새로고침 후에도 유지한다.
   - 정상·진행·경고·실패·선택 상태의 의미색은 테마가 바뀌어도 의미와 대비를 유지한다. WCAG AA 대비를 확인한다.
   - 밝은 테마에서도 큰 그림자·그라디언트·둥근 SaaS 카드·장식용 색상은 사용하지 않는다.

10. **금지 목록**

   다음 중 하나라도 있으면 §2.1 미달로 기록한다: gradient, glassmorphism, 큰 rounded card,
   큰 shadow, hero section, 카드 3~4개만으로 화면 구성, 챗봇 중심 홈 화면, 로봇·뇌·sparkle
    아이콘으로 AI를 장식하는 UI, 근거 없는 hardcoded KPI, 의미 없는 애니메이션, 한 화면에
    서로 다른 판단 단계의 정보를 모두 배치하는 mega-dashboard.

Critic은 CSS를 `rg -n "gradient|backdrop-filter|border-radius|box-shadow" web/`로 검사하고,
폰트·색상 토큰·레이아웃 스크린샷을 기준 실물과 대조한다. 시각적 취향이 아니라 위 규칙과
실제 데이터 밀도·운영 흐름으로 비판한다. 기준을 통과한 화면도 그 상태를 종착점으로 보지
않고, 더 높은 정보 밀도·맥락 연결·판단 가능성·근거 추적성을 확보할 수 있는지 다시 묻는다.

#### 실제 브라우저 검증 전제

S6 Critic은 소스 코드와 서버 응답만 보고 화면을 통과시킬 수 없다. 실제 브라우저에서 다음을
직접 확인하고 증거를 남겨야 한다.

- `http://127.0.0.1:8000/` 또는 명시된 실제 실행 URL을 브라우저로 열 것
- 1440×900·1280×800 PC viewport에서 스크린샷을 생성할 것
- 정책 설정 → 실행 → 모니터 → 결과 → 원본 근거의 탑다운 이동을 실제로 수행할 것
- 콘솔 오류, 네트워크 오류, 수평 overflow, 클릭 불가 상태, 새로고침 복원을 확인할 것
- 스크린샷·콘솔·네트워크 결과를 `docs/gauntlet/evidence/<stage>-<round>/`에 보관할 것

Codex Desktop의 연결 브라우저/CDP를 사용할 수 있으면 그것을 사용한다. CLI에서는
Playwright와 Chromium으로 브라우저를 직접 기동하는 검증 harness를 사용한다. 연결 브라우저도
Playwright harness도 없으면 화면을 실제로 보지 못한 상태이므로 S6을 통과시킬 수 없다.
이 상태는 `BROWSER_BLOCKED`로 기록하며, 코드가 빌드되고 HTTP 200을 반환한다는 이유만으로
시각 검증을 대신하지 않는다.

CLI의 기본 검증 명령은 다음과 같다.

```powershell
Push-Location web/ui
npm install
npx playwright install chromium
npm run gauntlet:screen
Pop-Location
```

검증기는 FastAPI를 필요하면 로컬에서 기동하고, 정책·모니터·결과·계약·시스템 workspace를
1440×900·1280×800에서 직접 열어 스크린샷과 `manifest.json`을
`docs/gauntlet/evidence/s6-screen/`에 남긴다. Critic은 이 증거를 읽지 않고 “브라우저를
사용했다”고 주장할 수 없다.

#### S6 UI/UX 스킬 전제

S6 전담 Builder와 UI Critic은 `ui-ux-pro-max` 스킬을 사용해 정보 구조, 시각 계층,
PC 폭별 상태, 접근성, 테마 토큰을 설계·검토한다. 이 스킬은 장식용 스타일 생성기가 아니라
Gotham형 임무 workspace의 품질을 높이기 위한 보조 기준이다. §2.1의 실제 데이터·탑다운
탐색·과밀 방지·light/dark 규칙보다 우선하지 않는다.

Claude Code CLI에서 작업을 시작하기 전에 다음을 실행한다.

```text
/plugin marketplace add nextlevelbuilder/ui-ux-pro-max-skill
/plugin install ui-ux-pro-max@ui-ux-pro-max-skill
```

설치되지 않았거나 스킬을 호출할 수 없는 환경에서는 사용했다고 주장하지 않는다. 이 경우
Critic은 `UI_UX_SKILL_BLOCKED`로 기록하고, §2.1 기준과 실제 브라우저 증거만으로 진행 여부를
사람에게 보고한다.

### 2.2 기능 기준 — 통과/실패가 갈리는 검사

| # | 기준 | 검사 방법 |
|---|---|---|
| B1 | 목업 데이터 금지 | 모든 수치가 실제 `metrics/day_*.jsonl` / `checkpoints/*.json`에서 나온다. 하드코딩된 숫자 0개 |
| B2 | 정책 검증 일치 | 화면 검증 결과가 `scripts/sim/policy_preflight.py` 실행 결과와 100% 일치 |
| B3 | 실행 중 안전 | 콘솔의 어떤 조작도 돌고 있는 시뮬레이션을 죽이거나 지연시키지 않는다 |
| B4 | 첫 화면 2초 | 진행 중인 run이 있을 때 모니터 화면 첫 렌더까지 2초 이내 |
| B5 | 대용량 내성 | 19MB짜리 `day_*.jsonl`을 읽어도 브라우저가 멈추지 않는다 (서버 측 집계) |
| B6 | 중단 복원 | 페이지를 새로고침해도 진행 상태가 그대로 복원된다 |
| B7 | PC 해상도 | 1440×900 / 1280×800 두 화면에서 가로 스크롤과 중첩 세로 스크롤 없이 동작 |
| B8 | 중복 실행 차단 | 이미 run이 돌고 있을 때 실행이 **물리적으로 불가능**해야 한다. UI 비활성화만으로는 미달 — 서버 측 lock으로 막고, lock 보유자·시작시각을 화면에 표시 |

> **B8이 왜 치명적인가.** 2026-08-02 18:59, 두 번째 `chain_p2.sh`가 기동되면서 첫 단계인 `neo4j stop`이
> 2시간 42분째 돌던 시뮬레이션의 DB 연결을 끊어 죽였다. Day 0의 4,500/7,500 지점이었고 연산은 전량 소실됐다.
> 콘솔에 실행 버튼을 다는 순간 이 사고의 재현 난이도가 급격히 낮아진다. **lock 없는 실행 기능은 반려한다.**

### 2.3 데이터 계약 기준

콘솔이 읽어야 하는 실제 산출물의 형태는 다음에서 확인한다.

```
C:\Users\srdyh\gpu_exp_data\20260802\
  out_BASE\      완료된 7일 run (metrics/ checkpoints/ timing/ summary.json events.jsonl)
  out_FINAL\     장기 run — 8/17까지 장기 시계열
  rescue\        중단된 run — Day 0만 존재하는 부분 상태
  logs_scripts\  run_*.log 42개 + chain_*.sh
```

`rescue/`는 특히 중요하다. **불완전한 run을 어떻게 표시할 것인가**의 실물 테스트 케이스다.

### 2.4 DASOL 최종 보고서 생성 기능 기준

`origin/dasol` 브랜치의 보고서 생성 기능을 웹에서 사용할 수 있어야 한다. 소스 구조를
다시 작성하지 않고, 아래 기존 흐름을 웹 API의 백그라운드 작업으로 감싼다.

```text
정책 JSON + run 범위
        ↓
scripts/report/catalog.py
  적용 가능한 분석 자동 판정
        ↓
scripts/report/menu.py
  분석 선택 또는 --all
        ↓
scripts/report/narrate.py + 기존 generate_final_report 계산 함수
        ↓
scripts/report/build_html.py
  차트가 포함된 self-contained FINAL_REPORT.html
```

웹 기능은 다음을 제공한다.

- `보고서 허브`: run·정책·데이터 기준일·기존 보고서 목록과 생성 이력
- `보고서 생성 workspace`: run, 시작일, 기간, 정책 시행일, 분석 항목, 인터뷰 포함 여부를 단계적으로 선택
- 분석 카탈로그가 적용 불가능하다고 판단한 항목은 메뉴에서 비활성화하고 이유를 표시한다. UI가 임의로 DID를 강제하지 않는다.
- 생성 중에는 별도 report job 상태·로그·진행 단계를 보여주며, 시뮬레이션 실행 화면을 가리지 않는다.
- 생성 후 `FINAL_REPORT.html`, Markdown, 차트·부록을 원본 경로와 함께 열람한다. 브라우저에 대용량 원본을 한 번에 적재하지 않는다.
- 보고서의 모든 숫자는 생성된 JSON·metrics·checkpoint·차트 파일의 출처 링크를 가진다. 해설 LLM은 숫자를 새로 만들지 않는다.
- 보고서 생성은 실제 run snapshot에 대해서만 수행한다. 실행 중인 mutable 산출물을 동시에 읽어 결론을 만들지 않으며, 필요한 경우 대기 상태로 둔다.
- 현재 프로젝트의 국내 모델 제약을 따른다. 모델 선택 UI에서 외국 모델을 노출하지 않고, 서버 측 EXAONE 설정만 사용한다.

웹 API는 임의의 shell command를 받아 실행하지 않는다. S2가 `menu.py`의 인자와 허용된
산출물 root를 구조화된 request로 검증하고, report lock·실행 상태·출력 경로를 서버에서 통제한다.
보고서 생성 기능이 simulation lock을 우회하거나 실행 중인 GPU/Neo4j 작업을 방해하면 반려한다.

보고서 job은 `completed` 표시만으로 충분하지 않다. 서버는 선택한 run의
`summary.json`, `events.jsonl`, `poi_summary.json`, 요청 범위의 `metrics/`, `timing/`,
`checkpoints/`를 immutable manifest(SHA-256)로 고정하고, `menu.py`가 계산 직전에 같은
파일을 다시 검증해야 한다. 기존 계산 함수가 Neo4j를 읽는 동안에는
`DASOL_NEO4J_RUN_ID`가 선택한 run ID와 일치하지 않으면 생성하지 않는다. 비밀번호만
설정된 live DB는 snapshot 근거로 간주하지 않는다.

---

## 3. 분해 (독립적으로 개선 가능한 조각)

각 조각은 다른 조각을 기다리지 않고 개선될 수 있어야 한다.
조각 간 결합은 **S1이 확정한 데이터 계약**으로만 이뤄진다.

| ID | 조각 | 현재 검증 기준 | 선행 |
|---|---|---|---|
| S1 | 데이터 계약 + 픽스처 | jsonl → API 스키마 확정. 3종 run으로 픽스처 생성. 나머지 조각이 이걸로 독립 개발 가능 | — |
| S2 | 백엔드 API | 정책 CRUD·검증, run 목록·상태·진행률, 실행/report lock, report job, 산출물 서빙. B2·B3·B5·B8·B11 통과 | S1 |
| S3 | 정책 설정 화면 | 10분위 지급액·기간·사용처 제한 입력, 실시간 검증, JSON 미리보기. B2·B7 통과 | S1 |
| S4 | 실행 모니터 대시보드 | 일자별 진행, Stage1/Stage2/Dawn 병목, 실패율, ETA, 실행 제어. B1·B4·B6·B8 통과 | S1 |
| S5 | 결과 workspace + 보고서 생성 통합 | 기존 3D·리포트 열람과 DASOL 보고서 생성 job을 분리된 workspace로 제공. 렌더링·분석 계산 코드는 재작성하지 않음. B7 통과 | S1 |
| S6 | 앱 셸 + 디자인 시스템 | Palantir-inspired workspace 셸, 탑다운 navigation, 정보 분리, light/dark 토큰, 인스펙터, 로딩·빈 상태·에러 상태. §2.1 기준 | — |

**S1이 선행이다.** S1이 확정되기 전에 S2~S5를 시작하면 계약이 흔들려 재작업이 발생한다.
S6은 데이터에 의존하지 않으므로 S1과 병렬로 시작한다.

---

## 4. 빌더 / 크리틱 프로토콜

### 규칙

1. **빌더와 크리틱은 컨텍스트를 공유하지 않는다.** 크리틱은 빌더의 대화 이력·의도·변명을 보지 않고, 산출물과 이 문서만 본다.
2. **크리틱은 결함 탐지기가 아니라 지속 개선자다.** 기능이 작동하고 최소 기준을 통과해도 비판을 멈추지 않는다. 정보 구조, 사용 흐름, 표현의 명료성, 근거 연결, 성능, 확장 가능성에서 현재 수준을 넘어설 미흡함을 찾아낸다.
3. **크리틱은 코드를 고치지 않는다.** 대신 가장 영향력이 큰 개선 목표와 검증 가능한 수정 방향을 제시한다. 수정은 빌더가 하고, 개선 결과의 판정은 새 검토에서 다시 한다.
4. **크리틱은 요약이 아니라 실물을 본다.** 실행 가능한 웹은 직접 실행하고, API는 실제 응답을 요청하고, 화면은 지정 viewport에서 렌더링하며, 수치는 원본 산출물까지 추적한다.
5. **`PASS`는 종료가 아니다.** PASS는 현재 최소 기준을 충족했다는 뜻일 뿐이다. Critic은 반드시 `남은 미흡함`, `다음 수준의 목표`, `왜 지금 수정할 가치가 있는지`를 남긴다.
6. **같은 결함과 같은 전략을 반복하지 않는다.** 동일한 문제가 두 번 재현되면 Lead는 접근법을 바꾸거나 사람에게 에스컬레이션한다.
7. **지속성에는 경계가 필요하다.** 품질 개선을 계속 시도하되, 실행 시간·비용·권한·안전 경계에 도달하면 성공으로 기록하지 않고 `BOUNDED_STOP`으로 기록한다.

### 크리틱 보고 형식

```
조각: S4
기준 위반:
  - B1: hud 상단 "평균 응답 2.3초"가 하드코딩. metrics 파일에 해당 필드 없음
  - §2.1: 패널 여백이 styles.css의 legacy-panel(16px)과 불일치(8px), 밀도가 튐
미흡한 점:
  - 현재 상태 표는 읽히지만 정책 → run → day → bottleneck으로 내려가는 분석 경로가 끊겨 있음
  - inspector에 원본 checkpoint 링크는 있으나 선택한 metric과의 provenance 연결이 약함
다음 수준의 개선 목표:
  - 상위 run context를 유지한 채 bottleneck을 선택하고 원본 이벤트까지 2단계 이내로 내려갈 수 있게 구성
통과:
  - B4 (1.2초), B6, B7
판정: REOPEN — B1은 필수 수정이고, 통과 항목도 다음 개선 라운드로 유지
```

### 4.1 에이전트 배치와 건틀렛 루프

개발은 **S1~S6 전담 빌더**, 전체 범위를 조정하는 **Lead**, 산출물만 보고 판정하는 **독립 Critic**의 세 역할로 운영한다. 에이전트 수를 늘리는 것이 목적이 아니라, 각 조각의 책임과 검증 경계를 분리해 계약·기능·화면이 서로 오염되지 않게 하는 것이 목적이다.

| 역할 | 책임 | 금지 사항 |
|---|---|---|
| 총괄 Lead | S1 계약 잠금, 의존성·범위 결정, 크리틱 결과 분류, 통합 판정, 다음 조각 개방 | 기준을 무시한 강행, 시뮬레이션 엔진 직접 수정, 근거 없는 “통과” 판정 |
| S1 전담 빌더 | 실제 산출물에서 JSONL·체크포인트·요약·로그 계약 작성, 3종 run 픽스처와 API 스키마 생성 | UI/API를 먼저 구현, 가짜 값·임의 KPI 삽입 |
| S2 전담 빌더 | FastAPI의 정책·run·실행/report lock·SSE·report job·산출물 API 구현 | 엔진·GPU 실행 로직 변경, mock fallback 추가, 임의 shell command 실행 |
| S3 전담 빌더 | 10분위 정책 설정, 사전 검증, JSON 미리보기, 오류 상태 구현 | 정책 효과 방향을 프롬프트나 UI에서 유도 |
| S4 전담 빌더 | 실행 모니터, Stage1/Stage2/Dawn 병목, 실패·ETA·재개·lock 상태 구현 | 과도한 폴링, 프로세스 강제 종료, 지연·KPI 하드코딩 |
| S5 전담 빌더 | 기존 3D 뷰와 리포트를 앱 셸 안에 통합 | 기존 렌더링 코드 재작성, 브라우저에 대용량 원본 직접 적재 |
| S6 전담 빌더 | `ui-ux-pro-max`를 사용한 Palantir-inspired workspace 셸, 탑다운 navigation, 정보 분리, light/dark 토큰, 인스펙터, 상태·오류·빈 상태 구현 | AI 장식, 그라디언트, 글래스모피즘, 마케팅형 히어로 화면 |
| 독립 Critic | 새 컨텍스트에서 실물을 직접 검사하고 결함·미흡함·다음 수준의 개선 목표를 지속적으로 발굴 | 코드 수정, 빌더 의도 참작, 최소 기준 통과를 종료로 선언 |

#### 반복 순서

1. Lead가 해당 조각의 입력·범위·이번 라운드 검증 기준·재현 명령을 고정한다.
2. 전담 빌더가 자기 조각만 구현하고 변경 파일, 테스트, 픽스처/API 예시, 알려진 한계를 남긴다.
3. 독립 Critic이 빌더의 대화 이력 없이 결과물을 검토한다.
4. Critic은 `결함`, `미흡함`, `다음 수준의 개선 목표`를 각각 기록하고 가장 영향력이 큰 하나를 우선순위로 지정한다.
5. Lead는 `치명적`, `필수 수정`, `품질 향상`, `통합 검토`로 분류한다. `PASS`가 있어도 품질 향상 항목이 남아 있으면 조각을 닫지 않는다.
6. 빌더가 이전과 다른 전략으로 수정한 뒤 **새 컨텍스트의 Critic이 재검토**한다. 실패한 조각은 선행 조건으로 간주하지 않는다.
7. Lead가 라운드·증거·남은 미흡함·다음 목표를 기록한다. 기준 통과만으로 다음 조각을 자동 개방하지 않는다.

S1의 계약 버전이 잠기기 전에는 S2~S5의 구현을 통합하지 않는다. S6은 데이터 계약에 의존하지 않으므로 S1과 병렬로 개발할 수 있다. 각 조각의 개선 라운드가 끝난 뒤에는 전체 흐름을 처음부터 사용하는 **독립 Integration Critic**을 수행한다. Integration Critic은 단순한 용어·여백 스무딩만 하지 않고, 상위 임무 화면에서 하위 근거 화면까지 탑다운 흐름 전체에서 남은 미흡함을 다시 찾아낸다.

#### 에이전트 간 전달물

Lead가 각 빌더에 전달하는 최소 컨텍스트는 다음으로 제한한다.

- `docs/GAUNTLET_WEB_CONSOLE.md`
- 해당 조각의 계약 문서 `docs/gauntlet/contracts/<stage>.md`
- `web/fixtures/<stage>/`의 실제 픽스처
- 작업 파일 범위, B 기준, 재현·검증 명령

빌더는 다음을 남긴다.

- 변경 파일과 변경 이유
- 실행한 검증 명령과 결과
- JSONL/API/화면 상태의 실제 예시
- 미해결 문제와 다음 Critic이 확인할 지점

Critic은 `docs/gauntlet/critics/<stage>-<round>.md`에 보고서를 작성하고, Lead는 `docs/gauntlet/gates/<stage>.json`에 판정·커밋·Critic 경로·계약 버전·남은 미흡함·다음 개선 목표를 기록한다. 이 파일들이 있어야 CLI에서 다음 라운드를 재현할 수 있다.

### 4.2 Critic의 지속 개선 의무

Critic은 “문제가 없다”를 쉽게 선언하지 않는다. 각 라운드에서 다음 순서를 지킨다.

1. 실제 산출물과 §2 기준·실제 참고물·사용자 작업 흐름을 직접 대조한다.
2. 이미 통과한 부분도 현재 수준에서의 미흡함을 찾는다. 단, 사소한 취향이 아니라 사용자 판단·탐색·근거 추적·안전·성능에 영향을 주는 개선이어야 한다.
3. 남은 개선점을 영향도 순으로 정렬하고, 가장 큰 하나를 다음 Builder 작업으로 돌려보낸다.
4. 개선 전후를 같은 테스트·viewport·데이터·참고물로 비교한다. 변화가 없거나 퇴보하면 해당 라운드를 실패로 기록한다.
5. 두 번 연속 의미 있는 개선이 없을 때만 `INTEGRATION_REVIEW_REQUESTED`를 제안할 수 있다. 이는 완료 선언이 아니라 전체 Integration Critic에게 검토를 요청하는 상태다. 이때도 남은 한계와 더 높은 수준의 가능성을 보고서에 남긴다.

이 문서에는 에이전트가 작업 종료를 선언하는 상태가 없다. 기준 통과는 해당 라운드의 검증 결과이자 다음 분석·개선 단계로 넘어가기 위한 조건일 뿐이다. 전체 Integration Critic에게 검토를 요청한 뒤에도 인간이 작업을 멈추기 전까지 Critic은 남은 미흡함과 다음 개선 가능성을 계속 기록한다. 예산·시간·권한 때문에 멈추면 `BOUNDED_STOP`으로 남기며 성공으로 위장하지 않는다.

### 통합 일관성 패스

각 조각이 `INTEGRATION_REVIEW_REQUESTED` 상태로 전환된 뒤에도 **전체 일관성 검토**를 수행한다.
독립 개선의 부작용(용어 불일치, 여백 편차, 색상 드리프트)뿐 아니라 탑다운 탐색 흐름의 단절,
상위 상태와 하위 근거의 불일치, 전체 제품 수준에서의 미흡함을 다시 비판한다. 문제가 발견되면
해당 조각으로 되돌아가 새 개선 라운드를 연다.

---

## 5. 스택

| 계층 | 선택 | 비고 |
|---|---|---|
| 백엔드 | **FastAPI** (Python 3.11) | 저장소와 동일 런타임. 기존 `policy_preflight.py`를 서브프로세스로 재사용 |
| 프론트 | **React + Vite + TypeScript** | 실시간 모니터의 상태량이 많아 선택 |
| 실시간 | SSE (Server-Sent Events) | 폴링보다 가볍고, 단방향이라 WebSocket 불필요 |
| 배포 | Docker Compose | 기존 `docker-compose.prod.yml` 패턴 답습 |

디렉터리는 `web/` 아래에 둔다. 기존 `scripts/`는 건드리지 않는다.

```
web/
  api/        FastAPI
  ui/         Vite + React + TS
  fixtures/   S1이 생성한 픽스처
```

---

## 6. 실행 환경 / 배포

### 배포 대상

**`43.201.218.176`** — 콘솔(FastAPI + 정적 번들)이 여기서 돈다.

> 아직 이 호스트의 접속 수단(SSH 키·사용자·포트)과 기존 점유 서비스를 확인하지 않았다.
> **실제 배포 전에 사람 확인을 받는다.** 빌더는 배포 설정 파일까지만 만들고 배포를 실행하지 않는다.

시뮬레이션 자체는 GPU 서버(`proxy.tta-gpu.gov-nhncloud.com:30044`)에서 계속 돈다.
콘솔 호스트와 GPU 서버는 별개이므로, 콘솔이 GPU 서버의 산출물에 어떻게 접근할지가 S2의 설계 항목이다.

### GPU 서버 취급 규칙 (엄수)

- **읽기 전용 조회만.** 실행 중인 시뮬레이션을 절대 방해하지 않는다 (B3).
- 서버에서 명령을 돌릴 때 인자에 `run_simulation.py` 문자열을 넣지 않는다.
  `chain_p2.sh`의 `pgrep -f "run_simulation[.]py"` 감시 루프에 걸려 체인이 진행하지 못한다.
  대괄호 이스케이프(`[r]un_simulation`)를 쓴다.
- `pkill`·`kill`·`rm`·`mv`를 GPU 서버에서 실행하지 않는다.

### 로컬 개발

`C:\Users\srdyh\gpu_exp_data\20260802\`의 실제 산출물로 한다.
**서버 접속 없이 전 조각을 개발·검증할 수 있어야 한다.**

### PowerShell에서 웹 콘솔 열기

#### 이미 빌드된 번들 실행

FastAPI가 `web/ui/dist/`를 정적으로 제공하므로, 별도 프론트 서버 없이 8000번 포트에서
완성 번들을 연다.

```powershell
$Project = "C:\Users\srdyh\OneDrive\사진\바탕 화면\agent_simulate\kw_26_final_project\.claude\worktrees\silly-gagarin-0b181a"

Set-Location -LiteralPath $Project
python -m uvicorn web.api.app:app --host 127.0.0.1 --port 8000
```

다른 PowerShell 창에서 브라우저를 연다.

```powershell
Start-Process "http://127.0.0.1:8000/"
```

#### 프론트 개발 서버 실행

API는 8000번 포트, Vite는 5173번 포트를 사용한다.

```powershell
$Project = "C:\Users\srdyh\OneDrive\사진\바탕 화면\agent_simulate\kw_26_final_project\.claude\worktrees\silly-gagarin-0b181a"

Set-Location -LiteralPath "$Project\web\ui"
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

```powershell
Start-Process "http://127.0.0.1:5173/"
```

#### 브라우저를 직접 기동하는 S6 검증

CLI Critic은 연결된 브라우저를 기다리지 않고 Playwright/Chromium을 직접 기동한다.

```powershell
$Project = "C:\Users\srdyh\OneDrive\사진\바탕 화면\agent_simulate\kw_26_final_project\.claude\worktrees\silly-gagarin-0b181a"

Set-Location -LiteralPath "$Project\web\ui"
npm install
npx playwright install chromium
npm run gauntlet:screen
```

#### DASOL 보고서 생성 CLI 기준

`origin/dasol`의 `scripts/report/`가 현재 작업 트리에 반영된 뒤, 웹 API는 아래와 같은
구조화된 실행을 백그라운드 job으로 감싼다. 사용자가 임의 shell command를 입력하는 방식은
허용하지 않는다.

```powershell
$Project = "C:\Users\srdyh\OneDrive\사진\바탕 화면\agent_simulate\kw_26_final_project\.claude\worktrees\silly-gagarin-0b181a"
Set-Location -LiteralPath $Project
  python scripts/report/menu.py `
    --run-id FINAL `
    --policy-id P008 `
    --start 2026-05-25 `
    --days 4 `
    --policy-json data/neo4j_load/policies/P008.json `
    --policy-from 2026-05-27 `
    --snapshot-manifest output/sim/report/FINAL_REPORT_WEB.snapshot.json `
    --data-root C:\Users\srdyh\gpu_exp_data\20260802 `
    --all `
    --out output/sim/report/FINAL_REPORT_WEB.html
  ```

`snapshot-manifest`는 웹 API가 생성한 파일을 사용한다. 직접 CLI를 실행할 때도 먼저
완료 run에서 같은 manifest를 만들고, Neo4j에는 `DASOL_NEO4J_RUN_ID=FINAL`처럼
동일한 snapshot binding을 명시한다. 둘 중 하나라도 확인되지 않으면 보고서를
생성하지 않고 원본 미확인 상태를 표시한다.

### CLI 작업 시작 권한

권한 전체 허용 모드는 **현재 웹 프로젝트 작업 디렉터리에서만** 사용한다. GPU 서버 접속,
배포 호스트, 운영 DB, 자격증명 경로에서는 이 모드를 사용하지 않는다. 작업 디렉터리를 먼저
절대 경로로 고정한 뒤 CLI를 시작한다.

#### Claude Code CLI

```powershell
$Project = "C:\Users\srdyh\OneDrive\사진\바탕 화면\agent_simulate\kw_26_final_project\.claude\worktrees\silly-gagarin-0b181a"

Set-Location -LiteralPath $Project
claude --dangerously-skip-permissions
```

Claude Code 세션이 열린 뒤 S6 스킬을 설치한다.

```text
/plugin marketplace add nextlevelbuilder/ui-ux-pro-max-skill
/plugin install ui-ux-pro-max@ui-ux-pro-max-skill
```

#### Codex CLI

```powershell
$Project = "C:\Users\srdyh\OneDrive\사진\바탕 화면\agent_simulate\kw_26_final_project\.claude\worktrees\silly-gagarin-0b181a"

Set-Location -LiteralPath $Project
codex `
  --cd $Project `
  --dangerously-bypass-approvals-and-sandbox `
  "docs/GAUNTLET_WEB_CONSOLE.md를 단일 기준서로 사용해 실제 웹 개발을 시작하라. S1부터 구현하고, 기준서에 정의된 지속 개선 Critic·탑다운 workspace·정보 분리·light/dark 테마·DASOL 보고서 생성 연동·실제 브라우저 검증을 실행하라."
```

Codex의 전체 권한 옵션은 `--yolo`로 축약할 수 있지만, 작업 범위가 바뀌지 않도록 위의
절대 경로 고정과 GPU 서버 취급 규칙을 먼저 확인한다.
