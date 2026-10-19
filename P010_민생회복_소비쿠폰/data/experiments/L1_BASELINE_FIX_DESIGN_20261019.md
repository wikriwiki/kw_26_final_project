# L1 기준선 모사 개선 — 업종 예산 봉투와 필요 게이지 (2026-10-19, 결과를 보기 전에 고정)

브랜치 `fix/l1-baseline-fidelity` (출발점: `origin/P010_simulation` a99e5619 = P010·P017 본런 엔진 repo_ab3w_20261012b).
모든 변경은 `EXP_CATEGORY_BUDGET=1` 일 때만 켜진다. 꺼 두면 본런 엔진과 같아야 한다(단위 시험으로 확인).

## 1. 왜 — L1 경계 지도(정책 없는 평소 14일, 본 실행 5건 9,855명, 지출 20만 5천 건)

실측: 서울시 상권분석서비스 추정매출-행정동 2024(계절 맞춘 분기)·서울 생활인구 2026-03 평일. 판정 = 실측과의 가까움(1−JSD)을
단순 모형(균등 / 입력만으로 배분 / 상가 수 비례)·완벽한 모형(같은 건수로 실측에서 뽑기)과 견줌. 결과 파일 `tools/l1_eval/results/`.

| 차원 | 굵은 단위 → 세밀한 단위 (◐ 입력 이상 · ◔ 균등보다 나음 · △ 균등 수준 · ○ 균등보다 못함) |
|---|---|
| 누가 | 성별 △(실측 반반) · 연령대 ◔ · 연령×업종 ○ · 연령×자치구 ◐(합산만) · 연령×행정동 ○ |
| 사람 위치(평일 14시) | 권역 △ · 자치구 ◐ · 행정동 ◐ (기준: 하루 종일 집) |
| 어디서(지출) | 권역 ◔ · 자치구 △ · 행정동 ◔(합산만) · 행정동×업종 ○ — 상가 수 비례는 어느 단위에서도 못 이김 |
| 언제 | 주중/주말 ◔ · 요일 ○ · 시간대 ○ · 업종×시간대 ○ · 업종×요일 ○ · 행정동×시간대 △ |
| 무엇에 | 3분류 ○ · 10묶음 ○ · 63업종 ○ · 자치구×업종 ○ |

소비쿠폰(P010) 정책 없음 갈래: 음식점 지출 73%(실측 24%, **페르소나 입력 28%**), 주말 비중 46%(실측 24~27%, **입력 20%**),
점심(11~14시) 45%(실측 26%), 의료·의류·교육 각 1% 안팎(실측 21·10·6%). **입력은 실측에 가깝고 출력이 어긋난다** — 고칠 곳은 일정 생성과 결제 배분.

## 2. 원인(코드)

- `consumption.apply_consumption_model` 의 `EXP_PRICE_FIRST` 경로: 하루 가게 예산 `_pf_budget = 입력 소비액(평일/주말) ÷ 2.40 × 동별 가게 몫 × 계획 배수` 는 맞게 계산되지만,
  `_price_first_spends` 가 `ESSENTIAL_L1 = {식사, 마트, 편의점}` 을 **예산과 무관하게 전액 결제**한다. 예산은 카페·여가 같은 재량 지출만 막는다.
- 그래서 하루 지출 = LLM 이 1단계에서 잡은 결제 일정의 합. P010 정책 없음: 평일 기준액 12.3만 → 실제 2.0만(1.6건 × 1.28만), 주말 기준액 8.0만 → 실제 4.1만(2.2건 × 1.88만).
- 1단계 프롬프트에 이미 '평소 업종별 지출 구성(카드 실측)' 이 들어간다(`dawn_context._cat_line`). 알려 주는 방식은 효과가 없었다 → 구조로 고친다.
- 업종별 이월이 없어 병원·미용실·학원처럼 한 달에 한두 번 생기는 지출은 하루 단위 계획에 거의 나오지 않는다.

## 3. 무엇을 — `scripts/sim/category_budget.py` (작성됨)

1. `shares_for(persona, is_weekend)`: 페르소나 `cat_ratio_wd/we`(카드 실측)를 `bdc_category_map.fold` 로 시뮬 L1 12종에 접고, 입력 밖 업종 최소 몫 `MIN_SHARE=0.02` 후 재정규화.
2. `plan_day(...)` — **1단계 직후, 2단계 전에** 호출: 업종 봉투 = 하루 가게 예산 × 몫, 여유 = 봉투 + 업종별 이월.
   - 일상 외식(계기가 policy·appointment·rumor 가 아닌 `식사`)을 여유로 감당할 수 있는 끼니 수만 남기고 나머지는 집(근무 중이면 직장)에서 먹는 것으로 바꾼다.
   - 여유가 1회 구매액(`TICKETS`)을 넘었는데 오늘 일정에 없는 업종을 하루 최대 `MAX_INJECT=2` 건 넣는다(업종별 시간 창 `WINDOWS`, 근무자는 평일 근무 시간을 피해 18:30~21:00).
3. `allocate(...)` — `_price_first_spends` 대신: 월 납부(달마다 한 번)·비정기 의료·가전·정책 계기 일정은 전액 내되 봉투에서 빼고, 나머지는 업종 봉투 안에서 시간 순으로 결제(넘치면 기존과 같은 해시 확률 규칙). 업종별 이월을 `carry_out` 으로 낸다.
4. 이월 첫 값: 상태에 없으면 (사람, 업종) 해시 위상 × 1회 구매액 — 정책 전 주를 함께 돌리면 두 갈래가 같은 게이지에서 출발.

보정 원칙: 입력(페르소나 카드 업종 비율·평일/주말 소비액)만 쓴다. 검증 자료(추정매출·생활인구)는 쓰지 않는다. `TICKETS`·`WINDOWS` 는 가정(문서화), `EXP_CATEGORY_TICKETS`(JSON)로 바꿀 수 있다.

## 4. 남은 일 (클라우드 세션이 이어서)

1. **consumption.py**
   - `base_store_budget(daily, aid=None, dong_code=None)` 추가 — 계획 배수 1 일 때의 `_pf_budget` 과 같은 식(`spend_today(ANCHOR_PROPENSITY, daily)['total'] / ANCHOR_OVERSTATE × eligible_share_for_dong(...)`), `EXP_ELIGIBLE_CHANNEL` 이 꺼져 있으면 그 경로의 식을 그대로 따른다. 1단계 직후(2단계 전) 예산 추정에 쓴다.
   - `apply_consumption_model` 에 키워드 `category_shares: dict | None = None`, `budget_carry_cat: dict | None = None` 추가.
     `EXP_PRICE_FIRST` 경로에서 `category_budget.ENABLED and category_shares` 이면 `_price_first_spends` 두 호출(`_base_spends`, `desired_spends`)을 `category_budget.allocate(... monthly_subs=MONTHLY_SUBS, irregular_subs=IRREGULAR_SUBS ...)` 로 바꾸고,
     `_price_first_meta` 는 allocate 의 메타를 쓴다(carry_out 은 업종 합; 업종별 이월은 `meta['category']['carry_out']`). 반환 dict 에 `"category_budget": meta['category']` 추가.
2. **run_simulation.py `process_one`**
   - `call_stage1` 직후: 켜져 있으면 `shares, hints = category_budget.shares_for(ctx.persona, _kr_day_off(today))`(None 이면 건너뜀),
     `budget = consumption.base_store_budget(daily_we|daily_wd, aid)`, `carry = category_budget.carry_from_state((ctx.state or {}).get('budget_carry_cat'), aid)`,
     `s1.events, plan_meta = category_budget.plan_day(s1.events, ..., make_event=lambda **kw: Stage1Event(**kw))`.
     `has_work = bool(ctx.persona.get('work_poi_id'))`.
   - `apply_consumption_model(...)` 에 `category_shares=shares`, `budget_carry_cat=carry` 전달(켜져 있을 때만).
   - `night_create_state(... budget_carry_cat=(cm_meta.get('category_budget') or {}).get('carry_out'))`.
   - 지표(metrics) 줄에 `cm_category_budget`(envelopes·carry_in/out·paid_by_cat)·`cm_category_plan`(meals_converted·injected) 추가.
3. **plan_writer.py** `NIGHT_STATE_CYPHER` 에 `s.budget_carry_cat = $budget_carry_cat`(JSON 문자열, 꺼져 있으면 null), `night_create_state(budget_carry_cat: dict | None = None)` 인자 추가.
4. **dawn_context.py** 상태 조회에 `s.budget_carry_cat AS budget_carry_cat` 추가.
5. **단위 시험** `tests/unit/sim/test_category_budget_20261019.py`
   - 꺼져 있을 때 `apply_consumption_model` 결과가 바뀌지 않음(같은 입력, 플래그 off).
   - `shares_for` 합 1, 입력 밖 업종 ≥ MIN_SHARE, 주말/평일 입력을 따로 씀.
   - `plan_day`: 보호 계기(policy/appointment/rumor) 외식은 절대 바꾸지 않음, 감당 못 하는 일상 외식만 집/직장으로, 결정론(같은 입력 → 같은 출력), 시간 정렬 유지, 넣는 일정 ≤ MAX_INJECT, 시간 충돌 30분 회피, 근무자 평일 근무 시간 회피.
   - `allocate`: 업종 합이 봉투+이월을 넘지 않음(필수 지출 제외), 월 납부는 달에 한 번, 정책 계기는 전액 결제, 이월 상·하한.
   - 기존 `tests/unit/sim` 전체를 돌려 실패 목록이 출발점과 같음을 확인(엔진 기록상 기존 실패 19개).
6. 커밋 메시지 끝: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. main 에 합치지 않는다.

## 5. 실행과 판정 (환경은 사용자가 준비 — GPU 서버·Neo4j)

- 같은 명부(예: P010 150명 `p010_ab3w_small_150.json`)로 **정책 없음만** 정책 전 7일 + 7일, 옛 엔진(플래그 off)과 새 엔진(`EXP_CATEGORY_BUDGET=1`)을 같은 LLM·같은 시작 그래프에서 각각 돌린다.
  나머지 플래그는 P010 본런 `engine.json` 과 같게(`EXP_PRICE_FIRST=1`, `EXP_PLAN_DRIVES_TOTAL=1`, `EXP_ELIGIBLE_CHANNEL=1`, `EXP_CLEAN_PROMPT=1`, `EXP_NO_MIN_OUTING=1` 등).
- 판정(결과 보기 전에 고정): `tools/l1_eval/l1_multiscale.py` 로 두 런의 사다리를 같은 사람·같은 날로 다시 그린다.
  - 주 지표: 무엇에(3분류·10묶음)와 언제(주중/주말·요일·시간대)가 ○ → ◔ 이상으로 오르는가. 음식점 비중이 입력(28%) 쪽으로, 주말 비중이 입력(20%) 쪽으로 오는가.
  - 지키는 지표: 누가(성·연령) 판정이 떨어지지 않는가, 사람 위치가 떨어지지 않는가, 하루 지출 수준(평일 평균)이 ±20% 안인가.
  - 정책 반응을 막지 않는지는 다음 단계(정책 있음 갈래)에서 받기 전 위약·정책 계기 일정 수로 본다.
- 150명이면 세밀한 단위는 판정할 수 없다(표본). 굵은 단위만 본다.

## 6. 알고 넣은 한계

- 1회 구매액·시간 창은 가정이다. 결과를 보고 고치지 않는다(바꾸면 새 판으로 기록).
- 공간(상가 수 비례를 못 이김, 도심 낮 인구 없음)은 이번 변경 범위 밖 — 직장 재배정·가게 후보 범위는 다음 단계.
- 넣은 일정은 집 기준(anchor=residence)이라 공간 판정은 나빠질 수 있다. 지키는 지표로 본다.

## 7. 구현 기록 (2026-10-19, 클라우드 세션 — 1~6절은 그대로 두었다)

브랜치: 예전 `fix/l1-baseline-fidelity`(85798a3) 를 지우고, `final_backup_20261019` 를 main 에 병합한 뒤
(wikriwiki/kw_26_final_project#3, main 트리 = final_backup 트리) main 에서 다시 만들었다. 엔진은 정책별 폴더 중
`P010_민생회복_소비쿠폰/`(= `P010_simulation` a99e561 트리, 바이트 단위로 같음)이고, 85798a3 의 파일(이 문서·
`scripts/sim/category_budget.py`·`tools/l1_eval/`)을 이 폴더 아래로 옮겼다. 이 문서의 경로는 모두 이 폴더 기준이다.

### 4절 남은 일 — 한 것

| 항목 | 상태 |
|---|---|
| 4-1 `consumption.base_store_budget` · `apply_consumption_model(category_shares, budget_carry_cat)` · 반환 `category_budget` | 했다 |
| 4-2 `run_simulation.process_one` — 1단계 직후 `plan_day`, 소비 모형에 몫·이월 전달, 밤 상태 저장, 지표 `cm_category_budget`·`cm_category_plan` | 했다 |
| 4-3 `plan_writer` — `s.budget_carry_cat`(JSON 문자열, 꺼져 있으면 null) · `night_create_state(budget_carry_cat=)` | 했다 |
| 4-4 `dawn_context` — 상태 조회에 `budget_carry_cat` | 했다 |
| 4-5 단위 시험 `tests/unit/sim/test_category_budget_20261019.py` (15개) | 했다, 모두 통과 |

### 구현하면서 정한 것 · 고친 것

1. **최소 몫(고침)**: `shares_for` 가 바닥(0.02)을 깐 뒤 전체를 다시 나눠, 입력 밖 업종 몫이 0.018 안팎으로 MIN_SHARE 아래로
   내려갔다(4-5 시험 "입력 밖 업종 ≥ MIN_SHARE" 에 걸림). 바닥에 걸린 업종을 MIN_SHARE 로 고정하고 나머지를 남은 몫 안에서 비례로 나눈다.
2. **거래 없는 날의 게이지(고침)**: 가게 결제가 하나도 없는 날은 소비 모형이 일찍 돌아와 업종 이월이 저장되지 않았고(null),
   다음 날 게이지가 해시 위상으로 다시 시작했다 — 집에 머문 날 쌓여야 할 필요가 지워진다. 그런 날도 `allocate([], [])` 로
   그날 봉투를 업종마다 그대로 이월한다(예산은 `base_store_budget`, 계획 배수 1).
3. **이월은 업종별로만**: 켜져 있으면 `allocate` 의 예산은 `_pf_budget`(합계 이월 `budget_carry_in` 을 더하지 않음) — 상태의
   `budget_carry` 가 업종 이월의 합이므로 더하면 같은 돈을 두 번 센다. `price_first.carry_in/out` 은 업종 합이다.
4. **이미 낸 월 납부는 넣지 않는다(사용자 결정 2026-10-19)**: 교육·건강 게이지가 차서 학원·기타교육·헬스장을 넣었는데 이번 달 이미
   냈으면 0원이라 게이지가 줄지 않고 다음 날 또 들어갔다(오프라인 14일 점검: 150명 중 9명이 7일 이상). `plan_day` 가
   `monthly_subs`·`monthly_paid` 를 받아 그 업종은 그날 넣지 않는다(다음으로 밀린 업종이 그 자리에 들어간다).
5. `base_store_budget` 에 `is_weekend` 인자를 더했다 — 적립 채널이 꺼진 경로에서 계획 기준선을 평일/주말로 찾는 식을 따르기 위해서다.
   본 실험(적립 채널 켜짐)에서는 쓰이지 않는다.
6. 4-5 의 "업종 합이 봉투+이월을 넘지 않음" 은 3절의 "넘치면 기존과 같은 해시 확률 규칙" 과 함께 읽어, **넘는 결제는 업종마다
   마지막 한 건뿐이고 그 앞까지의 합은 봉투+이월(필수 지출을 뺀 것) 이하**로 시험한다(무작위 300판).

### 실제로 돌린 검사 (모델·Neo4j 없이)

- 꺼져 있을 때: 고정 입력(평일 섞인 하루·외출 없는 하루)에서 `apply_consumption_model` 결과와 바뀐 일정이 출발점 엔진(a99e561)과
  **바이트 단위로 같다**. 고정값 `tests/fixtures/category_budget_20261019/off_snapshot_a99e561.json`. 업종 인자를 넘겨도 꺼져 있으면 같다.
- `tests/unit/sim` 전체: 실패 5개 — 출발점과 같은 목록(test_full_experiment_recovery 2, test_policy_prompt_timing 2,
  test_price_first_20261007::test_ticket_label_and_unmapped). 이 환경 기준이며 엔진 기록의 "기존 실패 19개" 와 수가 다른 것은 환경 차이다.
  `tests/unit`(sim 제외)도 출발점과 실패·수집 오류 목록이 같다(fastapi 미설치 등).
- 고친 1·2 는 고치기 전 코드로 되돌리면 각각 해당 시험이 실패하는 것을 확인했다.
- 오프라인 14일 점검(150명 명부, `agents_final.json.bak_orig` 의 업종 비율, 하루 점심 한 끼 일정만 둔 가상 일정): 넣은 일정
  사람-날당 1.02건, 0원 결제로 넣은 일정 0건. **LLM 일정이 아니므로 결과 예측이 아니다** — 배관 점검일 뿐이다.
- 하지 않은 것: 실제 모델 실행, Neo4j 상태 저장·조회 왕복, 실측 대비 판정.

## 8. 실행 순서 (5절 — 기존 실행기 그대로, 사용자 결정 2026-10-19)

`tools/run_ab3w.sh` 를 고치지 않고 `AB_CASE=p010` 으로 두 번 돌린다. 실행기는 정책 있음 갈래도 함께 돌린다(사람-날 3,150, 정책 없음만이면
2,100). **판정은 정책 없음 갈래(`off`)와 정책 전 주(`pre`)만 쓴다.** 정책 있음 갈래는 5절 "다음 단계"(받기 전 위약·정책 계기 일정 수)에 쓴다.

0. **입력 확인** — 출발 그래프(`AB_PRE_GRAPH`, 기본 `/data/backup_20260927_pre_pilot`)를 복원한 Neo4j 에서 명부 150명 모두
   `cat_ratio_wd`·`cat_ratio_we` 가 있어야 한다(없는 사람은 이 수정이 적용되지 않는다 — `shares_for` 가 None).
   `MATCH (a:Agent) WHERE a.id IN $ids RETURN count(a), count(a.cat_ratio_wd), count(a.cat_ratio_we)` — 비어 있으면
   `scripts/neo4j_load/09_backfill_catratio.py` 로 채운 그래프를 출발 그래프로 쓴다(두 런이 같은 그래프여야 한다).
1. **옛 엔진(플래그 꺼짐)** — 이 브랜치 그대로, `EXP_CATEGORY_*` 를 모두 지운 셸에서:
   ```
   cd <저장소>/P010_민생회복_소비쿠폰
   unset EXP_CATEGORY_BUDGET EXP_CATEGORY_TICKETS EXP_CATEGORY_MIN_SHARE EXP_CATEGORY_MAX_INJECT
   AB_CASE=p010 AB_TAG=l1old AB_ROSTER=$PWD/data/experiments/p010_eligibility_20261012/p010_ab3w_small_150.json \
   AB_PRE_DAYS=7 AB_POST_DAYS=7 AB_NEO_ON=… AB_BOLT_ON=… AB_HTTP_ON=… AB_NEO_OFF=… AB_BOLT_OFF=… AB_HTTP_OFF=… \
   bash tools/run_ab3w.sh
   ```
2. **새 엔진** — 같은 셸 설정에 `export EXP_CATEGORY_BUDGET=1` 만 더하고 `AB_TAG=l1new`. 동시에 돌리면 Neo4j 쌍을 다르게 준다.
   - `EXP_CATEGORY_BUDGET` 은 `common_env` 가 지우지 않으므로 그대로 모든 단계에 닿고, `engine.json` 의 설정과 실행 지문에 들어간다
     (`EXP_*` 전부가 지문). 이어 돌릴 때 플래그가 다르면 실행기가 멈춘다.
   - 같은 LLM(`AB_LLM_BASE_URL`·`AB_MODEL_ID`)·같은 `AB_PRE_GRAPH`·같은 명부. 그 밖의 플래그는 실행기가 P010 본런과 같게 넣는다
     (`EXP_PRICE_FIRST=1`·`EXP_PLAN_DRIVES_TOTAL=1`·`EXP_ELIGIBLE_CHANNEL=1`·`EXP_CLEAN_PROMPT=1`·`EXP_NO_MIN_OUTING=1`·p010 `CASE_EXPORTS`).
3. **새 엔진 첫날 확인** — `$BASE/pre/metrics/day_2025-07-14.jsonl` 의 행에 `cm_category_budget`·`cm_category_plan` 이 있어야 한다
   (없으면 플래그가 닿지 않은 것이다). 옛 엔진 행에는 이 열이 없다.
4. **판정(5절, 결과 보기 전 고정)** — 두 런의 `$BASE` 를 zip 으로 묶어 `tools/l1_eval/extract_run_v2.py <zip> <pickle>` →
   `tools/l1_eval/l1_multiscale.py` 의 `RUNS`·실측 경로를 두 pickle 로 바꿔 같은 사람·같은 날(정책 전 7일 + `off` 7일)로 사다리를 다시 그린다.
   150명이라 굵은 단위만 본다.
