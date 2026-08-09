# 데이터 계약 — 시뮬 산출물 → 콘솔 API

> S1 산출물. `docs/GAUNTLET_WEB_CONSOLE.md` §3의 선행 조각.
> S2(백엔드)·S3(정책화면)·S4(모니터)·S5(시각화통합)는 **이 문서와 `web/fixtures/`만 보고** 독립 개발한다.

## 0. 이 문서의 표기 규칙

| 표기 | 뜻 |
|---|---|
| **실측** | `C:\Users\srdyh\gpu_exp_data\20260802\` 의 실제 파일을 파싱해 확인한 필드·값 |
| **코드 확인** | 저장소의 writer 코드(`scripts/sim/*.py`)에서 확인했으나 실측 3종 run에 값이 나타나지 않은 것 |
| **미확인** | 실측·코드 어느 쪽에서도 확정하지 못한 것. **화면에 추정치를 그리지 말고 "알 수 없음"으로 표시한다** |
| **설계** | 파일에 존재하지 않고 콘솔이 새로 정의하는 것 (lock, SSE 등). 값이 아니라 구조만 정의 |

기준 B1(목업 데이터 금지)에 따라, **"실측"·"코드 확인"이 아닌 필드는 API 응답에 넣지 않는다.**
값을 모르면 필드를 지우지 말고 `null` + `unknown[]` 배열에 키 이름을 넣는다 (§4).

---

## 1. 실측 소스 — 3종 run

```
C:\Users\srdyh\gpu_exp_data\20260802\
  out_BASE\                   완료 run   7일  ×  200 agent   (정책 P010 적용, 무정책 대조군 아님 — §7.9)
  out_FINAL\                  완료 run  28일  ×  200 agent
  rescue\out_BASE7500\        중단 run   1일  × 7,500 목표 / 4,533 완료  ← 불완전 표현 테스트 케이스
  logs_scripts\               파일 42개 = run_*.log 24개 + chain_*.log 3개 + chain_*.sh + .py 유틸
  archive\                    events/dump 사본 (콘솔이 읽지 않음)
```

**run 루트 = `metrics/` 를 직접 자식으로 갖는 디렉터리.**
`rescue/` 는 run 루트가 아니라 run 루트의 부모다. 스캐너는 한 단계 내려가서 찾아야 한다.

### 1.1 run 디렉터리 파일 인벤토리 (실측)

| 파일 | writer | out_BASE | out_FINAL | rescue/out_BASE7500 |
|---|---|:-:|:-:|:-:|
| `metrics/day_<YYYY-MM-DD>.jsonl` | `run_simulation.run_day` (agent 완료마다 append) | 7 | 28 | 1 |
| `checkpoints/done_<day>.json` | 동, 500명마다 + 일자 종료 시 | 7 | 28 | 1 |
| `checkpoints/failed_<day>.json` | 동, **일자 종료 시에만** | 7 | 28 | **0** |
| `timing/day_<day>.json` | `timing_metrics.write_day_timing_report`, **일자 종료 시에만** | 7 | 28 | **0** |
| `timing/slow_<day>.json` | `timing_metrics.slow_cases`, 일자 종료 시에만 | 7 | 28 | **0** |
| `summary.json` | `run_simulation.main`, 매 일자 종료마다 원자적 갱신 | ✅ | ✅ | **없음** |
| `stage1_failures.jsonl` | `stage1_intent.call_stage1` (재시도 실패마다 append) | ✅ | ✅ | ✅ |
| `events.jsonl` | **`export_run.py` (별도 운영 스크립트, 저장소 밖)** | ✅ | ✅ | **없음** |
| `poi_summary.json` | 동 | ✅ | ✅ | **없음** |

> **핵심**: 중단 run에는 `summary.json` / `timing/` / `failed_*.json` / `events.jsonl` 이 **구조적으로** 없다.
> 파일이 없는 것은 오류가 아니라 "그 단계까지 못 갔다"는 정보다. §4에서 이를 어떻게 표현할지 정한다.

---

## 2. 파일별 스키마 (실측)

### 2.1 `metrics/day_<day>.jsonl` — agent 1명 = 1줄

**최대 크기 실측: `rescue/.../day_2025-07-14.jsonl` = 19,599,953 bytes (18.7MB), 4,533행.**
행 하나가 평균 4.3KB. **브라우저로 절대 보내지 않는다** (기준 B5, §5 참조).

성공 행(`status:"ok"`)의 최상위 필드는 **`out_BASE` 83개 / `out_FINAL` 83개 / `rescue` 82개**
(전 일자 전 행 스캔, 합집합 83개). run 사이의 차이는 단 하나: `timing_t_night_finalize`.
`rescue` 는 첫날 하나뿐이라 그 키가 등장할 일자 자체가 없어 82개다 (바로 아래 표 참고).
**"82개"를 상수로 박지 마라.** run 이 며칠짜리인지에 따라 82/83 이 갈린다.

#### 식별·상태
| 필드 | 타입 | 비고 |
|---|---|---|
| `aid` | str | `AGT_<행정동10자리>_<F\|M>_<연령대>_<일련번호>` — 앞 8자리가 행정동 코드 |
| `status` | `"ok"` \| `"error"` | |
| `elapsed` | float(초) | agent 1명 총 소요 |

#### 단계 소요 (`timing_*` 접두 — phase 레벨)
`timing_t_dawn`, `timing_t_s1`, `timing_t_s2`, `timing_t_write_plan` : float(초), 4종 모두 항상 존재.

`timing_t_night_finalize` : float(초). **Day 0에는 없고 Day 1 이후에만 존재** (전날 정산이므로).
실측 — **3종 run 전부 "첫날에만" 없다.** 규칙에 예외가 없다:

| run | 일자 수 | 키 있는 일자 | 없는 일자 |
|---|--:|--:|---|
| `out_BASE` | 7 | 6 | `2025-07-21` (첫날) |
| `out_FINAL` | 28 | 27 | `2025-07-21` (첫날) |
| `rescue/out_BASE7500` | 1 | 0 | `2025-07-14` (첫날 = 유일한 일자) |

→ **키 부재를 정상으로 처리해야 한다.** 이 때문에 `timing/day_*.json` 의 `timings` path 개수도
첫날만 58, 나머지는 59다(§3.5).

#### 결과 요약
`n_events`(int) `n_includes`(int) `n_visited_memories`(int) `avg_sat`(float 0~1)
`balance`(int, 원) `mood`(float) `fatigue`(float) `tokens_in`(int) `tokens_out`(int) `policy_hits`(int)

#### 정책 지갑
`grant_applied_today` `grant_expired_today` `policy_spend_today` `grant_remaining_total`
`policy_spend_corrected` — 모두 int(원).

#### 소비 모델 (`cm_` 접두, **33개** — 3종 run 전부 동일)
`cm_propensity` `cm_propensity_center` `cm_day_multiplier` `cm_planned_total` `cm_today_total`
`cm_today_total_incl_online` `cm_online_total` `cm_online_share` `cm_online_share_source`(str, 실측값 `"seoul_smallbiz"`)
`cm_personal_total` `cm_anchor_total` `cm_eligible_base` `cm_additional_from_grant` `cm_plan_over_anchor`
`cm_intended_grant_today` `cm_grant_carry_in` `cm_grant_carry_out` `cm_grant_plan_days`
`cm_grant_choice_mode`(bool) `cm_grant_choice_share_mean`(float) `cm_grant_extra_rate`(float) `cm_substituted`(int)
`cm_selected_policy_liquidity` `cm_policy_requested_total` `cm_policy_allocated_total`
`cm_policy_eligible_spend_total` `cm_policy_eligible_event_count` `cm_policy_payment_coverage`(float)
`cm_policy_liquidity_relief` `cm_mechanical_policy_uplift`
(위 30개 + 아래 "항상 null" 절의 `cm_grant_posture` `cm_mpc_new_share_effective` `cm_mpc_new_share` = **33개**)

`spend_decile` (int 1~10) — 정책 분위 화면의 조인 키.
> **`spend_decile` 은 `null` 일 수 있다.** `rescue` Day 0 실측 ok 4,533행 중 **5행이 null**.
> `1~10` 을 전제로 조인/그룹핑하면 그 행들이 조용히 사라진다. 집계 규약은 §3.4 참고.

#### 재시도·중첩 타이머
`s1_attempts`(int) `s2_attempts`(int)
`s1_timing` / `s2_timing` / `prompt_timing` / `dawn_timing` : **중첩 dict**.
- 공통 규칙: `t_` 로 시작하는 숫자 leaf 가 초 단위 타이머다 (`timing_metrics._collect_timing_leaves`).
- `dawn_timing.persona_cache_hit` / `policy_cache_hit` : bool → 캐시 적중률의 원천.
- `s1_timing.attempts[]` / `s2_timing.attempts[]` : 시도별 `{attempt, t_llm, tokens_in, tokens_out, status, t_total, ...}`.
  `s2_timing.attempts[].call_kind` 는 실측 `"initial"` 관측.
- `s2_timing.candidate_detail` : `{t_group_resolve, t_query_exact, t_query_l1_dong, t_query_l1_district, t_enrich, t_sort_split, n_groups, n_query_exact, n_query_l1_dong, n_query_l1_district}`.

#### Stage 2 폴백 카운터 (`fb_` 접두 + `review_lookup_count`) — int 12개
`fb_resolve_dong` `fb_cand_sub_match` `fb_cand_l1_dong` `fb_cand_l1_district` `fb_cand_all_empty`
`fb_hallucinations_corrected` `fb_hallucinations_dropped` `fb_order_mismatch` `fb_missing_picks_filled`
`fb_pool_split_groups` `fb_pool_split_events`

#### 3종 run 전부에서 항상 `null` 인 필드 — **화면에 넣지 마라**
`s1_grant_use` `s1_grant_style` `s1_grant_spread_days` `s1_grant_plan_reason` `s1_grant_extra_spend`
`s1_online_share` `cm_grant_posture` `cm_mpc_new_share_effective`
→ 코드상 존재하지만 이 3개 run 어디에도 값이 없다. **미확인**으로 취급한다.
`cm_mpc_new_share` 는 대부분 null, 일부 float (BASE Day0 실측 200명 중 55명).
`s1_daily_propensity` `s1_grant_kept_share` 는 항상 float (실측).

#### 실패 행 (`status:"error"`) — 필드가 **5개뿐**
```json
{"aid": "...", "status": "error", "elapsed": 219.89, "error": "<앞 200자>", "trace": "<뒤 500자>"}
```
실측: `out_FINAL/metrics/day_2025-08-03.jsonl` 1건, `day_2025-08-14.jsonl` 1건.
→ **UI는 error 행에서 `avg_sat`·`balance` 등을 읽으려 하면 안 된다.** 5개 필드만 있다.
`error` 는 `str(e)[:200]`, `trace` 는 `traceback.format_exc(limit=3)[-500:]` 로 잘려 있다(코드 확인).
실측된 `trace` 는 앞이 잘려 `"ess_one\n ..."` 처럼 문장 중간에서 시작한다. **파싱 대상이 아니라 원문 표시용이다.**

#### 행 수 ≠ 완료 agent 수 (중요)
- `metrics` 행 수 = ok 행 + error 행. `agents_ok` 만 따로 세야 한다.
- resume 시 `run_day()` 가 jsonl 을 **재작성**하며 aid당 ok 1행만 남기고 error 행은 폐기한다(코드 확인).
  → **중단→재개된 run의 metrics 행 수는 시점에 따라 달라진다. 캐시 키에 mtime을 포함해야 한다.**
- 실측 divergence: `rescue` 는 metrics 4,533행 / `done_` 체크포인트 4,500개.
  체크포인트는 500명 단위 스냅샷이라 항상 **뒤처진다**. 둘 다 보여주고, 진행률의 분자로는 `agents_ok`(metrics)를 쓴다.

### 2.2 `checkpoints/done_<day>.json`
`["AGT_...", ...]` — 정렬된 aid 문자열 배열. 그 외 필드 없음.
실측 크기: 200명 5,732B / 4,500명 129,528B. **7,500명이면 약 216KB** → 목록 자체를 프런트로 보내지 않는다.

### 2.3 `checkpoints/failed_<day>.json`
`[]` 또는 §2.1의 error 행 객체 배열. 실측 최대 1건. **일자 종료 시에만 기록**되므로 중단 run에는 없다.

### 2.4 `timing/day_<day>.json` — 일자 병목 리포트
```
agents_ok        int
agents_error     int
timings          { "<path>": {n,total,avg,p50,p95,p99,max} }   실측 58개 path
counters         { "dawn.n_memory_returned"|"stage1.n_llm_calls"|"stage2.n_llm_calls": {동일 통계} }
cache            { persona_hit_rate, policy_hit_rate }          0~1 float
policy_payment   { llm_requested_total, system_allocated_total, eligible_spend_total,
                   eligible_event_count, payment_coverage, agents_using_policy,
                   agent_usage_rate, liquidity_relief_total, expired_wallet_total }
bottleneck_rank  [ {path, total_sec, avg_sec, p95_sec} ]  상위 30, total_sec 내림차순
```
`path` 네임스페이스(실측): `phase.*` `dawn.*` `prompt.*` `stage1.*` `stage2.*` `stage2.candidate_detail.*`.
`bottleneck_rank` 는 `*.t_total` 을 **제외**한다(중복 합산 방지, 코드 확인). `timings` 에는 포함된다.
**모니터 화면의 "Stage1/Stage2/Dawn 병목"은 이 파일이 정본이다.** 직접 계산하지 마라.

### 2.5 `timing/slow_<day>.json` — 느린 agent 원본
`[{aid, slow:{dawn?|stage1?|stage2?}, dawn_timing, prompt_timing, s1_timing, s2_timing, tokens_in, tokens_out, s1_attempts, s2_attempts}]`
임계값은 환경변수 `SLOW_DAWN_SEC`(기본 2) / `SLOW_STAGE1_SEC`(60) / `SLOW_STAGE2_SEC`(60).
**임계값은 파일에 기록되지 않는다 → 실행 시 임계값은 미확인.** 화면에 "2초/60초"를 하드코딩하지 마라.
실측 크기 531,751~729,811 B/일 (BASE+FINAL 35개 일자). **페이지네이션 필수** (§5).

### 2.6 `summary.json`
```json
{"summary": [ {day, ok, err, agent_elapsed_sec, night2_elapsed_sec, elapsed_sec,
               timing_top:[{path,total_sec,avg_sec,p95_sec}] (최대 10)} ],
 "args": {start, days, limit, gu, workers},
 "completed_at": "ISO8601"}
```
- `args.limit` = agent 목표 수 (실측 BASE/FINAL 모두 200). `null` 이면 전수 → **목표 수 미확인**.
- `args.gu` 실측 `null`.
- **`updated_at` 과 `completed_at` 은 배타적이다** (코드 확인): 진행 중에는 `updated_at`, 종료 시 `completed_at`.
  실측 3종은 전부 `completed_at`. 진행 중 파일은 미실측이므로 **양쪽 키를 모두 optional 로 다뤄라.**
- `summary[]` 는 **완료된 일자만** 들어간다. 진행 중인 일자는 없다.

### 2.7 `stage1_failures.jsonl` — LLM 출력 파싱 실패 (agent 실패와 다름)
`{aid, day, attempt(int), temp(float), error_type(str), error(str≤300), finish_reason(str), raw_excerpt(str≤800)}`
실측 `error_type` **3종** — `"ValidationError"` `"JSONDecodeError"` **`"ValueError"`**.
실측 `finish_reason` **2종** — `"stop"` **`"length"`**.
> `ValueError` 와 `length` 는 BASE 에는 없다. **BASE 만 보고 값 목록을 확정하면 놓친다.**
> 화면이 `error_type` 을 하드코딩한 2개 배지로 렌더하면 나머지가 조용히 사라진다 —
> **미지의 값은 원문 그대로 표시**하고, 색 매핑에만 알려진 값을 쓴다.

**run 전체가 한 파일에 누적**된다(일자별 파일 아님). `day` 로 필터한다.
실측 분포:

| run | 건수 | 크기 | `error_type` | `finish_reason` |
|---|--:|--:|---|---|
| BASE | 32 | 52,861 B | ValidationError 21 / JSONDecodeError 11 | stop 32 |
| FINAL | 203 | 336,311 B | ValidationError 141 / JSONDecodeError 61 / **ValueError 1** | stop 202 / **length 1** |
| BASE7500 | 101 | 164,357 B | ValidationError 68 / JSONDecodeError 26 / **ValueError 7** | stop 101 |
> **재시도로 복구된 실패도 여기 남는다.** 이 파일의 건수는 agent 실패 수가 아니다.
> `attempt` 가 0..2 로 늘어나며, 최종 실패한 agent만 `checkpoints/failed_*.json` 에 나타난다. 화면에서 이 둘을 섞지 마라.

### 2.8 `events.jsonl` — 결제 원장 (run 종료 후 별도 export)
`{day_type, l1, sub, amt(int,원), sp(str: JSON 문자열), ex(int|null), wba(bool|null), elig(bool), dong(null), day}`
- **`sp` 는 dict 가 아니라 JSON 문자열이다.** 실측 `"{}"` 또는 `"{\"P010\": 6251}"` → 서버에서 파싱한다.
- `dong` 은 **실측 3종 전부 100% null** (POI.adm_cd 미적재). 지도 조인에 쓸 수 없다 → **미확인**.
- `ex`(extra_spent) / `wba`(would_buy_anyway) 는 null 비율이 높다
  (BASE 7,785건 중 각 6,001 / 6,004 null — 77%. FINAL 30,189건 중 22,365 / 22,369 — 74%).
- **`l1` distinct 는 run 마다 다르다: BASE 15종 / FINAL 18종.**
  > 짧은 run 은 업종이 덜 등장할 뿐이다. **업종 축을 상수 18개로 고정하지 마라.**
  > `by_l1` / `by_day_l1` 의 축은 응답에서 관측된 값으로만 만든다. 없는 업종을 0으로 채우면
  > "그 업종에서 소비가 없었다"는 거짓 사실을 그린다 — 실제로는 "관측 자체가 없다"이다.
- 크기: BASE 1,236,982 B / 7,785행, FINAL 4,799,810 B / 30,189행. **서버 집계 후 전송.**

### 2.9 `poi_summary.json`
`{"poi_total": int, "poi_eligible": int}` — 필드 2개뿐. 실측 두 run 모두 `{543924, 491675}`.

### 2.10 `logs_scripts/run_<RUN>.log` — run 디렉터리 **밖**의 보조 소스
`run_simulation.py` stdout. 파싱 가능한 실측 라인:
```
  agents: 7500, days: 7, start: 2025-07-14, workers: 48
  output: /data/exp001/out_BASE7500
[Day 0 2025-07-14] processing 7500 agents with 48 workers
  4500/7500 (ok=4500, err=0) @ 0.5/s, ETA 6439s
[Day 0 2025-07-21] agent phase done in 631s — ok=200, err=0
  [캐시] persona=0.0% policy=6.5% | slow=164명 → /data/exp001/out_BASE/timing
  [Night2] Conversation +20 (약속=1, 이슈=0, 추천=13, 기타=6) in 33s
[Day 0 2025-07-21] done in 664s (agent=631s, Night2=33s)
```
> **로그는 보조(advisory)다. 정본이 아니다.**
> 실측 반증: `run_BASE7500.log` 헤더는 `workers: 48` 인데 `chain_p2.sh` 는 `--workers 128` 로 기동한다.
> 로그 파일이 재기동으로 덮여쓰였을 가능성이 있다. **`summary.json` 이 있으면 항상 그쪽이 이긴다.**
> 로그에서 온 값은 API에서 `log_hint` 라는 별도 객체에 담아 출처를 분리한다 (§3.2).
> 로그 파일과 run 디렉터리를 잇는 것은 `output:` 줄의 경로뿐이며, 그 경로는 **GPU 서버 기준 절대경로**다.
> 로컬 사본 경로와 다르므로 **basename 매칭**이 필요하다.

---

## 3. API 리소스 설계

베이스: `/api`. 전부 `Content-Type: application/json; charset=utf-8`.
**날짜는 언제나 `YYYY-MM-DD` 문자열**, 금액은 원 단위 정수, 시간은 초 단위 float.

각 리소스의 실물 응답 예시는 `web/fixtures/` 에 있다. 파일명이 곧 계약이다.

### 3.1 `GET /api/runs` → `runs.index.json`
```ts
{ total: number,
  items: Array<{
    run_id: string,            // 디렉터리명에서 "out_" 접두 제거. 실측: BASE | FINAL | BASE7500
    root: string,
    status: "completed" | "incomplete",
    first_day: string | null,  // metrics 파일명 기준
    last_day: string | null,
    days_present: number,      // metrics/day_*.jsonl 개수
    days_planned: number | null,   // summary.args.days. null = 미확인
    agents_target: number | null,  // summary.args.limit. null = 미확인
    completed_at: string | null,
    artifacts: { summary_json, events_jsonl, poi_summary_json,
                 stage1_failures_jsonl, timing_dir, checkpoints_dir, metrics_dir }, // 전부 boolean
    unknown: string[]          // §4
  }>,
  unknown: string[] }          // ← 리소스 최상위에도 있다 (§4.1.3)
```
`status` 판정 규칙(실측 3종으로 검증): `summary.completed_at` 이 있고 `summary.args.days === days_present` 이면 `completed`, 아니면 `incomplete`.

### 3.2 `GET /api/runs/{run_id}` → `run.<ID>.detail.json`
`runs.index` 항목(`run_id` `root` `status` `completed_at` `artifacts` 포함) + 다음. `updated_at` 도 함께 낸다(§2.6).
```ts
{ days_present: string[], days_with_timing: string[],
  days_with_done_checkpoint: string[], days_with_failed_checkpoint: string[],
  plan: { source: "summary.json:args" | null, start_day, planned_days, agents_target, workers },
  log_hint: null | { source_file, agents_target, planned_days, start_day, workers,
                     output_dir, last_progress_line },   // ← 출처가 다르다. plan 과 섞지 마라
  day_summaries: Array<{day, ok, err, agent_elapsed_sec, night2_elapsed_sec, elapsed_sec}>,
  unknown: string[] }
```
`day_summaries` 는 `summary.json` 의 `summary[]` 에서 `timing_top` 을 뺀 것(용량). 병목은 §3.5에서 따로 받는다.

### 3.3 `GET /api/runs/{run_id}/days` → `run.<ID>.days.json`
일자별 진행 시계열. **모니터 화면의 주 데이터 소스.**
```ts
{ run_id, total, items: Array<{
    day: string,
    agents_ok: number,               // metrics 의 status==="ok" 행 수
    agents_error: number,
    metrics_rows: number,            // ok + error. agents_ok 와 다를 수 있다 (§2.1)
    counts_source: "metrics_aggregate" | "status_scan",  // 위 3개 카운트를 어느 경로로 얻었나 (B4, 아래)
    checkpoint_done_count: number|null,
    checkpoint_failed_count: number|null,   // null = failed_*.json 없음 = 일자 미종료
    agents_target: number|null,
    progress_ratio: number|null,     // agents_ok / agents_target. target 미확인이면 null
    day_complete: boolean,           // summary.summary[] 에 해당 day 가 있는가
    elapsed_sec: number|null, agent_elapsed_sec: number|null, night2_elapsed_sec: number|null,
    timing_report_present: boolean,
    policy_payment: object|null,     // timing/day_*.json 의 policy_payment 그대로
    metrics_bytes: number,
    unknown: string[]
  }>,
  unknown: string[] }              // ← 리소스 최상위에도 있다 (§4.1.3)
```
> **`progress_ratio` 가 null 이면 진행 바를 그리지 마라.** 0%도 100%도 아니다 — 분모를 모른다.
> `run.BASE7500.days.json` 이 정확히 그 상태다.

#### 3.3.1 B4(첫 화면 2초)를 이 리소스가 어떻게 지키는가 — **계약 사항**

> **문제.** 이 리소스는 일자마다 `agents_ok`/`agents_error`/`metrics_rows` 를 필요로 한다.
> 그 값의 정본은 `metrics/day_*.jsonl` 뿐이다(체크포인트는 500명 단위로 뒤처진다, §2.1).
> 그런데 **진행 중인 일자는 캐시할 수 없다** — 파일이 계속 늘어나므로 매 요청 재집계다.
> 전체 집계(`aggregate_day`)로 이걸 하면 19.6MB 일자 하나에 아래 실측 시간이 들고, B4 2초 기준을 구현 단계에서 반드시 깬다.

**실측 (이 저장소 기준, `_build_fixtures.py` 의 두 함수를 같은 파일에 대해 3회 실행한 최소값):**

| 경로 | 19.6MB / 4,533행 | 36개 일자 전체 |
|---|--:|--:|
| `aggregate_day` (전체 집계) | **0.92초** | 2.6초 |
| `status_scan` (카운트 3종) | **0.044초** | **0.13초** |

파일 캐시가 차갑거나 원격 마운트면 `aggregate_day` 는 더 느려진다(콜드 실측 3.7초까지 관측).
**비율이 요점이다 — 같은 파일에서 21배.**

**계약: §3.3 은 카운트 3종을 얻는 데 전체 집계를 요구하지 않는다.**

1. 서버는 **경량 경로(`status_scan`)** 를 쓸 수 있다. JSON 파싱 없이 줄 단위 바이트 검사만 한다.
   `status` 는 항상 두 번째 키이고, error 행의 `error`/`trace` 안에 같은 패턴이 나와도
   따옴표가 이스케이프되므로(`\"status\": \"ok\"`) 오탐이 없다.
   판정 문자열은 `b'"status": "ok"'` / `b'"status": "error"'` 두 개다.
2. **동치성은 검증된 사실이다.** 3종 run **36개 일자 전부**에서 `status_scan` 과 `aggregate_day` 의
   `rows`/`agents_ok`/`agents_error` 가 **완전히 일치**했다(불일치 0건). 근사치가 아니다.
3. 응답은 어느 경로를 썼는지 **`counts_source` 로 밝힌다.** 픽스처는 전량 `"metrics_aggregate"` 다
   (생성기는 어차피 전체 집계를 하므로). S2 가 경량 경로를 쓰면 `"status_scan"` 이 된다.
   **소비자는 두 값 모두에서 같은 숫자를 기대해도 된다.**
4. §3.4(일자 상세 집계)는 이 완화의 대상이 **아니다.** 분포·10분위·폴백 카운터는 전체 집계가 있어야 한다.
   그래서 §3.4 는 **사용자가 특정 일자를 열 때 지연 로드**한다(§5.2-6). 첫 화면에 넣지 마라.

> **첫 화면 예산 배분 (B4).** 첫 렌더 = §3.1 + §3.3 뿐이다.
> 완료된 일자는 §5.2-3 대로 영구 캐시(파일이 더 이상 안 자란다)이고,
> 진행 중인 일자만 매 요청 `status_scan` 이 돈다 → 일자 1개당 실측 0.044초.
> §3.4/§3.5/§3.6 을 첫 화면에 끌어오면 이 예산이 무너진다.

### 3.4 `GET /api/runs/{run_id}/days/{day}` → `run.<ID>.day.<day>.json`
**19MB jsonl 을 서버에서 접은 결과.** 응답 실측 6.7~7.5KB.
```ts
{ run_id, day, source_file, source_bytes, aggregated_server_side: true,
  rows, status_counts: {ok?, error?, malformed?}, agents_ok, agents_error,
  sums:          { [필드명]: number },              // 실측 30키
  distributions: { [필드명]: {n,total,avg,p50,p95,max} },  // 실측 9키: elapsed, timing_t_*(4), avg_sat, balance, mood, fatigue
  fallback_counts: { [fb_* | review_lookup_count]: number },   // 실측 12키
  attempt_counts:  { "s1=1": n, "s1=2": n, "s2=1": n, ... },
  llm_call_totals: { stage1: n, stage2: n },
  cache: { persona_hit_rate: number|null, policy_hit_rate: number|null },
  by_spend_decile: Array<{spend_decile: 1..10 | null,          // ← null 버킷이 존재한다
                          agents, grant_applied_today,
                          grant_remaining_total, policy_spend_today,
                          cm_policy_allocated_total, cm_today_total_incl_online}>,
  spend_decile_unknown_agents: number,   // null 버킷의 크기. 0 이면 전량 10분위로 분해된 것
  error_samples: Array<{aid,status,elapsed,error,trace}>,   // 최대 10건
  _fields_not_aggregated: string[],
  unknown: string[] }
```

#### `by_spend_decile` — 항등식이 계약이다

> **`sum(by_spend_decile[].agents) === agents_ok` 는 언제나 성립해야 한다.**
> `spend_decile` 이 결측인 행을 조용히 버리면 이 항등식이 깨지고, 화면은
> "10분위 전량 분해"를 그린 채 일부 인원을 잃는다. §4.1.4("부분 계산은 부분이라고 말한다") 위반이다.

- 결측 행은 **`spend_decile: null` 버킷 하나로 모은다.** 버리지 않는다.
- `null` 버킷은 **배열의 항상 마지막**이다. 숫자 분위와 섞어 정렬하면 파이썬에서 `TypeError` 가 난다.
- 결측이 1행이라도 있으면 `unknown` 에 `"spend_decile"` 이 들어간다. 전량 결측일 때만이 아니다.
- `spend_decile_unknown_agents` 로 그 크기를 따로 실어 준다 → 화면이 응답만 보고 항등식을 검산할 수 있다.

**실측**: `rescue/out_BASE7500` Day 0 = ok 4,533행 중 **5행이 null**.
`run.BASE7500.day.2025-07-14.json` 이 그 실물이다 (`spend_decile_unknown_agents: 5`, `unknown: ["spend_decile"]`,
null 버킷 `agents:5` / `cm_today_total_incl_online:344,475` / 정책 지갑 4개 필드는 전부 0).
나머지 3개 day 픽스처는 `spend_decile_unknown_agents: 0`, `unknown: []`.

> UI 규칙: `spend_decile_unknown_agents > 0` 이면 10분위 차트에 **"분위 미상 N명" 열을 반드시 함께 그린다.**
> 10개 막대만 그리고 합계를 `agents_ok` 로 라벨하면 거짓말이 된다.

#### `_fields_not_aggregated` — 표본이 아니라 명세다

`_fields_not_aggregated` 는 **의도적 명세**다. "이건 안 보낸다"를 문서가 아니라 응답으로 선언한다.
필요해지면 S2가 `sums`/`distributions` 목록에 추가하고 이 배열에서 빠진다.

산출 규칙 (참조 구현 `_build_fixtures.py::aggregate_day`):

1. **전 행 스캔이다. 표본이 아니다.** 앞 N행만 보면 드물게만 나타나는 필드를 놓쳐
   이 목록이 '명세'가 아니라 '표본 관측'이 된다. 상수 메모리(집합 하나)면 되므로 전수로 돈다.
2. **다른 키로 접혀 응답에 이미 반영된 필드는 제외한다.** 이름이 `sums`/`distributions`/`fallback_counts`
   에 없다는 이유로 "안 보낸다"고 선언하면 소비자가 같은 값을 두 번 요청한다. 제외 대상 7종:

   | 원본 필드 | 응답의 어디로 갔나 |
   |---|---|
   | `status` | `status_counts` / `agents_ok` / `agents_error` |
   | `spend_decile` | `by_spend_decile` |
   | `s1_attempts` · `s2_attempts` | `attempt_counts` |
   | `dawn_timing` | `cache` (`persona_cache_hit` / `policy_cache_hit`) |
   | `s1_timing` | `llm_call_totals.stage1` |
   | `s2_timing` | `llm_call_totals.stage2` |

3. 따라서 **`_fields_not_aggregated` 와 `sums`∪`distributions`∪`fallback_counts` 의 교집합은 항상 공집합**이다.
   실측 4개 day 픽스처 전부에서 교집합 0, 목록 길이 24.

### 3.5 `GET /api/runs/{run_id}/days/{day}/bottlenecks` → `run.<ID>.day.<day>.bottlenecks.json`
**키 집합은 `available` 값과 무관하게 항상 같다** (§4.1.6). 해당 없는 값은 `null`.
```ts
{ run_id, day,
  available: boolean,
  degraded: boolean,                  // optional 아님. available:true 면 false
  reason: string|null,                // available:false 일 때만 문자열
  degraded_note: string|null,
  agents_ok: number, agents_error: number,        // 양쪽 모두 항상 채워진다
  bottleneck_rank: Array<{path,total_sec,avg_sec,p95_sec}>|null,   // 상위 30
  cache: object|null, policy_payment: object|null,
  counters: object|null, timings: object|null,
  fallback_rank: Array<{path,total_sec,avg_sec,p95_sec}>|null,     // phase.* 4개만
  fallback_source: string|null,
  unknown: string[] }
```

| | `available: true` | `available: false` |
|---|---|---|
| 채워짐 | `bottleneck_rank` `cache` `policy_payment` `counters` `timings` | `reason` `degraded_note` `fallback_rank` `fallback_source` |
| `null` | `reason` `degraded_note` `fallback_rank` `fallback_source` | `bottleneck_rank` `cache` `policy_payment` `counters` `timings` |
| `degraded` | `false` | `true` |
| `unknown` | `[]` | `["bottleneck_rank","cache","policy_payment","counters","timings"]` |

> `available:false` 여도 **빈 화면을 내지 마라.** `fallback_rank` 로 phase 4개(`phase.t_s1/t_s2/t_dawn/t_write_plan`)는
> metrics 의 `timing_t_*` 에서 재계산할 수 있다. `stage1.*`/`stage2.*`/`dawn.*` 하위 경로와 `cache`·`policy_payment` 만 미확인이다.
> `run.BASE7500.day.2025-07-14.bottlenecks.json` 이 이 상태의 실물이다.

`timings` 의 path 개수 실측 **58~59** (`phase.t_night_finalize` 가 Day 0 에만 없어 첫날이 58, 나머지가 59).
`bottleneck_rank` 는 실측 전부 30개.

### 3.6 `GET /api/runs/{run_id}/days/{day}/slow?limit=&offset=` → `run.<ID>.day.<day>.slow.json`
키 집합은 `available` 값과 무관하게 항상 같다 (§4.1.6).
```ts
{ run_id, day,
  available: boolean,
  reason: string|null,                 // optional 아님
  total: number|null, limit: number,
  sorted_by: "max(slow.*) desc",       // available 과 무관하게 항상 같은 문자열
  phase_counts: { dawn?: n, stage1?: n, stage2?: n }|null,
  items: Array<slow 원본 객체>,        // available:false 면 []
  unknown: string[] }
```
`total` 만 먼저 주고 `items` 는 페이지로 준다. 픽스처는 `limit=15` 기준 66.9KB — 원본 558KB(BASE Day 0)의 1/8.

- `available:true` → `reason:null`, `unknown:["slow_thresholds_sec"]`.
  **성공 응답에도 `unknown` 이 비어 있지 않다.** 임계값이 파일에 기록되지 않기 때문이다 (§2.5).
- `available:false` → `total:null`, `phase_counts:null`, `items:[]`,
  `unknown:["total","phase_counts","items","slow_thresholds_sec"]`.
  실물: `run.BASE7500.day.2025-07-14.slow.json`.

### 3.7 `GET /api/runs/{run_id}/days/{day}/failed` → `run.<ID>.day.<day>.failed.json`
`checkpoints/failed_<day>.json` 원본. **파일이 없어도 404 를 내지 않는다** (§4.1.1).
```ts
{ run_id, day, source_file: string,
  available: boolean, reason: string|null,
  total: number|null,                  // 0 과 null 은 다르다 (아래)
  items: Array<error 행>,              // 없으면 []
  unknown: string[] }
```
> **`total: 0` 과 `total: null` 을 같은 화면으로 그리지 마라.**
> `0` = "그 날은 끝났고 실패가 없었다"(성과). `null` = "그 날이 안 끝나서 파일 자체가 없다"(미확인).
> 실물 3종이 전부 있다 — `run.BASE.day.2025-07-21.failed.json`(`total:0`, `unknown:[]`),
> `run.FINAL.day.2025-08-03.failed.json`(`total:1`, 실제 실패 행),
> `run.BASE7500.day.2025-07-14.failed.json`(`total:null`, `unknown:["failed_checkpoint"]`).

### 3.8 `GET /api/runs/{run_id}/failures?day=&limit=` → `run.<ID>.failures.json`
`stage1_failures.jsonl` 의 **파싱 실패**(§2.7). agent 실패(§3.7)와 별개 리소스로 둔다.
```ts
{ run_id, available: boolean, reason: string|null,
  total: number|null, by_day: {[day]: n}|null, by_error_type: {[type]: n}|null,
  limit: number, items: [...], unknown: string[] }
```
집계(`by_day`,`by_error_type`)는 전량 스캔, `items` 는 페이지. 픽스처 20KB (원본 336KB의 1/16).
실측 3종 전부 `available:true` — **중단 run(`BASE7500`)에도 이 파일은 있다**(101건). "중단이면 못 본다"가 아니다.

### 3.9 `GET /api/runs/{run_id}/events/summary` → `run.<ID>.events.summary.json`
키 집합은 `available` 값과 무관하게 항상 같다 (§4.1.6).
```ts
{ run_id,
  available: boolean, reason: string|null,   // optional 아님
  source: string|null,
  poi_summary: {poi_total,poi_eligible}|null,
  totals: {events, amt, policy_paid, extra_spent, coupon_eligible_events, would_buy_anyway}|null,
  day_type_counts: {weekday?, ...}|null,
  policy_paid_by_policy_id: { "P010": number }|null,   // sp 문자열을 서버에서 파싱해 합산
  by_day: [{day, ...위 6개}]|null, by_l1: [{l1, ...}]|null,
  by_day_l1: [{day,l1,events,amt,policy_paid}]|null,
  null_only_fields: string[],                // 실측 ["dong"]
  unknown: string[] }
```
4.8MB / 30,189행 → 52KB (90:1). `null_only_fields` 는 "값이 항상 null이라 조인에 못 쓴다"는 경고를 응답에 실은 것.

- `available:true` → `unknown:["dong"]`. **여기도 성공 응답의 `unknown` 이 비어 있지 않다** —
  `null_only_fields` 와 같은 사실을 §4 의 공통 규약으로도 선언한다.
- `available:false` → 데이터 7키 전부 `null`, `source:null`,
  `unknown:["totals","day_type_counts","policy_paid_by_policy_id","by_day","by_l1","by_day_l1","poi_summary"]`.
  실물: `run.BASE7500.events.summary.json` (`reason`: events.jsonl 없음 — export 미실행).

### 3.10 정책
| 엔드포인트 | 픽스처 | 비고 |
|---|---|---|
| `GET /api/policies` | `policies.index.json` | `data/neo4j_load/policies/P*.json` 스캔 |
| `GET /api/policies/{id}` | `policy.P010.detail.json` | 원본 JSON을 **`policy` 키 아래 통째로** + 파생값. 최상위에 펼치지 않는다 |
| `POST /api/policies/{id}/validate` | `policy.P010.validate.json` | §3.11 |
| `PUT /api/policies/{id}` | — | **S2 설계 항목.** S1은 스키마만 고정(§3.12) |

```ts
// GET /api/policies
{ total, source_dir: "data/neo4j_load/policies", items: Array<PolicyIndexItem>, unknown: string[] }

// GET /api/policies/{id}
{ file, policy: <원본 JSON 통째로>, source_dir,
  grant_key_effective, grant_key_source, unknown: string[] }
```

`PolicyIndexItem`:
`{id, file, name, type, announce_date, effective_from, effective_until, target_districts,`
`benefit_categories, grant_key, grant_key_effective, grant_key_source, poi_restricted,`
`has_decile_grants, has_income_grants, unknown}`.

#### `grant_key` — 파일값과 실효값을 분리한다

> **`grant_key` 를 파일에서 읽은 값 하나로 내보내면 화면이 거짓말을 한다.**
> 실측: 4개 정책 중 **P008 · P009 · P011 파일에는 `grant_key` 키가 아예 없다.** 그대로 내보내면 `null` 이고,
> 화면은 "지급 기준 미정"으로 렌더한다. 그런데 검증기는 그 정책들을 **`income` 기준으로 정상 동작시킨다.**
> "미정"이 아니라 "기본값 적용 중"이다.

`policy_preflight.py` 와 **같은 규칙**으로 실효값을 도출해 세 필드로 나눠 보낸다:

```
policy_preflight.py:74   grant_key = pol.get("grant_key") or "income"
policy_preflight.py:133  row["grant_key"] = "spend_decile" if decile_grants else grant_key
```

| 필드 | 뜻 | 실측 |
|---|---|---|
| `grant_key` | **파일에 적힌 값 그대로.** 없으면 `null` | P010 만 `"spend_decile"`, 나머지 3개 `null` |
| `grant_key_effective` | 검증기·시뮬이 **실제로 쓰는** 값 | P010 `"spend_decile"`, P008/P009/P011 `"income"` |
| `grant_key_source` | `"file"` \| `"default"` | P010 `"file"`, 나머지 3개 `"default"` |

> 화면에는 `grant_key_effective` 를 쓰고, `grant_key_source === "default"` 일 때 "기본값" 뱃지를 붙인다.
> `decile_grants` 가 있으면 파일의 `grant_key` 가 무엇이든 실효값은 `spend_decile` 이다 (§3.12의 우선순위 규칙).

### 3.11 검증 응답 — 기준 B2의 계약
검증은 **`scripts/sim/policy_preflight.py` 를 서브프로세스로 실행**하고 stdout을 파싱한다.
자체 검증 로직을 다시 구현하면 B2("100% 일치")를 만족할 수 없다.

```ts
{ policy_id, exit_code: number, ok: exit_code === 0,
  verdict: string,                       // "READY — 시뮬 구동 가능" | "FAIL N건 — 수정 후 재실행"
  counts: {pass?: n, warn?: n, fail?: n},
  checks: Array<{grade: "pass"|"warn"|"fail", message: string}>,
  prompt_preview: string,                // Dawn 프롬프트에 실제로 들어갈 정책 카드 (여러 줄, 들여쓰기 원문 유지)
  prompt_preview_persona: string|null,   // 미리보기가 어떤 대상자를 가정했는지. 헤더 괄호 안 문자열
  db_wiring_checked: boolean,            // NEO4J_URI 유무. false 면 치명 점검을 건너뛴 상태다
  stderr: string, command: string[],
  unknown: string[] }                    // db_wiring_checked 가 false 면 ["db_wiring"]
```

**stdout 파싱 규칙 (실측 확인).** 참조 구현은 `_build_fixtures.py::run_preflight`.

- 등급 줄은 `  ` + 이모지로 시작. `✅`=U+2705 → pass, `⚠️ `=U+26A0+U+FE0F+공백 → warn, `❌`=U+274C → fail.
  **`⚠️` 는 2 코드포인트다.** 첫 글자(U+26A0)로 판정하고 U+FE0F 를 lstrip 해야 한다. (여기서 한 번 틀렸다.)
- `결과:` 로 시작하는 줄이 최종 판정.
- `exit_code` 는 fail 개수가 0이면 0, 아니면 1 (코드 확인).

**`prompt_preview` 는 줄 접두로 줍지 않는다 — 블록으로 잘라낸다.**

> ⚠️ **접두 `-` 규칙은 틀렸다. 쓰지 마라.**
> `policy_preflight.py:138/150` 이 헤더를 찍고 **바로 다음 줄부터** `dawn_context._format_policy()` 의
> 반환값을 통째로 출력한다. 그 반환값에는 `dawn_context.py:507` 의 `_format_policy_facts()` 가 만드는
> **2칸 들여쓴 `  배경: …` 줄**이 섞여 있다. 이 줄이 정책 `description` **원문**이다.
> `-` 접두 규칙은 이 줄을 통째로 버린다. 실측 누락량:
> **P008 127자 / P009 143자 / P010 286자 / P011 84자** (P010 은 미리보기 593자 중 286자 = 48%).
> 버리면 화면에서 정책 `description` 을 고쳐도 미리보기가 변하지 않는다 → B2 위반.

블록 규칙:

| | 조건 | 동작 |
|---|---|---|
| **열림** | 줄이 `---` 로 시작하고 `프롬프트 미리보기` 를 포함 | `in_preview = true`. **헤더 줄 자체는 버린다** (preflight 라벨이지 카드 원문이 아니다) |
| **닫힘** | 등급 줄(이모지) / `결과:` 줄 / `=` 로만 이뤄진 구분선 | `in_preview = false` |
| **수집** | 그 사이의 모든 줄 | `rstrip()` 만 하고 **왼쪽 들여쓰기를 보존해 append** |

`prompt_preview_persona` 는 헤더의 괄호 안을 뽑는다: `/프롬프트 미리보기\s*\((?P<persona>.*?)\)\s*-*$/`.
실측: `"소비 1분위, 지급 당일"`(P010, decile), `"소득 '중상' 대상자, 지급 당일"`(P009/P011), `"소득 '중' 대상자, 지급 당일"`(P008).

**미리보기 블록은 등급 줄보다 먼저 나온다** (실측 P010 stdout: 6행 헤더 → 7~10행 카드 → 11행부터 `✅`).
"검사 결과를 다 읽은 뒤 마지막에 미리보기가 온다"고 가정하면 깨진다.

실측 P010 stdout (행 번호는 stdout 기준):
```
  6|--- P010 프롬프트 미리보기 (소비 1분위, 지급 당일) ---     ← 열림, 헤더는 버림
  7|- P010 [지원금] 민생회복 소비쿠폰 1차 | 2025-07-21~…
  8|  배경: 정부가 소상공인·자영업자 지원과 …               ← 2칸 들여쓰기. 접두 규칙이 버리던 줄
  9|- P010: 대상(소비 1분위) | 지급액 400,000원 | …
 10|- 판단 원칙: 소비 필요·시점·총액·POI는 …
 11|  ✅ id 존재                                            ← 닫힘
```

실측 결과 (로컬, `NEO4J_URI` 미설정):
| 정책 | exit | pass | warn |
|---|---|---|---|
| P008 (facility) | 0 | 10 | 1 |
| P009 (grant, income tier) | 0 | 13 | 2 |
| P010 (grant, spend_decile + poi_restricted) | 0 | 16 | 1 |
| P011 (grant, income tier + poi_restricted) | 0 | 15 | 2 |

> **`NEO4J_URI` 미설정이면 DB 배선 점검(`check_db_wiring`)이 통째로 건너뛰어지고 warn 1건이 붙는다.**
> "applied_to 0건 → 정책이 존재해도 아무 agent도 못 봄"이라는 **치명 결함을 로컬에서는 잡을 수 없다.**
> 화면은 이 warn 을 "검증 미완료" 배지로 명확히 구분해 표시해야 한다. 초록 체크로 뭉뚱그리면 안 된다.
> **실측된 4개 정책 모두 fail 이 0건이다. fail 상태 UI를 검증할 실물 픽스처가 없다 → 미확인.**
> S3는 fail 렌더링을 만들되, 그 화면의 정확성은 실제 fail 을 내는 정책이 생길 때 검증한다.

### 3.12 정책 JSON 스키마 (실측 4개 파일 + preflight 코드)
| 필드 | 타입 | 필수 | 근거 |
|---|---|---|---|
| `id` | str | ✅ | preflight 필수 6종 |
| `name` | str | ✅ | 〃 |
| `type` | str | ✅ | 유효값 `grant\|subsidy\|regulation\|facility\|campaign\|tax\|transit\|environment` (코드) / 실측 `grant`,`facility` |
| `description` | str | ✅ | Dawn 프롬프트에 **원문 주입**. 톤이 시뮬 결과에 직접 영향 |
| `effective_from` / `effective_until` | `YYYY-MM-DD` | ✅ | `from <= until` 검사 |
| `announce_date` | `YYYY-MM-DD` | — | 실측 4/4 존재 |
| `target_districts` | str[] | — | 실측 `["서울특별시"]`, `["강남구"]` |
| `benefit_categories` | str[] | — | `categories.yaml` 어휘와 대조. 실측 `[]` 또는 `["식사","카페","디저트","여가"]` |
| `benefit_rate` / `cap_per_agent` | number\|null | — | 실측 4/4 모두 null |
| `grant_key` | `"spend_decile"` \| `"income"`(기본) | — | **키 자체가 없는 게 실측 3/4** (P008/P009/P011). 없으면 `income`. §3.10 |
| `decile_grants` | `{"1".."10": 양의 정수}` | grant 조건부 | 키는 **문자열** |
| `excluded_deciles` | (str\|int)[] | — | 실측 `[]` |
| `income_grants` | `{"하"\|"중하"\|"중"\|"중상"\|"상": 양의 정수}` | grant 조건부 | |
| `excluded_income` | str[] | — | 실측 `["상"]` |
| `poi_restricted` | bool | — | true면 사용처 제한. P010/P011 |
| `notes` | str | — | 실측 4/4 존재. 운영 메모, 시뮬에 미주입 |

**검증 규칙 (preflight와 동일하게 구현하지 말고 preflight를 호출할 것):**
`decile_grants` 와 `income_grants` 가 동시에 있으면 **`decile_grants` 우선** + warn.
지급/제외 집합이 겹치면 fail. 지급액이 `int` 이고 `> 0` 이 아니면 fail.

### 3.13 실행 lock (기준 B8) — **설계**
파일 시스템에 lock 실물이 없다. **구조만 정의하고, 값은 S2가 런타임에 만든다.**
```ts
GET /api/runner/lock →
{ held: boolean,
  holder: { pid: number|null, host: string, started_at: string, run_id: string|null,
            command: string, output_dir: string } | null,
  acquired_by_console: boolean,
  stale_after_sec: number }
POST /api/runner/start → 409 Conflict + 위 객체  (lock 보유 중일 때)
```
`held:true` 면 실행 엔드포인트가 **서버에서** 409 를 낸다. UI 비활성화는 기준 미달이다(B8).
`holder` 는 반드시 화면에 표시한다 — 누가·언제부터.

**이 요구의 실측 근거**는 `runner.lock.evidence.json` 에 있다 (`logs_scripts/chain_p2.log` 원문):
```
[08-02 16:16:33] 1구간 BASE7500 기동 PID=1257679 (무정책 7/14~07/20)
[08-02 18:59:01] 2단계 시작 — 7,500명 순차 전후
[08-02 18:59:01] Day 0 덤프 복원          ← 이 단계의 `neo4j stop` 이 위 PID를 죽였다
```
2시간 42분, Day 0 4,500/7,500 지점에서 소실. 그 잔해가 `rescue/out_BASE7500` 이다.

### 3.14 진행 스트림 (SSE) — **설계**
`GET /api/runs/{run_id}/stream` → `text/event-stream`.
- 이벤트 페이로드는 **§3.3 `items[]` 의 원소 1개와 동일한 형태**로 고정한다. 새 스키마를 만들지 않는다.
- 서버는 `metrics/day_*.jsonl` 의 크기·mtime 변화를 감지해 **증분 구간만** 재집계한다. 전체 재파싱 금지.
- 폴링 폴백은 `GET /api/runs/{run_id}/days` 그대로. **SSE가 죽어도 화면이 동작해야 한다** (B6).

### 3.15 시각화·리포트 산출물 (S5) — **부분 확인**
저장소 `output/sim/` 실측: `report/FINAL_REPORT_5D_FULL.html`(601KB), `visualization/index.html`(51KB),
`visualization/sim_standalone*.html`(128~147MB), 대응 `.zip`(7~35MB).
> **이 파일들은 `gpu_exp_data` 3종 run과 파일명·메타데이터로 연결되어 있지 않다.**
> 어떤 run에서 생성됐는지 산출물만으로는 **미확인**이다.
> S5는 `GET /api/runs/{run_id}/artifacts` 를 정의하되, run↔산출물 매핑 규칙은
> **사람이 확인한 뒤** 확정한다. 추정 매핑을 화면에 사실처럼 표시하지 마라.
> 147MB HTML은 `<iframe>` 으로 감싸 원본 그대로 서빙한다 — 파싱하거나 재작성하지 않는다(비목표).

---

## 4. 불완전 run 표현 규칙

> `rescue/out_BASE7500` 가 이 규칙의 실물 테스트 케이스다.

### 4.1 원칙
1. **파일이 없는 것은 에러가 아니다.** 404 를 내지 말고, 리소스는 200으로 돌려주되 필드를 `null` 로 둔다.
2. **모르는 값에 0을 넣지 않는다.** `agents_target: 0` 은 거짓말이다. `null` + `unknown` 이 정답.
3. **모든 리소스는 최상위에 `unknown: string[]` 을 갖는다.** 값이 null인 이유가 "미확인"임을 명시적으로 선언한다.
   프런트는 `unknown.includes("x")` 로 "알 수 없음" 뱃지를 그린다. `== null` 만 보고 판단하지 않는다.
4. **부분 계산은 부분이라고 말한다.** §3.5의 `degraded:true` + `degraded_note`.
5. **`status: "incomplete"` 는 실패가 아니다.** "중단됨"과 "실패함"을 색으로 구분한다.
6. **응답의 키 집합은 상태에 따라 달라지지 않는다.** `available:false` 라고 필드를 빼지 않는다.
   해당 없는 값은 `null` 로 두고 키는 남긴다 (§4.1.6).

#### 4.1.3 보충 — `unknown` 이 붙는 위치와 범위

`unknown` 은 **리소스 최상위**의 필수 키다. optional 이 아니고, 정상 응답에서도 생략하지 않는다(`[]`).
현재 픽스처 **36개 전부가 최상위 `unknown` 을 갖는다.** 8종 리소스 전부가 대상이다:

| 리소스 | 최상위 `unknown` | 항목(`items[]`)별 `unknown` |
|---|:-:|---|
| §3.1 `runs.index` | ✅ | ✅ run 마다 (`days_planned`/`agents_target`/`completed_at`) |
| §3.2 `run.detail` | ✅ | — (items 없음) |
| §3.3 `days` | ✅ | ✅ 일자마다 (`agents_target`/`elapsed_sec`/`day_complete`/`timing_report`) |
| §3.4 `day` 집계 | ✅ | — |
| §3.5 `bottlenecks` | ✅ | — |
| §3.6 `slow` | ✅ | ❌ **`items[]` 는 `timing/slow_*.json` 원본 객체 그대로다. 손대지 않는다** |
| §3.7 `failed` | ✅ | ❌ 원본 error 행 그대로 |
| §3.8 `failures` | ✅ | ❌ 원본 `stage1_failures` 행 그대로 |
| §3.9 `events.summary` | ✅ | — |
| §3.10 `policies.index` / `policy.detail` | ✅ | ✅ 정책마다 |
| §3.11 `validate` | ✅ | — |
| `runner.lock.evidence` | ✅ | — |

> **`items[]` 의 원소에까지 `unknown` 을 붙이지 마라.** §3.6/§3.7/§3.8 의 `items` 는 원본 파일의 행을
> **무가공으로** 실어 나르는 통로다. 여기에 필드를 주입하면 "픽스처 = 원본"이라는 회귀 테스트 근거가 깨진다.
> 시계열/목록형(§3.1·§3.3·§3.10)의 원소는 서버가 조립한 객체이므로 `unknown` 을 갖는다.

**`unknown: []` 은 "완전함"의 뜻이 아니다.** `available:true` 인 §3.6 은 `["slow_thresholds_sec"]`,
§3.9 는 `["dong"]` 을 늘 달고 있다. 성공했는데도 구조적으로 못 아는 값이 있다는 뜻이다.

#### 4.1.6 보충 — `available:false` 의 키 집합

`available` 이 `false` 로 바뀌어도 **응답의 키 집합은 그대로다.** 값만 `null` 로 바뀐다.

> 근거: 프런트가 `resp.totals.amt` 를 읽을 때 `available` 을 먼저 검사하도록 강제하는 것보다,
> 키가 항상 있고 값이 `null` 인 편이 렌더 코드의 분기를 없앤다. 조건부로 키를 지우면
> TypeScript 의 optional(`?`) 이 전염되어 모든 소비 지점에 옵셔널 체이닝이 붙는다.
> `run.BASE.day.2025-07-21.bottlenecks.json`(true) 과 `run.BASE7500.day.2025-07-14.bottlenecks.json`(false)
> 의 **최상위 키 16개가 정확히 같다** — 이것이 본보기다.

`reason` · `degraded` 를 optional(`reason?`)로 선언하지 마라. `string|null` · `boolean` 이다.
§3.5(bottlenecks) · §3.6(slow) · §3.7(failed) · §3.8(failures) · §3.9(events.summary) 5종 전부에 적용된다.

### 4.2 rescue 케이스에서 실제로 모르는 것 (실측)
| 항목 | 왜 모르나 | API 표현 |
|---|---|---|
| 목표 agent 수 | `summary.json` 없음 | `agents_target: null`, `unknown:["agents_target"]`. `log_hint.agents_target: 7500` 은 **참고용 별도 표시** |
| 계획 일수 / 시작일 | 〃 | `plan.planned_days/start_day: null`, `log_hint` 로만 보조 |
| 일자 소요시간 | 〃 (`summary[]` 에 그 일자 없음) | `elapsed_sec: null` |
| 그 일자가 끝났는지 | 〃 | `day_complete: false` — "미완료"가 아니라 **"완료 여부 미확인"**으로 문구를 쓴다 |
| 병목 하위 경로 | `timing/` 없음 | `available:false` + `fallback_rank`(phase 4개만) |
| 최종 실패 agent 목록 | `failed_*.json` 없음 | `checkpoint_failed_count: null` |
| 결제 이벤트 / POI 요약 | `events.jsonl` 없음 (export 미실행) | `available:false` + reason |
| 진행률 | 분모(목표 수)가 없음 | `progress_ratio: null` → **진행 바를 그리지 않는다** |

### 4.3 아는 것 — 여기까지는 정확히 보여준다
`agents_ok: 4533`, `metrics_rows: 4533`, `checkpoint_done_count: 4500`, `metrics_bytes: 19,599,953`,
`elapsed/timing_t_*/avg_sat/balance` 전체 분포, spend_decile 10분위 분해, 폴백 카운터, LLM 호출 수,
캐시 적중률(`policy_hit_rate: 0.764394`), stage1 파싱 실패 전량.
→ **"중단된 run이라 아무것도 못 본다"는 잘못된 결론이다.** 4,533명분 데이터는 온전하다.

### 4.4 진행 중 run (미실측)
실측 3종에 "지금 돌고 있는 run"은 없다. 다음은 **코드에서 도출한 규칙**이며 실측으로 확인되지 않았다:
- `summary.json` 에 `updated_at` 이 있고 `completed_at` 이 없으면 진행 중.
- 진행 중인 일자는 `metrics/day_X.jsonl` 은 있는데 `timing/day_X.json` 과 `checkpoints/failed_X.json` 이 없는 일자.
- 그 일자의 `done_X.json` 은 500명 단위로만 갱신되므로 최대 499명 뒤처진다.
→ S2는 이 규칙으로 구현하되, **실제 진행 중 run으로 검증하기 전에는 "확정"으로 표시하지 않는다.**

---

## 5. 대용량 대응 — 서버 측 집계 지점 (기준 B5)

### 5.1 절대 프런트로 보내지 않는 파일
압축비는 **픽스처와 그 원본을 짝지어** 잰 값이다(둘 다 실물). "실측 최대"와 다른 파일을 비교하지 않았다.

| 파일 | 짝지은 원본 | 대체 리소스 | 실측 응답 | 압축비 |
|---|--:|---|--:|--:|
| `metrics/day_*.jsonl` | **19,599,953 B** / 4,533행 | §3.4 일자 집계 | **7,009 B** | **2,796:1** |
| `events.jsonl` (FINAL) | 4,799,810 B / 30,189행 | §3.9 집계 | 53,315 B | 90:1 |
| `events.jsonl` (BASE) | 1,236,982 B / 7,785행 | §3.9 집계 | 15,721 B | 79:1 |
| `timing/slow_*.json` | 558,763 B (BASE Day 0) | §3.6 페이지 | 68,456 B | 8:1 (limit=15) |
| `stage1_failures.jsonl` | 336,311 B (FINAL) | §3.8 집계+페이지 | 20,780 B | 16:1 (limit=12) |
| `checkpoints/done_*.json` | 129,528 B (4,500명) | 개수만 (`checkpoint_done_count`) | 정수 1개 | — |
| `timing/day_*.json` | 15,938~16,171 B | 그대로 전송해도 무방 | 동일 | 1:1 |
| `visualization/*.html` | **147MB** | iframe 직접 서빙 (§3.15) | — | — |

`timing/slow_*.json` 의 **전 일자 최대는 729,811 B** 다(위 표는 픽스처가 뽑힌 Day 0 기준 558,763 B).
`checkpoints/done_*.json` 은 7,500명이면 약 216KB 로 늘어난다(4,500명 129,528 B 의 선형 외삽 — **미실측**).

### 5.2 구현 제약 (S2 필수)
1. **jsonl 은 반드시 라인 스트리밍.** `json.loads` 를 줄 단위로, 전체 `read()` 금지.
   구현 참조: `web/fixtures/_build_fixtures.py::aggregate_day`. 4,533행 × 4.3KB 를 상수 메모리로 접는다.
2. **집계 결과 캐시 키 = `(경로, size, mtime_ns)`.**
   §2.1대로 resume 시 jsonl 이 재작성되므로 mtime 없는 캐시는 오답을 낸다.
3. **완료된 일자는 불변**(더 이상 append 되지 않음) → 영구 캐시 가능.
   진행 중 일자만 재계산한다. 판별: 같은 일자의 `timing/day_*.json` 존재 여부.
4. **SSE 증분**: 마지막으로 읽은 byte offset 을 보관하고 그 뒤만 파싱한다.
5. **집계는 워커 스레드/프로세스에서.** 19MB 파싱이 이벤트 루프를 막으면 B3(실행 중 안전)·B4(2초)를 함께 깬다.
6. **B4 대응**: 첫 렌더는 §3.1 + §3.3(캐시된 일자) 만으로 충분하다.
   §3.4/§3.6 은 사용자가 특정 일자를 열 때 지연 로드한다.

### 5.3 읽기 전용 보장 (기준 B3)
콘솔은 `metrics/`·`checkpoints/`·`timing/`·`summary.json` 을 **O_RDONLY 로만** 연다.
쓰기·이동·삭제·잠금 금지. 시뮬레이터가 같은 파일에 append 중이므로,
읽는 도중 마지막 줄이 잘려 있을 수 있다 → **`JSONDecodeError` 는 정상 상황이다.**
해당 줄을 조용히 버리고 `status_counts.malformed` 로 센다 (§3.4). 예외를 올리지 마라.

---

## 6. run_id · 경로 규칙

- `run_id` = run 루트 디렉터리명에서 `out_` 접두를 제거. 실측: `BASE`, `FINAL`, `BASE7500`.
- **`run_id` 는 전역 유일하지 않을 수 있다.** `out_BASE` 와 `rescue/out_BASE7500` 은 부모가 다르다.
  같은 이름이 두 곳에 나타나면 S2가 `<부모디렉터리>/<run_id>` 로 한정해야 한다. 지금 3종은 충돌 없음.
- run 루트 탐색: 데이터 루트에서 **깊이 2까지** `metrics/` 를 가진 디렉터리를 찾는다.
  깊이 1(`out_BASE`)과 깊이 2(`rescue/out_BASE7500`) 둘 다 실측에 존재한다.
- `logs_scripts/` 는 run 루트가 아니다. run 과의 연결은 `run_<run_id>.log` 파일명 규칙 + `output:` 줄 basename.

---

## 7. 다른 조각이 반드시 알아야 할 주의점

1. **`sp` 는 JSON 문자열이다** (§2.8). `JSON.parse` 를 프런트에서 하지 말고 서버에서 접어 보낸다.
2. **`events.jsonl` 의 `dong` 은 전부 null이다** (§2.8). 행정동 지도 조인의 근거로 쓸 수 없다.
   대신 `metrics.aid` 앞 8자리가 행정동 코드다 — 다만 이 해석은 **aid 명명 규칙에서 읽은 것이며 별도 검증이 필요하다.**
3. **error 행은 필드가 5개뿐이다** (§2.1). 목록 컴포넌트가 공통 필드를 가정하면 깨진다.
4. **`stage1_failures`(파싱 실패) ≠ `failed_*.json`(agent 실패)** (§2.7). 실측 BASE: 전자 32건, 후자 0건. 같은 화면에 합산 금지.
5. **`timing_t_night_finalize` 는 Day 0에 없다** (§2.1). 키 부재를 정상 처리하라.
6. **검증은 preflight 서브프로세스로만** (§3.11). 재구현하면 B2 불통과.
7. **`⚠️` 파싱 함정** (§3.11). 2 코드포인트다.
8. **로그는 정본이 아니다** (§2.10). `summary.json` 이 이긴다.
9. **`out_BASE` 는 이름과 달리 정책이 적용된 run이다.** `events.jsonl` 에 `{"P010": ...}` 결제가 820건 실측된다.
   run_id 만 보고 "무정책 대조군"이라고 화면에 쓰지 마라. **run↔정책 매핑은 산출물에 없다 → 미확인.**
   (`chain_p2.sh` 를 보면 `BASE7500`=무정책 / `POL7500`=P010 이라는 명명 의도가 있으나, `out_BASE`/`out_FINAL` 이
   어떤 체인에서 나왔는지는 확인되지 않았다.)
10. **`slow_*.json` 의 임계값은 파일에 없다** (§2.5). "60초 초과" 같은 문구를 하드코딩하지 마라.
11. **금액은 원 단위 정수**, 비율은 0~1 float. 퍼센트 변환은 표시 계층에서만.
12. **인코딩은 전부 UTF-8**, 한국어 값이 그대로 들어 있다 (`"70대이상"`, `"seoul_smallbiz"`, `l1` 값 전부).
    파일 입출력에 반드시 `encoding="utf-8"` 을 명시한다 (Windows 기본 cp949 로 깨진다 — 실제로 겪었다).

---

## 8. 픽스처

`web/fixtures/` 에 **JSON 36개, 합계 405,200 B (395.7KB), 개별 최대 68,456 B** (`run.BASE.day.2025-07-21.slow.json`).
각 파일이 곧 위 엔드포인트의 실물 응답이다. 전부 200KB 이하.
재생성: `python web/fixtures/_build_fixtures.py` (읽기 전용, `SIM_DATA_ROOT` 로 소스 경로 변경 가능).
상세는 `web/fixtures/README.md`.

> 개수·크기는 픽스처를 재생성하면 바뀐다. **이 문장을 고치지 않고 픽스처만 늘리지 마라.**
