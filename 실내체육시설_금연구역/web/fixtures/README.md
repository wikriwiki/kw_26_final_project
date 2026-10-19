# 픽스처 — 각 파일이 대표하는 상황

> 계약 본문은 `web/CONTRACT.md`. 이 문서는 **어떤 파일이 어떤 상황의 실물인가**만 다룬다.
> **전부 실제 산출물에서 뽑았다. 손으로 쓴 숫자는 하나도 없다** (기준 B1).

## 재생성

```bash
python web/fixtures/_build_fixtures.py
# 소스 경로 변경: SIM_DATA_ROOT=... python web/fixtures/_build_fixtures.py
```
소스: `C:\Users\srdyh\gpu_exp_data\20260802\` + 저장소 `data/neo4j_load/policies/*.json`.
생성기는 **읽기 전용**이다. `scripts/` `data/` `output/` 를 건드리지 않는다.
정책 검증 픽스처는 `scripts/sim/policy_preflight.py` 를 실제로 실행해 stdout 을 파싱한 결과다.

**JSON 36개, 합계 405,200 B (395.7KB), 개별 최대 68,456 B** (`run.BASE.day.2025-07-21.slow.json`, 제한 200KB).

> 개수·크기를 늘리거나 줄이면 **이 줄과 `CONTRACT.md` §8 을 같이 고친다.**
> 재측정: `python -c "import glob,os;f=glob.glob('web/fixtures/*.json');print(len(f),sum(os.path.getsize(x) for x in f),max(os.path.getsize(x) for x in f))"`

---

## 1. 대표하는 3가지 상황

| run_id | 원본 | 대표 상황 | 콘솔이 시험받는 것 |
|---|---|---|---|
| **BASE** | `out_BASE` | 정상 완료. 7일 × 200명, 실패 0 | 기본 행복 경로 |
| **FINAL** | `out_FINAL` | 장기 완료. 28일 × 200명, **agent 실패 2건 실재** | 긴 시계열 + 에러 상태 |
| **BASE7500** | `rescue/out_BASE7500` | **중단.** Day 0만, 7,500 목표 중 4,533명에서 죽음 | 불완전 표현 + 19MB 내성 |

---

## 2. run 리소스

### 공통 진입
| 파일 | KB | 상황 |
|---|--:|---|
| `runs.index.json` | 1.9 | `GET /api/runs`. 3종이 한 화면에 섞여 나오는 상태. `BASE7500` 만 `status:"incomplete"` 이고 `unknown` 이 3개 차 있다 |
| `runner.lock.evidence.json` | 1.2 | B8 lock 이 막아야 하는 **실제 사고**. `chain_p2.log` 원문 7줄 — 16:16:33 기동, 18:59:01 두 번째 체인이 `neo4j stop` |

### BASE — 정상 완료
| 파일 | KB | 상황 |
|---|--:|---|
| `run.BASE.detail.json` | 3.0 | `summary.json` 이 있어 `plan` 이 전부 채워진 상태. `unknown: []` |
| `run.BASE.days.json` | 6.2 | 7일 전부 `progress_ratio: 1.0`, `day_complete: true`. **진행 바가 정상 동작하는 기준선** |
| `run.BASE.day.2025-07-21.json` | 6.8 | Day 0 집계. **정책 지급 당일** — `grant_applied_today` 합 37,650,000원, `by_spend_decile` 10개 분위 전부 채워짐. `spend_decile_unknown_agents: 0` |
| `run.BASE.day.2025-07-21.bottlenecks.json` | 16.5 | `available:true`. `bottleneck_rank` 30개 + `timings` 58경로 + `cache` + `policy_payment` **완전판** (첫날이라 58, 나머지 날은 59) |
| `run.BASE.day.2025-07-21.slow.json` | 66.9 | 원본 558,763 B → limit=15. `total:164`, `phase_counts` 로 어느 단계가 느린지. `unknown:["slow_thresholds_sec"]` — **성공인데도 임계값은 모른다** |
| `run.BASE.day.2025-07-21.failed.json` | 0.2 | **`total:0` 의 실물.** "그 날이 끝났고 실패가 없었다". BASE7500 의 `total:null`(미확인)과 **다른 화면으로 그려야 한다** |
| `run.BASE.failures.json` | 20.1 | stage1 파싱 실패 32건. `by_error_type` = ValidationError 21 / JSONDecodeError 11, `by_day` 7일 분포. **`ValueError`·`length` 는 여기 없다 — FINAL/BASE7500 을 함께 봐야 값 목록이 완성된다** |
| `run.BASE.events.summary.json` | 15.4 | 1,236,982 B / 7,785행 → 15,721 B (79:1). `policy_paid_by_policy_id: {"P010": 4,857,655}`, `l1` **15종** |

> `BASE` 는 이름과 달리 **P010이 적용된 run이다.** 무정책 대조군으로 라벨하지 마라 (CONTRACT §7.9).

### FINAL — 장기 완료 + 실패 실재
| 파일 | KB | 상황 |
|---|--:|---|
| `run.FINAL.detail.json` | 8.9 | 28일치 `day_summaries`. 시계열 차트 축이 길어졌을 때의 레이아웃 시험 |
| `run.FINAL.days.json` | 24.5 | 28행. **2025-08-03 과 2025-08-14 만 `agents_ok:199`, `metrics_rows:200`** — 행 수와 성공 수가 갈리는 실물 |
| `run.FINAL.day.2025-08-17.json` | 6.8 | 마지막 날 집계 |
| `run.FINAL.day.2025-08-17.bottlenecks.json` | 16.6 | 완전판 병목 |
| `run.FINAL.day.2025-08-17.slow.json` | 66.1 | 페이지 |
| **`run.FINAL.day.2025-08-03.json`** | 7.5 | **`error_samples` 에 진짜 실패 행 1건.** 필드가 `aid/status/elapsed/error/trace` 5개뿐인 그 행이다 |
| **`run.FINAL.day.2025-08-03.failed.json`** | 0.8 | `checkpoints/failed_2025-08-03.json` 원본. `Stage1 failed after 3 attempts: Expecting ',' delimiter...` + traceback |
| `run.FINAL.failures.json` | 20.3 | stage1 파싱 실패 203건 (28일 누적). **`error_type` 에 `ValueError` 1건, `finish_reason` 에 `length` 1건** — BASE 에 없는 값이 여기 있다 |
| `run.FINAL.events.summary.json` | 52.1 | **4,799,810 B / 30,189행 → 53,315 B (90:1).** `by_day_l1` 28일 × **18업종**(BASE 는 15업종 — 축을 상수로 고정하지 마라) |

> `run.FINAL.day.2025-08-03.*` 는 **에러 상태 UI를 실물로 검증할 유일한 픽스처**다.
> 8-14 도 같은 상황이지만 픽스처는 첫 번째 것만 만들었다.

### BASE7500 — 중단 run (핵심 테스트 케이스)
| 파일 | KB | 상황 |
|---|--:|---|
| `run.BASE7500.detail.json` | 1.1 | `plan` 이 **전부 null**, `unknown` 4개. 대신 `log_hint` 에 `agents_target:7500`, `last_progress_line:"4500/7500 (ok=4500, err=0) @ 0.5/s, ETA 6439s"` — **출처가 분리되어 있다** |
| `run.BASE7500.days.json` | 0.7 | 1행. `progress_ratio: null` / `elapsed_sec: null` / `day_complete: false` / `timing_report_present: false`, `unknown` 4개. **진행 바를 그리면 안 되는 상태** |
| **`run.BASE7500.day.2025-07-14.json`** | 6.8 | **19,599,953 B → 7,009 B (2,796:1).** 기준 B5의 증거. 4,533명 분포·폴백 카운터 전부 온전. **`spend_decile_unknown_agents: 5`** — 아래 참고 |
| `run.BASE7500.day.2025-07-14.bottlenecks.json` | 1.2 | `available:false` + `degraded:true`. `fallback_rank` 에 `phase.t_s1/t_s2/t_dawn/t_write_plan` 4개만. **빈 화면이 아니라 축소된 화면.** `available:true` 픽스처와 최상위 키 16개가 동일 |
| `run.BASE7500.day.2025-07-14.slow.json` | 0.3 | `available:false`. `timing/slow_*.json` 이 없다. 키는 그대로 두고 값만 null |
| **`run.BASE7500.day.2025-07-14.failed.json`** | 0.3 | **`total:null` 의 실물.** `unknown:["failed_checkpoint"]`. `failed_*.json` 은 일자 종료 시에만 쓰이므로 중단 run 에 없다 |
| `run.BASE7500.events.summary.json` | 0.5 | `available:false` + reason. export 단계까지 가지 못했다. 데이터 7키 전부 null |
| `run.BASE7500.failures.json` | 20.3 | **중단 run에도 stage1 파싱 실패는 남아 있다**(101건, `available:true`). "아무것도 못 본다"가 틀렸다는 증거 |

**이 run에서 아는 것 / 모르는 것**
- 안다: `agents_ok:4533`, `checkpoint_done_count:4500`, `metrics_bytes:19,599,953`,
  `policy_hit_rate: 0.764394`, elapsed p50 96.32초 / p95 159.504초 / max 403.04초, stage1 파싱 실패 101건 전량.
- 모른다: 목표 agent 수, 계획 일수, 시작일, 일자 소요시간, 완료 여부, 병목 하위 경로, 최종 실패 목록, 결제 이벤트.

> **10분위는 "전량"이 아니다.** ok 4,533행 중 **5행의 `spend_decile` 이 null** 이다.
> 이 5행은 버려지지 않고 `by_spend_decile` 의 **`spend_decile: null` 버킷**(배열 마지막)에 들어간다.
> 덕분에 `sum(by_spend_decile[].agents) == agents_ok == 4533` 항등식이 성립한다.
> 화면은 `spend_decile_unknown_agents > 0` 일 때 **"분위 미상 5명" 열을 반드시 함께 그린다** (CONTRACT §3.4).
> 나머지 3개 day 픽스처는 이 값이 0 이고 `unknown` 이 비어 있다 — **양쪽 분기를 다 시험할 수 있다.**

---

## 3. 정책 리소스

| 파일 | KB | 상황 |
|---|--:|---|
| `policies.index.json` | 2.5 | 4건 목록. `type` 이 `grant` 3 / `facility` 1, `grant_key_effective`·`poi_restricted` 로 카드가 갈린다 |
| `policy.P010.detail.json` / `.validate.json` | 2.8 / 3.7 | **소비 10분위 차등 + 사용처 제한.** 10분위 입력 화면의 주 케이스. pass 16 / warn 1. **`grant_key_source:"file"` 인 유일한 정책** |
| `policy.P009.detail.json` / `.validate.json` | 1.2 / 2.9 | **소득 5분위 tier + 제외 계층(`excluded_income:["상"]`).** pass 13 / **warn 2** — 소득 기준 근사 경고가 실물로 나온다 |
| `policy.P011.detail.json` / `.validate.json` | 1.2 / 3.1 | 소득 tier + 사용처 제한 조합. pass 15 / warn 2 |
| `policy.P008.detail.json` / `.validate.json` | 1.4 / 2.2 | **`type:"facility"` — 지급액이 아예 없는 정책.** 지급 관련 필드가 전부 빠진 카드 렌더 시험. pass 10 / warn 1 |

### `grant_key` — 파일에 없는 게 정상이다

**P008 · P009 · P011 파일에는 `grant_key` 키가 아예 없다.** 그래도 검증기는 `income` 기준으로 정상 동작한다.
그래서 픽스처는 세 필드로 나눠 싣는다 (CONTRACT §3.10):

| 정책 | `grant_key`(파일값) | `grant_key_effective`(실효) | `grant_key_source` |
|---|---|---|---|
| P008 | `null` | `income` | `default` |
| P009 | `null` | `income` | `default` |
| P010 | `spend_decile` | `spend_decile` | `file` |
| P011 | `null` | `income` | `default` |

> 화면이 `grant_key` 만 읽으면 4건 중 3건이 **"지급 기준 미정"**으로 렌더된다 — 거짓이다.
> `grant_key_effective` 를 쓰고, `grant_key_source === "default"` 에 "기본값" 뱃지를 붙인다.

### `prompt_preview` — `배경:` 줄이 들어 있어야 정상이다

각 `.validate.json` 의 `prompt_preview` 는 **Dawn 프롬프트에 실제로 주입되는 정책 카드 원문**이다.
정책 화면의 "미리보기" 패널은 이 문자열을 **줄바꿈·들여쓰기 그대로** 보여주면 된다.

실측 4건 모두 **4줄**이고, 그중 **2번째 줄이 2칸 들여쓴 `  배경: …`** 이다. 이 줄이 정책 `description` 원문이다.

| 정책 | 미리보기 총 길이 | 그중 `배경:` 줄 | 비중 |
|---|--:|--:|--:|
| P008 | 348자 | 127자 | 36% |
| P009 | 434자 | 143자 | 33% |
| P010 | 593자 | 286자 | **48%** |
| P011 | 396자 | 84자 | 21% |

> **회귀 검사.** `prompt_preview` 에 `배경:` 이 없으면 파싱이 틀린 것이다.
> `-` 로 시작하는 줄만 줍는 규칙이 이 줄을 통째로 버린다 (CONTRACT §3.11).
> 그 상태에서는 정책 `description` 을 고쳐도 미리보기가 변하지 않아 **B2 를 못 지킨다.**
> `prompt_preview_persona` 는 미리보기가 가정한 대상자다 (P010 `"소비 1분위, 지급 당일"`).

`db_wiring_checked` 는 4건 모두 `false` 이고 `unknown` 은 `["db_wiring"]` 이다.

> **fail 픽스처는 없다.** 저장소의 4개 정책 전부 `exit_code:0` 이다.
> S3는 fail 렌더링을 구현하되, 실물 검증은 실제로 fail 하는 정책이 생길 때 한다. 지어내지 마라.
> 4건 모두 `NEO4J_URI 미설정 → DB 배선 점검 생략` warn 을 갖는다. 이건 "검증 완료"가 아니라
> **"치명 항목을 아직 못 봤다"**는 뜻이다 (CONTRACT §3.11). 초록 체크로 묶지 마라.

---

## 4. 화면별 최소 픽스처 세트

| 조각 | 먼저 붙일 파일 |
|---|---|
| **S3 정책 설정** | `policies.index.json` → `policy.P010.detail.json` → `policy.P010.validate.json` (+ warn 2건 케이스로 `policy.P009.validate.json`, 지급 없는 케이스로 `policy.P008.*`) |
| **S4 실행 모니터** | `runs.index.json` → `run.BASE.days.json`(정상) → `run.BASE7500.days.json`(불완전) → `run.BASE.day.2025-07-21.bottlenecks.json`(완전) → `run.BASE7500.day.2025-07-14.bottlenecks.json`(축소) → **`failed` 3종을 다 붙일 것**: `run.BASE.day.2025-07-21.failed.json`(`total:0`) / `run.FINAL.day.2025-08-03.failed.json`(`total:1`) / `run.BASE7500.day.2025-07-14.failed.json`(`total:null`) → `runner.lock.evidence.json`(B8) |
| **S5 시각화 통합** | `runs.index.json` + `run.*.events.summary.json`. 3D/리포트 HTML 은 `output/sim/` 실물을 iframe 으로 |
| **S2 백엔드** | 전부. 자기 응답과 픽스처가 **바이트 단위로 같아야** 한다 (§5) |

---

## 5. S2 에게 — 픽스처는 회귀 테스트다

`_build_fixtures.py` 의 함수들은 **참조 구현**이다.

| 함수 | 대응 엔드포인트 |
|---|---|
| `scan_run` | `GET /api/runs`, `GET /api/runs/{id}` |
| `aggregate_day` | `GET /api/runs/{id}/days/{day}` — **19MB 스트리밍 집계의 정답** |
| `status_scan` | `GET /api/runs/{id}/days` 의 **B4 경량 경로.** 카운트 3종만, 19.6MB 를 0.044초에 (`aggregate_day` 0.92초의 1/21). 36개 일자 전부에서 `aggregate_day` 와 결과 완전 일치 — CONTRACT §3.3.1 |
| `day_progress` | `GET /api/runs/{id}/days` |
| `bottlenecks` | `GET /api/runs/{id}/days/{day}/bottlenecks` (degraded 폴백 포함) |
| `slow_page` / `failures_page` | 페이지 리소스 |
| `events_summary` | `GET /api/runs/{id}/events/summary` — `sp` 문자열 파싱 포함 |
| `run_preflight` | `POST /api/policies/{id}/validate` — **`⚠️` 2 코드포인트 파싱 포함** |
| `policy_index` | `GET /api/policies` |

FastAPI 응답을 같은 소스에 대해 뽑아 이 파일들과 diff 하면 B1 회귀를 자동으로 잡을 수 있다.
스키마를 바꿔야 하면 **`_build_fixtures.py` 와 `CONTRACT.md` 를 먼저 고치고 픽스처를 재생성한 뒤** 구현을 따라오게 한다.
