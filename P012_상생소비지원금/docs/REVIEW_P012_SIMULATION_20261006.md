# P012_simulation 코드 리뷰와 실험 코드 정합성 보고서

- 작성: 2026-10-06, Claude (doing_gyu 요청)
- 목적: **현재 돌고 있는 금연구역 실험(No_SmokingZone_EXP)과 P012 계열 실험 코드가 같은 시뮬레이션 논리로 돌게 한다.** 한쪽에서 찾은 고침이 다른 쪽에 빠지거나, 같은 함수가 양쪽에서 다르게 동작하는 일을 막는 것이 목표다.
- 리뷰 대상: `origin/P012_simulation` `7bc8e45f`(main과의 분기점 `77ec86cf` 이후 444커밋, 830파일). 정합성 대조는 `7bc8e45f`를 기준으로 했고, 이후 올라온 `d9813f42`에서 핵심 항목을 다시 확인했다.
- 검증 방식: 영역별 리뷰어 5명과 하위 검토 4건. 모든 지적은 실제 코드를 열어 확인한 것만 남겼다. 실행 중인 실험 코드는 Vast 서버에서 읽기 전용으로 복사해 함수 단위로 대조했다.

---

## 1. 결론

1. **기준 논리는 "실행 중인 금연 실험의 논리"로 정한다.** 실제로 돌고 있는 코드는 `frozen v22`(소스 압축본 SHA256 `e6ab8021…`)에 런타임 패치 층 7개를 얹은 것이다. 저장소에서 이와 같은 소스는 `No_SmokingZone_EXP`(`17d21f64`)와, 이번에 정합 보정을 더한 `claude/no-smoking-ops-parity`이다.
2. **P012의 `scripts/sim`은 9월 29일 공개본(`7e9cc723`) 기준이다.** 금연 실험의 최신 재시도·야간 수정(`679b17af`)이 빠져 있다. "doinggyu 시뮬레이터를 기준으로 삼았다"는 말과 달리, 지금 돌고 있는 논리와는 Stage1, Stage2, Night2 재시도 경로가 다르다.
3. **P012의 `EXP_*` 정책 기능은 플래그가 꺼져 있으면 금연 실험과 동작이 같다.** 함수 단위로 확인했다. 따라서 공통 논리를 맞추는 데 필요한 것은 아래 3절의 이식 목록뿐이다.
4. **양쪽에 공통으로 있는 시뮬레이션 버그(4절)는 지금 고치지 않는다.** 금연 실험은 OFF/ON 두 조건이 같은 코드로 돌아야 하고(점수화할 때 `code_sha256` 일치를 검사함), 지금 OFF 14일 중 10일째다. 이 버그들은 실험이 끝난 뒤 **양쪽에 동시에** 고쳐야 다시 같은 논리가 된다.
5. **P012만의 결함(5절)이 많다.** 보고서 숫자에 직접 영향을 주는 것과, 병합 과정에서 빠진 고침이 포함된다. 지금 상태로는 main 병합이 어렵다(6절).

---

## 2. 기준 논리: 실행 중인 금연 실험

| 층(앞이 우선) | 역할 |
|---|---|
| `runtime_hotfix_v22_gpu_pool` | LLM 요청을 GPU 풀 프록시로 보내는 것만 담당. 생성·검증·저장 논리는 바꾸지 않음 |
| `night_progress` | `run_day`, Night2 완료·복구(`night_completion`, `night_recovery`), 쌍별 6회 예산, skip 영수증 |
| `retry_contract` | Stage1/Stage2/`grounded_schema` 전체 교체(2017-11-24부터 적용) |
| `grounding`, `retry` | 이전 근거·재시도 프롬프트 |
| `night2_recovery` | Night2 `evidence_ref` 0 채움("E12" → "E0012") |
| `night2`, `skip` | Night2 재시도 제한, skip 날의 State 저장 |
| frozen v22 `scripts/sim` | 기반 |

- 실행 설정: `POLICY_BACKTEST_DETERMINISTIC=1`(temperature 0), JSON 문법 모드 `json_object`, Policy 노드 0개, grounded 모드.
- `No_SmokingZone_EXP`의 `scripts/sim`은 공통 모듈 전부에서 실행 논리와 함수 단위로 같다. 유일한 차이였던 Night2 0 채움은 `claude/no-smoking-ops-parity`의 `9b5a665e`로 넣었다.

---

## 3. 두 버전을 같은 논리로 맞추기 위해 P012에 필요한 작업

### 3.1 금연 실험의 재시도·야간 수정 이식 (필수)
`git diff 7e9cc723 679b17af -- scripts/sim`을 적용한다. 대상은 8개 파일이다. `7bc8e45f`에는 충돌 없이 적용되는 것을 확인했고, `run_simulation` 조각만 줄 위치가 +27 밀린다.

| 파일 | 바뀌는 논리 |
|---|---|
| `grounded_schema.py` | 재시도 피드백(사유 900자, 실패 항목 발췌, 900/500 절단) |
| `stage1_intent.py` | 같은 오답 반복 시 조기 중단 제거, `with_agents`·`HH:MM` 검사, 실제 호출 수로 시도 집계, "재시도 n/m" 접두 |
| `stage2_poi.py` | 빠진 order만 다시 요청, 허용 POI 힌트를 피드백에 통합, 이미 받은 선택의 동일 반복 무시 |
| `night_intent_llm.py`, `interview_evidence.py` | 쌍별 재시도 예산, skip된 쌍 처리 |
| `night_completion.py`, `night_recovery.py`(신규) | 쌍별 6회 예산을 재시작해도 이어감, 소진 쌍은 skip 영수증과 함께 봉인, 날짜는 막지 않음 |
| `run_simulation.py` | 인라인 Night2 블록을 `complete_night`로 교체 |

**이 이식이 빠져 있으면:** grounded·temperature 0 조건에서 P012 코드는 재시작할 때마다 같은 Night2 호출 3번을 반복하다 실패한다. 이 경우 고집스러운 쌍 하나가 그날 전체를 영구히 막을 수 있다. 금연 실험에서 실제로 겪은 문제다.

### 3.2 Night2 근거번호 0 채움 (필수)
- 커밋: `claude/no-smoking-ops-parity`의 `9b5a665e`. `night_intent_llm._extract_first_json`에서 "E12"를 "E0012"로 바꾼다.

### 3.3 운영 코드 수정 (권장, 양쪽 동일 파일)
`deploy/vast/run_shared.py`, `backup_checkpoint.py`, `watch_pipeline.py`는 두 브랜치에서 바이트까지 같다. 커밋 `1f12e18c`를 그대로 가져가면 된다.

| 문제 | 수정 |
|---|---|
| 감시 스크립트가 남은 simulator를 실제로는 종료하지 못하고, 그 상태에서 Neo4j를 멈춤(`os.killpg(pid)`는 그룹 리더가 아니면 실패하는데 그 실패를 무시함) | 실제 프로세스 그룹에 신호를 보내고, 살아남은 simulator가 있으면 백업을 건너뜀 |
| 복구기가 감독자를 재시작하는 동안 오래된 `failed` 상태를 보고 새 실행 밑에서 Neo4j를 멈춤(운영에서 1회 발생) | 감독자가 없고 `failed`가 180초 넘게 유지될 때만 동작 |
| 백업 실패 시 수 GB 임시 파일이 남아 디스크를 채움 | 커밋되지 않은 시도의 압축본과 덤프를 정리(커밋된 Dec-2 분기 덤프는 보존) |
| `neo4j stop`이 실패하거나 시간을 넘기면 재시작하지 않음 | 정지를 시도했으면 항상 재시작하고 Bolt 포트로 확인 |
| 감독자 동시 실행 틈 | 접두어별 `flock` |
| 근거 감사 출력에 stderr 한 줄만 섞여도 JSON이 깨지고 재개가 영구히 막힘 | stderr를 별도 로그로 보내고, 파싱이 성공해야 결과 파일로 교체 |

### 3.4 `d9813f42`에 새로 생긴 병합 버그 (필수)
- `scripts/sim/stage2_poi.py:927`의 `system_prompt = active_stage2_system()`이 837행에서 고른 grounded 프롬프트(`SYSTEM_STAGE2`)를 **조건 없이 덮어쓴다.** 그래서 grounded 실행에서 옛 일반 프롬프트(`SYSTEM_S2`)가 쓰인다.
- 수정안: `if not grounded_experiment:` 아래로 옮긴다.
- 그 밖에 `d9813f42`는 다음과 같이 금연 실험 논리와 더 멀어졌다. 3.1을 적용할 때 함께 정리해야 한다.
  - Night2가 쌍당 3회만 시도하고, 재시작하면 예산을 잊고, 실패하면 날짜가 막힌다. 쌍 ID도 입력에서 그대로 받는다.
  - 1일자 월 초기화가 조건 없이 동작한다.
  - `execution_fingerprint`에 `LLM_BASE_URL`, `PYTHONHASHSEED` 등이 항상 들어간다.

### 3.5 플래그를 꺼도 동작이 달라지는 P012 변경 (결정 필요)
아래 항목은 Policy 노드가 없으면 실행되지 않지만, 정책 실험에서는 금연 실험과 다르게 동작한다.
- `dawn_context._format_policy_status`와 `mechanisms._PRINCIPLE`에서 "지출을 늘리지 않는다" 원칙 문장이 빠졌다. 모든 지갑·캐시백 정책 실행의 프롬프트가 바뀐다. P010 동결 문구도 바뀐다.
- `NIGHT_STATE_CYPHER`가 모든 State에 `income_today`(기본값 0)를 쓴다. 잔액에는 영향이 없다.
- 지원금 날짜를 문자열이 아니라 날짜로 비교한다. `visible_from_receipt` 필터도 생겼다(지급 일정이 없으면 원래와 같음).

`EXP_*` 기능(`consumption`의 계획 기준선·인정 채널, `income`, 월 초기화, `EXP_AGENT_DAY_MAX_ATTEMPTS`, `EXP_NO_SKIP`)은 **기본값이 꺼짐이어야 한다.** 꺼져 있을 때 금연 실험과 같은 동작인지는 확인했다. 단, 실행 환경에 엉뚱한 `EXP_*` 변수가 남아 있으면 실행 지문이 바뀐다.

---

## 4. 양쪽에 공통으로 있는 버그 — 실험 종료 후 동시에 수정

금연 실험의 OFF/ON 코드는 실험이 끝날 때까지 바꿀 수 없다. 그래서 아래 항목은 **양쪽을 같은 커밋으로 동시에** 고친다. 한쪽만 고치면 두 버전의 논리가 다시 갈라진다.

| # | 문제 | 위치 | 영향 |
|---|---|---|---|
| C1 | `retryable`을 계산만 하고 쓰지 않음. 코드 버그나 DB 오류 같은 결정적 오류도 6번 재시도한 뒤 `skipped`로 봉인하고 다음 날짜로 넘어감 | live `run_simulation.py:762`, P012 `:769`(`d9813f42` `:891`) | 한 조건에서만 나는 버그가 skip으로 숨어 OFF/ON 비교를 편향시킬 수 있음. 오류를 분류해 결정적 오류는 실행을 멈추게 해야 함 |
| C2 | 재시도 라운드가 같은 seed로 같은 요청을 반복(`begin_llm_scope`가 라운드마다 0부터, temperature 0). `llm_client`는 `max_retries=0`이라 통신 장애도 시도 예산을 씀. Stage1은 통신 오류 6번으로 하루 예산을 한 라운드에 소진 | live `run_simulation.py:429,446`, `llm_client.py:142`; P012 `:426,443` | 재시도가 사실상 무의미하고, 짧은 장애에도 skip이 생김. 라운드 번호를 seed 범위에 넣고, 통신 오류는 예산과 분리해 백오프 재시도해야 함 |
| C3 | 월 경계 처리: live는 `month_spent`를 초기화하지 않아 12월 1일 이후 프롬프트의 "월 누적"에 11월분이 섞임. P012는 1일에 초기화하지만, 1일이 skip되면 전월 누적이 넘어감 | live `plan_writer.py:519`, P012 `plan_writer.py:708`, `agent_day_store.py:82-95` | 이전 State 날짜의 월과 오늘의 월이 다르면 초기화하도록 통일하고, `save_skipped_day`에도 같은 규칙 적용 |
| C4 | 지원금이 수령일 당일에만 지급됨. 그날이 skip되면 영영 받지 못함 | live `run_simulation.py:336`, P012 `plan_writer.py:308` | 수령일 이후 처음 처리되는 날에 지급 |
| C5 | skip된 날이 이전 State의 날짜별 필드(`income_today`, `yesterday_satisfaction`)를 그대로 복사하고 `online_spent=null`. `main()`이 `summary.json`을 이번 실행 날짜만으로 덮어씀 | `agent_day_store.py:82-95`, `run_simulation.py` `main()` | 날짜별 필드는 비우거나 표시하고, summary는 날짜를 병합해 기록 |
| C6 | Night2 쌍 키 처리 방식이 다름(live는 입력과 다른 참가자 응답을 거부, P012 `d9813f42`는 입력에서 키를 가져옴) | `night_intent_llm` | 한 방식으로 통일 |
| C7 | Stage2 `finish=="length"` 재시도, 비grounded 무모델 대체 차단, 정책 실행용 `poi_restriction` | P012 쪽에만 있었다가 일부가 병합에서 빠짐 | 공통 코드로 가져와 양쪽 기본 동작으로 |

---

## 5. P012만의 결함 (리뷰 결과)

### 5.1 병합(`7bc8e45f`)에서 빠진 P012 쪽 고침
- **Stage2 무모델 대체:** grounded가 아니면 Stage2가 완전히 실패할 때 상위 5개 후보 중 무작위 매장을 조용히 채운다(`stage2_poi.py:1233-1243`). 과거 P012 실행에서 500명 중 106명이 이렇게 채워진 문제가 되살아났다(`d9813f42`에서 다시 넣음, 기본 꺼짐).
- **빠진 고침:** Stage2 출력 길이 초과 재시도, Night2 쌍 키·누락 집계, `mechanisms.poi_restriction`(이전 병합에서도 한 번 사라졌던 고침), `policy_preflight`의 `--require-db`·`--expect-no-policy`, `FALLBACK_LABEL`.
- **빠진 정책 기능:** P016 즉시할인(`instant_discount.py`), `mechanisms/generic.py`, `covid_no_distancing` 환경, 고정 명부 `fetch_roster`, 정산 기준 MPC, `paired_environment_fingerprint`, 실행 기록의 프롬프트 해시.
- **프롬프트 v40~v53, v5* 삭제:** `P012_v53_*.json`, `P013_v53_policy_*.json` 등이 v53을 전제로 하는데, `SIM_PROMPT_VARIANT=v53`을 주면 KeyError가 난다.
- **조용히 무시되는 설정:** 이전 P012/P013 실행 설정 `EXP_BALANCE_DAYS=39`, `EXP_SEED_SANGSAENG`, `EXP_SPLIT_ANCHOR`를 오류 없이 무시하고 다른 조건으로 돈다.
- **P012 HEAD 시뮬레이터 코드 자체의 문제:**
  - `EXP_*` 전체가 실행 지문에 들어가, 오류 메시지가 안내하는 `EXP_AGENT_DAY_MAX_ATTEMPTS` 복구 절차가 재개 거부로 막힌다(`experience_provenance.py:29`, 실제 확인).
  - `day_resume.py:50`이 시도 횟수 6을 하드코딩한다.
  - `consumption._plan_baseline`이 빈 dict를 먼저 공개해 스레드 경합이 생긴다(`:134-149`).
  - 설정 파일이 없으면 오류 없이 예전 동작으로 돌아간다.
  - `Stage1Exhausted`가 실제 호출 수와 무관하게 6회로 보고한다(`stage1_intent.py:694`).

### 5.2 채점표 교체 (`data/experiments/scoring_table.json`)
- HEAD에서 P016 지표(C1~C3, E1)와 모든 `empirical_audit` 블록, `P012.result_r2_v5` 등의 결과 블록이 빠졌다.
- `build_experiment_comparison`의 모든 보고서 모드가 `extra=['C1','C2','C3','E1']`로 멈춘다. 동결된 v53 매니페스트의 표 SHA와도 맞지 않는다.
- P012·P015 지표가 `sign_scoreboard`와 `error_budget`에서 조용히 빠진다.
- EM-4 순위 이름이 Category와 맞지 않아 0원으로 합산된다.
- P012-1 기준값 단위가 바뀐다(0.2082 로그포인트 → 20.82%).
- 보고서 단위 테스트 13건 이상이 실패한다.
- **어느 채점표가 기준인지 정하고, 스크립트·테스트·해시를 함께 맞춰야 한다.**

### 5.3 보고서 숫자 오류 (`scripts/report`)
| 심각도 | 문제 | 위치 |
|---|---|---|
| 높음 | K13 1인당 캐시백을 전체 시민으로 나눔. KDI는 수령자 기준이라 KDI 대비 약 0.23배가 0.07배로 보고됨 | `score_p012_two_arm.py:274-275` |
| 높음 | 사용처 규칙이 없는 지원금 정책(P010·P011·P013)의 "사용 가능 매장 오프라인 지출"이 항상 0(실행으로 재현) | `export_policy_daily_ledger.py:187-213`, `paired_coupon_effect.py:19` |
| 높음 | P015 HO-1에 정책 미적용(baseline) 실행 값(+16.1%, n=199)이 보고됨 | `core_indicator_table.py:85-106`, `steerability_map.py:95` |
| 중상 | 업종 지표 K3~K8에 캐시백 비대상 매장 지출이 포함됨(`kdi:X` → `kdiE:X`) | `score_p012_two_arm.py:249-254` |
| 중상 | 비율 지표를 정수로 반올림(0.48 → 0) | `score_to_block.py:217-222` |
| 중간 | 로그 계수를 %로 표시(P012-1 차이 −0.82%p가 실제는 −3.15%p) | `build_experiment_comparison.py` 약 500행, `build_rounds_page.py:36` |
| 중간 | EM-4 기준값이 원문(+3.6%p)과 다름(+3%p) | 채점표 EM-4 설명 |
| 중간 | 신뢰구간을 점추정과 다른 단위로 표시 | `build_results_overview.py:205` |
| 중간 | OFF/ON 쌍을 비교할 때 실행 지문·소스 지문을 검사하지 않음. 거리두기 쌍은 어느 쪽이 ON인지 확인하지 않음 | `score_multi_policy_proxies.py:138-160` |
| 중간 | `v50_verdict`의 DS-1 판정 기준이 사전 등록(1%p)과 다름(0.2) | `v50_verdict.py:374`, `steerability_map.py:60` |
| 중간 | 날짜 범위를 양 끝 이틀만 읽음 | `online_share_policy_response.py:42-44`, `threshold_response_shape.py:170-172` |
| 중간 | P012-4·P012-6을 다른 모집단(전체 vs 수령자)과 비교 | `_surface_comparison` |
| 중간 | 아주 작은 부호검정 p값이 0.0으로 반올림된 뒤 1로 처리됨 | `score_p012_two_arm.py:732,746` |
| 낮음~중간 | DS-6 순위 방향 반대, P013 주간을 목록 순서로 자름, 방향 일치 집계 부풀림, `recompute_plan_channel`의 arm을 경로 부분문자열로 거름 등 | 각 리뷰 원문 |

### 5.4 실험 실행·채점 (`tools`, `scripts/experiments`)
- **높음:** 이전 라운드 셸 스크립트에 `set -e`가 없다. 리셋, Day0 seed, 시뮬레이션의 종료 코드를 확인하지 않고 점수 파일을 "완료" 표시로 쓴다. 실패한 실행도 채점되고, 이후 재실행에서는 건너뛴다(`tools/run_p013_ruler.sh`, `run_p016.sh`, `run_p012_28d.sh`, `run_split_p012.sh`, `run_plan_p012.sh`, `run_scope_fact_round.sh`, `run_case_trend_round.sh`).
- **중간:**
  - P012 채점 전에 OFF/ON 일치를 확인하는 장치가 없다. `P012M_DAYS` 기본값이 스크립트마다 31과 7로 다르다.
  - OFF가 아직 보존되지 않은 ON 그래프를 덮어쓸 수 있다(`run_multi_v53_short_arm_20260928.sh:82-83`).
  - 재개 경로의 `SIM_RUN_ID`에 `$RUN_REVISION`이 빠져 재개가 항상 거부된다.
  - 서버 안에만 있는 사본을 `external_copy_verified`로 표시한다.
  - `analyze_policy_stance`가 공통 선행 설계를 처리하지 못하고, 수집 오류를 "모름"으로 센다.
  - probe 스크립트가 이전 결과로 판정할 수 있다.
  - `deploy_scoring_fixes.sh`가 실패해도 계속 진행한다.
- **코호트:**
  - **P012 명부(1,000·3,000명)가 성별과 연령을 독립으로 가정해 배분**됐다. 모든 연령대의 여성 비율이 0.518로 같다. 실제 결합표를 쓴 P013 명부는 60세 이상 0.547, 30대 0.499다. 명부를 다시 만들어야 한다(`freeze_demographic_matched_cohort.py:217-219`).
  - 동 배정이 실제 거주지가 아니라 ID 코드 기준이다(444명이 다름, `:205`).
  - `patch_split_anchor.py:47`이 패치를 심지 않고 "이미 있음"으로 끝난다.
  - P013 소진율 보정의 분모가 실제 지급액이 아니다.
  - 표본 크기 산식이 검정력 약 50% 기준이라 필요 규모를 약 2배 과소 추정한다.

### 5.5 배포 (`deploy/gpu_pool_kw26` 등)
- kw26 프록시의 로컬 동시 처리 상한에 `/health` 같은 비생성 요청까지 묶여, 최대 1,800초 대기한다. 원격이 죽으면 대기 요청이 몰린다.
- kw26 프록시가 포트 30000을 쓴다. 이 포트는 `llm_client`가 환경변수가 없을 때 자동으로 고르는 기본 포트라서, 설정 없이 시작한 프로세스가 의도치 않게 프록시를 타고 하드웨어가 섞일 수 있다.
- kw26 `keeper.sh`에 잠금과 `set -e`가 없다.
- 일부 런타임 패치 층이 해시 불일치에도 `RuntimeError`만 내서, sitecustomize 예외가 무시되고 패치가 일부만 적용된 채 실행될 수 있다(`SystemExit`로 바꿔야 함).
- 상위 호출의 제한시간이 백업 자체 제한시간보다 짧다.
- `run_shared`의 상태 파일 업로드가 같은 경로에 `--immutable`로 올라가는데 실패 시 내용을 다시 써서, 이후 업로드가 영구히 실패한다.
- Dec-2 덤프가 로컬에 없으면 post 단계를 재개할 수 없다.

### 5.6 저장소 위생 (공개 저장소)
- **Neo4j 고정 암호:** main에 17곳, 이 브랜치에서 13곳이 새로 생겼다(`tools/run_*.sh`, `experiments/NEXT_STEPS.md` 등). 실제 서버 암호를 바꾸고 스크립트는 환경변수만 쓰게 해야 한다. 이력에 남은 값은 암호를 바꾸는 것 말고는 무력화할 수 없다.
- A100 서버 주소·포트·사용자가 노트북과 스크립트에 하드코딩돼 있다.
- `data/neo4j_load/agents/agents_final.json.bak_orig`(22MB)가 `.gitignore` 규칙을 이름 차이로 피해 커밋됐다. `output/` 아래 파일 59개도 강제로 추가됐다.

---

## 6. 테스트와 병합 판단

| 대상 | 실패 | 오류 | 통과 |
|---|---|---|---|
| P012 HEAD `7bc8e45f` | 39 | 25 | 1,704 |
| 병합 전 P012 쪽 `557d7de2` | 4 | 1 | 1,668 |
| 병합 전 doinggyu 쪽 `af066e92` | 15 | 1 | 1,262 |

- HEAD에만 있는 새 실패 43건은 모두 병합 불일치 때문이다. 환경 문제가 아니다.
  - `test_propensity_mode`: 23건
  - `tests/unit/report`: 20건
  - `test_final_bundle_is_true`: 2건
- 커밋 메시지의 "새 실패 없음"은 `tests/unit/sim`에만 해당한다. 그것도 sim 테스트 파일 약 45개를 지운 결과다.
- origin/main과 add/add 충돌이 15건 있다. `P016.json`은 실제 사업기간(2020-07-30~11-30)으로 고친 브랜치 쪽을 써야 한다.
- **판단: 지금 상태로는 main 병합 불가.** 3절 이식, 5.1·5.2 복원 또는 폐기 기록, 5.3 숫자 수정, 충돌 해결이 먼저다.

---

## 7. 앞으로 두 버전을 같은 논리로 유지하는 규칙

1. **공통 경로는 하나의 소스에서 고친다.** 공통 경로는 dawn context, Stage1/Stage2, 근거·재시도 프롬프트, agent-day 재시도/skip/봉인, Night2, State/outbox 저장, 실행 지문, 재개 검사다. 한쪽에서 고치면 같은 커밋을 다른 쪽에 바로 cherry-pick하고, 이 문서 같은 대조표를 갱신한다.
2. **정책 전용 기능은 `EXP_*` 플래그 뒤에 두고 기본값은 꺼짐.** 꺼져 있을 때 공통 경로와 바이트 단위로 같은 동작인지 테스트로 고정한다.
3. **실행 중인 실험이 있으면 그 실험의 논리를 고정 기준으로 둔다.** 공통 버그 수정은 실험이 끝난 뒤 양쪽에 함께 넣는다.
4. **병합은 "한쪽 채택"으로 하지 않는다.** 병합 전후로 공통 모듈의 함수 단위 비교와 전체 단위 테스트를 돌리고, 테스트를 지워서 통과시키지 않는다.

## 부록 A. 대조에 쓴 자료
- **실행 코드:** Vast 서버 `/workspace/no-smoking-project-v22-perf-final/scripts/sim`과 `/workspace/no-smoking-runtime-hotfix-v22-*`를 읽기 전용으로 복사했다. 런타임 적재 순서대로 재구성해 함수 단위(AST)로 비교했다.
- **이식 확인:** `git diff 7e9cc723 679b17af -- scripts/sim`을 `7bc8e45f`에 적용하는 시험을 별도 사본에서 했다(충돌 없음).
- **금연 쪽 수정 브랜치:** `claude/no-smoking-ops-parity`
  - `1f12e18c` 운영 코드 수정
  - `9b5a665e` Night2 0 채움

## 부록 B. 확인하지 못한 것
- 서버 실행 환경에 엉뚱한 `EXP_*` 변수가 있는지.
- 같은 POI의 카테고리가 여러 개일 때 `head(collect(c))`가 임의로 하나를 고르는 문제가 실제 그래프에 있는지.
- `d9813f42`의 전체 변경 중 이 문서에 적은 항목 밖의 것은 줄 단위로 다시 보지 않았다.
- P012 실행 중 실제로 C1(결정적 오류가 skip으로 숨는 경우)이 일어났는지는 `attempts_*.jsonl`을 봐야 알 수 있다.
