# Claude 인수인계 — Vast 본 실험을 보존하며 Colab GPU로 가속

작성: 2026-09-30 KST. **마지막 실제 서버 확인: 2026-09-29 15:37:12 UTC = 2026-09-30 00:37:12 KST.** PID와 진행량은 당시 관측값이며 작업 시작 때 다시 조회한다.

## 1. 사용자 목표와 지켜야 할 원칙

사용자는 같은 날짜의 앞 에이전트는 Vast, 뒤 에이전트는 Colab에서 처리하고 전원 완료 후 다음 날짜로 넘어가는 가속 방안을 제안했다. Colab 상품은 “가장 비싼 걸로 구독할게”라고 답했으며, 이 작업부터 Claude에게 맡기겠다고 요청했다. **구독 완료·실제 GPU 배정은 아직 확인되지 않았다.**

- 노원·서초·송파 **1,154명**, 모델·기간·정책 조건과 완료 데이터를 보존한다. Day1부터 다시 실행하거나 새 baseline을 만들지 않는다. 이미 봉인된 `ok`/`skipped`도 임의 재계산하지 않는다.
- 기존 계산을 계속 진행하면서 별도로 구현·검증하고, **다음 안전한 날짜 경계에 최대한 빨리 적용**한다. 다음 14일짜리 프로세스까지 미루지 않는다.
- 현재 Python 프로세스에는 분산 기능을 새로 읽는 hot reload가 없다. 파일만 교체하고 적용됐다고 보고하지 않는다. 필요하면 백업과 작업 배출 후 계산 프로세스만 통제된 인계를 한다. Vast 인스턴스·기존 모델을 불필요하게 재시작하지 않는다.
- 개별 agent-day/야간 pair는 최초 시도+최대5회 재시도 규칙을 보존하고 소진 시 해당 대상만 `skipped`로 남긴다. Stage 내부 검증 호출, 외부 agent-day 시도, 통신 재전송을 구분하고 연결 교체로 예산을 초기화하지 않는다.
- 일상적 개별 실패로 전체 실험이나 Vast를 자동 중지하지 않는다. 계산이 중단되면 원인을 확인하고 기존 데이터에서 안전하게 재개한다. GPU를 아무 작업 없이 방치하지 않는다. 정상적인 백업을 위한 작업 배출은 계산 실패와 구별한다.
- 무결성 검사를 무력화하거나 실패한 응답을 임의로 고쳐 성공으로 만들지 않는다. 봉인된 결과를 지우거나 과거 checkpoint로 되감아 이미 완료한 계산을 잃지 않는다.
- 새 Vast 임대·모델/연구 조건 변경·실제 구매는 이번 인수인계의 실행 지시가 아니다. 필요한 Colab 로그인/GPU 할당만 사용자에게 구체적으로 요청한다.

## 2. 작업 위치와 읽을 순서

```text
C:\Users\srdyh\OneDrive\사진\바탕 화면\agent_simulate\kw_26_final_project\.claude\worktrees\silly-gagarin-0b181a
```

브랜치 `No_SmokingZone_EXP`. 먼저 `git branch --show-current`, `git status --short`로 기존 변경을 확인한다.

1. 루트 `AGENTS.md`
2. 이 문서
3. `docs/HANDOFF_NO_SMOKING_ZONE.md`의 **맨 위 최신 실험 기록**
4. `deploy/vast/local/ACTIVE_DEPLOYMENT.json`
5. `deploy/colab/README.md`
6. `deploy/vast/runtime_hotfix_v22_night_progress/` 및 이전 runtime chain

`deploy/vast/local/`는 Git 제외다. 새 clone만 받으면 ACTIVE·검증 영수증·복구 사본이 없을 수 있다. 이 경우 원래 작업 폴더의 자료를 확인하며 baseline을 다시 만들지 않는다. 별도 연구비 정산 기록은 서버 실행 지시가 아니다.

기존 미커밋 변경은 실제 장애 대응 결과이므로 보존한다. 작성 시 변경/신규 파일은 `scripts/sim/{grounded_schema,interview_evidence,night_intent_llm,run_simulation,stage1_intent,stage2_poi,night_completion,night_recovery}.py`, `deploy/vast/runtime_hotfix_v22_night_progress/`, `runtime_hotfix_v22_retry_contract/`, `deploy/colab/`, 관련 tests 및 HANDOFF였다. 신규 `tmp/`도 있었으며 이번 작업에서 내용을 변경하지 않았다. reset/clean 또는 원격 브랜치 덮어쓰기를 하지 않는다.

## 3. 실제 실행 상태와 접속

**00:37 KST 읽기 전용 확인:**

| 항목 | 관측값 |
|---|---|
| Vast | `52220534`, L40S 1장 |
| pipeline | `running / shared-pre` |
| supervisor / simulator | `21477 / 21675` |
| 서버 복구기 / 모델 | `21476 / 74` |
| 모델 health | HTTP 200 |
| 별도 backup 프로세스 | 당시 없음 |
| 디스크 여유 | `/workspace` 8.260GiB |
| 완료 날짜 | Day1~Day6, 날짜별 1,154개 고유 metrics·완료 백업 마커 |
| 현재 날짜 | **Day7 `2017-11-25`: 198/1,154, ok198/skipped0** |

이번 조회에서 Day1~Day7의 완결된 metrics JSONL 행 integrity seal과 날짜별 고유 Agent ID를 확인했다. **새 Neo4j State/outbox 전수 대조, Drive 원격 재조회, 새 복원 시험은 하지 않았다.** 진단 JSON은 `deploy/vast/local/handoff_claude_colab_20260930.json`이다.

PowerShell SSH 예시:

```powershell
$vastKeyPath = Join-Path $env:USERPROFILE '.ssh/no_smoking_vast'
ssh -i $vastKeyPath -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=12 -p 45025 root@95.3.33.46 'date -u'
```

기존 로컬 개인키를 사용한다. 키 본문·API 키·DB 암호·rclone 토큰을 채팅/문서/노트북 출력/Git에 넣지 않는다. 접속 거부만으로 서버 중지를 단정하지 말고 필요하면 Vast 콘솔의 현재 포트를 확인한다.

### 활성 경로

| 용도 | 경로 |
|---|---|
| frozen project | `/workspace/no-smoking-project-v22-perf-final` |
| project Python | `/workspace/no-smoking-project-v22-perf-final/.venv-no-smoking/bin/python` |
| 원본 archive | `/workspace/source-exaone-sglang-3gu-1154-v22-perf-final.tar.gz` |
| config | `/workspace/expedited-full-1154-v22-day2-config.json` |
| prefix | `integration-main-v22-1154` |
| pipeline | `/workspace/no-smoking-results/integration-main-v22-1154-pipeline.json` |
| 현재 run | `/workspace/no-smoking-results/integration-main-v22-1154-shared-pre` |
| 실행 로그 | `/workspace/no-smoking-results/integration-main-v22-1154-shared-pre.run.log` |
| 복구기 로그 | `/workspace/no-smoking-results/integration-main-v22-1154-recovery-controller.log` |
| 복구기 원장 | `/workspace/no-smoking-results/integration-main-v22-1154-resume-control.jsonl` |
| 활성 extension | `/workspace/no-smoking-runtime-hotfix-v22-night-progress` |
| startup | `/workspace/onstart.sh` |
| 현재 Neo4j OFF | `/workspace/no-smoking-neo4j-integration-main-v22-1154-pre/off`, Bolt `127.0.0.1:17791` |
| 모델 API | `http://127.0.0.1:8000/v1`, health `/health` |

**내부 run_id는 `/workspace/no-smoking-results/integration-main-v19-1154-shared-pre`를 승계했다.** 폴더의 v22와 다르다고 수정하거나 기존 seal을 다시 쓰지 않는다. 후속 run은 `integration-main-v22-1154-post-off`/`post-on`이다.

### 활성 SHA256

| 산출물 | SHA256 |
|---|---|
| frozen v22 archive | `e6ab8021766613426a615de4f1322b2d315864b857080d4fe047c5e2e4277f7f` |
| night-progress manifest | `19f2a2eaf2587b8ee85da0e5f14278877a86fd4de543346bd31bb325383455eb` |
| night-progress sitecustomize | `b22ba7cc53837a3c9d0194c50a3e4f06f84319f95eeb870add8f588cf976db92` |
| night-progress resume_runtime.py | `c798e9f8d0ff15cad7178639f846ffe3c4c20925833f4fc1039004c1c85701d6` |
| onstart.sh | `48e96c24682684fcc984e7387f301d45443f41a06dc2130b49acdd40763f23ed` |
| 이전 retry-contract manifest | `7f0c6f2d4ed7b956b1371898b3736db511150625ae71db1f6044819520166b99` |

night-progress 3파일과 onstart 해시는 이번 SSH에서 재확인했다. 원본 archive/이전 체인 전체를 이번에 다시 해시한 것은 아니다. sitecustomize는 이전 retry-contract → grounding/State/night 패치에 의존한다. 상세 경로·해시는 `resume_runtime.py`의 `validate()`와 sitecustomize chain을 읽는다. 새 폴더 하나를 복사한다고 전체 체인이 대체되지 않는다.

## 4. 동일하게 유지할 모델·설계

- **현재 모델:** `LGAI-EXAONE/EXAONE-4.5-33B-AWQ`. 과거 대화의 DASHQ INT2/INT4가 아니다.
- revision: `31e6a965d0661bbe4a8b895e22a77f8271772ba0`
- 실제 관측 패키지: SGLang `0.0.0.dev11420+g6757c9f90`, torch `2.9.1`, transformers `5.8.0`
- 모델 실행기: `/workspace/venv_sgl/bin/python -m sglang.launch_server`
- context16384, TP1, max-running-requests16, mem-fraction-static0.88, attention `triton`, grammar `outlines`, 서버 seed42. 현재 workers8, 호출 timeout600초.
- 요청별 model/temperature/seed/max_tokens/schema/프롬프트/토크나이저도 맞춘다. 같은 서버 seed만으로 충분하지 않다. 다른 GPU/커널의 비트 단위 동일 출력은 보장되지 않으므로 실행 출처를 기록한다.
- 공통 선행 `2017-11-19~12-02`14일 → 공통 최종 그래프에서 `2017-12-03~12-16` OFF14일·ON14일 → 감사·점수·백업. 현재 감독자는 OFF/ON을 순차 실행한다. 총 `1,154 × 42 = 48,468 agent-days`; 사후 인터뷰는 별도다.

## 5. 백업·복구 조건과 남은 문제

최신 검증 기록은 **Day6 완료 복구점**, `2026-09-29 14:03:47 UTC` 업로더 MD5 검증이다. 이번 receipt도 이 복구점을 가리켰다(00:37 KST 기준 약1시간33분 경과).

```text
no_smoking_drive:No_SmokingZone_EXP_Backups/runs/integration-main-v22-1154-shared-pre/checkpoints/2017-11-24/20260929T132007Z-9fd553c5
checkpoint SHA256: c407aee6acc86606d9c96c680af9102d2a99620b0d9c1f564e13008ac935e687
graph SHA256: 699d0ce52866f89cb048a470123b64809232f6c8258f3a125642def8be8c3d94
day archive SHA256: 8f21b7cf1bb84dca59c7d3d0d3d5be19ef29ef318f25e8ad0bf365a95c2b1a63
```

- 당시 업로더4파일 MD5 검증. 14:11 UTC 별도 원격 목록에서4파일 SHA256/크기와 로컬 checkpoint/committed2파일 MD5 일치. 직후 committed 본문 재다운로드는 공유 OAuth403으로 실패했다. 본문 재다운로드/새 DB 복원까지 성공한 것으로 쓰지 않는다.
- `recoverable_backup.json`, `backup_completed_YYYY-MM-DD.json`, 원격 `committed.json`을 확인한다. 단순 존재나 검증 시각 갱신을 새 그래프 생성으로 취급하지 않는다.
- 날짜 완료 및 마지막 검증 백업10시간 뒤 quiescent hook이 작업자를 배출해 백업한다. 완전한 새 복구점이12시간을 넘으면 원인을 진단한다. 실행 중 DB에 offline dump를 중복 실행하지 않는다.
- **일일 run archive에는 이전 날짜가 전부 들어 있지 않다.** parent checkpoints 또는 검증된 이전 full-run도 복원에 필요하다. graph만 보존하고 metrics/evidence/attempts/runtime 계보를 빠뜨리지 않는다.
- 외부 로컬 사본: `deploy/vast/local/recovery-20260929-day5-complete`, `deploy/vast/local/recovery-20260928-day5`. Day6 사본으로 혼동하지 않는다.
- 공유 Drive OAuth `403 rateLimitExceeded`는 간헐적으로 지속된다. 전용 OAuth 전환은 미완료다. 기존 인증 설정을 임의로 덮어쓰거나 Drive를 작업 큐로 쓰지 않는다.
- Vast 여유 공간 약8.26GiB. 두 번째 모델/대형 DB 사본을 추가하기 전에 공간을 계산하고 유일한 원본·미검증 결과를 지우지 않는다.

과거 skipped State.id 유일성 충돌, 근거 번호 앞자리0 형식, 야간1쌍 실패로 전체를 막던 문제는 현재 패치 체인으로 수정됐다. 유효 응답은 재사용하고 소진 pair는 증거와 함께 skipped로 남긴다. **Day2/Day5 일회성 복구 스크립트를 다시 실행하지 않는다.** 특히 `switch_once.py`, `continue_handoff.py`, `resume_external_snapshot.py`는 현재 일반 재개 도구가 아니다.

## 6. 자동 복구기와 작업자 충돌

서버 `resume_runtime.py`는15초마다 감독/시뮬레이터/백업을 검사하고 종료된 계산을 기존 데이터에서 재개한다. lifetime flock과 writer 확인으로 중복 실행을 막으며, 재개 오류는30~300초 간격으로 재시도한다. 미완료 Vast 자동 stop은 억제되고 전체 완료·점수·Drive 검증 뒤 stop은 유지된다.

살아 있지만 응답하지 않는 모델은 자동 강제 종료하지 않는다. 실제 재부팅/운영 중 강제 crash 시험은 미실행이다. 모든 장애가 자동 해결된다고 가정하지 않는다.

계산 프로세스를 교체할 때 복구기가 기존 버전을 다시 올릴 수 있다. **검증된 범용 유지보수 모드가 있다고 가정하지 말고**, 기존 복구기와 새 worker/queue/backup의 소유권·인계 순서·잠금·복귀를 함께 설계한다.

Codex `vast-drive` **6시간 감시도 남아 있다.** 이번 문서 작성만으로 자동화나 복구 정책을 변경하지 않았다. Claude가 배포할 때 HANDOFF에 작업 소유자·유지보수 구간을 기록하고 Codex 감시의 변경 작업과 조율한다. 문서를 읽었다고 감시가 자동 정지된 것은 아니다. **실제 서버 변경은 한 작업자만 수행한다.** PC가 꺼져도 Vast 실행·백업·복구기는 지속되며 Codex 감시는 PC와 앱이 켜져야 작동한다.

## 7. Colab 준비물과 권장 설계 — 구현 전

준비된 것은 `deploy/colab/README.md`와 조회 전용 `no_smoking_gpu_check.ipynb`다. JSON/코드 컴파일은 통과했지만 **Colab 실제 실행·모델 적재·분산 코드·벤치마크는 미실행**이다.

- notebook SHA256: `db0367a09457cdc91d69281b41fac5cdcd43c0059a3873070ce181c7e40d1dc2`
- README SHA256: `88c96590bf3b22cb7943ec36b18b19cb6eb32ec44cac16e3943d73bfc0841431`

[공식 Colab FAQ](https://research.google.com/colaboratory/faq.html)는 유료도 GPU/가용성이 변하고 충분한 컴퓨팅 단위가 있는 Pro+ 연속 실행이 최대24시간이라고 안내한다. 실제 배포 시 현재 제한·잔액·할당 GPU를 확인한다. 무료 제한/자원 할당을 우회하지 않는다. 유료 구독 자체가 특정 GPU나 주 단위 무중단 실행을 보장하지 않는다.

### 제안 구조

1. **Vast에 단일 DB·원장·감독자.** Colab은 동일 모델 추론을 받아 응답을 반환한다. DB/백업 자격증명을 Colab에 복제하지 않는다.
2. 중앙 배정기는 `(run, arm, day, aid 또는 pair, stage, attempt, input hash)`별 소유권을 관리한다. 앞/뒤 정렬만으로 중복을 막을 수 없다. 고정 반분보다 각 GPU 실측 처리량에 맞게 미할당 작업을 배분한다.
3. 검증·State/outbox·metrics 봉인·시도 예산은 Vast가 책임진다. 날짜가 끝날 때까지 Colab에만 결과를 모으지 않고 작은 단위로 보존한다.
4. 연결 종료/timeout을 처리하는 heartbeat·임대·소유권 세대 번호를 둔다. 미완료 요청은 Vast로 재배정하되 늦은/중복 응답을 두 번 저장하지 않는다. 이미 진행 중인 Vast 작업을 중복 배정하지 않는다.
5. 하루1,154명 ok/skipped 확인 → 야간 성공+skipped accounting → 백업 → 다음 날짜. Colab이 사라져도 Vast가 남은 작업을 끝낼 수 있어야 한다. 야간 추론도 가속할 수 있으나 날짜 경계를 유지한다.
6. 인증·암호화된 전송 채널을 사용한다. Colab pull 방식/외부 추론 endpoint 등 실제 환경에 맞는 방법을 선택한다. 구체 네트워크 경로는 아직 확인·구현하지 않았다.

이는 앞선 Codex의 **제안**이며 확정·배포된 구조가 아니다. 더 단순한 구현으로 데이터 보존·중복 방지·장애 복귀를 충족할 수 있으면 근거를 제시해 선택한다. 독립 DB 두 개를 날짜 말에 합치는 방식은 현재 원장·트랜잭션·계보와 충돌한다.

### 확인할 코드

| 코드 | 핵심 |
|---|---|
| `scripts/sim/run_simulation.py` | `process_one`, `run_day`, 병렬 workers, 일일/야간 경계·백업 배출 |
| `scripts/sim/dawn_context.py` | 이전 날짜 State/Memory 입력과 날짜 일관성 |
| `scripts/sim/agent_day_store.py` | completed 검사·Agent 잠금·State/outbox 원자성·fingerprint |
| `scripts/sim/llm_client.py` | singleton client·endpoint·요청 설정·응답 추적 |
| `scripts/sim/stage1_intent.py`, `stage2_poi.py` | 입력/후보 검증·오류 피드백 재시도 |
| `scripts/sim/night_recovery.py`, `night_completion.py`, `night_intent_llm.py` | 성공 캐시·pair 예산·skipped accounting·저장 |
| `scripts/sim/experience_provenance.py` | source/execution fingerprint, 설정·계보 |
| `deploy/vast/run_shared.py` | 선행 → OFF → ON, 분기·감사·점수·백업 |
| `deploy/vast/runtime_hotfix_v22_night_progress/` | 실제 활성 함수/wrapper/복구기와 이전 patch chain |

환경 변수만 바꿔서는 실행 중 singleton client가 갱신되지 않는다. 두 번째 모델만 띄운다고 workers8이 자동 분산되지 않는다. source/execution fingerprint를 비활성화하거나 새 코드를 과거 코드로 위장하지 않는다. 새 runtime hash·적용 날짜·worker 환경·입력/출력 hash를 추적한다.

## 8. 수행 순서와 완료 기준

1. 최신 실제 실행·데이터·백업·디스크를 조회하고 기존 정상 계산을 유지한다.
2. Colab 구독/로그인·GPU는 사용자에게 확인하고 준비된 노트북 또는 `!nvidia-smi`로 확인한다. GPU를 기다리는 동안 가능한 큐/검증/인계 구현은 진행한다.
3. 동일 revision/설정 모델을 Colab에 적재하고 본 결과에 합치지 않는 소량 시험으로 메모리·검증 통과·처리량을 측정한다. 다른 모델로 조용히 대체하지 않는다. 현재 dev SGLang과 설치 버전이 다르면 실제 호환성을 확인한다.
4. 격리 검사: 중복/늦은 응답, 연결 종료·재접속, 예산 보존, 마지막 에이전트의 날짜 경계, 야간 mixed/all-skipped, DB commit 뒤 파일 기록 전 중단 복구, 백업 중 원격 요청 배출, 자동 복구기 인계 충돌.
5. 적용 직전 기존 완료 파일 해시/State/outbox와 최신 그래프+run 백업을 검증한다. 부모 archive·실제 runtime까지 복원 가능해야 한다. 운영 DB를 초기화하지 않는다.
6. 다음 안전한 날짜 경계에서 한 번 인계하고, 문제가 있으면 **현재 결과를 유지한 Vast 단독 경로**로 복귀하도록 한다. 이미 봉인한 날짜·대상은 재실행하지 않는다.
7. 두 GPU 실제 작업량, 중복0, seal/State/outbox 일치, 오류율, Colab 종료 시 Vast 이어받기, 다음 날짜 진전·Drive 백업을 확인한다. 강제 장애 시험은 격리 테스트에서 먼저 수행한다.
8. 실측 ETA·비용을 갱신하고 새 경로/해시/활성 시각·검사 결과·미완료를 HANDOFF/ACTIVE에 기록한다. 서버 복구기/감시가 새 런타임을 인식하도록 함께 인계한다.

보고는 **설계 / 코드 구현 / 오프라인 검사 / Colab 실제 모델 시험 / 본 실험 적용 / 가속 실측 / 복구·백업 확인**을 구분한다. 사용자는 계획만 반복하는 것을 원하지 않는다. 가능한 구현과 검증을 수행하고 사용자 로그인이 필요한 지점만 요청한다.

최근 Day6 실측은 계산9.904시간+백업 등0.728시간=10.632시간/일. 동급 지속 처리량 GPU 추가의 이상적 계산은 `9.904/2+0.728=5.680시간/일`이다. **실측 가속률·확정 납기가 아니다.** 연결 수명·재적재·통신·병목을 포함해 다시 측정한다.

## 9. 인수인계 작성 작업의 범위

브랜치/기존 변경·HANDOFF/ACTIVE·Colab 준비물·복구 코드를 읽었고, 위 SSH 조회와 전용 문서/프롬프트 작성을 수행했다. 서버 코드 변경·프로세스 재시작·새 모델 호출·DB 쓰기·새 dump·Drive 재조회·자동화 변경·구매는 하지 않았다. 기존 미커밋 파일을 보존했다.

전달 프롬프트: `docs/PROMPT_CLAUDE_COLAB.md`.
