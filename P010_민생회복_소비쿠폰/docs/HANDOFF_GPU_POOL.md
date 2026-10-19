# 인수인계 — GPU 추론 풀(Vast + Colab) 준비 상태

작성: 2026-09-30 01:40 KST (Claude, 워크트리 `colab-gpu-integration-999c0b`, 브랜치 `claude/colab-gpu-integration-999c0b`, 미커밋).
본 실험 인수인계는 `No_SmokingZone_EXP` 브랜치의 `docs/HANDOFF_NO_SMOKING_ZONE.md`, `docs/HANDOFF_CLAUDE_COLAB.md`. 이 세션은 그 워크트리(`silly-gagarin-0b181a`)에 쓸 수 없어 이 파일에 기록했다. **ACTIVE_DEPLOYMENT.json은 변경하지 않았다(서버 배포 없음).**

## 1. 사용자 결정 (2026-09-30)
- Colab Pro+ 결제 완료(사용자 진술). GPU 배정은 아직 미확인.
- 구조: **Vast의 simulator 1개가 DB·원장·검증·시도 예산·백업을 모두 유지**하고, Vast의 추론 프록시가 LLM 요청 단위로 Vast GPU와 Colab GPU에 나눈다. 홀짝 고정 분배는 채택하지 않았다(느린 쪽 대기·Colab 종료 시 정지 위험).
- 두 번째 Vast 인스턴스: 사용자가 대여를 원했으나 **로컬 Vast API 키가 2FA 세션 없음으로 401** → 콘솔에서 사용자가 직접 대여해야 한다. 후보: offer `38575675`(머신 39567, L40S, $0.801/h, verified, 드라이버 570, 최대 170일; 디스크 표시 90GB라 200GB 가능 여부 확인 필요). 대여되면 같은 프록시에 원격 백엔드로 추가한다.
- 사용자 취침 중 요구: **실험이 멈추거나 Day1부터 다시 시작하면 안 된다.** → 이 세션은 서버 프로세스·설정·파일을 하나도 바꾸지 않았다.

## 2. 실제 서버 관측 (읽기 전용 SSH)
| 시각(UTC) | 관측 |
|---|---|
| 09-29 16:03 | 모델 PID 74, 복구기 21476, supervisor 21477, simulator 21675, health 200, `/workspace` 여유 8.3~9GB |
| 09-29 16:19 | Day7 `2017-11-25` metrics 284행, 약 123명/시간 → Day7 완료 예상 ≈ 09-29 23:30 UTC(09-30 08:30 KST) |
| 최근 검증 복구점 | Day6 complete, checkpoint `c407aee6…`, graph `699d0ce5…`, verified 14:03:47 UTC |

- 병목: SGLang scheduler 프로세스 CPU 100%(단일 코어), GPU 사용률 ~40%, 8요청 동시 생성 합계 ~78 tok/s(요청당 ~9.7). outlines JSON 문법 처리로 추정. Stage1 1회 LLM 145초(입력 6,755·출력 1,132 토큰) 예시 확인.
- run log의 Neo4j `defunct connection` 경고 12건(진행은 계속). 원인 미진단.
- 날짜 백업 임시 사용량: dump ≈0.47GB + run archive ≈0.2GB, 업로드 검증 후 dump 삭제 → 오늘 밤 공간 충분.

## 3. 코드 동일성 확인 (서버 해시 vs 로컬 `silly-gagarin-0b181a`)
- frozen archive SHA `e6ab8021…` 재확인. 실행 PYTHONPATH는 패치 7층: night-progress → retry-contract → grounding → retry → night2-recovery → night2 → skip → frozen.
- frozen `.py` 405개 중 397 동일, 8개 상이(`run_simulation, agent_day_store, stage1_intent, stage2_poi, grounded_schema, interview_evidence, night_intent_llm, watch_pipeline`).
- 서버 패치 파일 53개: 로컬 존재분 33개 동일, `skip/onstart.sh` 상이(활성 `/workspace/onstart.sh`는 로컬 night-progress 사본과 동일), 19개 로컬 부재(grounding·retry의 stage1/stage2·canonical_evidence_ref, night2 복구 스크립트, skip/watch_pipeline, 로그 등).
- **결론: 로컬 코드는 활성 코드와 같지 않다. 새 GPU 환경은 로컬이 아니라 서버 사본/해시로 재현한다.**

## 4. 반드시 지킬 제약 (코드에서 확인)
- `no_smoking_zone.py run`의 resume은 `workers`, `server_config`, `engine_settings`, `code_sha256` 등 변경 시 거부(`Resume cannot change the frozen execution`). **workers=8 유지.** 두 GPU는 이 8개 동시 요청을 나눠 받는다(스케줄러 CPU 병목 완화가 가속 근거 — 실측 전).
- `engine_settings`는 `SIM_ EXP_ POLICY_ CONSUMPTION_ STAGE1_ STAGE2_ NIGHT_ MOBILITY_ HUB_` 접두만 기록 → `SGLANG_BASE_URL` 변경은 fingerprint/identity에 들어가지 않는다. `execution_fingerprint()`도 base URL·workers 미포함.
- `run_shared.py:88`이 `SGLANG_BASE_URL=http://127.0.0.1:8000/v1`을 **하드코딩**(frozen·해시 검증 파일). simulator 21675 환경에서도 확인. 따라서 프록시(127.0.0.1:30000)를 띄워도 현재 실험은 자동으로 타지 않는다(안전). 적용하려면 이전 패치와 같은 방식의 **새 최상위 런타임 층**(sitecustomize가 자식 프로세스에서 `SGLANG_BASE_URL`을 프록시로 바꾸고 다음 층 sitecustomize를 이어 실행) + 그 층을 PYTHONPATH에 넣는 **복구기 새 버전**이 필요하다. 모델·인스턴스는 재시작하지 않는다.
- **다가오는 위험:** `run_shared.py`의 post pair 생성이 `NO_SMOKING_NEO4J_MIN_FREE_GB='10'`을 요구. 현재 여유 ≈9GB → 공통 선행 종료(Day14 `2017-12-02`, 현재 속도로 약 3~4일 뒤) 시 OFF/ON Neo4j 복원이 막힐 수 있다. 후보: `/root/.cache/pip` 5.1GB(캐시), 은퇴 run DB `no-smoking-neo4j-main-v15`/`-v17` 각 9.2GB(백업·은퇴 여부 확인 후). **삭제는 아직 하지 않았다.**

## 5. 이번에 만든 것 (코드 구현 + 오프라인 검사만; 서버 미배포)
| 파일 | SHA256 | 내용 |
|---|---|---|
| `deploy/gpu_pool/gpu_pool_proxy.py` | `21032228a2af72374bcb9b38a58f3a4dc616bb895c9535c6aa69c4ceedef5d44` | 표준 라이브러리 프록시. 원격은 `/get_server_info` 생성 식별 필드가 로컬과 전부 같을 때만 투입, health 상실·거부·5xx·끊김 시 같은 바이트를 로컬로 1회 재전송(시도 예산 소모 없음), 늦은 원격 응답 폐기·기록, 요청/응답 SHA·처리 백엔드 JSONL 기록, `/pool/status`, `/pool/drain` |
| `tests/unit/deploy/test_gpu_pool_proxy.py` | `5274e13ff38a1d8ee4b4a29ff29cbfd258f5b17ff7cc055b03516a01b4f74a18` | 가짜 SGLang 10개 시험: 바이트 보존·식별 불일치 차단·분산·요청 중 원격 사망·500/리셋·거부 후 재검증 재투입·drain·비생성 경로 로컬·로컬 실패 은폐 금지. **Windows Python 3.13에서 3회 연속 10 passed.** Linux/3.11 미실행 |
| `deploy/gpu_pool/vast_venv_sgl_freeze.txt` | `db4b8ae34250a6c69ca59990abf43cf2db44ee4cde25e604adfc9691ab99f5ef` | Vast `venv_sgl` 전체 pip freeze(229줄; sglang = `lkm2835/sglang@6757c9f9`, torch 2.9.1, outlines 0.1.11, Python 3.11) |
| `deploy/gpu_pool/build_colab_notebook.py` | `ab0348ab…`(생성기) | 노트북 생성 |
| `deploy/gpu_pool/colab_sglang_worker.ipynb` | `692f891dc98a9adf58be5943ccc64a0765573a24bf7d5644a57660aff3b720c7` | GPU≥40GB 확인 → uv Python3.11 venv에 freeze 설치 → 가중치 1회 다운로드 → **한 GPU에 동일 서버 N개**(VRAM 30GB당 1개, 최대 4, 포트 8000+i, `mem_fraction_static=0.88/N`) → 서버별 식별 확인 → 소량 JSON 성능 시험 → 역방향 터널(Vast 127.0.0.1:18001+i) → 상태 루프. 코드 셀 구문 검사만 통과, **Colab 실제 실행 안 함** |

- Colab 전용 SSH 키: 로컬 `~/.ssh/no_smoking_gpu_pool`(지문 `SHA256:tbZq7Ihdhc7Y7tKFVGQfJ9nSXSmuZru6XjVMyNzlCcQ`). **Vast에는 아직 등록하지 않았다.** 등록 시 `authorized_keys`에 `restrict,port-forwarding,permitlisten="127.0.0.1:18001",command="/usr/sbin/nologin"`을 붙여 셸 불가·해당 포트만 허용(Vast OpenSSH 8.9 — permitlisten 지원).
- Colab 약관(FAQ 확인): SSH·분산 워커 제한은 무료 등급만. "원격 프록시 연결"은 전 등급 금지 — 본 구조는 Colab이 자기 모델 서버를 우리 서버로 내보내는 것이라 해당하지 않는다고 판단했지만 남은 위험으로 기록.

## 6. 미완료와 다음 순서
1. (사용자) Colab A100/H100 런타임 연결, Secrets 3개 등록, 노트북 실행. 설치·모델 적재·성능 셀 결과 확인.
2. Vast에 제한 키 등록(위 옵션), 프록시를 **포트 30000**에 keeper 루프로 기동(현재 실험에는 영향 없음 — 하드코딩 8000). Colab 터널 연결 후 `/pool/status`로 admission 확인, 프록시 경유 소량 요청으로 Colab 실측.
3. 런타임 층 `gpu-pool` 구현·해시 manifest·복구기 새 버전(PYTHONPATH 앞에 추가) + 오프라인 시험. 시도 기록이 in-flight 중단 시 어떻게 남는지 확인(`retry_contract`, `night_recovery`).
4. Codex `vast-drive` 감시·서버 15초 복구기와 소유권 조율 후, **다음 날짜 완료+백업 검증 직후** 복구기→supervisor/simulator만 통제 교체. 이미 봉인된 날짜 재실행 금지. 문제 시 기존 복구기(night-progress `c798e9f8…`)로 복귀 = Vast 단독.
5. 디스크 10GB 조건 해소(4절) — 공통 선행 종료 전 필수.
6. 실측: 백엔드별 처리량·오류율·fallback 수, Colab 종료 시 Vast 복귀, 다음 날짜 진전·Drive 백업. 결과를 `ACTIVE_DEPLOYMENT.json`·본 HANDOFF에 반영.

## 7. 동시 작업자 주의
`No_SmokingZone_EXP` 브랜치에 09-30 00:56~00:57 KST 커밋 `679b17af`, `17d21f64`(기존 미커밋 retry/Colab 준비물 커밋)가 이 세션 도중 생겼고, 01:1x에 HANDOFF가 다시 수정되고 있었다. 다른 에이전트(Codex 등)가 활동 중일 수 있으니 서버 변경 전 확인한다.

## 8. G4 사용 계획 (2026-09-30 사용자: "G4, VRAM 약 90GB")
- G4 = RTX PRO 6000 Blackwell 96GB로 추정(실제 배정 시 노트북 1번 셀로 확인). 병목이 SGLang 스케줄러 단일 CPU 코어이므로 한 GPU에 같은 서버를 여러 개(기본 3개 = 96GB/30GB) 띄워 스케줄러를 늘린다. `mem_fraction_static`·포트는 identity 필드가 아니며 프록시 admission 대상은 그대로다.
- 상한: frozen `workers=8` → 전체 동시 LLM 요청 최대 8. Vast 1개 + Colab 3개 = 스케줄러 4개면 서버당 약 2요청. 서버를 더 늘려도 이득이 거의 없다.
- **미확인 위험:** Blackwell(sm_120)에서 Vast와 같은 `sglang-kernel 0.4.1`/torch 2.9.1 cu128/AWQ 커널이 동작하는지. 동작하지 않으면 다른 버전이 필요하고 이는 `version` identity 불일치 → 사용자 결정 필요(조용히 대체 금지).
- Vast 제한 키 등록 시 `permitlisten`을 사용할 포트 수만큼 나열: `permitlisten="127.0.0.1:18001",permitlisten="127.0.0.1:18002",permitlisten="127.0.0.1:18003"`. 프록시 remotes: `colab-0..2` → `http://127.0.0.1:18001..18003`.

## 9. 본 실험 적용 완료 (2026-09-30 11:05~11:09 UTC) — 위 5~6절의 "미배포" 상태를 대체
- **사용자 승인:** "예약해"(날짜 경계 자동 교체). 실행 시 경계는 이미 지나 있었다(Day8 `2017-11-26` 완료 백업 검증 10:03:49 UTC, checkpoint `e4bfc9a1faac0aac3aeef50b7efa4c5b0cfcc93c4da5c46fb3dcfe755e484699`; Day9 `2017-11-27` 126행 진행 중) → 스크립트가 즉시 수행.
- **교체 기록:** `/workspace/no-smoking-results/integration-main-v22-1154-gpu-pool-switch.log`. 기존 복구기 21476·supervisor 21477·simulator 21675 정지(pipeline `failed / Pipeline interrupted`) → 새 복구기 28601·supervisor 28787·simulator 29104. 재개 로그 `verified completed day ... no model calls or graph writes`, `processing 1028 agents`(126행 보존). 완료 날짜 파일 8개 해시 불변(`/workspace/no-smoking-gpu-pool/switch-pre-hashes-20260930T110543Z.txt`), `SWITCH OK: rows 126 -> 127`. 모델 서버(PID 74)·Neo4j·인스턴스 재시작 없음.
- **활성 경로/해시 (서버):**
  - 런타임 층 `/workspace/no-smoking-runtime-hotfix-v22-gpu-pool/`: manifest `3ce5ea010f86f7ce0fc695ca0807d4ed539d65450dac974d0e0bce143da6bb0a`, sitecustomize `1ca17ba77042175c1b07be4727a6b48a070c6e22af812a5f0736d38088b28c90`, resume_runtime `e832da0de6a94ac353ccac4e2ebff414d650f9ac10f7dd128413dfeec5c67c2d`, onstart `f1168c49d00e1b1c5d77146f98a1c53ccb3915f5206f585aa6ea0dbc881321f9`. 로컬 사본: `deploy/vast/runtime_hotfix_v22_gpu_pool/`.
  - `/workspace/onstart.sh` = 위 onstart(이전 것은 `/workspace/onstart.sh.pre-gpu-pool-20260930T110543Z`, sha `48e96c24…`).
  - 프록시 `/workspace/no-smoking-gpu-pool/`: `gpu_pool_proxy.py` `68515d2568c600250b736c523557d2f1cbf0203c06839256c06e99fb8cd95b36`, `config.json`(포트 30000, remotes `colab-g4-0`=18001·`colab-g4-1`=18002 capacity 1.5, remote_timeout 300), `keeper.sh`(flock, 1초 재시작 — kill 후 재기동 실측), `switch_at_boundary.sh` `a81eeddb…`.
  - 로그: 요청별 출처 `…-gpu-pool.jsonl`(backend, request/response sha256, fallback), 프로세스별 경로 결정 `…-gpu-pool-routing.jsonl`.
  - 새 복구기는 night-progress 복구기(`c798e9f8…`)를 해시 검증 후 그대로 불러 쓰고 같은 `persistent-recovery.lock`을 쓴다 → 옛 복구기를 다시 띄워도 잠금으로 종료된다.
- **오프/복귀:** `touch /workspace/no-smoking-gpu-pool/DISABLED` → 다음에 시작하는 simulator부터 8000 직결(실행 중 프로세스는 그대로; 프록시는 Colab 없으면 어차피 전부 local). 완전 복귀는 새 복구기·supervisor 정지 후 `onstart.sh.pre-gpu-pool-*`로 복원·실행.
- **서버에서 실제 확인한 검사:** (a) 층 격리 시험 A/B/C: 현재 체인→8000, 새 체인→30000+night-progress 설치 유지, DISABLED→8000. (b) execution/source fingerprint가 새 체인에서도 `3bb9b019…`/`d96436a0…`로 cohort 기록과 동일. (c) 복구기 dry-run. (d) OpenAI 클라이언트로 프록시 경유 3요청이 local/colab-g4-0/colab-g4-1에 분산. (e) 제한 키: 셸 거부·미허용 포트 거부·허용 포트만.
- **Colab 실측 (G4 = RTX PRO 6000 Blackwell 97.9GB, 48 vCPU, nvcc 12.8.93 = Vast와 동일):** 서버 2개(8000: mem_fraction 0.293/KV 33,418; 8001: 0.636/KV 96,179) identity 차이 없음. 터널 경유·입력 6,577토큰·동시 6요청: 합계 125.0 tok/s, 요청당 21.0 tok/s. 같은 시각 Vast 운영 8요청 합계 ~72 tok/s. 서버 1→2개 이득 ~16%(1개 8요청 175 tok/s, 2개 203 tok/s; 입력 3.1K). 3개째는 메모리 여유 부족으로 미사용.
  - 설치 시 필요했던 것: freeze는 `--no-deps`로 설치(Vast는 sglang 뒤에 transformers 5.8.0을 덮어씀), 서버 실행 PATH에 `/content/venv_sgl/bin`(ninja)과 `/usr/local/cuda/bin`, 두 번째 서버의 `mem_fraction_static`은 남은 메모리 기준으로 계산, Secrets의 개인키는 줄바꿈 복원 필요. Python 3.11.16(Vast 3.11.14).
- **미검증/실패한 것:** Colab 세션이 09:25 UTC에 유휴로 끊겼다(실행 중 셀이 없었음 — 상태 루프 셀 안내 누락). 그래서 **두 GPU로 본 실험을 실제 처리한 구간은 아직 0**이고 합산 가속·오류율·Colab 종료 시 in-flight 복귀는 본 실험에서 미측정(프록시 단위 시험만 통과). 교체 후 요청 32건 전부 local, request_failed 0.
- **운영 중 실수:** Vast `authorized_keys` 끝에 개행이 없어 제한 키가 기존 키 줄에 붙었다가 백업(`authorized_keys.bak-gpu-pool-20260930T071054Z`)에서 복원 후 재등록(기존 키 접속 확인).
- **다음:** 사용자가 Colab을 다시 실행(상태 루프 셀을 계속 실행) → 터널 재연결 시 프록시가 identity 재검증 후 자동 투입. 그 뒤 백엔드별 처리량·fallback·날짜 소요시간을 측정해 기록. `runtime_changes/`에 gpu-pool 층 tarball·기록 추가, `No_SmokingZone_EXP`의 HANDOFF·`ACTIVE_DEPLOYMENT.json` 반영(이 세션은 그 워크트리에 쓸 수 없음). Codex `vast-drive` 감시에 새 복구기 경로 전달. 남은 위험: 프록시 프로세스 사망 시 1~2초간 in-flight 요청 실패(시도 1회 소모), post 전환 시 디스크 10GB 조건(pre pair 6.7GB 삭제 후 확보 예상, 미검증).

## 10. 사용자 결정과 실측 갱신 (2026-10-01 00:2x KST)
- **결정: "OFF/ON에서도 두 GPU 섞어서 돌려".** 설정 변경 없음(현재 프록시·런타임 층이 post-off/post-on에도 그대로 적용된다). `DISABLED` 플래그를 만들지 않는다. 요청별 처리 서버 기록으로 arm별 GPU 비율을 사후 점검할 수 있다.
- Colab 재투입 11:30 UTC 이후 끊김 없음(15:22 UTC 기준), fallback 0, request_failed 0, 누적 요청 local 1,145 / colab 2,078.
- 하루 단위 실측: Day8 `2017-11-26`(Vast 단독) agent 32,635s + Night2 2,514s = 35,150s. Day9 `2017-11-27`(교체 후 1,028명, 초반 약 25분은 Vast 단독) agent 13,216s + Night2 1,059s = 14,275s, ok 1,149 / skipped 5. Day9 완료 백업 Drive 검증 15:14:52 UTC.
- 처리량: 풀 적용 구간 약 300명/시간(이전 120~127명/시간).
- 관측: 백업 직후 Neo4j `defunct connection`으로 Day10 첫 시도 1건 실패(기존에도 12회 있던 현상, 재시도로 처리).
- 세션 내 1시간 모니터링: `/workspace/no-smoking-gpu-pool/pool_monitor.py`(읽기 전용)를 매시 37분 실행.

## 11. 디스크 정리 (2026-10-01 05:0x~05:43 UTC, 사용자 승인 "그래")
- 이유: OFF/ON 전환 후 여유 약 6GB + 하루 0.4~0.6GB 증가 → post 도중 디스크 고갈 예상.
- `/root/.cache/pip` 5.1GB 삭제(캐시).
- 이전 시도 DB `no-smoking-neo4j-main-v15`, `-v17`(정지 상태)을 `deploy/gpu_pool/retire_old_pairs.py`로 arm별 오프라인 덤프 → Drive `no_smoking_drive:No_SmokingZone_EXP_Backups/retired/neo4j-main-v1{5,7}/{off,on}/neo4j.dump` 업로드·MD5 검증 → 두 arm 모두 검증 후 폴더 삭제. 영수증: 서버 `/workspace/no-smoking-checkpoints/retired-pairs/main-v1{5,7}.json`(Drive에도 업로드), 로그 `/workspace/no-smoking-results/retire-old-pairs.log`.
  - v15 off 789,595,101B sha256 `84018be787600d4c671355a391e10769b3fb4b3c38f0f72b872ef03dfd39d8a9`; on 841,287,255B `03f3bcc5604bac7b3a4bb19a86577db9b673ff5d1a3ec798f0cfaafc7d9700da`
  - v17 off 782,110,152B `7a60c23a03bdc1ffc19abe952cd4b28edd4c50c539121b1401317ae347c2576b`; on 841,287,171B `c9feb2793ff9bf5ea4441820c239bd7a57eee8539f78282511f54fe3ab85bb4c`
- 결과: `/workspace` 여유 7GB → **30GB**. 실험 프로세스·현재 pre DB·모델 캐시·체크포인트는 건드리지 않음.
- **주의(발견):** v17/off에 9-23 비정상 종료로 남은 `run/neo4j.pid`(29170)가 현재 simulator의 스레드 ID와 겹쳐 `neo4j status`가 "running"으로 오판했다. 이 상태에서 `neo4j stop`을 호출하면 실험 프로세스를 죽일 수 있다. 정리 스크립트는 pid 파일 대신 실제 JVM(argv[0]=java)의 경로 참조로 판정하고 `neo4j stop`을 호출하지 않는다. 다른 오래된 Neo4j 홈을 다룰 때도 같은 주의가 필요하다.

## 12. 두 번째 Vast 인스턴스 (2026-10-05, 사용자 대여)
- 인스턴스 `54298237`, L40S, $0.562/h, **기존 `52220534`와 같은 물리 머신 27249**(같은 공인 IP, SSH 포트 44660). 이미지 동일(`pytorch/pytorch@sha256:39236c0a…`), 드라이버 570.133.20, Python 3.11.14, nvcc 12.8 — 기존과 같음.
- 설치: `/workspace/gpu-worker/setup.sh`(= `deploy/gpu_pool/vast_worker_setup.sh`)로 freeze `--no-deps` 설치 + 모델 revision 다운로드(10:38 UTC 완료). **추가로 `libnuma1 2.0.14-3ubuntu2`(기존 서버와 같은 버전) apt 설치가 필요했다**(없으면 `sgl_kernel` import 실패).
- 실행: `/workspace/gpu-worker/keeper.sh`(= `deploy/gpu_pool/vast_worker_keeper.sh`)가 기존 서버의 recorded argv 그대로(mem 0.88, 포트 8000) SGLang을 유지하고 역방향 터널 `127.0.0.1:18003`을 연다. 같은 머신이라 공인 IP 헤어핀이 막혀 **내부 주소 `172.17.0.8:22`**로 접속(HostKeyAlias로 기존 호스트키 고정). keeper의 flock fd가 자식에게 상속돼 재기동이 막혔던 문제를 `9>&-`로 수정.
- 검증: 기존 서버에서 `127.0.0.1:18003` identity diff **NONE**, KV 86,671 토큰(기존과 동일). 소량 시험(입력 6,567토큰, 동시 4, 워밍업 후) 합계 약 80 tok/s, 요청당 21.3 tok/s(같은 프롬프트라 prefix 캐시 영향 가능 — 실측은 본 실험에서).
- **발견: 기존 서버 `/root/.ssh/authorized_keys`가 2026-10-01 21:12 UTC에 계정 키 1줄로 재작성돼 제한 키가 사라져 있었다**(Vast 키 동기화로 추정). 그 이후 Colab도 재접속 불가였다. 10-05 11:06 UTC 백업 후 같은 제한 옵션으로 재등록, 기본 키 접속·셸 차단 확인. 재발 가능 → 모니터링 시 확인 필요.
- 프록시 등록: `reload_proxy_when_idle.sh config.v3-vast2.json`을 11:08:47 UTC에 예약(백업 중 + 전 백엔드 inflight 0 3회 확인 시에만 재시작). remotes에 `vast2-l40s`(18003, capacity 1.0) 추가. 로그 `…-gpu-pool-reload.log`.
