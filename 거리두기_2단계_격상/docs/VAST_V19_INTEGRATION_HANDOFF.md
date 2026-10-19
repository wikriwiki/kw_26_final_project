# 고정 후보 v19 실제 검증 절차

2026-09-24. 사용자 지시: 전체 코드 점검 → 통합 검증 → 본 실험. 본 실험을 돌리며 오류마다 코드를 바꿔 재시작하는 방식은 중단한다.

## 현재 상태와 고정 입력

- Vast 52220534는 진단 자료 Drive 검증 보존 후 01:57 UTC 중지했다. 02:36~02:46 UTC 기존 인스턴스 시작 요청 세 번 모두 `Required resources are currently unavailable, state change queued.`로 실패했다. 02:42 UTC에도 actual=exited, cur/intended=stopped이며 SSH 연결이 거부된다. **새 임대는 금지**다. 이후 시간은 실제 CLI로 다시 확인한다.
- 후보 `deploy/vast/local/source-exaone-sglang-3gu-1154-audit-v19.tar.gz`, SHA256 `6c1cb3cc42f923742f7505121149a7f33e2df9d24eb18f57627d7bd51022e9c0`. 657개 소스·통계 파일을 archive 내부에서 다시 해시 검증했다. 원격에 아직 전송하지 않았다. 패키지 뒤에 갱신한 로컬 보고서·인수인계는 패키지 내용과 다를 수 있으나 실행 코드는 수정하지 않았다.
- 오프라인 sim/deploy 994 passed, 7 skipped, 7 subtests passed, 기준 브랜치에서도 재현된 비활성 outside_need 실패 3개. persona 5 passed. compileall 통과. GPU 생성·42일 통합·실제 복원은 **미실행**이다.
- 진단 입력: `deploy/vast/local/recorded-probes-v19.jsonl` 86건(계획36, 장소36, 야간14), `deploy/vast/local/integration-bundle-v19.tar.gz` 8명(원래 명단·흡연 라벨 유지). 8명은 기술 검증 전용이다. 본 실험 3구 1,154명 번들은 바꾸지 않았다.
- 실제 검증이 실패하면 본 실험은 차단하고 진단 자료를 보존한다. 원인을 종합 점검하는 동안 유료 GPU는 중지한다. 임의 결과·근거·완료 마커를 만들어 통과시켜서는 안 된다.

## 기존 인스턴스가 사용 가능해진 뒤

1. `.venv-vast/Scripts/vastai.exe show instances --raw` 응답을 출력 전 필요한 필드만 선택한다. start는 기존 52220534만 허용된다. 응답에 불가/대기가 나오면 반복 요청하지 않고 다음 감시까지 대기한다. 키·계정 전체 JSON·환경변수·rclone 설정 내용을 출력하지 않는다.
2. SSH는 기존 `~/.ssh/no_smoking_vast`, 직전 endpoint `95.3.33.46:45025`, root. CLI의 ssh_port=20534는 프록시 ssh3.vast.ai 포트였으므로 직접 IP 포트와 혼동하지 않는다. 재시작 뒤 endpoint를 확인한다.
3. SCP로 후보 archive, probes, integration bundle과 아래 `deploy/vast/local/` helper 세 개를 `/workspace/`로 전송하고 SHA를 대조한다: `activate_audit_v19.py`, `restore_roundtrip_v19.py`, `validate_db_faults_v19.py`. 이 helper들은 문법 검사만 통과했고 실제 동작은 아직 미검증이다.
4. 기존 `/workspace/no-smoking-project/.venv-no-smoking/bin/python /workspace/activate_audit_v19.py` 실행. 이는 Drive quota(검증+본 실행 80GiB 여유)·로컬 18GiB 여유, archive SHA를 확인한 뒤 **새** `/workspace/no-smoking-project-audit-v19`에 추출한다. 기존 v18 소스/DB를 덮어쓰지 않는다. 가상환경·tokenizer만 기존 설치에 symlink한다. 결과는 `/workspace/integration-v19-config.json`, `/workspace/main-v19-config.json`; 본 실험은 시작하지 않는다. 중간 실패 시 이미 만든 경로를 확인하고 기존 자료를 임의 삭제하거나 helper를 무작정 반복하지 않는다.
5. 기존 비밀 설정 `/workspace/no-smoking-project/deploy/vast/local.env`를 source하되 출력하지 않는다. `HF_HOME=/workspace/huggingface`, `MAX_MODEL_LEN=16384`, `MAX_NUM_SEQS=16`, TP1, mem0.88, triton을 명시하고 새 프로젝트 `deploy/vast/serve_sglang.sh`를 실행한다. 모델은 고정 LG EXAONE AWQ revision `31e6a965d0661bbe4a8b895e22a77f8271772ba0`, SGLang commit `6757c9f904cdb8ae9028a394a2108d079b9e088c` 그대로다. 다른 HF cache에 모델을 중복 다운로드하지 않는다.
6. server health·revision·config·GPU 메모리를 확인한 뒤 새 프로젝트 Python으로 `deploy/vast/validate_recorded_requests.py --cases /workspace/recorded-probes-v19.jsonl --out /workspace/no-smoking-results/integration-v19-recorded-replay`를 실행한다. 86건 전부 통과, OOM/출력 잘림/컨텍스트 오류 여부와 재시도 수를 실제로 읽는다. 대표 응답의 의미와 후보 선택도 직접 검토한다. JSON schema 통과를 의미적 정확성이라고 보고하지 않는다.
7. 재현 검사가 통과하면 같은 환경에서 `deploy/vast/run_shared.py --config /workspace/integration-v19-config.json` 실행. 로그는 `/workspace/no-smoking-results/integration-v19-continuation.log`, 상태는 `integration-v19-pipeline.json`. 새 DB pre 17751/17752, post 17753/17754. 8명×42일=336 agent-days; 공통 선행·12월3일 경계·OFF/ON·전체 증거 감사·점수와 Drive 백업까지 실제 검증한다. 하루 backup은 offline dump라 시간이 든다.
8. 같은 supervisor의 중단/재개도 확인한다. SIGTERM은 supervisor에 보내 child process group이 끝났는지 확인한다. 코드·명단·설정 변경 없이 같은 config로 재개하며 `--resume`가 완료 날짜를 건너뛰고 부분 날짜의 DB outbox를 재사용하는지 검사한다. 근거 손상은 정상 복구로 취급하지 않는다.
9. 세 단계 완료 후 `/workspace/restore_roundtrip_v19.py`를 실행한다. OFF/ON 마지막 graph를 Drive에서 실제 다운로드·SHA256 대조한 다음 이 검증이 만든 post DB만 정리한다. Dec2 체크포인트 및 14일 증분 archive를 Drive에서 다시 받아 **새** `/workspace/no-smoking-neo4j-integration-v19-restore`(17761/17762)에 적재한다. 검증된 복원 계보를 바인딩한 뒤 새 결과 폴더에서 14일을 `--resume`하여 추가 모델 호출·metrics/증거 변경 없이 완료되는지 확인한다. 원본 Day0 및 v14~18 DB는 정리하지 않는다.
10. 같은 복원 pair의 ON DB에서 `/workspace/validate_db_faults_v19.py`를 실행한다. 연구 결과와 분리된 기술 시험 fixture로 Conversation/Memory/outbox 작성 직후 예외를 넣어 전부 rollback되는지, 정상 재시도 후 outbox 복구가 되는지 확인한다. synthetic fixture를 실험 산출물에 섞지 않는다.
11. 실제 통과 증거·SHA들을 `/workspace/integration-v19-gate.json`에 기록한다. 필수 필드는 status=passed, source_sha256, replay_passed=true, restore_passed=true, full_shared_pipeline_passed=true이며 각각 실제 파일·로그·Drive 위치로 뒷받침한다. 실패/미실행 상태에서 true로 바꾸지 않는다. gate와 검사 결과를 Drive에 checksum 검증 보존한다.
12. 검증용 DB·dump는 Drive에 실제 검증 보존되었고 사용 중이 아님을 확인한 뒤 **해당 검증 경로만** 정리해 본 실행 공간을 확보한다. 같은 패키지의 `run_shared.py --config /workspace/main-v19-config.json`으로 1,154명 본 실행을 시작한다. 초기 결과·프로세스·완전 그래프 startup backup을 확인하고 ACTIVE_DEPLOYMENT와 본 인수인계를 갱신한다.

## 새 실행의 백업·감시

- 백업은 기존 progress-only 타이머를 사용하지 않는다. 날짜 완료마다 dump+그 날짜 근거, 시작 시/직전 검증 10시간 후에는 worker를 비우고 **부분 graph+전체 run 파일**을 백업한다. 원격 MD5 및 마지막 committed.json 뒤에 `recoverable_backup.json`을 쓴다.
- 12시간을 넘기기 전에 backup이 끝나는지 감시한다. source/runtime/roster 변경, 영구 호출 오류, 증거 불일치, DB 불일치, backup 실패는 후속 날짜/분기 실행을 차단해야 한다.
- GPU 작업 종료·모든 점수와 Drive 백업 완료 후 인스턴스를 중지한다. 사후 인터뷰는 아직 실행하지 않았다. 인터뷰 필요 여부를 사용자에게 따로 알린다.
- 현재 감시의 역할은 기존 인스턴스 가용성 확인 → 위 통합 검증 계속 → 통과한 동일 패키지의 본 실험 실행 → 정상 진행/백업 감시다. 대기/정상 상태가 그대로면 알리지 않고 실패·완료·사용자 조치가 필요할 때만 보고한다.
