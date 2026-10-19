# 2026-09-24 전체 실행 경로 점검

사용자 지시: 본 실험을 반복 재시작하며 수정하지 말고, 전체 활성 코드를 점검하고 통합 검증을 통과한 뒤 실행한다. 대상 3구 1,154명, 고정 흡연 라벨, 14일 공통 선행 + OFF/ON 각 14일, LG EXAONE AWQ revision과 정책 조건은 유지한다.

## 비용 및 기존 기록 보존

- Vast 52220534: CLI에서 `actual_status=exited`, `cur_state=stopped`, `intended_status=stopped` 확인. GPU 중지 중이며 저장공간은 유지한다.
- v18 launcher/continuation/12h 파일 백업 프로세스 종료. 점검 중 `vast-drive`를 PAUSED로 했다가 02:45 UTC 기존 서버 가용성 및 준비된 통합 검증을 이어가는 ACTIVE 감시로 갱신했다. 본 실험은 아직 재시작하지 않았다.
- Drive `No_SmokingZone_EXP_Backups/diagnostic/audit-pause-v18-20260924/`에 4,125개 파일 + 부분 그래프 + diagnostic.json + committed.json 보존. 전부 원격 MD5 확인 및 완료 마커 재조회.
- 기록 archive SHA256 `3635411d2f85fcfd197abf6b14a2c62375fe08e618b1f9327861523c3857b3bd`; 부분 graph SHA256 `13ddd1c519ddd5b5a0cd359cc7774fc5b5c458dfbd0b05bc01d664c50239fd6c`. 기록 archive는 로컬에도 회수·해시 확인했다.

## 확인된 문제와 수정 범위

1. **입력 길이 검사 범위 부족.** 보존된 v14/v15/v17/v18 증거 22,060개를 seal 검증하고 고유 프롬프트 5,915개를 고정 LG tokenizer로 전수 계수했다. 최대 입력 10,095 + 출력 예약 2,400 + 여유 128 = 12,623. 18개 요청은 이전 12,288 설정도 초과한다. 후보 서버 컨텍스트 16,384를 선언하고 runner가 실제 서빙 설정·revision과 일치하는지 검사하도록 수정했다. 실제 GPU 메모리·처리량 검증은 아직 필요하다.
2. **상충된 근거 출력 계약.** Stage1은 evidence_quote를 필수라고 쓰면서 뒤에서는 쓰지 말라고 했다. evidence_ref로 통일하고 Stage1·Night도 실제 입력에 존재하는 번호만 출력할 수 있는 JSON schema를 추가했다. Stage1의 직전 관측 블록도 번호 부여 범위에 포함했다. 의미적 타당성은 별도이며 번호 검증만으로 추론이 사실이라고 주장하지 않는다.
3. **다른 활동의 POI 선택 허용.** Stage2의 전체 후보 union enum을 order/그 order의 후보가 함께 묶인 선택지로 바꿨다. 완성된 선택은 보존하고 누락 항목만 재요청하는 기존 방식, minItems 제거 및 maxItems 상한은 유지한다. 금연 실험과 무관한 지원금·실시간 리뷰 출력 필드는 금연용 schema에서 제외했다.
4. **중첩 재시도와 비용 통제.** SDK의 숨은 재시도를 끄고 호출 timeout을 명시했다. 문맥 초과·영구 HTTP4xx는 즉시 전파한다. 동일 잘못된 응답을 동일 오류로 반복하면 중단한다. 바깥 에이전트 재시도는 실제 일시 장애만 대상으로 하고, 동시에 workers 수만큼만 작업을 제출한다. 영구 오류 시 남은 명단의 호출을 멈추고 진행 중 결과를 회수한다. 사용 토큰 집계는 마지막 응답뿐 아니라 재시도 합계도 포함한다.
5. **야간 저장의 원자성.** Conversation/Memory/관계와 완료 outbox를 한 DB transaction으로 저장한다. DB commit 뒤 로컬 마커 쓰기 전에 중단돼도 검증된 outbox에서 마커만 복구할 수 있게 했다. 실제 DB 장애 주입 검증은 아직 필요하다.
6. **같은 코드의 재개 기능 부재.** runner의 `--resume`은 roster·모델·소스·설정·날짜·팔을 기존 manifest와 비교한다. 완료 날짜는 그래프의 metrics outbox와 파일·야간 증거를 맞춘 후 건너뛴다. 부분 append는 마지막 미완성 행에 한해서만 원문을 보존하고 DB outbox로 재처리한다. 변경된 코드로 과거 실행에 이어붙이는 것은 허용하지 않는다.
7. **12시간 백업에 그래프 없음.** 새 방식은 시작 시와 마지막 검증 후 10시간에 새 호출을 쉬고 기존 호출이 끝난 시점에 전체 진행 파일과 offline graph dump를 함께 Drive에 업로드한다. 정상 날짜 종료 백업도 유지한다. 원격 checksum·완료 마커 후에만 로컬 복구 영수증을 갱신한다. 기존 progress-only 타이머는 새 본 실행에 쓰지 않는다. 실제 Drive 다운로드·DB 복원·이어 실행 검증은 아직 필요하다.

## 실제 완료한 검사

- 변경 전 sim/deploy: 984 passed, 7 skipped, 7 subtests passed, 기존 outside_need 기대값 불일치 3 failed.
- 변경 후 집중 회귀: 110 passed, 7 subtests passed.
- 최종 전체 sim/deploy: 994 passed, 7 skipped, 7 subtests passed, 동일 기존 outside_need 3 failed. 이 helper는 현재 금연 실행 경로에서 호출되지 않는 과거 별도 실험 도구다. 이 문서의 통과 숫자는 실제 LG 생성이나 실증 평가를 뜻하지 않는다.
- persona 별도 프로세스: 5 passed. sim/persona를 같은 pytest 프로세스에 모으면 기존 `_common` 모듈 이름 충돌로 수집 오류가 생겨 분리 실행했다.
- 전수 기록 감사 JSON: 로컬 `deploy/vast/local/audit-pause-v18-20260924/recorded-audit.json`. 본문에는 시민의 원문·비밀정보를 복제하지 않는다.

## 본 실험 재개 전 남은 필수 검증

고정 후보 SHA256 `6c1cb3cc42f923742f7505121149a7f33e2df9d24eb18f57627d7bd51022e9c0`의 archive 내부 657개 파일은 일치한다. 02:36~02:46 UTC 기존 Vast 재시작 요청은 자원 부족으로 모두 대기 응답을 받았다. **GPU 재현·통합·실제 복원은 미실행**이며 본 실행 시작도 차단되어 있다. `docs/VAST_V19_INTEGRATION_HANDOFF.md`에 사용 가능 시 수행할 고정 절차·helper를 기록했다.

1. 고정 패키지의 소스/통계/모델·토크나이저 해시와 전체 검사 결과 확정.
2. 이전 실패·최장 요청을 포함한 실제 LG 구조화 생성, 16개 동시 요청·OOM·출력 잘림 검사.
3. 진단용 별도 표본에서 여러 날짜의 기억 누적과 정책 전후, 공통 선행 → 그래프 복제 → OFF/ON → 점수까지 한 패키지로 통합 실행.
4. DB 저장 중 장애/마커 유실·재개 시 중복 없음, Drive에서 받은 dump·결과를 실제로 복원해 검증.
5. Drive 용량과 전체 42개 날짜 dump 보관량, 디스크 여유 및 백업 실패 시 후속 단계 차단 확인.
6. 통과한 동일 패키지로 본 실험 실행, 시간별 감시 재개. 검증 실패 시 본 실험은 시작하지 않고 원인을 종합 점검한다.
