# Codex → Claude 인수인계 — 2026-09-20

## 먼저 읽을 결론과 사용자 요청

사용자가 **현재 실행 중인 실험까지만 마무리하고 나머지는 Claude에게 넘기라**고 요청했다.
해당 실험과 결과 감사·백업을 완료했다. 새 GPU 실험은 시작하지 않았다.
마지막 계획/결제 프로세스와 로컬 미러는 종료됐다. 모델 서빙 프로세스는 그대로다.

**현재 방식에 문제가 없다고 결론 내릴 수 없다.** v25는 초기 계획·결제 검사를 통과했지만,
추가 반복에서 배송 전에 세제를 사용하는 불가능한 계획이1건 발생했고 실제 결제에서도
실패했다. 이 행을 삭제·0원 대체·자동 시간 보정하지 않았다. 따라서 v25/v7은 현재 후보이며
미시 실행과 실제 정책 효과의 방향·대략적 크기까지 검증된 최적판이 아니다.

목표는 정책별 정답을 주입하지 않는 범용 시민 행동 프롬프트다. 공통 문구와 정책·사회 배경·
시민 상태 입력을 분리한다. 제도상의 금액·가격·자격 같은 입력 사실은 필요하지만,
원하는 소비 부호나 실측 효과 목표를 공통 지시·평가 피드백으로 주면 안 된다.
사용자는 정확한 효과 수치보다 방향과 대략적인 크기의 검증을 원한다.

**P010 원본 불변, P015 재실험 금지.** 모델은 EXAONE-4.5-33B-AWQ다.
P010 SHA256: `82bf0f7645e8455befc9108cda18d1ba270a97006575ff4315e0524c71e93611`.
기존 `HANDOFF_CODEX.md` 본문과 `INSIGHTS.md`의 v5 확정·방향12/12 판단은 역사적 기록이다.
[검증 아키텍처 감사](VALIDATION_ARCHITECTURE_V3.md)의 반박과 현재 결과를 우선한다.

## 작업 위치와 서버

- 로컬: `G:\내 드라이브\Kw\final_project`, PowerShell.
- 브랜치: `exp/p010-bok-validation`.
- GitHub: `https://github.com/wikriwiki/kw_26_final_project.git`.
- 실험 코드: 서버 `/data/validation_v3/repo`. 기존 `/data/repo`와 공유 DB를 건드리지 않았다.
- SSH: `ssh -i outofmemory.pem -p 10022 outofmemory@123.37.28.167`.
- 서버 제공자는 사용자 확인상 **메가존클라우드**다. 회수일·보존 보장 조건은 확인되지 않았다.
- A100-SXM4-80GB, `/data`는 별도1TB ext4 볼륨. 서버 밖 복사 없이 영구 보존을 보장하지 않는다.
- `/data/venv/bin/python`: 단위검사·원문 감사·자료 준비.
- `/data/venv_sgl/bin/python`: 모델 호출·transformers·xgrammar.
- SGLang `http://localhost:8000/v1`, native `/generate`, qwen3 reasoning parser.
- 마지막 `/v1/models` 조회에서 `LGAI-EXAONE/EXAONE-4.5-33B-AWQ`를 확인했다.
- tokenizer: `/data/hf_cache/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/31e6a965d0661bbe4a8b895e22a77f8271772ba0`.

`/data/stage8.sh`는 초기화 작업이 있어 실행하지 않는다. Neo4j 환경 변수만 필요하다면
스크립트의 `export NEO4J_` 줄만 읽는 방식으로 분리한다. 암호나 개인키를 출력·커밋하지 않는다.
실험 코드는 SCP로 격리 체크아웃에 배포했으므로 서버 Git HEAD만 보고 실행 코드를 추정하지 말고
각 run의 `code/`, manifest 해시, `system.txt`, 요청 원문을 확인한다.

로컬 Git 쓰기 시 사용한 설정:

```powershell
$env:GIT_OBJECT_DIRECTORY='C:/Users/Administrator/gitobj/kw26'
$env:GIT_ALTERNATE_OBJECT_DIRECTORIES=(Join-Path (Get-Location) '.git/objects')
```

기존 `.git/AUTO_MERGE.lock` 경고가 있어도 실제 커밋·push는 성공했고 원격 HEAD를 확인했다.
임의로 lock을 삭제하지 않는다. 무관한 미추적 대용량 파일이 많으므로 `git add .`를 쓰지 않는다.
팀장의 `codex/fe039529-execution-audit`에서 필요한 측정/회계 보호는 선택적으로 반영했다.
관련 커밋은 `a4610ea`, `3a5d132`; [반영 평가](MERGE_ASSESSMENT_CODEX_AUDIT.md) 참조.
브랜치 전체를 무비판적으로 병합한 상태라고 해석하지 않는다.

## 현재 프롬프트와 실행 계약

- 계획: `scripts/sim/prompts/v25.py`.
- 결제: `scripts/sim/prompts/asset_transaction_v7.py`(v6에 일반 자원·배송 지시를 추가).
- 실제 전체 문구와 해시: [CURRENT_UNIVERSAL_PROMPT.md](CURRENT_UNIVERSAL_PROMPT.md).
- 계획512추론 토큰/2048답변 토큰, 온도0.2,30분 시각 문법,18시 이후 마지막 집 활동.
- 결제1024추론 토큰/4096답변 토큰, 온도0.2, v4 선택 문법·잔액 입력.
- 사건 ID·시각/장소·확정 근무/수업의 일부는 문법으로 보장된다. 이를 모델이 자발적으로
  현실 생활을 학습한 증거로 세지 않는다. 후처리 시각 보정은0이다.
- 후보/취득 제안이 전혀 없으면 결정적으로 미구매를 반환한다. 모델 성공 건수와 분리한다.
- 실제 결제에서 회계와 물리적 재고를 별도로 검사한다. 회계상 가능해도 배송 전 사용은 실패다.

범용 문구에 특정 정책 이름·원하는 방향·실측 목표는 없다. 다만 문구만으로 모든 문제가
해결된 것은 아니다. 재고와 상품 목록 같은 입력, 실행 문법, 결제 엔진도 함께 결과를 결정한다.

## 마지막 실험의 확정 결과

동일한 알려진12명×4개 기전×on/off=96조건이다. 새로운 홀드아웃이 아니다.
`daily_v1`은 v24/v25를 동일한 당일 조건·seed52001로 비교했다.
`daily_v2`는 v25를 고정하고 seed56001/57001로192계획을 추가했다.
세 계획 반복의 결제 seed는53001로 같다.

| 실험 | 완료와 검사 결과 | 실제 모델 결제 / 결정적 미구매 |
|---|---|---:|
| daily_v1 v24 | 계획96/96, 결제96/96 | 70 / 26 |
| daily_v1 v25 | 계획96/96, 결제96/96 | 43 / 53 |
| 자원 사실 시험 | 결제·배송·재고8/8 | 8 / 0 |
| daily_v2 seed56001 | 계획96/96, 결제96/96 | 50 / 46 |
| daily_v2 seed57001 | 시간/장소 계획96/96, **결제95/96** | 42 / 54 |

daily_v2 계획 평균715.42토큰, 원문 요청384개를 감사했다.48개 균형 표본의 전체 행동열을
읽었으며, 나머지144개는 자동 원문 검사만 수행했다. 별도로 자원 사용 활동9개를 확인했다.
인접 동일 활동/장소 반복4건, 근무 시작 이후 사무실 준비0건, 의무 반복1건이었다.
동일 조건의 반복 행동열 거리0.366~0.416과on/off 거리0.381~0.448은 비슷한 규모다.
이 거리는 소비 효과 크기·신뢰구간·유의성 지표가 아니다.

daily_v2 실제 모델 결제 선택92개를 모두 검토했다. 결정적 미구매100건은 별도다.
성공 행의 요청182개와 실패 행의 요청2개를 감사해 총184개를 확인했다.
일반 감사기는 실패 행에서 일찍 중단하므로 해당2개는 별도 보충 감사로 처리했다.
관련 근거: `docs/experiment_evidence/daily_v2_purchase_final_review.json`.

### 실패1건의 정확한 내용

- 계획 seed57001, `AGT_11620565_F_70대이상_001`, `distancing/on`.
- 초기 세제0회분,14:00 구매 검토,15:00 세탁. 입력상 배송90분이므로15:30 도착이다.
- 실제 모델은14시 구매를 선택했지만15시 사용은 여전히 불가능했다.
- 결제 attempt key: `a940f36166db55224758136187c362b0cb9945ae3ed074f7ab195bc4101ccd43`.
- 오류: `Physical inventory shortage: event:6/detergent_dose`.
- 결제 원문과 회계 자체는 검사에 통과했고, 물리적 실패를 독립 재현했다.
- `resource_feasibility.py`의 최대 공급 검사도 이 계획을 불가능으로 판정한다.

이로 인해 두 번째 반복의 전체 연결 행렬은 적격하지 않다. `report_purchase_repeats.py`에
원래 반복까지 포함한 세 경로를 넣었을 때 실제로 통합 비교가 거부됐다. 실패 행을 제외한
효과나 성공 반복만의 평균으로 검증 성공을 만들지 않는다. 개별 연결 보고서에 남은 완전한
다른 기전의 기술값도 전체 정책 검증 완료의 근거가 아니다.

## 핵심 수정과 아직 안 된 것

1. 시민 서술이 다른 세대의 직업/소비 기준과 섞인 문제를 찾아 원래 백업 서술을 복원했다.
   공유 DB는 수정하지 않았다. [PERSONA_INPUT_LINEAGE_AUDIT.md](PERSONA_INPUT_LINEAGE_AUDIT.md).
2. 당일 근무·수업·선택적 빨래·재고를 on/off에 동일하게 제공했다. 이는 합성 가정이며
   실제 개인 시간표나 재고의 관측값이 아니다. [daily_v1 설계](DAILY_CONDITIONS_V1_EXPERIMENT.md).
3. 원문 회계·배송·자원 소비 재검사를 추가했다. `resource_feasibility.py`는 모든 후보를
   살 수 있다고 낙관해도 공급이 부족한 경우를 찾는다. 통과가 자금/배타적 선택의 가능성을
   보장하지는 않는다. **아직 계획 runner의 자동 사전 게이트로 연결하지 않았다.**
4. `daily_state_transition.py`와 `asset_day_checkpoint.py`의 `carry_needs_v1`은 남은 필요,
   현금·지갑·재고·배송을 이월한다. 실제v25 결제96건으로 첫날 체크포인트와 다음 날 초기
   상태를 생성했다.94건의 남은 필요를 유지하고2건의 완료 필요를 재생성하지 않았다.
   **다음 날 LLM 행동을 실행한 것은 아니다.** 새 필요·소득0, 남은 필요 지속은 명시적 감사 가정이다.
5. 무료 집밥·직장 식사는 음식 재고를 소진하지 않는다. 현재 좁은17종 합성 상품으로는
   가계 전체 수요를 설명할 수 없다. 미래 수요·소득·환급·기억·필요 만료의 실제 전이도 미완료다.
6. 실제 정책 효과와 비교할 개인/가구 단위, 기간, 자격, 비교군, 분모가 아직 완전히 정렬되지
   않았다. 과거 %p와% 혼동을 반복하지 않는다.2026년 음식 중심 메뉴를2020/2021년 가격으로 쓰지 않는다.

최신 실행 검사: 서버에서
`PYTHONPATH=scripts/sim /data/venv/bin/python -m pytest tests/unit/sim -q`
→ **271 passed in1.00s**. 이전 문서의373개는 당시 실행 범위의 기록이며 위 sim 디렉터리
검사와 합산하지 않는다. 단위시험 통과가 행동·효과 타당성의 증거를 대신하지 않는다.

## 다음 작업: 준비만 했고 실행하지 않음

`scripts/sim/prepare_resource_revision.py`는 이미 관찰한 자원 불가능 계획만 골라,
원래 조건·이전 계획·사실상의 자원 부족을 주고1회 재검토할 입력을 만든다.
특정 정책의 효과 방향·수치나 정답 일정을 주지 않는다. 원문은 그대로 남긴다.

입력 생성까지만 실행했다:

- 서버 `/data/validation_resource_revision_v1/source.json`.
- 로컬 `output/validation_resource_revision_v1/source.json`.
- SHA256 `ab6bf196713bc63a1dc893ba1dd1ae5ff40c23f3875e1216e27fe1c209c00a70`.
- **수정용 모델 호출0회, 모델 실행 config 미작성, 런처 없음.**
- 선택된 이미 본 실패1건의 진단이므로 새 홀드아웃·일반 성능 향상으로 해석하지 않는다.

Claude가 이어받을 때 권장 순서는 다음과 같다. 먼저 자원 검사를 계획→결제 사이에 연결하고,
수정 전략을 시험한다면 첫 실패·피드백·수정 출력을 모두 남기는 별도 프로토콜로 등록한다.
새 프로토콜은 새로운 완전한 실행에서 확인한다. 기존 실패 행 하나를 고쳐 넣고 원래
반복이 통과한 것으로 바꾸지 않는다. 이후 음식/재고/필요와 다일 상태를 보강하고, 외부
효과와 추정량을 맞춘 뒤 방향·대략적 크기를 검증한다. 단순 표본 확대만으로 이 공백을 덮지 않는다.

## 원본·백업과 복구

서버 run의 `manifest.json`, `frozen_inputs.json`, `system.txt`, `responses.jsonl`,
`attempts/`, `code/`가 기준이다. 요청을 호출 전에 저장하고 완료행은 fsync했다.
작업은 nohup/setsid로 실행해 SSH 단절과 분리했다. 로컬 미러는180초마다 이미 저장된
파일을 복사했지만 마지막 성공 복사 이후와 생성 중인 응답까지 보장하지 않는다.
Google Drive 원격 동기화/클라우드 보존은 확인하지 않았다.

아래 최종 압축본은 로컬과 서버에서 읽어 SHA256 일치를 확인했다. Git에는 감사/설정/
문서를 올렸으며 대용량 원문 압축본을 모두 올린 것은 아니다. **GitHub만 clone하면 원문이
복구된다고 생각하면 안 된다.**

| 로컬 output 아래 경로(서버는 같은 validation 폴더를 /data 아래 사용) | SHA256 |
|---|---|
| validation_daily_v1/v24_reviewed.tar.gz | `39dc7b7b83621b8a0405d6ae6fd3647582d93ddb1ab6e8d9f8e08932024ebf61` |
| validation_daily_v1/v25_reviewed.tar.gz | `b8b4132347bf57f762c7a48e4b7dbc7922ffcec053de5733ba9f5782b935a4e8` |
| validation_daily_v1/v24_purchase_reviewed.tar.gz | `5371bd593d68ab03832baba49fc0aa21a2275c972e39917b0ab0341db66dda01` |
| validation_daily_v1/v25_purchase_reviewed.tar.gz | `2be1c26e4fbd05d04ca525bc3b9431ce6dbe92071fd0f9a64880d0f29808781d` |
| validation_daily_stress_v1/development_reviewed.tar.gz | `9ce5343d31fdf928df9944ac75814ad7b74d86e363dd28da0df0b1974cc333ed` |
| validation_daily_v1/v25_state_carry.tar.gz | `ba01d76817a2744f488884c91b0e2d6e7edfda979a4a3fe0c99a551f8ffe4f16` |
| validation_daily_v2/development_reviewed.tar.gz | `20740208fc997ed6bd81d8ce4e864b942884b33e0809ec7eb835c13980575cfd` |
| validation_daily_v2/purchase_56001_reviewed.tar.gz | `77fab255afff512e245cd0963472f2d75df5ab1ed07fe5ce7b88d46ccdf8ab13` |
| validation_daily_v2/purchase_57001_reviewed.tar.gz | `745b0f8f45ee706226d67a84913b4e7b55e56460ccbf7190faf7f6d5b962e265` |

완료된 과거 실험을 재시작하지 않는다. PID1163230/1165325/1167232는 종료 확인했지만,
PID는 재사용될 수 있으므로 번호만 보고 프로세스를 정지하거나 재시작하지 않는다.
후속 작업은 현재 명령을 확인하고 새 출력 폴더에서 수행한다. 기존 출력이나 공유DB를 초기화하지 않는다.
