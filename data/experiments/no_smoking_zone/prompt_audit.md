# 금연 실험 프롬프트 감사

현재 금연 실험은 `no_smoking_v1`을 고정한다. 기존 기본값 `p010`은 과거 쿠폰 실험의 동결 기록으로 유지되며 금연 실행에 사용하지 않는다. 실험 기간은 2017-11-19~12-16, 시행일은 12월 3일이다. 이 문서는 코드·입력 계약의 오프라인 감사이며 EXAONE 실제 응답 품질이나 정책 효과 검증 결과가 아니다.

개인 상황 연결과 입력 보존을 강화한 최신 감사본은 `output/no_smoking_zone/prompt_audit_v3_reasoning`이다. Python 248개, 프롬프트 모듈 52개, 등록 변형 23개, 생성 호출 지점 15개를 목록화했고 합성 입력 46개를 검증했다. 정적 검사 및 소스 해시 일치 확인을 통과했다. `prompt_audit_v2`는 강화 이전 스냅샷으로 보존한다.

## 현재 사용하는 경로

| 단계 | 프롬프트와 입력 | 근거의 의미 |
|---|---|---|
| Stage 1 | `prompts/no_smoking_v1.py`, Dawn 사실, 당일 금연 규칙, 과거 관측 | 하루 계획. 짧은 공개 선택 설명과 사용자 입력의 4~1000자 원문 인용 |
| Stage 2 | `no_smoking_prompts.SYSTEM_STAGE2`, 후보·개인 상태·당일 규칙 | 장소/금액/만족도 제안. 실제 정산 결과와 구분 |
| Night | `no_smoking_prompts.SYSTEM_NIGHT`, 두 시민의 당일 활동·규칙 | 가상 상호작용 제안. 대화 녹취나 실제 찬반 응답이 아님 |
| 인터뷰 | `interview_agent.INTERVIEW_SYSTEM`, 저장된 증거 | 사후 설명. 당시 입력·선택 설명·실행 영수증·해석을 분리 |
| 찬반 수집 | `collect_policy_stances.SYSTEM`·고정 `QUESTION`, 시점 제한 증거 | 정책 입장 응답. 시행 전/OFF는 hypothetical, 시행 후 ON은 experienced이며 실제 이용 경험은 기록이 있을 때만 인정 |

단계별 본문 해시, 선택 조건, 전체 Python 소스 해시, 실제 생성 API 호출 지점은 감사 산출물에 기록한다. 등록된 프롬프트라고 해서 현재 Stage 1 출력 스키마와 호환되는 것은 아니다. 특히 `v14`의 상대시간 표현은 별도 변환이 필요하다. `v22~v39`, transaction 및 asset_transaction 계열은 별도 검증 경로이며 금연 실행 경로가 아니다. `backfill_night_reasoning.py`가 나중에 채운 이유는 당시 생성된 원본 근거로 취급하지 않는다.

## 검토 결과와 수정

- 쿠폰용 프롬프트의 지원금 사용·동적 사건·외출 유도와 금연 규제의 부조화를 확인하여 독립적인 중립 본문을 추가했다. 흡연 여부만으로 찬반, 방문, 지출, 만족도의 방향을 정하지 않는다.
- 시행 이전 OFF/ON은 같은 당일 규칙을 받는다. 시행일부터 ON에만 금연 규칙이 나타난다. 미상 흡연 상태는 미상으로 유지한다. 시설 폐업·흡연실 존재를 추정하지 않는다.
- 본문은 결과 통계·목표 효과·미래 정책 지식을 보충하지 않도록 요구한다. renderer는 정의된 Dawn block만 받고 `ground_truth`와 `future_policy_schedule` 같은 추가 key를 무시한다. 허용된 block 안의 잘못된 정보까지 자동으로 판별한다는 뜻은 아니다.
- 공개 설명과 정확한 입력 인용을 구분한다. 문자열 일치 검사는 인용 출처를 확인하지만 설명의 타당성·인과관계·심리적 진실을 증명하지 않는다. 숨은 사고과정 기록으로 취급하지 않는다.
- Stage 1/2는 관련된 자기 상황·필요·제약을 골라 왜 그것이 이번 선택에 중요한지 2~3문장으로 연결한다. 단순한 일상은 1문장도 가능하다. 실제 대안·상충이 있을 때만 우선순위와 감수하는 부담을 설명하며 매번 장단점을 나열하지 않는다. 기존 출력 토큰 예약을 늘리지 않으므로 모든 이벤트를 장문으로 만들지 않는다.
- 인터뷰는 기록된 개인 상황·관측 사실, 현재의 예상, 가치 판단을 구분한다. 실제 관련된 상충의 우선순위, 입장이 달라질 가정적 조건, 판단에 중요한 미확인 사항을 설명할 수 있다. 분량이나 인용 수를 채우려고 새 개인사·욕구·경험을 만들거나 양면적 입장을 강요하지 않는다.
- 일반 `ask_grounded` 경로도 최근 영수증보다 최신 개인 상황 원문을 먼저 보존한다. 그 항목이 문자 예산에 들어가지 않으면 개인 상황을 조용히 버리지 않고 실패한다. 개인 상황 자체가 없는 입력은 `personal_context_missing`을 표시한다. 운영 찬반 수집의 토큰 기반·출처 연결 선택기는 별도 경로이며, 선택된 정보가 개인의 전체 사정을 대표한다고 보장하지 않는다.
- 금연 Stage 1/2의 실제 개인 블록은 생활 서술을 자르거나 첫 줄을 버리지 않으며 취미·직업·통근·위치·자원 정보를 같은 명시적 필드 목록으로 전달한다. 미기록과 실제 0을 구분하고 집 체류시간을 재택근무로 바꾸지 않는다. 길이가 늘어난 입력은 정확한 토큰 검사를 통과해야 한다.
- Stage 2의 가격은 모형 참고치로 표시하고, 예측 만족도와 찬반을 구분한다. Night는 근거가 약하면 `기타`를 허용하며 실제 대화가 있었다고 주장하지 않는다.
- Stage 2에는 기존 **유료 방문과 양수 지출 제안** 제약이 남아 있다. 중립적인 문구만으로 구조적 행동 편향이 없다고 결론 낼 수 없다. 실제 정산에서 생기는 예산 보정·대체 선택은 별도 진단 정보이며 시민의 선호나 경험으로 바꾸지 않는다.
- `experience.prompt_block`의 거래 영수증 관측·입장 계약은 별도의 입력 경로로 소스 목록에 포함한다. 금연 규제에 대한 찬반 측정 가능 여부는 관측/인터뷰 분석 계약으로 별도 검증해야 한다. 혜택 사용이나 만족도를 곧바로 찬반으로 바꾸면 안 된다.

## 재현

```powershell
python scripts/experiments/audit_simulation_prompts.py --out output/no_smoking_zone/prompt_audit_next --tokenizer-path output/no_smoking_zone/tokenizer_exaone45_awq
python scripts/experiments/audit_simulation_prompts.py --verify output/no_smoking_zone/prompt_audit_v3_reasoning/manifest.json
python -m pytest tests/unit/sim/test_prompt_audit.py -q
```

`--tokenizer-path`를 생략하면 토큰 예산은 **미검증**으로 기록한다. 이 도구는 네트워크·DB·LLM을 호출하지 않는다. Stage 2/Night 실제 renderer 실행에는 로컬 Python 의존성이 필요하며 연결 시도는 차단한다. prompt 모듈은 본문을 합성하는 로컬 정의만 import하고, 실행 시 외부 호출이 있는 probe/검증 스크립트는 AST로만 읽는다.

산출물: `manifest.json`, `callsites.json`, `prompt_catalogue.json`, `active_surfaces.json`, `source_inventory.json`, `fixtures.json`, `checks.json`, `token_budget.json`, `report.md`. 다시 실행하면 해당 출력 폴더의 감사 파일을 갱신한다. `--verify`는 변경·삭제·추가된 시뮬레이션 소스를 탐지하므로 이후 코드 수정 뒤에는 재감사해야 한다.

## 실제 확인 범위

모든 `scripts/sim/**/*.py`, `scripts/experiments/collect_policy_stances.py`, `scripts/experiments/probe_stance_reasoning.py`를 구문 분석하여 생성 API와 prompt 표현/renderer 후보를 목록화한다. 동적 Python 호출 그래프 전체를 증명하는 도구는 아니며 HTTP 후보도 별도 기록한다. 금연 활성 본문은 사람의 의미 검토를 거쳤다. 최신 렌더 검사는 Stage 1 26개, Stage 2 6개, Night 4개, 인터뷰 10개로 총 46개의 **합성 사례**다. 이 수에는 같은 흡연 상태·다른 직업/시간/취향/자원의 행동 입력 4개와 찬반 비교·정보 부족 probe 4개가 포함된다. Stage 1 개인 비교는 실제 Dawn 개인 포매터, Stage 2는 실제 후보 입력 빌더, 찬반 probe는 실제 토큰 기반 선택기를 거쳐 렌더한다. 관련 개인 정보가 선택 후 입력에 남아 있는지 확인하며 특정 찬반 답변을 정답으로 두지 않는다. 모델 인터뷰는 오프라인 감사에서 실행하지 않는다.

공식 `LGAI-EXAONE/EXAONE-4.5-33B-AWQ` revision `31e6a965d0661bbe4a8b895e22a77f8271772ba0`의 tokenizer JSON과 chat template만 사용한다. 세 파일의 SHA256을 검사하고 `enable_thinking=False` 텍스트 입력을 토큰화한다. 실행과 같은 `prompt_budget.load_tokenizer`를 사용한다. 현재 로컬 Transformers 4.57.6에는 새 `TokenizersBackend` 자동 등록이 없어 같은 serialized tokenizer를 `PreTrainedTokenizerFast`로 직접 로딩한다. 서버의 Transformers 5.8.0/GPU 실행을 검증했다는 뜻은 아니다.

아래는 최신 v3 감사의 측정값이다. 찬반 계약의 출력 예약은 `collect_policy_stances.MAX_OUTPUT_TOKENS`(현재 1600)을 참조한다.

| 합성 사례 | 최대 입력 토큰 | 출력 예약 | 128토큰 여유 포함 합계 |
|---|---:|---:|---:|
| Stage 1 | 1708 | 2200 | 4036 |
| Stage 2 | 1572 | 2400 | 4100 |
| Night | 815 | 900 | 1843 |
| 찬반 수집 | 3081 | 1600 | 4809 |
| 인용 구조화 인터뷰 | 884 | 800 | 1812 |
| 기존 자유형 인터뷰 | 587 | 400 | 1115 |

모두 8192 이내지만 실제 7,500명의 기억·후보 수·재시도 입력의 최대 길이를 보장하지 않는다. 실행 runner는 `SIM_PROMPT_TOKEN_GUARD=required`를 고정하고 실제 HTTP 요청 직전에 정확한 입력 토큰+출력 예약+128 여유를 검사한다. 초과 요청은 모델 호출 전에 실패한다. 찬반 수집은 별도 전체 증거 원본을 보존하면서 들어갈 수 있는 항목 전체만 고르고 누락 개수를 공개한다. 사전학습으로 기억한 정책 지식, 실제 EXAONE의 형식 준수·해석 편향·정책 반응은 GPU 파일럿과 별도 누출 검증이 남아 있다.

## 논리적 설명의 품질은 별도 평가

정확한 인용이나 풍부한 필드가 있다는 것만으로 좋은 설명은 아니다. 실제 GPU 응답은 (1) 구체적 자기 상황의 사용, (2) 그 상황이 선택·정책 고려사항에 왜 관련되는지, (3) 실제 상충이 있을 때의 우선순위, (4) 관측·예상·가치 판단 구분, (5) 관련 조건과 모르는 점, (6) 사실을 만들어내거나 인구학적 라벨로 결론을 정하지 않는지를 검토해야 한다. 내부 사고과정을 수집하거나 문장이 길다는 이유로 높은 점수를 주지 않는다. 현재 오프라인 감사는 입력 전달·출처 일치·형식·토큰 한도를 확인하며 이러한 의미적·논리적 품질을 자동으로 검증했다고 주장하지 않는다.

토크나이저 출처: [LG 공식 고정 revision 파일](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B-AWQ/tree/31e6a965d0661bbe4a8b895e22a77f8271772ba0).
