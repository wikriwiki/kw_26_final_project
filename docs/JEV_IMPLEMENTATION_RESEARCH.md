# Jev의 가속 아이디어와 EXAONE 구현에 적용할 범위

확인일: 2026-09-22. 이 문서는 공개 원리를 검토한 연구 기록이며, 우리 모델의 속도·예측 품질 검증 결과가 아니다. GPU 연결 전에는 코드와 CPU에서 실행 가능한 계약·수치 검증을 준비한다. 모델 가중치 다운로드와 학습은 별도 실행 단계다.

## 1. 빠른 이유: 답변 문장을 만들지 않는다

Jev는 문장이나 JSON을 토큰 단위로 생성하는 대신, 미리 정한 선택지에 대한 확률과 구조화된 답을 반환한다. TypeSafe는 병렬 출력과 RLCD(Reinforcement Learning for Calibrated Decisions)를 설명하지만, 공개 자료만으로 내부 학습법·모델 구조 전체를 재현할 수는 없다. 공식 발표의 70–500ms와 40–200배 수치는 제공자의 특정 비교 결과이며 우리 시뮬레이션에서 보장되는 수치가 아니다. 발표도 짧은 입력이 시연에 유리했다고 밝힌다. [TypeSafe 공식 발표](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [RLCD 설명](https://docs.typesafe.ai/introduction/machine-learning-primer)

우리 구현에서 가져올 수 있는 핵심은 다음과 같다.

1. 상태와 후보를 읽는다.
2. EXAONE을 한 번 순전파한다.
3. 마지막 위치에서 선택 코드의 점수만 읽는다.
4. 코드가 확률·선택 결과를 구성한다. 설명문·JSON 생성과 파싱은 생략한다.

이는 **EXAONE 기반 제한 선택 분류기**이며 Jev의 비공개 모델을 복제했다는 의미가 아니다. 입력을 읽는 연산은 남으므로, 긴 기억·후보 목록에서는 입력 처리 시간이 병목이 될 수 있다.

## 2. 참고 구현에서 가져올 아이디어

| 자료 | 확인한 아이디어 | 우리 적용 시 주의점 |
|---|---|---|
| [awesome-jev](https://github.com/tanxarx/awesome-jev) | 관련 공개 구현을 찾는 목록 | 목록의 성능 소개만 채택하지 않고 원본 구현의 조건을 확인한다. |
| [SemIf](https://github.com/TheoLeeCJ/SemIf) | 직접 선택지 점수 읽기, 동일 상태의 prefix 재사용, 질문 suffix 병렬 처리 | 동일 모델 비교에서도 생성 답변과 직접 점수가 항상 일치하지 않았다. 캐시·BF16 경로도 일부 선택이 달라졌다. 우리 데이터로 검증한다. |
| [openjev-sglang](https://github.com/ekzhang/openjev-sglang) | 단일 토큰 선택 코드 검증, 선택지 logprob 정규화, 공통 prefix 재사용 | 일반 API의 top-k logprob가 모든 후보를 포함한다고 가정하지 않는다. 엔트로피 기반 confidence는 정답 확률이 아니다. |
| [Bespoke Nimble](https://github.com/bespokelabsai/nimble) | 허용 후보 점수에 직접 교차엔트로피 학습, LoRA, 사실 하나를 바꾼 대응 사례 | 공개 예제는 Qwen 기반이다. LG 기반 모델에는 학습 원리를 별도 구현한다. 합성 라벨을 현실 정답으로 취급하지 않는다. |
| [Decider](https://github.com/Mapika/decider) | 선택 슬롯 점수화, 상태 또는 고정 질문 prefix 캐시 | schema-first 입력은 해당 프로젝트에서도 정확도 손실이 있었다. 단순 프롬프트 재배열도 품질 변경으로 평가한다. |

위 자료의 숫자는 서로 다른 모델·하드웨어·문제에서 측정됐다. 서로의 속도 숫자를 우리 EXAONE 1.2B의 예상치로 옮기지 않는다.

## 3. EXAONE에서 가장 먼저 구현할 추론 경로

[LG 공식 EXAONE 4.0 1.2B 모델 카드](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B)는 Transformers 4.54.0 이상과 기본 비추론 채팅 형식을 안내한다. 원본 모델과 토크나이저 revision을 함께 고정한다. GPU 연결 전에는 이미 있는 로컬 가중치만 명시적으로 읽거나 가중치 없이 코드 계약을 검사하고, 서버 시작만으로 다운로드하지 않게 한다.

가장 단순한 기준 구현은 `model.generate()` 대신 `model(..., logits_to_keep=1)`로 마지막 위치 점수를 얻는 것이다. EXAONE의 공식 Transformers 구현에 이 인자가 있다. 따라서 긴 입력 전체 위치의 어휘 점수를 보관할 필요가 없다. [Transformers 4.54 EXAONE 구현](https://github.com/huggingface/transformers/blob/v4.54.0/src/transformers/models/exaone4/modeling_exaone4.py)

추가 최적화로 마지막 hidden state에 필요한 선택 코드의 출력층 행만 곱할 수 있다. 다만 LoRA·출력층 변경·정밀도에 따라 기준 구현과 같은지 먼저 검사해야 하므로, 첫 기준선에서는 마지막 위치의 전체 점수를 얻고 필요한 후보만 선택하는 방식이 단순하다.

### 선택 코드 검증

`A`, `B`처럼 보이는 문자열도 토크나이저·앞 공백·출력 문맥에 따라 토큰이 달라질 수 있다. 후보 ID 자체를 점수화하지 않고 별도 선택 코드를 사용한다.

- `encode(code, add_special_tokens=False)`가 정확히 한 토큰인지 확인한다.
- **실제 완성된 프롬프트 뒤에 코드를 붙여 재토큰화**한다. 프롬프트 토큰이 그대로 보존되고, 검증한 한 토큰만 추가되는지 확인한다.
- 코드끼리 토큰 ID가 서로 다르고 특수 토큰이 아닌지 확인한다.
- 두 자리 숫자나 다중 토큰 이름의 첫 토큰만 읽는 근사 방식은 거부한다.
- 검증에 실패하거나 후보 수가 지원 범위를 넘으면 기존 EXAONE 경로로 넘긴다.

이 검사는 모델·토크나이저·템플릿 revision과 실제 프롬프트 접미사에 연결한다. 시작 시 독립 문자열 검증만 하고 실제 문맥 검사를 생략하지 않는다.

### 배치와 캐시

- `eval()`과 추론 모드를 적용한다. 다른 길이의 입력을 배치할 때 attention mask와 position IDs를 맞춘다.
- 마지막 위치를 읽는 구현은 왼쪽 패딩을 사용하거나 각 행의 실제 마지막 토큰을 명시적으로 찾아야 한다. 오른쪽 패딩 위치의 점수를 읽으면 안 된다.
- 먼저 캐시 없는 기준선에서 단일 요청과 배치 요청의 점수·선택을 비교한다.
- prefix 캐시는 동일 토큰 prefix일 때만 사용한다. 서로 다른 사람의 상태·정책·지갑·기억을 같은 것으로 취급하지 않는다.
- branch별 KV 상태가 서로 변경되지 않도록 격리한다. 캐시 추가 전후의 점수·선택·메모리 사용을 측정한다.
- 여러 독립 에이전트는 배치할 수 있지만, 앞선 소비·방문이 다음 결정의 입력이 되는 이벤트를 독립 질문으로 병렬화하지 않는다.

## 4. 장소·지출·만족도의 관계를 보존한다

Jev형 API의 독립 질문은 같은 상태를 각각 평가한다. 질문을 여러 개 보낸다고 한 질문이 다른 질문의 답을 자동으로 반영하는 것은 아니다. [TypeSafe 공식 인터페이스 설명](https://docs.typesafe.ai/introduction)

우리의 장소·소비액·만족도는 서로 연결된다. 따라서 다음 중 하나로 관계를 표현한다.

- 장소를 선택한 다음, 선택 장소와 앞선 지출을 조건으로 소비·만족도 등을 판단한다.
- 관계가 일관된 전체 결정 묶음을 선택지로 구성한다. 후보 묶음이 너무 커지면 기존 모델로 넘긴다.

어느 방식이든 원래 Stage 2 계약과 공통 회계·기억·감정 갱신을 유지한다. 만족도를 상수로 채우거나 선택지 점수에서 긍정 감정을 임의로 만들어내지 않는다. 설명문이 후속 기억·인터뷰에 쓰이면, 구조화된 근거 참조로 충분한지 검증하고 불충분한 경우 큰 모델로 넘긴다.

## 5. 확률을 세 가지로 구분한다

| 값 | 의미 | 사용 범위 |
|---|---|---|
| 후보 점수 정규화 결과 | 제공된 선택지 사이에서 모델이 상대적으로 부여한 점수 | 직접 점수 기준선과 분석용 |
| 품질 위험·보류 점수 | 작은 모델 결과를 채택하면 품질이 나빠질 위험 | 별도 검증 자료로 보정한 라우팅용 |
| 행동 확률 | 실제 사람·집단이 방문하거나 소비할 분포 | 관측 자료 또는 검증된 시뮬레이션 분포에 맞춘 샘플링용 |

후보 softmax가 0.9라고 정답률이나 실제 방문 확률이 90%인 것은 아니다. 선택지 밖 결과의 가능성이 정규화 과정에서 사라질 수 있으므로 `DEFER` 등 보류 경로와 사전 위험 검사를 둔다. 비슷하게 좋은 식당이 여러 개인 상황은 정상적인 다양성일 수 있으므로 낮은 최대 확률만으로 실패라 판정하지 않는다. 확률 보정은 사용 문제의 별도 자료에서 수행한다. [SemIf 확률 보정 문서](https://github.com/TheoLeeCJ/SemIf/blob/master/docs/CALIBRATION.md), [TypeSafe confidence 설명](https://docs.typesafe.ai/confidence)

학습·보정·최종 평가는 분리한다. 같은 인물·같은 시나리오의 대응 사례가 경계를 넘지 않게 묶어서 나눈다. 교사 EXAONE 일치율과 현실 방문·감정 예측력은 다른 지표다. 최종 채택 조건은 기존 [가속 설계도](SIMULATION_ACCELERATION_DESIGN.md)의 장기 분기 실험과 비열등성 평가를 따른다.

## 6. 출처와 라이선스 기록

SemIf 코드는 [MIT](https://github.com/TheoLeeCJ/SemIf/blob/master/LICENSE), Decider 코드는 [Apache-2.0](https://github.com/Mapika/decider/blob/main/LICENSE), awesome-jev 목록은 [CC0](https://github.com/tanxarx/awesome-jev/blob/main/LICENSE)로 표시되어 있다. openjev-sglang와 Nimble의 확인한 루트 목록에서는 별도 LICENSE 파일을 찾지 못했으므로, 원리를 참고하는 것과 소스 복사를 구분한다.

LG 가중치에는 별도 [EXAONE 라이선스](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/blob/main/LICENSE)가 적용된다. 파생 모델의 이름은 EXAONE으로 시작하도록 요구하며 대회 참여·배포에도 조건이 있다. 이 문서는 대회 규정 충족을 판정하지 않는다. 모델 이름·revision·데이터 출처·학습 설정을 기록하고 제출 전 대회 규정과 해당 라이선스를 함께 확인한다.
