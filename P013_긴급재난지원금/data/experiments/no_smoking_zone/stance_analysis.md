# 정책 찬반·군집 분석 계약 v2 (기존 v1 읽기 지원)

이 분석의 대상은 **시뮬레이션 에이전트가 시점 제한 인터뷰에서 명시적으로 표현한 입장**이다. 실제 시민의 관측 태도, 숨겨진 내적 동기, 정책의 인과효과를 뜻하지 않는다. 흡연자라는 이유로 반대로, 방문·매출이 늘었다는 이유로 찬성으로 분류하지 않는다. 현재는 구현과 합성 사례 검증을 완료했으며, 실제 GPU 인터뷰·실증 찬반 정확도 결과는 없다.

## 수집과 근거

`scripts/experiments/collect_policy_stances.py`가 완료된 실행에 대해 사후 인터뷰를 수집한다. 인터뷰를 시뮬레이션 상태나 후속 행동에 다시 넣지 않는다. 질문 본문·ID·SHA256은 이 파일의 `QUESTION`, `QUESTION_ID`, `QUESTION_SHA256`으로 고정한다. 정책의 시설 범위와 규칙만 제시하며 연구 PDF의 효과값은 제시하지 않는다.

권장 측정일은 시행 전 마지막 날 **2017-12-02**, 시행 후 마지막 날 **2017-12-16**이다. OFF/ON 각각에서 같은 질문을 사용한다. OFF 전·후와 ON 전은 `hypothetical`, ON 후는 `experienced`로 기록한다. `experienced`는 시행 후 맥락이라는 뜻이며 개인이 해당 시설을 실제 이용했다는 증거는 아니다. 이 맥락을 합쳐 하나의 찬반 비율로 만들지 않는다.

근거 패킷은 `scripts/sim/interview_evidence.py`가 해당 에이전트·실행·실험 팔의 커밋된 기록에서 만든다. 지정일 이후 근거를 사용할 수 없다. 실행 영수증, 당시 모형의 공개 설명, 대체 처리 진단, 모형이 생성한 상호작용은 출처를 구분한다. 공개 설명은 검증되지 않은 주관적 진술이며 실제 녹취나 숨겨진 사고과정으로 해석하지 않는다. 인터뷰는 본인의 상황과 정책 입장을 연결하는 공개 설명을 요청하며, 상세한 내부 사고과정을 요청하거나 저장하지 않는다.

수집기는 전체 패킷과 입력 길이를 제한한 선택 패킷, 실제 요청·공개 응답, 호출 ID·해시를 보존한다. 선택되지 않은 근거와 누락된 날짜는 사건이 없었다는 뜻이 아니다. 분석 결과의 `evidence_coverage`에도 누락 날짜·밤 기록과 선택 내역을 남긴다. 에러나 인용 검증 실패를 찬반으로 고쳐 채우지 않는다.

## 자기 상황에 근거한 공개 설명 v2

새 수집은 `schema_version=2`, `provenance.argument_contract_version=2`를 사용한다. 기존 찬반·답변·확신도·대표 인용에 다음 `response.argument`를 추가한다.

| 필드 | 공개 설명의 역할 | 상한 |
| --- | --- | --- |
| `personal_situation` | `{claim, evidence:[{evidence_id, quote}]}`: 입장에 관련된 자신의 상황과 출처 | 3개 |
| `considerations` | `{claim, basis, direction, personal_relevance, evidence}`: 고려사항과 개인에게 관련되는 이유 | 4개 |
| `weighing` | 무엇을 더 중요하게 보아 현재 입장을 택했는지 설명 | 1,600자 |
| `conditions` | `{condition, possible_change}`: 조건이 달라지면 입장을 어떻게 재검토할 수 있는지 | 2개 |
| `uncertainties` | 현재 알 수 없거나 자료가 부족한 점 | 3개 |

`basis`는 `recorded_fact`, `inference`, `value_judgment`를 구분한다. `recorded_fact`도 해당 기록이 현실에서 관측되었다는 인증이 아니라, 제공된 모형 기록에 근거한다는 **응답자의 선언**이다. `direction`은 고려사항이 정책에 유리한지(`for`), 불리한지(`against`), 판단하기 어려운지(`uncertain`)를 표현한다. 이 필드로 최종 찬반을 자동 결정하지 않는다.

예를 들어 흡연자라는 자기 기록이 있어도 “이동 불편은 있을 수 있지만 실내의 쾌적함을 더 중요하게 보아 찬성한다”는 입장이 가능하다. 반대로 관련 불편을 더 중요하게 보는 입장도 가능하다. 어느 쪽이든 흡연 라벨 자체를 찬반 정답으로 사용하지 않는다. 실제 방문 자료가 없다면 경험을 지어내지 않고, 가치판단이나 조건부 입장·판단 유보를 표현할 수 있다.

`scripts/sim/stance_argument.py`의 `validate_argument(response, packet)`는 정확한 필드·자료형, 각 배열과 문자열의 상한, 같은 배열 안의 중복 항목, 모든 인용의 정확한 일치를 검사한다. 별도 역할 없이 추가한 키는 거부한다. 개별 근거 배열은 3개, 대표 `reasons`는 6개까지이며, claim·개인적 관련성은 각각 800자, 조건·변화·불확실성은 각각 600자, 인용은 1,000자, 전체 answer는 4,000자까지다. 이 상한은 입력 계약이며 문장의 품질 점수가 아니다.

상황·고려사항·판단 기준이 비어 있거나, `recorded_fact`인데 인용이 없거나, 고려사항의 개인적 관련성이 비어 있으면 재현 가능한 `quality_flags`를 남긴다. 이때 스키마가 유효한 응답은 `answered`와 원래 입장을 유지한다. 응답·주장 길이의 최소값, 문장 수, 찬반 양쪽을 모두 쓰는지 여부로 점수를 주지 않는다. 조건이나 불확실성 항목도 실제로 말할 내용이 없으면 빈 배열이 가능하다. 없는 장단점이나 경험을 강제로 채우지 않는다.

`quality_status=needs_enrichment`는 빠진 공개 연결을 검토하라는 표시이며, 부실 판정·재생성 명령·실패 상태가 아니다. `input_evidence_count`, 근거 종류별 개수, `limited_input_possible`도 함께 남긴다. 자료가 없거나 선택 과정에서 빠졌을 수 있는 경우, 짧은 판단 유보도 정상적인 응답이다. `sufficient_for_review` 역시 논리가 맞다는 판정이 아니다. 모든 v2 결과는 `review_required=true`이며, 인용이 주장을 실제로 뒷받침하는지, 자기 상황과 관련되는지, 판단 기준과 입장이 일관되는지는 별도 의미 검토 대상이다. 정확한 인용 뒤에 근거 없는 일반화가 이어져도 문자열 검증만으로 “논리 통과”를 선언할 수 없다.

수집기가 저장한 `argument_quality`는 분석기가 다시 계산해 일치 여부를 확인한다. 품질 표시별 원래 찬반 구성과 전체 분모도 보고하며, 품질 표시가 있는 응답을 삭제하거나 중립으로 바꾸지 않는다. 자동 보완·재생성을 기본 동작으로 하지 않는다.

설계의 `measurement_contract={"record_schema_version":2,"argument_contract_version":2}`를 동결한다. 이 필드가 없는 과거 설계는 v1로 읽으며 같은 설계에 v1/v2 응답을 섞을 수 없다. 새 `make-design`은 수집기의 v2 질문·버전 상수를 사용한다. v1 자료를 v2로 재명명하거나 기존 군집 모델을 새 응답 방식에 그대로 사용하지 않는다.

## 입력과 판정

기계 판독 구조는 `stance_record.schema.json`, 추가 검증은 `analyze_policy_stance.validate_record(record, design=None)`에 있다. 실제 수집 결과는 `records/*.json`의 개별 봉인 JSON이다. 분석기는 이 디렉터리 또는 수집 루트, JSON 배열, JSONL을 읽는다. 잘못된 입력을 조용히 건너뛰지 않는다.

핵심 필드는 다음과 같다.

| 필드 | 의미 |
| --- | --- |
| `run_id`, `arm`, `agent_id`, `as_of_day`, `period` | 실행·에이전트·측정 시점 식별 |
| `question_id`, `question_sha256`, `measurement_context` | 같은 질문 및 가정적/시행 후 응답 구분 |
| `response_status` | `answered`, `no_response`, `error` |
| `response.stance` | 명시적 자기 응답의 `support`, `oppose`, `mixed`, `neutral`, `uncertain`, `unknown` |
| `response.answer`, `stance_quote` | 공개 답변과 그 안에 정확히 존재하는 입장 인용 |
| `response.confidence` | 0–1 자기 보고 확신도, 검증된 확률이 아님 |
| `response.reasons` | `{evidence_id, quote}` 배열; 인용문은 패킷 `text`의 정확한 부분문자열 |
| `evidence_packet` | 봉인된 시점 제한 패킷·원천/코호트 해시 |
| `provenance` | 모델·호출·요청·응답 해시, `source=structured_policy_feedback`, 합성 테스트 표시 |

경험을 주장하지 않는 가치판단이나 정보 부족 응답은 `reasons=[]`일 수 있다. 정확한 문자열 인용과 SHA 검사는 출처 연결 및 변경 여부를 검사하지만, 답변의 의미적 정확성·진실성·독립성을 보증하지 않는다. 자기 보고 라벨과 인용이 의미상 일치하는지는 별도 독립 주석 평가 대상이다.

`derive_stance(record, min_confidence=0.6)`는 아래를 구분한다.

| 결과 | 기준 |
| --- | --- |
| `support` / `oppose` / `mixed` / `neutral` | 충분한 자기 보고 확신도로 명시된 입장 |
| `uncertain` | 판단을 유보한다고 명시한 응답 |
| `unknown` | 잘 모른다는 명시 응답 또는 수집·검증 에러; `reason`으로 구분 |
| `insufficient_evidence` | 응답 기록 없음, 무응답, 설정 기준보다 낮은 확신도 |

낮은 확신도의 원래 라벨도 `reported_stance`로 보존한다. 기본 0.6은 검증된 분류 경계가 아닌 진단 설정이다. 정답 평가를 본 뒤 이 값을 최적화하면 같은 정답을 평가에 재사용할 수 없다. 중립은 누락의 대체값이 아니다.

## 실제 비지도 군집과 집단 요약

군집 입력은 답변 텍스트와 정확히 인용한 근거 텍스트이다. v2에서는 공개된 개인 상황, 고려사항·개인적 관련성, 판단 기준, 조건·가능한 변화, 불확실성도 사용한다. 동일한 텍스트 조각은 한 번만 포함한다. 단어·한국어 문자 3-gram TF-IDF, L2 정규화, 코사인 거리 기반 spherical k-means를 표준 라이브러리로 구현했다. 기본값은 `k=4`, seed `17001`, 최소 문서 빈도 `2`, 최대 특징 `1024`, 최대 반복 `40`이다. 첫 중심은 seed/응답 ID 해시, 이후 중심은 가장 먼 문서로 고르며 동점 순서도 고정한다. 같은 입력의 순서 변경에 대해 재현된다.

**OFF/시행 전 응답만** 어휘·IDF·중심을 학습한다. 모델 파일을 동결한 후 나머지 팔·시점을 같은 중심에 할당한다. 시행 후 응답 추가로 학습 모델을 바꿀 수 없다. 두 개 미만의 학습 문서에는 군집 생성을 보류한다. 어휘가 겹치지 않거나 최대 코사인 유사도가 기본 0.1보다 작으면 미할당으로 남긴다. 기본 `k`·유사도 기준은 실제 사회 집단 수나 검증된 경계가 아니다. 군집 수렴 여부·학습 거리·군집 크기와 미할당 비율을 함께 확인한다.

명시적 찬반 라벨·확신도·`direction`·`basis`·품질 표시 필드, 원시 인구 속성·흡연 라벨 열, 연구 정답은 군집 특징으로 넣지 않는다. 단, 본인이 답변이나 인용에서 흡연·나이 등을 언급하면 그 단어는 의미 내용으로 남는다. 따라서 완전한 인구 특성 비식별화나 공정성 보장을 주장하지 않는다. 해당 언급을 특정 찬반에 강제로 대응시키는 규칙은 없다. 판단 유보·낮은 확신도·구조 보완 표시가 있는 실제 텍스트도 군집화할 수 있으며, 군집에 들어갔다고 찬반 상태가 바뀌지 않는다.

각 군집에는 학습 중심의 주요 단어, 대표 인용·응답 ID, 팔/시점별 찬반 구성, Day 0에 고정된 나이·성별·소득·흡연 라벨 구성을 제시한다. 임의로 “반정책 흡연 집단” 같은 이름을 붙이지 않는다. 별도로 미리 정한 속성 집단별 비율도 집계한다. 흡연 라벨은 실측 흡연율에 기반한 합성 배정이며 개인 관측 흡연 여부가 아니다. 이 속성은 **결과를 설명하는 집계에만** 사용한다.

분모는 고정 코호트 전체와 명시적인 네 가지 입장(`support/oppose/mixed/neutral`)이 확인된 응답자를 각각 보고한다. 기본 5명 미만 집단은 작은 집단 표시를 하고 외부 집단 비교를 보류한다. 찬성·반대의 결측 범위는 미해결 상태가 전부 찬성 또는 반대일 수 있는 범위이다. 표본오차 신뢰구간·사람에 대한 예측 확률이 아니다.

한 팔·한 기간에 여러 시점의 응답이 있으면 기간 내 가장 늦은 실제 기록을 사용한다. 같은 날짜의 충돌 기록은 명시적으로 해결하기 전까지 거부한다. OFF/ON, 시행 전/후의 네 셀과 에이전트별 전→후 상태 전이를 보존하며, 없는 전 응답을 후 응답으로 채우지 않는다. 비교 가능한 군집을 위해 후 자료로 재학습하지 않는다.

## 실행 예시

아래 `run_off`, `run_on`은 완료된 28일 실행 폴더 자리표시자다. 아직 실행하지 않은 결과가 존재한다고 가정하지 않는다. 30명×1일 처리량 파일럿은 전후 태도 분석에 사용할 수 없다. 30명×28일 진단 실행은 같은 인터페이스를 사용할 수 있지만 7,500명 결과로 보고하지 않는다.

```sh
python scripts/experiments/collect_policy_stances.py --run-dir output/no_smoking_zone/run_off --as-of 2017-12-02 --out output/no_smoking_zone/stance/off_pre
```

수집 명령은 기본적으로 계획만 출력한다. 준비된 LG 서버로 실제 호출할 때 `--execute`를 추가한다. OFF 후/ON 전/ON 후도 각각 다른 출력 폴더로 수집한다. `--limit`는 명시적인 진단 부분집합이며, 분석에서 나머지 고정 코호트는 결측으로 남는다. 7,500명×2팔×2시점이면 별도 인터뷰 **30,000건**이 필요하므로 실제 수집 계획에 추가 호출량을 반영해야 한다.

```sh
python scripts/experiments/analyze_policy_stance.py make-design --bundle output/no_smoking_zone/full_7500_v1 --off-run-dir output/no_smoking_zone/run_off --on-run-dir output/no_smoking_zone/run_on --out output/no_smoking_zone/stance/design.json
python scripts/experiments/analyze_policy_stance.py fit --design output/no_smoking_zone/stance/design.json --records output/no_smoking_zone/stance/off_pre --k 4 --seed 17001 --out output/no_smoking_zone/stance/cluster_model.json
python scripts/experiments/analyze_policy_stance.py analyze --design output/no_smoking_zone/stance/design.json --model output/no_smoking_zone/stance/cluster_model.json --records output/no_smoking_zone/stance/off_pre output/no_smoking_zone/stance/off_post output/no_smoking_zone/stance/on_pre output/no_smoking_zone/stance/on_post --out output/no_smoking_zone/stance/report.json
python scripts/experiments/analyze_policy_stance.py evaluate --report output/no_smoking_zone/stance/report.json --out output/no_smoking_zone/stance/evaluation.json
```

`make-design`은 고정 번들, 두 실행의 코호트·기간·완료 상태, 첫날 커밋 패킷의 원래 실행 ID·원천 해시를 확인한다. 폴더를 옮겨도 실행 ID를 새 경로로 추정하지 않는다. 이후 각 응답의 실제 패킷도 이 설계와 대조한다. 기존 분석 출력은 덮어쓰지 않는다. 다른 독립 seed는 별도 설계·분석으로 실행하고 동일 인물을 서로 다른 관측자로 합쳐 분모를 늘리지 않는다.

## 정답 비교 인터페이스

현재 `ground_truth.json`의 매출 회귀계수, 카드 거래, 업주·종사자의 매출 감소 예상/체감, PM 측정치는 **찬반 라벨이 아니다**. 예를 들어 당구장 매출 +13.54%를 찬성률로, 매출 감소 응답을 정책 반대로 바꾸지 않는다. `evaluate`는 해당 파일이나 독립 정답이 없는 경우 `status=abstained`, `metrics=null`을 반환한다. 기존 경제·노출 검증과 태도 평가를 분리한다.

독립 정답을 공급할 때 `--truth`에 JSON을 지정한다. 공통 계약은 다음과 같다.

```json
{
  "kind": "individual_stance_labels",
  "independent": true,
  "partition_role": "holdout",
  "synthetic_fixture": false,
  "source": {
    "id": "독립 주석 자료 식별자",
    "sha256": "원천 파일의 64자리 SHA256",
    "annotation_or_measurement_protocol": "주석자·질문·블라인드 절차·판정 규칙"
  },
  "estimand": "report.json의 estimand와 같은 구조의 객체",
  "label_basis": "independent_annotation_of_same_responses",
  "classes": ["support", "oppose", "mixed", "neutral"],
  "labels": []
}
```

위 해시·`estimand` 문자열은 구조 설명용 자리표시자이며 검증을 통과하는 실제 데이터가 아니다. `labels`의 각 행은 `agent_id`, `arm`, `period`, `record_id`, `measurement_context`, `stance`를 가져야 한다. 독립 주석은 분석에 사용한 **그 응답**을 가리켜야 한다. `estimand`는 질문·대상/역할·정책·기간·단위·가중방식이 모두 일치해야 한다. 독립성은 공급자의 선언과 절차를 확인해야 하며 해시만으로 증명되지 않는다.

개별 평가는 macro-F1, balanced accuracy, accuracy, 혼동행렬, 클래스별 support/precision/recall/F1, 분류 범위를 낸다. 혼동행렬 열에는 미해결 상태도 포함하며 실질적 입장 정답에서의 기권을 누락시키지 않고 오류로 센다. macro-F1은 선언된 클래스 전체, balanced accuracy는 정답 표본이 있는 클래스의 recall 평균이며 0으로 나누는 값은 0이다. 이는 **생성된 응답에 대한 추출/주석 일치도**로서, 실제 인간의 찬반 예측 정확도가 아니다.

집단 통계는 `kind=group_stance_statistics`와 `groups`를 사용한다. 각 행은 `arm`, `period`, `group_by`, `group`, `measurement_context`, `denominator`, `n`, `counts`를 포함한다. `all_expected` 분모는 일곱 상태 전체, `resolved_responses`는 네 가지 명시적 입장 전체의 정수 count를 요구하고 합이 `n`과 같아야 한다. 같은 측정대상·질문·분모의 집단에만 구성비 차이, total variation distance, 평균 절대 구성비 오차를 낸다. 연구의 시설 업주/종사자를 합성 거주자와 같은 집단으로 재명명해서 비교할 수 없다.

검증 사례는 `tests/fixtures/policy_stance/` 및 `tests/unit/sim/test_policy_stance_analysis.py`에 있다. 학습용 문장과 별도 시행 후 검증 문장을 나누며, 입력 순서·메타데이터·선언 라벨 변경 불변성, 후 자료 누출 방지, 정확한 인용, 미응답 분모, 독립 정답 불일치 거부를 검사한다. 합성 사례의 점수는 소프트웨어 검증 기대값이며 실증 정확도가 아니다.
