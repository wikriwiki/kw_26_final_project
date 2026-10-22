# EXAONE 판단 모델: GPU 연결 전 준비된 구현

**개발 목표는 EXAONE을 기반으로 Jev 방식의 판단 모델 자체를 학습하는 것입니다.** 새로 추가한 `decision_model.py`는 EXAONE 본체에 전용 판단 출력층을 붙인 신경망입니다. 본체를 이벤트당 한 번 실행하고, 작은 출력층에서 앞선 선택을 반영해 장소·지출·만족도 등을 결정합니다. `decision_training`이 본체 LoRA와 출력층을 함께 학습하고, `decision_inference`가 저장된 모델을 오프라인 재실행합니다.

**새 모델의 구조·현재 상태·실행 명령은 [EXAONE 판단 모델 안내](../../../docs/EXAONE_DECISION_MODEL.md)를 먼저 읽으세요.** 실제 EXAONE 학습 가중치는 GPU와 실제 자료가 없어 아직 만들지 않았습니다. CPU 무작위 모델 검사는 학습 완료를 뜻하지 않습니다.

아래 내용은 **이전 선택 코드 방식의 비교 기준**과 두 모델이 공유하는 자료 수집·평가 안내입니다. 이전 `training`/`replay`와 새 `decision_training`/`decision_inference`의 체크포인트는 서로 호환되지 않습니다.

이전 비교용 모듈은 **EXAONE 4.0 1.2B의 선택지 점수를 직접 읽는 분류기**입니다. 문장·JSON을 생성하지 않습니다. Jev의 공개된 아이디어를 적용했으며 Jev의 비공개 RLCD를 재현한 모델은 아닙니다.

현재 단계는 데이터 수집·학습·병행 평가를 위한 구현입니다. **실제 EXAONE 가중치 학습, 현실 예측력 유지, 전체 시뮬레이션 속도 개선은 아직 검증되지 않았습니다.** GPU를 사용하지 않고 CPU의 작은 무작위 EXAONE으로 연산과 연결을 검증했습니다. 입력 전처리와 평가 도구는 별도 CPU 측정으로 개선을 확인했습니다.

실제 EXAONE 토크나이저 revision `3abf2810673c7c0778df64a73c2d52eab32d91c4`로 예시의 5개 판단 종류와 최대 42개 선택지의 단일 토큰 조건도 확인했습니다. 토크나이저 파일만 내려받았으며 모델 가중치는 다운로드하지 않았습니다. 실제 학습 자료에서도 문맥별 검사를 다시 수행합니다.

## 구현한 범위

- 실제 문맥에서 한 토큰인 선택 코드만 허용하고, 마지막 입력 위치의 출력층 점수만 계산합니다. 입력을 임의로 자르지 않습니다.
- 일상적 선택 가능 여부 → 장소 → 소비액 → 만족도 → 선택 요인 순서로 판단합니다. 앞 단계 선택을 뒤 단계 입력에 포함하고, 여러 거래도 순서대로 처리합니다.
- 긍정 감정이나 소비액을 상수로 채우지 않습니다. `DEFER`나 오류가 하나라도 발생하면 하루 묶음 전체를 보류합니다.
- 정책·부정적 기억·사회적 사건·상태 누락 등은 초기 적용 범위에서 제외합니다. 이 규칙은 검증 범위를 제한하는 임시 조건이며 학습된 정확도 보증이 아닙니다.
- 결정 전 스냅샷, 교사 모델 결과, 보정·리뷰 흔적, 모델 식별 정보, 지연을 기록합니다. 감정·기억·정책 정산 경로는 기존 그대로입니다.
- 교사 오류 보정·리뷰 추가 조회 사례는 일반 학습 정답으로 사용하지 않습니다. 실제 기록은 사람 ID 단위로 train/calibration/test에 나누고 합성 테스트 사례는 따로 둡니다.
- LoRA로 허용 선택지 점수에 대한 교차엔트로피를 학습하고, 별도 calibration 자료로 온도 보정할 수 있습니다. 보정도 현실 방문 확률이나 전체 시뮬레이션 품질을 보장하지 않습니다.

## 현재 의도적으로 활성화하지 않은 것

실제 결정을 바꾸는 `live` 모드는 없습니다. `off`, `record`, `shadow`만 지원합니다. 가중치 없는 상태나 평가 전 어댑터가 실제 시민의 행동·감정을 바꾸지 않게 하기 위함입니다. `shadow`도 기존 큰 모델의 결과만 시뮬레이션에 반영합니다.

실제 학습쌍·GPU가 제공되고 독립 실행과 현실 자료 검증을 통과하면 운영 적용 단계를 추가해야 합니다. 온도 보정 파일이나 교사 일치율만으로 운영 승인을 만들지 않습니다.

## CPU에서 추가 검증한 최적화

- 빠른 토크나이저에서 전체 `질문+선택 코드`를 묶어서 검사합니다. 모든 선택 코드의 실제 문맥 검사는 유지하며, 느린 토크나이저는 기존 개별 검사를 사용합니다.
- 동일 입력의 검증된 토큰만 재사용합니다. 기본 캐시는 최대 128개 질문·합계 65,536개 입력 토큰으로 제한합니다. **선택 결과나 감정 판단을 캐시하지 않습니다.** 토크나이저를 교체하거나 내부 설정을 바꾼 경우 `backend.clear_encoding_cache()`를 호출해야 합니다. `encoding_cache_size=0`으로 재사용을 끌 수 있습니다.
- 한 사람의 하루 판단에서 변하지 않는 상태는 한 번만 JSON으로 변환합니다. 각 단계의 앞선 장소·지출 선택은 계속 반영하며, 변경 전과 동일한 입력 문자열을 만듭니다.
- 로컬 가중치 로딩 실패는 같은 설정에서 60초 동안 재시도를 늦춥니다. 여러 worker가 동일한 실패를 반복하는 비용을 줄이고, 모델 설정을 바꾸면 바로 다시 시도합니다.
- 병행 평가와 학습 자료의 `--dry-run` 검사는 입력을 순차 처리합니다. 평가의 긴 원문과 개별 거래는 누적 보관하지 않습니다. 중복 확인 ID와 지연 측정값 등 일부 작은 자료는 여전히 건수에 비례해 증가합니다.
- 중복 학습 질문·실제 교사 출력 보정은 거부하되, 출력 보정이 아닌 사전 후보 검색 범위 확장은 구별해 유효한 자료를 유지합니다.

서로 다른 합성 질문 35개에서 캐시를 끄고 전처리만 측정한 결과는 58.1 → 19.5ms/질문이었습니다. CPU 한 대에서 7회 교차 측정한 중앙값이며, 전체 모델 추론 가속률은 아닙니다. 조건·검증 결과는 [성능 재점검 기록](../../../docs/EXAONE_SIMDECISION_PERFORMANCE_REVIEW.md)에 정리했습니다.

## 1. 지금 실행 가능한 CPU 확인

프로젝트 루트에서 실행합니다. 출력 경로는 새 실험마다 새 디렉터리를 사용하세요.

```powershell
python -m scripts.sim.fast_decision smoke --output sim_output/fast_decision/example/captures.jsonl
python -m scripts.sim.fast_decision.dataset --input sim_output/fast_decision/example/captures.jsonl --output sim_output/fast_decision/example/dataset
python -m scripts.sim.fast_decision.training --input sim_output/fast_decision/example/dataset/synthetic.jsonl --output sim_output/fast_decision/example/dry-run --allow-synthetic --dry-run
```

이 사례는 사람이 작성한 합성 검사 자료입니다. 모델이 실제로 예측했다는 기록이나 학습 완료 증거가 아닙니다. `--dry-run`은 자료 구조만 검사하며 토크나이저·가중치를 로드하지 않습니다.

## 2. 이후 실제 EXAONE 시뮬레이션에서 학습 자료 수집

큰 EXAONE 서버와 별도 실험용 Neo4j가 준비된 뒤 기존 실행 명령에 다음 환경 설정을 붙입니다. 아래는 이후 실행할 예시이며 이번 작업에서는 실행하지 않았습니다.

```powershell
$env:LLM_MODE = 'exaone_4_5'
$env:SIM_FAST_MODE = 'record'
$env:SIM_FAST_CAPTURE_PATH = 'sim_output/fast_decision/run-001/captures.jsonl'
python scripts/sim/run_simulation.py --start 2026-05-01 --days 3 --limit 100 --workers 8
```

실제 서빙 모델과 `LLM_MODE`가 일치해야 합니다. 대회에서 사용할 버전과 초기 DB 스냅샷도 고정하세요. 기록에 교사 모델 종류를 남기며, Qwen 교사 자료는 기본 학습 데이터에서 제외합니다.

`captures.pending.jsonl`은 교사 호출 전 스냅샷, `captures.jsonl`은 최종 교사 결과까지 있는 완료 기록입니다. 학습에는 완료 기록을 사용합니다. 여러 프로세스가 같은 파일에 동시에 쓰지 않도록 실행마다 별도 경로를 사용합니다. 한 프로세스의 여러 worker는 쓰기 잠금으로 보호됩니다.

스냅샷에는 모델 입력에 쓰인 페르소나와 기억이 담깁니다. 생성 파일은 기존 `sim_output/` 제외 규칙으로 Git에 올라가지 않습니다.

## 3. 실제 자료 분리와 검사

```powershell
python -m scripts.sim.fast_decision.dataset --input sim_output/fast_decision/run-001/captures.jsonl --output sim_output/fast_decision/dataset-001
python -m scripts.sim.fast_decision.training --input sim_output/fast_decision/dataset-001/train.jsonl --output sim_output/fast_decision/check-001 --dry-run
```

`manifest.json`에서 수집·거부 수와 집단 분리를 확인합니다. 자료가 적으면 calibration/test가 비어 있을 수 있습니다. 샘플을 복제해 채우지 말고 더 수집합니다. 같은 사람의 다른 날짜가 학습과 평가에 섞이지 않지만, 이 분리만으로 지역·시간 외삽 검증까지 완료되는 것은 아닙니다.

## 4. GPU 연결 후 실제 학습

GPU 드라이버에 맞는 PyTorch를 별도 환경에 설치하고 이 디렉터리의 `requirements.txt`를 설치합니다. 기본값은 로컬에 있는 가중치만 읽습니다. 첫 다운로드는 `--allow-download`로 명시합니다.

```powershell
$revision = '3abf2810673c7c0778df64a73c2d52eab32d91c4'
python -m scripts.sim.fast_decision.training --input sim_output/fast_decision/dataset-001/train.jsonl --output sim_output/fast_decision/EXAONE-SimDecision-v1 --revision $revision --device cuda --allow-download
```

결과에는 LoRA 어댑터와 `training_manifest.json`이 저장됩니다. LG 기반 파생 모델 이름은 `EXAONE-SimDecision-v1`로 둡니다. 입력 토큰 제한을 넘으면 오류로 중단하며, 기억을 자동으로 잘라 학습하지 않습니다. GPU 용량에 맞춰 입력 한도와 누적 학습 횟수를 조절하고, 긴 입력이 제외된 집단은 별도로 집계해야 합니다.

## 5. 기존 결정과 병행 비교

```powershell
python -m scripts.sim.fast_decision replay --input sim_output/fast_decision/run-001/captures.jsonl --output sim_output/fast_decision/replay-001.jsonl --revision $revision --adapter sim_output/fast_decision/EXAONE-SimDecision-v1 --device cuda
python -m scripts.sim.fast_decision.evaluation --input sim_output/fast_decision/replay-001.jsonl --output sim_output/fast_decision/comparison-001.json
```

`replay`는 읽은 스냅샷의 입력만 사용하고 교사 답은 모델에 주지 않습니다. 재현 실험에서는 학습에 사용하지 않은 사람들의 캡처를 별도 파일로 제공해야 합니다. 위 경로 예시는 명령 구조 설명이며 같은 학습 자료의 재실행 결과를 최종 성능으로 쓰면 안 됩니다.

평가기는 장소 일치율, 소비 오차, 만족도 오차·부정 경험 누락, 집단별 차이, 보류·실패와 지연을 기록합니다. 이는 교사와의 비교이고 실제 시민의 정답률이 아닙니다. 작은 모델 실패·보류 시간까지 포함한 실제 전체 시뮬레이션 가속은 별도 독립 실행에서 측정해야 합니다.

## 6. 확률 보정

```powershell
python -m scripts.sim.fast_decision.evaluation --score-examples sim_output/fast_decision/dataset-001/calibration.jsonl --training-manifest sim_output/fast_decision/EXAONE-SimDecision-v1/training_manifest.json --adapter sim_output/fast_decision/EXAONE-SimDecision-v1 --device cuda --output sim_output/fast_decision/calibration-scores-001.jsonl
python -m scripts.sim.fast_decision.evaluation --calibration-scores sim_output/fast_decision/calibration-scores-001.jsonl --training-manifest sim_output/fast_decision/EXAONE-SimDecision-v1/training_manifest.json --output sim_output/fast_decision/calibration-001.json
```

학습 인물과 겹치거나 합성 자료가 섞인 보정은 거부합니다. 파일에 모델·자료 식별자와 보정 범위를 기록합니다. 보정 결과는 연구용 산출물이며 현재 planner에 자동 적용하거나 운영 경로를 열지 않습니다.

`--score-examples`에는 `--batch-size`를 지정할 수 있습니다. 기본값은 1이며, 서로 독립적인 보정 질문만 묶습니다. 실제 GPU에서 메모리 사용과 단건 대비 점수 차이를 확인한 뒤 늘리세요. 시뮬레이션의 장소 → 지출 → 만족도 판단 순서는 이 옵션으로 바뀌지 않습니다.

## 중요한 한계와 다음 검증

- 현재 지출은 미리 정한 금액 후보, 만족도는 0.05 간격입니다. 정밀도 손실을 실제 자료로 측정해야 합니다. 지원 범위를 벗어난 지출은 학습 시 보류하고, 만족도 반올림이 기존 `<0.3` 부정 경험 기준을 넘으면 보류 라벨로 학습합니다. 여러 경험의 평균·장기 감정 변화에는 여전히 차이가 생길 수 있습니다.
- 여러 판단을 순서대로 하므로 한 번의 모델 호출만 하는 방식은 아닙니다. 각 호출에서 문장 생성을 제거한 것이며, 몇 배 가속된다는 실측 주장은 아직 없습니다.
- 점수의 최고값은 정답 확률이 아닙니다. 기본 `argmax`와 실험용 `sample` 모두 현실 행동 분포 검증이 필요합니다. `sample`은 독립 난수로 재현 가능하지만 보정된 시민 방문 확률을 의미하지 않습니다.
- KV 캐시 재사용은 수치·선택 변화 위험을 검증한 뒤 추가할 최적화입니다. 현재 모델은 KV 캐시를 사용하지 않으며, 위의 토큰화 결과 재사용과는 별개입니다.
- 실제 매출·방문·감정 자료와 장기 실행을 비교하고 사전 허용 오차를 통과하기 전에는 예측 성능 보존을 선언하지 않습니다.

공개 구현 조사와 출처: [Jev 연구 기록](../../../docs/JEV_IMPLEMENTATION_RESEARCH.md). 품질 보존 기준: [전체 설계](../../../docs/SIMULATION_ACCELERATION_DESIGN.md).
