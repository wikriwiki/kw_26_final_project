# 실내체육시설 금연구역 실험

> **현재 활성 설계(2026-09-23):** 기존 고정 7,500명에서 노원·서초·송파 거주자 1,154명(387·322·445명)을 선택했다. [새 설계](experiment_spec_3gu_1154_v1.json), [명단](cohort_3gu_1154_ids.json), [선정 감사](cohort_3gu_1154_selection.json), [Vast 시작 안내](../../../deploy/vast/START_3_DISTRICTS.md)를 사용한다. 아래의 7,500명 설명은 이전 설계의 검증 기록이다. 새 번들은 로컬 preflight만 통과했고 원격 쌍 DB/GPU 실행은 남아 있다.

브랜치 `No_SmokingZone_EXP`, 시작 기준 `origin/main@77ec86cf49397ac118fc63501f39abbb3244cab2`.

실험 규모는 **정확히 7,500명·대상 POI 308개**, 기간은 **2017-11-19~12-16의 28일: 시행 전 14일·후 14일**이다. 원본 그래프의 14,881명 중 적격 풀은 14,549명이다. 원래 BASE7500H 실행자 중 적격한 7,494명을 유지하고 소비 anchor가 없는 6명은 다른 적격자로 대체한 고정 명단과 흡연 라벨을 그대로 쓴다. 본 bundle `full_7500_v1`과 같은 명단 중 30명의 `pilot_7500_v1`은 인물·시설 준비 검사 blocker 0이다. **Day 0 `2017-11-18` 재구축·내보내기·해시 확인을 마쳤고, 한 개의 격리 Windows Neo4j 5.26.0에 실제 재복원한 뒤 두 bundle의 graph_preflight를 통과했다.** 이전 `2017-11-25` baseline의 검증 기록은 보존하지만 현재 일정에는 사용할 수 없다. 독립된 두 팔 복원·Linux 배포·GPU 시뮬레이션은 아직 검증하지 않았다.

## 근거와 평가

| 파일 | 용도 |
|---|---|
| `policy.json` | 시행일, 시설 범위, 실험 조건 |
| `smoking_rates.json` | 서울시 공식 2017 지역사회건강조사 성별·연령별 흡연율 |
| `ground_truth.json` | 실행 후에만 읽는 연구 비교값 |
| `source_manifest.json` | 원문과 검증 이미지의 SHA-256 |
| `evidence.md` | 표·페이지·분모·방법·자료 불일치 검토 |
| `experiment_spec.json` | 고정 실험 설계와 준비 상태 |
| `graph_source.json` | 원본 덤프, 격리 복원 조회, 코호트·POI 추출 및 초기화 상태 |
| `baseline_rebuild_validation.json` | 현재 28일 일정의 11월 18일 Day 0 재구축·검증 상태 |
| `baseline_rebuild_validation_prior_14d.json` | 이전 14일 일정의 11월 25일 Day 0 검증 기록 원본 보존 |
| `cohort_7500_ids.json` | 본 실험과 파일럿의 기준이 되는 고정 7,500명 명단 |
| `cohort_7500_selection.json` | 원본 실행 명단 복구 근거, 6명 대체 규칙·ID·해시 |

경제 분석 범위는 **노원구(11350), 서초구(11650), 송파구(11710)의 당구장과 실내골프연습장**이다. 스크린골프를 실내골프에 포함한다. 일반 스포츠·여가 카테고리나 골프용품점을 통째로 포함하지 않는다. 전국 정책 중 연구 대상 시설만 모델링하는 제한된 시나리오다.

법 시행은 **2017-12-03**, 연구의 회귀 정책변수는 **2017년=0, 2018년=1**이다. 당구 매출 +13.54%는 관측 카드자료의 조정 회귀 추정치다. 현재 시뮬레이션의 동일기간 ON/OFF 차이와 추정 대상이 다르므로 정확도 합격이나 인과효과 재현을 주장하지 않는다. 결제 건수는 방문자 수가 아니며, 비유의 결과를 효과 0으로 채점하지 않는다. 자세한 출처와 한계는 `evidence.md`를 따른다.

| 구간 | 날짜(양 끝 포함) | 일수 | OFF 팔 | ON 팔 |
|---|---|---:|---|---|
| Day 0 | 2017-11-18 | 초기 상태 | 같은 baseline | 같은 baseline |
| 시행 전 | 2017-11-19~12-02 | 14 | 금연정책 비활성 | 금연정책 비활성 |
| 시행 후 | 2017-12-03~12-16 | 14 | 금연정책 비활성 | 금연정책 활성 |

시행 전·후 집계를 각각 보존하고, 각 팔의 전후 변화와 시행 후 ON/OFF 차이를 구분해 보고한다. 관측 연구의 조정 월별 회귀효과와 같은 추정량으로 해석하지 않는다.

## 실측 통계 기반 흡연 라벨

[서울시 공식 통계집](https://news.seoul.go.kr/welfare/archives/260455) 인쇄 p31 그림10의 성별·연령별 조율을 사용한다. 원문 이미지와 숫자를 직접 대조했다. 집단별 인원×흡연율을 가장 가까운 정수로 반올림하고, 정렬한 ID에서 고정 시드로 해당 인원을 추출한다. 전체 비율 18.8%를 모든 집단에 일괄 적용하지 않는다.

개별 라벨은 **실측 인구 통계에 근거한 합성 배정**이다. 가상 인물의 실제 개인 건강정보를 관측한 값은 아니다. 성별 미상은 같은 연령 전체 비율, 19세 미만·나이 미상은 `unknown`을 사용한다. 페르소나의 나이·소득을 2017년으로 소급 변경하지 않는다.

모집단은 복원 그래프와 대조한 `graph_cohort_v2/personas.json`의 14,881명이다. 거주 POI→행정동 연결이 없는 321명과 양수 소비 anchor가 없는 11명, 총 332명을 제외해 적격 풀 14,549명을 얻었다. 소득·소비·나이는 원본 그래프와 대조하며 누락 나이를 임의로 채우지 않는다. **전체 14,881명에 먼저 흡연 상태를 배정한 다음 적격 풀 → 고정 실험 7,500명 → 파일럿 30명 순서로 선택**하므로 같은 사람의 라벨은 유지된다.

원본 덤프를 다시 격리 복원하여 2025-07-14의 State 7,500명과 Plan 7,500명이 일치하며 7월 14~20일 전체 ID 합집합도 같은 7,500명임을 확인했다. 일부 날짜에는 실패로 기록이 빠졌지만 명단 밖 ID는 없었다. 원본 명단 SHA-256은 `5137064448a4e4d2f098412faba0a7ba8d4668dcdf0847dbeea67e2ee1bfdf2b`다. 이 중 현재 적격자는 7,494명이고 나머지 6명은 양수 소비 anchor가 없다. 따라서 **이번 실험 명단은 원본 7,500명과 완전히 같지 않다.** 유지할 7,494명을 제외한 적격 풀에서 시드 `20171203`의 SHA-256 순위로 6명을 대체하며, 흡연 라벨이나 실험·연구 결과를 선정에 사용하지 않는다.

고정 명단 파일 `cohort_7500_ids.json`의 SHA-256은 `1358a7e060268131fcc1a0357cd2e66200cb42fb14ed5451dfd31da57ceaed59`다. `full_7500_v1`은 **흡연자 1,294명·비흡연자 5,813명·미상 393명**, `pilot_7500_v1`은 **4명·25명·1명**이다. 파일럿은 동일 7,500명 안의 부분집합이며 개인별 흡연 라벨이 일치함을 검증했다. 서울 전체 18.8%와의 차이는 코호트의 연령·성별 구성, 미성년자 및 선정에 따른다. 각 bundle의 `assignment_audit.json`에 전체 모집단 배정·적격 필터·실험 명단·파일럿 선택 단계를 보존한다.

이전 `full_ready_v2`의 흡연자 2,498명·비흡연자 11,271명·미상 780명은 **적격 풀 14,549명의 집계**다. `full_v1`, `full_ready_v2`, `pilot_ready_v2`는 본 실험 bundle로 사용하지 않는다.

## 로컬 준비와 검사

Python 3.10 이상. `prepare`, `preflight`, `smoke`, `score`는 표준 라이브러리만 필요하며 LLM과 DB에 연결하지 않는다. 프로젝트 루트에서 실행한다.

```powershell
# 준비된 고정 명단의 bundle 검사. 원격 DB/GPU 검사는 별도다.
python scripts/experiments/no_smoking_zone.py preflight --bundle output/no_smoking_zone/full_7500_v1
python scripts/experiments/no_smoking_zone.py preflight --bundle output/no_smoking_zone/pilot_7500_v1

# 다시 만들 때는 새 출력 경로를 사용한다. 기존 산출물을 덮어쓰지 않는다.
python scripts/experiments/no_smoking_zone.py prepare --agents output/no_smoking_zone/graph_cohort_v2/personas.json --eligible-ids output/no_smoking_zone/graph_cohort_v2/eligible_ids.json --cohort-ids data/experiments/no_smoking_zone/cohort_7500_ids.json --pois output/no_smoking_zone/staging_audit/verified_pois.json --seed 20171203 --simulation-seed 17001 --out output/no_smoking_zone/full_7500_v2

# 같은 모집단 라벨을 유지한 30명 파일럿
python scripts/experiments/no_smoking_zone.py prepare --agents output/no_smoking_zone/graph_cohort_v2/personas.json --eligible-ids output/no_smoking_zone/graph_cohort_v2/eligible_ids.json --cohort-ids data/experiments/no_smoking_zone/cohort_7500_ids.json --pois output/no_smoking_zone/staging_audit/verified_pois.json --limit 30 --out output/no_smoking_zone/pilot_7500_v2

python scripts/experiments/no_smoking_zone.py smoke --out output/no_smoking_zone/context_smoke.json
python -m pytest tests/unit/sim/test_no_smoking_context.py tests/unit/sim/test_no_smoking_experiment.py tests/unit/sim/test_no_smoking_poi_extraction.py tests/unit/sim/test_no_smoking_rebuild.py tests/unit/deploy -q
```

`prepare`는 누락 상태도 기록하고 0으로 종료한다. `preflight`는 blocker가 있으면 종료코드 2다. `smoke`는 명시적 가짜 테스트 시설로 프롬프트 연결만 확인하며 실제 경제효과 검증이 아니다. GPU·Neo4j 검사는 `run` 직전에 별도로 수행한다.

POI JSON은 다음 필드를 갖는 배열이다. 아래 예시의 ID는 스키마 설명용이며 실제 데이터로 실행하지 않는다.

```json
[
  {
    "poi_id": "실제 그래프의 POI ID",
    "district_code": "11650",
    "facility_type": "billiard",
    "classification_source": "실제 시설 업종코드/등록자료 또는 검토 근거"
  }
]
```

`facility_type`은 `billiard`, `indoor_golf`, `screen_golf`만 허용한다. 모든 구에 당구·실내골프가 있어야 준비 검사를 통과한다. 이는 최소 데이터 범위 검사이며 충분한 표본이나 결제 발생을 보장하지 않는다. 실제 그래프에서 `type=commerce`, `IN_DONG`, `IN_CATEGORY`와 상위 `여가` 분류도 확인한다.

현재 세 구의 여가/당구·스포츠 후보 915개에는 원본 L3 업종코드가 없고, 당구 카테고리에 복싱장·클라이밍장 등 상호가 불일치하는 사례가 있었다. `extract_no_smoking_pois.py`는 카테고리와 명시 상호를 함께 요구해 **당구 244개·스크린골프 63개·실내골프 1개**를 추출하고 607개를 보류했다. 노원은 당구 74개/골프 24개, 서초 69개/16개, 송파 101개/24개다. 일반 골프 명칭·브랜드만 있는 상호, 파크골프·실외·판매/피팅 가능성, 분류 충돌은 자동 포함하지 않는다.

`staging_audit/verified_pois.json`은 **기록된 상호·카테고리 기반의 보수적 분류 추론**이다. 파일명 `verified`는 실제 등록업종·실내 배치·현재 영업 여부의 확인을 의미하지 않으며 전체 시설의 대표 표본도 아니다. `verified_pois.audit.json`에 915개 모두의 원 상호·카테고리·판정 사유, 업종코드 미관측, 원본 덤프·입력·분류기·매핑 해시를 보존했다.

## DB 준비와 실행

2026-09-22 사용자 Downloads에서 `BASE7500H_nopolicy_7d.dump`와 같은 내용의 `(1)` 사본을 발견했다. 크기 785,727,594 bytes와 MD5가 기존 검증 기록과 일치하며 SHA-256도 기록했다([graph_source.json](graph_source.json)). 기존 Neo4j 5.26.0의 2025-07-14~20 무정책 실행 **후** 그래프다. 원본을 보존한 격리 복원본에서 Agent 14,881개, POI 543,924개, Category 93개, District 25개, Dong 427개와 과거 State·Plan·Memory·Conversation·Policy를 확인했다. 이 원본 SHA를 깨끗한 Day 0 baseline SHA로 사용하지 않는다.

이전 14일 계획에서는 원본 격리 복원, 첫 초기화의 메모리 부족 실패 후 배치 500개 재시도, 11월 25일 Day 0 생성·내보내기·한 개 Windows Neo4j 5.26.0에 실제 재복원까지 검증했다. 정적 노드 559,350개·관계 1,267,832개의 전후 fingerprint가 일치했고, 이후 7,500명과 30명의 graph_preflight도 통과했다. 이 검증은 [baseline_rebuild_validation_prior_14d.json](baseline_rebuild_validation_prior_14d.json)에 바이트를 바꾸지 않고 보존했다.

이전 baseline `output/no_smoking_zone/baseline_v1/neo4j.dump`는 841,216,715 bytes, SHA-256 `43004478ff00f016004522da6fcc84e42d063859bff58d2afa47c5739d221d4e`이며 **이전 일정에 대해서는 유효하지만 현재 일정에서는 사용하지 않는다.** Day 0가 11월 25일이므로 새 시작일 11월 19일의 전날 상태가 될 수 없기 때문이다. 당시 내보내기 전 14,549명 검사와 이후 7,500명 재복원 검사를 새 일정의 검증으로 이름만 바꾸지 않는다.

현재 일정은 원본을 새로 격리 복원하여 **2017-11-18 Day 0**로 재구축했다. `rebuild_28d_apply_v1.json`은 `status=complete`, `blockers=[]`이며 정적 노드 559,350개·관계 1,267,832개의 전후 fingerprint가 같고 새 State 14,881개를 확인했다. 새 덤프는 `output/no_smoking_zone/baseline_28d_v1/neo4j.dump`, **841,292,022 bytes**, SHA-256 **`6103c534628da29c6aafdbbe808ff530af5b0e47234466eb8b90d535f979830d`**다. 파일 수는 71개, 비압축 크기는 4,409,352,685 bytes다.

[baseline_rebuild_validation.json](baseline_rebuild_validation.json)은 이 새 일정의 보고서 해시·정적 fingerprint·Day 0·내보내기·실제 재복원 검증을 기록한다. `clean_reload_7500_28d_validation.json`에서 7,500명과 30명 모두 `graph_preflight_passed=true`, `all_checks_passed=true`이며 명단·흡연 라벨도 그대로다. 이 보고서 SHA-256은 `a45b75fd1a5a8fbad36e6d03a627b3ef92e2ab6b855b10ba4369f9648d594d3e`다. 해시 표식은 복원본에만 추가했고 원본 덤프는 수정하지 않았다.

검증은 **한 개의 격리 Windows DB**에서 수행했다. 원 그래프·Day 0 14,881명, 적격 풀 14,549명, 본 실험 7,500명을 구분하며 이 단일 복원 검사를 독립된 두 팔·Linux·LLM/GPU 검증으로 확대 해석하지 않는다. 해당 검사가 남아 있으므로 `live_run_ready=false`를 유지한다.

기존 Neo4j 일일 엔진(`scripts/sim/run_simulation.py`)을 재사용한다. 흡연 상태·시설 규칙은 Dawn/Stage1과 Stage2에 전달되며, 최종 결제 영수증을 대상 POI별로 집계한다. 정답 데이터는 이 경로에 들어가지 않는다. 금연 상태는 시설 이용공간의 규칙이며 실제 흡연실 보유·준수율·오염농도까지 생성하지 않는다.

필요한 복원 절차:

1. 원본 덤프의 정적 POI·에이전트·거주/직장 anchor·카테고리·가격/이동 reference를 격리 환경에서 대조한다. 현재 코호트는 이 대조에서 제외된 332명을 반영한다.
2. 시작일 `2017-11-19` 전날인 **`DAY_ZERO=2017-11-18`**의 State를 만든다. 기존 소비규모 기반 초기 잔액은 현대 페르소나 가정이다. 이전 11월 25일 State의 날짜만 바꾸어 완료된 검증처럼 취급하지 않는다.
3. 다른 Policy, 이전 Plan/Conversation, 실행 흔적이 있는 State가 없는 깨끗한 baseline을 확보한다. 과거 검증 DB를 덮어쓰지 않는다. 초기 KNOWS_POI 등의 날짜도 미래 자료가 유입되지 않게 검토한다.
4. baseline dump의 SHA-256을 기록하고 **동일 dump를 두 독립 Neo4j 인스턴스/DB에 복원**한다. Community Edition은 별도 인스턴스 두 개를 사용한다. 실제 hash 확인 후 각 복원 DB에 `ExperimentSnapshot {id: SHA256, sha256: SHA256}` 표식을 기록한다. 이 표식 자체는 동일성의 증명이 아니므로 dump checksum 검증이 선행되어야 한다.
5. 아래 환경변수를 설정하고, 두 팔의 환경·가격·모델·코드·코호트를 동일하게 유지한다. 원본/DB가 없으면 준비 완료로 표시하지 않는다.

기본 모델은 `LGAI-EXAONE/EXAONE-4.5-33B-AWQ` (`LLM_MODE=exaone_4_5`)이며 기존 SGLang 환경을 유지한다. 비추론 모드(`enable_thinking=False`)를 사용하고 자동으로 다른 모델로 전환하지 않는다. AWQ는 33B 모델의 4비트 버전이다. 동일 GPU 속도 비교는 아직 수행하지 않았으며 선택 근거와 고정 버전은 [model_selection.json](model_selection.json)에 기록했다.

필수 환경변수는 `NO_SMOKING_OFF_NEO4J_URI`, `NO_SMOKING_ON_NEO4J_URI`, `NO_SMOKING_OFF_NEO4J_DATABASE`, `NO_SMOKING_ON_NEO4J_DATABASE`, `NO_SMOKING_SNAPSHOT_SHA256`, `NEO4J_PASSWORD`, `LLM_BASE_URL`, `LLM_MODE`, `NO_SMOKING_SERVER_CONFIG`다. 마지막 변수는 `serve_sglang.sh`가 기록하는 버전·모델 revision 파일 경로이며 배포 실행기가 자동 지정한다. DB별 암호가 다르면 `NO_SMOKING_OFF_NEO4J_PASSWORD`, `NO_SMOKING_ON_NEO4J_PASSWORD`를 사용한다. 암호는 로그·명령행 인자로 기록하지 않으며 실제 env 파일은 Git에서 제외한다.

```bash
# 새 baseline 검증·두 팔 복원 후, 먼저 시행 전 하루의 30명 처리량 검사
python scripts/experiments/no_smoking_zone.py run --bundle output/no_smoking_zone/pilot_7500_v1 --arm off --start 2017-11-19 --days 1 --workers 4 --out output/no_smoking_zone/pilot_off_20171119

# 파일럿 검토 후 깨끗한 baseline을 다시 복원하여 본 실험 28일 실행
python scripts/experiments/no_smoking_zone.py run --bundle output/no_smoking_zone/full_7500_v1 --arm off --start 2017-11-19 --days 28 --workers 4 --out output/no_smoking_zone/off_28d_17001
python scripts/experiments/no_smoking_zone.py run --bundle output/no_smoking_zone/full_7500_v1 --arm on  --start 2017-11-19 --days 28 --workers 4 --out output/no_smoking_zone/on_28d_17001
python scripts/experiments/no_smoking_zone.py score --off output/no_smoking_zone/off_28d_17001 --on output/no_smoking_zone/on_28d_17001 --out output/no_smoking_zone/comparison_28d_17001.json
```

30명·1일 파일럿은 연결/처리량/오류/결제 발생 검사용이며 효과 추정이나 정책 전환 검증에 쓰지 않는다. 전환까지 확인할 때는 같은 30명으로 28일의 작은 진단 실행을 별도 수행할 수 있다. 본 실험은 정확히 7,500명·28일이며, 파일럿에서 실제 처리량과 결제·오류 상태를 확인한 뒤 진행한다. 28일 설정만으로 연구의 21개월 패널을 재현하지 않는다. 기본은 원래 LLM 경로이며 `SIM_FAST_MODE=off`를 강제한다.

점수기는 완료된 두 팔의 ID×일자 완전성, 코드/입력/모델/시드/환경 동일성을 검사하고, 시행 전·후 집계를 구분해 보고하며 시행 후 대상 매출·결제건수의 ON/OFF 및 구별·흡연별 차이를 계산한다. OFF 매출 0이면 변화율은 `null`이다. 연구값은 참고값으로 병기하며 자동 합격 판정을 내리지 않는다. 현재 단일 실행기이며 중단 런 재개·다중 시드 통합 신뢰구간은 아직 지원하지 않는다. 중단 시 출력과 실패 기록을 보존하고 깨끗한 snapshot에서 새로운 출력 경로로 다시 실행한다.

두 팔을 합하면 시드 1개당 **7,500명 × 28일 × 2팔 = 420,000 agent-days**다. 시드 `17001,17002,17003` 세 개를 모두 실행하면 **1,260,000 agent-days**이며 이는 계획량이지 자동 실행 지시가 아니다. 파일럿의 실측 처리량과 예산을 검토한 뒤 본 실행·추가 시드를 진행한다. 반복 시드는 흡연 배정 `--seed 20171203`을 유지하고 `--simulation-seed`만 바꾼 새 bundle을 쓰며, 매번 같은 새 baseline을 복원한다. GPU/병렬 계산에 의한 잔여 비결정성은 남는다.

## vast.ai

[deploy/vast/README.md](../../../deploy/vast/README.md)에 현재 오퍼 조회, 버전 고정 모델, SSH, 패키징, 서버 초기화, 시간 제한, 결과 회수와 종료 절차를 정리했다. 계정 로그인과 $400 잔액은 확인했으나 이번 준비 단계에서 서버를 임대하지 않았다. 검증된 baseline 덤프를 두 팔로 복원하고 실제 실행 검사를 마친 뒤 소규모 파일럿부터 실행한다.

## 후속 인터뷰와 집단 반응

현재 프롬프트, 인용 가능한 근거 기록, 전체 에이전트 완전성 감사, 시점별 인터뷰와 찬반 군집/평가는 [interview_and_stance.md](interview_and_stance.md)를 따른다. 실제 찬반 결과는 GPU 실행과 후속 인터뷰 전에는 존재하지 않는다.
