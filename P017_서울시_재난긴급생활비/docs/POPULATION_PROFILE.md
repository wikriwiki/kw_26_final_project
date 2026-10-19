# 실측 인구분포 표본과 소득 프로필

기존 여섯 정책 런은 소비 10분위 표본입니다. 새 기능은 아직 해당 결과에 적용하지 않았고, 서버·그래프·모델을 변경하거나 새 시뮬레이션을 실행하지 않았습니다. 코드의 준비와 정책 효과 검증 완료를 구분합니다.

`tools/freeze_population_matched_cohort.py`는 성별·연령대·행정동·소득의 공식 주변분포를 CPU에서 교정합니다. `population_gate_pass`는 표본 통계 관문이고 `runtime_input_binding_verified`는 그 표본의 소득이 실제 시민 프롬프트와 정책 자격에 같은 값으로 연결됨을 확인하는 별도 관문입니다. 둘 다 참이어야 `model_calls_allowed`가 참입니다. 가중 분포만 맞춰 비가중 실제 모델 시민이 목표와 다르게 뽑히면 진행하지 않습니다. 네 주변분포의 일치는 결합분포 검증이 아닙니다.

런타임은 기본값에서 기존 페르소나를 그대로 사용합니다. 명시적으로 두 환경변수를 함께 지정해야 새 프로필을 사용합니다.

```
POPULATION_PROFILE_FILE=/absolute/path/frozen_profile.json
POPULATION_PROFILE_EXPECTED_SHA=<that file's SHA256>
```

하나만 지정하거나, 파일/소스 해시가 맞지 않거나, 실행 중 설정·파일이 바뀌면 실패합니다. 프로필의 시민 목록을 그대로 쓰므로 `--limit`로 그 일부를 다시 뽑거나 `--gu`로 제한할 수 없습니다. 모든 시민의 정확한 나이·성별·실제 LIVES_AT 거주동을 모델 worker 시작 전에 그래프와 대조합니다. ID 접두어의 동 코드로 거주 POI를 대체하지 않습니다. 코드계·동 경계 빈티지의 공식 crosswalk 증거도 필요합니다.

프로필 스키마는 `frozen_population_profile_v1`입니다. `population_unit=resident_person`, `reference_year`, `assignment_kind`, `source_evidence`, `household_income_definition`, `admin_dong_code_system`, `official_admin_crosswalk_verified=true`, `policy_outcome_used_for_assignment=false`가 필요합니다. 합성시민에게 공식 가구소득 분포를 이용해 소득을 배정하는 경우 `assignment_kind=calibrated_synthetic_income_assignment`라고 명시합니다. 실제 개인의 관측 소득으로 부르지 않습니다.

`mapping_definition`은 `method=observed_income_band_mapping`, `income_band_order`, `band_to_tier`, `observed_band_direct_mapping=true`, `policy_outcome_used=false`, 원본 경로·해시가 있는 `source_evidence`를 담습니다. tier는 기존 정책 키인 하·중하·중·중상·상 중 하나이며, 관측 소득구간 순서를 뒤집으면 실패합니다. 이 순서를 임의 소득 5분위로 자동 변환하지 않습니다. 모름·무응답은 정책 자격 tier로 쓰지 않습니다. 구간을 정책 키로 묶는 근거가 아직 없으면 프로필을 활성화하지 않습니다.

각 `rows` 항목에는 `aid`, `sex`, 정확한 정수 `age`, 실제 거주 앵커의 `home_dong_code`, `income_band`, `income_tier`, `household_income_definition`이 있습니다. 정의는 프로필 전체와 일치해야 합니다. 가구의 월 세전 소득이면 그대로 가구·월·세전이라고 밝힙니다. 이를 개인 연봉, 일별 소비예산 또는 실측 개인소득으로 바꾸지 않습니다.

Dawn은 같은 프로필에서 `persona.income`과 `p_income_level`을 만들고, 정책 지급 판정도 그 값을 읽습니다. 프롬프트에는 해당 시민에게 배정된 소득구간과 가구 정의만 들어갑니다. 공식 집계 비율·목표 지표·실측 정책 효과값·소스 감사 표는 프롬프트에 넣지 않습니다. 개인별 기존 생활양식·직업·소비 앵커를 재추정하는 기능은 없으므로, 새 소득과 이 속성들의 결합 정합성은 따로 점검해야 합니다. `daily_income_by_aid` 소비예산 보충액을 이번 소득 할당으로 대체하지도 않습니다.

표본 도구의 배선 증거 `runtime_input_binding`은 동결 profile 파일·SHA, `population_runtime_binding_audit_v1` 감사 파일·SHA, 그리고 population_profile.py·dawn_context.py·run_simulation.py의 실제 소스 SHA를 담습니다. 감사는 같은 profile로 시민·정책 소득이 일치함, 전체 시민의 그래프 정체성 관문 통과, 집계 목표의 프롬프트 미주입을 기록해야 합니다. profile의 시민·성별·정확한 나이·실제 거주동·소득구간이 교정 표본과 다르면 배선 관문은 실패합니다.

검증은 합성 단위 테스트로 수행했습니다. 실측 목표 frame·구간→tier 근거·새 동결 profile이 실제로 완결되기 전에는 새 정책 검증이 가능하다고 판단하지 않습니다.
