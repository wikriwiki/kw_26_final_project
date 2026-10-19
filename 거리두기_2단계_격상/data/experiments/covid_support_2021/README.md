# 현재 서울 페르소나 × 2021 국민지원금·코로나 환경

2026-09-07 수집·검증. **자료 준비 단계이며 실제 시뮬레이터에 연결되지 않았다.** 기존 `data/neo4j_load/policies` 로더에 이 JSON을 전달하면 안 된다. 상세 코드 점검은 `docs/COVID_SUPPORT_2021_READINESS.md`를 참조한다.

현재 사람들의 나이·직업·소비 특성은 유지하고, 실험의 달력과 외부 환경에 2021년 규칙을 적용한다. 2021년 실제 인구를 복원한 백테스트와 구분한다. 날짜를 과거로 설정했다고 개인의 출생연도·소득을 자동으로 과거 값으로 바꾸지 않는다.

## 파일과 검증 상태

| 파일 | 내용 | 사용 범위 |
|---|---|---|
| `source_catalog.json` | 공식 출처 17개, URL·기관·목적 | 원문 추적 |
| `download_manifest.json`, `sources/` | 수집 시각·SHA256·원본 | 재현·교차검증 |
| `national_support_rules.json` | 최종 건강보험료 선정표 19행, 1인당 25만원, 신청·소멸·사용 규칙 | 정책 명세; 개인 자격 판정표 미구현 |
| `distancing_schedule.json` | 8/23~12/31의 7개 구간과 추석 가족모임 예외 | 선별한 규칙; 모든 시설별 법적 예외의 완전한 구현 아님 |
| `seoul_cases_daily.json` | 8/1~12/31 153일, 25개 자치구와 기타·타시도 | 후향적으로 정리된 역학 배경; 품질 표시 확인 필수 |
| `seoul_vaccination_review.json` | 접종자료 154행, 원본 값과 오류 후보 | 주입 보류 |
| `data_quality.json` | 자료 간 정합성 검사 결과 | 실행 가능 여부와 별개 |
| `context_preview_20210906.json` | 9/6 종로구 코로나 입력 예시 | 미리보기; 행동 제약 미적용 |
| `experiment_spec.json` | 비교군·측정·가정·실행 조건 | 설계 초안; 실행 설정 파일 아님 |

정책 시행 후 실적 자료(`grant_outcomes`, `grant_outcome_detail`, `seoul_eligibility_coverage`)는 외부 평가·집계 보정에만 쓴다. 에이전트에게 소비 증가율이나 실제 집행 성과를 알려주지 않는다.

## 날짜와 품질

자치구 XLSX의 `source_date`에 하루를 더한 날이 서울 전체 CSV의 `city_reference_date`와 대응한다. 이 정렬로 153일 중 152일의 확진 합계가 일치한다. 2021-09-29 자치구 합계 944명과 2021-09-30 서울 전체 943명은 불일치로 표시했다. 임의로 숫자를 맞추지 않았다. 25개 자치구만 합치면 기타·타시도 때문에 서울 총계와 다를 수 있다.

정확한 당시 발표 시각은 이 자료에 없다. 미리보기는 새벽 기준으로 `city_reference_date < simulation_date`인 과거 자료만 사용한다. 예컨대 9/6 입력의 마지막 `source_date`는 9/4다. 이는 보수적으로 가정한 발표 지연이며 당시 실제 발표판의 재현을 보장하지 않는다. 후속 분석은 1일 추가 지연에 대한 민감도를 확인한다. 최근 7일에 불일치가 있으면 이전 날짜로 몰래 대체하지 않고 오류를 낸다.

접종자료는 요청 기간의 모든 날짜를 포함하지만 12/29가 중복된다. 인원/분모로 계산한 비율과 공개 비율이 0.15%p 넘게 다른 행이 31개다. 단순 오탈자인지 분모·정의 차이인지 추가 확인이 필요하다. 원본의 빠진 0을 추정하여 채우지 않았다. 현재 `runtime_usable=false`; 개인별 접종 상태도 `unknown`이다. 공개 집계 접종률만으로 특정 페르소나의 접종 이력을 관측했다고 간주해서는 안 된다.

## 재생성

Python 3.10 이상. 원본 XLSX 읽기에만 `openpyxl`이 필요하다. 다운로드는 선택 사항이며 이미 저장한 원본으로 정규화할 수 있다.

```powershell
python scripts/experiments/fetch_covid_support_sources.py
python scripts/experiments/prepare_covid_support_data.py
python scripts/experiments/preview_covid_context.py --day 2021-09-06 --district 종로구 --output data/experiments/covid_support_2021/context_preview_20210906.json
python -m pytest tests/unit/test_covid_context_preparation.py -q
```

선택 재수집은 `fetch_covid_support_sources.py --only SOURCE_ID`를 사용한다. 재수집하면 원본과 해시가 갱신되므로 이미 시작한 실험의 입력은 별도 고정해야 한다. `distancing_schedule.json`과 `experiment_spec.json`은 출처를 확인하여 작성한 수동 명세다. 정규화 스크립트가 원문에서 모든 법적 조건을 자동 추출하는 것은 아니다.

이 폴더의 `.gitattributes`는 `sources/` 원본의 줄바꿈 자동 변환을 막는다. Windows/Linux에서 clone한 뒤에도 원본 SHA256 검사가 동일하게 동작하도록 유지해야 한다.

이 폴더는 Neo4j나 LLM에 연결하지 않는다. 실제 대상자 배정, 당시 가맹점 명부, 개인 접종·가구 자료, 시뮬레이터의 방역 강제 검증은 별도 준비가 필요하다.
