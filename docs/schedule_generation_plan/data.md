# data.md — 필요 입력 데이터셋

> 메인: [`./schedule_generation_plan.md`](./schedule_generation_plan.md)

에이전트 일일 스케줄 생성에 필요한 모든 데이터 목록. 개념·사용 방식은 메인 문서 참조.

**저장 매체**: 모든 데이터는 **Neo4j 그래프**에 적재한다(`schedule_generation_plan.md §4`). parquet/CSV는 그래프 빌드 입력 용도로만 사용되며, 런타임 조회는 Cypher 인덱스 조회로 일원화. 에이전트간 자연어 대화·정책 뉴스는 Graphiti의 NL→그래프 추출 파이프라인으로 처리(구조화 데이터는 직접 Cypher INSERT).

---

## 0.1 에이전트 기초 (이미 존재)

| # | 데이터셋 | 상태 | 왜 필요한가 |
|---|---|---|---|
| D1 | **에이전트 페르소나** (`output/agents/agents_final.json`) | ✅ 확보 | 60K 에이전트의 성별·연령·직업·소비분위·거주/직장 행정동 원천. Stage 1 프롬프트 `[페르소나]` |
| D2 | **통계 JSON 9종** (`output/stats/*.json`) | ✅ 확보 | 참조통계·`consumption_flow` 원천 |
| D3 | **행정동 코드 매핑** (`data/mapping/adm_code_mapping.csv`, `KIKcd_H.xlsx`) | ✅ 확보 | MOPAS↔NSO 코드 변환, POI `dong_code` 10자리 생성 |

## 0.2 POI 3종 (신규 확보 필요)

| # | 데이터셋 | 상태 | 왜 필요한가 |
|---|---|---|---|
| D4 | **소비 POI** — 소상공인 상가업소정보 | ❌ 확보 필요 | 이벤트의 모든 `place` 실명 출처. 환각 방지의 전제 |
| D5 | **거주 POI** — 공동주택 단지정보 | ❌ 확보 필요 | 에이전트 거주지 1건씩 부여. 하루 첫·마지막 이벤트 |
| D6 | **직장 POI** — 건축물대장 | ❌ 확보 필요 | 유직 에이전트 직장 1건씩 부여. 평일 09~18시 근무 |

## 0.3 카테고리 매핑 (신규 생성)

| # | 산출물 | 상태 | 왜 필요한가 |
|---|---|---|---|
| D13a | `data/mapping/categories.yaml` — 10 카테고리 × ~45 서브카테고리 정의 | ❌ 설계 필요 (T1c) | 프로젝트 공통 분류 어휘 (Stage 1 출력 스키마·인덱스 키). T1a/T1b 진행 중 서브 목록 확정 후 작성 |
| D13b | `data/mapping/mapping_upjong_to_sub.json` — 상가업소 업종코드 → (cat, sub) | ⚠️ 생성 필요 (T1a) | D4 POI 태깅 |
| D13c | `data/mapping/mapping_sb63_to_sub.json` — 신한카드 SB63 → (cat, sub) | ⚠️ 생성 필요 (T1b) | 통계 JSON과 카테고리 어휘 일관성 |
| D13d | `data/mapping/mapping_building_to_category.json` — 건축물 용도 → cat | ⚠️ 생성 필요 (T4) | D6 POI 카테고리 부여 |

## 0.4 가공 산출물 (본 모듈에서 생성)

| # | 산출물 | 상태 | 왜 필요한가 |
|---|---|---|---|
| D7 | 에이전트별 거주/직장 확정 → `:Agent`-`[:LIVES_AT]`/`[:WORKS_AT]`->`:POI` 엣지 | ⚠️ 생성 필요 (T6) | 60일 내내 재사용되는 개인 고정 앵커 (parquet 중간 산출물 폐기, 그래프 직적재) |
| D10 | `consumption_flow.json` — 거주 행정동 → 소비 유입지 확률분포 | ⚠️ 생성 필요 (T5.5) | 당일 intent zone 실측 기반 샘플링 (BDC 8번 가공) |
| D11 | 초기 인지 시딩 — `[:INITIAL_AWARENESS]` 엣지 bulk insert | ⚠️ 시딩 작업 필요 | 거주·직장·경로·원거리 랜드마크 ~80/agent (~4.8M 엣지). 기억 전체는 5종 엣지 분리 (`:VISITED`/`:HEARD_RUMOR`/`:SAW_SNS`/`:HEARD_POLICY`/`:INITIAL_AWARENESS`). 단골성·지인 추천은 `memory_context`·`social_context`가 Cypher로 추출. 개념: [`schedule_generation_plan.md §4.2`](./schedule_generation_plan.md) |
| D12 | **사전계산 `[:NEARBY]` 엣지** — KDTree로 초기 1회 Top-30 계산 → `(:Agent)-[:NEARBY {anchor,category,rank}]->(:POI)` (~3,600만) + `(:Dong)-[:NEARBY {category,rank}]->(:POI)` (~127K) | ⚠️ 빌드 스크립트 필요 | 런타임 Cypher 인덱스 조회로 <1 ms. KDTree는 **초기 빌드 도구로만**, 이후 폐기(런타임 상주 없음). zone 앵커는 계층 traversal `(:Dong)<-[:IN_DONG]-(:POI)`도 병용 가능 |

## 0.5 정책·이벤트

| # | 데이터셋 | 상태 | 왜 필요한가 |
|---|---|---|---|
| D9 | **정책 카탈로그** (`data/policy_events.yaml`) | ❌ 설계 필요 (T11) | 정책 공지·수혜조건·대상지역. awareness/전파 엔진 투입 |

---

> 정적 룩업(카테고리→운영시간은 `categories.yaml`에 동봉, 직업→건물용도는 D13d의 일부)·캘린더는 [`generation.md`](./generation.md)와 검증기에서 사용.
