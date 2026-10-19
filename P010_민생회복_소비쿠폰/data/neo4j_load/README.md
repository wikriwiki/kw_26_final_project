# data/neo4j_load/ — Neo4j Day 0 적재 입력

`docs/schedule_generation_plan/runtime_ontology.md §10` 기준 폴더.

> **실행 가이드**는 [`scripts/neo4j_load/README.md`](../../scripts/neo4j_load/README.md) 참조. 이 문서는 입력 데이터 현황만 다룸.

## 현재 상태 (2026-05-11 적재 완료 기준)

| 파일 | 상태 | 비고 |
|---|---|---|
| `admin/KIKcd_H.xlsx` | ✅ 확보 | `data/mapping/KIKcd_H.20251027(말소코드포함).xlsx`에서 복사. 서울 25 District + 427 Dong |
| `admin/adm_code_mapping.csv` | ✅ 확보 | 행안부↔SGIS 매핑 288 행정동 |
| `categories/categories.yaml` | ✅ 확보 | 소비자 관점 12대분류 (10 + 교육 + 기타), 93 L2 |
| `pois/소상공인...서울_202603.csv` | ✅ 확보 (raw) | 537,489건, 결측 0. `03_pois.py`가 직접 CSV 읽음 (parquet 변환 없음) |
| `pois/residence.csv` | ✅ 확보 | K-apt geocoded, 3,146건 (서울 공동주택 단지정보) |
| `pois/workplace.csv` | ⚠️ 부분 확보 | 38,438 / 149,531 (26%) — V-WORLD geocode 한도 미달. 재실행 필요 |
| `mapping/mapping_upjong_to_sub.json` | ✅ 자동 생성 | 247개 L3 → 12대분류 매핑 (fallback 0) |
| `agents/agents_final.json` | ✅ 확보 | `output/agents/agents_final.json`에서 복사 (23MB, **14,881 agent**) |
| `policies/*.txt` | ⚠️ 선택 | POC 1~3개. 정책 시뮬 시점에만 필요 |

## 소상공인 상가정보 가공 통계

- **537,489건** (서울 전체)
- 결측: 좌표·행정동·분류 모두 0건
- 행정동코드(NSO 8자리) 데이터에 직접 포함 — 좌표→행정동 변환 불필요
- 25개 자치구 모두 커버

### 12대분류 분포 (자동 매핑 후)

| L1 | 건수 | sub 수 |
|---|---|---|
| 식사 | 95,393 | 11 |
| 카페 | 24,116 | 1 |
| 디저트 | 6,375 | 2 |
| 주점 | 15,201 | 2 |
| 편의점 | 10,193 | 2 |
| 마트 | 19,469 | 7 |
| 미용 | 35,006 | 5 |
| 쇼핑 | 73,253 | 16 |
| 여가 | 19,156 | 7 |
| 건강 | 29,633 | 8 |
| 교육 | 45,080 | 3 |
| 기타 | 164,614 | 29 |

시뮬 핵심 10대분류(식사~건강) = 약 32.8만 건 (61%). 교육·기타(부동산·법무·광고 등 산업)는 적재만 하고 시뮬에서 거의 사용 안 됨.

## 다음 단계

### Phase 1 — 즉시 가능
적재 스크립트 `scripts/neo4j_load/`:
1. `:District` × 25, `:Dong` × ~424 (admin/)
2. `:Category` × ~92 (12 L1 + ~83 L2)
3. `:Agent` × 60K (agents/)
4. `:POI {type:'commerce'}` × ~530K + `:IN_DONG` + `:IN_CATEGORY` (pois/ + mapping/)

Day 0의 80%가 이 단계에서 끝납니다 (residence/workplace POI만 보류).

### Phase 2 — POI 보강 후
- `residence.parquet` 도착 시 `:LIVES_AT` 배정
- `workplace.parquet` 미도착이면 가상 생성으로 `:WORKS_AT`
- `:KNOWS_POI {source:'initial'}` + `:Memory {type:'initial'}` 시딩

### Phase 3 — 정책 시뮬
- `policies/*.txt` 또는 `*.json` 배치

## 폴더별 가이드
- [`admin/`](./admin/) — 행정동 마스터 (확보 완료)
- [`categories/README.md`](./categories/README.md) — 자동 생성된 어휘 검토 가이드
- [`pois/README.md`](./pois/README.md) — POI 3종 명세 + 자동 변환 흐름
- [`mapping/README.md`](./mapping/README.md) — 자동 매핑 결과 + 검토 가이드
- [`agents/`](./agents/) — 페르소나 (확보 완료)
- [`policies/README.md`](./policies/README.md) — 정책 파일 형식
