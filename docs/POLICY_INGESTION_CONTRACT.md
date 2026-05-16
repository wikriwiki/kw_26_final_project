# 정책 자동 적재 — 분업 인터페이스 명세

> 자연어 정책 .txt → 시뮬 본체에서 작동하는 `:Policy` 노드까지 가는 파이프라인의 **팀원 ↔ 사용자(시뮬) 사이 contract**.
>
> 팀원이 본 contract에 맞는 JSON 1개를 `data/neo4j_load/policies/{id}.json` 에 떨궈주면, 사용자의 loader가 자동으로 Neo4j에 적재하고 시뮬이 즉시 인지함.

---

## 1. 분업 흐름

```
[팀원 영역]                                              [공유 폴더]                              [사용자 영역]

policies/inbox/*.txt                                                                          
  → LLM 추출        (정책 자연어 → 구조화)                                                        
  → Pydantic 검증   (타입·날짜·범위·도메인)              →  data/neo4j_load/policies/{id}.json   →  load_p007.py 같은 loader
  → ValidatedPolicy 모델                                                                          → Neo4j (:Policy) + [:applied_to] + [:targets]
```

- **팀원의 책임 종료점**: `policies/{id}.json` 파일 1개 떨구기까지
- **사용자의 책임 시작점**: 그 JSON 읽어 그래프 적재 + 시뮬에서 활용
- 팀원은 **Neo4j 직접 안 만져도 됨**

---

## 2. 출력 JSON 스키마 (필수 필드 11개 + 선택 메타)

```json
{
  "id": "P008",
  "name": "서울시민 야간 소상공인 응원 캠페인",
  "type": "subsidy",
  "description": "한국어 자연어 한 문단으로 정책 효과·대상·조건 모두 기술. 시뮬 LLM이 이걸 자율 해석.",
  "benefit_rate": 0.3,
  "cap_per_agent": 50000,
  "announce_date": "2026-04-25",
  "effective_from": "2026-05-01",
  "effective_until": "2026-06-30",
  "target_districts": ["강남구", "마포구"],
  "benefit_categories": ["식사", "카페"],
  "_meta": {
    "source_file": "policy_001.txt",
    "source_file_hash": "sha256:...",
    "confidence": 0.92,
    "extracted_at": "2026-05-15T10:30:00"
  }
}
```

### 필드별 타입·제약

| 필드 | 타입 | 필수 | 규칙 |
|---|---|---|---|
| `id` | string | ✅ | `P` + 3자리 숫자. 기존 사용: P001~P007. 다음 할당: P008부터 |
| `name` | string | ✅ | 한국어 정책명 |
| `type` | enum string | ✅ | 7-enum (§3 참고) |
| `description` | string | ✅ | 자연어 2~5문장 (§4 참고) |
| `benefit_rate` | float\|null | ⚠️ | 0.0~1.0. 환급률·할인율. regulation/facility/campaign 정책은 null 가능 |
| `cap_per_agent` | int\|null | ⚠️ | 1인 한도(원). 위와 동일 |
| `announce_date` | ISO date | ✅ | YYYY-MM-DD |
| `effective_from` | ISO date | ✅ | YYYY-MM-DD |
| `effective_until` | ISO date | ✅ | YYYY-MM-DD, ≥ effective_from |
| `target_districts` | string[] | ✅ | 자치구명 정확 문자열 (§5). 빈 배열 = 서울 전체 |
| `benefit_categories` | string[] | ✅ | L1 카테고리명 (§6). 빈 배열 = 전체 commerce |
| `_meta` | object | ⛔ | 선택. loader가 무시. 검증·추적용 |

---

## 3. `type` enum 7종

| value | 의미 | 예시 |
|---|---|---|
| `subsidy` | 환급·쿠폰·바우처 | 소비쿠폰 10만원 |
| `regulation` | 규제 | 24시 영업 제한 |
| `facility` | 시설·인프라 | 야간 도서관 개관 |
| `campaign` | 홍보·캠페인 | "걷기 좋은 거리" |
| `tax` | 세제 혜택 | 부가세 환급 |
| `transit` | 교통 보조 | 대중교통 캐시백 |
| `environment` | 환경 인센티브 | 친환경 차량 보조 |

위 7개 안에 안 맞으면 → `requires_human_review=True` 처리하고 JSON 출력 X.

---

## 4. `description` 작성 가이드

- **자연어 한 문단** (2~5문장 권장)
- 시뮬 LLM이 이걸 보고 정책 효과를 자율 해석함 — **풍부할수록 시뮬 품질 좋아짐**
- **반드시 포함**:
  - 대상 시민·지역·업종
  - 혜택 메커니즘 (환급률·한도·시간대 등)
  - 정책 목적
- **자유 포함** (기존 ValidatedPolicy의 풍부한 필드는 모두 여기로 흡수):
  - `conditions` (대상 자격)
  - `restrictions` (제외 업종·조건)
  - `expected_behavior_effects` (기대 행동 변화)
  - `target_groups` (청년·소상공인 등)

### 예시 (P007 실제 사용)

> 서울특별시가 전 시민에게 발급하는 디지털 쿠폰. 서울 시내 어느 자치구의 가게·점포에서든 사용 가능하며 1인당 10만원 한도 내에서 결제액의 100%가 차감된다. 모든 commerce 카테고리(식사·카페·디저트·미용·쇼핑·여가·건강·교육·마트·편의점·주점·기타) 적용. 소상공인 매출 회복 + 시민 소비 진작 목적.

---

## 5. `target_districts` — 서울 25개 자치구명

빈 배열 `[]` = 서울 전체 (loader가 25개 다 매핑). 일부만 적용이면 아래에서 **정확한 문자열** 선택:

```
강남구, 강동구, 강북구, 강서구, 관악구, 광진구, 구로구, 금천구,
노원구, 도봉구, 동대문구, 동작구, 마포구, 서대문구, 서초구, 성동구,
성북구, 송파구, 양천구, 영등포구, 용산구, 은평구, 종로구, 중구, 중랑구
```

> ⚠️ 오타·약칭 금지: "강남", "강남區", "Gangnam-gu" 안 됨. 정확히 **"강남구"** 형태만 매칭.

---

## 6. `benefit_categories` — L1 카테고리 12종

빈 배열 `[]` = 전체 commerce. 일부만이면 **정확한 문자열**:

```
식사, 카페, 디저트, 주점, 편의점, 마트,
미용, 쇼핑, 여가, 건강, 교육, 기타
```

> ⚠️ 상세 sub_category(한식·일식·치킨·아메리카노 등)는 적지 마. L1만 받음.
> 자세한 sub 매핑은 `data/neo4j_load/categories/categories.yaml` 참고.

---

## 7. ID 명명 + 충돌 방지

- 형식: `P` + 3자리 숫자, 예: `P008`, `P009`
- **기존 사용 중 (충돌 X)**: P001(비활성), P002, P003, P004, P005, P006, P007(활성)
- **다음 할당 가능**: P008부터
- 같은 ID 두 번 출력 시 → loader가 MERGE로 덮어씀 (의도된 동작)

---

## 8. 출력 위치 + 파일명

| 케이스 | 위치 |
|---|---|
| 검증 통과 → 그래프 적재 대상 | `data/neo4j_load/policies/{id}.json` |
| `requires_human_review=True` | `data/neo4j_load/policies/needs_review/{id}.json` (사용자가 검토 후 위로 이동) |
| 검증 실패·LLM 실패 | `policies/failed/` |

---

## 9. 휴먼 리뷰 정책 처리

`requires_human_review=True`인 경우:

- ✅ JSON은 **`needs_review/` 폴더에 출력**
- ❌ 메인 `policies/{id}.json` 에는 출력하지 마 (loader가 자동으로 적재해버림)
- 📩 사용자에게 "리뷰 필요" 알림 (선택)

휴먼 리뷰 트리거 기준 (팀원이 정의한 것 그대로):
- `confidence < 0.7`
- `target_districts` 불명확
- `target_industries` (= `benefit_categories`) 불명확
- 기간·금액·조건 누락
- 정책 범위 해석 애매

---

## 10. 강제 룰 + 위반 시 동작

| 룰 | 위반 시 |
|---|---|
| `id` 형식 P+3자리 | loader가 reject |
| `type` 7-enum 외 값 | needs_review로 분리 |
| `target_districts` 자치구명 오타 | matched=0 → 정책 효과 없음 (silent fail) |
| `benefit_categories` L1 외 값 | matched=0 → 정책 효과 없음 (silent fail) |
| `benefit_rate` 0~1 범위 외 | reject |
| `cap_per_agent` 음수 | reject |
| `effective_from > effective_until` | reject |
| `announce_date > effective_from` | warn only |

> silent fail 방지를 위해 팀원의 **도메인 validator에 자치구명·카테고리명 enum 검증 포함 필수**.

---

## 11. 참고 자료

| 파일 | 내용 |
|---|---|
| `data/neo4j_load/policies/P007.json` | 실제 작동 예시 (서울 전체 + 전 commerce 적용 케이스) |
| `scripts/neo4j_load/load_p007.py` | 사용자 측 loader (참고용. 일반화된 적재 로직) |
| `data/neo4j_load/categories/categories.yaml` | 12개 L1 + 93개 L2 카테고리 정의 |
| `data/neo4j_load/admin/KIKcd_H.xlsx` | 25 자치구 + 427 행정동 정확한 명칭 |
| `docs/NEO4J_SETUP_GUIDE.md` | Neo4j 환경·DDL 통합 가이드 |
| `docs/schedule_generation_plan/runtime_ontology.md` | 정책 노드·엣지 온톨로지 |

---

## 12. FAQ

**Q. 팀원이 짠 ValidatedPolicy 모델 (15개 필드)은 버려야 하나?**

아니. **검증 통과까지는 그대로 사용**, 마지막 출력 dict로 변환할 때만 본 contract 형태로 매핑. 팀원 코드의 95%는 그대로 살아남음.

**Q. `target_regions`(서울시·수도권 같은 넓은 단위)·`target_groups`(청년·소상공인) 같은 풍부한 필드는?**

`description` 자연어 한 문단에 흡수. 시뮬 LLM이 description을 보고 자율 해석하는 구조라서 별도 필드로 두면 dead data가 됨.

**Q. `benefit_amount` (정액 금액) vs `cap_per_agent` 차이는?**

본 contract에선 `cap_per_agent` 한 가지로 통합. 정액 지급 정책(예: "1인 5만원 지급")은 `benefit_rate=1.0 + cap_per_agent=50000`으로 표현.

**Q. PolicyEffect 별도 노드 (attractiveness_delta 등)는?**

만들지 마. 시뮬에서 안 읽음 → dead data. 정책 효과는 `description` 자연어로 표현하면 충분.

**Q. Watchdog 자동 감시는 유지?**

선택. 시뮬 워크플로는 "시뮬 시작 전 정책 모두 적재 → 시뮬 실행"이라 도중 추가 케이스 없음. 그러나 팀원이 이미 작성했으면 보존해도 무해.

**Q. 캐시 무효화 + summary 재생성은?**

빼도 됨. 시뮬 본체가 매번 fresh 적재 후 실행하는 구조 + vLLM prefix cache는 agent별로 자연스럽게 miss 처리됨.

---

## 13. 분업 인터페이스 — 최종 확정

```
팀원 종착지:  data/neo4j_load/policies/{P008,P009,...}.json   ← 본 contract 준수
사용자 시작지:  scripts/neo4j_load/load_p007.py 일반화 loader   ← 자동 적재
                → Neo4j (:Policy {id, name, type, description, benefit_rate,
                          cap_per_agent, announce_date, effective_from, effective_until})
                + 25개 [:applied_to]->(:District) 엣지
                + 12개 [:targets]->(:Category) 엣지
```

본 contract 안에 들어오는 JSON 1개만 출력해주면 끝. 나머지는 사용자가 처리.
