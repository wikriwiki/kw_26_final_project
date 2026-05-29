# 1일치 풀런 점검 — 2026-05-01

**파일**: `C:\Users\Administrator\sim_output\metrics/day_2026-05-01.jsonl`

## 1. 기본 통계

| 항목 | 값 |
|---|---:|
| 총 처리 agent | 14,560 |
| 성공 | 14,543 (99.88%) |
| 실패 | 17 (0.12%) |
| 평균 elapsed/agent | 36.6s |
| 총 tokens_in | 124,938,090 (8,591/agent) |
| 총 tokens_out | 17,137,289 (1,178/agent) |
| 평균 만족도 | 0.584 |
| Stage 1 retry | 260 (1.8%) |
| Stage 2 retry | 1 |

## 2. Stage 2 Fallback

외출 이벤트 (cand_calls): **77,187** / INCLUDES total **127,822**

| 단계 | 건수 | 비율 |
|---|---:|---:|
| 1차 sub 매칭 | 63,329 | 82.05% |
| 2차 L1 dong fallback | 4,526 | 5.86% |
| 3차 L1 district | 1 | 0.00% |
| 4차 모두 비어 있음 (드롭) | 9,331 | 12.09% |

- 환각 자동 보정 (random Top-5): **18,547** (145.10‰)
- 환각 드롭 (후보 자체 없음): **4,081** (31.93‰)
- order_mismatch (다른 order POI): **22,505** (29.16% of cand_calls)
- missing_picks_filled: **6,446** (50.43‰)
- resolve_dong placeholder: **0** (0이어야 정상)

## 3. trigger 분포 + reasoning 적재율 (Stage 1)

외출 INCLUDES: **67,884**

- reasoning 적재율: **100.0%** (67,884)
- trigger 적재율: **100.0%** (67,884)
- pick_reason 적재율: **63.1%** (42,864)

**trigger 분포**:

| trigger | 건수 | 비율 |
|---|---:|---:|
| top_category | 48,269 | 71.11% |
| habit | 18,392 | 27.09% |
| mood | 731 | 1.08% |
| none | 420 | 0.62% |
| appointment | 39 | 0.06% |
| rumor | 26 | 0.04% |
| policy | 5 | 0.01% |
| health | 2 | 0.00% |

## 4. Plan / INCLUDES / 외출 비율

- Plan 노드: **14,543**
- INCLUDES 엣지: **127,822**
- 외출 이벤트: **67,884** (53.1% of INCLUDES)
- 내부 (집/직장): 59,938

## 5. 환각 검출 (모든 poi_id가 :POI 노드인지)

- INCLUDES: 127,822 / valid POI: 127,822 → 환각 **0건** (✅ 0건)

## 6. trigger별 평균 만족도

| trigger | 평균 만족도 | n |
|---|---:|---:|
| health | 0.650 | 2 |
| top_category | 0.575 | 48,269 |
| rumor | 0.557 | 26 |
| habit | 0.553 | 18,392 |
| appointment | 0.541 | 39 |
| none | 0.525 | 420 |
| mood | 0.510 | 731 |
| policy | 0.444 | 5 |

## 7. Night Phase 1 — visited Memory + KNOWS_POI

- visited Memory (day=2026-05-01): **0** (다음 날 새벽에 적재 — 0이면 아직 미처리)
- KNOWS_POI 갱신 (last_visit=2026-05-01): **0**

## 8. Night Phase 2 — Conversation 분포

| intent | n |
|---|---:|
| 기타 | 5,606 |
| 약속 | 18 |
| 추천 | 7,298 |
| **합계** | **12,922** |

- rumor Memory 적재: **7,298** (추천+이슈 합과 일치해야 정상)

## 9. 샘플 reasoning (LLM 출력 품질 육안 점검)

**Stage 1 reasoning (5건):**

- `AGT_11290685_F_30대_001` @ 20:00 | **쇼핑** → 동성푸드마켓 | trigger=`top_category`
  > 평일 Top 카테고리에 백화점 36% 포함. 신혼 생활에 맞춰 화장품 구매를 위한 쇼핑.

- `AGT_11440700_F_40대_002` @ 08:30 | **식사** → 국제식당 | trigger=`top_category`
  > 평일 Top 카테고리에 한식 13% 포함. 출근 전 한식으로 아침 식사.

- `AGT_11500593_F_40대_003` @ 15:00 | **쇼핑** → 엘르벨ellebelle | trigger=`habit`
  > 평일 소비 패턴에 쇼핑 카테고리가 포함됨. 일상적인 의류 구매.

- `AGT_11215847_F_70대이상_003` @ 08:00 | **할인점/슈퍼마켓** → 중한슈퍼 | trigger=`top_category`
  > 평일 Top 카테고리에 할인점/슈퍼마켓 22% 포함. 절약형 소비 성향으로 필수품 구매를 위해 일찍 외출.

- `AGT_11590530_F_30대_005` @ 17:30 | **식사** → 노가리먹태포차 | trigger=`top_category`
  > 평일 Top 1위 한식 8% + 거주 동 근처에서 저녁 식사.

**Stage 2 pick_reason (3건):**

- `AGT_11545700_M_20대_001` | **식사** → 채선당금천 | factor=`distance`
  > 채선당금천 — 0.12km로 가장 가까운 곳. 한식 카테고리에선 단골이 우선시되며, 이 POI는 방문 0회지만 aff=0.50으로 상대적으로 높다.

- `AGT_11350619_F_50대_002` | **학원** → 케이엠에듀수학학원 | factor=`distance`
  > 케이엠에듀수학학원 — 이벤트 7의 후보 중 가장 가까운 0.05km, 단골이 아니지만 탐색을 위해 선택.

- `AGT_11215847_F_50대_002` | **쇼핑** → 신발할인매장 | factor=`distance`
  > 신발할인매장 — 같은 카테고리 중에서 affinity 0.50로 가장 높고, 거리 0.06km로 가장 가까움. 라이프스타일 실속형이라 단골을 선호하지만, 이 경우 단골이 없으므로 가장 가까운 곳 선택.

**Night Phase 2 Conversation reasoning (3건):**

---

## 🎯 v3 변경 사항 중점 점검

### 10-A. reasoning 품질 (Stage 1)

외출 reasoning 67,884건

- 평균 길이: **48자** (min 15, max 130)
- 너무 짧음 (<30자) — placeholder 의심: **499** (0.74%)

**페르소나·근거 인용 비율 (높을수록 깊이 있는 reasoning):**

| 인용 종류 | 건수 | 비율 |
|---|---:|---:|
| 페르소나 (Top·라이프스타일·소득·직업) | 62,538 | 92.1% |
| 정책 (바우처·쿠폰·환급·policy_id) | 26 | 0.0% |
| 과거 만족도 (어제 sat·만족도) | 633 | 0.9% |
| 약속·지인 (AGT_) | 102 | 0.2% |

### 10-B. trigger 정합성

오늘(2026-05-01) 정책 발효 여부: **❌ 비활성 (baseline)**

✅ baseline 일자, policy trigger 5건 (정상)
- top_category 비중: **71.1%** (60%↑이면 LLM이 안전 라벨로 쏠림 — diversity 낮음)
- 다양성: rumor 26 · mood 731 · appointment 39

### 10-C. pick_factor 분포 (Stage 2 단골 vs 탐색)

| factor | 건수 | 비율 |
|---|---:|---:|
| distance | 32,160 | 75.0% |
| known | 3,432 | 8.0% |
| satisfaction | 3,052 | 7.1% |
| novelty | 2,923 | 6.8% |
| random | 859 | 2.0% |
| affinity | 354 | 0.8% |
| rumor | 76 | 0.2% |
| appointment | 2 | 0.0% |
| repetition | 2 | 0.0% |
| consistency | 2 | 0.0% |
| category_match | 1 | 0.0% |
| necessity | 1 | 0.0% |

✅ 균형 (known 8.0% / novelty 6.8%)

### 10-D. 강남 vs 비강남 매출 (단일 정책 검증)

정책 대상 카테고리 (카페·디저트) 매출:

| 자치구 | 매출 | 이벤트 수 |
|---|---:|---:|
| **강남 (정책 대상)** | 2,766,000원 | 461 |
| 비강남 (대조군) | 42,912,000원 | 7,150 |

- 강남 1인당 ~1,520원 / 비강남 1인당 ~3,368원
⚠️ baseline 일자인데 강남이 비강남 대비 **-54.9%** — 인구·소득 분포 효과로 일부 차이는 정상, 30%↑이면 점검 필요

### 10-E. v3 신규 필드 적재율

| 필드 | 적재율 | 비고 |
|---|---:|---|
| Stage1 reasoning | 100.0% | ✅ 95%+ 정상 |
| Stage1 trigger | 100.0% | ✅ 95%+ 정상 |
| Stage2 pick_reason | 63.1% | 외출 이벤트만 (50~70% 정상) |
| Stage2 pick_factor | 63.1% | 외출 이벤트만 (50~70% 정상) |
| Night Conversation reasoning | 0.0% | ⚠️ 95%+ 정상 |

---

## 진단 결과 요약

- ✅ 성공률 99%+ 정상
- ✅ 환각 0건
