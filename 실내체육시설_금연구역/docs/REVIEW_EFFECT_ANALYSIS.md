# 리뷰 데이터가 에이전트 의사결정에 미친 영향 — 분석 보고서

> 작성일: 2026-06-29
> 데이터: `neo4j_3d_v2.dump` (437MB) — 3일치 시뮬 그래프 (Neo4j 5.x, 5.26으로 적재)
> 분석 방법: 덤프를 Neo4j에 적재 후 Cypher 탐색 + 코드(main 브랜치) 직접 검토

---

## 핵심 결론 (TL;DR)

**리뷰는 에이전트 의사결정에 영향을 미쳤다 — 단, 작고 특정적인 채널로.**

- 카카오 별점·리뷰가 POI 선택의 **결정적 동인이었던 비율 0.2%**(117건), **고려·언급된 비율 3.0%**(2,073건).
- 리뷰가 쓰일 땐 거의 항상 **고별점·긍정 평가 쪽으로**(긍정 838 : 부정 4 ≈ 200:1), **식당·카페·건강** 같은 경험재에서 작동.
- 전체 선택은 여전히 **습관(과거 만족 기억) 47% + 근접성 37%** 가 지배하며, 리뷰는 입소문(2.9%)보다도 아래.
- ⚠️ 기존 구조는 **별점 값·조회 이력을 저장하지 않아** 영향의 정밀 측정(인과)이 제한적 → 본 분석 후 **계측 코드를 추가**해 향후 정밀 분석이 가능하도록 함.

---

## 1. 데이터 개요

| 항목 | 값 |
|---|---|
| Agent | 15,000 |
| POI | 540,388 (commerce 위주, type: commerce/residence) |
| Plan (에이전트-일 계획) | 22,500 = **7,500/일 × 3일** (05-01 평일, 05-02·03 주말) |
| INCLUDES (의사결정 이벤트) | 173,162 (그중 외출 commerce 68,757) |
| Conversation | 463 / Memory 7 / **Policy 0** |

**중요**: POI 노드 속성은 `id, name, lat, lon, dong_code, type` 6개뿐 — **별점/리뷰 속성이 그래프에 없음.** 리뷰는 런타임에만 쓰이고 영속화되지 않기 때문(아래 2장).

---

## 2. 리뷰 처리 메커니즘 (코드 검토)

리뷰 기능 = **"POI 별점·리뷰 가용성 인지 + LLM 선택적 lookup"** (커밋 `6f060c4`).

```
[외부] 카카오 SQLite DB  (scripts/sim/poi_review_lookup.py)
   C:/Users/Administrator/naver_crawl/sqlite/kakao_enrich.db
        │  C_<상가번호> → COM_<상가번호> 매칭 → 평균별점·리뷰수·리뷰텍스트 추출
        ▼
[Stage2] 선택적 2-pass lookup  (scripts/sim/stage2_poi.py)
   1차: LLM이 review_lookup_requests 에 "확인할 POI" 적어 반환 (본인 판단)
   2차: 코드가 ★별점·리뷰 fetch → 프롬프트에 "★4.5 (N리뷰)+리뷰글" 첨부 → 재호출 → 최종 picks
        ▼
[영속화] scripts/sim/plan_writer.py — INCLUDES 관계에 기록
   ✅ pick_factor(선택 동인 태그) · pick_reason(선택 사유 텍스트)  ← 리뷰 영향이 여기 남음
   ❌ 별점 값 · 조회한 POI 목록 · review_lookup_count → (기존엔) 저장 안 함
```

**시사점**: 그래프 덤프에 리뷰 데이터가 없는 것은 *설계상 정상*. 리뷰의 영향은 Stage2의 `pick_factor`/`pick_reason`에만 흔적으로 남는다. (Stage1 `reasoning`에는 거의 없음 — 리뷰는 *POI 선택*(Stage2) 단계의 기능이지 *의도*(Stage1) 단계가 아니므로.)

---

## 3. 정량 분석 결과

### 3-1. POI 선택 동인 순위 (`pick_factor`, commerce 58,710건 중)

| 동인 | 건수 | 비중 |
|---|---|---|
| satisfaction (과거 만족·습관) | 27,654 | **47.1%** |
| distance (근접성) | 21,684 | **36.9%** |
| known (단골·친숙) | 2,741 | 4.7% |
| appointment (약속) | 2,534 | 4.3% |
| random | 2,046 | 3.5% |
| rumor (입소문) | 1,729 | 2.9% |
| **review (별점·리뷰)** | **117** | **0.2%** |
| policy_spend | 95 | 0.16% |

→ 의사결정은 **습관과 근접성**이 압도(합 84%). 리뷰는 7위로 입소문보다도 작다.

### 3-2. 리뷰 영향 규모 (commerce 68,757건 기준)

- 리뷰가 **결정적 동인**(`pick_factor='review'`): **117건 (0.17%)**
- 리뷰를 **고려·언급**(`pick_reason`에 별점/리뷰/★): **2,073건 (3.0%)**

### 3-3. 리뷰는 "고품질로 유도" (언급 2,036건 분석)

| 신호 | 건수 |
|---|---|
| 긍정·고별점 인용 (긍정/좋/높은 별점/친절) | **838** |
| 부정 인용 (낮은/별로/안 좋) | **4** |
| "5.0 / ★5" 최고별점 인용 | 441 |

→ 리뷰를 쓸 때 **거의 전적으로 고별점·긍정 쪽**(≈200:1). 에이전트가 리뷰를 "좋은 곳을 고르는 근거"로 활용.

### 3-4. 리뷰가 작동하는 카테고리 (경험재 집중)

| 카테고리 | pick_factor=review | pick_reason 리뷰언급 |
|---|---|---|
| 식사 | 64 | 1,225 |
| 카페 | 14 | 284 |
| 여가 | 10 | 177 |
| 건강 | 17 | 168 |
| 마트 | 8 | 65 |
| 미용/편의점/디저트 | 소수 | 26/26/15 |

→ **식당·카페·건강** 등 품질 불확실성이 큰 경험재에서 리뷰가 의미. 편의점·마트 같은 commodity엔 거의 무영향.

### 3-5. 질적 증거 (`pick_factor='review'` 실제 사례)

> "다움치과는 긍정적 리뷰와 높은 별점(4.6) 바탕으로 선택. 신규 장소지만 신뢰할 정보가 있음."
> "카카오 별점 5.0으로 신규 한식점 선택. 두부전문점이라는 점이 페르소나의 한식 선호와 일치."
> "★★5.0(1리뷰) 한식점. 신규이지만 높은 별점을 고려해 선택."

특징: 리뷰는 종종 **신규(미경험) 가게를 시도할 신뢰 근거**로 작동.

---

## 4. 결론 및 해석

1. **리뷰는 영향을 미쳤으나 주변부 채널**이다. 의사결정 구조는 *습관(만족 기억)·근접성*이 지배하고, 리뷰는 그 위에 얇게 작동한다.
2. 리뷰가 작동할 땐 **합리적 방향**(고별점·긍정·경험재 집중)으로 움직여, 기능이 *의도대로* 동작함을 시사한다.
3. 낮은 비중(0.2%)이 **"설득력이 약해서"인지 "노출이 적어서"인지는 이 데이터로 분리 불가** — '선택적 lookup'이라 노출량이 기록되지 않기 때문.

---

## 5. 한계 (이 덤프 기준)

| 한계 | 영향 |
|---|---|
| 별점 *값*·조회 POI 목록 미저장 | "본 별점 대비 얼마나 높은 곳 골랐나"(선택지 내 비교) 불가 |
| 조회 여부(lookup) 미기록 | "리뷰 본 결정 vs 안 본 결정" 비교 불가 → 노출↔설득 분리 불가 |
| `actual_satisfaction` 7건뿐 | "리뷰픽이 실제 만족↑로 이어졌나" 불가 |

---

## 6. 후속 조치 — 계측 코드 추가 (구현 완료)

위 한계를 풀기 위해 **추가 LLM 호출 없이**(기존 2-pass의 버려지던 데이터만 캡처) 다음을 저장하도록 코드를 수정함:

**Plan 노드**: `reviews_seen`(본 리뷰 전체 JSON), `review_lookup_count`(조회 수), `review_changed_count`(리뷰로 바뀐 선택 수)
**INCLUDES 관계**: `review_seen`, `seen_rating`, `seen_rating_count`, `review_snippet`, `pre_review_poi`(리뷰 전 선택), `review_changed`

수정 파일: `stage2_poi.py`, `run_simulation.py`, `plan_writer.py` (성능: 추가 LLM·SQLite·Neo4j 호출 0, 시간복잡도 불변).

### 이걸로 *재실행 시* 가능해지는 분석
- **lookup→선택**: `review_seen`으로 리뷰 본/안 본 결정 비교
- **선택지 내 별점 비교**: `reviews_seen`에 기각된 후보 별점까지 있어 "고른 별점 vs 본 후보 평균" 측정
- **설득 전환율**: `review_changed_count / review_lookup_count`
- **변경 사례 추적**: `pre_review_poi`(전) → 최종 POI(후) + `pick_reason`(근거)

---

## 부록 — 재현 방법

```bash
# 1) 덤프 적재 (Community DB는 파일명 neo4j.dump 필요)
docker run --rm -v <dump_dir>:/dumps -v neo4j_review:/data neo4j:5.26 \
  neo4j-admin database load neo4j --from-path=/dumps --overwrite-destination=true
# 2) 서버 기동
docker run -d --name neo4j_review -v neo4j_review:/data -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/analyze123 neo4j:5.26
```

```cypher
// 동인 순위
MATCH (:Plan)-[r:INCLUDES]->() WHERE r.pick_factor IS NOT NULL
RETURN r.pick_factor AS factor, count(*) AS c ORDER BY c DESC;

// 리뷰 결정적 픽
MATCH (:Plan)-[r:INCLUDES]->() WHERE r.pick_factor='review' RETURN count(*);

// 리뷰 언급 + 카테고리
MATCH (:Plan)-[r:INCLUDES]->()
WHERE r.pick_reason CONTAINS '별점' OR r.pick_reason CONTAINS '리뷰' OR r.pick_reason CONTAINS '★'
RETURN r.category, count(*) ORDER BY count(*) DESC;
```
