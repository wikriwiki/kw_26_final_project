# 시뮬레이션 타당성 검증 — 지표·결과·근거

**대상**: 에이전트가 누적 생성한 행동 로그 ↔ 시뮬·페르소나 생성에 미사용된 독립(held-out) 실측 데이터.
**데이터**: 5일 P008 run + 2일 dasol run (각 표본 5,000명).
**검증 지표 세트**: 최신 생성형 통행수요 모델링 논문(아래 [A])과 **동일한 지표 세트(코사인 유사도 + JSD + MAPE)** 를 채택.

> [A] *Next-Generation Travel Demand Modeling with a Generative Framework for Household Activity Coordination* (arXiv 2507.08871, 2025) — 생성 시뮬레이션의 OD 행렬을 **코사인 유사도 0.97**, VMT 분포를 **JSD 0.006 / MAPE 9.8%** 로 검증. 지표를 데이터 유형에 맞춰 사용: **분포→JSD, 공간 구조→코사인, 크기→MAPE.**

---

## ① 검증 지표 및 결과

| # | 지표 | 무엇을 보는가 | 코사인 | JSD | MAPE | 판정 |
|---|------|---------------|--------|-----|------|------|
| 1 | 시간대 활동 분포 | 하루 중 소비가 언제 | **0.918~0.937** | 0.058 | 37% | ✅ |
| 2 | 공간 소비 분포 | 어느 구에 소비가 몰리나 | **0.945** | 0.027 | 34% | ✅ 거시 |
| 3 | 공간 집중도(지니) | 소비 집중 정도 | 시뮬 0.199 ≈ 실측 0.215 | — | — | ✅ |
| 4 | 이동거리 분포 | 집→소비지 거리 | 멱지수 α≈2.07 (중꼬리) | — | — | △ |
| 5 | 재방문·단골 분포 | 단골 형성 | 1회 51% → 단조 중꼬리 | — | — | △ |
| 6 | 소비 불평등(지니) | 개인 간 소비 격차 | 0.240 (소득 0.39보다 낮음) | — | — | △ |
| 7 | 소득–소비 탄력성 | 소득↑→소비↑ | ρ=0.255 (기준 0.7) | — | — | ❌ |

> 논문 [A] 대비: 코사인 0.92~0.95 (논문 0.97에 근접). JSD·MAPE 절대값은 **정답 데이터 해상도**(본 검증은 동 단위 매출지수 프록시)와 표본 규모에 의존.

---

## ② 지표별 상세

### 1. 시간대 활동 분포 (Temporal Activity Distribution)
- **정의**: 24시간 중 소비 활동 시각의 분포. 코사인(형태)·JSD(분포 발산)로 비교.
- **정답 데이터**: BDC `temporal_activity_by_demo` (성·연령별 시간대 활동) — held-out.
- **결과**: 코사인 0.918(P008)~0.937(dasol), **피크 정오 12시 일치**, JSD 0.058.
- **근거**: 분포 비교에 JSD·코사인을 쓰는 방식은 논문 [A] 및 Grimm et al., *Pattern-Oriented Modeling of Agent-Based Complex Systems*, **Science 310 (2005)**.

### 2. 공간 소비 분포 (Spatial Consumption Distribution)
- **정의**: 25개 자치구별 소비 발생량 분포의 일치도. 논문 [A]의 OD 검증과 동일하게 코사인 + JSD + MAPE 적용.
- **정답 데이터**: BDC `dong_context.b069_sales`(동별 매출지수, held-out) + 서울시 우리마을가게 상권분석서비스 추정매출(공개).
- **결과**: 코사인 0.945, JSD 0.027, MAPE 34%. 최대 상권 **강남 1위 재현**.
- **근거**: 논문 [A] (OD 행렬 코사인 0.97 검증); FHWA·Cambridge Systematics, *Travel Model Validation and Reasonableness Checking Manual* (2010).

### 3. 공간 집중도 (Spatial Concentration, Gini)
- **정의**: 소비가 소수 지역에 몰리는 정도(0=균등, 1=완전집중).
- **정답 데이터**: 동별 매출(b069_sales) 기반 실측 분포.
- **결과**: 시뮬 0.199 ≈ 실측 0.215 → 동일 수준 집중(과집중 없음).
- **근거**: Gini, C. (1912), *Variabilità e mutabilità*.

### 4. 이동거리 분포 (Trip-Length / Human Mobility)
- **정의**: 거주지→소비지 직선거리 분포의 멱법칙(중꼬리) 여부.
- **정답 데이터**: 확립된 인간 이동 경험법칙.
- **결과**: median 0.23km, 멱지수 α≈2.07 (중꼬리 형태 재현).
- **근거**: González, Hidalgo & Barabási, *Understanding individual human mobility patterns*, **Nature 453 (2008)**; 적합법 Clauset, Shalizi & Newman, *Power-law distributions in empirical data*, **SIAM Review 51 (2009)**.

### 5. 재방문·단골 분포 (Revisit / Loyalty)
- **정의**: (에이전트, 상점) 방문 횟수 분포의 중꼬리 형태.
- **정답 데이터**: 소비자 충성도 정형사실.
- **결과**: 1회 51.3% / 2회 24.8% / … 단조 감소(중꼬리).
- **근거**: Zipf 법칙; Ehrenberg, *Repeat-Buying*.

### 6. 소비 불평등 (Consumption Inequality, Gini)
- **정의**: 개인 간 1인당 소비 격차.
- **정답 데이터**: 통계청 소득 지니 0.39(2023); 소비 지니는 그보다 낮음.
- **결과**: 0.240 — 방향·범위 정합.
- **근거**: Friedman 항상소득가설(소비평탄화); 통계청 가계금융복지조사.

### 7. 소득–소비 탄력성 (Income–Consumption Elasticity)
- **정의**: 소득과 소비액의 피어슨 상관(정상 ρ>0.7 기대).
- **정답 데이터**: 경제학적 기대(소득↑→소비↑).
- **결과**: ρ=0.255 (기준 미달). 방향은 정상, 기울기 약함 — 식별된 모델 한계.
- **근거**: Engel 법칙.

---

## ③ 지표 선택 근거 (논문 [A] 기준)

| 데이터 유형 | 지표 | 이유 |
|---|---|---|
| 확률 분포 (시간·업종·활동) | **JSD** | 두 분포의 형태 차이를 재는 유계·대칭 발산 (분포 비교 1순위) |
| 공간 구조 (OD·자치구 분포) | **코사인 유사도** | 규모 무시·패턴(방향) 정렬 측정 |
| 집계 크기 (소비량·통행량) | **MAPE** | 상대 오차(%)로 수준 검증, JSD(형태)와 쌍으로 사용 |

**핵심**: 시간·공간 분포(1·2·3)는 논문 [A]와 동일 지표 세트로 정합 확인(코사인 0.92~0.95, OD 논문 0.97 근접), 미시 행동(4·5·6)은 형태 정합, 소득 탄력성(7)만 미달.

---

### 출처
- Next-Generation Travel Demand Modeling with a Generative Framework (arXiv 2507.08871, 2025)
- Grimm et al., *Pattern-Oriented Modeling of Agent-Based Complex Systems*, Science 310 (2005)
- González, Hidalgo & Barabási, Nature 453 (2008); Clauset, Shalizi & Newman, SIAM Review 51 (2009)
- FHWA·Cambridge Systematics, *Travel Model Validation and Reasonableness Checking Manual* (2010); UK *DMRB* (GEH)
- Gini (1912); 통계청 가계금융복지조사·가계동향조사(2024)
