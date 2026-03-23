# 데이터 파이프라인 및 ETL

## 1. 데이터 소스

### 실제 데이터 (`data/`)

| 파일 | 코드 | 내용 |
|------|------|------|
| `7.서울시 내국인 성별 연령대별(행정동별).csv` | B079 | 행정동×성별×연령대×업종별 카드소비 |
| `8.서울시 내국인의 개인카드 기준 유입지별(행정동별).csv` | B079 | 소비지×유입지×업종별 카드소비 |
| `2.서울시민의 일별 시간대별(행정동).csv` | B079 | 행정동×시간대×업종별 카드소비 |
| `블록별 성별연령대별 카드소비패턴.csv` | B063 | 블록×성별×연령대 세부 소비 |
| `내국인(집계구) 성별연령대별.csv` | B063 | 집계구×성별×연령대 소비 |
| `내국인(집계구) 유입지별.csv` | B063 | 집계구 유입지별 소비 |
| `내국인(블록) 일자별시간대별.csv` | B063 | 블록×요일×시간대 소비 |
| `KT 월별 시간대별 성연령대별 유동인구.csv` | B009 | 격자×시간대×성연령대 유동인구 |
| `KT 월별 성연령대별 거주지별 유동인구.csv` | B009 | 격자×거주지 유동인구 (OD) |
| `행정동별 상권발달 개별지수.csv` | B069 | 행정동×5개 지수 |
| `카드소비 업종코드.csv` | — | SS 업종코드 매핑 |
| `신한카드 내국인 63업종 코드.csv` | — | SB 업종코드 매핑 |
| `월세임대 예측시세.csv` | B068 | 임대 시세 |
| `PURPOSE_250M_202403.csv` | B078 | 250m 격자 생활이동 목적 |

### 데이터 모드 전환

```python
# config.py
USE_SYNTHETIC = True   # 합성 데이터 (기본)
USE_SYNTHETIC = False  # 실제 데이터 (풀 데이터 확보 후)
```

---

## 2. ETL 파이프라인 (`etl_transform.py`)

```
Step 1: build_industry_map()
  → SS/SB 업종코드 → 통합 업종 대분류 매핑

Step 2: build_consumer_segments()
  → (행정동 × 성별 × 연령대) 세그먼트 프로토타입
  → 건당 평균 소비금액 + TOP3 업종

Step 3: build_time_patterns()
  → 행정동별 피크 소비 시간대

Step 4: build_od_matrix() + build_kt_od()
  → 카드 소비 유입지 OD 매트릭스
  → KT 유동인구 OD (주중/주말 비율)

Step 5: build_district_profiles()
  → 상권발달지수 기반 행정동 유형 (HH/HL/LH/LL)

Step 6: generate_agent_profiles()
  → 세그먼트에서 개별 에이전트 샘플링 (소비금액, 행동 성향)
```

---

## 3. 좌표 시스템 (`geo_utils.py`)

KT 유동인구 데이터의 좌표는 **EPSG:5186** (Korea 2000 / Central Belt 2010) 사용.

```
TM 좌표 예시: X=199637.3, Y=553456.6
→ WGS84 변환: lat=37.580662, lng=126.995894 (서울 을지로 부근)
```

```python
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
lng, lat = transformer.transform(x, y)
```

---

## 4. 데이터 이슈 및 해결 방법

### 4-1. 샘플 데이터 제약

| 데이터 | 샘플 행 수 | 예상 풀 데이터 |
|---|---|---|
| B079 카드소비 (행정동별) | 199~200행 | 수십만~수백만 행 |
| B063 카드소비패턴 (블록/집계구) | 500행 | 수백만 행 |
| B009 KT 유동인구 | 500행 | 수천만 행 |
| B069 상권발달지수 | 424행 | 수천 행 |

**영향**: 149개 행정동만 커버, commuter 81% 편중
**해결**: 합성 데이터(`data/synthetic/`)로 서울 전체 417개 행정동 커버

### 4-2. B069 상권발달지수 — LL/LH/HL/HH 분류 부재

**기획서 가정**: B069에 `상권변화지표` 컬럼이 있어서 직접 분류 가능
**실제 데이터**: 5개 수치형 지수만 존재

```
컬럼: DATE, ADSTRD_CD, SALES, INFRASTRUCTURE, STORE, POPULATION, DEPOSIT
예시: 201907, 11260575, 9.11, 11.47, 23.19, 24.49, 60.92
```

**해결 방법**: 2축 기반 4분류 파생

```
축1 (기존 안정성) = (STORE + SALES) / 2
축2 (성장 동력)   = (POPULATION + DEPOSIT) / 2
각 축의 중앙값 기준으로 High/Low 판별 → HH, HL, LH, LL
```

합성 데이터 결과: 417개 행정동 → HH:145, LL:143, HL:65, LH:64

### 4-3. 업종 코드 이중 체계

| 데이터 | 코드 체계 | 개수 |
|---|---|---|
| B079 (행정동별 카드소비) | 업종대분류 한글명 | ~20종 |
| B063 블록별 카드소비패턴 | ss코드 | 75종 |
| B063 집계구별 | sb코드 | 63종 |

**해결**: 대분류(class1) 기준으로 통합 → 138개 코드 매핑

### 4-4. 시간 범위 불일치

| 데이터 | 시간 범위 |
|---|---|
| B079 카드소비 | 2021.01 ~ 2025.12 |
| B063 카드소비패턴 | 2016.01 ~ 2022.08 |
| B069 상권발달지수 | 2018.01 ~ 2019.09 |

**해결**: B063은 구조적 패턴만 추출, B069는 상대적 지수로 사용

### 4-5. 개별 거래 데이터 부재

모든 카드소비 데이터는 **집단 합산** 형태:
```
B079: "을지로3가 + 남성 + 30대 + 한식" → 카드이용금액합계: 1,115,948원, 건수: 43건
```

**해결**:
```python
avg_per_transaction = card_amount / card_count  # 건당 평균 (중앙값 약 17,000원)
monthly_spending = avg_per_transaction × monthly_visits × noise
# 결과: 평균 74만원, 범위 7만~167만원
```

### 4-6. 행정동코드 자릿수 불일치

| 데이터 | 코드 길이 | 예시 |
|---|---|---|
| B079 카드소비 | 10자리 | 1114055000 |
| B069 상권발달지수 | 8자리 | 11140550 |
| KT 유동인구 | 8자리 | 11140550 |

**해결**: B079 10자리의 앞 8자리 사용

```python
b079_adm_cd[:8] == b069_adm_cd  # 매칭 성공
```

---

## 5. 합성 프로토타입 데이터 (`data/synthetic/`)

### 생성 스크립트

```bash
python src/generate_synthetic.py
# → data/synthetic/ 에 저장
```

### 합성 데이터 사양

| 항목 | 샘플 데이터 | 합성 데이터 |
|---|---|---|
| 행정동 커버 | 149개 | **417개** (25개 구 × ~17개 동) |
| KT 격자 셀 | 536개 | **2,048개** (동당 3~7개) |
| Work 매칭률 | 75% | **100%** |
| Home 매칭률 | 53% | **100%** |
| 상권분류 unknown | 97% | **0%** |
| 세그먼트 분포 | commuter 81% 편중 | commuter 55%, resident 23%, weekend 11%, evening 11% |
| B079 gender_age | ~200행 | ~97,000행 |
| KT OD 쌍 | 499개 | **7,388개** |

### 한계
- TM 좌표를 구 중심 ± 정규분포로 생성하므로 한강 위에 격자가 찍힐 수 있음 (풀 데이터로 자연 해결)
- 소비 패턴이 실제 서울 상권 특성과 다를 수 있음

---

## 6. 에이전트 좌표 매핑

### 거주지(Home) / 소비지(Work) 분리

```
[B079 adm_cd] ──── 소비 행정동 ──── work_lat/lng
                                     └── 해당 행정동 내 KT 격자 랜덤 선택 + jitter

[KT OD 데이터] ── 소비지→거주지 역매핑 ── home_lat/lng
                   └── 유동인구 가중 랜덤으로 거주지 행정동 추정
```

### 매핑 파이프라인

```
[KT 유동인구 CSV]
    │  격자ID, X_COORD, Y_COORD, ADMI_CD
    ▼
[격자 맵 구축] (geo_utils.build_grid_map)
    │  격자ID → (행정동코드 8자리, lat, lng)
    ▼
[행정동→격자 역매핑] (geo_utils.build_dong_to_grids)
    │  행정동코드 → [격자1, 격자2, ...]
    ▼
[거주지 매핑 구축] (geo_utils.build_residence_mapping)
    │  KT OD에서 소비지→거주지 유입 가중 확률 계산
    ▼
[에이전트 좌표 할당] (geo_utils.assign_coordinates_to_agents)
    │  work_lat/lng ← 소비 행정동 격자
    │  home_lat/lng ← KT OD 기반 거주지 행정동 격자
    │  current_lat/lng ← 세그먼트별 초기 위치
```

### 초기 위치 규칙

| 세그먼트 | 초기 위치 |
|---|---|
| commuter | 소비지(work) |
| evening_visitor | 소비지(work) |
| resident | 거주지(home) |
| weekend_visitor | 거주지(home) |

### 시뮬레이션 중 이동

```python
# rule_engine.move_agent()
if last_outing:
    dong = last_outing["dong"]
    grids = dong_grids.get(dong, [])
    chosen = random_grid(grids)
    agent["current_lat"] = chosen["lat"] + jitter
else:
    agent["current_lat"] = agent["home_lat"] + jitter  # 재택: 집으로 복귀
```

### 시각화 파일

| 파일 | 용도 |
|---|---|
| `output/agent_map_initial.html` | 초기 에이전트 위치 |
| `output/agent_map_final.html` | 시뮬레이션 종료 후 최종 위치 |
| `output/snapshot_dayNNN.html` | N일차 스냅샷 (7일 간격) |
| `output/weekly_stats.json` | 주간 소비 통계 |
