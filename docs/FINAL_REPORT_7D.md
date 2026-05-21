# 서울 상권정책 시뮬레이션 — 최종 보고서

**작성일**: 2026-05-21 14:58 KST

**기간**: 2026-05-01 ~ 2026-05-07 (7일) · **모델**: Qwen3-14B-AWQ · **정책 시행**: 2026-05-02

## 1. 시뮬레이션 조건 요약

| 항목 | 값 |
|---|---:|
| 기간 | 2026-05-01 ~ 2026-05-07 |
| 일수 | 7 |
| 정책_시행일 | 2026-05-02 |
| Agent 수 | 14,881 |
| Plan 수 | 101,708 |
| INCLUDES 엣지 | 842,011 |
| State 수 | 116,586 |
| Memory(visited) | 444,914 |
| Memory(rumor) | 51,275 |
| Conversation 약속 | 349 |
| Conversation 추천 | 58,553 |
| Conversation 이슈 | 1 |
| Conversation 기타 | 46,256 |

## 2. 정책 시행 전 vs 후 매출 추이

![1인당 매출](FINAL_REPORT_7D.d/fig2a_per_capita.png)

![변화율](FINAL_REPORT_7D.d/fig2b_change_rate.png)

![DID](FINAL_REPORT_7D.d/fig2c_did.png)

### 평균 일간 매출 비교

| 자치구 | 시행 전 | 시행 후 | 변화율 |
|---|---:|---:|---:|
| 강남 (정책 대상) | 21,630,000원 | 19,506,000원 | **-9.82%** |
| 비강남 (대조군) | 353,682,000원 | 291,490,000원 | -17.58% |

**DID (정책 순효과)**: 강남 변화율 − 비강남 변화율 = **+7.76%p**

## 3. 간접 영향 (Spillover) — 강남 vs 인접 자치구

![spillover](FINAL_REPORT_7D.d/fig3_spillover.png)

강남구 정책이 인접한 서초·송파에 어떤 파급을 줬는지, 멀리 떨어진 강북과 비교. 인접 자치구의 강남 대비 매출 격차가 좁혀지면 spillover로 해석.

## 4. 소비자 행동·심리 분석

### 4-1. 방문 목적·이동 패턴 — 결정 동기 (trigger) 분포

![trigger](FINAL_REPORT_7D.d/fig4_1_triggers.png)

외출(집·직장 제외)의 결정 동기 분류:

| 동기 | 건수 | 비율 |
|---|---:|---:|
| Top 카테고리 | 324,745 | 69.11% |
| 습관 | 112,687 | 23.98% |
| 컨디션 | 13,700 | 2.92% |
| 정책 | 8,013 | 1.71% |
| 소문 | 6,886 | 1.47% |
| 기타 | 2,706 | 0.58% |
| 약속 | 1,134 | 0.24% |
| neighbor | 5 | 0.0% |
| campaign | 5 | 0.0% |
| health | 4 | 0.0% |
| life_style | 2 | 0.0% |
| workplace | 1 | 0.0% |

### 4-2. 단골 vs 신규

![단골](FINAL_REPORT_7D.d/fig4_2_regulars.png)

| 구분 | 관계 수 |
|---|---:|
| 신규 (1회 방문) | 62,745 |
| 재방문 (2~4회) | 62,435 |
| 단골 (5회+) | 30,695 |

전체 KNOWS_POI(방문 경험 있음) 관계 수: **155,875**

### 4-3. 만족도·피드백 — 어떤 동기로 외출했을 때 더 만족했나

![만족도](FINAL_REPORT_7D.d/fig4_3_satisfaction.png)

| 동기 | 평균 만족도 | 표본 수 |
|---|---:|---:|
| health | 0.637 | 4 |
| workplace | 0.58 | 1 |
| Top 카테고리 | 0.571 | 324,745 |
| 습관 | 0.55 | 112,687 |
| 소문 | 0.543 | 6,886 |
| 컨디션 | 0.54 | 13,700 |
| 약속 | 0.538 | 1,134 |
| 기타 | 0.519 | 2,706 |
| campaign | 0.518 | 5 |
| 정책 | 0.507 | 8,013 |
| neighbor | 0.48 | 5 |
| life_style | 0.405 | 2 |

## 5. 1대1 인터뷰 — 페르소나별 대표 (positive / negative / neutral)

### 5-positive — 샘플 없음

### 5-negative — 샘플 없음

### 5-neutral — 샘플 없음

## 부록

- 본 보고서는 `scripts/sim/generate_final_report.py`로 자동 생성됨.
- 시뮬 원본 데이터는 Neo4j에 보존 (Plan/State/Memory/Conversation 노드).
- 인터랙티브 시각화: `output/sim/visualization/sim_standalone.html`
