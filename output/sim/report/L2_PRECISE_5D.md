# L2 spillover 정밀 측정 (4-case 분리)

- 시뮬 기간: 2026-05-18 ~ 2026-05-22 (5일)
- 정책 시행일: 2026-05-20
- baseline 일자: 2026-05-19
- 정책 활성 일자: 2026-05-20, 2026-05-21, 2026-05-22

## 1. 분석 동기

`day_health_check.py` 10-D 의 L2 (도보권 거주) 그룹은 거주지만 보고 분류한다. 그러나 P008 정책의 `POLICY_CYPHER`는 `LIVES_AT OR WORKS_AT` 둘 다 fetch하므로, 거주 L2 + 직장 L1(사업 구간)인 agent는 *direct + spillover 혼재* 신호를 받는다. 이 스크립트는 거주 L2 를 직장 위치별 4 case 로 분리해, *진정한 spillover* 만 추출한다.

## 2. 분모 (거주 L2 agent의 직장 분포)

| Case | 의미 | n |
|---|---|---:|
| **pure_L2** | 거주 L2 ∩ 직장 비강남(또는 직장 없음) — 순수 spillover | 63 |
| **mixed_L2_L1work** | 거주 L2 ∩ 직장 L1 — direct + spillover 혼재 | 42 |
| **both_walk** | 거주 L2 ∩ 직장 L2 — 양쪽 도보권 | 65 |
| **L2_gangnam_other** | 거주 L2 ∩ 직장 강남 비도보권 — spillover (직장 동선 일부 강남) | 20 |
| (Control) | 비강남 거주 | 13,748 |

> 분모는 시뮬 첫날 기준 (agent population은 변하지 않음).

## 3. baseline 일자 평균 1인당 매출 (카페·디저트)

| Case | 1인당 매출 (원) | vs Control |
|---|---:|---:|
| pure_L2 | 1,238 | -30.7% |
| mixed_L2_L1work | 1,857 | +4.0% |
| both_walk | 3,415 | +91.3% |
| L2_gangnam_other | 2,400 | +34.4% |
| Control (비강남 거주) | 1,785 | — |

## 4. 정책 활성 일자 평균 1인당 매출

| Case | 1인당 매출 (원) | vs Control |
|---|---:|---:|
| pure_L2 | 2,444 | +55.1% |
| mixed_L2_L1work | 2,857 | +81.3% |
| both_walk | 4,615 | +192.9% |
| L2_gangnam_other | 3,400 | +115.8% |
| Control | 1,576 | — |

## 5. DID — 정책 효과 (baseline 격차 → 정책 활성 격차)

| Case | baseline 격차 | 정책 활성 격차 | **DID 효과** |
|---|---:|---:|---:|
| pure_L2 | -30.7% | +55.1% | **+85.8%p** |
| mixed_L2_L1work | +4.0% | +81.3% | **+77.3%p** |
| both_walk | +91.3% | +192.9% | **+101.6%p** |
| L2_gangnam_other | +34.4% | +115.8% | **+81.4%p** |

> **pure_L2 의 DID 효과**가 P008 의 진정한 spillover 지표. `day_health_check.py` 의 L2 통합 수치는 이 4 case 의 가중평균으로, mixed_L2_L1work 의 direct 효과가 일부 섞여있다.

