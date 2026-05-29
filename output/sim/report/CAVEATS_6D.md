# 시뮬레이션 데이터 한계 및 해석 주의사항 (6일 풀런)

> 본 보고서의 모든 수치를 해석할 때 반드시 함께 읽어야 할 한계 문서.
> 시뮬 기간: 2026-05-18 ~ 2026-05-23 (6일).
> 정책 시행일: 2026-05-20 ~ 2026-05-23 (P008 강남역-역삼역 보행친화거리, 4일).
> baseline: 2026-05-18·5/19 (정책 미활성). 단, 5/18 은 아래 한계로 인해 5/19 단일 일자 baseline 권장.

---

## 1. 시뮬 흐름 변경 — 6일로 단축

원안: 2026-05-18 ~ 5/24 (7일). 실제: **2026-05-18 ~ 5/23 (6일)** 로 단축.

- 마감 데드라인(2026-05-29 21:00) 까지 7일 시뮬을 완료할 수 없음.
- 시뮬 v8 (`--start 2026-05-21 --days 4`) 의 Day 2 (5/23) Night Phase 완료 시점에 자동 종료 후 시각화 빌드.
- 5/24 데이터는 Day 3 본 작업이 ~50 agent 정도 진입한 시점에서 강제 종료. **분석에서 5/24 통째로 무시**.

정책 활성 일자가 4일(5/20·5/21·5/22·5/23)로 baseline 2일과 거의 동일 — DID 가중평균 안정성은 유지.

---

## 2. 5/18 — Conversation 데이터 0건

### 무엇이 일어났나
- 원래 시뮬에서 12,477건의 Conversation 정상 생성 (Night Phase 2).
- WSL OOM 추정으로 시뮬 silent kill → resume 시 Night Phase 2 가 재실행되며 12,455건 *중복* 추가 (총 24,932).
- 분석 신뢰성 회피 위해 **Conv 5/18 통째 삭제**.

### 결과
- 5/18 에는 상호작용(약속·추천·이슈·기타) 데이터가 **0건**.
- 5/18 Plan / State / visited Memory (67,806) 는 정상 보존 (CREATE 이지만 done_aids 로 agent 중복 차단).
- 5/18 rumor Memory 5,388 건은 그대로 유지 — *부모 Conv 가 삭제됐지만 자식 rumor 는 1배 정상 수치*. ("dangling" 의미는 출처가 끊겼다는 뜻이지 수치가 부풀려졌다는 뜻 아님. 5/19=5,349, 5/20=5,291, 5/21=5,143 와 비슷.)

### 해석 시 주의
- baseline 1인당 외출·매출 (`day_health_check` 10-D, layered DID) 은 5/18·5/19 평균으로 계산 가능 — Plan·매출 신호는 정상.
- 그러나 **trigger=appointment 분포·약속 횟수 같은 Conv 기반 지표는 5/19 단일 일자로만 계산**해야 함.
- DID 비교 시 baseline 일수 비대칭에 주의: Conv 기반 지표는 baseline n=1, 정책 활성 n=4. 통계적 유의성보다는 효과 크기 위주로 해석.

---

## 3. 5/20 — visited Memory 0건

### 무엇이 일어났나
- Day 3 (5/20) 본 작업 완료 후 Night Phase 1·2 진행 중 시뮬 silent kill.
- Conversation 5/20 (12,996) · Memory.rumor 5/20 (5,291) 은 부분 적재됨 (Night 2 가 거의 끝나갈 무렵 죽음).
- **Memory.visited day=5/20 = 0** — 5/20 에 어떤 POI 를 방문했는지의 기록이 영원히 누락.

### 결과
- `KNOWS_POI` 의 5/20 갱신이 누락 — 5/20 첫 방문 POI 의 `visit_count`·`avg_satisfaction`·`last_visit`·`recent_visit_dates` 가 업데이트되지 않음.
- 단골화 지표 (5/19 → 5/21 갭) 에서 5/20 만큼의 가속이 누락.

### 해석 시 주의
- 단골 vs 신규 비율 · KNOWS_POI 기반 reuse rate 는 **약간 underestimate** 됨. 단골 비율이 보고서보다 실제로 조금 더 높음.
- 매출·외출 지표는 영향 없음 (Plan/State 정상).
- 5/20 Conv 는 부분 적재 — 12,996 건이 정상 분량인지 1배인지 검증 필요 (5/19=정상 분량과 비교).

---

## 4. 5/19 Plan 의 잔여 영향 (영구)

Day 2 (5/19) 처리 시점에 5/18 의 24K 중복 Conv 가 *아직 살아있었음*. Dawn 컨텍스트가 그것을 fetch 해서 LLM 에 노출했고, 일부 reasoning 이 그 영향을 받음.

- 5/19 의 `trigger=appointment` 카운트가 **부풀려졌을 가능성**. 5/20 이후와 비교해 비정상 증가 패턴이면 본 영향으로 해석.
- 5/19 Plan/State 본 적재는 정상 (done_aids 로 agent 중복 차단).
- 5/20 dawn 시점엔 이미 5/18 Conv 가 삭제되어 있어서 이후 날짜에는 영향 없음.

---

## 5. 5/24 — 분석 무시

자동 종료 daemon 의 신호로 `day_2026-05-24.jsonl` 첫 라인 등장 시점에 시뮬 kill. Day 3 본 작업이 ~50 agent 정도 진입한 상태에서 종료됨.

- 5/24 Plan / State 가 ~50 agent 분량 Neo4j 에 적재될 수 있음.
- **본 보고서는 5/24 데이터를 모든 집계에서 제외**.
- L2 정밀 측정 (`precise_l2_did.py`) · DID 분석 · 차트 모두 `--days 6` (5/18~5/23) 로 한정.

---

## 6. L2 spillover 측정의 한계와 정밀 측정

### 한계
- `day_health_check.py` 10-D 의 L2 (도보권 거주) 그룹은 *거주지만* 기준으로 분류.
- 그러나 P008 정책의 `POLICY_CYPHER` 는 `LIVES_AT OR WORKS_AT` 둘 다 fetch 함.
- 즉 **거주 L2 + 직장 L1 (사업 구간)** 인 agent 는 *direct + spillover 혼재* 신호를 받음.
- 본 보고서 (`day_health_check` 출력) 의 L2 격차 수치는 이 혼재된 평균.

### 정밀 측정 (`L2_PRECISE_6D.md` 참조)
거주 L2 를 직장 위치별 4 case 로 분리:

| Case | 의미 |
|---|---|
| `pure_L2` | 거주 L2 ∩ 직장 비강남(또는 없음) — **진정한 spillover** |
| `mixed_L2_L1work` | 거주 L2 ∩ 직장 L1 — direct + spillover 혼재 |
| `both_walk` | 거주 L2 ∩ 직장 L2 — 양쪽 도보권 |
| `L2_gangnam_other` | 거주 L2 ∩ 직장 강남 비도보권 |

→ **`pure_L2` 의 DID 효과만 보고서의 "spillover 지표" 로 인용**. 그 외 케이스는 보조 분석.

---

## 7. 자동 보고서 (`FINAL_REPORT_6D.md`) 수치 해석 가이드

| 자동 보고서 항목 | 해석 시 주의 |
|---|---|
| 매출 추이 (직접 영향) | 영향 없음 (Plan/매출 정상). 안정적 신호. |
| 매출 추이 (간접 영향 = L2) | **이 한계 문서의 §6 정밀 측정 결과로 보완 필요** |
| 단골 vs 신규 비율 | 5/20 visited Memory 누락으로 단골 비율 살짝 underestimate |
| trigger 분포 | 5/19 의 appointment 부풀림 가능. 5/20 이후만 따로 보면 안전 |
| 인터뷰 답변 | 5/18 / 5/20 의 일부 인용이 빠질 수 있음 (페르소나 답변 풍부도 살짝 ↓) |
| 자치구 매출 표 | 매장 위치 기준 (Plan/INCLUDES) — 영향 없음 |

---

## 8. 원인 요약 (재발 방지)

- 시뮬 코드의 `resume` 메커니즘이 *day 별 retry agent 있을 때 Night Phase 전체를 재실행*.
- Conversation / Memory 가 `CREATE` 라 중복 적재 — 진정한 멱등 아님.
- **다음 라운드 개선 후보**:
  - Night Phase 에도 idempotency (Conv id 기반 MERGE 또는 done_<day>_night.json 체크포인트).
  - 또는 시뮬 재시작 시 `--skip-night-if-resume` 옵션.
  - Night Phase 별도 entry point 분리 (`run_night.py`) → 본 작업과 독립적으로 멱등 보장.

---

## 9. 보고서에 명시할 단일 문구 (요약)

> 본 시뮬레이션은 2026-05-18 ~ 5/23 (6일) 풀런이며, 실행 중 두 차례 silent kill 이 발생해 5/18 Conversation·5/20 visited Memory 가 누락되었다. 두 누락 모두 Plan·매출 등 핵심 지표에는 영향 없으나, Conv·KNOWS_POI 기반 보조 지표는 약간의 과소측정 가능성을 안고 있다. P008 spillover 측정 시 거주 L2 + 직장 L1 케이스가 direct 효과를 일부 포함하므로, 진정한 spillover 는 `L2_PRECISE_6D.md` 의 `pure_L2` DID 수치를 기준으로 한다.
