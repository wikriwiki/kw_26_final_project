# 시뮬레이션 데이터 한계 및 해석 주의사항 (5일 풀런)

> 본 보고서의 모든 수치를 해석할 때 반드시 함께 읽어야 할 한계 문서.
> 시뮬 기간: 2026-05-18 ~ 2026-05-22 (5일).
> 정책 시행일: 2026-05-20 ~ 2026-05-22 (P008 강남역-역삼역 보행친화거리, 3일).
> baseline: 2026-05-18·5/19 (정책 미활성). 단, 5/18 은 아래 한계로 인해 **5/19 단일 일자 baseline 권장**.

---

## 1. 시뮬 흐름 변경 — 7일 → 5일로 단축

원안: 2026-05-18 ~ 5/24 (7일). 실제: **2026-05-18 ~ 5/22 (5일)** 로 단축.

- 마감 데드라인(2026-05-29 21:00) 까지 6일 이상 시뮬을 완료할 수 없음. 시뮬 1일 = 약 12시간(본 작업 9.5h + Night Phase 2.5h) 소요.
- 시뮬 v8 (`--start 2026-05-21 --days 4`) 진행 중 5/22 Night Phase 완료 시점(= 5/23 본 작업 시작 신호)에 종료.
- 5/23·5/24 는 시뮬하지 않음. **분석에서 5/23 이후 통째로 제외**.

정책 활성 3일(5/20·5/21·5/22), baseline 2일(5/18·5/19). DID 분석에 충분.

---

## 2. 5/18 — Conversation 데이터 0건

### 무엇이 일어났나
- 원래 시뮬에서 12,477건의 Conversation 정상 생성 (Night Phase 2).
- WSL OOM 추정으로 시뮬 silent kill → resume 시 Night Phase 2 가 재실행되며 12,455건 *중복* 추가 (총 24,932).
- 분석 신뢰성 확보 위해 **Conv 5/18 통째 삭제**.

### 결과
- 5/18 에는 상호작용(약속·추천·이슈·기타) 데이터가 **0건**.
- 5/18 Plan / State / visited Memory (67,806) 는 정상 보존 (CREATE 이지만 done_aids 로 agent 중복 차단).
- 5/18 rumor Memory 5,388 건은 그대로 유지 — *부모 Conv 가 삭제됐지만 자식 rumor 는 1배 정상 수치* (5/19=5,349, 5/20=5,291, 5/21=5,143 와 비슷). "출처 Conv 없는 가십"이지 수치가 부풀려진 것 아님.

### 해석 시 주의
- baseline 1인당 외출·매출 (layered DID) 은 5/18·5/19 평균으로 계산 가능 — Plan·매출 신호 정상.
- 그러나 **trigger=appointment 분포·약속 횟수 같은 Conv 기반 지표는 5/19 단일 일자로만 계산**.
- Conv 기반 지표는 baseline n=1, 정책 활성 n=3. 통계적 유의성보다 효과 크기 위주로 해석.

---

## 4. 5/19 Plan 의 잔여 영향 (영구)

Day 2 (5/19) 처리 시점에 5/18 의 24K 중복 Conv 가 *아직 살아있었음*. Dawn 컨텍스트가 그것을 fetch 해서 LLM 에 노출했고, 일부 reasoning 이 그 영향을 받음.

- 5/19 의 `trigger=appointment` 카운트가 **부풀려졌을 가능성**. 5/20 이후와 비교해 비정상 증가면 본 영향으로 해석.
- 5/19 Plan/State 본 적재는 정상.
- 5/20 dawn 시점엔 이미 5/18 Conv 가 삭제되어 있어 이후 날짜엔 영향 없음.

---

## 5. 5/23·5/24 — 미시뮬 (분석 제외)

5일 단축으로 5/23·5/24 는 시뮬하지 않음. 본 보고서는 5/18~5/22 만 집계. 모든 차트·DID·정밀측정이 `--days 5` 한정.

---

## 6. L2 spillover 측정의 한계와 정밀 측정

### 한계
- `day_health_check.py` 10-D 의 L2 (도보권 거주) 그룹은 *거주지만* 기준으로 분류.
- 그러나 P008 정책의 `POLICY_CYPHER` 는 `LIVES_AT OR WORKS_AT` 둘 다 fetch.
- 즉 **거주 L2 + 직장 L1 (사업 구간)** 인 agent 는 *direct + spillover 혼재* 신호를 받음.

### 정밀 측정 (`L2_PRECISE_5D.md` 참조)
거주 L2 를 직장 위치별 4 case 로 분리:

| Case | 의미 |
|---|---|
| `pure_L2` | 거주 L2 ∩ 직장 비강남(또는 없음) — **진정한 spillover** |
| `mixed_L2_L1work` | 거주 L2 ∩ 직장 L1 — direct + spillover 혼재 |
| `both_walk` | 거주 L2 ∩ 직장 L2 — 양쪽 도보권 |
| `L2_gangnam_other` | 거주 L2 ∩ 직장 강남 비도보권 |

→ **`pure_L2` 의 DID 효과만 "spillover 지표"로 인용**. 그 외는 보조 분석.

---

## 7. 자동 보고서 (`FINAL_REPORT_5D.md`) 수치 해석 가이드

| 항목 | 해석 시 주의 |
|---|---|
| 매출 추이 (직접 영향, L1) | 영향 없음 (Plan/매출 정상). 안정적 신호. |
| 매출 추이 (간접 영향, L2) | §6 정밀 측정 결과(`L2_PRECISE_5D.md`)로 보완 |
| 단골 vs 신규 | 5/20 visited Memory 복구로 **정확** |
| trigger 분포 (5/19) | appointment 부풀림 가능 — 5/20 이후만 보면 안전 |
| trigger 분포 (5/21) | "새가게_탐색" 약간 부풀림 가능 (5/20 회상 부재 잔여 영향) |
| 인터뷰 답변 | 5/18 상호작용 인용 빠짐 (5/19~5/22 로 보완) |

---

## 8. 원인 요약 (재발 방지)

- 시뮬 `resume` 메커니즘이 *day 별 retry agent 있을 때 Night Phase 전체를 재실행* → Conversation/Memory `CREATE` 중복.
- visited Memory 가 `day_idx==0` 에서 skip → 재시작 경계일(5/20)에서 누락.
- **다음 라운드 개선 후보**:
  - Night Phase idempotency (Conv id MERGE 또는 done_<day>_night.json 체크포인트).
  - `--skip-night-if-resume` 옵션.
  - 재시작 시 `--start` 가 첫날이어도 yesterday visited Memory 생성 (이전 시뮬 데이터 있으면).

---

## 9. 보고서에 명시할 단일 문구 (요약)

> 본 시뮬레이션은 2026-05-18 ~ 5/22 (5일) 풀런이며, 실행 중 두 차례 silent kill 이 발생해 5/18 Conversation 이 누락(삭제)되고 5/20 visited Memory 가 누락되었다. 5/20 visited Memory 와 KNOWS_POI 는 사후 복구했으나, 5/18 Conv(0건)와 5/21 trigger 분포의 미세 왜곡은 잔존한다. 두 사고 모두 Plan·매출 등 핵심 지표에는 영향이 없다. P008 spillover 측정 시 거주 L2 + 직장 L1 케이스가 direct 효과를 일부 포함하므로, 진정한 spillover 는 `L2_PRECISE_5D.md` 의 `pure_L2` DID 수치를 기준으로 한다.
