# 개발 계획 — 구조 분석 + 광역 소비(A)·인간형 소비(B) 재설계

작성 시점 기준 코드(`scripts/sim/*`) 전수 검토. 1부는 구조/성능 진단, 2부는 두 문제의 해결 설계(최신 논문 근거 포함).

---

# 1부. 아키텍처 구조 요약

## 1.1 일일 파이프라인 (`run_simulation.py` → `process_one`)

```
Day 루프(순차)  ─ agent ThreadPoolExecutor(workers)
  └ process_one(aid, day):
      ① build_dawn_context  (7개 Cypher: persona/state/memory/appointment/policy/social/knows_poi)
      ② grant 적용          (apply_grant_to_prev_state → 어제 State.balance += 지원금)  ※grant일이면 ctx 재빌드
      ③ Stage1 (LLM)        동선/시간/카테고리/anchor + reasoning/trigger   [stage1_intent.py]
      ④ Stage2 (LLM)        이벤트별 POI 선택 + actual_spent + policy_spend + 만족도  [stage2_poi.py]
      ⑤ merge_to_final_events → validate_policy_spend → track_policy_usage
      ⑥ write_plan (:Plan)-[:INCLUDES]->(:POI)
      ⑦ night_finalize_yesterday (visited Memory + KNOWS_POI) / night_create_state (잔액·mood·fatigue)
  └ Night2: 상호작용 쌍 선정 + 의도분류 LLM → Conversation(약속/소문) 적재
```

핵심 데이터 흐름: **소비액·만족도·정책사용액은 전부 Stage2 LLM이 결정**, 정책 cap만 사후 추적. State.balance = 전날잔액 − 오늘지출.

---

# 2부. 구조적·비논리적·성능 문제 진단

## 2.1 구조/논리 결함

### ❶ 공간 범위 폐쇄성 — **Problem A의 직접 원인**
- `stage1_intent.py` `_format_dawn_blocks`(L389~393)가 LLM에 줄 수 있는 zone anchor 코드를 **거주 동·직장 동 단 2개로 못박음**: `## zone anchor 코드 (반드시 이 값들 중 하나만 사용)`. SYSTEM_PROMPT(L358)도 "거주/직장 동 코드를 그대로 복사".
- Stage2 후보도 `STAGE2_CANDIDATE_CYPHER`(dawn_context L181)가 `IN_DONG {code:$dong_code}` — **anchor 동 내부 POI만** fetch. fallback도 같은 자치구(district)까지.
- `resolve_dong`(stage2 L90)은 8자리 아닌 코드를 **home_dong으로 되돌림** → 설령 LLM이 다른 동을 시도해도 무효처리.
- 결과: 도시 전역 상권 매력(홍대·강남·동대문)을 표현할 구조가 **아예 없음**. 주말 광역 이동 불가능.
- ※ 다행히 스키마(`_check_anchor`)는 임의 `zone:<8자리>`를 허용 → **수정은 프롬프트+후보데이터 차원**, 그래프 스키마 변경 불필요(저위험).

### ❷ 소비의 rule-base 앵커링 — **Problem B의 직접 원인**
- `stage2_poi.py` SYSTEM_S2(L279)가 **`daily_wd`(고정 일일예산)를 직접 노출** → LLM이 그 값 근처로 회귀(객단가가 외출횟수의 함수가 됨).
- `_ensure_positive_spend`(L71~84) fallback이 **`daily_wd × 0.4`로 cap** → 누락 거래도 daily_wd에 종속.
- 지원금은 `apply_grant_to_prev_state`로 **balance에만 가산**되고 소비 결정식엔 안 들어감 → 저소득이 grant 받아도 소비 안 늘림(**역진 MPC**: 검증서 하 6.0% < 중상 43.7%).
- `daily_wd`는 페르소나 생성 시 BDC 소비분위에서 뽑은 **단일 고정 숫자**(`generate_agents.py`) → 예산제약·저축동기·소득탄력성 부재. 소비 지니 0.25(실측 0.42)로 과소.

### ❸ 소비 결정의 신호 빈약
- Stage2 후보에 거리·avg_satisfaction만 있고 **객단가/가격대·혼잡도·매력도** 신호 없음 → 금액 결정이 통념 fallback에 의존.

### ❹ State JSON-in-string 중복 파싱
- `grant_received`/`grant_remaining`/`policy_used`/`policy_lifecycle`이 문자열 JSON으로 저장돼 `_read_state_json`·`_parse_*`가 여러 파일에서 반복 파싱. 논리오류는 아니나 유지보수 부담·버그 표면적↑.

## 2.2 성능 저하 지점

| # | 위치 | 문제 | 개선 |
|---|---|---|---|
| P1 | `build_stage2_candidates*` (dawn_context L510~543) | 호출마다 **새 `driver_session()`** 오픈. agent-day당 이벤트그룹 수만큼 + dawn 7 + night 다수 → 세션 수십 개 × 7500 × workers | 한 agent-day는 **세션 1개 재사용**(컨텍스트 매니저 주입) |
| P2 | `fetch_candidates_for_events` | 이벤트 그룹별 Cypher **순차 round-trip** | `UNWIND $groups` 단일 쿼리로 배치 |
| P3 | grant일 `build_dawn_context` **2회** (process_one L212) | 전체 7쿼리 재조회 | 잔액만 in-place 패치 |
| P4 | `STAGE2_CANDIDATE_CYPHER`의 `point.distance` 매 호출 계산 | POI×anchor 거리 런타임 산출 | anchor-동 거리 사전계산/캐시 |
| P5 | 거리 정렬 강제 `ORDER BY km ASC` | 근접 POI 편향(②와 결합해 locality 강화) | 매력도·소득 신호 결합 정렬 |

> 권장: P1·P2를 먼저 잡으면 동일 throughput에서 Neo4j 커넥션 압박이 크게 준다(현재 workers의 2~3배 pool 권장 주석과도 정합).

---

# 3부. Problem A 해결 설계 — 광역 소비(주말 홍대·동대문)

## 3.1 진단 재확인
에이전트는 **갈 수 있는 곳 자체가 거주/직장 동으로 제한**돼 있다. 사람은 평일엔 생활권, 주말엔 매력 상권으로 **광역 이동**한다. 빠진 두 요소: ⓐ **도시 전역 상권 매력도(attraction)**, ⓑ **거리 비용(distance decay)** 과 그 둘의 결합 선택.

## 3.2 논문 근거
- **Huff 소매 중력모형** (Huff 1964, *Defining and Estimating a Trading Area*, J. Marketing): 소비자가 상권 j를 택할 확률
  `P(j) ∝ 매력도_j / 거리_ij^β`. 소매 목적지 선택의 표준 공간상호작용 모형.
- **AgentMove** (Feng et al., 2024/25): 도시를 **multi-layer graph**로 인코딩하고 ‘spatio-temporal memory + world knowledge + collective pattern’으로 다음 목적지를 예측 — LLM 단독의 환각·일관성 문제를 **집단 패턴 prior**로 보정.
- **LLM × gravity 결합** 흐름 (Toward LLM-Agent-Based Modeling of Transportation Systems, arXiv:2412.06681; TrajLLM arXiv:2502.18712): 통계 모형(중력/이산선택)을 prior로 깔고 LLM이 reasoning으로 채택.

## 3.3 구현안 (단계적, 저위험순)

**① 광역 상권 허브 카탈로그 (데이터, 이미 보유)**
- `output/stats/dong_context.json`의 `b069_sales`(동별 매출지수)로 **매력도** 산출 → 상위 상권 동·자치구 랭킹(강남역·홍대·동대문·명동·여의도 등) + 대표 카테고리(쇼핑·여가·식사).
- 산출물: `hub_catalog.json` = [{dong_code, name, attraction(=b069_sales), top_l1s}]. **신규 데이터 수집 불필요.**

**② Stage1 zone 후보 확장 (프롬프트 구조 변경)**
- `_format_dawn_blocks`의 "반드시 이 둘만" 제약을 제거하고, **오늘 갈 수 있는 zone 후보 목록**을 제공:
  - (a) 거주/직장 동(생활권) (b) 약속·친구 동 (c) 과거 방문 동(KNOWS_POI) (d) **매력 허브 Top-K**.
- 의도/요일 가중: **주말 + 여가·쇼핑·식사 의도일 때 허브 가중↑**, 평일·생활밀착(편의점·약국)은 생활권 고정 → 동시에 2.1의 평일 과다 이동도 억제.

**③ 목적지 선택을 Huff prior + LLM 하이브리드**
- 광역 이동을 LLM에 전부 맡기지 않고, **Huff 확률로 허브 후보를 샘플링**해 "오늘 끌리는 광역 상권 후보"로 Stage1에 제시(거리·매력·`mobility_level`(이동분위) 반영). LLM은 reasoning으로 채택/기각 → AgentMove의 collective-pattern prior 역할(환각·쏠림 완화).
- β(거리 민감도)·허브 Top-K·주말 가중은 **검증 지표(공간 cosine 유지 + 분산↑)로 튜닝**.

**④ Stage2 후보 fetch를 anchor 동 → 선택된 목적지 동으로** 일반화(쿼리는 그대로, dong_code만 허브 코드 허용). `resolve_dong`의 home 되돌림을 **허브 카탈로그 화이트리스트면 통과**로 완화.

**기대효과**: 공간 cosine(0.94) 유지하면서 동(洞) 단위 과집중(시뮬 0.312)을 완화하고 주말 광역 분포를 재현.

---

# 4부. Problem B 해결 설계 — 인간형 소비 + 저소득 정책반응

## 4.1 진단 재확인
소비가 **고정 `daily_wd`에 앵커**돼 (ⓐ 소득·지원금에 비탄력, ⓑ 개인 간 격차 압축, ⓒ 저소득 grant 미사용=역진 MPC). "사고·판단을 에이전트에 위임" 못 하고 rule-base에 가깝다.

## 4.2 논문 근거
- **EconAgent** (Li et al., ACL 2024, *Large Language Model-Empowered Agents for Simulating Macroeconomic Activities*, arXiv:2310.10436): LLM이 **소비성향 propensity ∈ [0,1]** 을 출력 → **소비액 = propensity × 가용자산(소득+저축)**. 저축 많을수록 성향↓, 정책·금리에 반응. **rule-based/learning-based보다 현실적 거시현상** 산출이라 보고.
- **한계소비성향(MPC) 이질성** (Jappelli & Pistaferri 2014, *Fiscal Policy and MPC Heterogeneity*, AEJ:Macro): **저소득일수록 MPC↑**. 우리 역진을 바로잡는 타깃값.
- **항상소득가설** (Friedman 1957): 일시 지원금(transitory)도 유동성제약 가구(저소득)에선 즉시 소비로 전환 → 저소득 grant 소비↑가 정상.

## 4.3 구현안

**① `daily_wd` 직접노출 폐기 → 소비성향 출력 (EconAgent 방식)**
- Stage2(또는 Stage1.5)에서 하루 단위로 LLM이:
  - (a) **오늘 가용예산** = f(State.balance, 일상 소득흐름, **지원금 잔여**) 인식,
  - (b) **소비성향 propensity[0,1]** 결정(페르소나 소득·저축·mood·정책 반영),
  - (c) **오늘 총지출 = propensity × 가용예산**,
  - (d) 거래별 금액은 **동선·카테고리 비율로 LLM이 분배**(사용자가 앞서 제안한 "하루 총액 비율 배분"과 정합).
- `daily_wd`는 프롬프트에서 제거하고 **성향 prior**로만 내부 참조(절약/소비형 범위 가드).

**② 가용예산에 지원금 포함 → 역진 자동 해소**
- 가용예산이 grant만큼 커지므로 **같은 propensity라도 지출↑**. 저소득은 baseline가 작아 grant의 상대비중↑ → **MPC↑**(저소득>고소득) 회복. 별도 if-분기 없이 구조로 해결.

**③ 저소득 정책반응 페르소나 렌즈 실효화**
- 현재 SYSTEM_PROMPT(stage1 L258)의 "소비분위 낮을수록 작은 혜택도 크게 와닿음"이 daily_wd 고정에 의해 무력화됨. **propensity 구조에서는 실제로 작동**. + "지원금 = 추가 가처분소득" 인식을 저소득 페르소나에 명시.

**④ 예산제약·저축동기 실제화**
- `balance`가 **소비 상한으로 실제 작동**(잔액 부족→억제, 여유→일부 저축/이연). EconAgent의 savings 반응 도입 → 소비 분산(지니) 현실화.

**⑤ (선택) 객단가 신호 주입**
- Stage2 후보에 카테고리 평균 객단가(BDC `consumption_detail`/decile)를 **참고치**로 제공 → 금액 현실화(rule이 아니라 prior).

**기대효과**: 소비 지니 0.25→실측 0.42 방향으로 상승, MPC 역진(하 6%<중상 44%) → 정상화(저소득↑). 발표 한계 ②③ 동시 해소.

---

# 5부. 우선순위·리스크·검증 연계

1. **데이터 선행**: 허브 카탈로그(A①)·객단가 prior(B⑤) 생성 — 기존 `output/stats`만으로 가능.
2. **성능 선정리**: P1(세션 재사용)·P2(후보 배치) — 구조 변경 전 비용 절감.
3. **A 적용**: Stage1 zone 후보 확장 + Huff prior.
4. **B 적용**: Stage2 propensity 소비 + 가용예산(지원금 포함).
5. **재검증**: 기존 지표(`validate_korea.py`, `validate_backup_new.py`)로 before/after — 공간 분산↑·소비 지니↑·MPC 정상화 확인.

**공통 리스크 & 가드레일**
- LLM 자유도↑ → 환각·일관성. → **통계 prior 병행**: A는 Huff 확률 샘플링, B는 propensity 범위 가드(저소득 하한·고소득 상한)·예산상한. (AgentMove의 collective-prior, EconAgent의 [0,1] 제약과 동일 철학.)
- 두 변경 모두 Stage 프롬프트/후보 데이터 레벨 — **그래프 스키마 불변(저위험)**.

## 참고문헌
- Huff, D. L. (1964). Defining and Estimating a Trading Area. *Journal of Marketing, 28*(3), 34–38.
- Li, N., et al. (2024). EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities. *ACL 2024*. arXiv:2310.10436.
- Feng, J., et al. (2024/2025). AgentMove: LLM-based agentic framework for zero-shot next location prediction. arXiv.
- Toward LLM-Agent-Based Modeling of Transportation Systems: A Conceptual Framework (2024). arXiv:2412.06681.
- TrajLLM: A Modular LLM-Enhanced Agent-Based Framework for Realistic Human Trajectory Simulation (2025). arXiv:2502.18712.
- Jappelli, T., & Pistaferri, L. (2014). Fiscal Policy and MPC Heterogeneity. *AEJ: Macroeconomics, 6*(4), 107–136.
- Friedman, M. (1957). *A Theory of the Consumption Function*. Princeton University Press.
