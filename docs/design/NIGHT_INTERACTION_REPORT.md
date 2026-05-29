# 🌙 Night Phase 2 — 에이전트 상호작용 추가 구현 리포트

**작성일**: 2026-05-15 (노션 의도분류 v2 정합 반영) · **시뮬 대상**: 풀런 결과 (14,560 agent × 3일, 강남 vs 비강남 DID)

이 문서는 **3가지를 한꺼번에 정리**합니다:

1. 실제 시뮬에서 부여된 두 agent의 **기억·일정·상호작용** 데이터
2. 이번 세션에 새로 추가된 **Night Phase 2** (상호작용 대상 선정 + 의도 분류) 내용
3. 전체 시뮬 시스템의 **현재 구현 상태** 한눈에

---

## 1. 실제 시뮬 결과 — Agent 2명 표본

### 1.1 강남구 거주 + 정책 P001 영향권 agent

`AGT_11680531_M_40대_003`

**페르소나**:
- 40대 M / 생애주기 = 자녀 퇴학, 자동차·여가 소비 증가
- 직업: **교육 컨설팅 실장** · 소득 중상 · 성향 보통
- 평일 일평균 소비: **146,970원**
- 거주: **강남구 논현2동 — 동양파라곤** · 직장: 논현1동 WIN빌딩 (통근 12분)
- 라이프스타일: "자녀 교육 후 여가 소비 증가"

**State 시계열 (Day 0 시드 → Day 3)**:
| 날짜 | balance | mood | fatigue | yesterday_sat |
|---|---|---|---|---|
| 2026-04-30 | 4,000,000 | 0.50 | 0.30 | 0.50 |
| 2026-05-01 | 3,952,000 | 0.52 | 0.70 | 0.57 |
| 2026-05-02 | 3,904,000 | 0.55 | 0.80 | 0.60 |
| 2026-05-03 | 3,868,000 | **0.57** | 0.70 | 0.62 |

→ mood가 0.50 → 0.57로 점진 상승, balance가 매일 ~50K씩 감소 (외출 이벤트당 8~12K 소비).

**📅 2026-05-02 (P001 활성 D+0) 하루 일정**:
| time | anchor | category | poi | satisfaction | intent |
|---|---|---|---|---|---|
| 08:00 | residence | 집 | 동양파라곤 | 0.52 | 기상 |
| 08:30 | residence | 편의점 | 동양파라곤 | 0.52 | 아침 간단 음식 |
| 09:00 | residence | 집 | 동양파라곤 | 0.59 | 아침 식사 및 휴식 |
| 10:00 | zone:11680531 | **식사** | 디에이치 컴퍼니 | 0.66 | 점심 식사 |
| 12:00 | residence | 집 | 동양파라곤 | 0.59 | 휴식 및 TV 시청 |
| 14:00 | zone:11680531 | **카페** | 이지스터커피&베이커리 | 0.54 | 휴식 및 커피 마시기 |
| 16:00 | residence | 집 | 동양파라곤 | 0.56 | 오후 휴식 및 책 읽기 |
| 19:00 | zone:11680531 | **식사** | 라프푸드코트 | 0.74 | 저녁 식사 |
| 21:00 | residence | 집 | 동양파라곤 | 0.69 | 취침 준비 |

→ 정책 대상(식사·카페·디저트) 외출 3건. **만족도 룰 +0.10 효과** 반영됨 (강남+식사·카페 매칭).

**📅 2026-05-03 (D+1) 일정 중 핵심 이벤트**:
- 12:30 카페 (이지스터커피&베이커리) — **재방문** (어제 visited Memory 효과로 KNOWS_POI affinity 상승)
- 17:00 식사 (타쿠타쿠) — intent: **"정책 환급 대상 식사로 저녁 식사"** ← LLM이 P001 description 자율 인식

**visited Memory (D+1, D+2 누적)**:
| day | poi | satisfaction | importance |
|---|---|---|---|
| 2026-05-01 | 동양파라곤 | 0.68 | 1.52 |
| 2026-05-01 | WIN빌딩 | 0.64 | 1.46 |
| 2026-05-02 | 라프푸드코트 | 0.74 | 1.61 |
| 2026-05-02 | 디에이치 컴퍼니 | 0.66 | 1.49 |
| 2026-05-02 | 이지스터커피&베이커리 | 0.54 | 1.31 |

**KNOWS_POI 단골 Top 5** (시뮬 종료 시점):
| POI | 카테고리 | visit_count | avg_sat | affinity | source |
|---|---|---|---|---|---|
| 라프푸드코트 | 식사 | 1 | 0.74 | 0.60 | visited |
| WIN빌딩 | (직장) | 1 | 0.64 | 0.59 | visited |
| 동양파라곤 | (거주) | 6 | 0.58 | 0.57 | visited |
| 디에이치 컴퍼니 | 식사 | 1 | 0.66 | 0.56 | visited |
| 이지스터커피&베이커리 | 카페 | 1 | 0.54 | 0.55 | initial → visited |

### 1.2 종로구 거주 + 정책 영향 없는 baseline agent

`AGT_11110700_M_50대_001`

**페르소나**:
- 50대 M / 생애주기 = **은퇴**
- 직업: 은행 퇴직 후 자문역 · 소득 중상 · 성향 **절약형**
- 평일 일평균 소비: **2,863원** (강남 대비 약 1/50)
- 거주: **종로구 숭인제1동 — 롯데캐슬천지인** · 직장 없음
- 라이프스타일: "조용한 생활, 건강 유지에 집중"

**State 시계열**:
| 날짜 | balance | mood | fatigue | yesterday_sat |
|---|---|---|---|---|
| 2026-04-30 | 500,000 | 0.50 | 0.30 | 0.50 |
| 2026-05-01 | 440,000 | 0.53 | 0.70 | 0.60 |
| 2026-05-02 | 380,000 | 0.54 | 0.90 | 0.56 |
| 2026-05-03 | 308,000 | 0.55 | 0.90 | 0.57 |

→ balance 시작액이 강남(4M) 대비 1/8 수준(500K). 일별 ~60-70K 소비. **fatigue 0.90으로 매우 높음** — 은퇴자라 이벤트 11개씩 처리하면 룰 상 피로 누적.

**📅 2026-05-02 일정 (집 위주)**:
| time | category | poi | satisfaction | intent |
|---|---|---|---|---|
| 07:30 | 집 | 롯데캐슬천지인 | 0.59 | 기상 |
| 08:10 | 편의점 | 롯데캐슬천지인 | 0.54 | 아침 식자재 및 음료 구매 |
| 09:30 | **마트** | 롯데캐슬천지인 | 0.43 | 주간 식자재 쇼핑 |
| 12:30 | **한식** | 롯데캐슬천지인 | 0.51 | 점심 식사 |
| 14:00 | 집 | 롯데캐슬천지인 | 0.62 | 건강 운동 (가벼운 스트레칭) |
| 15:30 | **디저트** | 롯데캐슬천지인 | 0.54 | 간단한 디저트 |
| 19:00 | **한식** | 롯데캐슬천지인 | 0.56 | 저녁 식사 |
| 20:00 | 집 | 롯데캐슬천지인 | 0.67 | 책 읽기 및 취침 준비 |

→ 모든 외출이 anchor=residence 안에서 일어남 (집 근처). **POI도 거주지(롯데캐슬천지인) 그대로** — 직장 없고 이동성 낮은 은퇴자 패턴.

**KNOWS_POI**:
| POI | visit_count | avg_sat | affinity | source |
|---|---|---|---|---|
| 롯데캐슬천지인 | 10 | 0.54 | 0.51 | visited |

→ **단일 POI만 10번 방문**. 시뮬 사슬은 잘 작동하지만 외출 다양성 X.

### 두 agent 비교 인사이트

| 항목 | 강남 40대 직장인 | 종로 50대 은퇴자 |
|---|---|---|
| 평일 소비액 | 146,970원 | 2,863원 (1/50) |
| 외부 외출 카테고리 | 식사·카페·미용 등 | 거의 집 안에서만 |
| 정책 P001 효과 | ✅ 식사·카페 방문 빈도 ↑, intent에 "정책 환급" 언급 | ❌ 강남 거주 아니라 컨텍스트 미주입 |
| KNOWS_POI 다양성 | 5개 (단골 + 신규) | 1개 (집만 반복) |
| Memory 누적 | 8개 commerce 방문 | 10개 모두 같은 POI |
| Conversation 참여 | 0건 (5/02 매칭 풀에 안 잡힘) | 0건 |

---

## 2. 이번 세션에 추가된 내용

### 2.1 Night Phase 2 — 상호작용 대상 선정 알고리즘

📄 [`scripts/sim/night_interaction.py`](../scripts/sim/night_interaction.py)

**입력**: 매일 자정 시뮬 종료 시점의 Plan·State·Memory·KNOWS

**3축 점수**:
```
InteractionScore(A, B) = 0.4·Exposure + 0.3·Relationship + 0.3·Urgency
```

- **Exposure** (0~1): 같은 동·±1시간 시간 겹침. `freq×0.6 + avg_overlap×0.4`
- **Relationship** (0~1): KNOWS 엣지(colleague 0.6 / neighbor 0.4) + 과거 Conversation count
- **Urgency** (0~1): 정책 인지 비대칭(+0.5) + Memory(policy/sns) 누적 차이 + mood 극단(<0.3 우울 / >0.7 흥분) + fatigue 보정

**임계값·매칭**:
- 임계값 0.3 미만 제외
- agent당 max 2회 그리디 매칭 (점수 내림차순)
- 후보 추출: 같은 (dong, hour) bucket + 인접 hour + KNOWS 양쪽 외출

**5/02 실측**:
- 후보 쌍 **22,381**
- 점수 ≥ 0.3 통과 211
- 그리디 매칭 후 **148쌍 선정**
- 처리 시간 **8.7초** (Python in-memory)
- exposure-주도 (현 시뮬은 정책 lifecycle·Conversation·감정 극단 다 0이라 urgency·relationship 작음)

### 2.2 의도 분류 LLM (노션 v2 정합)

📄 [`scripts/sim/night_intent_llm.py`](../scripts/sim/night_intent_llm.py)

노션 v2(2026-05-15 갱신) 명세 그대로 구현:

**입력**: MATCHING_ANALYSIS + AGENT_A + AGENT_B 3 블록
- persona는 `agent_id`, `job`, `lifestyle`, `mood`, `fatigue`만 사용
- daily_log는 시간순 전체 이벤트 (정규화 한 줄: `time | dong | poi | category | activity`)

**LLM**: SGLang/vLLM 자동감지 — 기본 Qwen3-32B-AWQ (Pydantic 검증 + 재시도 2회)

**출력 JSON 스키마** (v2, flat topic + 확장 plan_signal):
```json
{
  "intent": "약속|이슈|추천|기타",
  "initiator_id": "<AGENT_A.agent_id>",
  "recipient_id": "<AGENT_B.agent_id>",
  "topic_type": "policy|poi|category|none",
  "topic_value": "<입력에서 추출한 값 또는 null>",
  "plan_signal": {
    "should_inject": false,
    "target_day_offset": null,
    "target_time": null,
    "meeting_location_hint": null
  }
}
```

**v1 → v2 변경점**:
- ❌ `speech_act` 폐기 (intent 4종으로 충분)
- ❌ `topic_entity {type, value}` 중첩 → ✅ flat `topic_type` + `topic_value`
- ❌ `time_horizon: "D+N"` 문자열 → ✅ `target_day_offset` int + `target_time` "HH:MM" + `meeting_location_hint` 자유 문자열
- ❌ `:WITH` 엣지 폐기 → ✅ `PARTICIPATES_IN.role` 속성으로 흡수 (`initiator`/`recipient`)
- ❌ 이슈 `OCCURRED_IN(Dong)` 분기 폐기 → ✅ topic_type ∈ `{policy}` 단일 (`ABOUT_POLICY`만)
- ✅ 이슈·추천 둘 다 recipient에 **Memory{type:'rumor'}** 적재 (노션 §5)
- ✅ `(Memory)-[:FROM_CONVERSATION]->(Conversation)` 신규 엣지
- ✅ `importance = urgency × 0.6 + relationship × 0.4` 산식 (노션 §9)
- ✅ Memory id 형식 `MEM_RUMOR_<recipient>_D<YYYYMMDD>_<n>`
- ❌ 기타 `KNOWS.strength +0.02` 보강 폐기 → ✅ 기타는 Conversation만 (노션 §4)

**Intent별 그래프 적재 (v2)**:
| intent | Conversation 필드 | Memory | 추가 엣지 |
|---|---|---|---|
| **약속** | `should_inject=true`, `target_day_offset`, `target_time`, `meeting_location_hint` | ❌ | `meeting_location_hint`가 POI 매칭 시 `(c)-[:MENTIONS_POI]->(poi)` |
| **이슈** | `topic_type='policy'`, `topic_value=<Policy.id>` | ✅ Memory{rumor} (recipient, importance) + `[:FROM_CONVERSATION]` | `(c)-[:ABOUT_POLICY]->(:Policy)` |
| **추천** | `topic_type ∈ {poi, category}` | ✅ Memory{rumor} (recipient, importance) + `[:FROM_CONVERSATION]` | `(c)-[:MENTIONS_POI]->(poi)` + `(b)-[:KNOWS_POI{source:'rumor'}]->(poi)` MERGE + `(m)-[:ABOUT_POI]->(poi)` |
| **기타** | `topic_type='none'` | ❌ | base만 (PARTICIPATES_IN×2) |

**Plan 자동 주입 (Dawn ④)**:
- Conversation에 저장된 `should_inject=true AND date(c.day) + duration({days:target_day_offset}) = date(today)`로 조회
- 별도의 `:SEEDS_PLAN` 엣지는 두지 않고 Conversation 자체를 source of truth로 사용
- `dawn_context.APPOINTMENT_CYPHER`가 daily Dawn에서 자동 실행

**5/02 148쌍 실측** (v1 시점 데이터 — v2 재실행 시 분포는 달라질 수 있음):
- 처리 시간 **297초 (5분, 16 workers)**
- 결과 분포: 기타 143 / 이슈 3 / 추천 2 / 약속 0
- 적재된 Conversation **188** (이전 dry-run 20 + 148 + 일부)
- Memory(rumor) 2건 생성, MENTIONS_POI 2건
- 약속 0이라 Plan 자동 주입 미발생

**샘플 추천**:
- `AGT_11680700_F_40대_003 → AGT_11680700_M_30대_005`: 호정순대국 추천 (식사 POI)
- recipient에 KNOWS_POI{source:'rumor', affinity:0.5} MERGE + Memory{rumor, topic_type:'poi', topic_value:'호정순대국'} 생성

**샘플 이슈**:
- `AGT_11680600_F_40대_004 → AGT_11680600_M_50대_001`: P001 정책 정보 공유
- v2에서는 topic_type='policy' + topic_value='P001' → Conversation에 `[:ABOUT_POLICY]->(Policy{id:'P001'})` 직접 연결
- recipient에 Memory{rumor, source:'AGT_11680600_F_40대_004', topic_type:'policy', topic_value:'P001'} 생성

### 2.3 통합 흐름

```
매일 시뮬 종료 (Night)
       │
       ▼
[A] night_interaction.py
    └─ 후보 쌍 추출 (동·hour bucket + KNOWS) → 3축 점수 → 그리디 매칭
       │ 산출: 쌍 list (aid_a, aid_b, score, exposure, relationship, urgency)
       ▼
[B] night_intent_llm.py
    └─ 각 쌍에 LLM 호출 (16 workers, SGLang/vLLM 자동감지)
       │ 입력: MATCHING_ANALYSIS + AGENT_A·B daily_log + State
       │ 출력: intent + topic_type/value + plan_signal{should_inject,offset,time,hint}
       ▼
[C] Conversation 적재 (Cypher) — intent별 분기 (노션 §4·§5·§9)
    └─ :Conversation CREATE
       + (a)-[:PARTICIPATES_IN {role:'initiator'}]->(c)
       + (b)-[:PARTICIPATES_IN {role:'recipient'}]->(c)

       (노션 §5: rumor = 다른 agent한테 들은 모든 정보 / policy = Watchdog 시드 전용)
       importance = urgency × 0.6 + relationship × 0.4

       ├─ 약속 → should_inject + target_day_offset + target_time + meeting_location_hint
       │        → meeting_hint 매칭 시 (c)-[:MENTIONS_POI]->(poi)
       │        → D+offset Dawn ④ APPOINTMENT_CYPHER가 자동 조회 → Plan에 강제 진입
       ├─ 이슈 → recipient에 :Memory{type:'rumor', source, topic_type, topic_value, importance}
       │        + (m)-[:FROM_CONVERSATION]->(c)
       │        + topic_type='policy' 이면 (c)-[:ABOUT_POLICY]->(:Policy)
       ├─ 추천 → :Memory{rumor} (위와 동일)
       │        + (c)-[:MENTIONS_POI]->(poi)
       │        + (m)-[:ABOUT_POI]->(poi)
       │        + (b)-[:KNOWS_POI{source:'rumor', affinity:0.5}]->(poi) MERGE
       └─ 기타 → Conversation만 (별도 엣지 없음 — 노션 §4)
```

### 2.4 메인 루프 통합 (노션 매일 사이클 정합)

`run_simulation.py`의 `run_day()` 끝에 Night Phase 2 hook이 통합되어, **매일 자정 자동 실행**:

```python
def run_day(agents, today, day_idx, workers=64):
    # ... 기존: ThreadPoolExecutor로 agent별 process_one (Dawn → Plan → Phase 1/3) ...

    # ═══ Night Phase 2 (노션 다이어그램 매일 자정 자동) ═══
    from night_interaction import select_interaction_pairs
    from night_intent_llm import run_intent_classification
    pairs = select_interaction_pairs(today, verbose=False)
    if pairs:
        stats = run_intent_classification(today, pairs, workers=workers, verbose=False)
        print(f"  [Night2] Conversation +{stats['write']['created']} "
              f"by_intent={stats['write']['by_intent']} "
              f"rumor_mem={stats['write']['rumor_memory']}")
```

→ 다음 날(D+1) Dawn ④의 `APPOINTMENT_CYPHER`가
`should_inject=true AND date(c.day) + duration({days: target_day_offset}) = date($today)` 조건으로
약속 큐를 자동 조회 → Stage 1 프롬프트에 주입.

---

## 3. 전체 시뮬 시스템 구현 상태

### 3.1 아키텍처

```
┌──────────── Day 0 (1회) ────────────┐
│  scripts/neo4j_load/01–08            │
│  - District/Dong/Category/POI/Agent  │
│  - LIVES_AT·WORKS_AT·KNOWS·KNOWS_POI │
│  - State 시드 (4/30)                  │
└──────────────────────────────────────┘
              │
              ▼
┌──────────── 매일 시뮬 (Day t) ───────────┐
│  Dawn (scripts/sim/)                       │
│   ① dawn_context.py — 7종 Cypher 사전 조회 │
│   ② stage1_intent.py — 의도 LLM 호출       │
│   ③ stage2_poi.py — POI 결정 LLM 호출      │
│   ④ plan_writer.py — Plan 적재 + 만족도 룰 │
│                                            │
│  낮 (시뮬) — simulate_satisfaction         │
│                                            │
│  Night                                     │
│   ⑤ plan_writer.night_finalize_yesterday   │
│      어제 INCLUDES → visited Memory + KNOWS_POI 갱신 │
│   ⑥ 🆕 night_interaction.py                │
│      후보 쌍 + 3축 점수 + 그리디 매칭        │
│   ⑦ 🆕 night_intent_llm.py                 │
│      LLM 의도 분류 → Conversation 적재     │
│   ⑧ plan_writer.night_create_state         │
│      오늘 State CREATE (mood/fatigue 갱신) │
└────────────────────────────────────────────┘
              │
              ▼
        다음 날 Dawn으로 (체인 반복)
```

### 3.2 모듈 구성

| 모듈 | 역할 | 상태 |
|---|---|---|
| `scripts/neo4j_load/` | Day 0 정적 적재 (01~08 + run_all + validate) | ✅ 완성 |
| `scripts/sim/dawn_context.py` | 7종 Cypher 사전 조회 → 텍스트 블록 | ✅ |
| `scripts/sim/stage1_intent.py` | Stage 1 LLM (의도·카테고리·anchor) | ✅ |
| `scripts/sim/stage2_poi.py` | Stage 2 LLM (POI 확정) | ✅ |
| `scripts/sim/plan_writer.py` | Plan 적재 + 만족도 룰 + Night Phase 1/3 | ✅ |
| `scripts/sim/run_simulation.py` | 메인 루프 + checkpoint | ✅ |
| `scripts/sim/evaluate.py` | KPI 측정 (DID·환각·만족도 등) | ✅ |
| **`scripts/sim/night_interaction.py`** | 🆕 3축 점수 + 그리디 매칭 | ✅ 신규 |
| **`scripts/sim/night_intent_llm.py`** | 🆕 의도 분류 LLM + Conversation 적재 | ✅ 신규 |
| `scripts/sim/export_visualization.py` | 시각화용 JSON dump | ✅ |
| `scripts/sim/build_standalone_html.py` | 단일 HTML 빌드 (공유용) | ✅ |

### 3.3 그래프 온톨로지 (최종)

**정적 노드 (Day 0)**: `:District` 25 · `:Dong` 427 · `:Category` 93 · `:POI` 543,924 · `:Agent` 14,881

#### `:POI` type 3종 상세

| type | 카운트 | 예시 | 용도 |
|---|---|---|---|
| **residence** | 2,909 | 동양파라곤, 롯데캐슬천지인, 무악현대아파트 | 거주지 (K-apt 공동주택 단지) |
| **workplace** | 3,526 | WIN빌딩, 코엑스타워, RDL Tower | 직장 (건축물대장 빌딩) |
| **commerce** | **537,489** | 라프푸드코트, 호정순대국, 이지스터커피&베이커리 | 가게·점포 (소상공인 데이터) |

#### `:POI {type:'commerce'}` 카테고리 분포 (12 L1)

`categories.yaml` + `mapping_upjong_to_sub.json` 기준 — 풀런 그래프 실측 분포:

| L1 카테고리 | 카운트 | sub 어휘 (예시) |
|---|---|---|
| **식사** | 95,393 | 한식, 일식, 중식, 양식, 분식, 치킨, 피자, 아시안, 기타요식 등 11종 |
| **카페** | 24,116 | 카페 |
| **디저트** | 6,375 | 베이커리, 아이스크림 |
| **주점** | 15,201 | 일반주점, 호프 |
| **편의점** | 10,193 | 편의점, 담배 |
| **마트** | 19,469 | 슈퍼마켓, 식료품, 청과, 정육, 수산, 음료소매, 종합소매 |
| **미용** | 35,006 | 미용실, 네일, 마사지, 욕탕·신체관리, 피부관리 |
| **쇼핑** | 73,253 | 의류, 화장품, 가구, 가전·통신, 안경, 시계·귀금속, 문구, 반려동물 등 16종 |
| **여가** | 19,156 | PC방, 노래방, 당구, 볼링, 스포츠, 여행사, 유원지·오락 |
| **건강** | 29,633 | 병원, 의원, 치과, 한의원, 약국, 헬스장, 건강보조식품 |
| **교육** | 45,080 | 학원, 교육지원, 기타교육 |
| **기타** | 164,614 | 부동산, 법무, 컨설팅, 광고, 사진, 세탁, 수리, 디자인, 숙박, 주유소 등 29종 |

→ 정책 P001 대상(식사+카페+디저트)은 합계 **125,884개** POI (commerce의 23.4%).

**정적 엣지**:
- `:HAS_DONG`, `:ADJACENT_TO`, `:IN_DONG`, `:IN_CATEGORY`
- `:LIVES_AT`, `:WORKS_AT`
- `:KNOWS`, `:KNOWS_POI` (집계 캐시, in-place 갱신)

**런타임 노드 (시뮬 진행 중 누적)**:
- `:State` (agent×day, 잔액·mood·fatigue)
- `:Plan` (agent×day, INCLUDES 엣지에 이벤트 인라인)
- `:Memory {type:'visited'|'rumor'|'sns'|'policy'}` (시계열 raw 기록)
  - rumor: `{id, type:'rumor', day, source, topic_type, topic_value, importance}` — Night Phase 2가 적재
  - visited: `{id, day, poi_id, satisfaction}` — Night Phase 1이 적재
  - policy / sns: Watchdog/SNS 채널이 시드
- `:Conversation {intent, initiator_id, recipient_id, topic_type, topic_value, should_inject, target_day_offset, target_time, meeting_location_hint}` 🆕 v2
- `:Policy` (정책 카탈로그, type: subsidy/regulation/facility/campaign)

**런타임 엣지**: `:HAS_STATE`, `:HAS_PLAN`, `:INCLUDES`, `:REMEMBERS`, `:ABOUT_POI`,
- 🆕 v2: `:PARTICIPATES_IN {role:'initiator'|'recipient'}`, `:MENTIONS_POI`, `:ABOUT_POLICY`, `:FROM_CONVERSATION`
- `:applied_to`, `:targets`
- 폐기: `:WITH`, `:OCCURRED_IN`, `:SEEDS_PLAN` (v2에서 모두 다른 표현으로 흡수)

### 3.4 풀런 결과 요약 (3일, 14,560 agent)

| 항목 | 값 |
|---|---|
| 시뮬 시간 | 21시간 52분 |
| 일별 계획 생성 (성공) | 42,566 (97.4%) |
| 환각 (poi_id ∉ POI) | **0건** (4,385 INCLUDES 전부 valid) |
| 평일 직장 출근 준수 | 100% |
| visited Memory 누적 | 91,631 |
| KNOWS_POI 갱신 (단골 형성) | 24,450 |
| 정책 P001 DID 순효과 | **+17.0%p** |
| **신규: Conversation** | **188** (5/02만 적용, 기타 179 / 이슈 5 / 추천 4) |
| **신규: rumor Memory** | **2** |
| 총 LLM 토큰 | in 158M / out 15M |

### 3.5 노션 다이어그램 정합성 점검

노션 "Night — 상호작용 대상 선정 알고리즘" 페이지의 3-Phase 일별 사이클을 코드와 1:1 대응:

| 노션 Phase | 코드 구현 위치 | 상태 |
|---|---|---|
| **Phase 1: 낮 - 로그 수집** (Daily Activity Buffer) | `plan_writer.write_plan()` — `:Plan-[:INCLUDES]->:POI` 적재 | ✅ |
| **Phase 2: 밤 - 상호작용 및 의도 결정** | `night_interaction.py` + `night_intent_llm.py` | ✅ |
|  ↳ 상대별 상호작용 점수 계산 | `night_interaction.fetch_all` + `find_candidate_pairs` | ✅ |
|  ↳ Exposure (시간/빈도) | `calc_exposure` (0.6×freq + 0.4×avg_overlap) | ✅ |
|  ↳ Relationship (친밀도/역할) | `calc_relationship` (0.5×base + 0.5×intimacy) | ✅ |
|  ↳ Urgency (정보/감정) | `calc_urgency` (정책 비대칭 + mood 극단 + fatigue 보정) | ✅ |
|  ↳ 점수 기반 랭킹 | `select_interaction_pairs` (`scored.sort` + 그리디 매칭) | ✅ |
|  ↳ 상호작용 대상 Persona B 선정 | greedy `max_pairs_per_agent=2` | ✅ |
|  ↳ LLM 실행: 대화 의도 추론 | `night_intent_llm.classify_intent` (SGLang/vLLM, Qwen3-32B-AWQ 기본) | ✅ |
|  ↳ Input Data (Persona A & B 프로필 + 하루 일과) | `build_user_block` (MATCHING_ANALYSIS + AGENT_A + AGENT_B) | ✅ |
|  ↳ LLM 분류 → 약속/이슈/추천/기타 | Pydantic `IntentOutput` 검증 (enum + topic_type 강제) | ✅ |
|  ↳ Conversation 노드 생성 | flat topic + plan_signal 4필드 | ✅ |
| **Phase 3: 상태 업데이트** | `write_conversations` intent별 분기 | ✅ |
|  ↳ 만약 약속일 경우 다음 날 plan에 추가 | `should_inject=true AND day+target_day_offset==today` → Dawn ④ APPOINTMENT_CYPHER 자동 조회 | ✅ |
|  ↳ Memory Stream 업데이트 → Memory DB | 이슈·추천 모두 Memory{rumor} + FROM_CONVERSATION (importance=urg×0.6+rel×0.4) | ✅ |
| **매일 자정 자동 실행** | `run_simulation.run_day()` 끝에 hook 통합 | ✅ |

### 3.6 의도 분류 입출력 계약 (노션 v2 §1·§2 정합)

**입력** (`build_user_block`):
- ✅ MATCHING_ANALYSIS 블록 (interaction_score, exposure, relationship, urgency)
- ✅ AGENT_A / AGENT_B 블록 (role, agent_id, job, lifestyle, mood, fatigue)
- ✅ daily_log 정규화 한 줄 포맷: `time: HH:MM | dong: ... | poi: ... | category: ... | activity: ...`
- ✅ persona는 agent_id, job, lifestyle, mood, fatigue만 사용
- ✅ daily_log 전체 시간순, 발췌하지 않음
- ✅ residence/workplace/home anchor 이벤트도 포함

**출력 JSON** (`IntentOutput`, v2):
```python
{
  "intent": "약속|이슈|추천|기타",
  "initiator_id": str,
  "recipient_id": str,
  "topic_type": "policy|poi|category|none",   # flat (구 topic_entity 폐기)
  "topic_value": str|None,
  "plan_signal": {
    "should_inject": bool,
    "target_day_offset": int|None,             # D+N의 N (구 time_horizon "D+N" 문자열 폐기)
    "target_time": "HH:MM"|None,               # ✨ v2 신규
    "meeting_location_hint": str|None,         # ✨ v2 신규 (POI 매칭 실패 시 자유 문자열)
  }
}
```
- ✅ Pydantic field_validator로 intent·topic_type enum 강제
- ✅ 노션 규칙: `약속`일 때만 plan_signal 4필드 채움 / 다른 intent는 전부 null·false
- ✅ intent별 필드 매핑: 이슈=topic_type:policy, 추천=topic_type:poi|category, 기타=topic_type:none
- ✅ 분류 우선순위 (약속 > 이슈 > 추천 > 기타) SYSTEM 프롬프트에 반영
- ✅ 코드펜스·자연어 금지 → `_extract_json` + Pydantic 검증

### 3.7 미구현 / 향후 작업

| 항목 | 우선순위 |
|---|---|
| 페르소나 sensitivity (소득·spending_tendency 별 정책 효과 차등) | 중 |
| 정책 lifecycle S0~S5 + awareness 진행 모델 | 중 |
| 쿠폰 잔액 추적 (cap_per_agent 도달 후 효과 0) | 중 |
| Memory{type:'sns'} 적재 채널 (SNS 영향력자 모델) | 낮음 |
| 60일 풀런 (현재 GPU로 약 18일 소요, 인프라 확장 필요) | 낮음 |

### 3.6 사용법 (재현)

```bash
# 1) Day 0 적재 (Neo4j 5.x + 입력 데이터 준비)
python scripts/neo4j_load/apply_constraints.py
python scripts/neo4j_load/run_all.py

# 2) 시뮬 (vLLM Qwen3-32B-AWQ 가동 후)
python scripts/sim/run_simulation.py --start 2026-05-01 --days 3 --workers 16

# 3) 🆕 Night 상호작용 (시뮬 day별로)
python scripts/sim/night_interaction.py --day 2026-05-02 \
  --dump output/sim/interactions_2026-05-02.json

python scripts/sim/night_intent_llm.py --day 2026-05-02 \
  --pairs output/sim/interactions_2026-05-02.json --workers 16

# 4) KPI 평가
python scripts/sim/evaluate.py --start 2026-05-01 --days 3
```

---

## 📌 결론

두 모듈(`night_interaction.py` + `night_intent_llm.py`)로 **노션 v2 다이어그램의 3-Phase 일별 사이클이 코드와 1:1 정합**:
- Phase 1 (Daily Activity Buffer) — 기존 `plan_writer.write_plan`이 담당
- **Phase 2 (상호작용 + 의도 결정)** — Exposure·Relationship·Urgency 3축 + LLM 의도 분류 + Conversation 적재
- **Phase 3 (상태 업데이트)** — intent별 분기 (노션 §4·§5·§9 정합):
  - **약속** → Conversation에 `should_inject + target_day_offset + target_time + meeting_location_hint` 저장 → D+offset Dawn ④ `APPOINTMENT_CYPHER`가 자동 조회 → Stage 1 프롬프트 주입
  - **이슈** → recipient에 `:Memory{rumor, source, topic_type:'policy', topic_value:<Policy.id>, importance}` + `[:FROM_CONVERSATION]` + `(c)-[:ABOUT_POLICY]->(:Policy)`
  - **추천** → recipient에 `:Memory{rumor, topic_type:'poi'|'category'}` + `[:FROM_CONVERSATION]` + `(c)-[:MENTIONS_POI]` + `(m)-[:ABOUT_POI]` + `(b)-[:KNOWS_POI{source:'rumor'}]` MERGE
  - **기타** → Conversation만 (노션 §4 — 별도 효과 없음)
- **매일 자정 자동 실행** — `run_simulation.run_day()` 끝에 hook 통합
- **importance 산식** — `urgency × 0.6 + relationship × 0.4` (노션 §9)
- **Memory id 형식** — `MEM_RUMOR_<recipient>_D<YYYYMMDD>_<n>`

기존 풀런(3일, v1 의도 분류) 결과는 5/02 한 day만 사후 적용. **v2 스펙 도입 후 다음 풀런부터** 매일 자동 사슬로 동작하여 D+1 시뮬에 약속·rumor·정책 정보 전파가 자연 반영됨.

> 노션 v2 본문이 명시한 "매일 시뮬레이션이 끝난 뒤 (Night 단계)" + 다이어그램 Phase 1/2/3 사이클 + "만약 약속일 경우 다음 날 plan에 추가" + "이슈·추천은 Memory{rumor}로 적재 + FROM_CONVERSATION으로 출처 보존" — 모두 코드에 반영됨.
