# Simulation — 매일 사이클 실행 가이드

Neo4j Day 0 그래프가 준비된 다음, **매일 시뮬을 돌리는 본체**. Dawn(아침 계획) → 낮(시뮬) → Night(상호작용·기억 정리) 한 사이클을 N일 반복.

## TL;DR

```bash
# 0. 사전 조건
#    - Neo4j 5.x 실행 중 + Day 0 적재 완료 (scripts/neo4j_load/run_all.py)
#    - SGLang 또는 vLLM 서버 가동 중 (Qwen3-32B-AWQ 기본)
#    - .env에 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD

# 1. LLM 서버 띄우기
bash scripts/serve/serve_qwen32b.sh    # 또는 qwen9b / exaone

# 2. 시뮬 (테스트: 강남구 100명 × 3일)
python scripts/sim/run_simulation.py --start 2026-05-01 --days 3 --gu 11680 --limit 100 --workers 16

# 3. 풀런 (14,560 agent × 3일, ~13–22시간)
python scripts/sim/run_simulation.py --start 2026-05-01 --days 3 --workers 16

# 4. KPI 평가 (DID·환각·만족도)
python scripts/sim/evaluate.py --start 2026-05-01 --days 3

# 5. 시각화 HTML (단일 파일로 팀원 공유)
python scripts/sim/export_visualization.py --start 2026-05-01 --days 3
python scripts/sim/build_standalone_html.py
```

---

## 매일 사이클 아키텍처

```
┌── Dawn (자정) ─────────────────────────────────┐
│  ① dawn_context.py — 7종 Cypher → 텍스트 블록  │
│  ② stage1_intent.py — 의도·카테고리·anchor LLM │
│  ③ stage2_poi.py — POI 확정 LLM                │
│  ④ plan_writer.py write_plan — :Plan-[:INCLUDES]│
└────────────────────────────────────────────────┘
                  ↓
┌── 낮 (시뮬레이션) ──────────────────────────────┐
│  run_simulation.simulate_satisfaction           │
│  (룰 기반 만족도 — 정책 효과는 자연어 LLM 경로) │
└────────────────────────────────────────────────┘
                  ↓
┌── Night (자정 직전) ───────────────────────────┐
│  Phase 1: plan_writer.night_finalize_yesterday │
│           visited Memory + KNOWS_POI 갱신       │
│  Phase 2: night_interaction.py                  │
│           3축 점수 + 그리디 매칭                │
│           night_intent_llm.py                   │
│           LLM 의도 분류 → Conversation 적재     │
│  Phase 3: plan_writer.night_create_state        │
│           오늘 State CREATE (mood/fatigue)      │
└────────────────────────────────────────────────┘
                  ↓
            다음 날 Dawn으로 (체인 반복)
```

---

## 파일 구성

### 코어 (Dawn + Plan)

| 파일 | 역할 |
|---|---|
| `dawn_context.py` | 7종 고정 Cypher 쿼리 + 텍스트 블록 포맷터 (Stage 1 프롬프트에 주입) |
| `stage1_intent.py` | Stage 1 LLM — 의도 시퀀스 + 카테고리 + anchor 생성 |
| `stage2_poi.py` | Stage 2 LLM — 각 이벤트의 구체 POI 확정 (KNOWS_POI + 거리 기반) |
| `plan_writer.py` | Plan 적재 + 만족도 룰 + Night Phase 1·3 (visited Memory, State CREATE) |
| `llm_client.py` | SGLang/vLLM 자동감지 + 모델 레지스트리 (qwen32b/qwen9b/exaone) |

### Night (상호작용)

| 파일 | 역할 |
|---|---|
| `night_interaction.py` | Phase 2 — 상호작용 대상 선정 (Exposure·Relationship·Urgency 3축 + 그리디 매칭) |
| `night_intent_llm.py` | Phase 2 — 의도 분류 LLM (약속/이슈/추천/기타) + Conversation·Memory{rumor} 적재 |

> Night Phase 2의 상세 설계는 [`docs/NIGHT_INTERACTION_REPORT.md`](../../docs/NIGHT_INTERACTION_REPORT.md), 노션 다이어그램 ↔ 코드 1:1 매핑은 [`docs/NIGHT_NOTION_DIAGRAM_MAPPING.md`](../../docs/NIGHT_NOTION_DIAGRAM_MAPPING.md) 참고.

### 메인 루프 + 평가 + 시각화

| 파일 | 역할 |
|---|---|
| `run_simulation.py` | 메인 루프 — ThreadPoolExecutor로 agent 병렬, 일자별 chain + Night Phase 2 hook |
| `evaluate.py` | KPI 측정 — DID 분석, 환각률, 만족도, 정책 lifecycle |
| `export_visualization.py` | 그래프 → JSON dump (D3.js·Leaflet 입력용) |
| `build_standalone_html.py` | dump JSON + 템플릿 → 단일 HTML (오프라인 공유 가능) |

---

## 환경 변수

| 변수 | 기본 | 용도 |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j 접속 |
| `NEO4J_USER` / `NEO4J_PASSWORD` | (필수) | 인증 |
| `SIM_OUTPUT_DIR` | `~/sim_output` | 체크포인트·메트릭·interactions_<day>.json 저장 |
| `VIZ_OUT_DIR` | `<프로젝트루트>/output/sim/visualization` | 시각화 JSON 저장 |
| `LLM_MODE` | `qwen32b` | `qwen9b` (개발) / `exaone` (대회용) 전환 |
| `SGLANG_BASE_URL` | auto (30000 → 8000) | LLM 서버 URL 명시 |

> **Google Drive 경로 주의**: `G:\내 드라이브\...` 같은 Drive Stream 가상 파일시스템에 `SIM_OUTPUT_DIR`를 두면 ~7,000 write 이후 OSError 22 발생. 로컬 디스크(`C:\Users\<user>\sim_output\`)에 두는 것을 권장.

---

## 메인 루프 단독 실행 옵션

```bash
python scripts/sim/run_simulation.py \
  --start 2026-05-01 \         # 시뮬 시작일 (Day 0 다음날)
  --days 3 \                   # 며칠 진행할지
  --gu 11680 \                 # 자치구 코드로 필터 (optional, 강남=11680)
  --limit 100 \                # agent 수 제한 (optional, 테스트용)
  --workers 16                 # ThreadPoolExecutor worker 수
```

체크포인트가 `$SIM_OUTPUT_DIR/run_*/` 아래 저장됨. 도중에 죽으면 같은 명령으로 재실행 시 이어서 진행.

---

## Night Phase 2 단독 실행 (디버그용)

메인 루프는 매일 자정 자동으로 Night Phase 2를 호출하지만, 사후 분석/재처리가 필요하면 day별로 직접 호출 가능:

```bash
# 1. 후보 쌍 추출 + JSON dump
python scripts/sim/night_interaction.py \
  --day 2026-05-02 \
  --dump $SIM_OUTPUT_DIR/interactions_2026-05-02.json

# 2. LLM 의도 분류 + Conversation 적재
python scripts/sim/night_intent_llm.py \
  --day 2026-05-02 \
  --pairs $SIM_OUTPUT_DIR/interactions_2026-05-02.json \
  --workers 16
```

---

## 그래프와의 입출력 계약

### Dawn에서 **읽음** (7종 Cypher)

`dawn_context.py`가 각 agent에 대해 다음을 가져와 Stage 1 프롬프트로 변환:

1. **Persona** — `:Agent` 정적 속성 + `:LIVES_AT`/`:WORKS_AT` anchor
2. **State** — 어제 잔액·mood·fatigue·정책 lifecycle
3. **Memory Top-N** — 최근 30일, `importance × exp(-days/14)` 정렬
4. **Appointment** — `should_inject=true AND day+offset=today`인 Conversation
5. **Policy** — 거주·직장 동에 `applied_to`된 활성 정책
6. **Social** — `:KNOWS` strength 상위 N명
7. **KNOWS_POI summary** — 카테고리별 인지 POI 분포

### Night에서 **씀**

| 위치 | 만드는 노드/엣지 |
|---|---|
| `plan_writer.write_plan` | `:Plan` + `[:INCLUDES]` 이벤트 인라인 |
| `plan_writer.night_finalize_yesterday` | `:Memory{type:'visited'}` + `[:ABOUT_POI]` + `:KNOWS_POI` 갱신 (visit_count, affinity) |
| `plan_writer.night_create_state` | 오늘 `:State` + `[:HAS_STATE]` |
| `night_intent_llm.write_conversations` | `:Conversation` + `[:PARTICIPATES_IN {role}]` + 이슈·추천 시 `:Memory{type:'rumor'}` + `[:FROM_CONVERSATION]` |

---

## 관련 문서

| 문서 | 내용 |
|---|---|
| [`docs/NEO4J_SETUP_GUIDE.md`](../../docs/NEO4J_SETUP_GUIDE.md) | Neo4j Day 0 환경·DDL·적재 통합 가이드 |
| [`docs/NIGHT_INTERACTION_REPORT.md`](../../docs/NIGHT_INTERACTION_REPORT.md) | Night Phase 2 통합 보고 (v2 의도분류 정합 반영) |
| [`docs/NIGHT_NOTION_DIAGRAM_MAPPING.md`](../../docs/NIGHT_NOTION_DIAGRAM_MAPPING.md) | 노션 다이어그램 12박스 ↔ 코드 1:1 매핑 |
| [`docs/SGLANG_MIGRATION.md`](../../docs/SGLANG_MIGRATION.md) | vLLM → SGLang 마이그레이션 (RadixAttention·structured output) |
| [`docs/schedule_generation_plan/agent_ontology.md`](../../docs/schedule_generation_plan/agent_ontology.md) | 정적 노드 5종 + 엣지 명세 |
| [`docs/schedule_generation_plan/runtime_ontology.md`](../../docs/schedule_generation_plan/runtime_ontology.md) | 런타임 노드 5종 (State/Plan/Memory/Conversation/Policy) + 엣지 |

---

## 알려진 한계

- 14,560 agent × 3일 풀런 기준 SGLang(Qwen3-32B-AWQ) ~13–15시간, vLLM ~22시간
- `LLM_MODE=qwen9b`는 토큰량 60% 절감되나 의도 분류 정확도 약간 낮음 (개발·디버그 권장)
- `LLM_MODE=exaone`은 한국어 자연스러움 최상 (대회·시연 권장), 추론 속도는 Qwen3-32B와 유사
- KNOWS_POI 단일 직접 엣지 캐시 사용 — 시뮬 도중 in-place 갱신만 (`visit_count`, `affinity`)되고 신규 인지는 추천 의도 분류에서만 추가됨
