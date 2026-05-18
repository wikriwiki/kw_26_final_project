# `scripts/policy_pipeline/` — 정책 추출·검증·적재 파이프라인

`agent_persona` 브랜치의 **하드코딩 P007 적재**(`load_p007.py`) 를 대체하는,
일반화된 정책 처리 파이프라인.

## 두 가지 입력 경로

### 1. 자연어 텍스트 (LLM 추출)
`data/policies/inbox/*.txt` 또는 `*.json` 에 정책 원문을 떨군다 → Watchdog 감지 →
LLM Structured Output 추출 → 도메인 검증 → Scope 분석 → Neo4j 적재 → 캐시 무효화.

```bash
python -m scripts.policy_pipeline.watch
# 다른 터미널에서
cp my_new_policy.txt data/policies/inbox/
```

### 2. 수기 검수 JSON (LLM 우회)
`data/neo4j_load/policies/P007.json` 같은 이미 정형화된 JSON. LLM 단계를 건너뛰고
바로 검증 → Scope → Neo4j 적재.

```bash
# 기존 load_p007.py 와 동등한 결과 (P001 비활성화 포함)
python -m scripts.policy_pipeline.inject_json \
    data/neo4j_load/policies/P007.json \
    --deactivate-others-from 2026-05-01

# 단순 적재
python -m scripts.policy_pipeline.inject_json data/neo4j_load/policies/P007.json

# Neo4j 안 건드리고 검증 + Scope 만 확인
python -m scripts.policy_pipeline.inject_json data/neo4j_load/policies/P007.json --dry-run
```

기존 `scripts/neo4j_load/load_p007.py` 는 이 새 명령어로 forward 하는
thin wrapper 로 남았다 (하위 호환).

### 3. 스케줄 일괄 주입 (시뮬 시작 전 권장)
여러 정책을 "언제부터 언제까지 활성화할지" 한 YAML 에 모아 적고 일괄 주입한다.
시뮬 도중 동적 적재는 하지 않는다 — Neo4j 의 `effective_from / effective_until`
필드와 `dawn_context.POLICY_CYPHER` 의 날짜 필터가 활성 시점을 자동 결정한다.

```yaml
# data/policies/schedule.yaml
sim_start: 2026-05-01            # 상대 일수의 기준
policies:
  - file: P007.json              # 절대 날짜
    effective_from: 2026-05-06
    effective_until: 2026-06-30
    deactivate_others: true      # 적재 시 다른 정책 비활성

  - file: P008.json              # 시뮬 시작 + N일 (상대)
    effective_from_day: 12
    effective_until_day: 45
```

```bash
python -m scripts.policy_pipeline.apply_schedule              # 적재
python -m scripts.policy_pipeline.apply_schedule --dry-run    # 날짜만 검증
```

규칙:
- 절대(`effective_from`) 와 상대(`effective_from_day`) 동시 지정 금지
- 상대 일수를 하나라도 쓰면 `sim_start` 필수
- 정책 JSON 원본은 건드리지 않음 — 적재 시점에만 override

---

## 처리 흐름

```
파일 stable 확인 (watcher)
  ↓ file_hash 계산
dedup.py  ── 이미 처리된 hash 면 skip
  ↓
loader.py → PolicyDocument
  ↓
extractor.py
  ├ Structured Output (OpenAI 클라우드)    ← 가능 시 우선
  └ JSON object 폴백 (SGLang 자체 호스팅)   ← scripts/sim/llm_client 재사용
  ↓ ExtractedPolicy
validator.py  ── 도메인 룰 (자치구 화이트리스트, L1 카테고리, 모호어, 전국 범위)
  ├ VALIDATED      → 후속 처리
  ├ NEEDS_REVIEW   → data/policies/failed/ 로 보존
  └ FAILED         → 동일
  ↓
scope.py → PolicyScope (textual + Neo4j GraphReader 로 확장)
  ↓
neo4j_writer.py
  MERGE (:Policy {id, name, type, ..., benefit_rate, cap_per_agent, dates})
  MERGE (Policy)-[:applied_to]->(:District)   # target_districts
  MERGE (Policy)-[:applied_to]->(:Dong)       # scope.affected_dongs
  MERGE (Policy)-[:targets]->(:Category)      # parent IN benefit_categories
  ↓
invalidator.py + summary_jobs.py
  context_version bump + 요약 재생성 잡 enqueue
  ↓
state → APPLIED, archive(processed/)
```

---

## Neo4j 스키마 매핑

| Pydantic 필드 | Neo4j 속성/엣지 |
|--------------|----------------|
| `policy_id` | `:Policy.id` (UNIQUE) |
| `title` | `:Policy.name` |
| `summary` | `:Policy.description` |
| `policy_type` | `:Policy.type` (subsidy/coupon/...) |
| `benefit_rate` | `:Policy.benefit_rate` |
| `cap_per_agent` | `:Policy.cap_per_agent` |
| `announce_date` / `effective_from` / `effective_until` | 동일 (date) |
| `source_file_hash` | `:Policy.raw_json_ref` |
| `target_districts` (list) | `(:Policy)-[:applied_to]->(:District {name})` |
| `target_dongs` / scope.affected_dongs | `(:Policy)-[:applied_to]->(:Dong {code})` |
| `benefit_categories` (L1 이름) | `(:Policy)-[:targets]->(:Category {parent IN L1})` |

빈 `benefit_categories` 는 "전체 commerce" 의미 — `:targets` 엣지를 만들지 않음.
시뮬레이션의 POLICY_CYPHER 가 `OPTIONAL MATCH (pol)-[:targets]->(cat)` 으로 받기 때문에
빈 경우 자연스럽게 "카테고리 제한 없음" 으로 해석된다.

---

## 파일 책임

| 파일 | 책임 |
|------|------|
| `vocabulary.py` | **단일 진실의 원천**. 25 자치구, 12 L1 카테고리, 모호어 사전, 정규화 헬퍼 |
| `models.py` | Pydantic 도메인 모델 3단 (`PolicyDocument` → `ExtractedPolicy` → `ValidatedPolicy`). Neo4j 스키마와 1:1 매핑 |
| `loader.py` | 파일 → `PolicyDocument` (raw_text + hash) |
| `dedup.py` | file_hash 기반 중복 처리 차단 |
| `extractor.py` | LLM 호출 + ExtractedPolicy 변환. structured output 우선 |
| `llm_client.py` | OpenAI/SGLang structured output 래퍼. 재시도 + 자동 폴백 |
| `validator.py` | 도메인 룰 → VALIDATED / NEEDS_REVIEW / FAILED |
| `scope.py` | PolicyScope 산출. GraphReader Protocol |
| `neo4j_reader.py` | GraphReader 의 Neo4j 구현. `_common.driver_session()` 재사용 |
| `neo4j_writer.py` | ValidatedPolicy + Scope → MERGE Cypher |
| `state.py` | 상태머신 + JSONL 감사 로그 (DETECTED→EXTRACTING→VALIDATED→APPLIED) |
| `archive.py` | inbox → processed/ 또는 failed/ |
| `cache_keys.py`, `version_registry.py`, `invalidator.py`, `summary_jobs.py` | L3 캐시·요약 워커가 들어왔을 때를 위한 scaffolding. 현재는 JSONL append-only 로 신호만 남김 |
| `pipeline.py` | 위 모듈들의 직렬 오케스트레이터. `process_policy_file()`, `inject_validated_payload()` |
| `watch.py` | Watchdog Observer + 워커 스레드. `--polling` 옵션 (WSL) |
| `inject_json.py` | `load_p007.py` 대체. CLI 진입점 |

---

## 환경 설정

`scripts/policy_pipeline` 자체는 추가 환경변수가 필요 없다. Neo4j 인증은 기존
`scripts/neo4j_load/_common.py` 의 `.env` 로드를 재사용:

```bash
# data/neo4j_load/.env 필요
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
```

LLM 추출 경로(자연어 파일)를 쓸 때만:
```bash
# scripts/sim/llm_client 가 사용
SGLANG_BASE_URL=http://localhost:30000/v1   # SGLang
# 또는
OPENAI_API_KEY=sk-...                       # OpenAI 클라우드 (structured output 권장)
```

---

## 테스트

```bash
python -m pytest tests/unit/policy_pipeline -v
# 48 passed
```

Neo4j 통합 테스트는 실제 Neo4j 인스턴스가 필요해 별도. 위 단위 테스트는
`_StubSession` / `NullGraphReader` / `_StubLLM` 으로 외부 의존성을 모두 모킹.

---

## 기존 `load_p007.py` 와의 비교

| 항목 | 구 `load_p007.py` | 신 `inject_json.py` |
|------|-------------------|---------------------|
| 대상 파일 | P007.json 하드코딩 | 임의 JSON 경로 인자 |
| 검증 | 없음 | Pydantic 자기일관성 + 도메인 룰 |
| 자치구 매핑 | 25개 하드코딩 | 화이트리스트 + 비매핑 자치구 자동 경고 |
| 카테고리 매핑 | 없음 | `benefit_categories` → `:targets → Category` |
| Scope 분석 | 없음 | textual + Neo4j GraphReader 확장 |
| 동 단위 적재 | 없음 | scope.affected_dongs → `:applied_to → :Dong` |
| 캐시 무효화 신호 | 없음 | JSONL 로 무효화 키 + summary 잡 enqueue |
| 다른 정책 비활성화 | P001 하드코딩 | `--deactivate-others-from YYYY-MM-DD` 옵션 |
| 감사 로그 | 콘솔 출력만 | `output/policy_pipeline/*.jsonl` 영구 보존 |
| 상태 추적 | 없음 | DETECTED→EXTRACTING→VALIDATED→APPLIED 머신 |

---

## TODO

- 통합 테스트: 실제 Neo4j 인스턴스 + P007.json 적재 → POLICY_CYPHER 가 정상 동작하는지
- L3 캐시 store 어댑터 (Redis) — invalidator 가 산출한 키를 실제로 무효화
- summary worker — `summary_jobs.jsonl` 을 소비해 LLM 으로 요약 생성
- 정책 영향도 수치화 (modifier 계수) — 도메인 calibration 후 별도 모듈
