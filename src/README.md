# `src/` — 소스 코드 루트

본 시스템은 **하이브리드 LLM-ABM 시뮬레이션**으로, 다음 4개 컴포넌트로 구성됩니다.
각 컴포넌트는 폴더로 분리되어 있으니, **새 파일을 추가할 때 반드시 해당 폴더의 README를 먼저 읽고 위치를 결정**하세요.

## 폴더 맵

| 폴더 | 역할 | 다이어그램 박스 |
|------|------|----------------|
| `core/` | 도메인 모델 (Pydantic) — Agent, State, Plan, Memory 등 | Neo4j Internal Schema |
| `phases/dawn/` | Phase 1: 자정에 하루 Plan 생성 (Plan as Log) | 🌅 Dawn |
| `phases/night/` | Phase 2: 하루 종료 시 상호작용 & 메모리 업데이트 | 🌙 Night |
| `policy_pipeline/` | Async 정책 주입 (Watchdog → LLM → Cypher) | 📂 Async |
| `infra/llm/` | vLLM/SGLang 엔진, Batch Controller | 🛠 Shared Infra |
| `infra/cache/` | Prompt Prefix Cache, Global Community Summary | 🛠 Shared Infra |
| `graph/queries/` | Neo4j Cypher 쿼리 (도메인별) | Neo4j DB |
| `graph/migrations/` | Neo4j 스키마/인덱스 마이그레이션 | Neo4j DB |
| `prompts/` | LLM 프롬프트 템플릿 (Jinja2) | — |

## 진입점 (루트에 위치)

| 파일 | 역할 |
|------|------|
| `config.py` | 경로, Neo4j 엔드포인트, LLM 엔드포인트, 배치 크기 등 전역 설정 |
| `orchestrator.py` | Day t 루프 (Dawn → Night → 다음 날) |
| `cli.py` | 진입점 (`python -m src.cli --day 1`) |

## 새 코드를 어디에 둘지 헷갈릴 때

- 데이터 클래스(Pydantic) → `core/`
- LLM 호출이 들어가는 비즈니스 로직 → `phases/dawn` 또는 `phases/night`
- Neo4j 쿼리 → `graph/queries/`
- 정책 파일 처리 → `policy_pipeline/`
- 인프라(LLM 엔진, 캐시) 래퍼 → `infra/`

**한 함수에 여러 책임이 섞이면 해당 폴더의 README가 정의한 책임을 어기는 것입니다. 분리하세요.**
