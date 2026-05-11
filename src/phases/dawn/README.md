# `src/phases/dawn/` — Phase 1: 일일 계획 수립 (Plan as Log)

**자정 t에 실행.** 각 에이전트의 Day t 계획을 LLM으로 생성하고, 이를 그대로 Daily Log로 저장합니다.

## 실행 흐름

```
자정 t · Day t 시작
   ↓
context_collector.py  ← Neo4j 병렬 Cypher 7종
   ↓
prompt_bundler.py     ← persona + state + memory + 정책 + POI 번들링
   ↓
plan_generator.py     ← vLLM 호출 → Plan == Daily Log 저장
   ↓
Day t Plan 완성
```

## 예상 파일

| 파일 | 역할 |
|------|------|
| `context_collector.py` | Neo4j에서 페르소나/상태/메모리/주변 POI/적용 정책을 **병렬 Cypher 7종**으로 수집 |
| `prompt_bundler.py` | 수집한 컨텍스트를 `prompts/plan_generation.jinja2`에 주입해 LLM 입력 생성 |
| `plan_generator.py` | `infra/llm/batch_controller.py`로 배치 호출 → 결과를 `core.Plan`으로 파싱 → `graph/queries/plan_episode.py`로 저장 |

## 규칙

- **이 폴더의 함수는 직접 Neo4j 드라이버를 호출하지 않음.** 모든 쿼리는 `src/graph/queries/`를 거침.
- **이 폴더의 함수는 직접 HTTP 호출하지 않음.** LLM은 `src/infra/llm/`를 거침.
- 프롬프트 문자열을 코드에 하드코딩하지 말 것 → `src/prompts/` 템플릿 사용
- Plan == Episode 등치: Plan 생성 시 즉시 `INCLUDES` 엣지로 Episode 노드 생성

## 입력/출력

- **입력**: `agent_id`, `day`
- **출력**: `core.Plan` 객체 + Neo4j에 `(Agent)-[:HAS_PLAN]->(Plan)-[:INCLUDES]->(Episode)` 영구화
