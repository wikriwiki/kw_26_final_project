# `src/graph/queries/` — 도메인별 Cypher 쿼리

**Cypher를 Python 함수로 래핑.** 호출자는 `agent_id`를 넘기고, 함수는 Pydantic 객체를 반환.

## 예상 파일

| 파일 | 역할 | 주 호출자 |
|------|------|----------|
| `context.py` | **Dawn용 컨텍스트 7종** 병렬 조회 (persona/state/memory/relations/POI/policies/episodes) | `phases/dawn/context_collector.py` |
| `memory.py` | `MemoryStream` 읽기/`REMEMBERS` 엣지 추가 | `phases/night/memory_writer.py` |
| `plan_episode.py` | Plan/Episode CRUD | `phases/dawn/plan_generator.py`, `phases/night/activity_buffer.py` |
| `conversation.py` | `Conversation`/`Interaction` 저장 | `phases/night/interaction_summary.py` |
| `policy.py` | Policy 노드 주입 + scope 조회 | `policy_pipeline/cypher_builder.py` |
| `spatial.py` | 행정구/행정동/POI 조회 + `ADJACENT_TO` 탐색 | 여러 곳 |

## 규칙

- **함수 시그니처에 Cypher 문자열 노출 금지.** 호출자는 함수만 안다.
  ```python
  # 좋음
  def get_persona(agent_id: str) -> Persona: ...
  # 나쁨
  def run_query(cypher: str, params: dict) -> Any: ...
  ```
- 반환 타입은 `core/`의 Pydantic 모델로 변환 — `dict` 또는 `neo4j.Record` 반환 금지
- 병렬 조회는 `asyncio.gather` 또는 `concurrent.futures` — `context.py`가 이걸 활용
- 트랜잭션이 필요한 경우 함수 docstring에 명시
