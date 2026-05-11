# `tests/` — 테스트

`pytest` 기반. `src/`의 폴더 구조를 미러링.

## 폴더 맵

| 폴더 | 용도 |
|------|------|
| `unit/` | 의존성 없는 순수 단위 테스트 (Pydantic 모델, 점수 계산 등) |
| `integration/` | Neo4j/LLM 등 외부 의존성 포함. `testcontainers-neo4j` 사용 권장 |
| `fixtures/` | 샘플 Persona/Policy/Plan JSON. `conftest.py`에서 로딩 |

## 규칙

- **`unit/`에서 외부 호출 금지.** Neo4j 접근이 필요하면 `integration/`로 옮길 것
- LLM 테스트는 mock 클라이언트 사용 (`infra/llm/engine_client.py`에 인터페이스만 노출되어 있어야 함)
- 테스트 파일명: `test_<모듈명>.py` — 미러링 예시:
  ```
  src/phases/dawn/plan_generator.py
  tests/unit/phases/dawn/test_plan_generator.py
  ```
- fixtures는 작게 유지. 큰 데이터셋은 `data/seed/`에서 가져올 것
