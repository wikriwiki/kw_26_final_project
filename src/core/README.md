# `src/core/` — 도메인 모델

**Pydantic 모델만 정의하는 곳.** 비즈니스 로직, I/O, LLM 호출, DB 접근 금지.

## 예상 파일

| 파일 | 모델 | 비고 |
|------|------|------|
| `agent.py` | `Agent`, `Persona` | 페르소나 = 정적 속성, Agent = Persona + 런타임 상태 ref |
| `state.py` | `State` | 위치, 감정, 소지금 등 시점별 상태 |
| `plan.py` | `Plan`, `Episode` | Plan = Day t 계획, Episode = 실제 발생 단위 (Plan == Log) |
| `memory.py` | `MemoryItem`, `MemoryStream` | 기억의 단위 + 시계열 컨테이너 |
| `conversation.py` | `Conversation`, `Interaction` | 대화/약속/추천의 raw 형태 |
| `policy.py` | `Policy`, `PolicyScope` | 정책 + 영향 범위(행정동 리스트) |

## 규칙

- **순수 데이터만**: `@validator` 외에 메서드를 거의 두지 말 것
- **외부 의존성 금지**: `neo4j`, `httpx`, `requests` 같은 모듈을 import하면 안 됨
- **타입 힌트 필수**: 모든 필드에 타입 명시
- **상속보다 합성**: `Agent`는 `Persona`를 상속하지 말고 필드로 보유

## 이유

도메인 모델이 인프라에 의존하면 단위 테스트가 어려워지고, Neo4j 스키마 변경이 비즈니스 로직 전체로 전파됩니다.
