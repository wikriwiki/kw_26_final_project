# `src/graph/` — Neo4j / Graphiti 인터페이스

**모든 Neo4j 접근은 이 폴더를 거친다.** 다른 폴더에서 `neo4j.GraphDatabase`를 직접 import하면 PR 리젝.

## 폴더 맵

| 폴더/파일 | 역할 |
|-----------|------|
| `client.py` | Neo4j 드라이버 싱글톤 + 세션 관리 |
| `schema.py` | 노드/엣지 라벨 상수 (`Agent`, `Plan`, `KNOWS`, `APPLIED_TO` 등) |
| `migrations/` | 초기 스키마/인덱스를 정의하는 `.cypher` 파일 |
| `queries/` | 도메인별 Cypher 쿼리 모듈 |

## 스키마 (다이어그램 기준)

### 에이전트 도메인
- `(Agent)-[:HAS_STATE]->(State)`
- `(Agent)-[:REMEMBERS]->(MemoryStream)`
- `(Agent)-[:PARTICIPATES_IN]->(Conversation)`
- `(Agent)-[:HAS_PLAN]->(Plan)-[:INCLUDES]->(Episode)`
- `(Agent)-[:KNOWS]->(Agent)` *(인텐트=`약속` 시 사용)*
- `(Agent)-[:KNOWS_POI]->(POI)`
- `(Agent)-[:LIVES_AT|WORKS_AT]->(행정동)`

### 공간/정책 도메인
- `(Policy)-[:APPLIED_TO]->(행정구)` *Seoul → 행정구 → 행정동*
- `(Policy)-[:TARGETS]->(업종 카테고리)`
- `(행정동)-[:ADJACENT_TO]->(행정동)`
- `(업종 카테고리)-[:HAS]->(POI)`
- `(Episode)-[:OCCURRED_IN]->(POI)`

자세한 속성/제약은 `docs/05_graph_schema.md` (TBD)에 작성.

## 규칙

- 트랜잭션 경계는 호출자가 정함 (이 폴더는 `session.execute_read/write`까지)
- 쿼리에 비즈니스 로직 박지 말 것 — Cypher가 1000줄 넘어가면 도메인 로직이 새고 있음
- 모든 쿼리는 **파라미터 바인딩** — 문자열 포매팅 절대 금지 (인젝션)
