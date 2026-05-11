# `src/graph/migrations/` — Neo4j 스키마/인덱스 마이그레이션

**`.cypher` 파일로 작성.** Python 코드 두지 말 것.

## 명명 규칙

```
NNN_<목적>.cypher
```

예시:
- `001_agent_domain.cypher` — Agent/State/Plan/Episode/Memory 라벨 + 인덱스
- `002_spatial_policy_domain.cypher` — Seoul/행정구/행정동/POI/Policy + 인덱스
- `003_constraints.cypher` — UNIQUE 제약, NOT NULL 등

## 실행

`scripts/init_neo4j.py`가 `migrations/` 폴더의 파일을 **파일명 순서**대로 실행. 한번 적용된 마이그레이션은 재실행 안전(`IF NOT EXISTS`) 해야 함.

## 규칙

- 기존 파일 수정 금지 — 새 마이그레이션 파일 추가로 변경
- 인덱스는 자주 조회되는 속성에: `Agent.id`, `행정동.code`, `Plan.day`
- 제약은 ID 류에: `CREATE CONSTRAINT FOR (a:Agent) REQUIRE a.id IS UNIQUE`
