# `scripts/` — 운영 스크립트

**일회성 또는 운영 작업용 스크립트.** 시뮬레이션 메인 루프에 포함되지 않는 작업만.

## 예상 파일

| 파일 | 역할 |
|------|------|
| `init_neo4j.py` | `src/graph/migrations/` 의 `.cypher` 순서대로 실행 (최초 1회) |
| `seed_data.py` | `data/seed/` 의 행정동/POI/Persona를 Neo4j에 로딩 |
| `reset_simulation.py` | Plan/Episode/Memory/Conversation 노드 일괄 삭제 (정적 도메인은 보존) |
| `bench_llm.py` | vLLM/SGLang throughput 벤치마크 |

## 규칙

- **재실행 안전**: 같은 스크립트를 두 번 돌려도 망가지지 않게
- 파괴적 스크립트(`reset_*`)는 `--yes` 플래그 없으면 확인 프롬프트 띄울 것
- 모든 스크립트는 `python -m scripts.<name>` 형태로 실행 가능해야 함
- 운영 로직은 여기 두지 말고 `src/`로 — 여기는 그걸 호출하는 얇은 진입점만
