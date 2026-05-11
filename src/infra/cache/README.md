# `src/infra/cache/` — 캐시 인프라

**프롬프트 프리픽스 캐시와 Global Community Summary 관리.**

## 예상 파일

| 파일 | 역할 |
|------|------|
| `prompt_cache.py` | APC (Automatic Prefix Caching) — 공통 프리픽스 hit 시 token 절약 |
| `community_summary.py` | Global Community Summary — 커뮤니티별 요약을 L3에 캐시 |

## 규칙

- 캐시 키 설계는 신중히 — persona ID, day, policy 버전 등이 키에 포함되어야 stale 방지
- Community Summary는 **정책 변경 시 무효화** (`policy_pipeline/summary_refresher.py`가 트리거)
- TTL과 무효화 정책을 README에 명시할 것 (각 캐시 구현체 docstring에)

## 캐시 vs 영속화

| 데이터 | 위치 |
|--------|------|
| 프롬프트 프리픽스 | 이 폴더 (휘발성, vLLM 측 또는 Redis) |
| Community Summary | 이 폴더 + Neo4j 영속화 (재시작 후 복원) |
| Memory Stream | Neo4j (이 폴더 아님) |
| Plan/Episode | Neo4j (이 폴더 아님) |
