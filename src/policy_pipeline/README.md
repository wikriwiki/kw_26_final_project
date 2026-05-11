# `src/policy_pipeline/` — Async 정책 주입 파이프라인

**메인 시뮬레이션 루프와 독립적으로 실행되는 백그라운드 프로세스.**
새 정책 파일이 `data/policies/inbox/`에 떨어지면 자동으로 처리합니다.

## 실행 흐름

```
data/policies/inbox/*.{md,pdf,txt}
   ↓
watcher.py             ← Watchdog 파일 감지
   ↓
preprocessor.py        ← LangChain으로 텍스트 추출 + 구조화
   ↓
validator.py           ← Pydantic으로 Policy 스키마 검증
   ↓
cypher_builder.py      ← (Policy) 노드 + (APPLIED_TO)→행정구/행정동 엣지 생성
   ↓
scope_analyzer.py      ← 영향 범위 행정동 추출 → 메인 컨텍스트에 반영
   ↓
summary_refresher.py   ← 영향 받는 커뮤니티의 L3 Summary 백그라운드 재생성
   ↓
data/policies/processed/  ← 처리 완료 파일 이동
```

## 예상 파일

| 파일 | 역할 |
|------|------|
| `watcher.py` | `watchdog.Observer`로 `data/policies/inbox/` 감시 |
| `preprocessor.py` | LangChain 기반 텍스트 추출/정규화 (PDF→텍스트, 마크다운 파싱) |
| `validator.py` | Pydantic `Policy` 스키마 검증. 실패 시 inbox에 `*.failed` 파일 남기기 |
| `cypher_builder.py` | 검증된 Policy를 Neo4j 노드/엣지로 변환 |
| `scope_analyzer.py` | 정책의 `TARGETS` (업종) + `APPLIED_TO` (지역)로 영향 행정동 산출 |
| `summary_refresher.py` | 영향 받는 커뮤니티의 Global Community Summary 재생성 큐잉 |

## 규칙

- **메인 시뮬레이션 루프를 블로킹하지 않을 것.** 모든 작업은 async/큐 기반.
- 실패한 정책은 inbox에 남기고 알림. 절대 조용히 삭제하지 않음.
- L3 Summary 재생성은 무거운 작업 → `infra/async_runner.py`에 위임

## 진입점

별도 프로세스로 실행: `python -m src.policy_pipeline.watcher`
