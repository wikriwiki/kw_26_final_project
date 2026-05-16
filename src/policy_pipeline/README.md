# `src/policy_pipeline/` — Async 정책 주입 파이프라인

**메인 시뮬레이션 루프와 독립적으로 실행되는 백그라운드 프로세스.**
새 정책 파일이 `data/policies/inbox/`에 떨어지면 자동으로 처리해
Scope 산출 + 캐시 무효화 + 요약 재생성 큐 등록까지 진행.

---

## 처리 흐름 (현 구현)

```
data/policies/inbox/*.{txt,json}
   ↓  watcher.py (Watchdog Observer + 안정성 체크 + 큐 dedup)
   ↓  pipeline.process_policy_file()
   ├─ dedup.py             file_hash 가 이미 처리됨? skip
   ├─ loader.py            → PolicyDocument
   ├─ extractor.py         OpenAI Structured Output → ExtractedPolicy
   ├─ validator.py         도메인 룰 → ValidatedPolicy | NEEDS_REVIEW | FAILED
   ├─ [VALIDATED 경로]
   │   ├─ scope.py         PolicyScope 산출 (textual + GraphReader)
   │   ├─ invalidator.py   version_registry bump + cache_keys 열거
   │   ├─ summary_jobs.py  community/industry/seoul-wide 잡 enqueue
   │   └─ archive.py       inbox → data/policies/processed/{YYYYMMDD}/
   └─ [NEEDS_REVIEW / FAILED 경로]
       └─ archive.py       inbox → data/policies/failed/{YYYYMMDD}/ (원본 보존)
```

Neo4j 적재(`APPLIED` 상태 전이)는 별도 PR. 현재 파이프라인은 `VALIDATED` 까지만 진전.

---

## 파일 책임

| 파일 | 책임 |
|------|------|
| `models.py` | Pydantic 도메인 모델 (`PolicyDocument` → `ExtractedPolicy` → `ValidatedPolicy`). 자기일관성 검증만. |
| `vocabulary.py` | **단일 진실의 원천**. 서울 자치구 화이트리스트, 모호어/전국 표현 사전, 정규화 헬퍼. |
| `loader.py` | `.txt` / `.json` → `PolicyDocument` + 파일 해시. |
| `extractor.py` | LLM 호출 + JSON 검증. `complete_structured()` 가 있으면 OpenAI Structured Output 사용. |
| `validator.py` | 도메인 룰 (자치구 화이트리스트, 모호어, 업종/대상 어휘) → `VALIDATED` / `NEEDS_REVIEW` / `FAILED`. |
| `dedup.py` | `file_hash` 가 이전에 처리됐는지 확인 (state 로그 기준). |
| `scope.py` | `PolicyScope` 산출. `GraphReader` Protocol 로 Neo4j 조회 분리 (현 `NullGraphReader` stub). |
| `cache_keys.py` | 캐시 키 포맷 (`community_summary:{dong}:{ver}`, `industry_summary:{ind}:{ver}` 등). |
| `version_registry.py` | JSON 파일 기반 버전 카운터. context_version / policy_version / summary_version. |
| `invalidator.py` | Scope → version bump → 무효화 키 리스트 산출. |
| `summary_jobs.py` | 영향 받은 동/업종/서울 단위로 재생성 잡을 JSONL append-only 큐에 enqueue. |
| `archive.py` | inbox → processed/ 또는 failed/ 로 파일 이동. 일자별 폴더 + 해시 prefix. |
| `state.py` | 상태머신 (`DETECTED → EXTRACTING → VALIDATED/NEEDS_REVIEW/FAILED`) + JSONL 감사 로그. |
| `pipeline.py` | 위 모듈들을 직렬로 호출하는 오케스트레이터. 단일 함수 `process_policy_file(path, llm)`. |
| `watcher.py` | Watchdog Observer + 큐 + 워커 스레드. `polling=True` 로 WSL/NFS 폴백. |

---

## 핵심 설계 결정

1. **Structured Output** — `OpenAIChatClient.complete_structured(prompt, ExtractedPolicy)` 가 OpenAI 의
   strict JSON Schema 강제를 사용. 응답이 Pydantic 모델 인스턴스로 직접 옴. 코드펜스/think 태그
   파싱 규칙이 사라짐. JSON object 모드 폴백(`complete()`)도 유지.

2. **3단 Pydantic 계층** — `PolicyDocument` → `ExtractedPolicy` → `ValidatedPolicy`.
   `to_validated_policy()` 가 `requires_human_review=True` 면 거부 → downstream 이 더러운 데이터 못 봄.

3. **단일 진실의 원천** — 어휘/룰 상수가 두 모듈에 중복되던 문제 해소.
   `vocabulary.py` 한 곳에서만 정의.

4. **review_reasons** — `requires_human_review: bool` 만 두던 정보 손실을 해소.
   *왜* 검토 필요한지를 reason 리스트에 누적.

5. **버전 기반 stale 처리** — 캐시를 실제로 삭제하지 않고 `context_version` 을 +1.
   기존 키는 자연 만료. 인프라 의존(Redis 등) 없음.

6. **원문 보존, 로그 슬림** — JSONL 감사 로그는 `raw_text` 를 빼고 `raw_text_ref=file_hash` 만.
   원문 자체는 `data/policies/processed/` 또는 `failed/` 폴더에 파일로 보존.

7. **재시도 + 영구장애 분기** — LLM 호출 실패 시 RateLimit/Timeout/5xx 는 지수 백오프 재시도,
   4xx (BadRequest) 는 즉시 raise.

---

## 진입점

```bash
# 백그라운드 워커 실행
python -m src.policy_pipeline.watcher

# WSL/네트워크 마운트 환경: polling observer 사용
python -m src.policy_pipeline.watcher --polling
```

샘플 1건 수동 실행 (LLM 호출 발생):
```bash
python -m scripts.extract_policy_sample_openai data/policies/samples/policy_001_normal_consumption_coupon.txt
```

---

## 테스트

```bash
python -m pytest tests/unit/policy_pipeline -v
```

45개 단위 테스트 — vocabulary / models / validator / scope / cache 무효화 / pipeline (모킹 LLM) 커버.

---

## 미구현 / TODO

- **Neo4j 적재** — `ValidatedPolicy → Cypher write` → `APPLIED` 상태 전이.
  `src/graph/queries/policy_writer.py` 로 별도 PR.
- **Section 5 정책 영향도 수치화 (modifier)** — 도메인 전문가 calibration 대기.
- **요약 워커** — 현재는 `summary_jobs.jsonl` 에 enqueue 만. 워커가 큐를 소비해
  실제 LLM 으로 요약 생성하는 부분은 별도 PR.
- **Prompt Prefix Cache 실측 invalidation** — SGLang 서버측 prefix cache 그룹 라벨링
  필요. 현재는 `invalidation_prefix_cache_groups` 라벨만 산출.
