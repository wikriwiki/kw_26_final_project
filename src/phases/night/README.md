# `src/phases/night/` — Phase 2: 상호작용 & 메모리

**하루 종료 시점에 실행.** Day t의 Plan(=Log)을 활용해 에이전트 간 상호작용을 시뮬레이션하고, 결과를 Memory Stream에 반영합니다.

## 실행 흐름

```
Daily Activity Buffer (Plan을 Log로 활용)
   ↓
target_selector.py     ← Exposure × Relation × Urgency 점수로 후보 선정
   ↓
대상 Persona B 선정
   ↓
intent_classifier.py   ← LLM 의도 분류
   ↓
interaction_summary.py ← 기타/약속/이슈/추천 카테고리로 요약
   ↓
memory_writer.py       ← Memory Stream 업데이트
   ↓
plan_injector.py       ← (약속이면) 다음 날 Plan에 주입 ─── Feedback Loop
```

## 예상 파일

| 파일 | 역할 |
|------|------|
| `activity_buffer.py` | Day t Plan을 메모리 버퍼로 로딩 (`graph/queries/plan_episode.py` 사용) |
| `target_selector.py` | Exposure(공간), Relation(KNOWS), Urgency(긴급도) 3개 점수 합산해 상위 N명 선정 |
| `intent_classifier.py` | LLM으로 상호작용 의도 분류 (small model 권장) |
| `interaction_summary.py` | 의도별 요약 — `기타`/`약속`/`이슈`/`추천` 4종 |
| `memory_writer.py` | 요약을 `MemoryItem`으로 변환 후 `REMEMBERS` 엣지 추가 |
| `plan_injector.py` | 의도 == `약속`이면 상대방의 다음 날 Plan에 Episode 강제 주입 |

## 규칙

- **target_selector는 LLM 호출 금지** (비용 폭발). 점수 기반 휴리스틱만.
- 의도 분류와 요약은 **분리된 LLM 호출**로 — 의도 결과에 따라 요약 프롬프트가 달라짐
- Feedback Loop(약속 → 다음 날 Plan)는 `plan_injector`에서만 발생. Dawn 코드는 이걸 모름.

## 입력/출력

- **입력**: Day t Plan 전체 (전 에이전트)
- **출력**: Memory Stream 업데이트 + (선택적) Day t+1 Plan 주입
