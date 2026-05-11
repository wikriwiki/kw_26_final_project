# `src/prompts/` — LLM 프롬프트 템플릿

**모든 프롬프트는 Jinja2 템플릿(`.jinja2`)으로 작성.** Python 문자열로 하드코딩 금지.

## 예상 파일

| 파일 | 사용처 |
|------|--------|
| `plan_generation.jinja2` | `phases/dawn/plan_generator.py` — Day t Plan 생성 |
| `intent_classification.jinja2` | `phases/night/intent_classifier.py` — 의도 분류 (small) |
| `interaction_summary.jinja2` | `phases/night/interaction_summary.py` — 4종 요약 |
| `policy_extraction.jinja2` | `policy_pipeline/preprocessor.py` — 정책 텍스트 → 구조화 |

## 규칙

- **변수 명은 일관되게**: `{{ persona }}`, `{{ state }}`, `{{ memory }}`, `{{ day }}`, `{{ policies }}`
- 시스템 프롬프트와 유저 프롬프트를 분리해서 작성 (`---SYSTEM---`, `---USER---` 마커)
- 출력 포맷은 **JSON Schema를 프롬프트 안에 명시** + 코드에서 Pydantic으로 재검증
- 프롬프트 변경 시 PR 설명에 **모델/버전/예시 입출력**을 첨부할 것

## 이유

프롬프트는 코드만큼 중요한 산출물. 분리해야 (1) 변경 추적 쉬움, (2) 비개발자 리뷰 가능, (3) A/B 테스트 가능.
