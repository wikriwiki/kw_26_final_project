# FINAL_REPORT 생성 — 하이브리드 아키텍처 (국내 모델 대응)

> **변경 이유**: 국내 트랙 규정상 Claude Code(자율 에이전트)를 쓸 수 없다.
> "정책 JSON + dump를 던지면 Claude가 알아서 전부 생성" 방식은 폐기하고,
> **숫자 계산은 코드가 하고, LLM(K-exaone 등 국내 모델)은 좁은 역할(해설 문장·인터뷰
> 답변 생성)만 맡는 구조**로 재설계했다.
>
> 팀 피드백에서 나온 3안 중 **3번(분석 함수 메뉴 + 사용자 선택)을 뼈대로, 1번의
> "사용자에게 필요한 분석을 먼저 물어본다" 아이디어를 메뉴 선택으로 결합**했다.

---

## 왜 순수 대화형(1안 단독)이 아닌가

기존 설계(Claude 단독 에이전트)의 핵심은 Phase 1 "분석 설계 결정" — 정책 JSON을 보고
대조군이 있는지, DID가 타당한지, 어떤 지표를 뽑을지를 **LLM이 스스로 판단**하는 것이었다.
이건 Claude 수준에서는 맡길 만하지만, 국내 소형 모델(K-exaone 등)에게 자유 대화로
맡기면 잘못된 대조군을 DID라고 우기거나 숫자를 지어낼 위험이 커진다.

그래서 **그 판단 로직 자체를 파이썬 코드로 고정**했다. LLM은 이미 계산된 숫자를
문장으로 바꾸는 것과, 에이전트 trace를 인용해 인터뷰 답변을 쓰는 것만 담당한다.

---

## 아키텍처

```
정책 JSON (data/neo4j_load/policies/*.json)
        │
        ▼
scripts/report/catalog.py
  ├─ AnalysisSpec 목록 — 각 항목: id, label, applicable(ctx)->bool, run(...)
  ├─ applicable(ctx) = 코드가 판단 (LLM 아님)
  │     · income_grants 있음 → "소득 군집별 DID" 적용 가능
  │     · target_cats 있음   → "지역×업종 DID" 적용 가능
  │     · target_districts에 특정 구 있음 → "spillover" 적용 가능
  │     · trigger 분포/만족도/단골-신규는 항상 적용 가능
  └─ run(...) = generate_final_report.py의 기존 섹션 함수 재사용 (숫자·차트 계산)
        │
        ▼  (사용자가 메뉴에서 원하는 분석을 선택)
scripts/report/menu.py  ← CLI 진입점, 대화형 메뉴
        │
        ├─ 선택된 각 분석 결과(dict) → scripts/report/narrate.py
        │     LLM 호출 1회 = "이 숫자를 3~5문장으로 해설해줘" (숫자 생성 금지 규칙 명시)
        │
        ├─ 인터뷰 → generate_final_report.section5_interviews (기존 로직 그대로)
        │     scripts/sim/interview_agent.py 가 trace 인용 규칙을 이미 강제함
        │
        ▼
scripts/report/build_html.py
  → self-contained HTML 1개 (차트 base64 내장, 외부 리소스 0개) — 유일한 산출물
```

**LLM 호출은 두 곳뿐이다**:
1. `narrate.narrate()` — 계산된 JSON을 한국어 해설 문단으로 변환
2. `interview_agent.ask()` (기존 로직 재사용) — 에이전트 trace를 인용한 1인칭 인터뷰 답변

이 두 호출 모두 `scripts/sim/llm_client.py`의 OpenAI 호환 클라이언트를 통과하므로,
**국내 모델로 교체할 때 report 쪽 코드는 한 줄도 안 바꿔도 된다** — `llm_client.MODELS`에
K-exaone을 서빙하는 vLLM/SGLang 엔드포인트를 `ModelSpec`으로 등록하거나,
`LLM_MODE` / `LLM_BASE_URL` 환경변수만 그 엔드포인트로 돌리면 된다.

---

## 구현된 파일

| 파일 | 역할 |
|---|---|
| `scripts/report/catalog.py` | 분석 카탈로그 — 5개 `AnalysisSpec`(sales, spillover, triggers, regulars, satisfaction), 각각 `applicable(ctx)`로 적용 가능 여부 코드 판단, `run(...)`으로 `generate_final_report.py` 기존 함수 재사용 |
| `scripts/report/narrate.py` | 좁은 LLM 호출 — 계산된 숫자 → 한국어 해설 문단. 시스템 프롬프트에 "새 숫자 생성 금지" 명시 |
| `scripts/report/build_html.py` | 선택된 분석 결과 리스트 → self-contained HTML 조립. `generate_final_report.py`의 `HTML_STYLE`/`_figure`/`_h` 재사용해 기존 톤 유지 |
| `scripts/report/menu.py` | CLI 진입점 — 정책 선택(`select_policy_json`) → 적용 가능 분석 메뉴 출력 → 사용자가 번호로 선택(또는 `--all`) → 실행 → 해설 생성 → 인터뷰 → HTML 저장 |

### 실행 예시

```bash
# 대화형 — 정책 선택 후, 적용 가능한 분석 중 원하는 것만 골라 실행
python scripts/report/menu.py --start 2026-05-25 --days 4 \
    --policy-from 2026-05-27 --out docs/FINAL_REPORT.html

# 메뉴 없이 적용 가능한 분석 전부 실행 (기존 자동 생성과 동일한 커버리지)
python scripts/report/menu.py --start 2026-05-25 --days 4 \
    --policy-from 2026-05-27 --all --out docs/FINAL_REPORT.html

# narrate()에 쓸 모델을 국내 모델로 지정 (llm_client.MODELS에 등록된 key)
python scripts/report/menu.py --start 2026-05-25 --days 4 \
    --policy-from 2026-05-27 --model exaone --out docs/FINAL_REPORT.html
```

---

## 분석 카탈로그 상세

| id | label | 적용 조건 (코드) | 내부적으로 재사용하는 기존 함수 |
|---|---|---|---|
| `sales` | 정책 시행 전후 매출 효과 (DID) | `income_grants` 또는 `target_cats` 존재 | `gfr.run_section2` (grant형 → 소득 군집별 DID, 그 외 → 지역×업종 DID) |
| `spillover` | 간접 영향 (Spillover) | `target_districts`에 `서울특별시` 외 지역 존재 | `gfr.run_section3` |
| `triggers` | 방문 목적·이동 패턴 (trigger 분포) | 항상 | `gfr.section4_1_triggers` |
| `regulars` | 단골 vs 신규 | 항상 | `gfr.section4_2_regulars` |
| `satisfaction` | 만족도·피드백 (동기별) | 항상 | `gfr.section4_3_satisfaction` |

새 분석을 추가하려면 `scripts/report/catalog.py`에 `AnalysisSpec` 하나를 더 등록하면 된다
(숫자 계산 함수는 기존 `generate_final_report.py`에 있는 걸 재사용하거나 새로 작성).

---

## narrate() 시스템 프롬프트 (전문)

```
당신은 정책효과 시뮬레이션 보고서의 해설 작성자입니다.
이미 계산이 끝난 분석 결과(JSON)가 주어집니다.

규칙:
- JSON에 없는 숫자를 새로 만들거나 재계산하지 마세요. 있는 숫자만 문장으로 풀어쓰세요.
- 통계적 인과를 단정하지 마세요. 표본 수(n)가 작으면 "제한적으로 해석해야 한다"처럼 명시하세요.
- 정책 담당 공무원이 읽는다고 가정하고, 3~5문장의 자연스러운 한국어로 해설을 작성하세요.
- 수치는 JSON 표기를 그대로 인용하세요 (단위·반올림 임의 변경 금지).
- 결과가 기대와 다르거나 효과가 미미해도 그대로 서술하세요 (긍정적으로 포장 금지).
```

## 인터뷰 프롬프트

인터뷰는 새로 만들지 않고 `scripts/sim/generate_final_report.py::section5_interviews` +
`scripts/sim/interview_agent.py`를 그대로 재사용한다. 이미 다음을 갖추고 있다:

- 정책 유형별 질문 뱅크(`_QUESTIONS_BY_TYPE`) + 공통 5문항 + 정책 동적 질문(Q1)
- `INTERVIEW_SYSTEM` 프롬프트 — reasoning/trigger/pick_reason/pick_factor trace를
  반드시 인용하고, trace에 없는 사실은 지어내지 않도록 강제하는 규칙 포함
- positive/negative/neutral 라벨 기반 대표 샘플 추출 (`find_label_sample`)

---

## 국내 모델(EXAONE) 연결 — 완료

`scripts/sim/llm_client.py`의 `MODELS`에 `"exaone"` (`LGAI-EXAONE/EXAONE-4.0-32B-AWQ`,
family="exaone")이 이미 등록돼 있었다. `scripts/report/menu.py --model`의 **기본값을
`exaone`으로 설정**해, `narrate()`와 인터뷰(`ask()`) 둘 다 별도 지정 없이 국내 모델을
쓰도록 배선했다:

- `interview_agent.ask()`에 `mode: str | None = None` 파라미터를 추가 (기존 호출부는
  전부 하위 호환 — 인자 안 주면 이전과 동일하게 동작).
- `generate_final_report.section5_interviews()`에도 `mode` 파라미터를 추가해
  `ask()`로 그대로 전달.
- `menu.py`가 `--model`(기본 `exaone`) 값을 `narrate()`와 `section5_interviews()` 양쪽에
  동일하게 넘긴다.

실행 시 실제로 EXAONE 서버(vLLM/SGLang, `served_model_name=exaone-4.0-32b-awq`)가
`SGLANG_BASE_URL` 또는 `LLM_BASE_URL`로 떠 있어야 한다. 다른 모델로 바꾸려면
`--model qwen8b`처럼 `llm_client.MODELS`의 다른 키를 넘기면 된다.

## 남은 작업 / 확장 여지

- 현재 인터뷰는 1명(positive)만 뽑는다 — P009 예시처럼 3명(positive/negative/neutral)으로
  늘리려면 `section5_interviews` 호출을 라벨별로 3번 반복하도록 `menu.py`에서 확장.
