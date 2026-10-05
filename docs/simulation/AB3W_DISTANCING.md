# 사회적 거리두기 2단계(2020-11-24) — 3주 A/B 실행 (2026-10-06)

엔진·실행기 설명: `docs/KW26_MERGE_20261005.md`. 검수: `data/experiments/AB3W_REVIEW_20261005.md`.

## 창과 두 갈래
- 정책 전 주(공통): 2020-11-17 ~ 11-23 (`covid_2021`: 11-17·18 은 1단계, 11-19~23 은 1.5단계)
- 정책 있음: 11-24 ~ 11-30 `covid_2021` — 실제 일정(2단계: 식당 21시 이후 포장·배달, 카페 종일 포장·배달, 21시 운영 제한 시설, 유흥 5종 집합금지)
- 정책 없음: 11-24 ~ 11-30 `covid_2020_hold_1123` — 11-23 의 1.5단계를 이어 간다. 확진 소식은 그날 것으로 정책 있음과 같다
- 정답지(서울연구원)는 2단계 이상 vs 그 아래 단계를 비교한다 — 그래서 '거리두기 전혀 없음'이 아니라 직전 단계를 대조로 쓴다
- 명부: `output/cohort/distancing_ab3w_2000.json`(2020-10 서울 주민등록 분포)

## 돌리기
```
AB_CASE=distancing AB_TAG=small AB_ROSTER=output/cohort/distancing_ab3w_small_150.json AB_PRE_DAYS=3 AB_POST_DAYS=4 bash tools/run_ab3w.sh
AB_CASE=distancing AB_TAG=main AB_ROSTER=output/cohort/distancing_ab3w_2000.json \
  AB_LLM_BASE_URL=http://localhost:<중계 포트>/v1 AB_WORKERS=64 bash tools/run_ab3w.sh
python scripts/report/paired_distancing_effect.py --restricted <BASE>/on/distancing.ledger.jsonl \
  --control <BASE>/off/distancing.ledger.jsonl --control-arm control_hold --roster <BASE>/roster.json \
  --start 2020-11-24 --end 2020-11-30 --json-out <BASE>/distancing_score.json
```

## 실측 지표와 출력
DS-1(식사·한식), DS-2(소매), DS-4(카페): distancing 원장. DS-3 은 DS-1·DS-2 에서 계산. 포장·매장 구분은 없다.

## 한계
- 규칙은 Stage1 프롬프트로만 전달되고 후보 가게를 거르지 않는다. 에이전트의 하루가 대개 21시 전에 끝나 21시 제한은 거의 물리지 않는다.
- 대조 쪽 1.5단계는 단계표에 세부 규칙이 없어 단계 이름만 보인다(없는 규칙을 지어 넣지 않았다).
- 공통 프롬프트(v53)의 "외출을 너무 보수적으로 줄이면 부자연스럽다"는 두 갈래 모두에 들어가지만 제한 효과를 줄이는 쪽으로 민다 — 바꾸면 모든 정책의 기준 행동이 바뀌어 사용자 결정 대기.
