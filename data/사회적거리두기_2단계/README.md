# 사회적거리두기_2단계

**사회 배경 (정책 아님)**

정책 JSON 이 아니라 사회 배경이다. environment(covid_2021) 가 규제를 실어 온다.

---

## 정답지

출처 — 서울연구원 「코로나19 확산이 서울 지역에 미친 경제적 손실」(2021.4, 신한카드 서울 패널)

| 지표 | 기대 | 무엇을 재는가 |
|---|---|---|
| `DS-1` | - | 음식점 지출 감소 (실측 −14.1%) |
| `DS-2` | + | 소매업 지출 증가 (실측 +4.2%) — 같은 정책 안에서 부호가 갈리는 지점 |
| `DS-3` | rank | 소매 > 음식점 — '제약이면 다 줄어든다'로 수렴한 프롬프트를 잡는다 |
| `DS-4` | - | 카페 지출 감소 — 2단계에서 시간 무관 매장 이용 금지 |
| `DS-6` | rank | 관광특구 감소폭 > 발달상권 감소폭 (실측 −8.7% vs −4.4%) — hub 태그 확인 필요 |

## 이 폴더에 무엇이 있는가

```
policy_data/related/distancing_schedule.json
policy_data/related/seoul_cases_daily.json
policy_data/related/seoul_vaccination_review.json
policy_data/related/national_support_rules.json
policy_data/answer_key.json
policy_data/results.json
policy_data/SOURCES.md
```

`SOURCES.md` 에 각 파일이 어디서 왔는지 sha256 과 함께 적혀 있다.

## 우리가 잰 것

`policy_data/results.json` 에 채점표가 보관한 측정 기록이 그대로 들어 있다.
**한 런의 관측이며 검증 완료를 뜻하지 않는다** — 크기 비교의 조건은
`data/experiments/scoring_table.json` 의 `MAGNITUDE_CRITERION.audit_2026_09_20`
을 따른다.

---

> 원본은 옮기지 않았다. `data/neo4j_load/policies/` 가 정본이고 여기 있는 것은
> sha256 을 단 사본이다. 어긋나면 `organize_policy_data.py --check` 가 알려 준다.
