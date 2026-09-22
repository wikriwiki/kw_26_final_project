# (위약) 장보기 환급

**위약** · `P090` · 기전 `sector_voucher` · 2021-10-11 ~ 2021-10-31

위약. 있지도 않은 정책에 반응하는지 보는 검정이다.

---

## 정책이 무엇을 하는가

동네 마트와 슈퍼마켓에서 쓴 금액의 20%를 다음 달에 돌려줍니다. 1인 월 최대 2만원이며 수량이 정해져 있어 선착순으로 소진되면 받을 수 없습니다. 다른 업종에서 쓴 금액은 인정되지 않습니다.

## 정답지

출처 — 없음 — 위약. 기전 처리 여부를 보는 검정이다.

| 지표 | 기대 | 무엇을 재는가 |
|---|---|---|
| `PL-1` | + | 마트 지출 증가 — 가짜 정책의 대상 업종. 기전을 처리하면 오른다 |
| `PL-2` | 0 | 대상 아닌 업종은 무반응 — 위약이 아무 데나 효과를 만들지 않는지 |

## 이 폴더에 무엇이 있는가

```
policy_data/policy.json
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
