# 상생소비지원금(P012) 실행 인수인계

> EXP-001을 돌려본 사람 기준. **소비쿠폰과 달라지는 것만** 적는다.
> 설계: [POLICY_BACKTEST_SANGSAENG.md](POLICY_BACKTEST_SANGSAENG.md) · 업종 판정 근거: [SANGSAENG_UPJONG_CODE_FIX.md](SANGSAENG_UPJONG_CODE_FIX.md)

브랜치: `policy-backtest-validation`

---

## 0. 뭐가 다른가

캐시백이라 **에이전트한테 돈이 안 들어간다.** 정책지갑 없음, `[쿠폰]` 마커 없음. "2분기 월평균보다 3% 넘게 쓰면 초과분의 10%를 돌려준다"는 **정보만** 주고 실제로 더 쓰는지를 본다.

그래서 POI마다 "여기서 쓴 게 실적으로 인정되나"를 알아야 하고 → **적립/제외 라벨 백필**이 추가됐다. 이게 실행 절차상 유일한 차이다.

| | P010 쿠폰 | P012 상생 |
|---|---|---|
| 타입 | `grant` | `cashback` (지갑 미생성) |
| POI 라벨 | `coupon_eligible` (가맹점 CSV) | `sangsaeng_eligible` (**원천 업종코드**) |
| 마커 | `[쿠폰]` | `[적립]` |

---

## 1. ⚠️ 사전 준비 — 원본 상가 CSV

**git clone 외에 필요한 유일한 물건.** 덤프처럼 따로 올려야 한다.

### 왜 필요한가

POI를 Neo4j에 적재할 때 [`03_pois.py`](../scripts/neo4j_load/03_pois.py)가 원천 CSV의 **업종 소분류 코드(L3)를 읽고 그냥 버린다.** 그래서 DB에는 우리가 만든 대분류/중분류만 있고 L3가 없다.

그런데 상생 제외업종은 L3가 없으면 못 가른다. 우리 중분류 `일반주점` 안을 보면:

| 원천 L3 코드 | 업종 | 상생 |
|---|---|---|
| I21104 | 요리주점 (10,193개) | ✅ 적립 |
| I21101 / I21102 | **유흥주점·단란주점** (2,365개) | ❌ 제외 |

**이 둘은 상호명으로도 카테고리로도 구분이 안 된다.** L3 코드만이 가른다. 복권(R10410)도 `유원지·오락` 안에 같은 식으로 섞여 있다.

→ 그래서 **원천 CSV를 다시 읽어 DB의 POI와 조인**한다. 조인 키는 상가아이디이고, POI id가 `C_{상가아이디}` 규약이라 그대로 맞물린다. **POI를 다시 적재할 필요는 없다.**

### 어느 파일을 올리나

**Neo4j에 POI를 적재할 때 썼던 바로 그 CSV.**

```
소상공인시장진흥공단_상가(상권)정보_서울_YYYYMM.csv
```

| | |
|---|---|
| 놓을 위치 | `data/neo4j_load/pois/` (자동 탐색) — 또는 아무데나 두고 `--commerce-csv <경로>` |
| git에 있나 | ❌ 없음 (`.gitignore` 제외, 용량) — **덤프처럼 따로 전송** |
| 조인 키 | 상가아이디 → POI id `C_{상가아이디}` |
| 쓰는 컬럼 | 0열(상가아이디), 7열(상권업종소분류코드) — 소상공인공단 표준 포맷 |

> **다른 연월 파일밖에 없다면?** 돌아가긴 한다. 상가아이디는 등록 시 부여돼 세대 간 안정적이라 몇 달 차이면 조인율이 2~5% 떨어지는 수준이고, preflight가 WARN으로 통과시킨다. 다만 미조인 POI는 상호명·카테고리 룰로 판정되어 그만큼 유흥주점이 적립으로 남는다. **DB 만들 때 쓴 파일이 있으면 그게 정답.**

### 올리자마자 확인 (DB 불필요, 1초)

```bash
python scripts/neo4j_load/11_sangsaeng_eligibility.py --check-csv --commerce-csv <경로>
```

컬럼 위치·인코딩이 어긋나면 여기서 바로 잡힌다. 통과하면 `exit 0`.

---

## 2. 실행

본런 전에 3개. 합쳐서 5분 안쪽.

```bash
# ① 적립/제외 라벨 백필 — POI에 sangsaeng_* 속성 기록 (구조 변경 없음)
python scripts/neo4j_load/11_sangsaeng_eligibility.py --commerce-csv <CSV경로>

# ② 정책 적재
python scripts/neo4j_load/10_load_grant_policy.py data/neo4j_load/policies/P012.json

# ③ 사전점검 — FAIL이면 돌리지 말 것
python scripts/sim/policy_preflight.py data/neo4j_load/policies/P012.json
```

본런은 **평소 그대로.** 상생 때문에 바뀌는 인자 없다.

| | 재실행 |
|---|---|
| ① 백필 | **1회만** — `97_reset_run_artifacts.py`는 POI 속성을 안 지운다 |
| ② 정책 적재 | 리셋 돌렸으면 다시 (리셋이 Policy 노드를 지움) |

> `--dry-run`: 판정 분포만 출력, DB 미기록.

**백필이 쓰는 속성** — 에이전트가 보는 건 `sangsaeng_eligible` 하나뿐이고, 나머지는 채점용이다.

| 속성 | 용도 |
|---|---|
| `sangsaeng_eligible` | `[적립]` 표시 · 야간 실적 집계 |
| `sangsaeng_arm` | 제외 사유 5종 — **채점 전용** |
| `sangsaeng_kdi` | KDI 8분류 — 채점 전용 |
| `sangsaeng_src` | 판정 근거 (`upjong_code` / `rule_name` / `rule_fallback`) |

---

## 3. 날짜

`P012.json`의 `effective_from` / `effective_until`이 **정책이 켜져 있는 구간**이다. 지금 값은 예시이니 본인 런 구간에 맞춰 바꾼다.

> `effective_from` = 정책 켜고 싶은 날 (런 시작일 아님)
> `effective_until` ≥ 런 마지막 날

`dawn_context`가 날짜로 거르므로 **정책을 껐다 켜는 조작이 필요 없다.** 한 번에 쭉 돌리면 앞 구간엔 안 보이고 발효일부터 보인다 (EXP-001과 동일).

> ⚠️ **구간이 안 겹치면 에러 없이 조용히 "효과 없음"이 나온다.** 정책 노드도 배선도 정상이라 preflight도 통과한다. JSON 고쳤으면 `--start`와 한 번 대조할 것.

---

## 4. preflight — 상생에서 새로 보는 2개

### 백필 커버리지 (99% 미만 FAIL)
라벨 없는 POI가 있으면 야간 실적 집계에서 그 지출이 빠져 **에이전트가 보는 "문턱까지 남은 금액"이 틀어진다.** FAIL이면 ①을 다시.

### 업종코드 조인율 (95%↑ 정상 / 50% 미만 FAIL)

**백필은 CSV 없이 돌려도 에러 없이 끝난다** — 상호명·카테고리 룰이 대신 값을 채운다. 그러면 커버리지는 100%로 통과하는데 유흥주점 2,369개·복권 1,062개가 적립으로 남아 대조군이 깨진다. 그래서 "라벨이 있나"와 별개로 "코드로 판정했나"를 본다.

| | 뜻 | 할 일 |
|---|---|---|
| ✅ 95%↑ | 조인 정상 | 진행 |
| ⚠️ 50~95% | CSV 세대 불일치, 일부만 룰 판정 | 진행 가능 |
| ❌ 50%↓ | 조인이 통째로 실패 | 경로 확인 후 ①부터 다시 |

---

## 5. 시뮬 중

기본 모니터링 외에 하나만 — **실적 누적이 쌓이는지.** 0으로 고정이면 백필이 안 먹은 것이다.

```cypher
MATCH (s:State) WHERE s.day >= date('<발효일>')
RETURN s.day, avg(s.sangsaeng_month_spent) ORDER BY s.day
```

---

## 6. 채점 (런 종료 후)

전/후 구간을 각각 집계해서 비교한다. 시뮬과 무관한 사후 작업이다.

```bash
python scripts/sim/aggregate_period.py --start <전구간 시작> --days <일수> --tag off --out off.json
python scripts/sim/aggregate_period.py --start <후구간 시작> --days <일수> --tag on  --out on.json
python scripts/sim/validate_sangsaeng.py --off off.json --on on.json
# → output/sim/report/SANGSAENG_BACKTEST.{md,json}
```

판정은 **C2 단독** — 적립업종 소비의 에이전트별 쌍체차가 유의하게 +인가. `PASS` / `DIRECTION_ONLY`(방향만) / `NULL`.
C1(적립 vs 제외)은 제외업종 방문이 희소해 검정력이 없어 **참고 지표**로만 찍힌다. D1(KDI 업종 순위)·D2(분위별)는 보조.

> 무반응(NULL)도 유효한 결과다. 사후에 파라미터를 손봐 "성공"으로 재포장하지 않는 게 설계 원칙.

---

## 7. 함정

| 증상 | 원인 |
|---|---|
| 조인율 0%로 FAIL | CSV 경로. `--check-csv` 먼저 |
| CSV는 있는데 "인덱스가 비었다" | 컬럼 위치 불일치 — `03_pois.py`의 `C_COL`(id=0, 업종코드=7)과 대조 |
| 커버리지 100%인데 조인율 낮음 | CSV 없이/잘못된 CSV로 백필됨. 룰 fallback 상태 |
| 다 돌았는데 verdict가 NULL | 먼저 §3 날짜 겹침부터 확인 |
| `sangsaeng_month_spent` 계속 0 | 백필 미실행 → 재시뮬 필요 |
| `--seed` 인자 없다고 나옴 | 그런 인자 없음. 표본 재현성은 `--limit` 고정으로 확보됨 (설계문서 §11의 옛 명령어 주의) |

---

## 8. 참고

- **정렬 가점 안 씀** — `POLICY_POI_SORT_BOOST=0` 유지. 적립업종을 후보 상위로 올리면 효과를 만들어 넣는 셈이라 표시만 한다.
- **업종 판정이 100%는 아니다.** 한계는 [SANGSAENG_UPJONG_CODE_FIX.md](SANGSAENG_UPJONG_CODE_FIX.md) §10.
- 판정 로직만 검증 (DB·CSV 불필요):
  ```bash
  python scripts/sim/sangsaeng_eligibility.py           # 자체 테스트
  python scripts/sim/sangsaeng_eligibility.py --audit   # 247개 업종코드 전수 판정표
  ```
