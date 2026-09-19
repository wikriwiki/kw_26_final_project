# `codex/fe039529-execution-audit` 검토 — 무엇을 가져오고 무엇을 미룰까

## 2026-09-19 실제 반영 결과

2차 A단계가 끝나 `v5 유지`가 확정된 뒤 `0430c83`까지 다시 검토했다. 이번에는
시뮬레이션 행동을 바꾸지 않는 측정·판정 수정만 현재 브랜치에 옮겼다.

- `score_policy.py`: 업종 지출 0원을 관측으로 유지, 0원 INCLUDES 조회,
  쌍체차 순서 고정, 역순 날짜 거부, `_common` 패키지 import 명시
- `rank_candidates.py`: 후보 간 공통 정책이 없으면 순위를 만들지 않고 종료,
  이항 p값의 독립성·귀무확률 한계를 출력
- `tests/unit/sim/test_score_policy_execution_audit.py`: 업종 진입·이탈·0원 관측,
  쌍체 순서, 날짜 방향, 공통 정책 부재 회귀 검사

`consumption.py`, 방문 기억 시점, 원자적 agent-day 복구, 경험·정책입장 파이프라인은
가져오지 않았다. 이 변경들은 다음 날 프롬프트 또는 실제 소비 결과를 바꾸므로 기존
v5 훈련 결과와 이후 테스트 정책을 같은 시뮬레이터에서 비교한다는 조건을 깨뜨린다.
별도 실험 세대를 선언하고 훈련 정책 기준선을 다시 만들 때 검토한다.

이번 채점 수정도 과거 JSON 수치를 자동으로 고쳐 주지는 않는다. 과거 결과는 구 채점기로
측정한 역사 기록으로 유지하며, 새 채점기 결과와 한 표에 섞을 때는 세대를 구분한다.

doinggyu 가 읽어 달라고 한 브랜치를 살펴본 결과다. **가져올 것이 있다. 다만 지금
당장 병합하면 안 되는 것이 섞여 있어 순서가 중요하다.**

- 분기점: `fe03952` (우리도 가진 커밋). 그 위로 **6개 커밋, 31파일, +2,639 / −256**
- 분기 시점이 **우리 최근 작업 전부보다 앞선다** → 통째 병합은 우리 것을 지운다

---

## 0. 한 줄 결론

> **`score_policy.py` 의 측정 버그 수정 한 가지는 값이 크고 반드시 가져와야 한다.**
> 시뮬레이터 동작을 바꾸는 변경(`consumption.py`·`run_simulation.py`)은 **지금 병합하면
> 진행 중인 2차 선별이 통째로 비교 불능이 된다.** 실험이 끝난 뒤에 붙여라.

---

## 1. 통째 병합(merge/rebase)은 하지 마라

그쪽 브랜치는 `fe03952` 에서 갈라져 우리 최근 작업을 모른다.

| 우리 것 | 그쪽 브랜치 |
|---|---|
| `score_policy.py` 의 `sector_share:` 메트릭 | 없음 |
| `elig_spend_share` | 없음 |
| 진단(diagnostics) 블록 | 없음 |
| `apply_policy_eligibility` (정책별 적격 판정) | 없음 |
| `eligibility.py` 의 `require_same_district` | 없음 |
| `rank_round2.py` · `lookahead_probe.py` | 없음 |
| `prompts/v7~v9.py` · 2차 후보 | 없음 |
| `docs/HANDOFF_CODEX.md` · `INSIGHTS.md` | 없음 |

**파일 단위로 골라 옮겨라.** 특히 `score_policy.py` 는 양쪽이 서로 다른 부분을 고쳤으므로
**손으로 합쳐야 한다.**

---

## 2. 지금 가져올 것 — 측정 버그 (값이 크다)

### 2.1 0 원 지출은 "결측"이 아니라 "관측"이다 ★ 가장 중요

```python
# scripts/sim/score_policy.py — per_agent_daily
 acc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
+# Zero category spending is an observation, not a missing person.
+for row in rows:
+    acc[row["aid"]]
 for r in rows:
     if keep(r):
         acc[r["aid"]][r["d"]] += r["amt"]
```

**무엇이 문제였나.** 지금 우리 코드는 해당 업종에서 **한 번이라도 쓴 사람만** 딕셔너리에
넣는다. 그다음 `paired()` 가 off 와 on 의 교집합을 잡으므로, 결국 **양쪽 창에서 모두 그
업종을 쓴 사람만** 채점된다.

그러면 이렇게 된다.

- 무정책 때 마트를 안 쓰다가 정책 때 쓰기 시작한 사람 → **통째로 빠진다**
- 즉 **정책이 새로 끌어들인 사람이 집계에서 사라진다**
- 이것은 결과로 표본을 고르는 것(post-treatment selection)이라 **효과를 과소추정**한다

**얼마나 빠졌나.** 우리가 "관측부족"이라고 적은 것들이 대부분 이 현상이다.

| 지표 | 보고한 n | 실제 표본 | 빠진 사람 |
|---|---|---|---|
| P012-2 제외업종 | **4** | 200 | 196 |
| HO-4 마트 금액 (홀드아웃) | **114** | 200 | 86 |
| DS-4 카페 (거리두기 n=500) | **311** | 500 | 189 |
| DS-2 소매 | 481 | 500 | 19 |
| HO-2 여행·숙박·체육 | 극소 | 200 | 대부분 |

**중요 — 우리 대표 수치는 영향받지 않는다.** 홀드아웃 핵심인 HO-1 과 거리두기 진단값은
`per_agent_share` 를 쓰는데, 그 함수는 **총지출이 있는 사람 전부**를 분모에 넣으므로
처음부터 이 편향이 없다. 영향받는 것은 **금액 지표(`sector_spend:` · `excl_spend_paired`)**
뿐이다.

**그래도 반드시 고쳐야 하는 이유** — "관측부족"으로 접어 둔 지표들이 사실은 관측이
있었을 수 있다. 특히 **제외업종 무반응(P012-2)** 은 방어선 지표인데 n=4 라 판정을 포기
했었다. 200명 전부를 넣으면 판정 가능해진다.

### 2.2 쌍체차의 순서를 고정한다

```python
-return [on[a] - off[a] for a in on if a in off]
+return [on[a] - off[a] for a in sorted(set(on) & set(off))]
```

`boot_ci` 가 `rnd.randrange(k)` 로 리스트를 인덱싱하므로 **리스트 순서가 바뀌면 같은
시드라도 다른 CI 가 나온다.** 딕셔너리 순서는 삽입 순서라 런마다 달라질 수 있다.
정렬하면 재현성이 생긴다. **비용 0, 이득 명확.**

### 2.3 창 방향 검사

```python
+if e < s:
+    raise ValueError("date range must be ascending")
```

off/on 을 거꾸로 주면 지금은 빈 리스트가 나와 조용히 0건으로 채점된다. 막아 준다.

### 2.4 원장 조회에서 0원 행을 버리지 않는다

```python
-WHERE coalesce(i.actual_spent,0) > 0 AND toString(pl.day) IN $days
+WHERE toString(pl.day) IN $days
-RETURN ... i.actual_spent AS amt,
+RETURN ... coalesce(i.actual_spent, 0) AS amt,
```

2.1 과 짝이다. 계획은 했는데 0원 쓴 방문도 "그날 활동한 사람"으로 잡힌다.

> ⚠ 다만 이건 `per_agent_share` 의 분모에도 영향을 준다(0원 행은 합에 0 을 더하므로
> 값은 그대로지만, **총지출이 0 인 사람이 새로 들어와** `tot > 0` 필터에 걸려 제외된다).
> 함께 옮기되 **share 지표의 n 이 변하는지 확인**하라.

### 2.5 언제 옮길까

**2차 선별이 끝난 뒤.** 지금 A단계는 옛 채점기로 채점됐다. 중간에 채점기를 바꾸면
A단계와 B단계가 다른 자로 재어 비교가 안 된다.

**권장 순서**

1. 2차 B단계까지 지금 채점기로 끝낸다
2. 채점기를 고친다(2.1~2.4)
3. **원장이 남아 있는 런을 다시 채점**하고, 옛 수치와 **나란히** 보고한다
   — 원장은 다음 런의 `97_reset_run_artifacts.py` 가 지우므로, 다시 채점하려면
   그 런을 다시 돌려야 한다. 홀드아웃은 다시 돌리지 않는다(§5)
4. 앞으로의 런은 새 채점기로

---

## 3. 지금 병합하면 안 되는 것 — 시뮬레이터 동작 변경

아래는 **에이전트가 실제로 어떻게 행동하는지**를 바꾼다. 병합하는 순간 **그 전에 돌린
모든 런과 비교가 불가능**해진다. 우리는 지금 2차 선별 한가운데 있다.

### 3.1 `consumption.py` — 소비 태세가 "선택하지 않은 결제"를 되살리지 못하게 함

```python
-elif _posture > _cur:
-    _t = (_posture - _cur) / max(1e-9, 1.0 - _cur)
-    _choice_shares = [(s + _t * (1.0 - s)) if _elig[i] > 0 else s ...]
+elif _posture > _cur and _cur > 1e-9:
+    # A daily preference may scale selected payments, but cannot turn
+    # an explicitly unselected payment into consent.
+    _choice_shares = [min(1.0, s * _k) for s in _choice_shares]
```

**옳은 방향의 수정으로 보인다.** 예전 코드는 하루 소비 태세가 높으면 Stage2 가 0 으로
둔 결제까지 끌어올렸다 — 즉 **"안 사기로 한 것"을 태세가 뒤집었다.** 우리 프롬프트가
"닿지 않으면 평소와 같다"로 선택을 존중하도록 짜여 있는데 배관이 그 선택을 덮어쓰고
있었던 셈이다.

> **이것이 우리 실험에 주는 함의가 있다.** 섭동런에서 적립업종이 음수로 나왔던 현상,
> 그리고 지역화폐 총소비가 밴드를 넘던 현상이 이 되살리기와 관련 있을 수 있다.
> **다만 지금은 확인하지 마라** — 실험이 끝난 뒤 같은 창에서 전/후를 비교해야 한다.

### 3.2 `consumption.py` — 실제 현금으로 살 수 있는 만큼만 산다

```python
+# The final payment choice, not theoretical wallet capacity, must fund purchases.
+cash_required = sum(...) - int(allocation["total"])
+if own_balance is not None and cash_required > own_balance:
+    # 이분 탐색으로 장바구니를 줄인다
```

예전에는 잔액을 넘겨 살 수 있었다. 이것도 옳은 수정으로 보이나, **소비 총액의 분포를
바꾼다.** 우리가 "하루 총액이 페르소나 앵커에 묶여 있다"고 적어 둔 구조적 제약의
성질 자체가 달라질 수 있다.

### 3.3 `run_simulation.py` (+395 / −256)

하루 단위 원자성(atomic agent-day), 복구 시 증거 보존, 완전한 하루 장벽(complete-day
barrier). 메모리에 적어 둔 **"resume 시 Night Phase 중복 적재"** 문제를 정면으로 다루는
것으로 보인다. 값이 크지만 실행 경로를 광범위하게 바꾼다.

---

## 4. 나중에 보면 좋을 것

| 것 | 왜 |
|---|---|
| `tests/unit/sim/*` 7개 파일 (+800줄) | **공짜 안전망.** 병합 안 해도 읽고 우리 코드에 맞춰 가져올 가치가 있다. 특히 `test_policy_wallet_neutrality.py`(지갑 중립성), `test_policy_prompt_timing.py`(시점 누출) |
| `analyze_policy_response.py` (305줄) | 쌍 시나리오 분석 + **명시적 분모**. 우리 `score_policy.py` 와 목적이 겹치나 접근이 다르다 — "결측의 상한(missingness bounds)"을 명시하는 부분은 우리가 "관측부족"으로 접어 둔 것을 정량화하는 방법일 수 있다 |
| `evidence_contract.py` · `agent_day_store.py` | 해시로 원장 무결성을 검사. 우리가 겪은 "런 도중 백필로 오염" 같은 사고를 구조적으로 막는다 |
| `experience.py` 계열 (240+166+42줄) | 실행된 결제 → 관찰 → 정책 평가로 이어지는 경험 파이프라인. **새 기능**이라 우리 실험 설계와 맞물리는지 따로 판단해야 한다 |
| `docs/POLICY_RESPONSE_DESIGN.md` 등 문서 4개 | 설계 의도. 먼저 읽어라 |

---

## 5. 건드리면 안 되는 경계

- **홀드아웃(P015)은 이미 썼다.** 채점기를 고쳤다고 다시 채점하지 마라. 기준선 런의
  원장은 이미 지워졌고, 다시 돌리면 "한 번만"이 깨진다. HO-1(대표 수치)은 share 지표라
  이 수정의 영향을 받지 않으므로 **그대로 유효하다**
- **P010.json 수정 금지**
- 시뮬레이터를 바꾼 뒤의 수치와 그 전의 수치를 **한 표에 섞어 놓지 마라.** 섞을 거면
  어느 쪽 코드로 낸 것인지 칸을 나눠 적어라

---

## 6. 정리 — 추천 순서

1. **지금**: 아무것도 병합하지 않는다. 2차 B단계를 지금 코드로 끝낸다
2. **끝난 직후**: `score_policy.py` 의 §2.1~2.4 를 **손으로** 우리 파일에 옮긴다
   (통째 복사 금지 — 우리 sector_share·diagnostics·apply_policy_eligibility 가 지워진다)
3. 새 채점기로 **P012-2 제외업종**처럼 "관측부족"으로 접었던 지표를 다시 본다
4. 그다음 `tests/` 를 가져와 회귀 그물을 친다
5. 시뮬레이터 변경(§3)은 **별도 실험으로** — 같은 창에서 전/후를 돌려 무엇이
   달라지는지 먼저 재고, 그 결과를 기록한 뒤에 채택 여부를 정한다
6. `experience.py` 계열은 설계 문서를 읽고 우리 목표와 맞는지부터 판단한다
