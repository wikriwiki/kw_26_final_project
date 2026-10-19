# 판매자 에이전트 행동 알고리즘 명세 — 소비 알고리즘 정합판

> 작성일: 2026-07-04 · `SELLER_AGENT_DESIGN.md` v1.1 §4를 **실제 소비 코드에 정합**시켜 구체화한 문서
> 전제 코드: 커밋 `907e097`(가격 채널) + `98d32c2`(실측 계층) — `poi_price.py` / `stage2_poi.py` / `consumption.py`
> 원칙 유지: 소비자 경로 추가 LLM 호출 0 · O(1) 룩업 · 클램프 필수 · `SELLER_ENABLED` flag

---

## 0. 정합의 핵심 — 소비 알고리즘이 판매자의 "레버"를 이미 만들어 놨다

소비 알고리즘 재구현으로 다음 인과 사슬이 **이미 코드로** 존재한다:

```
가격배율(price_factor) → 후보 표시(₩·앵커) → LLM 선택 → 장바구니 지수 → 하루 총지출 → 잔액 → 다음날 선택
```

판매자 행동 알고리즘은 이 사슬의 **입력 노드들을 시간에 따라 움직이는 것**으로 정의된다.
그래서 정합판의 가장 중요한 결론 세 가지:

1. **판매자 가격은 한 줄 곱셈으로 전 소비에 전파된다.** 유효 가격배율 = `poi_price_factor × seller_price_level`.
   이 곱은 `stage2_poi.py:226`(후보 부착 지점) 한 곳에서 이뤄지고, 이후 장바구니→총지출→잔액은
   **`consumption.py` 수정 0**으로 자동 반응한다. (v1.1의 T4 "지출 배율" 태스크는 이미 구현된 셈 — 폐기)
2. **할인 비용은 원장에 내생화된다.** 할인 참여 가게의 factor에 ×0.9를 곱하면 소비자 지출(=판매자 매출)이
   실제로 10% 줄어 기록된다. v1.1에서 "할인비용 = 매출의 3%"로 외생 처리했던 것을 삭제하고,
   할인의 손실은 시뮬이 스스로 계산하게 한다 (행사 운영비 1%만 외생).
3. **"공짜 가격 인상"은 3중 채널로 차단된다.** 인상하면 ① 앵커 상승+아이콘으로 소비자가 인지·회피(예산 규칙),
   ② Huff 매력도 페널티로 원거리 수요 감소, ③ 그래도 오는 손님에겐 지출 ↑ — 즉 현실의
   가격-수요 트레이드오프가 성립한다.

---

## 1. 상태 공간 (셀 = 행정동 × L1업종, 서울 ~1,800개)

| 필드 | 초기값 | 범위 | 갱신 규칙 |
|---|---|---|---|
| `price_level` | 1.00 | **[0.85, 1.15]**, step 0.05 | tick. 같은 방향 2연속 후 1 tick 쿨다운 |
| `promo` | none | none / discount / event | tick. 지속 1 tick, 종료 후 **쿨다운 2 tick** |
| `quality_delta` | 0.0 | **[−0.20, +0.30]** | 투자 tick +0.03 / 미투자 tick −0.01 (자연 감가) |
| `marketing` | 0 | 0 / 1 | tick. 지속 1 tick |
| `open_delta_h` | 0 | [−2, +2], step 1 | tick |
| `ledger_margin` | 0 | 누적 (원) | **매일** (§5) |
| `prev_sales`, `prev_visits` | 0 | — | tick 관측 비교용 |
| `cooldowns` | {} | — | 행동별 잔여 tick |

행동 출력(LLM JSON, 이것만 허용 — 아니면 전체 no-op):

```json
{"price_move": -1|0|1, "promo": "none"|"discount"|"event", "quality_invest": 0|1,
 "marketing": 0|1, "open_delta": -1|0|1, "reason": "한 줄"}
```

---

## 2. 하루 알고리즘 (의사코드)

```python
# ── 소비자 루프 안 (run_simulation, merge 직후) — O(1)/이벤트 ──
for ev in events:                                # 이미 도는 루프에 1줄
    if ev.poi_id and ev.category not in INTERNAL:
        cell_acc[(dong_of(ev), ev.category)] += (ev.actual_spent, 1)
        #  ev.actual_spent 는 소비 알고리즘의 최종값(가격·할인 반영) — 판매자 매출 실측

# ── 하루 종료 (SellerPool.end_of_day) ──
def end_of_day(day, cell_acc):
    for cell in cells:                            # ① 원장: 전 셀, 매일
        ledger_update(cell, cell_acc.get(cell, (0, 0)))          # §5

    due = [c for c in cells if hash(c.sid) % TICK == day_idx % TICK]   # ② 스태거 (~N/3)
    obs   = [build_obs(c) for c in due]                                 # §3
    outs  = llm_batch(SELLER_SYSTEM, obs)                               # ~600 호출
    for c, out in zip(due, outs):
        act = parse_clamp(out, c)                 # ③ 범위·쿨다운 위반 필드는 무시, 파싱 실패 no-op
        apply_action(c, act)                      # ④ 상태 갱신 (§1 규칙)
        c.prev_sales, c.prev_visits = cell_acc.get(c.key, (0, 0))

    rebuild_effects()                             # ⑤ 소비자용 O(1) 맵 4+2종 (§4)
    persist_neo4j(due)                            # ⑥ UNWIND 1회 (SellerDecision + 상태)
```

`rebuild_effects()`가 만드는 맵 (전부 dict, 소비자 hot path에서 룩업만):

```python
level_map   : {(dong,l1): price_level}
disc_pois   : {poi_id}                 # 할인 참여 점포 (§4-2 π 규칙, seed 결정론)
tag_map     : {poi_id: "[세일중]"|"[이벤트]"|"[광고]"}
ad_slot     : {(dong,l1): poi_id}      # 셀 내 최고 별점 POI
attr_ovl    : {dong: multiplier}       # Huff 오버레이 (§4-1 ②)
window_ovl  : {(dong,l1): delta_h}
rating_ovl  : {(dong,l1): quality_delta}
```

---

## 3. 관측(입력) — 소비 알고리즘의 출력 단위와 일치

관측은 전부 **소비 코드가 실제로 기록하는 값**에서 만든다 (환각 관측 금지):

| 관측 필드 | 출처 | 의미 |
|---|---|---|
| `sales`, `d_sales_pct` | `cell_acc` 합 (= Σ actual_spent, 가격·할인 반영 후) | 이번 tick 매출, 직전 대비 % |
| `visits`, `d_visits_pct` | `cell_acc` 카운트 | 방문 건수 |
| `avg_ticket` = sales/visits | 계산 | 실현 객단가 |
| `anchor` | `unit_price_anchor(dong,l1) × price_level` | 동네 기준단가 (내 가격 반영) |
| `margin_cum` | 원장 (§5) | 누적 마진 |
| `price_level, promo, quality_delta(별점 환산), open_delta` | 상태 | 현재 정책 |
| `neighbors` | 같은 동 타업종 Δ% 1줄 + 인접동 동일업종 Δ% 1줄 | 경쟁·동네 경기 |
| `policy_line` | policy_pipeline 활성 정책 1줄 | 예: "P009 지원금 지급 중" |

**SELLER_SYSTEM 프롬프트 (전문)**:

```
당신은 서울 {gu} {dong}의 {l1} 업종 상인회 대표다. 관할 점포 {n_poi}곳(프랜차이즈 {fr}%),
주 고객층 {top_mix}, 동네 평균단가 {anchor:,}원, 평균 별점 {rating_eff:.1f}.
최근 {tick}일 실적: 매출 {sales:,}원({d_sales:+.0f}%), 방문 {visits}건({d_visits:+.0f}%),
객단가 {avg_ticket:,}원, 누적 마진 {margin_cum:,}원.
주변: {neighbors} {policy_line}
현재: 가격수준 {price_level:.2f}(동네 단가에 곱해짐), 프로모션 {promo}, 영업시간 {open_delta:+d}h.
가능한 행동과 대가:
- 가격 ±5%: 올리면 객단가↑ но 손님이 가격표(₩·앵커)를 보고 발길을 돌릴 수 있음. 잦은 변경 불가.
- 할인행사: 객단가 10%↓ 대신 손님 유인([세일중] 노출). 행사 운영비 매출의 1%. 연속 불가.
- 사은이벤트: 손님 유인만, 운영비 0.5%.
- 품질투자: 매출의 5%, 별점이 서서히 오름(방치 시 서서히 내림).
- 동네 광고: 매출의 4%, 우리 셀 대표 가게가 후보 목록에 [광고]로 보장 노출.
- 영업시간 ±1h: 연장 시 야간 손님 가능 но 운영비 시간당 0.5%.
목표: 무리한 출혈 없이 매출과 마진을 유지·개선. 다음 JSON만 출력:
{"price_move":-1|0|1,"promo":"none"|"discount"|"event","quality_invest":0|1,"marketing":0|1,"open_delta":-1|0|1,"reason":"한 줄"}
```

(~650 토큰 입력 / ~80 출력 × ~600셀/일 ≈ +0.4M tok/일, 호출 +4%)

---

## 4. 행동 → 소비자 효과 (코드 지점 단위 정의)

### 4-1. A1 가격 조정 — 소비 사슬의 곱 한 번

수정 지점은 `stage2_poi.fetch_candidates_for_events` (현재 224~227행) **한 곳**:

```python
lvl = seller.level_map.get((dong_code, l1), 1.0)              # O(1)
anchor_won = round500(unit_price_anchor(dong_code, l1) * lvl)  # ① 앵커에 반영 → 소비자 인지
for c in cands:
    band, f = poi_price(c["poi_id"], dong_code, l1)
    if c["poi_id"] in seller.disc_pois: f *= 0.90              # ② 할인 참여 점포
    c["price_band"], c["price_factor"] = band, clamp(f * lvl)  # ③ 유효배율 = poi × seller
    c["unit_anchor"] = anchor_won
    c["tag"] = seller.tag_map.get(c["poi_id"], "")             # ④ [세일중]/[이벤트]/[광고]
    c["lvl_icon"] = "(가격↑)" if lvl >= 1.10 else ("(가격↓)" if lvl <= 0.90 else "")
```

이후는 기존 사슬이 전부 처리한다 (**consumption.py·plan_writer 수정 불필요**, 계측 필드 1개 제외):

| 경로 | 이미 구현된 메커니즘 | 효과 |
|---|---|---|
| 방문당 지출 | `price_factor` → 장바구니 지수 → 하루 총액 (`consumption.py`) | 인상 시 객단가↑, 잔액↓ 가속 |
| 방문 선택 | ₩아이콘 + 앵커 + "잔액 빠듯하면 ₩" 규칙 (`SYSTEM_S2`) | 인상 시 회피 (LLM 채널) |
| 광역 수요 | `mobility.suggest_hubs` 오버레이: `A_j × (1 − EPS_PRICE×(lvl_dong−1) + EPS_PROMO×promo_share)` | 인상 동네로 원정 감소 |
| 기록 | INCLUDES `price_factor` (+신규 `seller_level`) | 검증·분해 분석 |

### 4-2. A2 판촉 — 할인은 내생 비용

- `discount`: 셀 내 **참여 점포만** factor ×0.90 + `[세일중]` 태그.
  참여 π = 40 + 20×(별점 p75 초과) − 20×프랜차이즈 [%], `sha1(poi_id+day)` 결정론.
  **비용**: 매출 감소 자체가 원장에 잡힘(내생) + 운영비 1% (외생).
- `event`: 태그 `[이벤트]` + Huff promo_share 가산만 (지출 불변), 운영비 0.5%.
- 계측: 태그가 노출된 후보가 선택되면 INCLUDES에 `promo_seen=true` (리뷰 계측 `review_seen` 패턴).

### 4-3. A3 영업시간 — 시간대 분포의 공급 제약

`visit_window`의 카테고리 시간창 산출에 `window_ovl[(dong,l1)]` 시간 가감 (키워드 인자, 기본 None).
연장 → 야간 이벤트에서 해당 셀 후보 노출 가능. 운영비 시간당 0.5% (원장).

### 4-4. A4 품질 — 리뷰 채널에 얹기

`poi_review_lookup.format_review_block` 호출부에서 `rating_eff = min(5.0, rating + rating_ovl[(dong,l1)])`.
기존 리뷰 2-pass(선택적 lookup)가 그대로 전달 채널 — 추가 호출 0.
동학: 투자 +0.03/tick (cap +0.30), 미투자 −0.01/tick (floor −0.20). 비용 매출의 5%.

### 4-5. A5 광고 — 후보 목록의 보장 슬롯

`fetch_candidates_for_events`에서 후보 확정 직후: `ad = ad_slot.get((dong_code,l1))`이 있고
후보에 없으면 **마지막 후보와 교체** + `[광고]` 태그. 선택은 여전히 소비자 LLM 몫.
비용 매출의 4%. 계측 `ad_seen`.

---

## 5. 원장 (경제 제약) — 소비 실측값 기반

매일, 전 셀:

```
revenue_d  = cell_acc[cell].sales                     # 소비 알고리즘의 최종 지출 = 실측 매출
cost_d     = revenue_d × cost_ratio(l1)               # 원가율: 소상공인실태조사 (T1)
           + revenue_base × rent_ratio(dong)          # 임대 프록시: 발달지수 분위 (T1)
           + promo운영비(1%|0.5%) + quality(5%) + marketing(4%) + 0.5%×|open_delta|
ledger_margin += revenue_d − cost_d
```

- 할인 손실·가격 인상 이득은 revenue_d에 **이미 들어 있다** — 소비 알고리즘이 계산해 준 것.
- `ledger_margin`이 연속 k tick 음수 → (Phase 2) 폐업 트리거 신호.

---

## 6. Rule-baseline (3-arm 비교군의 ② — 정확한 규칙)

LLM 기여분을 분리하기 위한 기계 판매자. 관측·클램프·효과는 동일, 결정만 규칙:

```python
def rule_policy(c, obs):
    a = NOOP.copy()
    if obs.d_sales <= -10 and obs.prev_d_sales <= -10 and not cd("promo"):
        a["promo"] = "discount"                        # 2연속 부진 → 할인
    elif obs.d_sales >= +15 and c.price_level < 1.10 and not cd("price"):
        a["price_move"] = +1                           # 수요 초과 → 인상
    elif obs.d_sales <= -15 and c.price_level > 0.90 and not cd("price"):
        a["price_move"] = -1                           # 수요 급감 → 인하
    a["quality_invest"] = int(margin_rate(c) > 0.10)   # 마진 여유 → 품질 유지
    a["marketing"] = int(obs.visits < p25_same_l1 and c.ledger_margin > 0)
    return a
```

---

## 7. 기대 부호표 — V1 검증의 가드레일

| 행동 | 1차(기계적) | 2차(LLM·Huff) | 순효과 가설 | 감시 지표 |
|---|---|---|---|---|
| 가격 +5% | 객단가 +5% | 방문 − (아이콘·앵커·Huff) | 매출 ±, 마진 조건부↑ | **가격↑ 셀의 방문 Δ 부호 음(−) 정합률 ≥60%** — "공짜 인상" 감시 |
| 할인 | 객단가 −10% | 방문 + (태그) | 매출 ±, 마진 ↓ 소폭 | `promo_seen`→선택 전환율 > 비노출 대비 |
| 품질 | 비용 −5% | 별점↑ → 리뷰픽·재방문 + | 지연된 매출 + | 별점→매출 탄력 부호 + |
| 광고 | 비용 −4% | 노출 → 신규 방문 + | 방문 + | `ad_seen` 전환율 수 % 대역 |
| 영업 +1h | 비용 − | 야간 방문 + | 야간 매출 비중 ↑ | 시간대 매출 vs OA-15572 실측 |
| (전체) | — | — | 가격변경빈도 tick당 5~30% (서비스 가격경직성 정합) | V1 |

---

## 8. 파라미터 (env, 기본값)

| 변수 | 기본 | 의미 |
|---|---|---|
| `SELLER_ENABLED` | 0 | 마스터 스위치 (0이면 모든 맵 비어 있음 = 기존 동작) |
| `SELLER_TICK_DAYS` | 3 | 의사결정 주기 (스태거) |
| `SELLER_EPS_PRICE` / `SELLER_EPS_PROMO` | 0.15 / 0.10 | Huff 오버레이 탄력 |
| `SELLER_DISCOUNT` | 0.90 | 할인 배율 |
| `SELLER_ICON_TH` | 0.10 | (가격↑↓) 아이콘 표시 문턱 |
| `SELLER_COST_PROMO/EVENT/QUALITY/MKT/HOUR` | 1% / 0.5% / 5% / 4% / 0.5% | 행동 비용 (매출 대비) |
| `SELLER_Q_STEP/Q_DECAY` | +0.03 / −0.01 | 품질 동학 |

---

## 9. v1.1 설계서 대비 달라진 점 (소비 알고리즘 정합의 결과)

| 항목 | v1.1 설계 | 정합판 |
|---|---|---|
| 지출 배율 태스크(T4) | merge에 별도 구현 | **폐기** — poi 가격 채널이 이미 처리, 곱 한 줄만 |
| 할인 비용 | 매출의 3% 외생 | **내생화** (factor ×0.9가 매출에 반영) + 운영비 1%만 |
| 가격 아이콘 | 밴드에 표시 | poi 밴드(₩)와 분리 — **판매자 편차분만** `(가격↑↓)` |
| 단가 인지 | 없음 | **앵커 × price_level** — 동네 단가가 실제로 오르는 걸 소비자가 봄 |
| 관측 | 매출·방문 | + **객단가 vs 앵커** (실현 단가와 기준 단가 비교 가능) |
| 구현 접점 | 5개 파일 | 소비자측은 사실상 `stage2_poi` 1개 함수 + mobility/visit_window/review 각 1줄 훅 |

---

## 10. 구현 태스크 (정합판 T'1~T'6)

| # | 내용 | AC |
|---|---|---|
| T'1 | `seller_agent.py`: SellerPool(상태·원장·클램프·쿨다운·스태거·LLM배치·rule_policy·effects 맵) | 단위테스트: 클램프/쿨다운/no-op/원장/π 결정론 |
| T'2 | `stage2_poi.py`: §4-1 코드블록 (lvl 곱·앵커·태그·아이콘·광고슬롯) + INCLUDES `seller_level`/`promo_seen`/`ad_seen` | SELLER_ENABLED=0 → 프롬프트·이벤트 diff 0 |
| T'3 | `mobility.py` attr 오버레이 / `visit_window` 시간창 오버레이 / `poi_review_lookup` rating 오버레이 (각 키워드 인자 1개) | None이면 기존 동일 |
| T'4 | `run_simulation.py`: cell_acc 1줄 + end_of_day 배선 + env | 파일럿 3일 런 crash 0, 호출 +≤5% |
| T'5 | `seller_writer.py`: Seller/SellerDecision UNWIND | 결정 수 = due 셀 수 |
| T'6 | `validate_seller.py`: §7 부호표 자동 판정 + 3-arm | V0~V3 리포트 |

파일럿: 1개 구 ~120셀 × 3일 → §7 부호표 확인 → 전역.
