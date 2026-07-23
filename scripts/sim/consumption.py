"""인간형 소비 — 소비 필요와 정책 결제수단을 분리한 소비성향 모델.

순수 함수 모듈. 오늘의 총소비는 평소 일일소비와 Stage1의 소비성향 p∈[0,1],
Stage2가 고른 POI 가격대로 정한다. 지원금은 총소비를 강제로 추가하는 재원이 아니라
Stage2가 선택할 수 있는 별도 결제수단이며, 실제 사용액은 후단 validator가
사용처·거래액·잔액 범위 안으로만 제한한다.

핵심 설계:
  planned_total    = Stage2 거래계획 × POI 가격지수
  day_multiplier   = 오늘 p / 동일 페르소나의 평상 p
  spend_today      = planned_total × day_multiplier
  policy_spend     = Stage2 선택값(강제 생성·재배분 금지)

따라서 정책 ON/OFF에서 일정·소비성향·POI가 같으면 총소비도 같고 결제수단만 달라진다.
정책의 추가 소비효과는 Stage1/2의 선택 변화, 유동성 제약 완화, 보존된 개인 잔액의
후속 행동을 통해서만 내생적으로 나타난다.
"""
from __future__ import annotations

ANCHOR_PROPENSITY = 0.70   # p0: 지원금 無·평상일에 이 값이면 지출 = daily_wd (BDC 앵커)
INTERNAL_CATS = {"집", "직장"}   # 머무름 — 소비 대상 아님

# 소득 등급별 소비성향 prior 중심값 — MPC 이질성(저소득↑) 반영.
# Jappelli & Pistaferri(2014): 저소득·유동성제약 가구일수록 한계소비성향 높음.
INCOME_PRIOR = {"하": 0.90, "중하": 0.82, "중": 0.74, "중상": 0.66, "상": 0.58}
_DEFAULT_CENTER = 0.74

# 소비성향 LLM 출력 허용 band(중심 ± band) 및 절대 클램프
TENDENCY_SHIFT = {"saver": -0.08, "spender": +0.08, "standard": 0.0}
BAND = 0.12
HARD_LO, HARD_HI = 0.15, 0.98


def classify_tendency(tendency: str | None) -> str:
    """페르소나 소비성향 문자열 → saver | standard | spender."""
    t = (tendency or "").strip()
    if not t:
        return "standard"
    if any(k in t for k in ("절약", "검소", "알뜰", "실속", "근검")):
        return "saver"
    if any(k in t for k in ("과소비", "충동", "플렉스", "과시", "소비형", "적극")):
        return "spender"
    return "standard"


def propensity_center(
    income_tier: str | None,
    balance: float | int | None = None,
    daily_wd: float | int | None = None,
    tendency: str | None = None,
) -> float:
    """소득·저축·성향 반영 소비성향 중심값 (LLM 출력의 prior·클램프 중심).

    - 소득 등급이 prior의 기본 골격(저소득↑).
    - 저축(balance)이 일일소비의 60일치보다 많으면 약간 하향(여유 버퍼).
    - 성향(saver/spender)으로 ±shift.
    """
    center = INCOME_PRIOR.get((income_tier or "").strip(), _DEFAULT_CENTER)
    # 저축 여유 보정 (작게) — 부유 버퍼가 클수록 소비성향 소폭↓
    if balance and daily_wd and daily_wd > 0:
        ratio = float(balance) / (float(daily_wd) * 60.0)
        if ratio > 1.0:
            center -= 0.05 * min((ratio - 1.0), 2.0) / 2.0   # 최대 −0.05
    center += TENDENCY_SHIFT.get(classify_tendency(tendency), 0.0)
    return max(HARD_LO, min(HARD_HI, center))


def clamp_propensity(
    p: float | None,
    income_tier: str | None,
    balance: float | int | None = None,
    daily_wd: float | int | None = None,
    tendency: str | None = None,
) -> float:
    """LLM 소비성향 출력을 prior 중심 ± BAND 로 클램프. None이면 중심값 사용.

    통계 prior(소득별 MPC 골격)를 벗어나지 못하게 가드 → LLM 노이즈에도 MPC 순서 보존.
    """
    center = propensity_center(income_tier, balance, daily_wd, tendency)
    lo = max(HARD_LO, center - BAND)
    hi = min(HARD_HI, center + BAND)
    if p is None:
        return round(center, 4)
    try:
        p = float(p)
    except (TypeError, ValueError):
        return round(center, 4)
    return round(max(lo, min(hi, p)), 4)


def available_today(daily_wd: float | int | None, grant_remaining: float | int = 0) -> float:
    """오늘의 평상 소비예산 기준.

    `grant_remaining`은 하위 호출 호환을 위해 남겨 두지만 총소비 예산에는 더하지 않는다.
    정책지갑 잔액을 여기에 더하면 매일 p×잔액이 강제로 인출되어 기하급수적으로 소진된다.
    """
    _ = grant_remaining
    return float(daily_wd or 0) / ANCHOR_PROPENSITY


def spend_today(
    propensity: float,
    daily_wd: float | int | None,
    grant_remaining: float | int = 0,
) -> dict:
    """오늘의 정책 비의존 총소비 기준을 계산한다.

    반환: {total, grant_part, own_part, available, propensity}
      total      = p × 평상 소비예산
      grant_part = 0 (결제수단 선택은 Stage2에 위임)
      own_part   = total (후단에서 실제 policy_spend만큼 개인부담이 대체됨)
    """
    p = max(0.0, min(1.0, float(propensity)))
    avail = available_today(daily_wd, grant_remaining)
    total = p * avail
    return {
        "total": int(round(total)),
        "grant_part": 0,
        "own_part": int(round(total)),
        "available": int(round(avail)),
        "propensity": round(p, 4),
    }


def distribute_budget(total: int, weights: list[float]) -> list[int]:
    """오늘 총지출을 이벤트별 가중치로 정수 배분(합 보존). 가중치 합 0이면 균등."""
    n = len(weights)
    if n == 0:
        return []
    s = sum(max(0.0, w) for w in weights)
    if s <= 0:
        weights = [1.0] * n
        s = float(n)
    raw = [total * max(0.0, w) / s for w in weights]
    out = [int(x) for x in raw]
    rem = total - sum(out)
    # 잔여(반올림 오차)는 소수부 큰 순으로 +1
    order = sorted(range(n), key=lambda i: raw[i] - out[i], reverse=True)
    for i in range(rem):
        out[order[i % n]] += 1
    return out


# POI 가격 반영 — 장바구니 가격지수의 하루 총지출 반영 한도.
# 평균적으로 basket≈1.0(poi_price 캘리브레이션)이라 집계 총량은 보존되고,
# 개별 일자의 고가/저가 선택만 총액에 반영된다.
BASKET_CLAMP = (0.80, 1.25)


def basket_price_index(weights: list[float], factors: list[float]) -> float:
    """선택된 거래들의 가중평균 가격배율 (전부 1.0이면 1.0 — 기존 동작 보존)."""
    ws = sum(weights)
    if ws > 0:
        idx = sum(w * f for w, f in zip(weights, factors)) / ws
    else:
        idx = (sum(factors) / len(factors)) if factors else 1.0
    lo, hi = BASKET_CLAMP
    return max(lo, min(hi, idx))


def _envelope_match(env: dict, e: dict) -> bool:
    """제한 예산 봉투가 이 거래에 사용 가능한가 — 정책 속성 기반 필터 합성.

    지원 필터 (전부 선택적, AND 결합 — 새 정책 유형은 필터 조합으로 표현):
      require_poi_eligible: True → 이벤트 POI가 쿠폰 사용처(coupon_eligible=True)여야 함
      categories: [L1...]   → 이벤트 카테고리가 목록에 포함
      dong_codes: {8자리..} → 이벤트 anchor 동이 집합에 포함 (지역화폐형)
    """
    if env.get("require_poi_eligible") and e.get("coupon_eligible") is not True:
        return False
    cats = env.get("categories")
    if cats and (e.get("category") not in cats):
        return False
    dongs = env.get("dong_codes")
    if dongs:
        anchor = e.get("anchor") or ""
        code = anchor.split(":", 1)[1].strip() if anchor.startswith("zone:") else ""
        if code not in dongs:
            return False
    return True


def apply_consumption_model(
    events: list[dict],
    *,
    daily: float | int | None,
    income_tier: str | None,
    tendency: str | None,
    balance: float | int | None,
    grant_avail: dict[str, int] | None = None,
    llm_propensity: float | None = None,
    restricted_envelopes: list[dict] | None = None,
) -> dict:
    """Stage2 결과(events)에 소비성향 모델을 적용 — 선택 보존 + 안전 검증.

    Stage2 LLM의 이벤트별 `actual_spent`를 실제 계획금액으로 존중한다. 선택 POI의
    가격배율과 Stage1의 '평소 대비 오늘 소비의향'을 곱해 최종 총액을 만들기 때문에
    정책이 이벤트·POI·계획금액·소비의향을 바꾸면 총소비도 증가하거나 감소할 수 있다.
    Stage2가 출력한 `policy_spend`는 그대로 보존하며 이 함수가 새로 만들거나
    거래 사이에 재배분하지 않는다.

    POI 가격 반영 (판매자 가격 채널 전제):
      거래 계획금액 × price_factor를 이벤트별 계획액으로 사용한다.
      → 비싼 곳을 고르거나 더 많은 소비를 계획하면 그날 지출이 실제로 증가한다.

    소비의향 반영:
      day_multiplier = 오늘 p / 동일 페르소나의 평상 p 중심값.
      지원금 잔액은 multiplier에 직접 들어가지 않는다. 다만 정책 정보를 본 Stage1이
      오늘 p를 다르게 판단했다면 그 선택은 총액에 반영된다.

    유동성:
      총소비 상한은 개인 잔액 + Stage2가 실제로 선택한 유효 정책결제액이다.
      정책지갑 전체 잔액이 아니라 선택한 결제액만 유동성을 완화하므로, 잔액 존재만으로
      소비가 생성되지는 않는다.

    제한 예산 봉투 (restricted_envelopes):
      정책 속성으로 사용 가능 거래를 표현하는 제약 메타데이터다.
        env = {"pid": "P010", "amount": 120000,
               "require_poi_eligible": True, "categories": None, "dong_codes": None}
      이 함수는 봉투 잔액을 소비액에 더하지 않는다. 후단 `validate_policy_spend`가
      Stage2 선택액에 대해 사용처·잔액·거래액 상한만 강제한다.

    events 를 in-place 수정(actual_spent, policy_spend). 반환: 메타 dict.
    """
    grant_avail = {k: int(v) for k, v in (grant_avail or {}).items() if int(v) > 0}
    grant_total = sum(grant_avail.values())
    envelopes = [e for e in (restricted_envelopes or []) if int(e.get("amount") or 0) > 0]

    commerce = [e for e in events if (e.get("category") not in INTERNAL_CATS) and e.get("poi_id")]
    if not commerce:
        return {"applied": False, "reason": "no_commerce"}

    # Stage2가 정한 절대 계획금액.
    weights = [max(0.0, float(e.get("actual_spent") or 0)) for e in commerce]
    # 선택 POI 가격배율 (merge_to_final_events가 부착, 없으면 1.0 — 하위호환)
    factors = [max(0.5, min(2.0, float(e.get("price_factor") or 1.0))) for e in commerce]
    basket_idx = basket_price_index(weights, factors)
    planned_weights = [w * f for w, f in zip(weights, factors)]
    # Stage2 절대 계획금액을 보존하되, POI 가격대 효과는 기존 BASKET_CLAMP 범위에서 반영한다.
    planned_total = int(round(sum(weights) * basket_idx))

    center = propensity_center(
        income_tier,
        balance=balance,
        daily_wd=daily,
        tendency=tendency,
    )
    p = clamp_propensity(
        llm_propensity,
        income_tier,
        balance=balance,
        daily_wd=daily,
        tendency=tendency,
    )
    day_multiplier = p / center if center > 0 else 1.0

    # Stage2가 선택한 결제액은 보존한다. 여기서는 진단용 요청액만 집계하고,
    # 실제 허용액은 후단 validate_policy_spend가 결정한다.
    requested_by_pid: dict[str, int] = {}
    for e in commerce:
        ps = e.get("policy_spend") or {}
        if not isinstance(ps, dict):
            continue
        for pid, amount in ps.items():
            try:
                value = int(amount)
            except (TypeError, ValueError):
                continue
            if value > 0:
                requested_by_pid[str(pid)] = requested_by_pid.get(str(pid), 0) + value

    envelope_requested: dict[str, int] = {}
    envelope_eligible_events: dict[str, int] = {}
    for env in envelopes:
        pid = str(env.get("pid") or "")
        idxs = [i for i, e in enumerate(commerce) if _envelope_match(env, e)]
        envelope_eligible_events[pid] = len(idxs)
        requested = 0
        for i in idxs:
            ps = commerce[i].get("policy_spend") or {}
            if not isinstance(ps, dict):
                continue
            try:
                requested += max(0, int(ps.get(pid, 0) or 0))
            except (TypeError, ValueError):
                continue
        envelope_requested[pid] = requested

    # 실제 선택된 정책결제액만 유동성으로 인정한다. 거래별 계획액·정책별 잔액·
    # 사용처 제약을 모두 만족하는 범위만 계산해, 과대 요청이 개인 잔액 부족을
    # 가리는 일이 없도록 한다. 최종 정수 보정은 validate_policy_spend가 담당한다.
    env_by_pid = {str(env.get("pid") or ""): env for env in envelopes}
    wallet_remaining = dict(grant_avail)
    for env in envelopes:
        wallet_remaining[str(env.get("pid") or "")] = int(env.get("amount") or 0)
    selected_by_pid: dict[str, int] = {}
    selected_policy_liquidity = 0
    for i, e in enumerate(commerce):
        ps = e.get("policy_spend") or {}
        if not isinstance(ps, dict):
            continue
        tx_room = max(0, int(round(planned_weights[i] * day_multiplier)))
        for raw_pid, raw_amount in ps.items():
            pid = str(raw_pid)
            try:
                requested = max(0, int(raw_amount))
            except (TypeError, ValueError):
                continue
            if requested <= 0 or tx_room <= 0:
                continue
            if pid in env_by_pid:
                if not _envelope_match(env_by_pid[pid], e):
                    continue
            elif pid not in grant_avail:
                continue
            allowed = min(requested, tx_room, max(0, wallet_remaining.get(pid, 0)))
            if allowed <= 0:
                continue
            selected_policy_liquidity += allowed
            selected_by_pid[pid] = selected_by_pid.get(pid, 0) + allowed
            wallet_remaining[pid] -= allowed
            tx_room -= allowed

    desired_total = int(round(planned_total * day_multiplier))
    affordability_cap: int | None
    try:
        affordability_cap = max(0, int(balance)) + selected_policy_liquidity
    except (TypeError, ValueError):
        affordability_cap = None
    total_adj = (
        min(desired_total, affordability_cap)
        if affordability_cap is not None
        else desired_total
    )
    spends = distribute_budget(total_adj, planned_weights)

    for e, sp in zip(commerce, spends):
        e["actual_spent"] = int(sp)

    normal_budget = spend_today(p, daily)

    return {
        "applied": True,
        "propensity": p,
        "propensity_center": round(center, 4),
        "day_multiplier": round(day_multiplier, 4),
        "planned_total": planned_total,
        "today_total": total_adj,
        "grant_part": sum(requested_by_pid.values()),
        "normal_budget": normal_budget["total"],
        "available": normal_budget["available"],
        "affordability_cap": affordability_cap,
        "selected_policy_liquidity": selected_policy_liquidity,
        "selected_policy_liquidity_by_pid": selected_by_pid,
        "policy_wallet_available": grant_total + sum(int(e.get("amount") or 0) for e in envelopes),
        "policy_spend_requested": requested_by_pid,
        "mechanical_policy_uplift": 0,
        "price_basket_idx": round(basket_idx, 4),
        "envelope_requested": envelope_requested,
        "envelope_eligible_events": envelope_eligible_events,
        "n_commerce": len(commerce),
    }


# =========================================================
# 자체 테스트
# =========================================================
if __name__ == "__main__":
    print("=== ① calibration: 지원금 없는 평상일, p=p0 ⇒ spend ≈ daily_wd ===")
    for wd in (30000, 50000, 90000):
        r = spend_today(ANCHOR_PROPENSITY, wd, 0)
        print(f"  daily_wd={wd:,} → total={r['total']:,} (available={r['available']:,})")

    print("\n=== ② 지원금 잔액은 총소비·인출액을 기계적으로 늘리지 않음 ===")
    cases = [
        ("하",   30000, 600000),
        ("중하", 45000, 450000),
        ("중",   60000, 250000),
        ("중상", 90000, 100000),
    ]
    print(f"  {'소득':<4} {'p_center':>8} {'grant':>9} {'total':>11} {'forced':>9}")
    for inc, wd, grant in cases:
        p = clamp_propensity(None, inc, balance=500000, daily_wd=wd)  # None → 중심값
        r = spend_today(p, wd, grant)
        print(f"  {inc:<4} {p:>8.3f} {grant:>9,} {r['total']:>11,} {r['grant_part']:>9,}")
        assert r["grant_part"] == 0
        assert r["total"] == spend_today(p, wd, 0)["total"]

    print("\n=== ③ 분배: 오늘 총지출 47,000원을 4개 이벤트(가중치)로 ===")
    print("  ", distribute_budget(47000, [3, 1, 2, 1]), "합", sum(distribute_budget(47000, [3, 1, 2, 1])))

    print("\n=== ④ apply_consumption_model: Stage2 결제 선택 보존 ===")
    evs = [
        {"category": "집", "poi_id": "R_1", "actual_spent": 0, "policy_spend": {}},
        {"category": "식사", "poi_id": "C_1", "actual_spent": 9000, "policy_spend": {"P009": 5000}},
        {"category": "카페", "poi_id": "C_2", "actual_spent": 5000, "policy_spend": {}},
        {"category": "마트", "poi_id": "C_3", "actual_spent": 30000, "policy_spend": {}},
    ]
    meta = apply_consumption_model(
        evs, daily=30000, income_tier="하", tendency="알뜰 절약형",
        balance=500000, grant_avail={"P009": 600000}, llm_propensity=None,
    )
    print("  meta:", meta)
    tot = sum(e["actual_spent"] for e in evs if e["category"] not in INTERNAL_CATS)
    gpt = sum(sum(e.get("policy_spend", {}).values()) for e in evs)
    print(f"  거래별 지출: {[e['actual_spent'] for e in evs if e['category'] not in INTERNAL_CATS]} 합={tot:,}")
    print(f"  지원금 사용분 합={gpt:,} (today_total={meta['today_total']:,}, grant_part={meta['grant_part']:,})")
    assert tot == meta["today_total"], "지출 합 = today_total 불일치"
    assert gpt == meta["grant_part"], "정책사용 합 = grant_part 불일치"
    assert evs[1]["policy_spend"] == {"P009": 5000}, "Stage2 결제 선택을 덮어쓰면 안 됨"
    assert meta["mechanical_policy_uplift"] == 0
    assert meta["price_basket_idx"] == 1.0, "price_factor 미지정이면 basket=1.0 (기존 동작 보존)"
    print("  ✔ 합 일치 검증 통과")

    print("\n=== ⑤ POI 가격 반영: 같은 조건에서 고가/저가 장바구니 → 총지출 반응 ===")
    def _mk(pf):
        return [
            {"category": "식사", "poi_id": "C_1", "actual_spent": 9000, "price_factor": pf, "policy_spend": {}},
            {"category": "카페", "poi_id": "C_2", "actual_spent": 5000, "price_factor": pf, "policy_spend": {}},
        ]
    results = {}
    for label, pf in [("저가(₩ 0.75)", 0.75), ("기본(1.0)", 1.0), ("고가(₩₩₩ 1.35)", 1.35)]:
        evs2 = _mk(pf)
        m = apply_consumption_model(
            evs2, daily=30000, income_tier="중", tendency="", balance=500000,
            grant_avail=None, llm_propensity=None,
        )
        results[pf] = m["today_total"]
        print(f"  {label}: today_total={m['today_total']:,} (basket={m['price_basket_idx']})")
    assert results[0.75] < results[1.0] < results[1.35], "가격대가 총지출에 단조 반영되어야 함"
    # 클램프 확인: basket ≤ 1.25
    assert results[1.35] <= int(round(results[1.0] * 1.25)) + 2
    print("  ✔ 가격 반응·클램프 검증 통과")

    print("\n=== ⑥ 제한 예산 봉투: 사용 가능성만 표현하고 소비를 강제하지 않음 ===")
    def _mk6():
        return [
            {"category": "식사", "poi_id": "C_E", "actual_spent": 10000, "price_factor": 1.0,
             "coupon_eligible": True, "policy_spend": {"P010": 8000}},
            {"category": "쇼핑", "poi_id": "C_D", "actual_spent": 30000, "price_factor": 1.0,
             "coupon_eligible": False, "policy_spend": {}},
        ]
    kw6 = dict(daily=30000, income_tier="중", tendency="", balance=500000,
               grant_avail=None, llm_propensity=None)
    base6 = _mk6(); apply_consumption_model(base6, **kw6)
    env6 = _mk6()
    m6 = apply_consumption_model(env6, **kw6, restricted_envelopes=[
        {"pid": "P010", "amount": 100000, "require_poi_eligible": True}])
    assert sum(e["actual_spent"] for e in base6) == sum(e["actual_spent"] for e in env6)
    assert env6[0]["policy_spend"] == {"P010": 8000}
    assert m6["envelope_requested"]["P010"] == 8000
    assert m6["envelope_eligible_events"]["P010"] == 1

    zero_use = _mk6()
    zero_use[0]["policy_spend"] = {}
    m7 = apply_consumption_model(zero_use, **kw6, restricted_envelopes=[
        {"pid": "P010", "amount": 100000, "require_poi_eligible": True}])
    assert zero_use[0]["policy_spend"] == {}
    assert m7["envelope_requested"]["P010"] == 0
    print("  ✔ 봉투가 사용액·총소비를 강제하지 않음")
