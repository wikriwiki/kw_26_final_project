"""인간형 소비 — 소비 필요와 정책 결제수단을 분리한 소비성향 모델.

순수 함수 모듈. 오늘의 총소비는 평소 일일소비와 Stage1의 소비성향 p∈[0,1],
Stage2가 고른 POI 가격대로 정한다. 지원금은 총소비를 강제로 추가하는 재원이 아니라
별도 결제수단이다. 소비가 이미 발생하기로 정해진 뒤 사용 가능한 거래에서는
정책지갑을 자기자금보다 먼저 배정하고, 후단 validator가 사용처·거래액·잔액을
한 번 더 검증한다.

핵심 설계:
  planned_total    = Stage2 거래계획 × POI 가격지수
  day_multiplier   = 오늘 p / 동일 페르소나의 평상 p
  spend_today      = planned_total × day_multiplier
  policy_spend     = 사용 가능한 실제 거래에 정책지갑 우선 배정

따라서 정책 ON/OFF에서 일정·소비성향·POI가 같으면 총소비도 같고 결제수단만 달라진다.
정책의 추가 소비효과는 Stage1/2의 선택 변화, 유동성 제약 완화, 보존된 개인 잔액의
후속 행동을 통해서만 내생적으로 나타난다.
"""
from __future__ import annotations

import os

ANCHOR_PROPENSITY = 0.70   # p0: 지원금 無·평상일에 이 값이면 지출 = daily_wd (BDC 앵커)
INTERNAL_CATS = {"집", "직장"}   # 머무름 — 소비 대상 아님

# [소진 rationing] 지원금 일일 인출 한도 비율.
# 실제 소비쿠폰은 유효기간(수개월) 내 며칠~몇 주에 걸쳐 사용된다(BOK 0주 20.9%·2주 70%).
# doing_gyu의 지갑우선결제(사용 가능 거래에 지원금 우선 배정)는 그대로 유지하되,
# 하루 인출을 '잔여 지원금 × 이 비율'로 상한해 소진율을 소비수준과 분리한다.
# 상한을 넘는 유상소비는 개인 잔액으로 결제되므로 총소비 수준은 유지된다.
# 1.0이면 상한 없음(=doing_gyu 원본 거동). 환경변수로 튜닝 가능.
COUPON_DAILY_DRAW_RATE = float(os.environ.get("COUPON_DAILY_DRAW_RATE", "0.25"))

# 소득 등급별 소비성향 prior 중심값 — MPC 이질성(저소득↑) 반영.
# Jappelli & Pistaferri(2014): 저소득·유동성제약 가구일수록 한계소비성향 높음.
INCOME_PRIOR = {"하": 0.90, "중하": 0.82, "중": 0.74, "중상": 0.66, "상": 0.58}
_DEFAULT_CENTER = 0.74

# 소비성향 LLM 출력 허용 band(중심 ± band) 및 절대 클램프
#
# BDC 소비앵커(s_daily_wd)는 그 사람의 '장기 평균 씀씀이'이지 오늘 하루의 지출액이 아니다.
# band를 좁게 잡으면 앵커가 사실상 그날 지출을 결정해 버려서, 목돈이 들어온 날·장을 몰아 보는
# 날·병원 가는 날 같은 그날의 사정이 반영될 자리가 없어진다. 실제 가계의 일별 지출 변동은
# 평균 대비 ±17%보다 훨씬 크다. 앵커는 중심값을 주는 한 요소로 두고, 그날 판단이 움직일 수
# 있는 폭을 넓힌다. 소득별 prior(INCOME_PRIOR)와 절대 클램프는 그대로 유지된다.
# [R61 측정] 밴드를 0.25로 넓혔더니 소득 등급별 중심값(0.869 vs 0.701)의 허용 구간이 서로
# 겹쳐 버렸고, LLM이 전 계층에 0.67을 답하는 탓에 계층 차이가 완전히 사라졌다(평상소비/앵커가
# 세 tier 모두 0.96배). 넓은 밴드는 'LLM에게 자유를 준다'가 아니라 '유일하게 작동하던 차이를
# 지운다'로 귀결됐다. 저장소 원래 값으로 되돌린다.
TENDENCY_SHIFT = {"saver": -0.08, "spender": +0.08, "standard": 0.0}
BAND = 0.12
HARD_LO, HARD_HI = 0.15, 0.98
# 하루 지출 중 배송 주문이 차지할 수 있는 최대 몫. LLM이 극단값을 답해도 하루 지출 전부가
# 가게 밖으로 빠져나가 방문 일정이 유명무실해지지 않도록 하는 안전장치일 뿐, 목표값이 아니다.
ONLINE_SHARE_CAP = float(os.environ.get("EXP_ONLINE_SHARE_CAP", "0.85"))

# [서울시 공개데이터 산출] 하루 지출 중 쿠폰 **사용처(소상공인)** 에서 나가는 몫.
#   서울시 상권분석서비스(추정매출-행정동) 2025년 — 생활밀착 63업종·425개 동·연 103.8조.
#   업종 목록에 백화점·대형마트·할인점·면세점이 없다 → 사용처 매출로 볼 수 있다.
#   103.8조 ÷ 서울 인구 940만 ÷ 365 = 1인·일당 30,255원.
#   우리 소비 앵커(s_daily_wd) 평균 119,371원(규격 교정 후) → 30,255/119,371 ≈ 0.2535.
# BOK 실측과 무관한 값이며, 앵커 과대와 비사용처 미분리를 함께 보정한다.
ELIGIBLE_SHARE_SEOUL = float(os.environ.get("EXP_ELIGIBLE_SHARE", "0.2535"))

# [폐기 2026-07-30] 소비수준별 '쿠폰 불가 업종' 지출 비중 표(BDC_OFFSITE_BY_LEVEL)는
# 업종 분류가 시뮬의 실제 사용처 판정(coupon_eligibility.py — 상호명 기준)과 어긋나 폐기했다.
# 상세 사유는 apply_consumption_model 안의 '기본 비활성' 주석 참조. 되살리려면 먼저
# BDC 업종별로 대형 브랜드와 동네 매장을 가를 수 있는 원자료(상호명 또는 매출규모)가 필요하다.

# 지원금 사용 계획(일) prior — 소득이 낮을수록 짧게(=빨리 소진).
#
# 근거:
#   · BOK 이슈노트 2026-13 32항 각주29 — Sahm(2019): 1인당 지급액이 커질수록 정책의
#     가시성(salience)이 높아져 정책효과가 지급액에 비례할 수 있다. 본 정책은 저소득일수록
#     지급액이 크다(기초생활수급자 40만 / 차상위 30만 / 일반 15만).
#   · BOK 31항 — 소득 1분위 MPC 0.25 > 5분위 0.17, 1·2차 서베이 모두에서 일관.
#   · Jappelli & Pistaferri(2014) — 유동성 제약이 큰 가구일수록 이전지출을 빨리 소비.
#
# ★ 해석상 주의(보고서 필수 기재): 이 prior는 **실측에서 관측된 방향을 모형 입력으로 넣은
#   캘리브레이션**이다. LLM(EXAONE-4.5-33B)이 지급액·소득에 따라 사용 계획을 차등하지 않아
#   (프롬프트 13종 실험에서 일관되게 실패) 내생적으로는 재현되지 않았다. 따라서 계층별
#   소진 속도의 '방향'은 본 시뮬의 예측이 아니라 설정값이며, 재현 대상에서 제외한다.
#   반면 소진 궤적·업종 구성·전체 MPC는 이 prior와 독립적으로 산출된 결과다.
GRANT_PLAN_PRIOR = {"하": 14, "중하": 16, "중": 22, "중상": 23, "상": 24}
_DEFAULT_PLAN_DAYS = 20
GRANT_PLAN_BAND = 2          # LLM 출력이 움직일 수 있는 폭(일)
GRANT_PLAN_LO, GRANT_PLAN_HI = 5, 60


def clamp_plan_days(v, income_tier: str | None) -> int:
    """LLM의 지원금 사용 계획(일)을 소득별 prior 중심 ± BAND 로 클램프. None이면 중심값."""
    center = GRANT_PLAN_PRIOR.get((income_tier or "").strip(), _DEFAULT_PLAN_DAYS)
    lo = max(GRANT_PLAN_LO, center - GRANT_PLAN_BAND)
    hi = min(GRANT_PLAN_HI, center + GRANT_PLAN_BAND)
    try:
        iv = int(round(float(v)))
    except (TypeError, ValueError):
        return center
    return max(lo, min(hi, iv))


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
      grant_part = 0 (실제 거래가 확정된 뒤 정책지갑 우선 정산)
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


def filter_active_grant_balances(
    previous: dict[str, int] | None,
    active_policies: list[dict] | None,
) -> tuple[dict[str, int], dict[str, int]]:
    """어제 정책지갑을 오늘도 유효한 grant와 만료·비활성 잔액으로 분리한다.

    Dawn 정책 목록은 오늘이 effective_from~effective_until 범위이고 에이전트의 적용
    지역에 해당하는 정책만 포함한다. 목록에서 사라진 잔액은 무제한 지갑으로 추정하지
    않고 만료·비활성 잔액으로 제거한다.
    """
    active_ids = {
        str(p.get("id") or "")
        for p in (active_policies or [])
        if p.get("type") == "grant" and p.get("id")
    }
    usable: dict[str, int] = {}
    inactive: dict[str, int] = {}
    for raw_pid, raw_amount in (previous or {}).items():
        pid = str(raw_pid)
        try:
            amount = max(0, int(raw_amount or 0))
        except (TypeError, ValueError):
            continue
        if amount <= 0:
            continue
        target = usable if pid in active_ids else inactive
        target[pid] = amount
    return usable, inactive


def _wallet_sort_key(wallet: dict) -> tuple:
    """제약이 좁은 지갑을 먼저 두는 결정적 순서."""
    cats = wallet.get("categories")
    dongs = wallet.get("dong_codes")
    has_cats = bool(cats)
    has_dongs = bool(dongs)
    constrained = int(bool(wallet.get("require_poi_eligible"))) + int(has_cats) + int(has_dongs)
    return (
        -constrained,
        len(cats) if has_cats else 10**9,
        len(dongs) if has_dongs else 10**9,
        str(wallet.get("pid") or ""),
    )


def _policy_wallet_specs(
    grant_avail: dict[str, int],
    envelopes: list[dict],
) -> list[dict]:
    """정책지갑 정산 순서를 만든다.

    사용처가 좁은 제한 지갑을 먼저 쓰고, 어디서나 쓸 수 있는 지갑을 나중에 쓴다.
    같은 유형 안에서는 입력 순서를 보존한다. 동일 정책 ID가 중복되면 제한 봉투 정의를
    우선하며 이중 계상하지 않는다.
    """
    specs: list[dict] = []
    seen: set[str] = set()
    for env in envelopes:
        pid = str(env.get("pid") or "")
        amount = max(0, int(env.get("amount") or 0))
        if not pid or amount <= 0 or pid in seen:
            continue
        specs.append({**env, "pid": pid, "amount": amount})
        seen.add(pid)
    for raw_pid, raw_amount in grant_avail.items():
        pid = str(raw_pid)
        amount = max(0, int(raw_amount or 0))
        if not pid or amount <= 0 or pid in seen:
            continue
        specs.append({"pid": pid, "amount": amount})
        seen.add(pid)
    return sorted(specs, key=_wallet_sort_key)


def _allocate_policy_capacity(
    events: list[dict],
    amounts: list[int],
    wallet_specs: list[dict],
) -> dict:
    """거래–정책지갑 최대흐름으로 사용 가능한 결제액을 최대화한다.

    단순 탐욕 배정은 범용 지갑을 앞 거래에 먼저 써서 뒤의 전용 지갑과 거래를 함께
    놓칠 수 있다. 정수 최대흐름은 입력 순서와 무관하게 전체 정책결제액을 최대화한다.
    이벤트·지갑 수가 매우 작아 LLM/DB 비용과 비교하면 계산비용은 무시할 수준이다.
    """
    tx_amounts = [max(0, int(v or 0)) for v in amounts]
    n_wallets = len(wallet_specs)
    n_events = len(events)
    allocations: list[dict[str, int]] = [{} for _ in events]
    eligible_flags = [
        any(_envelope_match(wallet, event) for wallet in wallet_specs)
        for event in events
    ]

    if not wallet_specs or not events:
        return {
            "total": 0,
            "by_pid": {},
            "allocations": allocations,
            "eligible_spend_total": sum(
                amount for amount, eligible in zip(tx_amounts, eligible_flags) if eligible
            ),
            "eligible_event_count": sum(
                1 for amount, eligible in zip(tx_amounts, eligible_flags) if amount > 0 and eligible
            ),
        }

    source = 0
    wallet_base = 1
    event_base = wallet_base + n_wallets
    sink = event_base + n_events
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(u: int, v: int, capacity: int) -> int:
        forward_idx = len(graph[u])
        reverse_idx = len(graph[v])
        graph[u].append([v, max(0, int(capacity)), reverse_idx, max(0, int(capacity))])
        graph[v].append([u, 0, forward_idx, 0])
        return forward_idx

    edge_refs: dict[tuple[int, int], tuple[int, int]] = {}
    # 지갑 한도가 하루 거래 총액보다 작을 때, 어느 거래에 쿠폰을 쓸지는 사람이 고른다.
    # 사람은 아침 편의점 몇천 원보다 병원비·장보기처럼 목돈이 나가는 자리에서 쿠폰을 꺼낸다.
    # 최대흐름 총액은 순서와 무관하므로, 큰 거래부터 간선을 놓아 그 선택을 반영한다.
    order = sorted(range(n_events), key=lambda i: -tx_amounts[i])
    for wi, wallet in enumerate(wallet_specs):
        wallet_node = wallet_base + wi
        add_edge(source, wallet_node, int(wallet.get("amount") or 0))
        for ei in order:
            event, amount = events[ei], tx_amounts[ei]
            if amount <= 0 or not _envelope_match(wallet, event):
                continue
            edge_idx = add_edge(wallet_node, event_base + ei, amount)
            edge_refs[(wi, ei)] = (wallet_node, edge_idx)
    for ei, amount in enumerate(tx_amounts):
        add_edge(event_base + ei, sink, amount)

    while True:
        level = [-1] * len(graph)
        level[source] = 0
        queue = [source]
        for u in queue:
            for to, capacity, _rev, _original in graph[u]:
                if capacity > 0 and level[to] < 0:
                    level[to] = level[u] + 1
                    queue.append(to)
        if level[sink] < 0:
            break

        cursor = [0] * len(graph)

        def send_flow(node: int, flow: int) -> int:
            if node == sink:
                return flow
            while cursor[node] < len(graph[node]):
                edge = graph[node][cursor[node]]
                to, capacity, reverse_idx, _original = edge
                if capacity > 0 and level[to] == level[node] + 1:
                    sent = send_flow(to, min(flow, capacity))
                    if sent > 0:
                        edge[1] -= sent
                        graph[to][reverse_idx][1] += sent
                        return sent
                cursor[node] += 1
            return 0

        while send_flow(source, 10**18) > 0:
            pass

    by_pid: dict[str, int] = {}
    for (wi, ei), (node, edge_idx) in edge_refs.items():
        edge = graph[node][edge_idx]
        used = edge[3] - edge[1]
        if used <= 0:
            continue
        pid = str(wallet_specs[wi]["pid"])
        allocations[ei][pid] = used
        by_pid[pid] = by_pid.get(pid, 0) + used

    return {
        "total": sum(by_pid.values()),
        "by_pid": by_pid,
        "allocations": allocations,
        "eligible_spend_total": sum(
            amount for amount, eligible in zip(tx_amounts, eligible_flags) if eligible
        ),
        "eligible_event_count": sum(
            1 for amount, eligible in zip(tx_amounts, eligible_flags) if amount > 0 and eligible
        ),
    }


def settle_policy_spend_priority(
    events: list[dict],
    *,
    grant_avail: dict[str, int] | None = None,
    restricted_envelopes: list[dict] | None = None,
    grant_use: float | None = None,
    # 거래별 '이 결제를 지원금으로 낼 몫'(0~1). Stage2가 건별로 고른 결제수단이다.
    # 주어지면 지갑우선결제 규칙 대신 이 선택을 그대로 따른다.
    choice_shares: list[float] | None = None,
) -> dict:
    """소비모델과 무관하게 실제 사용 가능 거래에 정책지갑을 우선 정산한다."""
    grants = {
        str(k): int(v)
        for k, v in (grant_avail or {}).items()
        if int(v or 0) > 0
    }
    envelopes = [
        e for e in (restricted_envelopes or [])
        if int(e.get("amount") or 0) > 0
    ]
    commerce = [
        e for e in events
        if (e.get("category") not in INTERNAL_CATS) and e.get("poi_id")
    ]
    requested_by_pid: dict[str, int] = {}
    for event in commerce:
        policy_spend = event.get("policy_spend") or {}
        if not isinstance(policy_spend, dict):
            continue
        for raw_pid, raw_amount in policy_spend.items():
            try:
                amount = max(0, int(raw_amount or 0))
            except (TypeError, ValueError):
                continue
            if amount > 0:
                pid = str(raw_pid)
                requested_by_pid[pid] = requested_by_pid.get(pid, 0) + amount

    wallet_specs = _policy_wallet_specs(grants, envelopes)
    # 지갑우선결제 강도 = LLM의 '오늘 지원금 사용의향'(grant_use). 1.0=가맹점 거래 전액을
    #   지원금으로(원본). 0.x=거래액의 그 비율만 지원금 배정, 나머지는 자기자금 → 소진을 늦춘다.
    #   현금이 빠듯한 사람은 높게(지원금 의존), 여유롭고 기한 넉넉하면 낮게(아껴 나눠 씀).
    #   grant_use가 None이면 실험용 환경변수(EXP_GRANT_USE)로 폴백.
    if grant_use is not None:
        _guse = max(0.0, min(1.0, float(grant_use)))
    else:
        _guse = float(os.environ.get("EXP_GRANT_USE", "1.0"))
    if choice_shares is not None and len(choice_shares) == len(commerce):
        # [결제 선택 모드] 쓸 수 있는 매장이라고 지원금이 자동으로 나가지 않는다. 계산할 때
        # 지원금을 꺼낼지 늘 쓰던 카드로 낼지는 건별로 이 사람이 정한 것이고, 여기서는 그
        # 선택을 회계적으로 따를 뿐이다. 지갑 잔액과 사용처 조건만 그 위에 걸린다.
        amounts = [
            max(0, int(round((e.get("actual_spent") or 0) * max(0.0, min(1.0, float(s or 0.0))))))
            for e, s in zip(commerce, choice_shares)
        ]
    else:
        amounts = [max(0, int((e.get("actual_spent") or 0) * _guse)) for e in commerce]
    # 하루 지출이 오늘 쓸 쿠폰보다 많으면, 쿠폰을 한 자리에 몰아 쓰기보다 그날 지출 전반에 얹어
    # 쓴다(밥값에도, 장값에도, 병원비에도 조금씩). 거래별로 받을 수 있는 몫을 지출 크기에 비례해
    # 두면 결과적으로 쿠폰 사용 업종 구성이 그날 소비 구성을 따라간다.
    _wallet_total = sum(int(w.get("amount") or 0) for w in wallet_specs)
    _spend_total = sum(amounts)
    if 0 < _wallet_total < _spend_total:
        _scale = _wallet_total / _spend_total
        amounts = [max(0, int(round(a * _scale))) for a in amounts]
    allocation = _allocate_policy_capacity(commerce, amounts, wallet_specs)
    for event, policy_spend in zip(commerce, allocation["allocations"]):
        event["policy_spend"] = policy_spend
    allocation["requested_by_pid"] = requested_by_pid
    allocation["wallet_available"] = sum(int(w["amount"]) for w in wallet_specs)
    return allocation


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
    grant_use: float | None = None,
    grant_spread_days: int | None = None,
    grant_extra_spend: float | None = None,
    grant_kept_share: float | None = None,
    grant_carry: int | None = None,
    # 지원금 받은 시점에 세운 사용 계획(일). 있으면 LLM의 오늘 답보다 이 값을 따른다.
    grant_plan_days: int | None = None,
    # 오늘 지출 중 동네 가게 계산대 밖으로 나가는 몫(0~1). POI 방문이 없는 지출이다.
    online_share: float | None = None,
    # 에이전트의 BDC 소비수준(1~10). 주어지면 위 실측 비중을 online_share보다 우선한다.
    spending_level: int | None = None,
) -> dict:
    """Stage2 결과(events)에 소비성향 모델을 적용 — 선택 보존 + 안전 검증.

    Stage2 LLM의 이벤트별 `actual_spent`를 실제 계획금액으로 존중한다. 선택 POI의
    가격배율과 Stage1의 '평소 대비 오늘 소비의향'을 곱해 최종 총액을 만들기 때문에
    정책이 이벤트·POI·계획금액·소비의향을 바꾸면 총소비도 증가하거나 감소할 수 있다.
    Stage2가 출력한 `policy_spend`는 원요청 진단값으로만 보존한다. 최종 소비액이
    정해진 뒤 사용 가능한 거래에는 정책지갑을 자기자금보다 먼저 배정한다.

    POI 가격 반영 (판매자 가격 채널 전제):
      거래 계획금액 × price_factor를 이벤트별 계획액으로 사용한다.
      → 비싼 곳을 고르거나 더 많은 소비를 계획하면 그날 지출이 실제로 증가한다.

    소비의향 반영:
      day_multiplier = 오늘 p / 동일 페르소나의 평상 p 중심값.
      지원금 잔액은 multiplier에 직접 들어가지 않는다. 다만 정책 정보를 본 Stage1이
      오늘 p를 다르게 판단했다면 그 선택은 총액에 반영된다.

    유동성:
      총소비 상한은 개인 잔액 + 오늘 계획된 사용 가능 거래에 실제로 배정할 수 있는
      정책결제액이다. 정책지갑 전체 잔액을 총소비에 더하지 않으므로 잔액 존재만으로
      소비가 생성되지는 않는다.

    제한 예산 봉투 (restricted_envelopes):
      정책 속성으로 사용 가능 거래를 표현하는 제약 메타데이터다.
        env = {"pid": "P010", "amount": 120000,
               "require_poi_eligible": True, "categories": None, "dong_codes": None}
      이 함수는 봉투 잔액을 소비액에 더하지 않는다. 소비계획이 정해진 뒤 일치하는
      실제 거래액 한도에서만 지원금을 우선 결제한다.

    events 를 in-place 수정(actual_spent, policy_spend). 반환: 메타 dict.
    """
    grant_avail = {k: int(v) for k, v in (grant_avail or {}).items() if int(v) > 0}
    grant_total = sum(grant_avail.values())
    envelopes = [e for e in (restricted_envelopes or []) if int(e.get("amount") or 0) > 0]
    wallet_total = grant_total + sum(int(e.get("amount") or 0) for e in envelopes)

    # 오늘 쓸 지원금 = 남은 지원금 ÷ (앞으로 며칠에 걸쳐 쓸지). 그 일수는 Stage1에서 본인이
    # 판단한 계획(grant_spread_days)이며, 여기서는 그 계획을 회계적으로 따를 뿐이다.
    # 계획이 없으면 실험용 환경변수 → 기존 상수 순으로 폴백하고, 그것도 없으면 상한 없음.
    # 계획은 지원금을 받은 시점에 세우고 그대로 이어 간다. 매일 다시 물으면 잔액이 줄수록
    # 짧게 답해 소진이 가속되는데(R54 측정: spread 20.0→17.4→15.9일, hazard 5.50→6.61%),
    # 실측 곡선은 반대로 감속한다(표1: 4주 76.4 → 5~8주 주당 4.1%p → 9~12주 주당 1.2%p).
    # 계획은 받은 날 한 번 정해지고(그때 소득별 prior로 클램프) 이후 그대로 이어진다.
    _plan = int(grant_plan_days or 0)
    # [폐기] 소득별 계획 prior(clamp_plan_days)는 쓰지 않는다. 소득 등급으로 사용 일수를
    # 직접 지정하는 것은 실측 방향을 답으로 넣는 것이라, 계층별 소진율이 시뮬의 결과가 아니라
    # 설정값이 된다. 프롬프트에 "저소득은 빨리 쓴다"고 적는 것과 형태만 다를 뿐 같은 일이다.
    _eff_spread = _plan if _plan > 0 else (int(grant_spread_days) if grant_spread_days else 0)
    if _eff_spread > 0:
        _draw = 1.0 / max(1, _eff_spread)
    else:
        _env_spread = float(os.environ.get("EXP_SPREAD_DAYS", "0") or 0)
        if _env_spread >= 1:
            _draw = 1.0 / _env_spread
        elif 0 < COUPON_DAILY_DRAW_RATE < 1:
            _draw = COUPON_DAILY_DRAW_RATE
        else:
            _draw = 1.0
    intended_grant_today = int(round(wallet_total * _draw))
    # [배급 표현] 사람이 실제로 정하는 것은 '남은 돈을 며칠로 나눌지'가 아니라 '지금 계산할 때
    # 지원금으로 낼지 자기 돈으로 낼지'다. 후자로 표현하면 오늘 나가는 금액이 그날 지출에 걸리므로,
    # 지갑이 클수록 하루 인출 한도가 작아져 구조적으로 느려지던 편향이 사라진다.
    # 남은 금액이 씀씀이에 비해 넉넉한 사람은 가려 쓸 이유가 없어 그냥 지원금으로 내고,
    # 하루이틀이면 사라질 사람은 아껴 두는 것 — 그 판단이 grant_use다.
    # [결제 선택 모드] 하루 인출 계획(지갑÷계획일수)으로 배급하지 않는다. 사람이 실제로 하는
    # 일은 '오늘 얼마를 꺼낼지' 정하는 것이 아니라 계산대에서 건별로 무엇으로 낼지 고르는 것이고,
    # 소진 속도는 그 선택의 결과여야 한다. 배급으로 두면 소진율이 1/계획일수로 고정돼 페르소나
    # 차이가 결과에 남지 않는다(측정: 배정 갭이 요청 갭보다 항상 압축됨).
    # 기본값 켜짐. 끄면 소진율이 다시 '잔액 ÷ LLM이 답한 계획일수'로 결정되는데, 그것은
    # 검증하려는 값(소진 속도)을 LLM 답변으로 직접 지정하는 순환이다. 되살리지 않는다.
    _choice_mode = os.environ.get("EXP_PAYMENT_CHOICE", "1") not in ("0", "false", "False")
    # 결제 선택 모드에서는 grant_use를 옛 '강도 배급'으로 쓰지 않는다. 하루 단위 태세
    # (오늘 쓸 수 있는 자리에서 얼마나 챙겨 쓰는가)로만 쓰며, 건별 선택을 그 태세에 맞춰
    # 재조정하는 목표값이 된다.
    _intensity_mode = (grant_use is not None) and not _choice_mode
    if _intensity_mode:
        _draw = 1.0
        intended_grant_today = wallet_total   # 상한은 지갑 잔액뿐. 실제 지출액은 grant_use가 정한다.
    # 어제 쓰려 했으나 그날 쓸 거래가 없어 못 쓴 몫은 사라지지 않는다. 계획은 그대로인데
    # 하루 소비가 적었을 뿐이므로, 다음에 쓸 데가 생기면 그만큼 앞당겨 쓰게 된다.
    # 이 이월이 없으면 '하루 인출 계획'이 큰 사람(지갑이 큰 쪽)만 매일 계획에 못 미쳐
    # 소진이 구조적으로 느려진다 — 지갑 크기와 무관해야 할 부분에서 생기는 편향이다.
    # 다만 밀린 몫을 무한정 쌓아 두지는 않는다. 며칠씩 계속 계획에 못 미치는 사람은
    # 계획 자체를 다시 잡지, 못 쓴 몫을 장부에 계속 적립해 두었다가 한꺼번에 쏟지 않는다.
    # 따라잡기는 '다음 기회에 하루치를 한 번 더' 수준까지로 본다.
    # 강도(grant_use)로 배급하면 '계획 미달'이라는 개념 자체가 없으므로 이월도 없다.
    # 상한 없이 이월한다. 이전에는 '하루치까지'로 묶었는데, 그 이유였던 후반부 폭주는
    # 계획을 고정한 뒤로는 생기지 않는다 — 일일 인출이 잔액/N 이라 잔액이 줄면 함께 줄고,
    # 아래에서 지갑 잔액으로 한 번 더 막힌다. 상한이 남아 있으면 계획 미달분의 회수가
    # 절반에서 멈춰(포착률 90.7%) 계획이 가장 짧은 tier만 손해를 본다.
    if _choice_mode:
        # 상한은 지갑 잔액뿐. 오늘 얼마가 나갈지는 아래 건별 선택이 정한다.
        _draw = 1.0
        intended_grant_today = wallet_total
    # 배급이 없으면 '계획에 못 미친 몫'이라는 개념도 없으므로 이월도 없다.
    _carry = 0 if (_intensity_mode or _choice_mode) else max(0, int(grant_carry or 0))
    if _carry > 0:
        intended_grant_today = min(wallet_total, intended_grant_today + _carry)
    _draw = (intended_grant_today / wallet_total) if wallet_total > 0 else _draw
    grant_avail_alloc = (
        grant_avail if _draw >= 1.0
        else {k: max(0, int(round(v * _draw))) for k, v in grant_avail.items()}
    )
    grant_avail_alloc = {k: v for k, v in grant_avail_alloc.items() if v > 0}
    # 봉투(제한 예산) 잔액도 같은 계획으로 하루 상한.
    envelopes_alloc = [
        {**e, "amount": max(0, int(round(int(e.get("amount") or 0) * _draw)))}
        for e in envelopes
    ] if _draw < 1.0 else envelopes
    envelopes_alloc = [e for e in envelopes_alloc if int(e.get("amount") or 0) > 0]

    commerce = [e for e in events if (e.get("category") not in INTERNAL_CATS) and e.get("poi_id")]
    if not commerce:
        # 오늘 거래가 아예 없으면 계획한 인출을 통째로 못 쓴 것이다. 그 몫이 여기서 사라지면
        # 이월의 정의('그날 쓸 거래가 없어 못 쓴 몫')와 코드가 어긋난다 — 다음 날로 넘긴다.
        return {
            "applied": False, "reason": "no_commerce",
            "grant_carry_in": _carry,
            "grant_carry_out": min(int(wallet_total), int(intended_grant_today)),
        }

    # Stage2가 정한 절대 계획금액.
    weights = [max(0.0, float(e.get("actual_spent") or 0)) for e in commerce]
    # 선택 POI 가격배율 (merge_to_final_events가 부착, 없으면 1.0 — 하위호환)
    factors = [max(0.5, min(2.0, float(e.get("price_factor") or 1.0))) for e in commerce]
    basket_idx = basket_price_index(weights, factors)
    planned_weights = [w * f for w, f in zip(weights, factors)]
    # Stage2가 건별로 고른 결제수단을 '그 거래의 몇 %를 지원금으로 냈는가'로 환산해 둔다.
    # 아래에서 총액이 앵커로 재조정되어도 이 비율은 그대로 따라간다 — 사람이 정한 것은
    # 금액이 아니라 '이 결제를 무엇으로 낼지'이기 때문이다.
    _choice_shares: list[float] = []
    for _e in commerce:
        _base = max(0.0, float(_e.get("actual_spent") or 0))
        _ps = _e.get("policy_spend") or {}
        _req = 0
        if isinstance(_ps, dict):
            for _v in _ps.values():
                try:
                    _req += max(0, int(_v or 0))
                except (TypeError, ValueError):
                    continue
        _choice_shares.append(min(1.0, _req / _base) if _base > 0 else 0.0)

    # [태세 정합] Stage2는 결제수단을 POI 선택에 딸린 부차 필드로 취급해, 어떤 지시를 넣어도
    # 건별 선택 비율이 0.19 근처로 고정됐다(R80~R84 측정: 40만 0.19 / 15만 0.19). 반면
    # Stage1은 하루를 통째로 보고 한 번만 판단하므로 태세(grant_use)에는 상황이 반영된다.
    # 그래서 '어느 결제에 쓸지'는 Stage2의 선택을 그대로 두고, '전체적으로 얼마나 챙겨 쓰는지'만
    # Stage1 태세에 맞춰 비례 조정한다. 둘 중 하나를 버리지 않는다.
    _posture = None
    if _choice_mode and grant_use is not None:
        try:
            _posture = max(0.0, min(1.0, float(grant_use)))
        except (TypeError, ValueError):
            _posture = None
    if _posture is not None:
        _elig = [
            (planned_weights[i] if commerce[i].get("coupon_eligible") is not False else 0.0)
            for i in range(len(commerce))
        ]
        _wsum = sum(_elig)
        if _wsum > 0:
            _cur = sum(_elig[i] * _choice_shares[i] for i in range(len(commerce))) / _wsum
            if _posture <= _cur and _cur > 1e-9:
                # 태세가 더 낮으면 고른 것들을 같은 비율로 줄인다(어느 결제를 골랐는지는 보존).
                _k = _posture / _cur
                _choice_shares = [s * _k for s in _choice_shares]
            elif _posture > _cur:
                # 태세가 더 높으면 각 결제를 전액 쪽으로 같은 정도만큼 끌어올린다.
                # 비례 배율은 Stage2가 0으로 둔 결제를 영원히 0으로 남겨 태세를 달성할 수 없다.
                _t = (_posture - _cur) / max(1e-9, 1.0 - _cur)
                _t = max(0.0, min(1.0, _t))
                _choice_shares = [
                    (s + _t * (1.0 - s)) if _elig[i] > 0 else s
                    for i, s in enumerate(_choice_shares)
                ]
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

    # Stage2가 제안한 결제액은 원인 분석용 요청액으로 먼저 집계한다.
    # 최종 결제액은 아래에서 실제 거래·사용처·잔액 기준으로 다시 정산한다.
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

    # [하이브리드 캘리브레이션] 소비수준을 구코드 daily_wd 앵커로 복원한다.
    # Stage2 계획액(planned_total)은 실측 s_daily_wd의 ~1/3로 저평가되므로, 총액 크기는
    # p×daily_wd/ANCHOR(=spend_today)에 POI 가격효과(basket_idx)를 곱한 값으로 앵커한다.
    # 이벤트/POI 선택 분포(planned_weights)와 지원금 우선정산(아래)은 그대로 보존한다.
    # day_multiplier는 진단값으로만 유지(총액에는 spend_today의 p가 이미 반영됨).
    # basket_idx는 1.0 미만으로 총액을 끌어내리지 않도록 클램프한다(하한 = daily_wd 앵커).
    # 평상 basket은 spend_today(=p×daily_wd/ANCHOR) 그대로 유지하고, 비싼 POI를 고르면
    # (basket_idx>1) 총액이 상향된다 — doing_gyu의 가격채널(비싼 곳→지출↑)은 보존.
    _anchor_total = int(round(spend_today(p, daily)["total"] * max(1.0, basket_idx)))
    # [정책 효과 경로 복원 2026-07-31]
    # 위 앵커는 '평상시 하루'의 크기다. 그런데 지원금이 들어온 날은 평상시가 아니다 —
    # 미뤄둔 병원, 바닥난 생필품, 아이 옷·신발처럼 목돈이 드는 일이 그날 일정에 들어온다
    # (Stage1 프롬프트가 그렇게 판단하도록 하고 있다).
    # 문제: 총액을 항상 앵커로 덮어쓰면 그 계획이 사라진다. 에이전트가 20만원짜리 지출을
    # 계획해도 총액은 앵커×p로 되돌아가고, p의 상한(0.98)은 앵커의 1.4배에 불과해
    # **정책 효과의 천장이 1.4배로 고정된다.** 12일치 지출에 해당하는 지원금은 흡수될 수 없다.
    # 조치: 에이전트가 계획한 금액이 앵커를 넘으면 그 계획을 따른다. 평상일에는 Stage2가
    # 앵커보다 낮게 잡으므로(구조적 저평가) 앵커가 그대로 유지되고, **특별한 날에만** 계획이
    # 총액을 끌어올린다. MPC 등 검증 대상 값이 입력으로 들어가지 않으므로 순환이 아니다 —
    # 증가분은 전적으로 에이전트 자신의 이벤트 계획에서 나온다.
    personal_total = max(_anchor_total, int(round(planned_total)))
    # [온라인 채널] 하루 지출이 전부 동네 가게 계산대에서 나가지는 않는다. 집에서 주문해 배송으로
    # 받는 지출은 POI 방문 없이 자기 돈으로 나가고, 소비쿠폰은 온라인 결제에 쓸 수 없다
    # (P010 사용처 조건). 이 채널이 없으면 하루 지출 전액이 가맹점으로 흘러 1인당 가맹점 지출이
    # 실제보다 크게 잡힌다(측정: 비가맹 지출 비중 1.6%, 1인당 가맹점 지출 59,184원/일).
    # 몫은 Stage1에서 본인이 판단하며(online_share), 여기서는 그 판단을 회계적으로 따를 뿐이다.
    # [활성 — 서울시 공개데이터로 근거 확보 2026-07-30]
    # 소비 앵커(daily_wd)는 '이 사람의 모든 카드소비'인데, 시뮬은 그 전액을 동네 가맹점 POI로
    # 흘려보내고 있었다(그래프에 대형마트·백화점 매장 실체가 없어 갈 곳이 없다). 그래서 쿠폰
    # 사용처 지출이 실제보다 크게 잡히고 소진이 과속했다.
    # 사용처 지출 수준을 서울시 공개데이터로 잡는다(BOK 무관):
    #   · 서울시 상권분석서비스(추정매출-행정동) 2025년 = 생활밀착 63업종, 425개 동, 연 103.8조.
    #     이 업종 목록에 백화점·대형마트·할인점·면세점이 **없다** → 곧 쿠폰 사용처 매출이다.
    #   · 103.8조 ÷ 서울 인구 940만 ÷ 365일 = 1인·일당 30,255원
    #   · 우리 앵커 평균 128,721원 → 사용처 비율 30,255/128,721 ≈ 0.235
    #   교차검증: 앵커를 서울 전체로 환산하면 441조/년인데 서울 개인카드는 약 180조(전국 900조의
    #   20%)이므로 앵커가 2.4배 과대하고, 소상공인 비중 103.8/180 = 0.58 → 53,000×0.58 = 30,740원.
    #   두 경로가 30,000원대로 수렴한다.
    # 이 값은 '앵커 과대'와 '비사용처 미분리'를 함께 보정한다. 둘을 분리하려면 BDC 동별 절대
    # 매출이 필요한데 확보되지 않았다(b069_sales는 지수값). 그 한계를 보고서에 명시할 것.
    _off = 1.0 - ELIGIBLE_SHARE_SEOUL
    _online_rate = max(0.0, min(ONLINE_SHARE_CAP, _off))
    _online_src = "seoul_smallbiz"
    # [이전 경로 폐기 기록]
    #  · 소비수준별 BDC 업종 비중 표: 업종 분류가 사용처 판정 규칙(상호명 기준)과 어긋나 폐기.
    #    '할인점/슈퍼마켓'(14.45%)은 동네 슈퍼가 대부분 사용 가능인데 전부 불가로 넣었었다.
    #  · LLM 판단(online_share): 계층 구분 없이 0.2 근처로 균일해 정보가 없었다(R82·R83 측정).
    #    필드는 진단용으로 남기고 비중 산정에는 쓰지 않는다.
    # EXP_ELIGIBLE_SHARE 로 민감도 실험 가능(기본값은 위 서울시 데이터 산출값).
    online_planned = int(round(personal_total * _online_rate))
    # 오프라인(가게 방문) 지출만 아래 POI 배분·지원금 정산을 거친다.
    personal_total = max(0, personal_total - online_planned)
    wallet_specs = _policy_wallet_specs(grant_avail_alloc, envelopes_alloc)

    # 쿠폰은 먼저 '어차피 하려던 소비'를 대체한다(그만큼 현금이 굳는다). 오늘 쓰려는 지원금이
    # 그 대체 가능액을 넘어서면, 넘는 만큼은 여력이 없어 미뤄왔던 소비가 오늘 실행되는 것이다.
    # 평소 소비가 큰 사람은 대체로 끝나 총소비가 그대로고, 평소 소비가 빠듯했던 사람일수록
    # 미뤄둔 필요가 풀려 총소비가 늘어난다 — 유동성 제약의 차이가 결과로 나타난다.
    _base_spends = distribute_budget(personal_total, planned_weights)
    _base_capacity = _allocate_policy_capacity(commerce, _base_spends, wallet_specs)
    eligible_base = int(_base_capacity["total"])
    # 오늘 실제로 쿠폰이 결제할 수 있는 금액(= 계획 소비 중 사용처 조건을 만족하는 부분,
    # 단 오늘 쓰려는 지원금 한도까지). 이 금액만큼 자기 현금이 굳는다.
    # 오늘 쿠폰이 대체하게 될 '어차피 했을 지출'. 강도 모드에서는 사용처 결제 중 지원금으로 내기로
    # 한 몫(eligible_base × grant_use)이 그 크기이고, 배급 모드에서는 하루 인출 계획이 상한이다.
    if _intensity_mode:
        substituted = min(wallet_total, int(round(eligible_base * max(0.0, min(1.0, float(grant_use))))))
    elif _choice_mode:
        # 오늘 지원금으로 결제하기로 고른 만큼이 대체액이다(사용처·지갑 한도 안에서).
        _chosen = sum(
            b * max(0.0, min(1.0, s)) for b, s in zip(_base_spends, _choice_shares)
        )
        substituted = min(wallet_total, eligible_base, int(round(_chosen)))
    else:
        substituted = min(intended_grant_today, eligible_base)
    # 굳은 현금 중 오늘 더 쓰는 데 돌리는 비율은 본인 판단(grant_extra_spend).
    # 값이 없으면 굳은 돈을 그냥 남겨 두는 것으로 본다(추가 소비 없음).
    # [MPC 산출 — 참고3 ⑤와 같은 형태] 지원금으로 결제한 건마다 '없었어도 했을 지출인가'를
    # 0/1로 받아, 결제금액으로 가중평균한 것이 그날의 신규 소비 유발 비중 m이다.
    #   m = Σ(그 건의 지원금 결제액 × 신규여부) / Σ(지원금 결제액)
    # 스칼라 하나(grant_kept_share)로 물으면 LLM이 계층 구분 없이 같은 값을 답한다(R84~R86
    # 측정). 건별 0/1은 각 지출을 실제로 들여다보게 하므로 형편 차이가 값에 남는다.
    # **이 값은 측정 전용이다.** 아래 소비 총액 계산에 들어가지 않는다(순환 방지).
    # BOK도 자기보고 서베이로 같은 값을 얻었고 그 한계를 34항에 명시했다 — 도구와 한계가 같다.
    # MPC = Σ(정책지갑 결제분 중 신규 소비) / Σ(정책지갑 결제액).
    # 신규분은 건별 참/거짓이 아니라 **금액**(extra_spent)으로 받는다. 같은 결제 안에서도
    # '평소 쓰던 만큼'과 '이 돈이 있어 더 쓴 만큼'이 섞이는데, 참/거짓으로 받으면 후자가
    # 통째로 0으로 버려져 생필품·외식 비중이 높은 우리 구성에서 체계적으로 과소 측정된다.
    # extra_spent는 결제 전체 기준이므로 정책 결제 비중만큼 안분한다.
    _mpc_new: float | None = None
    if _choice_shares:
        _w_tot = 0.0; _w_new = 0.0
        for _i, _e in enumerate(commerce):
            _amt = max(0.0, float(_base_spends[_i]))
            _c = _amt * max(0.0, min(1.0, _choice_shares[_i]))
            if _c <= 0: continue
            _ex = _e.get("extra_spent")
            if _ex is None:
                _wba = _e.get("would_buy_anyway")
                if _wba is None: continue
                _ex = 0.0 if _wba else _amt
            _ex = max(0.0, min(_amt, float(_ex)))
            _w_tot += _c
            _w_new += _ex * (_c / _amt if _amt > 0 else 0.0)
        if _w_tot > 0: _mpc_new = _w_new / _w_tot

    # ─────────────────────────────────────────────────────────────────────────
    # [폐기 — 순환 구조] 예전에는 여기서 MPC(또는 grant_kept_share)를 받아
    #   additional_from_grant = substituted × MPC/(1−MPC)
    # 로 '추가 소비'를 만들어 총지출에 더했다. 그리고 그 결과에서 다시 MPC를 측정했다.
    # 넣은 값이 그대로 나오는 순환이므로 "MPC가 실측과 일치한다"는 것이 재현이 아니었다.
    # MPC는 **사후 측정량**이며 소비 생성기로 쓰지 않는다(BOK 참고3 ⑤도 측정 정의다).
    #
    # 쿠폰 때문에 소비가 늘어난다면 그것은 에이전트 자신의 결정으로 나타나야 한다:
    #   ⑴ Stage1의 오늘 소비의향(daily_propensity)이 올라가 personal_total이 커지거나
    #   ⑵ Stage2에서 이벤트·결제금액이 늘어나 planned_weights가 커지거나
    # 둘 다 이미 모델 안에 있는 경로다. 여기서 공식으로 얹지 않는다.
    # grant_kept_share·grant_extra_spend는 진단·인터뷰용으로만 보존한다.
    # ─────────────────────────────────────────────────────────────────────────
    _extra_rate = 0.0
    additional_from_grant = 0
    desired_total = personal_total
    desired_spends = distribute_budget(desired_total, planned_weights)
    # 지원금 전액이 아니라 오늘의 계획된 사용 가능 거래액까지만 유동성으로 인정한다.
    capacity = _allocate_policy_capacity(commerce, desired_spends, wallet_specs)
    eligible_policy_liquidity = int(capacity["total"])
    policy_parts = [sum(a.values()) for a in capacity["allocations"]]
    own_needs = [
        max(0, desired - policy)
        for desired, policy in zip(desired_spends, policy_parts)
    ]
    affordability_cap: int | None
    try:
        # 배송 주문은 지원금으로 결제할 수 없으므로 자기 돈에서 먼저 빠진다. 남은 잔액만
        # 오프라인 지출의 자기부담분에 쓸 수 있다.
        _bal = max(0, int(balance))
        online_spent = min(online_planned, _bal)
        own_balance = _bal - online_spent
        affordability_cap = own_balance + eligible_policy_liquidity
    except (TypeError, ValueError):
        own_balance = None
        online_spent = online_planned
        affordability_cap = None

    if own_balance is None:
        spends = desired_spends
    else:
        # 정책으로 결제 가능한 거래분을 먼저 보존하고 자기자금 필요분만 잔액에 맞춰
        # 줄인다. 전체를 비례 축소하면 불가 거래가 남아 실제 결제 가능액을 넘을 수 있다.
        own_budget = min(sum(own_needs), own_balance)
        own_spends = distribute_budget(own_budget, [float(v) for v in own_needs])
        spends = [
            policy + own
            for policy, own in zip(policy_parts, own_spends)
        ]
    total_adj = sum(spends)

    for e, sp in zip(commerce, spends):
        e["actual_spent"] = int(sp)

    # 소비 필요·POI·총액이 모두 확정된 뒤 결제수단만 지원금 우선으로 정산한다.
    allocation = settle_policy_spend_priority(
        events,
        grant_avail=grant_avail_alloc,
        restricted_envelopes=envelopes_alloc,
        grant_use=grant_use,
        choice_shares=_choice_shares if _choice_mode else None,
    )

    normal_budget = spend_today(p, daily)
    allocated_total = int(allocation["total"])
    payment_coverage = (
        allocated_total / int(allocation["eligible_spend_total"])
        if int(allocation["eligible_spend_total"]) > 0
        else 0.0
    )
    own_only_cap = min(desired_total, own_balance) if own_balance is not None else desired_total

    return {
        "applied": True,
        "propensity": p,
        "grant_spread_days": int(grant_spread_days) if grant_spread_days else None,
        "intended_grant_today": intended_grant_today,
        "grant_carry_in": _carry,
        # 오늘 계획한 만큼 쓸 거래가 없었으면 그 차액이 내일로 넘어간다.
        "grant_carry_out": 0 if _intensity_mode else max(0, intended_grant_today - int(allocated_total or 0)),
        "grant_intensity_mode": _intensity_mode,
        "grant_choice_mode": _choice_mode,
        # 오늘 계획한 지출 중 지원금으로 내기로 고른 몫(가중평균). 진단용.
        "grant_choice_share_mean": (
            round(sum(_choice_shares) / len(_choice_shares), 4) if _choice_shares else 0.0
        ),
        "grant_posture": _posture,
        # 오늘 적용된 사용 계획(일). 지갑이 남아 있으면 내일 State에 그대로 이어 실린다.
        "grant_plan_days_effective": (_eff_spread if wallet_total > 0 else 0),
        "eligible_base": eligible_base,
        "substituted": substituted,
        "grant_extra_rate": _extra_rate,
        # 참고3 ⑤ 형태로 산출한 그날의 신규 소비 유발 비중(= MPC). None이면 판단 누락.
        # 사후 측정값. 소비 생성에 쓰이지 않는다(순환 방지). 보고서의 MPC는 이 값을
        # 지원금 결제액으로 가중해 집계한 것이다.
        "mpc_new_share": (round(_mpc_new, 4) if _mpc_new is not None else None),
        "additional_from_grant": additional_from_grant,
        "personal_total": personal_total,
        "anchor_total": _anchor_total,
        "plan_over_anchor": max(0, personal_total - _anchor_total),
        "propensity_center": round(center, 4),
        "day_multiplier": round(day_multiplier, 4),
        "planned_total": planned_total,
        "today_total": total_adj,
        "grant_part": allocated_total,
        "normal_budget": normal_budget["total"],
        "available": normal_budget["available"],
        "affordability_cap": affordability_cap,
        "selected_policy_liquidity": eligible_policy_liquidity,
        "selected_policy_liquidity_by_pid": capacity["by_pid"],
        "policy_spend_allocated": allocation["by_pid"],
        "policy_spend_allocated_total": allocated_total,
        "policy_eligible_spend_total": allocation["eligible_spend_total"],
        "policy_eligible_event_count": allocation["eligible_event_count"],
        "policy_payment_coverage": round(payment_coverage, 4),
        "policy_liquidity_relief": max(0, total_adj - own_only_cap),
        "policy_wallet_available": grant_total + sum(int(e.get("amount") or 0) for e in envelopes),
        "policy_spend_requested": requested_by_pid,
        "mechanical_policy_uplift": 0,
        # 배송 주문(POI 방문 없음, 지원금 결제 불가). today_total은 가게 지출 합계이므로
        # 이 금액은 별도 항으로 잔액에서 차감된다(plan_writer 야간 정산).
        "online_share_effective": round(_online_rate, 4),
        "online_share_source": _online_src,
        "online_planned": online_planned,
        "online_total": int(online_spent),
        "today_total_incl_online": total_adj + int(online_spent),
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

    print("\n=== ④ apply_consumption_model: 사용 가능한 거래에 정책지갑 우선 결제 ===")
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
    assert gpt == tot, "무제한 정책지갑은 실제 거래액까지만 우선 결제해야 함"
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

    print("\n=== ⑥ 제한 예산 봉투: 총소비는 보존하고 사용 가능 거래만 우선 결제 ===")
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
    assert env6[0]["policy_spend"] == {"P010": env6[0]["actual_spent"]}
    assert m6["envelope_requested"]["P010"] == 8000
    assert m6["envelope_eligible_events"]["P010"] == 1

    zero_use = _mk6()
    zero_use[0]["policy_spend"] = {}
    m7 = apply_consumption_model(zero_use, **kw6, restricted_envelopes=[
        {"pid": "P010", "amount": 100000, "require_poi_eligible": True}])
    assert zero_use[0]["policy_spend"] == {"P010": zero_use[0]["actual_spent"]}
    assert m7["envelope_requested"]["P010"] == 0
    print("  ✔ 봉투가 총소비를 강제하지 않고 결제수단만 우선 배정")
