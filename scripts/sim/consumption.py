"""인간형 소비 — 소비성향(propensity) 모델 (Problem B, EconAgent 방식).

순수 함수 모듈. 소비를 고정 `daily_wd`가 아니라 **가용자산에 대한 소비성향 p∈[0,1]**로
결정한다 (EconAgent, ACL 2024). 지원금이 가용자산에 들어가므로 같은 p라도 지출↑.

핵심 설계:
  available_today = daily_wd / ANCHOR_PROPENSITY + grant_remaining   # 가용자산(일 환산 + 지원금)
  spend_today     = p × available_today
  grant_part      = p × grant_remaining                              # 지원금 비례 인출
  → p 가 소득 낮을수록 높음(유동성 제약, Jappelli & Pistaferri 2014) ⇒ MPC 역진 구조적 해소.

calibration 보존: 지원금 없는 평상일에 p = ANCHOR_PROPENSITY 이면 spend = daily_wd
  (BDC 분위로 보정된 일일 소비) → 시간·공간·지니 검증치 유지.
"""
from __future__ import annotations

ANCHOR_PROPENSITY = 0.70   # p0: 지원금 無·평상일에 이 값이면 지출 = daily_wd (BDC 앵커)

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
    """오늘 가용자산 = 일일 가처분(daily_wd/p0) + 지원금 잔여."""
    base = float(daily_wd or 0) / ANCHOR_PROPENSITY
    return base + max(0.0, float(grant_remaining or 0))


def spend_today(
    propensity: float,
    daily_wd: float | int | None,
    grant_remaining: float | int = 0,
) -> dict:
    """오늘 총지출과 지원금/자기부담 분해.

    반환: {total, grant_part, own_part, available, propensity}
      total      = p × 가용자산
      grant_part = p × 지원금잔여 (비례 인출) — 정책 사용액(MPC 분자)
      own_part   = total − grant_part
    """
    p = max(0.0, min(1.0, float(propensity)))
    gr = max(0.0, float(grant_remaining or 0))
    avail = available_today(daily_wd, gr)
    total = p * avail
    grant_part = min(p * gr, gr, total)
    own_part = max(0.0, total - grant_part)
    return {
        "total": int(round(total)),
        "grant_part": int(round(grant_part)),
        "own_part": int(round(own_part)),
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


# =========================================================
# 자체 테스트
# =========================================================
if __name__ == "__main__":
    print("=== ① calibration: 지원금 없는 평상일, p=p0 ⇒ spend ≈ daily_wd ===")
    for wd in (30000, 50000, 90000):
        r = spend_today(ANCHOR_PROPENSITY, wd, 0)
        print(f"  daily_wd={wd:,} → total={r['total']:,} (available={r['available']:,})")

    print("\n=== ② MPC 역진 해소: 소득별 지원금 소비 (prior 중심 propensity 사용) ===")
    # P009 지급액: 하 60만 / 중하 45만 / 중 25만 / 중상 10만
    cases = [
        ("하",   30000, 600000),
        ("중하", 45000, 450000),
        ("중",   60000, 250000),
        ("중상", 90000, 100000),
    ]
    print(f"  {'소득':<4} {'p_center':>8} {'grant':>9} {'grant_part':>11} {'MPC(1일)':>9}")
    for inc, wd, grant in cases:
        p = clamp_propensity(None, inc, balance=500000, daily_wd=wd)  # None → 중심값
        r = spend_today(p, wd, grant)
        mpc = r["grant_part"] / grant if grant else 0
        print(f"  {inc:<4} {p:>8.3f} {grant:>9,} {r['grant_part']:>11,} {mpc:>8.1%}")

    print("\n=== ③ 분배: 오늘 총지출 47,000원을 4개 이벤트(가중치)로 ===")
    print("  ", distribute_budget(47000, [3, 1, 2, 1]), "합", sum(distribute_budget(47000, [3, 1, 2, 1])))
