"""범용 기전 — **등록되지 않은 어떤 정책이 들어와도** 받아 준다.

## 왜 필요한가

기전마다 모듈을 만드는 구조는 새 정책이 올 때마다 코드를 요구한다. 그래서 두
자리에서 조용히 틀린 말이 나갔다.

    label()      등록 안 된 기전 → "[interest_subsidy] 소상공인 이자지원"
                 한글 문장 한가운데에 영문 식별자가 박힌다

    principle()  등록 안 된 기전 → **지갑 원칙**이 떨어졌다
                 "정책 사용처에서 정책지갑으로 낼지 늘 쓰던 카드로 낼지는…"
                 지갑이 없는 정책에게 **있지도 않은 지갑을 사실처럼** 말한다

둘째가 특히 나쁘다. 라벨은 어색할 뿐이지만 이쪽은 **거짓을 사실로 주입**한다.

## 무엇을 하고 무엇을 하지 않는가

이 모듈은 **정책 JSON 에 선언된 것만** 읽어서 옮긴다. 추측하지 않는다.

    한다    1인 한도 · 사용처 표시 · 적용 기간처럼 **뜻이 하나뿐인 값**
    안 한다  비율의 의미 추정. `rate: 0.2` 가 깎아 주는 것인지 돌려주는 것인지는
            필드만 봐서 알 수 없다 — 그것은 정책 본문(description)이 이미 말한다

제도의 규칙은 `description` 이 한국어 산문으로 싣고 있다. 여기서 또 해석하면
같은 사실이 두 번 들어가고, 잘못 해석하면 두 번째가 틀린 채로 들어간다.

## 판단 원칙

기전을 모르면 **기전에 대해 아무 말도 하지 않는다.** 공통 문장만 두고,
조건은 정책 블록을 보라고만 한다. 방향은 어느 쪽으로도 말하지 않는다.
"""
from __future__ import annotations

PTYPE = "__generic__"

# 어떤 정책에도 참인 말만. 기전을 특정하지 않고, 무엇을 하라고도 하지 않는다.
PRINCIPLE = (
    "소비 필요·시점·총액·POI는 본인의 평소 습관, 자산, 일정에 따라 판단한다. "
    "정책의 조건은 정책 블록에 적힌 그대로이며, 그 조건이 오늘 본인 선택에 "
    "닿는지는 본인이 정한다."
)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def facts(row: dict) -> list[str]:
    """뜻이 하나뿐인 선언값만 한 줄씩. 비율의 의미는 추정하지 않는다."""
    out: list[str] = []
    cap = row.get("cap_per_agent")
    if _num(cap):
        out.append(f"1인 누적 한도 {int(cap):,}원")
    monthly = row.get("purchase_cap_monthly")
    if _num(monthly):
        out.append(f"1인 월 한도 {int(monthly):,}원")
    # 업종별 조건이 선언돼 있으면 이름만 옮긴다. 조건 문구는 본문이 말한다.
    sectors = row.get("sectors")
    if isinstance(sectors, dict) and sectors:
        out.append("대상 업종 — " + ", ".join(str(k) for k in sectors))
    return out


def status(pid: str, row: dict, persona: dict, state: dict, today=None) -> str:
    """개인별 한 줄 — 본인에게 걸리는 제약만 적는다.

    지갑이 없는 정책에 잔액을 적으면 없는 돈을 만들어 준다. 그래서 잔액은
    쓰지 않고, 정책이 선언한 **한도와 사용처 조건**만 옮긴다.
    """
    parts = [f"- {pid}:"]
    cap = row.get("cap_per_agent") or row.get("purchase_cap_monthly")
    if _num(cap):
        parts.append(f"1인 한도 {int(cap):,}원")
    marker = (row.get("eligible_marker") or "").strip()
    if row.get("poi_restricted") and marker:
        parts.append(f"{marker} 표시 매장에서만 적용")
    elif row.get("poi_restricted"):
        parts.append("정해진 표시가 붙은 매장에서만 적용")
    if (row.get("use_scope") or "") == "home_district":
        gu = (persona.get("home_gu") or "").strip()
        parts.append(f"{gu} 안에서만" if gu else "사는 자치구 안에서만")
    if len(parts) == 1:
        parts.append("조건은 정책 블록 참조")
    return " | ".join([parts[0] + " " + parts[1]] + parts[2:]) if len(parts) > 1 else parts[0]
