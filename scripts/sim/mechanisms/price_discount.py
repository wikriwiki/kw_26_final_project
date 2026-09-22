"""할인 구매 상품권 — 지역사랑상품권류.

**지갑을 만들지 않는다.** 새로 생기는 돈이 아니라 자기 돈을 싸게 바꾼 것이다.
그래서 총소비를 늘리지 않고 **쓸 수 있는 장소만 옮긴다** — 조세재정연구원
분석에서 소비지출 규모는 영향이 미미하고 구매처만 동네상권으로 이동했다.

이 구분이 이 기전의 전부다. 지갑형(`wallet`)처럼 다루면 "돈이 생겼으니 더 쓴다"
가 되어 실측과 반대 방향이 나온다.

정책 JSON 파라미터
    discount_rate         할인율 (0.10 = 10%)
    purchase_cap_monthly  1인 월 구매한도(액면가)
    use_scope             "home_district" — 거주 자치구 내에서만
    eligible_marker       후보 목록 표시 (예: "[지역]")
"""
from __future__ import annotations

PTYPE = "price_discount"


def status(pid: str, row: dict, persona: dict, state: dict,
           today=None) -> str:
    """개인별 한 줄. 잔액이 아니라 '살 수 있는 조건'을 적는다."""
    rate = row.get("discount_rate") or row.get("rate") or 0.0
    cap = int(row.get("purchase_cap_monthly") or row.get("cap") or 0)
    parts = [f"- {pid}: 액면가 {float(rate)*100:.0f}% 할인 구매"]
    if cap > 0:
        pay = int(round(cap * (1 - float(rate))))
        parts.append(f"이번 달 최대 {cap:,}원어치 ({pay:,}원 내면 {cap:,}원어치)")
    if (row.get("use_scope") or "") == "home_district":
        # 페르소나에 자치구가 없으면 이름을 대지 않는다 — 행정동을 자치구라고
        # 부르면 틀린 사실을 프롬프트에 넣게 된다(예: "무악동 자치구").
        gu = (persona.get("home_gu") or "").strip()
        parts.append(f"{gu} 안의 표시 가맹점에서만 사용" if gu
                     else "사는 자치구 안의 표시 가맹점에서만 사용")
    parts.append("새로 생긴 돈이 아니라 본인 돈을 싸게 바꾼 것")
    return " | ".join(parts)
