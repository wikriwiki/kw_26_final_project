"""할인 구매 상품권 — 지역사랑상품권류.

지갑을 만들지 않는다. 상품권으로 낸 결제는 액면가 대비 할인율만큼 덜 낸 것과 같아서,
회계는 결제 즉시 할인(instant_discount, 월 구매한도 x 할인율이 그 달 상한)으로 한다.

[2026-10-06] 프롬프트에서 "새로 생긴 돈이 아니다"를 뺐다. 소비를 줄이는 쪽으로 미는 문장이고
(지원금·캐시백·업종할인권에서 뺀 것과 같은 종류), 실측 결과를 행동으로 미리 정해 주는 문장이었다.
할인율·한도·남은 한도·쓸 수 있는 곳 같은 사실만 적는다.

정책 JSON 파라미터
    discount_rate         할인율 (0.10 = 10%)
    purchase_cap_monthly  1인 월 구매한도(액면가)
    use_scope             "home_district" — 거주 자치구 내에서만
    eligible_marker       후보 목록 표시 (예: "[지역]")
"""
from __future__ import annotations

PTYPE = "price_discount"


# 서울 자치구 코드(5자리) → 이름 (scripts/prep/merchant_stats.py 와 같은 표)
GU_NAME = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
    "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
    "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
    "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
    "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
    "11740": "강동구",
}


def _issuing_status(pid: str, row: dict, persona: dict, state: dict, today) -> str:
    """발행 구 상품권(P014 2020): 사는 구·직장 구 상품권의 할인율과 그 구의 이번 달 남은 구매 한도.

    할인율은 구마다 다르고(district_rates), 한도는 구별 월 구매한도다. 모두 원문의 사실이다.
    """
    import json as _json
    rate = float(row.get("discount_rate") or 0.0)
    rates = row.get("district_rates") or {}
    if isinstance(rates, str):
        rates = _json.loads(rates)
    cap = int(row.get("purchase_cap_monthly") or 0)
    raw = (state or {}).get("policy_used") or {}
    try:
        used = _json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        used = {}
    items = []
    for label, code in (("사는 구", str(persona.get("home_dong_code") or "")[:5]),
                        ("직장 구", str(persona.get("work_dong_code") or "")[:5])):
        if not code or code not in GU_NAME or any(code == c for _, c, _r in items):
            continue
        items.append((label, code, float(rates.get(code, rate))))
    parts = [f"- {pid}: 자치구 상품권 할인 구매(구마다 할인율이 다르다)"]
    for label, code, r in items:
        text = f"{label} {GU_NAME[code]} 상품권 {r*100:.0f}% 할인"
        if cap > 0 and today is not None and r > 0:
            got = 0
            try:
                got = max(0, int(used.get(f"{pid}:{code}@{today:%Y-%m}", 0) or 0))
            except (TypeError, ValueError):
                got = 0
            left = max(0, cap - int(round(got / r)))
            text += f", 이번 달 남은 구매 한도 {left:,}원어치"
        parts.append(text)
    if cap > 0:
        parts.append(f"구별로 한 달 {cap:,}원어치까지 구매")
    parts.append("상품권은 발행한 구 안의 표시 가맹점에서만 사용")
    return " | ".join(parts)


def status(pid: str, row: dict, persona: dict, state: dict,
           today=None) -> str:
    """개인별 한 줄. 잔액이 아니라 '살 수 있는 조건'을 적는다."""
    if (row.get("use_scope") or "") == "issuing_district":
        return _issuing_status(pid, row, persona, state, today)
    rate = row.get("discount_rate") or row.get("rate") or 0.0
    cap = int(row.get("purchase_cap_monthly") or row.get("cap") or 0)
    parts = [f"- {pid}: 액면가 {float(rate)*100:.0f}% 할인 구매"]
    if cap > 0:
        pay = int(round(cap * (1 - float(rate))))
        parts.append(f"이번 달 최대 {cap:,}원어치 ({pay:,}원 내면 {cap:,}원어치)")
    if cap > 0 and today is not None:
        # 이번 달 이미 할인받은 금액(사용량 열쇠 <정책>@<연-월>)에서 남은 구매 한도를 계산한다 — 사실이다.
        import json as _json
        raw = (state or {}).get("policy_used") or {}
        try:
            used = _json.loads(raw) if isinstance(raw, str) else dict(raw)
            got = max(0, int(used.get(f"{pid}@{today:%Y-%m}", 0) or 0))
        except (TypeError, ValueError):
            got = 0
        if float(rate) > 0:
            left = max(0, cap - int(round(got / float(rate))))
            parts.append(f"이번 달 남은 구매 한도 {left:,}원어치")
    if (row.get("use_scope") or "") == "home_district":
        # 페르소나에 자치구가 없으면 이름을 대지 않는다 — 행정동을 자치구라고
        # 부르면 틀린 사실을 프롬프트에 넣게 된다(예: "무악동 자치구").
        gu = (persona.get("home_gu") or "").strip()
        parts.append(f"{gu} 안의 표시 가맹점에서만 사용" if gu
                     else "사는 자치구 안의 표시 가맹점에서만 사용")
    return " | ".join(parts)
