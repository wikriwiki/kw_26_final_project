"""업종 한정 할인권 — 8대 소비쿠폰류. **홀드아웃 기전.**

훈련 정책이 쓰는 다섯 기전(`wallet`·`cashback`·`hours_limit`·`gathering_limit`·
`price_discount`) 어디에도 없는 조합이라 진짜 일반화 시험이 된다.

  · 정해진 **업종에서만** 적용된다
  · 할인 방식이 업종마다 다르다 — 정액 / 정률 / **횟수 문턱 환급**
  · 상한과 **선착순 수량**이 있어 늘 받을 수 있는 것이 아니다
  · 외식은 **요일·시간창**까지 걸린다 (금 16시 ~ 일 24시)

시행방안·보도자료에서만 만들었다. **효과분석은 열지 않았다** — 봉인 규칙
(docs/POLICY_ANSWERKEY_MATRIX.md §0). 여기에 실측 결과를 반영하면 홀드아웃이
아니게 된다.

정책 JSON 파라미터
    sectors: {
      "외식":     {"mode": "count_rebate", "min_amount": 20000,
                   "count": 3, "rebate": 10000,
                   "window": "금 16:00 ~ 일 24:00"},
      "여행":     {"mode": "rate", "rate": 0.30},
      "숙박":     {"mode": "flat", "amount": 30000},
      "농수산물": {"mode": "rate", "rate": 0.20, "cap": 10000},
      "체육":     {"mode": "rebate", "amount": 30000}
    }
"""
from __future__ import annotations

PTYPE = "sector_voucher"


def _one(name: str, spec: dict) -> str:
    mode = (spec.get("mode") or "").strip()
    if mode == "count_rebate":
        lo = int(spec.get("min_amount") or 0)
        n = int(spec.get("count") or 0)
        rb = int(spec.get("rebate") or 0)
        s = f"{name} {lo:,}원 이상 {n}회 결제하면 다음 결제에서 {rb:,}원 환급"
        w = spec.get("window")
        return s + (f" ({w} 결제만 인정)" if w else "")
    if mode in ("rate", "rate_rebate"):
        r = float(spec.get("rate") or 0) * 100
        cap = int(spec.get("cap") or 0)
        # 할인(결제 시 깎임)과 환급(나중에 돌려받음)은 소비 시점이 다르다.
        # 정책 설명과 기전 줄의 표현이 어긋나면 에이전트가 다른 사실을 본다.
        verb = "환급" if mode == "rate_rebate" else "할인"
        s = f"{name} {r:.0f}% {verb}"
        return s + (f" (최대 {cap:,}원)" if cap else "")
    if mode == "flat":
        return f"{name} {int(spec.get('amount') or 0):,}원 할인"
    if mode == "rebate":
        return f"{name} 이용료 {int(spec.get('amount') or 0):,}원 환급"
    return name


def facts(row: dict) -> list[str]:
    """정책 공통 사실 — 업종별 조건을 한 줄씩."""
    sectors = row.get("sectors") or {}
    out = [_one(k, v) for k, v in sectors.items() if v]
    # 선착순 사실은 정책 description 이 이미 말한다. 여기서 또 적으면 같은
    # 사실이 네 번(배경·기전 줄·개인 상태·판단 원칙) 반복되어 "받을 수 없다"가
    # 지배적 신호가 된다 — 위약에서 대상 업종이 오히려 -10.5% 로 줄었다.
    return out


def status(pid: str, row: dict, persona: dict, state: dict,
           today=None) -> str:
    sectors = row.get("sectors") or {}
    names = ", ".join(sectors) if sectors else "없음"
    return f"- {pid}: 적용 업종 — {names}"
