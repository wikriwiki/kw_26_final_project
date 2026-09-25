"""결제 즉시 할인 기전의 범용 회계.

정책 설명 문구만으로 할인효과를 대신하지 않는다. 매장 매출은 할인 전 금액이고,
시민의 자기부담은 할인 후 금액이다. 상품 자료가 없는 정책은 POI 적격 대리 지표의
한계를 평가 보고서에서 별도로 표시해야 한다.
"""
from __future__ import annotations

import json

from eligibility import Rules


def active_rate_discounts(policies: list[dict] | None) -> list[dict]:
    """정책 JSON에서 즉시 정률 할인·1인 누적 상한을 읽는다."""
    out = []
    for row in policies or []:
        if row.get("type") != "sector_voucher":
            continue
        raw = row.get("mech_params") or {}
        params = json.loads(raw) if isinstance(raw, str) else dict(raw)
        merged = {**params, **{k: v for k, v in row.items() if v is not None}}
        sectors = merged.get("sectors") or {}
        if len(sectors) != 1:
            raise ValueError("즉시 할인 회계는 업종별 규칙이 하나인 정책만 지원한다")
        sector = next(iter(sectors.values()))
        if sector.get("mode") != "rate":
            raise ValueError("즉시 할인 회계에 지원되지 않는 할인 방식")
        rate = float(sector.get("rate") or 0)
        cap = int(sector.get("cap") or merged.get("cap_per_agent")
                  or merged.get("cap") or 0)
        if not 0 < rate < 1 or cap <= 0:
            raise ValueError("즉시 할인율·누적 상한이 유효하지 않다")
        eligibility = merged.get("eligibility")
        if not isinstance(eligibility, dict) or eligibility.get("mode") != "include":
            raise ValueError("즉시 할인 정책에 명시적인 사용처 포함 규칙이 필요하다")
        if eligibility.get("require_same_district"):
            raise ValueError("즉시 할인 지역 제한을 확인할 거래별 장소 정보가 없다")
        out.append({"id": str(merged["id"]), "rate": rate,
                    "cap": cap, "rules": Rules(eligibility)})
    if len(out) > 1:
        raise ValueError("동시 즉시 할인 정책의 중복 적용 규칙이 정의되지 않았다")
    return out


def settle_instant_discounts(events: list[dict], amounts: list[int],
                             specs: list[dict], used_before: dict[str, int] | None = None) -> dict:
    """이벤트 순서대로 할인액과 남은 한도를 계산한다. 입력은 변경하지 않는다."""
    if len(events) != len(amounts):
        raise ValueError("할인 회계의 거래·금액 행 수가 다르다")
    prior = {str(k): max(0, int(v)) for k, v in (used_before or {}).items()}
    used = dict(prior)
    by_event = [{} for _ in events]
    eligible_gross = 0
    for i, (event, raw_amount) in enumerate(zip(events, amounts)):
        amount = max(0, int(raw_amount or 0))
        if amount <= 0 or not event.get("poi_id"):
            continue
        for spec in specs:
            eligible, _reason = spec["rules"].eligible(
                event.get("poi_name"), event.get("sub_category"),
                event.get("category"), event.get("upjong_l3"))
            if not eligible:
                continue
            pid = spec["id"]
            eligible_gross += amount
            remaining = max(0, spec["cap"] - used.get(pid, 0))
            discount = min(remaining, int(round(amount * spec["rate"])))
            if discount > 0:
                by_event[i][pid] = discount
                used[pid] = used.get(pid, 0) + discount
    by_pid = {spec["id"]: used.get(spec["id"], 0) - prior.get(spec["id"], 0)
              for spec in specs}
    return {"by_event": by_event, "by_pid": by_pid,
            "total": sum(by_pid.values()), "used_after": used,
            "eligible_gross": eligible_gross,
            "eligible_gross_basis": ("whole_poi_transaction_proxy" if specs else None),
            "product_lines_observed": False}
