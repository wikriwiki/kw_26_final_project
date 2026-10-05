"""결제 할인·환급 기전의 범용 회계.

정책 설명 문구만으로 할인효과를 대신하지 않는다. 매장 매출은 할인 전 금액이고,
시민의 자기부담은 할인 후 금액이다. 상품 자료가 없는 정책은 POI 적격 대리 지표의
한계를 평가 보고서에서 별도로 표시해야 한다.

[2026-10-06 일반화] 업종 하나짜리 정률 할인(P016)만 처리하던 것을 넓혔다.
  · 결제할 때 깎이는 것(정률 rate, 정액 flat) — 오늘 자기부담이 준다(by_event, by_pid).
  · 나중에 돌려받는 것(환급 rebate, 횟수를 채우면 환급 count_rebate, 정률 환급 rate_rebate)
    — 오늘 자기부담은 그대로다. 받을 돈으로 따로 기록한다(rebate_by_event, rebate_by_pid).
  · 업종마다 다른 조건·한도·시작일/종료일(sectors.<업종>.from/until), 요일·시각 창(window).
  · 할인 구매 상품권(price_discount, P014): 상품권으로 낸 결제는 액면가 대비 할인율만큼 덜 낸 것과 같다.
    월 구매한도(액면가) x 할인율이 그 달 할인 상한이다. 사는 자치구 안 가게에서만(require_same_district).
업종이 하나인 정률 할인은 사용량 열쇠가 정책 ID 그대로라 이전 런과 같은 값이 나온다.
"""
from __future__ import annotations

import json
from datetime import date

from eligibility import Rules

PAYMENT_MODES = frozenset({"rate", "flat"})
REBATE_MODES = frozenset({"rebate", "count_rebate", "rate_rebate"})


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _merged(row: dict) -> dict:
    raw = row.get("mech_params") or {}
    params = json.loads(raw) if isinstance(raw, str) else dict(raw)
    return {**params, **{k: v for k, v in row.items() if v is not None}}


def _sectors(merged: dict) -> dict:
    raw = merged.get("sectors") or {}
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def _minutes(hhmm: str) -> int:
    h, m = (int(x) for x in str(hhmm).split(":"))
    return h * 60 + m


def _period_suffix(per: str | None, today: date | None) -> str:
    if not per or today is None:
        return ""
    if per == "month":
        return f"@{today:%Y-%m}"
    if per == "week":
        y, w, _ = today.isocalendar()
        return f"@{y}-W{w:02d}"
    raise ValueError(f"알 수 없는 한도 기간: {per}")


def active_rate_discounts(policies: list[dict] | None, today: date | None = None,
                          gus: list[str] | None = None) -> list[dict]:
    """오늘 적용되는 할인·환급 규칙 목록. 정책 하나가 업종별로 여러 규칙을 낼 수 있다.

    gus: 이 사람이 상품권을 쓸 수 있는 자치구 코드(5자리) — 사는 구·직장 구. 할인 구매 상품권에만 쓴다.
    """
    out = []
    for row in policies or []:
        ptype = row.get("type")
        if ptype not in ("sector_voucher", "price_discount"):
            continue
        merged = _merged(row)
        pid = str(merged["id"])
        if ptype == "price_discount":
            rate = float(merged.get("discount_rate") or merged.get("benefit_rate") or 0)
            face_cap = int(merged.get("purchase_cap_monthly") or 0)
            eligibility = merged.get("eligibility")
            if isinstance(eligibility, str):
                eligibility = json.loads(eligibility)
            if not 0 < rate < 1 or face_cap <= 0:
                raise ValueError("할인 구매 상품권의 할인율·월 구매한도가 유효하지 않다")
            if not isinstance(eligibility, dict):
                raise ValueError("할인 구매 상품권에 명시적인 사용처 규칙이 필요하다")
            rates = merged.get("district_rates") or {}
            if isinstance(rates, str):
                rates = json.loads(rates)
            scope = merged.get("use_scope") or "home_district"
            if scope == "issuing_district":
                # 발행 구 안에서만 쓰고, 어느 구 상품권이든 산다 — 이 사람이 다니는 구(사는 구·직장 구)마다 규칙 하나.
                if gus is None:
                    raise ValueError("발행 구 상품권에는 이 사람의 자치구(사는 구·직장 구)가 필요하다")
                for gu in dict.fromkeys(g for g in gus if g):
                    gu_rate = float(rates.get(gu, rate))
                    out.append({"id": pid, "key": f"{pid}:{gu}" + _period_suffix("month", today),
                                "sector": None, "mode": "rate", "rate": gu_rate,
                                "cap": int(round(face_cap * gu_rate)), "rules": Rules(eligibility),
                                "require_same_district": False, "poi_gu": gu, "window": None})
                continue
            out.append({"id": pid, "key": pid + _period_suffix("month", today), "sector": None,
                        "mode": "rate", "rate": rate, "cap": int(round(face_cap * rate)),
                        "rules": Rules(eligibility),
                        "require_same_district": bool(eligibility.get("require_same_district")),
                        "window": None})
            continue
        sectors = _sectors(merged)
        if not sectors:
            raise ValueError("업종 할인권에 업종 규칙이 없다")
        policy_elig = merged.get("eligibility")
        if isinstance(policy_elig, str):
            policy_elig = json.loads(policy_elig)
        single = len(sectors) == 1
        for name, sector in sectors.items():
            sector = dict(sector or {})
            mode = sector.get("mode")
            if mode not in PAYMENT_MODES | REBATE_MODES:
                raise ValueError(f"지원되지 않는 할인 방식: {name} {mode}")
            if today is not None:
                if sector.get("from") and today < _as_date(sector["from"]):
                    continue
                if sector.get("until") and today > _as_date(sector["until"]):
                    continue
            eligibility = sector.get("eligibility") or policy_elig
            if not isinstance(eligibility, dict) or eligibility.get("mode") != "include":
                raise ValueError(f"업종 할인권 {name} 에 명시적인 사용처 포함 규칙이 필요하다")
            if eligibility.get("require_same_district"):
                raise ValueError("업종 할인권의 지역 제한은 정의되지 않았다")
            spec = {"id": pid, "sector": name, "mode": mode, "rules": Rules(eligibility),
                    "require_same_district": False, "window": sector.get("window")}
            if mode in ("rate", "rate_rebate"):
                spec["rate"] = float(sector.get("rate") or 0)
                spec["cap"] = int(sector.get("cap") or (merged.get("cap_per_agent") if single else 0)
                                  or (merged.get("cap") if single else 0) or 0)
                if not 0 < spec["rate"] < 1 or spec["cap"] <= 0:
                    raise ValueError(f"{name}: 할인율·누적 상한이 유효하지 않다")
            elif mode in ("flat", "rebate"):
                spec["amount"] = int(sector.get("amount") or 0)
                spec["min_amount"] = int(sector.get("min_amount") or 0)
                spec["tiers"] = list(sector.get("tiers") or [])
                spec["max_uses"] = int(sector.get("max_uses") or 0)
                spec["cumulative"] = bool(sector.get("cumulative"))   # 기간 누적 사용액이 문턱을 넘으면 한 번
                if spec["amount"] <= 0 and not spec["tiers"]:
                    raise ValueError(f"{name}: 정액이 없다")
            else:  # count_rebate
                spec["min_amount"] = int(sector.get("min_amount") or 0)
                spec["count"] = int(sector.get("count") or 0)
                spec["rebate"] = int(sector.get("rebate") or 0)
                spec["max_rebates"] = int(sector.get("max_rebates") or 0)
                spec["on_next"] = bool(sector.get("on_next"))          # count 회를 채운 다음 결제에서 환급
                spec["max_per_day"] = int(sector.get("max_per_day") or 0)
                spec["once_per_poi_per_day"] = bool(sector.get("once_per_poi_per_day"))
                if spec["count"] <= 0 or spec["rebate"] <= 0:
                    raise ValueError(f"{name}: 횟수·환급액이 유효하지 않다")
            # 사용량 열쇠: 업종 하나짜리 정률 할인은 정책 ID 그대로(이전 런과 같은 값).
            base = pid if (single and mode == "rate" and not sector.get("per")) else f"{pid}:{name}"
            spec["key"] = base + _period_suffix(sector.get("per"), today)
            out.append(spec)
    ids = {s["id"] for s in out}
    if len(ids) > 1:
        raise ValueError("동시 할인 정책의 중복 적용 규칙이 정의되지 않았다")
    return out


def _in_window(window, event: dict, weekday: int | None) -> bool:
    """window = {"from_weekday": 4, "from_time": "16:00", "to_weekday": 6, "to_time": "24:00"} (월=0).

    요일을 모르면(weekday None) 창이 있는 규칙은 적용하지 않는다 — 모르는 것을 인정으로 치지 않는다.
    """
    if not window:
        return True
    if weekday is None or not event.get("time"):
        return False
    t = _minutes(event["time"])
    start = window["from_weekday"] * 1440 + _minutes(window["from_time"])
    end = window["to_weekday"] * 1440 + _minutes(window["to_time"])
    now = weekday * 1440 + t
    return start <= now < end


def _flat_amount(spec: dict, amount: int) -> int:
    for tier in spec.get("tiers") or ():
        if tier.get("max") is None or amount <= int(tier["max"]):
            return int(tier["amount"])
    return spec["amount"]


def settle_instant_discounts(events: list[dict], amounts: list[int],
                             specs: list[dict], used_before: dict[str, int] | None = None,
                             weekday: int | None = None) -> dict:
    """이벤트 순서대로 할인·환급과 남은 한도를 계산한다. 입력은 변경하지 않는다."""
    if len(events) != len(amounts):
        raise ValueError("할인 회계의 거래·금액 행 수가 다르다")
    prior = {}
    for k, v in (used_before or {}).items():
        if isinstance(v, bool):
            continue
        try:
            prior[str(k)] = max(0, int(v))
        except (TypeError, ValueError):
            continue
    used = dict(prior)
    # 오늘 해당 거래가 없어도 활성 규칙의 열쇠는 닫는 잔액에 남긴다(run_simulation 이 저장한다).
    for spec in specs:
        used.setdefault(spec["key"], 0)
        if spec["mode"] in ("flat", "rebate", "count_rebate"):
            used.setdefault(spec["key"] + "#n", 0)
        if spec["mode"] == "rebate" and spec.get("cumulative"):
            used.setdefault(spec["key"] + "#sum", 0)
        if spec["mode"] == "count_rebate":
            used.setdefault(spec["key"] + "#paid", 0)
    by_event = [{} for _ in events]
    rebate_by_event = [{} for _ in events]
    eligible_gross = 0
    counted_today: dict[str, int] = {}        # 열쇠별 오늘 인정한 횟수(하루 상한)
    counted_poi: set[tuple[str, str]] = set()  # (열쇠, 가게) — 같은 업소 1일 1회
    for i, (event, raw_amount) in enumerate(zip(events, amounts)):
        amount = max(0, int(raw_amount or 0))
        if amount <= 0 or not event.get("poi_id"):
            continue
        counted = False
        for spec in specs:
            same = event.get("poi_same_gu")
            if spec.get("require_same_district") and same is not True:
                continue      # 자치구를 모르는 가게(후보풀 밖)는 인정하지 않는다
            if spec.get("poi_gu") and event.get("poi_gu") != spec["poi_gu"]:
                continue      # 발행 구 상품권은 그 구 안의 가게에서만
            # 고른 가게 자신의 업종으로 판정한다(없으면 계획 업종 — 후보풀 밖 POI).
            eligible, _reason = spec["rules"].eligible(
                event.get("poi_name"), event.get("poi_sub_category") or event.get("sub_category"),
                event.get("poi_category") or event.get("category"), event.get("upjong_l3"), same)
            if not eligible or not _in_window(spec.get("window"), event, weekday):
                continue
            if not counted:
                eligible_gross += amount
                counted = True
            pid, key, mode = spec["id"], spec["key"], spec["mode"]
            if mode in ("rate", "rate_rebate"):
                remaining = max(0, spec["cap"] - used.get(key, 0))
                value = min(remaining, int(round(amount * spec["rate"])))
                if mode == "rate":   # 한 거래의 결제 할인 합이 거래액을 넘지 않게
                    value = min(value, amount - sum(by_event[i].values()))
                if value > 0:
                    used[key] = used.get(key, 0) + value
                    target = by_event if mode == "rate" else rebate_by_event
                    target[i][pid] = target[i].get(pid, 0) + value
            elif mode == "rebate" and spec.get("cumulative"):
                # 기간 누적 사용액이 문턱을 넘는 결제에서 한 번 환급(사람당 max_uses, 기본 1)
                limit = spec["max_uses"] or 1
                if used.get(key + "#n", 0) >= limit:
                    continue
                used[key + "#sum"] = used.get(key + "#sum", 0) + amount
                if used[key + "#sum"] >= spec["min_amount"]:
                    used[key + "#n"] = used.get(key + "#n", 0) + 1
                    used[key] = used.get(key, 0) + spec["amount"]
                    rebate_by_event[i][pid] = rebate_by_event[i].get(pid, 0) + spec["amount"]
            elif mode in ("flat", "rebate"):
                if amount < spec["min_amount"]:
                    continue
                if spec["max_uses"] and used.get(key + "#n", 0) >= spec["max_uses"]:
                    continue
                value = min(amount, _flat_amount(spec, amount))
                if mode == "flat":
                    value = min(value, amount - sum(by_event[i].values()))
                if value > 0:
                    used[key + "#n"] = used.get(key + "#n", 0) + 1
                    used[key] = used.get(key, 0) + value
                    target = by_event if mode == "flat" else rebate_by_event
                    target[i][pid] = target[i].get(pid, 0) + value
            else:  # count_rebate — 조건을 채운 결제를 세고, count 번째에 환급한 뒤 다시 센다
                if amount < spec["min_amount"]:
                    continue
                if spec["max_rebates"] and used.get(key + "#paid", 0) >= spec["max_rebates"]:
                    continue
                if spec.get("max_per_day") and counted_today.get(key, 0) >= spec["max_per_day"]:
                    continue
                if spec.get("once_per_poi_per_day") and (key, event["poi_id"]) in counted_poi:
                    continue
                counted_today[key] = counted_today.get(key, 0) + 1
                counted_poi.add((key, event["poi_id"]))
                used[key + "#n"] = used.get(key + "#n", 0) + 1
                need = spec["count"] + (1 if spec.get("on_next") else 0)
                if used[key + "#n"] >= need:
                    used[key + "#n"] = 0
                    used[key + "#paid"] = used.get(key + "#paid", 0) + 1
                    used[key] = used.get(key, 0) + spec["rebate"]
                    rebate_by_event[i][pid] = rebate_by_event[i].get(pid, 0) + spec["rebate"]
    pids = sorted({s["id"] for s in specs})
    by_pid = {pid: sum(e.get(pid, 0) for e in by_event) for pid in pids}
    rebate_by_pid = {pid: sum(e.get(pid, 0) for e in rebate_by_event) for pid in pids}
    return {"by_event": by_event, "by_pid": by_pid,
            "total": sum(by_pid.values()), "used_after": used,
            "rebate_by_event": rebate_by_event, "rebate_by_pid": rebate_by_pid,
            "rebate_total": sum(rebate_by_pid.values()),
            "eligible_gross": eligible_gross,
            "eligible_gross_basis": ("whole_poi_transaction_proxy" if specs else None),
            "product_lines_observed": False}
