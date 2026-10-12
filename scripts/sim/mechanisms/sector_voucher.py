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

import os
import json

PTYPE = "sector_voucher"


_WEEKDAYS = "월화수목금토일"


def _window_text(w) -> str:
    if not w:
        return ""
    if isinstance(w, str):
        return w
    return (f"{_WEEKDAYS[int(w['from_weekday'])]} {w['from_time']} ~ "
            f"{_WEEKDAYS[int(w['to_weekday'])]} {w['to_time']}")


def _dates_text(spec: dict) -> str:
    frm, until = spec.get("from"), spec.get("until")
    if frm and until:
        return f" [{str(frm)[5:].replace('-', '/')}~{str(until)[5:].replace('-', '/')}]"
    if frm:
        return f" [{str(frm)[5:].replace('-', '/')}부터]"
    if until:
        return f" [{str(until)[5:].replace('-', '/')}까지]"
    return ""


def _one(name: str, spec: dict) -> str:
    return _one_body(name, spec) + _dates_text(spec)


def _one_body(name: str, spec: dict) -> str:
    mode = (spec.get("mode") or "").strip()
    if mode == "count_rebate":
        lo = int(spec.get("min_amount") or 0)
        n = int(spec.get("count") or 0)
        rb = int(spec.get("rebate") or 0)
        if spec.get("on_next"):
            s = f"{name} {lo:,}원 이상 {n}회 결제하면 {n + 1}번째 결제에서 {rb:,}원 환급(나중에 돌려받음)"
        else:
            s = f"{name} {lo:,}원 이상 결제를 {n}회 채우면 {rb:,}원 환급(나중에 돌려받음)"
        w = _window_text(spec.get("window"))
        return s + (f" ({w} 결제만 인정)" if w else "")
    if mode in ("rate", "rate_rebate"):
        r = float(spec.get("rate") or 0) * 100
        cap = int(spec.get("cap") or 0)
        # 할인(결제 시 깎임)과 환급(나중에 돌려받음)은 소비 시점이 다르다.
        # 정책 설명과 기전 줄의 표현이 어긋나면 에이전트가 다른 사실을 본다.
        verb = "환급" if mode == "rate_rebate" else "할인"
        s = f"{name} {r:.0f}% {verb}"
        # [2026-10-11] 농축산물 몫만 깎이고 한도가 유통업체마다 따로인 경우(P016 농할) — 사실 그대로 적는다.
        if spec.get("base") == "produce_share":
            s = f"{name}(국산 신선 농축산물) 값의 {r:.0f}%를 결제할 때 {verb}(그 밖의 물건은 할인 없음)"
        chains = list((spec.get("chains") or {}).keys())
        if chains and cap:
            return s + f" — 참여 유통업체({'·'.join(chains)})마다 1인 최대 {cap:,}원"
        return s + (f" (최대 {cap:,}원)" if cap else "")
    if mode == "flat":
        tiers = spec.get("tiers") or []
        if tiers:
            parts = []
            for t in tiers:
                if t.get("max") is not None:
                    parts.append(f"{int(t['max']):,}원 이하 {int(t['amount']):,}원")
                else:
                    prev = [x for x in tiers if x.get("max") is not None]
                    over = f"{int(prev[-1]['max']):,}원 초과 " if prev else ""
                    parts.append(f"{over}{int(t['amount']):,}원")
            return f"{name} 결제 1건 할인: " + ", ".join(parts)
        return f"{name} {int(spec.get('amount') or 0):,}원 할인"
    if mode == "rebate":
        lo = int(spec.get("min_amount") or 0)
        if spec.get("cumulative"):
            cond = f"기간 중 합계 {lo:,}원 이상 쓰면 " if lo else ""
        else:
            cond = f"{lo:,}원 이상 결제하면 " if lo else ""
        return f"{name} {cond}{int(spec.get('amount') or 0):,}원 환급(나중에 돌려받음)"
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
    """개인별 한 줄 — 업종 이름만이 아니라 **본인에게 걸리는 한도**까지 적는다.

    업종 이름만 적으면 1인 누적 한도가 있는 정책에서 모델이 제약을 못 본다.
    한도는 정책 JSON 의 sectors 에서 읽는다 — 특정 정책을 하드코딩하지 않는다.
    """
    sectors = row.get("sectors") or {}
    if not sectors:
        return f"- {pid}: 적용 업종 — 없음"
    raw_used = (state or {}).get("policy_used") or {}
    try:
        used = json.loads(raw_used) if isinstance(raw_used, str) else dict(raw_used)
        used_amount = max(0, int(used.get(pid, 0) or 0))
    except (ValueError, TypeError):
        used_amount = 0
    parts = []
    for name, spec in sectors.items():
        if (spec or {}).get("mode") == "count_rebate":
            # 지금까지 인정된 결제 횟수(사용량 열쇠 <정책>:<업종>[@기간]#n). 기간 열쇠는 오늘로 맞춘다.
            n_now = 0
            try:
                from instant_discount import _period_suffix
                key = f"{pid}:{name}" + _period_suffix((spec or {}).get("per"), today) + "#n"
                used_all = json.loads(raw_used) if isinstance(raw_used, str) else dict(raw_used)
                n_now = max(0, int(used_all.get(key, 0) or 0))
            except (ValueError, TypeError):
                n_now = 0
            parts.append(f"{name}(지금까지 인정된 결제 {n_now}회)")
            continue
        cap = int((spec or {}).get("cap") or 0)
        _chains = list(((spec or {}).get("chains") or {}).keys())
        if cap > 0 and (spec or {}).get("mode") == "rate" and _chains:
            # [2026-10-11] 체인마다 따로 센 한도(사용량 열쇠 <정책>:<체인>). **받은 할인만** 적는다.
            # 처음에는 '유통업체별 남은 할인 — 이마트 10,000원, 롯데마트 10,000원, …' 을 적었는데, 결제할 때 저절로 깎이는
            # 할인을 쓰지 않으면 사라지는 잔액(4곳 합 4만원)처럼 보이게 한다(10명 시험 t10b: 정책 있는 쪽 장보기 12건, 없는 쪽 0건,
            # 계기 대부분 '[policy]'). 사람은 영수증으로 받은 할인을 알 뿐 남은 잔액을 들고 다니지 않는다. 한도 규칙은 그대로 적는다.
            try:
                used_all = json.loads(raw_used) if isinstance(raw_used, str) else dict(raw_used)
            except (ValueError, TypeError):
                used_all = {}
            got = [f"{c} {int(used_all.get(f'{pid}:{c}', 0) or 0):,}원" for c in _chains if int(used_all.get(f'{pid}:{c}', 0) or 0) > 0]
            parts.append(f"{name}(지금까지 받은 할인 — {', '.join(got) if got else '없음'} · 유통업체마다 1인 최대 {cap:,}원)")
            continue
        if cap > 0 and (spec or {}).get("mode") == "rate" and len(sectors) == 1:
            parts.append(f"{name}(1인 할인 한도 {cap:,}원, 남은 할인 "
                         f"{max(0, cap - used_amount):,}원)")
        else:
            parts.append(f"{name}(1인 누적 {cap:,}원 한도)" if cap > 0 else name)
    line = f"- {pid}: 적용 업종 — {', '.join(parts)}"
    # [EXP_SCOPE_FACT] 범위의 산술 한 줄. **행동 방향이 아니라 계산이다.**
    #
    # 혜택은 위 업종에서만 붙는다. 그러므로 대상이 아닌 업종에서 줄여도 받는
    # 혜택은 한 푼도 늘지 않는다 — 제도의 정의에서 바로 나오는 사실이다.
    # 모델은 이것을 스스로 세우지 못하는 것으로 보인다: 위약에서 대상이 아닌
    # 업종이 함께 빠졌고(PL-2 -4.4%), 캐시백에서도 제외업종이 -13.9% 로
    # 동등성 밴드를 251% 넘었다. 하루 총액이 한 스칼라에 묶여 있어 한 업종이
    # 오르면 다른 업종이 내려가는 자리다.
    #
    # 무엇을 하라고 말하지 않는다. 대상 아닌 업종을 늘리라는 뜻이 아니다.
    if os.environ.get("EXP_SCOPE_FACT", "0") == "1":
        line += (" | 혜택은 위 업종에서만 붙는다 — 그 밖의 업종에서 줄여도 "
                 "받는 혜택이 늘지 않고, 거기서 쓴 돈이 혜택을 깎지도 않는다")
    return line
