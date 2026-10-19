"""업종별 예산 봉투와 필요 게이지 — EXP_CATEGORY_BUDGET=1 일 때만 쓰인다.

[왜 — 2026-10-19 L1 경계 지도]
정책이 없는 평소 14일(본 실행 5건, 9,855명)을 서울 실측과 비교했더니 하루 지출이 입력 기준액이
아니라 LLM 이 짠 일정의 결제 합으로 정해지고 있었다. 식사·마트·편의점(ESSENTIAL_L1)은 예산과
무관하게 전액 결제되므로, LLM 이 외식을 많이 잡으면 그대로 음식점 매출이 됐다.
  · 음식점 지출 비중 73% — 실측 24%, 페르소나 입력 28%
  · 주말 지출 비중 46% — 실측 24~27%, 페르소나 입력 20% (주말 하루가 평일의 1.7~2.2배)
  · 의료·의류·교육 각 1% 안팎 — 실측 21·10·6%
입력은 실측과 가까운데 출력이 어긋나므로 고칠 곳은 일정 생성과 결제 배분이다.
프롬프트에는 이미 '평소 업종별 지출 구성' 이 들어가 있었다(dawn_context._cat_line) — 알려 주는
방식은 효과가 없었으므로 구조로 고친다.

[무엇을]
1. 하루 가게 예산(엔진 기존 값: 입력 소비액 ÷ 2.40 × 동별 가게 몫 × 계획 배수)을 페르소나 입력
   업종 비율(평일·주말 따로)로 업종 봉투로 나눈다. 수준은 건드리지 않고 구성만 나눈다.
2. 봉투 안에서만 결제한다 — 식사·마트·편의점도. 월 납부(학원 등)·비정기 의료·정책 계기 일정은
   전액 내되 그 업종 봉투에서 뺀다(정책 반응 경로를 막지 않는다).
3. 업종마다 남은 돈은 이월로 쌓인다(필요 게이지). 게이지가 그 업종 1회 구매액을 넘었는데 오늘
   일정에 그 업종이 없으면 일정을 넣는다(업종별 시간 창, 하루 최대 MAX_INJECT 건).
4. 봉투로 감당할 수 없는 일상 외식(계기가 정책·약속·들은 이야기가 아닌 것)은 집(또는 근무 중이면
   직장)에서 먹는 것으로 바꾼다.

[보정 원칙] 입력(페르소나 카드 업종 비율·평일/주말 소비액)만 쓴다. 검증 자료(상권분석 추정매출·
생활인구)는 쓰지 않는다 — 쓰면 L1 검증이 순환이 된다. 1회 구매액·시간 창은 아래 상수(가정)로
두고 EXP_CATEGORY_TICKETS(JSON)로 바꿀 수 있다.

[같은 사람·두 갈래] 게이지의 첫 위상은 (사람, 업종) 해시로 정한다. 정책 전 주를 함께 돌리면
두 갈래가 같은 게이지에서 출발한다.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

try:
    from bdc_category_map import fold as _fold
except ImportError:  # pragma: no cover - package import path
    from .bdc_category_map import fold as _fold  # type: ignore

ENABLED = os.environ.get("EXP_CATEGORY_BUDGET", "0") == "1"

BUDGET_L1 = ("식사", "카페", "디저트", "편의점", "마트", "쇼핑", "미용", "건강", "교육", "여가", "주점", "기타")
INTERNAL = frozenset({"집", "직장"})

# 1회 구매액(원) — 가정. 게이지가 이만큼 차면 그 업종 일정이 생긴다.
_DEFAULT_TICKETS = {
    "식사": 9000, "카페": 5500, "디저트": 6000, "편의점": 6000, "마트": 25000, "쇼핑": 35000,
    "미용": 20000, "건강": 15000, "교육": 150000, "여가": 18000, "주점": 25000, "기타": 15000,
}


def _tickets() -> dict[str, int]:
    out = dict(_DEFAULT_TICKETS)
    raw = os.environ.get("EXP_CATEGORY_TICKETS")
    if raw:
        for k, v in json.loads(raw).items():
            out[str(k)] = int(v)
    return out


TICKETS = _tickets()
# 입력 상위 업종에 없는 업종의 최소 몫 — 입력이 상위 7개 업종만 주므로 나머지가 0 이 되지 않게 한다.
MIN_SHARE = float(os.environ.get("EXP_CATEGORY_MIN_SHARE", "0.02"))
MAX_INJECT = int(os.environ.get("EXP_CATEGORY_MAX_INJECT", "2"))
# 넣지 않는 업종: 식사는 LLM 의 끼니 계획을 따르고(봉투로 줄이기만), 기타는 갈 곳이 분명하지 않다.
NO_INJECT = frozenset({"식사", "기타"})

# 업종별 시간 창(분) — (평일, 주말). 가정이며 검증 자료에서 맞추지 않았다.
_H = lambda h, m=0: h * 60 + m  # noqa: E731
WINDOWS = {
    "카페": ((_H(10), _H(17)), (_H(11), _H(18))),
    "디저트": ((_H(14), _H(19)), (_H(13), _H(19))),
    "편의점": ((_H(19), _H(23)), (_H(12), _H(23))),
    "마트": ((_H(17, 30), _H(20, 30)), (_H(11), _H(19))),
    "쇼핑": ((_H(17, 30), _H(20, 30)), (_H(13), _H(19))),
    "미용": ((_H(13), _H(19)), (_H(11), _H(18))),
    "건강": ((_H(10), _H(17)), (_H(10), _H(13))),
    "교육": ((_H(15), _H(20)), (_H(10), _H(17))),
    "여가": ((_H(19), _H(22)), (_H(13), _H(21))),
    "주점": ((_H(19, 30), _H(23)), (_H(18, 30), _H(23))),
}
WORK_HOURS = (_H(9), _H(18))
AFTER_WORK = (_H(18, 30), _H(21))
DEFAULT_SUB = {"카페": "카페", "디저트": "베이커리", "편의점": "편의점", "마트": "슈퍼마켓", "쇼핑": "의류",
               "미용": "미용실", "건강": "의원", "교육": "학원", "여가": "스포츠", "주점": "일반주점"}
# 입력 업종 힌트(bdc_category_map) → 그래프 세부 업종 이름
HINT_SUB = {"의원": "의원", "병원": "의원", "치과": "치과", "약국": "약국", "한의원": "한의원", "헬스장": "헬스장",
            "의류": "의류", "화장품": "화장품", "가전": "가전·통신", "가전·가구": "가구", "가구": "가구",
            "생활용품": "생활용품", "노래방": "노래방", "스포츠": "스포츠", "PC방·오락": "PC방", "정육": "정육",
            "청과·수산": "청과", "욕탕": "목욕탕", "마사지": "마사지"}
PROTECTED_TRIGGERS = frozenset({"policy", "appointment", "rumor"})


def _u(*parts: Any) -> float:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:13]
    return int(h, 16) / float(16 ** 13)


def shares_for(persona: dict, is_weekend: bool) -> tuple[dict[str, float], dict[str, list[str]]] | None:
    """페르소나 입력 업종 비율(평일/주말)을 시뮬 업종으로 접는다. 입력이 없으면 None."""
    raw = persona.get("cat_ratio_we" if is_weekend else "cat_ratio_wd") or persona.get("cat_ratio_wd")
    if not raw:
        return None
    try:
        d = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return None
    folded = _fold(d)
    if not folded:
        return None
    shares = {l1: 0.0 for l1 in BUDGET_L1}
    hints: dict[str, list[str]] = {}
    for l1, share, hs in folded:
        key = l1 if l1 in shares else "기타"
        shares[key] += float(share)
        if hs:
            hints.setdefault(key, []).extend(hs)
    for k in shares:
        shares[k] = max(shares[k], MIN_SHARE)
    tot = sum(shares.values())
    return {k: v / tot for k, v in shares.items()}, hints


def envelopes(budget: float, shares: dict[str, float]) -> dict[str, int]:
    return {k: int(round(max(0.0, float(budget)) * s)) for k, s in shares.items()}


def carry_from_state(raw: Any, aid: str) -> dict[str, int]:
    """상태에 저장된 업종별 이월. 없으면 (사람, 업종) 해시 위상으로 0~1회 구매액 사이에서 시작한다."""
    stored: dict[str, int] = {}
    if raw:
        try:
            stored = {str(k): int(v) for k, v in (json.loads(raw) if isinstance(raw, str) else dict(raw)).items()}
        except (TypeError, ValueError):
            stored = {}
    return {k: stored.get(k, int(round(_u("phase", aid, k) * TICKETS[k]))) for k in BUDGET_L1}


def l1_of(category: str | None) -> str:
    c = category or ""
    return c if c in BUDGET_L1 else "기타"


def _minutes(t: str) -> int:
    return int(t[:2]) * 60 + int(t[3:5])


def _hhmm(m: int) -> str:
    m = max(0, min(23 * 60 + 45, int(m)))
    return f"{m // 60:02d}:{m % 60:02d}"


def _window(l1: str, is_weekend: bool, has_work: bool) -> tuple[int, int]:
    wd, we = WINDOWS[l1]
    lo, hi = we if is_weekend else wd
    if has_work and not is_weekend and lo < WORK_HOURS[1] and hi > WORK_HOURS[0]:
        return AFTER_WORK
    return lo, hi


def _pick_time(l1: str, aid: str, day: str, is_weekend: bool, has_work: bool, taken: list[int]) -> str:
    lo, hi = _window(l1, is_weekend, has_work)
    slots = list(range(lo, max(lo + 1, hi), 15))
    start = int(_u("time", aid, day, l1) * len(slots))
    for k in range(len(slots)):
        m = slots[(start + k) % len(slots)]
        if all(abs(m - t) >= 30 for t in taken):
            return _hhmm(m)
    return _hhmm(slots[start])


def plan_day(events: list, *, aid: str, day: str, is_weekend: bool, has_work: bool, budget: float,
             shares: dict[str, float], hints: dict[str, list[str]], carry: dict[str, int],
             make_event) -> tuple[list, dict]:
    """1단계 일정을 업종 봉투에 맞춘다(2단계 전에). make_event(**fields) 는 1단계 일정 객체를 만든다.

    반환: (일정 목록, 메타). 바뀐 일정에는 reasoning 끝에 표시를 붙인다.
    """
    env = envelopes(budget, shares)
    room = {k: env[k] + carry.get(k, 0) for k in BUDGET_L1}
    meta: dict[str, Any] = {"envelopes": env, "carry_in": dict(carry), "meals_converted": 0, "injected": []}

    # 끼니: 봉투로 감당할 수 있는 만큼만 밖에서 먹는다. 정책·약속·들은 이야기 계기는 그대로 둔다.
    meals = [e for e in events if getattr(e, "category", None) == "식사"]
    protected = [e for e in meals if (getattr(e, "trigger", None) or "") in PROTECTED_TRIGGERS]
    routine = [e for e in meals if e not in protected]
    afford = max(0, math.floor(room["식사"] / max(1, TICKETS["식사"]))) - len(protected)
    if len(routine) > max(0, afford):
        # 늦은 끼니부터 집으로 — 같은 날 안에서는 순서를 보존하는 쪽이 계획 변경이 적다.
        for e in sorted(routine, key=lambda x: _minutes(x.time))[max(0, afford):]:
            at_work = has_work and not is_weekend and WORK_HOURS[0] <= _minutes(e.time) < WORK_HOURS[1]
            e.category = "직장" if at_work else "집"
            e.sub_category = "식사"
            e.anchor = "workplace" if at_work else "residence"
            e.pinned_poi = None
            e.reasoning = ((e.reasoning or "").rstrip() + " [업종 예산: 이번 끼니는 "
                           + ("직장에서" if at_work else "집에서") + " 해결]").strip()
            meta["meals_converted"] += 1

    # 밀린 업종 일정 넣기
    planned = {l1_of(getattr(e, "category", None)) for e in events if getattr(e, "category", None) not in INTERNAL}
    due = [k for k in BUDGET_L1 if k not in NO_INJECT and k not in planned and room[k] >= TICKETS[k]]
    due.sort(key=lambda k: -room[k] / TICKETS[k])
    taken = [_minutes(e.time) for e in events]
    for k in due[:MAX_INJECT]:
        hint_subs = [HINT_SUB[h] for h in hints.get(k, []) if h in HINT_SUB]
        sub = hint_subs[0] if hint_subs else DEFAULT_SUB[k]
        t = _pick_time(k, aid, day, is_weekend, has_work, taken)
        taken.append(_minutes(t))
        ev = make_event(time=t, anchor="residence", category=k, sub_category=sub,
                        intent=f"{sub} 들르기", trigger="lifestyle",
                        reasoning=f"평소 지출 구성(카드 실측)상 {k} 지출이 밀려 있어 오늘 처리한다 [업종 예산: 필요 게이지]")
        events.append(ev)
        meta["injected"].append({"category": k, "sub": sub, "time": t})
    events.sort(key=lambda e: _minutes(e.time))
    return events, meta


def allocate(commerce: list[dict], planned: list[float], *, budget: int, shares: dict[str, float],
             carry: dict[str, int], aid: str | None, monthly_subs: frozenset, irregular_subs: frozenset,
             monthly_paid: set | None = None) -> tuple[list[int], dict]:
    """가격 우선 결제를 업종 봉투로 나눠 한다. consumption._price_first_spends 와 같은 꼴의 메타를 낸다."""
    amounts = [max(0, int(round(w))) for w in planned]
    env = envelopes(budget, shares)
    left = {k: env[k] + carry.get(k, 0) for k in BUDGET_L1}
    out = [0] * len(commerce)
    paid_set = set(monthly_paid or ())
    monthly = irregular = forced = skipped = 0
    by_cat = {k: 0 for k in BUDGET_L1}

    def sub_of(e):
        return e.get("sub_category") or e.get("sub") or ""

    # 1) 반드시 내는 것 — 월 납부(달마다 한 번), 비정기 의료·가전, 정책 계기 일정. 봉투에서는 뺀다.
    must = set()
    fixed_idx = set()          # 월 납부·비정기 — 기존 routine_total 정의에서 빠지는 것
    for i, e in enumerate(commerce):
        k = l1_of(e.get("category"))
        s = sub_of(e)
        if s in monthly_subs:
            must.add(i)
            fixed_idx.add(i)
            if s in paid_set:
                e["monthly_already_paid"] = True
                continue
            paid_set.add(s)
            out[i] = amounts[i]
            monthly += amounts[i]
        elif s in irregular_subs:
            must.add(i)
            fixed_idx.add(i)
            out[i] = amounts[i]
            irregular += amounts[i]
        elif (e.get("trigger") or "") == "policy":
            must.add(i)
            out[i] = amounts[i]
            forced += amounts[i]
        else:
            continue
        left[k] -= out[i]
        by_cat[k] += out[i]
    # 2) 나머지는 업종 봉투 안에서 시간 순으로
    for i, e in enumerate(commerce):
        if i in must:
            continue
        k = l1_of(e.get("category"))
        a = amounts[i]
        if a <= left[k]:
            out[i] = a
        elif left[k] > 0 and a > 0 and _u(aid or "", e.get("time") or "", e.get("poi_id") or "", i) < left[k] / a:
            out[i] = a
        else:
            out[i] = 0
            if a > 0:
                skipped += 1
                e["budget_skipped"] = True
                e["actual_spent_planned"] = a
            continue
        left[k] -= out[i]
        by_cat[k] += out[i]
    carry_out = {}
    for k in BUDGET_L1:
        hi = max(int(1.5 * TICKETS[k]), 3 * env[k])
        lo = -max(TICKETS[k], 3 * env[k])
        carry_out[k] = int(max(lo, min(hi, left[k])))
    routine_total = sum(o for i, o in enumerate(out) if i not in fixed_idx)
    meta = {"irregular_total": irregular, "monthly_total": monthly, "routine_budget": int(budget),
            "budget_skipped": skipped, "routine_total": int(routine_total), "policy_forced_total": forced,
            "carry_in": int(sum(carry.get(k, 0) for k in BUDGET_L1)), "carry_out": int(sum(carry_out.values())),
            "category": {"envelopes": env, "carry_in": {k: int(carry.get(k, 0)) for k in BUDGET_L1},
                         "carry_out": carry_out, "paid_by_cat": by_cat}}
    return out, meta
