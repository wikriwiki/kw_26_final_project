# -*- coding: utf-8 -*-
"""2021년 코로나 환경 — 서울.

`distancing_schedule.json`(방역 규제)과 `seoul_cases_daily.json`(확진 추이)에서
그날 에이전트가 알 수 있는 사실만 조립한다.

지키는 원칙
-----------
1. **미래를 보지 않는다.** 확진 집계는 공표 시차가 있어, 관측일이 시뮬레이션
   당일보다 앞선 자료만 쓴다. `date_semantics` 에 따라 관측일 = source_date + 1일.
2. **접종률은 쓰지 않는다.** 원자료가 `runtime_usable: False` (중복일자·정합성
   문제). 보정 없이 쓰면 없는 수치를 지어내는 셈이 된다. 필요해지면 정합화 후
   별도 커밋으로 추가한다.
3. **행동을 지시하지 않는다.** 규제와 유행 상태만 적고 소비를 늘리라/줄이라는
   문구는 넣지 않는다. 판단은 에이전트가 한다.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data" / "experiments" / "covid_support_2021"

# 2021년 추석 연휴
CHUSEOK = (date(2021, 9, 18), date(2021, 9, 22))


@lru_cache(maxsize=1)
def _regimes() -> list[dict]:
    d = json.loads((DATA / "distancing_schedule.json").read_text(encoding="utf-8"))
    return d.get("regimes") or []


@lru_cache(maxsize=1)
def _cases() -> dict[date, int]:
    """관측일 → 서울 신규 확진. 관측일 = source_date + 1일 (공표 시차)."""
    d = json.loads((DATA / "seoul_cases_daily.json").read_text(encoding="utf-8"))
    out: dict[date, int] = {}
    for r in d.get("records") or []:
        sd = r.get("source_date")
        n = r.get("total_cases")
        if not sd or n is None:
            continue
        y, m, dd = (int(x) for x in str(sd).split("-"))
        out[date(y, m, dd) + timedelta(days=1)] = int(n)
    return out


def _regime_for(day: date) -> dict | None:
    for r in _regimes():
        f = r.get("from"); u = r.get("until")
        if not f:
            continue
        fy, fm, fd = (int(x) for x in str(f).split("-"))
        start = date(fy, fm, fd)
        if u:
            uy, um, ud = (int(x) for x in str(u).split("-"))
            end = date(uy, um, ud)
        else:
            end = date(2021, 12, 31)
        if start <= day <= end:
            return r
    return None


def _meeting_line(pm: dict) -> str:
    if not pm:
        return ""
    before = pm.get("before_18_max")
    after = pm.get("from_18_max")
    base = []
    if before is not None and after is not None:
        base.append(f"사적모임 18시 이전 {before}인, 이후 {after}인")
    elif before is not None:
        base.append(f"사적모임 {before}인")
    tot = pm.get("vaccinated_exception_total") or pm.get(
        "restaurant_cafe_vaccinated_exception_total")
    venues = pm.get("vaccinated_exception_venues")
    if tot:
        where = "·".join(venues) if venues else "식당·카페"
        # 받침 유무로 조사 선택 — "가정은" / "카페는"
        josa = "은" if (ord(where[-1]) - 0xAC00) % 28 else "는"
        base.append(f"{where}{josa} 접종완료자 포함 시 최대 {tot}인")
    return ". ".join(base)


def build(day: date) -> dict:
    """그날의 사회 배경. 규제 구간을 못 찾으면 {} 를 돌려 섹션을 생략한다."""
    reg = _regime_for(day)
    if reg is None:
        return {}

    facts: list[str] = []

    # 확진 추이 — 당일보다 앞선 관측만
    cases = _cases()
    hist = [(d, n) for d, n in cases.items() if d < day]
    if hist:
        hist.sort()
        last_d, last_n = hist[-1]
        recent = [n for _, n in hist[-7:]]
        avg = round(sum(recent) / len(recent))
        facts.append(
            f"서울 신규 확진 {last_n:,}명 ({last_d.isoformat()} 기준, 최근 {len(recent)}일 평균 {avg:,}명)")

    cutoff = reg.get("dine_in_cutoff")
    if cutoff:
        after = reg.get("after_cutoff") or "매장 취식 불가"
        facts.append(f"식당·카페 매장 취식 {cutoff}까지. 이후 {after}")

    mline = _meeting_line(reg.get("private_meetings") or {})
    if mline:
        facts.append(mline)

    # 집합금지는 후속 구간이 다시 적지 않아도 해제된 것이 아니다. 원자료가 변경분만
    # 기술하므로, 명시가 없으면 직전 구간 값을 이어받는다. 이어받지 않으면 정책 주간
    # 한복판에서 규제가 사라진 것처럼 보이는 인공 변화가 생긴다.
    closed = reg.get("closed_facilities")
    if closed is None:
        for prev in reversed(_regimes()):
            if str(prev.get("from")) >= str(reg.get("from")):
                continue
            if prev.get("closed_facilities") is not None:
                closed = prev["closed_facilities"]
                break
    if closed:
        facts.append("집합금지: " + "·".join(str(c) for c in closed))

    level = reg.get("level")
    headline = f"수도권 사회적 거리두기 {level}단계" if level else "수도권 방역 조치 시행 중"
    fy, fm, fd = (int(x) for x in str(reg.get("from")).split("-"))
    if date(fy, fm, fd) == day:
        headline += " — 오늘부터 조치가 바뀐다"

    note = None
    if CHUSEOK[0] <= day <= CHUSEOK[1]:
        note = "추석 연휴. 가정 내 가족모임 예외가 적용되며 일반 식당 모임에는 확대되지 않는다"
    elif reg.get("note"):
        note = " ".join(str(reg["note"]).split())

    return {"headline": headline, "facts": facts, "note": note}
