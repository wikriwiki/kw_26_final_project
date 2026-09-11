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
    """사적모임 인원 제한 한 줄.

    원자료의 키 구조가 시기에 따라 다르다. 10/17 이전은 시간대로 인원을 가르고
    (before_18_max / from_18_max), 10/18 이후는 시간 구분이 사라져 총원으로 적는다
    (total_max / including_vaccinated_total). 한쪽만 처리하면 나머지 기간에
    사적모임 줄이 조용히 빠진다.
    """
    if not pm:
        return ""
    base: list[str] = []
    before, after = pm.get("before_18_max"), pm.get("from_18_max")
    total = pm.get("total_max") or pm.get("including_vaccinated_total")
    if before is not None and after is not None:
        base.append(f"사적모임 18시 이전 {before}인, 이후 {after}인")
    elif total is not None:
        base.append(f"사적모임 최대 {total}인")
        unvac = pm.get("unvaccinated_max") or pm.get("restaurant_cafe_unvaccinated_max")
        if unvac is not None:
            base.append(f"미접종자는 {unvac}인까지")
    elif before is not None:
        base.append(f"사적모임 {before}인")

    tot = pm.get("vaccinated_exception_total") or pm.get(
        "restaurant_cafe_vaccinated_exception_total")
    venues = pm.get("vaccinated_exception_venues")
    if tot:
        where = "·".join(venues) if venues else "식당·카페"
        josa = "은" if (ord(where[-1]) - 0xAC00) % 28 else "는"
        base.append(f"{where}{josa} 접종완료자 포함 시 최대 {tot}인")

    solo = pm.get("restaurant_cafe_unvaccinated_without_pass_exception")
    if solo:
        base.append(f"식당·카페 미접종자는 {solo}")
    return ". ".join(base)


def _effective(reg: dict, key: str):
    """값이 없으면 직전 구간에서 이어받은 실효값. 원자료가 변경분만 적기 때문."""
    if reg.get(key) is not None:
        return reg[key]
    for prev in reversed(_regimes()):
        if str(prev.get("from")) >= str(reg.get("from")):
            continue
        if prev.get(key) is not None:
            return prev[key]
    return None


def _materially_changed(reg: dict) -> bool:
    """직전 구간과 견줘 시민이 체감할 내용이 실제로 달라졌는가.

    원자료는 구간을 나눠놓고 변경분만 기술한다. 그래서 항목이 비어 있는 것은
    "해제"가 아니라 "그대로"다. 이어받은 실효값끼리 비교해야 없던 변화를
    프롬프트에 만들어 넣지 않는다.
    """
    keys = ("level", "dine_in_cutoff", "private_meetings", "closed_facilities",
            "other_22h_restricted_examples", "midnight_closing_examples",
            "vaccine_pass_examples", "vaccine_pass_expansion_examples")
    prev = None
    for r in _regimes():
        if str(r.get("from")) >= str(reg.get("from")):
            break
        prev = r
    if prev is None:
        return True
    return any(_effective(reg, k) != _effective(prev, k) for k in keys)


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

    # 값이 비었을 때 "그대로"인지 "해제"인지는 원자료가 dine_in_cutoff_note 로 구분한다.
    # 위드코로나(11/1)처럼 해제된 구간에서 이전 22:00 을 이어받으면 없는 규제를 말하게 된다.
    if reg.get("dine_in_cutoff") is None and reg.get("dine_in_cutoff_note"):
        cutoff = None
    else:
        cutoff = _effective(reg, "dine_in_cutoff")
    if cutoff:
        after = _effective(reg, "after_cutoff") or "매장 취식 불가"
        facts.append(f"식당·카페 매장 취식 {cutoff}까지. 이후 {after}")
        # 같은 시간제한이 걸린 다른 시설. 학원·영화관·PC방은 소비처라 행동에 직접 닿는다.
        others = _effective(reg, "other_22h_restricted_examples")
        if others:
            facts.append(f"{cutoff}까지 운영 제한: " + "·".join(str(x) for x in others))
    else:
        # 시간제한이 풀린 구간. 원자료가 "모든 매장이 24시간 영업한다는 뜻은 아님"을 단서로 단다.
        if reg.get("dine_in_cutoff_note"):
            facts.append("식당·카페 영업시간 제한 해제")
        mid = _effective(reg, "midnight_closing_examples")
        if mid:
            facts.append("24시까지 운영 제한: " + "·".join(str(x) for x in mid))

    mline = _meeting_line(_effective(reg, "private_meetings") or {})
    if mline:
        facts.append(mline)

    # 집합금지는 후속 구간이 다시 적지 않아도 해제된 것이 아니다. 원자료가 변경분만
    # 기술하므로, 명시가 없으면 직전 구간 값을 이어받는다. 이어받지 않으면 정책 주간
    # 한복판에서 규제가 사라진 것처럼 보이는 인공 변화가 생긴다.
    vp = _effective(reg, "vaccine_pass_examples")
    if vp:
        line = "방역패스(접종증명·음성확인) 적용: " + "·".join(str(x) for x in vp)
        ext = _effective(reg, "vaccine_pass_expansion_examples")
        if ext:
            line += " / 확대: " + "·".join(str(x) for x in ext)
        facts.append(line)

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
    # 구간이 나뉘어도 내용이 같으면 시민에게는 바뀐 게 없다. 자료상 경계일 뿐인데
    # "오늘부터 바뀐다"를 띄우면 있지도 않은 변화를 프롬프트에 만들어 넣게 된다.
    if date(fy, fm, fd) == day and _materially_changed(reg):
        headline += " — 오늘부터 조치가 바뀐다"

    # 구간의 note 는 "이 요약에 미포함" 같은 자료 한계 메모라 시민이 알 내용이 아니다.
    # 프롬프트에 넣으면 의미 없는 문장이 들어가고, 구간이 바뀔 때 문장만 달라져
    # 없던 변화가 생긴 것처럼 보인다. 실제 세상 사실(추석)만 남긴다.
    note = None
    if CHUSEOK[0] <= day <= CHUSEOK[1]:
        note = "추석 연휴. 가정 내 가족모임 예외가 적용되며 일반 식당 모임에는 확대되지 않는다"

    return {"headline": headline, "facts": facts, "note": note}
