"""집안 내구재 보유 상태 — 정책과 무관한 페르소나 사실.

큰 지출은 '오늘 갑자기 사고 싶어진 것'이 아니라 '언젠가 바꿔야 하는데 미뤄둔
것'이다. 우리 에이전트는 매일 아침 백지에서 하루를 계획하므로 이 축이 아예
없었고, 그 결과 가전·가구 결제가 750 에이전트일에 4건 · 건당 1.3만원에 그쳤다
(무정책 3일 × 150명 실측). 같은 구간의 BDC 기준선은 총소비의 8.15%(오프라인
재정규화) 또는 2.25%(전체 분포)인데 시뮬은 0.37%였다 — 6~22배 부족.

빈도를 올리는 것으로는 못 메운다. 실측의 가전·가구는 **드물고 큰** 지출인데
우리는 잦고 작았다. 필요한 것은 방문 횟수가 아니라 '미뤄둔 교체'라는 상태다.

설계 원칙
  ① 정책과 독립 — 무정책·정책 구간에 동일하게 부여한다. 목록에 캐시백·문턱
     같은 정책 정보를 일절 넣지 않는다. 정책이 없어도 냉장고는 늙는다.
  ② 상태만 주고 판단은 에이전트가 한다 — "사라"가 아니라 "몇 년째 쓰고 있다".
     오늘 살지, 다음 달로 미룰지, 그냥 더 쓸지는 에이전트가 정한다.
  ③ 결정론 — agent_id 해시. 같은 사람은 언제 돌려도 같은 목록을 갖는다.
     따라서 무정책 구간과 정책 구간에서 동일하며, 쌍체차 검정이 성립한다.

가격·교체주기는 실측이 아니라 2021년 기준 경험 상수다(poi_price._L1_FALLBACK_WON
과 같은 지위). BDC 는 업종별 금액 비중만 주고 건단가·교체주기를 주지 않는다.
"""
from __future__ import annotations

import hashlib
import os

# 켜짐 여부. 기본 꺼짐 — P010 검증본 렌더를 바이트 그대로 유지한다.
ENABLED = os.environ.get("EXP_DURABLES", "0") == "1"

# 한 사람에게 보여 줄 최대 품목 수. 목록이 길면 '오늘의 할 일'처럼 읽힌다.
MAX_SHOWN = int(os.environ.get("EXP_DURABLES_MAX", "2"))

# (품목, 교체주기(년), 시세 하한(만원), 시세 상한(만원))
_HOME = [
    ("냉장고", 11, 90, 160),
    ("세탁기", 10, 60, 110),
    ("에어컨", 10, 70, 130),
    ("TV", 8, 60, 150),
    ("청소기", 8, 30, 70),
    ("전자레인지", 9, 15, 30),
    ("가스레인지", 11, 25, 50),
]
_PERSONAL = [
    ("노트북", 5, 80, 150),
    ("휴대폰", 3, 90, 140),
]
_FURNITURE = [
    ("소파", 9, 50, 130),
    ("침대", 10, 60, 150),
    ("식탁", 12, 40, 90),
    ("책상·의자", 12, 20, 60),
]

# 생애주기별 보유 품목. 1인 가구는 가구·대형가전이 적다.
_SOLO = {"사회초년생", "1인가구", "청년", "학생"}


def _h(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


def _owned(agent_id: str, life_stage: str | None, age_group: str | None) -> list[tuple]:
    """이 사람이 가지고 있는 내구재 목록(결정론)."""
    ls = (life_stage or "").strip()
    solo = any(k in ls for k in _SOLO) or (age_group or "").startswith("20")
    items = list(_HOME if not solo else _HOME[:4])
    items += _PERSONAL
    if not solo:
        items += _FURNITURE
    else:
        # 1인 가구도 침대·책상은 있다.
        items += [x for x in _FURNITURE if x[0] in ("침대", "책상·의자")]
    # 품목별 보유 여부 — 전부 다 갖고 있지는 않다.
    out = []
    for it in items:
        if _h(agent_id, "own", it[0]) % 100 < 82:
            out.append(it)
    return out


# 오래된 물건이라고 다 말썽인 것은 아니다. 10여 개 내구재를 가진 집이면 그중
# 하나쯤 교체주기를 넘긴 것은 늘 있으므로(자체 점검 97%), 희소성으로 조절하면
# 현실을 왜곡한다. 대신 **작동 상태**로 가른다 — 대부분은 아직 쓸 만하고,
# 일부만 증상이 있다. 무엇을 할지는 여전히 에이전트가 정한다.
# 품목당 증상률 8%. 교체시기를 넘긴 품목이 보통 4개쯤이므로 사람 기준으로는
# 약 4분의 1이 증상 있는 물건을 하나 갖는다. 나머지는 '아직 쓸 만하다'로 읽힌다.
_COND = [
    (92, ""),                    # 아직 쓸 만함 — 증상 없음
    (98, "소리가 커졌다"),         # 가벼운 증상
    (100, "요즘 말썽이다"),        # 실사용 문제
]


def _cond(agent_id: str, name: str) -> str:
    r = _h(agent_id, "cond", name) % 100
    for cut, text in _COND:
        if r < cut:
            return text
    return ""


def pending(agent_id: str, life_stage: str | None = None,
            age_group: str | None = None) -> list[dict]:
    """교체 시기를 지난 품목 — 증상 있는 것 우선, 최대 MAX_SHOWN 개.

    사용연수는 0 ~ 교체주기×1.6 의 결정론 난수다. 교체주기를 넘긴 것만 노출한다.
    """
    if not agent_id:
        return []
    got = []
    for name, cycle, lo, hi in _owned(agent_id, life_stage, age_group):
        span = cycle * 1.6
        years = round(span * ((_h(agent_id, "age", name) % 10000) / 10000.0), 1)
        if years < cycle:
            continue
        got.append({"name": name, "years": years, "cycle": cycle,
                    "lo_man": lo, "hi_man": hi, "cond": _cond(agent_id, name),
                    "over": round(years - cycle, 1)})
    # 증상 있는 것 우선, 그다음 오래된 순
    got.sort(key=lambda d: (0 if d["cond"] else 1, -d["over"]))
    # 증상이 하나도 없으면 한 건만 — 멀쩡한 물건 목록을 늘어놓으면 '오늘의 할 일'
    # 처럼 읽힌다. 증상이 있을 때만 두 번째 줄까지 보여 준다.
    keep = MAX_SHOWN if (got and got[0]["cond"]) else 1
    return got[:keep]


def format_block(agent_id: str, life_stage: str | None = None,
                 age_group: str | None = None) -> str:
    """페르소나 블록 끝에 붙일 한 줄. 비었으면 빈 문자열.

    '사라'는 말을 쓰지 않는다. 연수·통상 교체주기·시세·작동 상태만 사실로 적는다.
    """
    if not ENABLED:
        return ""
    items = pending(agent_id, life_stage, age_group)
    if not items:
        return ""
    parts = []
    for d in items:
        tail = d["cond"] or "아직 쓸 만하다"
        parts.append("{} {}년째({}, 보통 {}년쯤 바꾼다, 요즘 {}~{}만원)".format(
            d["name"], int(d["years"]), tail, d["cycle"], d["lo_man"], d["hi_man"]))
    return "집안 물건: " + ", ".join(parts)


# =========================================================
# 자체 점검 — 노출률과 품목 분포가 상식 범위인지
# =========================================================
if __name__ == "__main__":
    import sys
    from collections import Counter
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    stages = ["사회초년생", "기혼유자녀", "중년", "노년", "1인가구"]
    n = 5000
    cnt = Counter()
    have = sym = 0
    for i in range(n):
        aid = "AGT_TEST_%05d" % i
        ls = stages[i % len(stages)]
        p = pending(aid, ls, "30대")
        if p:
            have += 1
        if any(d["cond"] for d in p):
            sym += 1
        for d in p:
            cnt[d["name"]] += 1
    print("교체시기 지난 물건이 있는 사람: {:.1f}%".format(100 * have / n))
    print("그중 증상까지 있는 사람:       {:.1f}%".format(100 * sym / n))
    print("품목 분포:", ", ".join("%s %d" % (k, v) for k, v in cnt.most_common()))
    print()
    for i in (0, 1, 2, 7, 11, 23):
        aid = "AGT_TEST_%05d" % i
        print(aid, "|", format_block(aid, stages[i % len(stages)], "40대") or "(없음)")
