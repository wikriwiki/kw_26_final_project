"""BDC 업종명 → 시뮬 카테고리 어휘 접기.

에이전트에게 "평소 업종별 지출 구성"을 보여 줄 때, BDC 원본 이름을 그대로 주면
에이전트가 우리 어휘로 잘못 옮긴다. 7차 실측(무정책 3일 × 150명)에서 확인했다 —
"취미/오락 10%, 실내골프/헬스 4%" 를 본 에이전트가 그것을 L1 '여가' 로 읽고
여행사·유원지를 골라, 여행·레저가 9.87% → 14.87% 로 **악화**했다(BDC 기준선 4.86%).
주유소처럼 우리 POI 어휘에 아예 없는 이름도 섞여 있었다.

그래서 BDC 이름을 **에이전트가 실제로 출력해야 하는 어휘(L1 12종)** 로 접어서 준다.
L1 만으로는 가전·가구가 '쇼핑' 에 묻히므로 주요 세부업종을 괄호로 덧붙인다.

오프라인 POI 소비가 아닌 항목(PG·전자상거래·세금·보험·통신요금·택시 등)은 버리고
나머지를 100% 로 재정규화한다. 시뮬은 하루 총액을 전부 오프라인 POI 에 쓰므로
그 쪽이 에이전트가 마주하는 현실과 맞는다.
"""
from __future__ import annotations

import json
import os

# 표시 방식: fold(접어서) / raw(BDC 원본 그대로) / off(표시 안 함)
MODE = os.environ.get("EXP_CATLINE", "fold").strip().lower()

# 시뮬 L1 12종 — Stage1 이 출력해야 하는 카테고리 어휘
L1S = ("식사", "카페", "디저트", "편의점", "마트", "쇼핑",
       "미용", "건강", "교육", "여가", "주점", "기타")

# BDC 업종명 → (시뮬 L1, 세부 힌트|None)
# 세부 힌트는 L1 만으로 행동이 갈리지 않는 곳에만 단다.
_MAP: dict[str, tuple[str, str | None]] = {
    # 식사
    "한식": ("식사", None), "양식": ("식사", None), "중식": ("식사", None),
    "일식": ("식사", None), "기타요식": ("식사", None), "패스트푸드": ("식사", None),
    "구내식당": ("식사", None),
    # 카페 · 디저트
    "커피전문점": ("카페", None), "제과점": ("디저트", None),
    # 편의점
    "편의점": ("편의점", None),
    # 마트
    "할인점/슈퍼마켓": ("마트", None), "대형마트": ("마트", None),
    "슈퍼마켓 기업형": ("마트", None), "슈퍼마켓 일반형": ("마트", None),
    "정육점": ("마트", "정육"), "농수산물": ("마트", "청과·수산"),
    "기타음/식료품": ("마트", None), "기타식품": ("마트", None),
    # 쇼핑 — 가전·가구가 여기 묻히므로 힌트를 반드시 단다
    "가전": ("쇼핑", "가전"), "기타가전/가구": ("쇼핑", "가전·가구"),
    "가구": ("쇼핑", "가구"), "인테리어/건축자재/주방기구": ("쇼핑", "가구·건자재"),
    "인테리어": ("쇼핑", "가구·건자재"), "컴퓨터/소프트웨어": ("쇼핑", "가전"),
    "의복/의류": ("쇼핑", "의류"), "패션잡화": ("쇼핑", "의류"),
    "백화점": ("쇼핑", None), "쇼핑몰": ("쇼핑", None), "면세점": ("쇼핑", None),
    "생활잡화/수입상품점": ("쇼핑", "생활용품"), "기타유통": ("쇼핑", None),
    "화장품": ("쇼핑", "화장품"), "시계/귀금속": ("쇼핑", None),
    "서점": ("쇼핑", None), "문화용품": ("쇼핑", None),
    "사무기기/문구용품": ("쇼핑", "문구"), "악기/음반": ("쇼핑", None),
    "완구/아동용자전거": ("쇼핑", None), "수제용품점": ("쇼핑", None),
    "스포츠/레저용품": ("쇼핑", "오락용품"), "애완동물": ("쇼핑", "반려동물"),
    "식물·꽃": ("쇼핑", None), "안경": ("쇼핑", None),
    "중고품판매점": ("쇼핑", None), "전용매장": ("쇼핑", None),
    # 미용
    "미용실": ("미용", None), "미용서비스": ("미용", None),
    "싸우나/목욕탕": ("미용", "욕탕"), "안마/마사지": ("미용", "마사지"),
    # 건강
    "일반병원": ("건강", "의원"), "종합병원": ("건강", "병원"),
    "치과병원": ("건강", "치과"), "기타의료": ("건강", None),
    "약국": ("건강", "약국"), "한의원": ("건강", "한의원"),
    "건강보조식품": ("건강", None), "보건소": ("건강", None),
    "실내골프/헬스": ("건강", "헬스장"),
    # 교육
    "학원": ("교육", None), "학원/학습지": ("교육", None),
    "유치원": ("교육", None), "독서실": ("교육", None),
    # 여가
    "여행사/항공사": ("여가", "여행사"), "노래방": ("여가", "노래방"),
    "게임방/오락실": ("여가", "PC방·오락"), "취미/오락": ("여가", None),
    "종합레저타운/놀이동산": ("여가", "유원지"), "영화/공연": ("여가", None),
    "스포츠시설": ("여가", "스포츠"), "실내/실외골프장": ("여가", "스포츠"),
    "운동경기관람": ("여가", None), "예식장/결혼서비스": ("여가", None),
    # 주점
    "유흥업소": ("주점", None), "주류판매": ("주점", None),
    # 기타 — 우리 POI 에 있으나 위 11종에 안 들어가는 것
    "주유소": ("기타", "주유소"), "LPG가스": ("기타", "주유소"),
    "자동차서비스": ("기타", "차량정비"), "자동차용품": ("기타", "차량정비"),
    "세탁소": ("기타", "세탁"), "동물병원": ("기타", "동물병원"),
    "모텔/여관/기타숙박": ("기타", "숙박"), "부동산중개": ("기타", None),
    "법률/사무서비스": ("기타", None), "연구/번역서비스": ("기타", None),
    "고속버스/철도/여객선": ("기타", None), "오토바이": ("기타", None),
    "장례식장/묘지/장의사": ("기타", None), "체인점": ("기타", None),
    # 이름 변형 — 자체 점검에서 미매핑으로 잡힌 것들
    "LPG": ("기타", "주유소"), "생활잡화": ("쇼핑", "생활용품"),
    "패션/잡화": ("쇼핑", "의류"), "슈퍼마켓": ("마트", None),
    "유아교육": ("교육", None), "유흥주점": ("주점", None),
    "호텔/콘도": ("기타", "숙박"), "업무서비스": ("기타", None),
    "생활서비스": ("기타", None), "기타": ("기타", None),
    "여행사": ("여가", "여행사"), "실외골프/스키": ("여가", "스포츠"),
    "기타유흥업소": ("주점", None), "화원": ("쇼핑", None),
    "교육용품": ("쇼핑", "문구"), "기타쇼핑": ("쇼핑", None),
    "기타서비스": ("기타", None),
}

# 오프라인 POI 소비가 아니라 접을 수 없는 것 — 버리고 재정규화한다.
_DROP = {
    "결제대행(PG)", "전자상거래(다품목취급)", "택시", "세금공과금", "보험",
    "통신요금(이동/시내전화)", "통신요금(PC통신/무선호출)", "'통신요금(이동",
    "'통신요금(PC", "학교등록금", "상품권/복권", "방문판매/다단계판매",
    "방문판매/다단계", "주차장", "신차판매", "수입자동차", "중고차판매",
    "자동차판매", "ZZ_나머지", " ZZ_나머지", "개인(정보없음)", "*",
    "PG결제", "정류소/주차장",
}


def fold(ratios: dict) -> list[tuple[str, float, list[str]]]:
    """BDC 비중 dict → [(L1, 비중, [세부힌트...])] 내림차순, 100% 재정규화.

    매핑에 없고 버림목록에도 없는 이름은 '기타' 로 보낸다(조용히 사라지지 않게).
    """
    agg: dict[str, float] = {}
    hints: dict[str, dict[str, float]] = {}
    total = 0.0
    for name, v in (ratios or {}).items():
        try:
            w = float(v)
        except (TypeError, ValueError):
            continue
        if w <= 0 or name in _DROP:
            continue
        # 이미 시뮬 L1 어휘로 들어온 값은 그대로 둔다(픽스처·수기 입력 대비).
        got = _MAP.get(name)
        if got is None:
            got = (name, None) if name in L1S else ("기타", None)
        l1, hint = got
        agg[l1] = agg.get(l1, 0.0) + w
        total += w
        if hint:
            hints.setdefault(l1, {})
            hints[l1][hint] = hints[l1].get(hint, 0.0) + w
    if total <= 0:
        return []
    out = []
    for l1, w in agg.items():
        hs = hints.get(l1) or {}
        top = [k for k, _ in sorted(hs.items(), key=lambda x: -x[1])[:2]]
        out.append((l1, w / total, top))
    out.sort(key=lambda x: -x[1])
    return out


def line(raw: str | dict | None, min_share: float = 0.02) -> str:
    """페르소나 블록에 넣을 한 줄. 비었으면 빈 문자열.

    구성은 **사실로만** 제시한다. "이 비중 안에서 움직여라" 는 지시를 붙였더니 모든
    지출이 '어차피 했을 것' 이 되어 MPC 가 0.174 → 0.075 로 무너진 적이 있다(T4·T5).
    """
    if MODE == "off" or not raw:
        return ""
    try:
        d = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return ""
    if not d:
        return ""

    if MODE == "raw":   # 구동작(7차까지) — A/B 대조용
        top = sorted(d.items(), key=lambda x: -float(x[1]))[:8]
        body = ", ".join(f"{k} {100*float(v):.0f}%" for k, v in top if float(v) > 0.004)
        return ("평소 업종별 지출 구성(카드 실측): " + body) if body else ""

    parts = []
    for l1, share, hs in fold(d):
        if share < min_share:
            continue
        tag = f"{l1} {100*share:.0f}%"
        if hs:
            tag += "(" + "·".join(hs) + ")"
        parts.append(tag)
    if not parts:
        return ""
    return "평소 업종별 지출 구성(카드 실측): " + ", ".join(parts)


# =========================================================
# 자체 점검 — 매핑 커버리지와 출력 예시
# =========================================================
if __name__ == "__main__":
    import sys
    from collections import Counter
    from pathlib import Path
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    src = Path(__file__).resolve().parents[2] / "data" / "neo4j_load" / "agents" / "agents_final.json.bak_orig"
    data = json.loads(src.read_text(encoding="utf-8"))

    seen, unmapped = Counter(), Counter()
    amt_mapped = amt_drop = amt_unmapped = 0.0
    for r in data:
        sp = r.get("spending") or {}
        wd = sp.get("weekday_top_categories") or {}
        s0 = sp.get("daily_spending_weekday") or 0
        for k, v in wd.items():
            seen[k] += 1
            w = s0 * float(v)
            if k in _DROP:
                amt_drop += w
            elif k in _MAP:
                amt_mapped += w
            else:
                amt_unmapped += w
                unmapped[k] += 1
    tot = amt_mapped + amt_drop + amt_unmapped
    print("BDC 업종 {}종 등장".format(len(seen)))
    print("  매핑됨   {:6.2f}%".format(100 * amt_mapped / tot))
    print("  의도 제외 {:6.2f}%  (PG·전자상거래·세금 등)".format(100 * amt_drop / tot))
    print("  미매핑   {:6.2f}%  {}".format(
        100 * amt_unmapped / tot,
        ", ".join(k for k, _ in unmapped.most_common(10)) or "없음"))
    print()
    for r in data[:4]:
        wd = (r.get("spending") or {}).get("weekday_top_categories") or {}
        print(r["agent_id"])
        print("   raw  :", line(wd) if MODE == "raw" else
              "평소 업종별 지출 구성(카드 실측): " +
              ", ".join(f"{k} {100*float(v):.0f}%"
                        for k, v in sorted(wd.items(), key=lambda x: -x[1])[:8]
                        if float(v) > 0.004))
        print("   fold :", line(wd))
        print()
