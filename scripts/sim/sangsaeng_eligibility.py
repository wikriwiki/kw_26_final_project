# -*- coding: utf-8 -*-
"""상생소비지원금 실적 산정 업종 판정 — POI별 적립/제외 여부.

정책 기준 (기재부·KDI 2021 상생소비지원금, 네거티브 방식):
  적립(실적 인정): 소상공인 중심 대부분 업종.
  제외:  대형마트·대형백화점(아울렛·복합몰)·면세점·대형전자전문점·대형종합온라인몰·
         홈쇼핑·유흥·사행·신차구입·명품전문매장·실외골프장·비소비성지출.

판정 축 = 소상공인시장진흥공단 표준 업종 L3 코드 (docs/SANGSAENG_UPJONG_CODE_FIX.md)
  우리 카테고리(L1/L2)는 L3를 병합한 소비자 관점 분류라 제외업종이 적립업종과
  한 덩어리로 뭉쳐 있다. 대표적으로 L2 '일반주점' =
      I21104 요리주점 10,193 (적립)  +  I21101/I21102 유흥주점 2,365 (제외)
  이라서 카테고리로도 상호명으로도 갈라지지 않는다. L3 코드만이 정확한 분리 축이다.

판정 우선순위:
  ① L3 코드가 제외 목록에 있으면      → arm 확정            (근거: code_*)
  ② 상호명 룰 — 코드가 못 닿는 항목만  → arm 확정            (근거: name_*/brand_*)
       · 카지노·경마장류 (사행, 전용 L3 코드 부재)
       · 대형마트·백화점 브랜드 (대형업태, 전용 L3 코드 부재)
  ③ 코드가 있고 ①②에 안 걸리면        → eligible 확정       (근거: code_eligible)
  ④ 코드 미확보 POI                   → 기존 L2 룰 fallback (근거: sub_*/ok)

  ②가 ①보다 뒤인 이유: 코드로 확정된 arm이 상호명 오탐에 덮이면 안 된다.
  ②가 ③보다 앞인 이유: 코드가 요리주점이어도 상호가 "○○카지노"면 사행으로 잡아야 한다.

네 층위의 arm을 반환한다 (validate_sangsaeng C1 대조군 선별용):
    excluded_luxury          = 시계·귀금속 (명품 근사)          — 반응 가능
    excluded_vice            = 유흥주점·복권 (+상호명 사행)      — 반응 가능
    excluded_nonconsumption  = 부동산·법무·회계세무             — 반응 원천 차단(위약)
    excluded_other           = 대형업태 (상호명 브랜드 룰)       — 잔여 버킷
    eligible                 = 적립

성인업소(성인용품점·안마방류)는 제외하지 않는다 — 전용 L3 코드가 없고(G22199·S20802
잡화 통에 혼재), 상호명도 실측상 복합어 매칭 0건 / '성인' 단독은 100% 오탐
("삼성인쇄"·"성인피아노학원")이라 잡을 수단이 없다. docs §4.3 참조.

사용:
  런타임  — stage2 후보에 DB값(p.sangsaeng_eligible) 없을 때 fallback 판정
  백필    — scripts/neo4j_load/11_sangsaeng_eligibility.py 가 원천 CSV의 L3 코드로 전 POI 기록
  검증    — python scripts/sim/sangsaeng_eligibility.py          (케이스 테스트)
            python scripts/sim/sangsaeng_eligibility.py --audit  (매핑 247코드 전수 판정)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# =========================================================
# ① L3 업종코드 → arm 확정 테이블
# =========================================================
# 명품 proxy. L2 '시계·귀금속'과 1:1 (수리업 S20603은 판매가 아니라 적립).
_CODE_LUXURY = {"G21701"}

# 유흥·사행. L2로는 각각 '일반주점'·'유원지·오락'에 병합되어 있어 코드로만 분리된다.
_CODE_VICE = {
    "I21101",   # 일반 유흥 주점  2,269
    "I21102",   # 무도 유흥 주점     96
    "R10410",   # 복권 발행/판매업 1,038
}

# 비소비 서비스 = 위약(placebo) arm.
# 주의: 정책 원문의 '비소비성 지출'(보험료·세금·공과금)과 개념이 다르다.
#       "정책상 제외라서"가 아니라 "캐시백에 반응할 수 없어서" 대조군이다 (docs §4.1).
_CODE_NONCONSUMPTION = {
    "L10203",                                                     # 부동산 중개/대리
    "M10301", "M10302", "M10303", "M10306", "M10307", "M10399",   # 법무
    "M10401", "M10402", "M10499",                                 # 회계·세무
}

# =========================================================
# ② 상호명 룰 — 전용 L3 코드가 없어 코드로 분리 불가능한 것만
# =========================================================
# 사행: 매핑 247코드에 카지노·경마 전용 코드가 없다. 상호명이 유일한 수단.
#   '경정'·'토토' 단독은 쓰지 않는다 — "자동차 경정비", "카페토토로" 오탐 때문.
_NAME_GAMBLING = re.compile(r"카지노|경마장|경륜장|경정장|스포츠토토")

# 대형업태: L2 '슈퍼마켓'·'가전·통신'에 대형/소형이 뭉쳐 있어 코드로 분리 불가.
#   실측 교정 (서울사랑상품권 가맹점 140,140건 대조, docs §9-7):
#     · 부분일치는 190건 매칭인데 대부분 오탐이었다 — "국대떡볶이이마트목동점",
#       "깐부치킨 압구정현대백화점", "던킨 목동홈플러스점", "다비치안경 시흥홈플러스앞점"처럼
#       대형점에 입점했거나 인근임을 지점명으로 표시한 소상공인이다.
#       → 상호 '맨 앞'에서만 매칭(_LARGE_ANCHOR). 190건 → 19건.
#     · 일반명사(백화점·면세점·아울렛)는 브랜드 앵커 필수 — 없으면 "고기백화점"·
#       "정육백화점"·"백화점약국"이 오탐.
#     · 개별 오탐 3건 교정: 노브랜드버거(SSG 가맹 햄버거)·에브리데이 단독어
#       ("에브리데이샐러드"·"에브리데이짐")·공백 낀 "이마트 24".
_BRANDS_LARGE = (
    r"이마트(?!\s*24)|홈플러스|롯데마트|코스트코|이케아|트레이더스"
    r"|이마트\s*에브리데이|롯데슈퍼|GS더프레시|노브랜드(?!버거)"
    r"|하이마트|전자랜드|디지털프라자|베스트샵"
    r"|(?:롯데|신세계|현대|갤러리아|AK|스타필드|타임스퀘어)\s*백화점"
    r"|(?:롯데|신세계|신라|현대|동화|HDC)\s*면세점"
    r"|(?:롯데|신세계|현대|모다|마리오|스타필드)\s*아울렛"
)
# 법인 접두어와 짧은 그룹 표기만 건너뛰고 상호 맨 앞을 요구한다.
_NAME_LARGE = re.compile(
    r"^(?:\(주\)|㈜|주식회사|주\)|\(유\))?\s*(?:LG|SSG)?\s*(?:" + _BRANDS_LARGE + r")"
)

# =========================================================
# ④ L2 카테고리 룰 — 코드 미확보 POI의 fallback
# =========================================================
_SUB_LUXURY = {"시계·귀금속"}
_SUB_NONCONSUMPTION = {"부동산", "법무", "회계·세무"}


# 반환 arm 상수
ARM_ELIGIBLE = "eligible"
ARM_EXCLUDED_LUXURY = "excluded_luxury"                  # 시계·귀금속 (명품)
ARM_EXCLUDED_VICE = "excluded_vice"                      # 유흥·사행
ARM_EXCLUDED_NONCONSUMPTION = "excluded_nonconsumption"  # 부동산·법무·회계세무 (위약)
ARM_EXCLUDED_OTHER = "excluded_other"                    # 대형업태 (잔여)

EXCLUDED_ARMS = (
    ARM_EXCLUDED_LUXURY,
    ARM_EXCLUDED_VICE,
    ARM_EXCLUDED_NONCONSUMPTION,
    ARM_EXCLUDED_OTHER,
)

# 근거코드 → 판정 출처. 백필 스크립트가 p.sangsaeng_src 에 기록한다.
_SRC_BY_WHY = {
    "code_luxury": "upjong_code",
    "code_vice": "upjong_code",
    "code_nonconsumption": "upjong_code",
    "code_eligible": "upjong_code",
    "name_gambling": "rule_name",
    "brand_large": "rule_name",
    "sub_luxury": "rule_fallback",
    "sub_nonconsumption": "rule_fallback",
    "ok": "rule_fallback",
}


def src_of(why: str) -> str:
    """근거코드 → 판정 출처(upjong_code / rule_name / rule_fallback)."""
    return _SRC_BY_WHY.get(why, "rule_fallback")


def arm_from_code(code: str | None) -> tuple[str, str] | None:
    """L3 업종코드가 제외 목록에 있으면 (arm, 근거). 아니면 None."""
    c = (code or "").strip().upper()
    if not c:
        return None
    if c in _CODE_LUXURY:
        return ARM_EXCLUDED_LUXURY, "code_luxury"
    if c in _CODE_VICE:
        return ARM_EXCLUDED_VICE, "code_vice"
    if c in _CODE_NONCONSUMPTION:
        return ARM_EXCLUDED_NONCONSUMPTION, "code_nonconsumption"
    return None


def sangsaeng_arm(
    name: str | None,
    sub: str | None,
    l1: str | None = None,
    upjong_l3: str | None = None,
) -> tuple[str, str]:
    """(arm, 근거코드) 반환. 우선순위는 모듈 docstring 참조.

    upjong_l3 는 키워드 기본값 None — 코드를 모르는 호출부(런타임 fallback)는
    기존과 동일하게 L2 룰로 동작한다.
    """
    n = (name or "").strip()
    c = (upjong_l3 or "").strip().upper()

    # ① 코드 제외 목록
    hit = arm_from_code(c)
    if hit:
        return hit

    # ② 상호명 룰 — 코드가 못 닿는 영역만
    if n:
        if _NAME_GAMBLING.search(n):
            return ARM_EXCLUDED_VICE, "name_gambling"
        if _NAME_LARGE.search(n):
            return ARM_EXCLUDED_OTHER, "brand_large"

    # ③ 코드가 있고 ①②에 안 걸림 → 적립 확정
    if c:
        return ARM_ELIGIBLE, "code_eligible"

    # ④ 코드 미확보 → L2 룰 fallback
    s = (sub or "").strip()
    if s in _SUB_LUXURY:
        return ARM_EXCLUDED_LUXURY, "sub_luxury"
    if s in _SUB_NONCONSUMPTION:
        return ARM_EXCLUDED_NONCONSUMPTION, "sub_nonconsumption"
    return ARM_ELIGIBLE, "ok"


def is_sangsaeng_eligible(
    name: str | None,
    sub: str | None,
    l1: str | None = None,
    upjong_l3: str | None = None,
) -> tuple[bool, str]:
    """(적립 여부, 근거코드). is_coupon_eligible 과 동일 시그니처 (백필 재사용)."""
    arm, why = sangsaeng_arm(name, sub, l1, upjong_l3)
    return (arm == ARM_ELIGIBLE), why


# =========================================================
# --audit : 매핑 247코드 전수 판정 (DB·CSV 불필요)
# =========================================================
_MAPPING_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "neo4j_load" / "mapping" / "mapping_upjong_to_sub.json"
)

# 적립이어야 하는데 헷갈리기 쉬운 코드 — 하나라도 제외로 가면 실패 (docs §8.1)
_WATCHLIST = [
    ("I21104", "요리 주점 — 유흥주점과 같은 L2"),
    ("I21103", "생맥주 전문"),
    ("R10407", "노래방 — 유흥주점과 별개"),
    ("R10311", "골프 연습장 — 제외는 실외골프장"),
    ("S20802", "마사지/안마 — 일반 마사지원"),
    ("R10404", "전자 게임장 — 사행 아님"),
    ("S20603", "시계/귀금속 수리업 — 판매 아님"),
    ("G20911", "가방 소매업 — 명품전문매장 아님"),
    ("G21801", "예술품 소매업 — 화랑"),
    ("G20202", "자동차 부품 — 제외는 신차 구입"),
]

# TA 합격 기준 (docs §8.1). 원천 _count 합이므로 정확히 일치해야 한다.
_EXPECT = {
    ARM_EXCLUDED_VICE: 3403,
    ARM_EXCLUDED_LUXURY: 2404,
    ARM_EXCLUDED_NONCONSUMPTION: 40614,
}


def run_audit() -> int:
    """매핑 전 코드를 판정 함수에 통과시켜 arm별 집계. 반환값 = 실패 건수."""
    raw = json.loads(_MAPPING_PATH.read_text(encoding="utf-8"))
    by_arm: dict[str, list[tuple[str, str, int]]] = {}
    total = 0
    for code, v in raw.items():
        n = int(v.get("_count") or 0)
        total += n
        arm, _why = sangsaeng_arm(None, v.get("sub"), v.get("cat"), upjong_l3=code)
        by_arm.setdefault(arm, []).append((code, v.get("_l3_name", ""), n))

    print(f"=== 매핑 {len(raw)}개 코드 전수 판정 ===\n")
    bad = 0
    for arm in (ARM_EXCLUDED_VICE, ARM_EXCLUDED_LUXURY, ARM_EXCLUDED_NONCONSUMPTION,
                ARM_EXCLUDED_OTHER):
        rows = sorted(by_arm.get(arm, []), key=lambda r: -r[2])
        n_sum = sum(r[2] for r in rows)
        print(f"[{arm}] {n_sum:,}개 / {len(rows)}종")
        for code, nm, n in rows:
            sub = raw[code]["sub"]
            print(f"  {code:8s} {nm:24s} {n:>7,}   (현재 L2: {sub})")
        exp = _EXPECT.get(arm)
        if exp is not None:
            mark = "✔" if n_sum == exp else "✘"
            if n_sum != exp:
                bad += 1
            print(f"  {mark} 기대 {exp:,} / 실측 {n_sum:,}")
        print()

    elig = by_arm.get(ARM_ELIGIBLE, [])
    print(f"[{ARM_ELIGIBLE}] {sum(r[2] for r in elig):,}개 / {len(elig)}종")
    print("  ★ 반드시 여기 있어야 하는 것 (오분류 감시):")
    elig_codes = {r[0] for r in elig}
    for code, note in _WATCHLIST:
        if code not in raw:
            print(f"    ? {code:8s} 매핑에 없음 — 확인 필요")
            bad += 1
            continue
        ok = code in elig_codes
        mark = "✔" if ok else "✘"
        if not ok:
            bad += 1
        print(f"    {mark} {code:8s} {raw[code].get('_l3_name',''):24s} "
              f"{raw[code].get('_count',0):>7,}  — {note}")

    n_ex = total - sum(r[2] for r in elig)
    print(f"\n제외 합계 {n_ex:,} / 전체 {total:,} ({100 * n_ex / max(total, 1):.2f}%)")
    if n_ex / max(total, 1) >= 0.10:
        print("  ✘ 제외 비율 10% 이상 — 상생은 네거티브라 대부분 적립이어야 정상")
        bad += 1
    else:
        print("  ✔ 제외 비율 10% 미만")
    return bad


# =========================================================
# 케이스 단위 테스트 (docs §8.2 T1)
# =========================================================
_CASES = [
    # (upjong_l3, name, sub, expect_arm, 검증 의도)
    # ── ① 코드로 확정되는 제외 ──
    ("I21101", "황금성", "일반주점", ARM_EXCLUDED_VICE, "상호에 단서 없어도 코드로 포착"),
    ("I21102", "블루문", "일반주점", ARM_EXCLUDED_VICE, "무도 유흥"),
    ("R10410", "복권명당", "유원지·오락", ARM_EXCLUDED_VICE, "복권"),
    ("G21701", "순금나라", "시계·귀금속", ARM_EXCLUDED_LUXURY, "명품 proxy"),
    ("L10203", "서울공인중개사", "부동산", ARM_EXCLUDED_NONCONSUMPTION, "위약 arm"),
    ("M10402", "정직한세무회계", "회계·세무", ARM_EXCLUDED_NONCONSUMPTION, "위약 arm"),
    # ── ★ 같은 L2인데 코드로 갈리는 핵심 케이스 ──
    ("I21104", "이자카야 하나", "일반주점", ARM_ELIGIBLE, "★ 요리주점은 적립"),
    ("I21103", "호프의전설", "호프", ARM_ELIGIBLE, "생맥주 전문"),
    ("R10404", "○○게임장", "유원지·오락", ARM_ELIGIBLE, "★ 전자게임장은 적립"),
    ("R10407", "코인노래연습장", "노래방", ARM_ELIGIBLE, "노래방은 적립"),
    ("S20802", "○○안마원", "마사지", ARM_ELIGIBLE, "일반 안마원은 적립"),
    ("S20603", "시계수리명장", "수리", ARM_ELIGIBLE, "수리업은 적립"),
    # ── ② 상호명 룰이 코드보다 앞서는 케이스 ──
    ("I21104", "○○카지노바", "일반주점", ARM_EXCLUDED_VICE,
     "★ 코드는 적립인데 상호가 사행 — ②가 ③보다 앞"),
    ("G20404", "홈플러스 익스프레스 문래점", "슈퍼마켓", ARM_EXCLUDED_OTHER,
     "★ 코드는 슈퍼마켓인데 상호가 대형마트"),
    # ── ② 상호명 룰 오탐 방지 ──
    (None, "고기백화점", "정육", ARM_ELIGIBLE, "★ 백화점 앵커화 — 동네 정육점"),
    (None, "황금한우정육백화점", "정육", ARM_ELIGIBLE, "★ 실제 CSV 상호"),
    (None, "면목백화점약국", "약국", ARM_ELIGIBLE, "★ 실제 CSV 상호"),
    (None, "롯데백화점 본점", "기타상품", ARM_EXCLUDED_OTHER, "★ 앵커 붙은 진짜 백화점"),
    (None, "신세계면세점 명동", "기타상품", ARM_EXCLUDED_OTHER, "앵커 붙은 면세점"),
    (None, "삼성인쇄", "인쇄", ARM_ELIGIBLE, "★ '성인' 패턴 미도입 — 넣었다면 오탐"),
    (None, "성인피아노학원", "학원", ARM_ELIGIBLE, "★ 실제 CSV 상호"),
    (None, "동대문여성인력개발센터", "고용서비스", ARM_ELIGIBLE, "★ 실제 CSV 상호"),
    (None, "스피드경정비", "차량정비", ARM_ELIGIBLE, "★ '경정' 단독 미사용 — 자동차 경정비"),
    (None, "카페토토로", "카페", ARM_ELIGIBLE, "★ '토토' 단독 미사용"),
    # ── ② 대형 브랜드는 유지 (상호 맨 앞에서만) ──
    (None, "이마트 성수점", "종합소매", ARM_EXCLUDED_OTHER, "고유 브랜드"),
    (None, "롯데마트", "종합소매", ARM_EXCLUDED_OTHER, "실제 CSV 상호"),
    (None, "홈플러스(주)익스프레스 삼전역점", "슈퍼마켓", ARM_EXCLUDED_OTHER, "실제 CSV 상호"),
    (None, "LG하이마트", "가전·통신", ARM_EXCLUDED_OTHER, "그룹 표기 허용"),
    (None, "전자랜드 용산점", "가전·통신", ARM_EXCLUDED_OTHER, "대형 전자"),
    (None, "동네컴퓨터수리", "가전·통신", ARM_ELIGIBLE, "동네 수리점"),
    # ── ★ 실측 오탐 회귀 (CSV 140,140건 대조로 발견, docs §9-7) ──
    #    대형점에 입점했거나 인근임을 지점명으로 표시한 소상공인 — 전부 적립
    (None, "국대떡볶이이마트목동점", "분식", ARM_ELIGIBLE, "★ 이마트 입점 떡볶이집"),
    (None, "깐부치킨 압구정현대백화점", "치킨", ARM_ELIGIBLE, "★ 백화점 입점 치킨집"),
    (None, "던킨 목동홈플러스점", "베이커리", ARM_ELIGIBLE, "★ 홈플러스 입점"),
    (None, "다비치안경 시흥홈플러스앞점", "안경", ARM_ELIGIBLE, "★ 홈플러스 '앞'"),
    (None, "CU 용산전자랜드 본관점", "편의점", ARM_ELIGIBLE, "★ 전자랜드 입점 편의점"),
    (None, "메가엠지씨커피 응암이마트점", "카페", ARM_ELIGIBLE, "★ 이마트 입점 카페"),
    (None, "마곡 아이마트 안경원", "안경", ARM_ELIGIBLE, "★ 아'이마트' 부분일치 오탐"),
    (None, "노브랜드버거 군자역점", "기타식사", ARM_ELIGIBLE, "★ 노브랜드버거는 가맹 햄버거"),
    (None, "에브리데이샐러드", "식료품", ARM_ELIGIBLE, "★ '에브리데이' 단독어"),
    (None, "에브리데이짐", "헬스장", ARM_ELIGIBLE, "★ '에브리데이' 단독어"),
    (None, "이마트24 성수점", "편의점", ARM_ELIGIBLE, "가맹 편의점 — lookahead"),
    (None, "이마트 24 관악조원점", "편의점", ARM_ELIGIBLE, "★ 공백 낀 이마트 24"),
    # ── ★ 소비쿠폰과 다른 지점: 직영 프랜차이즈는 상생 적립 ──
    (None, "스타벅스 강남대로점", "카페", ARM_ELIGIBLE, "★ 직영도 상생은 적립"),
    (None, "올리브영 신림점", "화장품", ARM_ELIGIBLE, "★ 직영도 상생은 적립"),
    (None, "다이소 봉천점", "생활용품", ARM_ELIGIBLE, "★ 직영도 상생은 적립"),
    # ── ④ 코드 미확보 fallback ──
    (None, "금은방 순금나라", "시계·귀금속", ARM_EXCLUDED_LUXURY, "L2 fallback"),
    (None, "정직한세무회계", "회계·세무", ARM_EXCLUDED_NONCONSUMPTION, "L2 fallback"),
    (None, None, "한식", ARM_ELIGIBLE, "정보 없으면 관대하게 적립"),
]


def run_cases() -> int:
    """케이스 테스트. 반환값 = 실패 건수."""
    bad = 0
    for code, name, sub, exp_arm, note in _CASES:
        arm, why = sangsaeng_arm(name, sub, None, upjong_l3=code)
        ok = arm == exp_arm
        if not ok:
            bad += 1
        mark = "✔" if ok else "✘"
        print(f"  {mark} {str(code or '-'):7s} | {str(name):24s} | {str(sub):10s} "
              f"→ {arm:24s} ({why}, src={src_of(why)})")
        if not ok:
            print(f"      기대 {exp_arm} — {note}")
    return bad


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if "--audit" in sys.argv:
        n_bad = run_audit()
        print("\nAUDIT OK" if n_bad == 0 else f"\nAUDIT FAILED — {n_bad}건")
        sys.exit(0 if n_bad == 0 else 1)

    n_bad = run_cases()
    print("\nALL OK" if n_bad == 0 else f"\n{n_bad} case(s) failed")
    sys.exit(0 if n_bad == 0 else 1)
