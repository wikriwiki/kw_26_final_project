# -*- coding: utf-8 -*-
"""민생회복 소비쿠폰 사용처 판정 — POI별 쿠폰 사용 가능 여부 (순수 룰).

정책 기준 (행안부, 2025):
  가능:  연매출 30억 이하 소상공인 매장 — 전통시장·동네마트·식당·카페·편의점·
        미용실·약국·의원·학원·프랜차이즈 "가맹점" 등
  불가:  대형마트·기업형 슈퍼(SSM)·백화점·면세점·대형 전자판매점·프랜차이즈 "직영점"·
        유흥·사행·환금성 업종·온라인 전자상거래

시뮬 근사 (한계 명시):
  - "연매출 30억"은 관측 불가 → 원천이 소상공인시장진흥공단 상가 DB라
    기본값 eligible=True 가 실제와 부합. 예외만 룰로 제외:
    ① 상호명 브랜드 패턴 (대형마트·SSM·백화점·면세·대형가전·직영 프랜차이즈 대표)
    ② L2 카테고리 (환금성: 시계·귀금속 / 비소비 서비스)
    ③ 상호명 업태 패턴 (유흥·사행 — L2 '주점'은 호프·일반주점이라 가능,
       단란·유흥주점은 이름으로만 구분)
  - 프랜차이즈 직영/가맹 구분은 대표 직영 브랜드(스타벅스·올리브영·다이소 등)만 반영.

사용:
  런타임  — stage2 후보에 DB값(p.coupon_eligible) 없을 때 fallback 판정
  백필    — scripts/neo4j_load/09_coupon_eligibility.py 가 이 룰로 전 POI에 속성 기록
"""
from __future__ import annotations

import re

# ① 상호명 제외 브랜드/시설 (대형마트·SSM·백화점·면세·대형가전·직영 프랜차이즈)
#    주의: '이마트24'는 가맹 편의점이라 가능 — negative lookahead.
_NAME_EXCLUDE = re.compile(
    r"이마트(?!24)|홈플러스|롯데마트|코스트코|이케아|트레이더스"
    r"|에브리데이|롯데슈퍼|GS더프레시|노브랜드"
    r"|백화점|면세점|아울렛"
    r"|하이마트|전자랜드|디지털프라자|베스트샵"
    r"|스타벅스|올리브영|다이소"
)

# ③ 상호명 유흥·사행·환금 패턴
_NAME_VICE = re.compile(r"단란주점|유흥주점|룸살롱|나이트클럽|카지노|복권|로또|성인|안마방")

# ② L2 카테고리 제외
_SUB_EXCLUDE = {
    "시계·귀금속",                      # 환금성
    "부동산", "법무", "회계·세무",       # 비소비 서비스 (수수료·공과금성)
}

# L1 전체 제외는 없음 (주점 L2=호프·일반주점은 사용 가능이 실제 기준)


def is_coupon_eligible(name: str | None, sub: str | None, l1: str | None = None) -> tuple[bool, str]:
    """(사용 가능 여부, 사유 코드). 판단 불가 정보는 관대하게 True (기본 소상공인 가정)."""
    n = (name or "").strip()
    if n:
        if _NAME_EXCLUDE.search(n):
            return False, "brand_large_or_direct"
        if _NAME_VICE.search(n):
            return False, "vice_or_cashable"
    if (sub or "") in _SUB_EXCLUDE:
        return False, "category_excluded"
    return True, "ok"


# =========================================================
# 자체 테스트
# =========================================================
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cases = [
        # (name, sub, expect)
        ("김밥천국 역삼점", "분식", True),
        ("이마트 성수점", "종합소매", False),
        ("이마트24 R성수점", "편의점", True),      # 가맹 편의점 — 가능
        ("홈플러스 익스프레스 문래점", "슈퍼마켓", False),
        ("롯데백화점 본점", "기타상품", False),
        ("스타벅스 강남대로점", "카페", False),     # 직영 프랜차이즈
        ("메가엠지씨커피 신림점", "카페", True),    # 가맹 프랜차이즈 — 가능
        ("GS25 관악점", "편의점", True),
        ("동네정육식당", "정육", True),
        ("금은방 순금나라", "시계·귀금속", False),  # 환금성
        ("황금성 단란주점", "일반주점", False),
        ("호프의전설", "호프", True),               # 일반 호프 — 가능
        ("연세밝은치과의원", "치과", True),
        ("서울부동산공인중개사", "부동산", False),
        ("복권명당", "기타상품", False),
        (None, "한식", True),
    ]
    bad = 0
    for name, sub, expect in cases:
        got, why = is_coupon_eligible(name, sub)
        mark = "✔" if got == expect else "✘"
        if got != expect:
            bad += 1
        print(f"  {mark} {str(name):24s} | {str(sub):8s} → {got} ({why})")
    assert bad == 0, f"{bad} case(s) failed"
    print("\nALL OK")
