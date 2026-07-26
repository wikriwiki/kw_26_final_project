# -*- coding: utf-8 -*-
"""상생소비지원금 실적 산정 업종 판정 — POI별 적립/제외 여부 (순수 룰).

정책 기준 (기재부·KDI 2021 상생소비지원금, 네거티브 방식):
  적립(실적 인정): 소상공인 중심 대부분 업종.
  제외:  대형마트·대형백화점(아울렛·복합몰)·면세점·대형전자전문점·대형종합온라인몰·
         홈쇼핑·유흥·사행·신차구입·명품전문매장·실외골프장·비소비성지출.

우리 데이터 근사 (한계 명시 — docs/POLICY_BACKTEST_SANGSAENG.md §4.3·§8):
  원천이 소상공인시장진흥공단 상가 DB라 기본값 eligible=True 가 실제와 부합
  (우리 POI ≈ 전부 적립업종). 예외만 룰로 제외:
    ① 상호명 브랜드 패턴 (대형마트·SSM·백화점·면세·대형가전) — 있으면 제외.
       ※ 소비쿠폰과 달리 프랜차이즈 직영(스타벅스·올리브영·다이소)은 상생 적립이라
          제외하지 않는다 (coupon_eligibility.py 와의 결정적 차이).
    ② 상호명 유흥·사행 패턴 (단란·유흥주점·룸살롱·카지노·복권 등) — 제외.
       ※ L2 '주점'(호프·일반주점)은 요식 적립. 유흥주점만 이름으로 구분.
    ③ L2 카테고리 (시계·귀금속 = 명품 proxy / 비소비 서비스: 부동산·법무·회계세무).

세 층위의 '제외'를 구분해 반환한다 (validate_sangsaeng C1 대조군 선별용):
    excluded_luxury  = 시계·귀금속 (명품 근사, 유일하게 sub로 깨끗이 분리되는 C1 arm)
    excluded_other   = 유흥·사행·대형업태·비소비 (상호명/기타 sub — 오염·일부만)
    eligible         = 적립

사용:
  런타임  — stage2 후보에 DB값(p.sangsaeng_eligible) 없을 때 fallback 판정
  백필    — scripts/neo4j_load/11_sangsaeng_eligibility.py 가 이 룰로 전 POI에 속성 기록
"""
from __future__ import annotations

import re
import sys

# ① 상호명 제외 브랜드/시설 (대형마트·SSM·백화점·면세·대형가전)
#    소비쿠폰(_NAME_EXCLUDE)과 달리 스타벅스·올리브영·다이소는 뺐다 — 상생은 이들 적립.
#    '이마트24'는 가맹 편의점이라 적립 — negative lookahead 유지.
_NAME_LARGE = re.compile(
    r"이마트(?!24)|홈플러스|롯데마트|코스트코|이케아|트레이더스"
    r"|에브리데이|롯데슈퍼|GS더프레시|노브랜드"
    r"|백화점|면세점|아울렛"
    r"|하이마트|전자랜드|디지털프라자|베스트샵"
)

# ② 상호명 유흥·사행 패턴 (상생 제외 — 캐시백 메시지에 반응할 업종 아님)
_NAME_VICE = re.compile(r"단란주점|유흥주점|룸살롱|나이트클럽|카지노|복권|로또|성인|안마방")

# ③-a 명품 proxy = 시계·귀금속 (C1 주력 제외 arm). 단독 취급을 위해 분리.
_SUB_LUXURY = {"시계·귀금속"}

# ③-b 비소비성 서비스 (수수료·공과금성) — 제외이나 C1 arm 아님
_SUB_NONCONSUMPTION = {"부동산", "법무", "회계·세무"}


# 반환 arm 상수
ARM_ELIGIBLE = "eligible"
ARM_EXCLUDED_LUXURY = "excluded_luxury"      # 시계·귀금속 (C1 placebo)
ARM_EXCLUDED_OTHER = "excluded_other"        # 유흥·사행·대형·비소비


def sangsaeng_arm(name: str | None, sub: str | None, l1: str | None = None) -> tuple[str, str]:
    """(arm, 사유코드) 반환. arm ∈ {eligible, excluded_luxury, excluded_other}.

    판단 불가 정보는 관대하게 eligible (기본 소상공인 가정, 데이터 현실 §4.3).
    """
    n = (name or "").strip()
    s = (sub or "").strip()
    if n:
        if _NAME_LARGE.search(n):
            return ARM_EXCLUDED_OTHER, "brand_large"
        if _NAME_VICE.search(n):
            return ARM_EXCLUDED_OTHER, "vice_or_gambling"
    if s in _SUB_LUXURY:
        return ARM_EXCLUDED_LUXURY, "luxury_proxy"      # 시계·귀금속 = 명품 근사
    if s in _SUB_NONCONSUMPTION:
        return ARM_EXCLUDED_OTHER, "non_consumption"
    return ARM_ELIGIBLE, "ok"


def is_sangsaeng_eligible(name: str | None, sub: str | None, l1: str | None = None) -> tuple[bool, str]:
    """(적립 여부, 사유코드). is_coupon_eligible 과 동일 시그니처 (백필 재사용)."""
    arm, why = sangsaeng_arm(name, sub, l1)
    return (arm == ARM_ELIGIBLE), why


# =========================================================
# 자체 테스트
# =========================================================
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cases = [
        # (name, sub, expect_eligible, expect_arm)
        ("김밥천국 역삼점", "분식", True, ARM_ELIGIBLE),
        ("이마트 성수점", "종합소매", False, ARM_EXCLUDED_OTHER),
        ("이마트24 성수점", "편의점", True, ARM_ELIGIBLE),          # 가맹 편의점 — 적립
        ("홈플러스 익스프레스 문래점", "슈퍼마켓", False, ARM_EXCLUDED_OTHER),
        ("롯데백화점 본점", "기타상품", False, ARM_EXCLUDED_OTHER),
        ("신세계면세점 명동", "기타상품", False, ARM_EXCLUDED_OTHER),
        # ★ 소비쿠폰과 다른 지점: 직영 프랜차이즈는 상생 적립
        ("스타벅스 강남대로점", "카페", True, ARM_ELIGIBLE),
        ("올리브영 신림점", "화장품", True, ARM_ELIGIBLE),
        ("다이소 봉천점", "생활용품", True, ARM_ELIGIBLE),
        # 일반 술집(호프·일반주점)은 요식 적립, 유흥주점만 제외
        ("호프의전설", "호프", True, ARM_ELIGIBLE),
        ("역전할머니맥주 신림점", "일반주점", True, ARM_ELIGIBLE),
        ("황금성 단란주점", "일반주점", False, ARM_EXCLUDED_OTHER),
        ("VIP 룸살롱", "일반주점", False, ARM_EXCLUDED_OTHER),
        ("복권명당 로또", "기타상품", False, ARM_EXCLUDED_OTHER),
        # 명품 proxy = 시계·귀금속 (C1 arm)
        ("순금나라 금은방", "시계·귀금속", False, ARM_EXCLUDED_LUXURY),
        ("스와치 매장", "시계·귀금속", False, ARM_EXCLUDED_LUXURY),
        # 비소비 서비스
        ("서울부동산공인중개사", "부동산", False, ARM_EXCLUDED_OTHER),
        ("정직한세무회계", "회계·세무", False, ARM_EXCLUDED_OTHER),
        # 적립업종 표본
        ("연세밝은치과의원", "치과", True, ARM_ELIGIBLE),
        ("전자랜드 용산점", "가전·통신", False, ARM_EXCLUDED_OTHER),  # 대형 전자
        ("동네컴퓨터수리", "가전·통신", True, ARM_ELIGIBLE),
        (None, "한식", True, ARM_ELIGIBLE),
    ]
    bad = 0
    for name, sub, exp_el, exp_arm in cases:
        el, why = is_sangsaeng_eligible(name, sub)
        arm, _ = sangsaeng_arm(name, sub)
        ok = (el == exp_el) and (arm == exp_arm)
        mark = "✔" if ok else "✘"
        if not ok:
            bad += 1
        print(f"  {mark} {str(name):22s} | {str(sub):10s} → 적립={el} arm={arm} ({why})")
    assert bad == 0, f"{bad} case(s) failed"
    print("\nALL OK")
