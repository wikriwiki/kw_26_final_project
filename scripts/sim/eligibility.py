"""정책 사용처·적격 판정 — 규칙 명세(JSON)를 읽는 범용 평가기.

정책마다 판정 코드를 새로 쓰면(`sangsaeng_eligibility.py` 처럼) 소비 프롬프트를
일반화해도 **배관에서 1:1 결합이 되살아난다.** 홀드아웃 정책을 코드 수정 없이
붙일 수 있어야 일반화 주장에 실체가 생긴다.

## 두 가지 모드

    exclude  기본 적격. 목록에 걸리면 제외.      (P010 소비쿠폰, P012 상생)
    include  기본 부적격. 목록에 들어야 적격.    (8대 소비쿠폰 같은 업종 한정 할인)

## 판정 우선순위 (exclude 모드)

    ① 업종코드 제외 목록      — 가장 신뢰도 높음
    ② 상호명 규칙             — 코드가 못 닿는 영역(대형업태 브랜드)
    ③ 코드가 있고 ①②에 안 걸림 → 적격 확정
    ④ 코드 미확보 → 세분류(sub) 규칙 fallback

이 순서는 `sangsaeng_eligibility.sangsaeng_arm()` 의 검증된 순서를 그대로 옮긴 것이다.
`--verify` 로 두 구현의 판정이 전건 일치하는지 대조할 수 있다.

## 규칙 명세 형태

    {
      "mode": "exclude",
      "exclude": {
        "codes":  {"luxury": ["G21701"], "vice": [...], "nonconsumption": [...]},
        "subs":   {"luxury": ["시계·귀금속"], "nonconsumption": ["부동산", ...]},
        "name_regex": "이마트|홈플러스|..."
      },
      "include": {"codes": [...], "subs": [...], "l1s": [...]}
    }

`arm` 은 제외 사유를 남기는 이름이다. 채점에서 대조군을 가를 때 쓴다.
"""
from __future__ import annotations

import re
from typing import Any

ARM_ELIGIBLE = "eligible"
ARM_EXCLUDED_LUXURY = "excluded_luxury"
ARM_EXCLUDED_VICE = "excluded_vice"
ARM_EXCLUDED_NONCONSUMPTION = "excluded_nonconsumption"
ARM_EXCLUDED_OTHER = "excluded_other"
ARM_EXCLUDED_SECTOR = "excluded_sector"     # include 모드에서 대상 업종이 아님

EXCLUDED_ARMS = (
    ARM_EXCLUDED_LUXURY, ARM_EXCLUDED_VICE,
    ARM_EXCLUDED_NONCONSUMPTION, ARM_EXCLUDED_OTHER, ARM_EXCLUDED_SECTOR,
)

_ARM_OF_KIND = {
    "luxury": ARM_EXCLUDED_LUXURY,
    "vice": ARM_EXCLUDED_VICE,
    "nonconsumption": ARM_EXCLUDED_NONCONSUMPTION,
    "other": ARM_EXCLUDED_OTHER,
}


class Rules:
    """규칙 명세를 판정 가능한 형태로 굳혀 둔다. 정책당 한 번만 만든다."""

    __slots__ = ("mode", "ex_codes", "ex_subs", "name_re",
                 "in_codes", "in_subs", "in_l1s")

    def __init__(self, spec: dict[str, Any] | None):
        s = spec or {}
        self.mode = (s.get("mode") or "exclude").strip()

        ex = s.get("exclude") or {}
        self.ex_codes: dict[str, str] = {}
        for kind, codes in (ex.get("codes") or {}).items():
            arm = _ARM_OF_KIND.get(kind, ARM_EXCLUDED_OTHER)
            for c in codes or ():
                self.ex_codes[str(c).strip().upper()] = arm
        self.ex_subs: dict[str, str] = {}
        for kind, subs in (ex.get("subs") or {}).items():
            arm = _ARM_OF_KIND.get(kind, ARM_EXCLUDED_OTHER)
            for x in subs or ():
                self.ex_subs[str(x).strip()] = arm
        pat = ex.get("name_regex")
        self.name_re = re.compile(pat) if pat else None

        inc = s.get("include") or {}
        self.in_codes = {str(c).strip().upper() for c in (inc.get("codes") or ())}
        self.in_subs = {str(x).strip() for x in (inc.get("subs") or ())}
        self.in_l1s = {str(x).strip() for x in (inc.get("l1s") or ())}

    # -----------------------------------------------------
    def arm(self, name: str | None, sub: str | None,
            l1: str | None = None, upjong_l3: str | None = None) -> tuple[str, str]:
        """(arm, 근거코드). 우선순위는 모듈 docstring 참조."""
        n = (name or "").strip()
        c = (upjong_l3 or "").strip().upper()
        s = (sub or "").strip()
        p = (l1 or "").strip()

        if self.mode == "include":
            # 대상 업종에 들어야만 적격. 코드 > 세분류 > L1 순으로 본다.
            if c and c in self.in_codes:
                return ARM_ELIGIBLE, "code_included"
            if s and s in self.in_subs:
                return ARM_ELIGIBLE, "sub_included"
            if p and p in self.in_l1s:
                return ARM_ELIGIBLE, "l1_included"
            return ARM_EXCLUDED_SECTOR, "not_in_sector"

        # ① 업종코드 제외 목록
        if c:
            hit = self.ex_codes.get(c)
            if hit:
                return hit, "code_" + _kind_of(hit)
        # ② 상호명 규칙 — 코드가 못 닿는 영역
        if n and self.name_re is not None and self.name_re.search(n):
            return ARM_EXCLUDED_OTHER, "brand_large"
        # ③ 코드가 있고 ①②에 안 걸림 → 적격 확정
        if c:
            return ARM_ELIGIBLE, "code_eligible"
        # ④ 코드 미확보 → 세분류 fallback
        if s:
            hit = self.ex_subs.get(s)
            if hit:
                return hit, "sub_" + _kind_of(hit)
        return ARM_ELIGIBLE, "ok"

    def eligible(self, name: str | None, sub: str | None,
                 l1: str | None = None, upjong_l3: str | None = None) -> tuple[bool, str]:
        a, why = self.arm(name, sub, l1, upjong_l3)
        return (a == ARM_ELIGIBLE), why


def _kind_of(arm: str) -> str:
    for k, v in _ARM_OF_KIND.items():
        if v == arm:
            return k
    return "other"


# =========================================================
# --verify : sangsaeng_eligibility 와 전건 일치 확인
# =========================================================
def _p012_spec() -> dict:
    """현행 P012 판정을 규칙 명세로 옮긴 것. 동작이 같아야 한다."""
    import sangsaeng_eligibility as S
    return {
        "mode": "exclude",
        "exclude": {
            "codes": {
                "luxury": sorted(S._CODE_LUXURY),
                "vice": sorted(S._CODE_VICE),
                "nonconsumption": sorted(S._CODE_NONCONSUMPTION),
            },
            "subs": {
                "luxury": sorted(S._SUB_LUXURY),
                "nonconsumption": sorted(S._SUB_NONCONSUMPTION),
            },
            "name_regex": S._NAME_LARGE.pattern,
        },
    }


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import sangsaeng_eligibility as S

    spec = _p012_spec()
    rules = Rules(spec)

    # 매핑표 전수 + 케이스 목록으로 두 구현을 대조한다.
    cases: list[tuple] = []
    for c in sorted(S._CODE_LUXURY | S._CODE_VICE | S._CODE_NONCONSUMPTION):
        cases.append((None, None, None, c))
    try:
        mp = json.loads(S._MAPPING_PATH.read_text(encoding="utf-8"))
        for code, v in (mp.get("mapping") or mp).items():
            if isinstance(v, dict):
                cases.append((None, v.get("sub"), v.get("cat"), code))
    except Exception as e:
        print(f"  (매핑표 미사용: {e})")
    for row in S._CASES:
        cases.append((row[1], row[2], None, row[0]))
    for sub in sorted(S._SUB_LUXURY | S._SUB_NONCONSUMPTION):
        cases.append((None, sub, None, None))
    for nm in ("이마트 성수점", "홈플러스 강서", "동네컴퓨터수리", "성인피아노학원",
               "LG하이마트", "전자랜드 용산점", "그냥가게"):
        cases.append((nm, None, None, None))

    same = diff = 0
    ex = []
    for name, sub, l1, code in cases:
        a1, w1 = S.sangsaeng_arm(name, sub, l1, upjong_l3=code)
        a2, w2 = rules.arm(name, sub, l1, upjong_l3=code)
        if a1 == a2:
            same += 1
        else:
            diff += 1
            if len(ex) < 5:
                ex.append((name, sub, code, f"기존={a1}({w1})", f"신규={a2}({w2})"))
    print(f"대조 {len(cases):,}건 | 일치 {same:,} | 불일치 {diff:,}")
    for e in ex:
        print("   ", e)
    print()
    print("판정 결과:", "동일 — 교체 가능" if diff == 0 else "다름 — 교체 금지")
    sys.exit(0 if diff == 0 else 1)
