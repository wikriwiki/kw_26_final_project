"""
scripts/persona/reconcile.py  (방식 C — 규칙기반 모순 detect + 봉합)
====================================================================
방식 B(conditional-graft)는 NVIDIA 사람서 소비를 파생해 대체로 안전하지만,
SES 힌트가 동네 대표분위의 ±2 보정뿐이라 잔여 모순이 남는다. 예:
  - 전문직(SES 0.9)인데 고소비 동네가 아니라 분위 4 부여 → SES↔소비 gap 큼
  - 취미에 "와인·해외여행·골프" 많은데 소비분위 1~2 → 서사↔소비 모순

이 모듈은 LLM 없이 **규칙으로 모순을 검출**하고, 선택적으로 소비분위를
SES 방향으로 **봉합(reconcile)** 한다. 봉합 후 잔여 경고는 `_match.warnings` 에
기록되어 사후 감사 가능.

방식 B와의 관계: B + 이 레이어 = 방식 C(hybrid). reconcile 끄면 순수 B.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_quant_from_cell  # noqa: E402


# 소비분위(1~10) → [0,1] 정규화 (SES 와 같은 척도로 비교)
def _level_to_unit(level: int) -> float:
    return (level - 1) / 9.0


# 고급 소비 키워드 — 서사에 많은데 저소비면 모순 신호
_LUXURY_HOBBY_KW = ("와인", "골프", "해외여행", "미식", "오마카세", "요트", "승마",
                    "갤러리", "클래식", "수입차", "명품")
# 저비용/검소 키워드 — 많은데 고소비면 모순 신호
_FRUGAL_HOBBY_KW = ("산책", "도서관", "공원", "텃밭", "등산", "동네", "재테크", "절약")


def check_consistency(persona: dict, ses: float,
                      gap_threshold: float = 0.4) -> list[str]:
    """페르소나의 SES↔소비 / 서사↔소비 정합성 검사. 경고 리스트 반환.

    gap_threshold: SES 와 소비분위(정규화) 차이 임계.
    """
    warnings: list[str] = []
    lv_wd = persona["spending"]["weekday_spending_level"]
    consume_unit = _level_to_unit(lv_wd)
    gap = consume_unit - ses

    # 1) SES ↔ 소비 gap
    if abs(gap) > gap_threshold:
        direction = "고소비_저SES" if gap > 0 else "저소비_고SES"
        warnings.append(f"ses_consume_gap:{direction}:{round(gap, 2)}")

    # 2) 서사 ↔ 소비 모순 (취미 키워드)
    hobbies = " ".join(str(h) for h in (persona.get("nvidia_persona", {}).get("hobbies") or []))
    if any(k in hobbies for k in _LUXURY_HOBBY_KW) and lv_wd <= 2:
        warnings.append("luxury_hobby_low_spend")
    if sum(k in hobbies for k in _FRUGAL_HOBBY_KW) >= 2 and lv_wd >= 9:
        warnings.append("frugal_hobby_high_spend")

    # 3) 직업 ↔ 소비 (전문직인데 극저소비, 구직/은퇴 제외)
    job = persona["personal"].get("job") or ""
    if ses >= 0.8 and lv_wd <= 2 and not any(x in job for x in ("구직", "전직", "무직", "은퇴")):
        warnings.append("high_ses_job_low_spend")

    return warnings


def reconcile_spending(persona: dict, ses: float, profile: dict, deciles: dict,
                       rng: random.Random, max_pull: int = 2,
                       gap_threshold: float = 0.4) -> dict:
    """SES↔소비 gap 이 크면 소비분위를 SES 방향으로 최대 max_pull 만큼 당김.

    당긴 뒤 daily_spending·tendency 를 재계산. 반환: 갱신된 persona(dict, 복사 아님).
    봉합 후에도 남는 경고는 _match.warnings 에 기록.
    """
    lv_wd = persona["spending"]["weekday_spending_level"]
    lv_we = persona["spending"]["weekend_spending_level"]
    target = round(ses * 9) + 1   # SES → 목표 분위 [1,10]
    gap_units = abs(_level_to_unit(lv_wd) - ses)

    pulled = False
    if gap_units > gap_threshold:
        # 목표 분위 쪽으로 max_pull 한도 내 이동
        def _pull(lv):
            if lv < target:
                return min(target, lv + max_pull)
            return max(target, lv - max_pull)
        new_wd = _pull(lv_wd)
        new_we = _pull(lv_we)
        if (new_wd, new_we) != (lv_wd, lv_we):
            quant = build_quant_from_cell(profile, deciles, rng,
                                          spending_level_override=(new_wd, new_we))
            # top_categories 는 기존(취미보정된) 유지 — 분위만 봉합
            quant["spending"]["weekday_top_categories"] = persona["spending"]["weekday_top_categories"]
            quant["spending"]["weekend_top_categories"] = persona["spending"]["weekend_top_categories"]
            persona["spending"] = quant["spending"]
            persona["behavior"] = quant["behavior"]
            persona["personality"]["spending_tendency"] = quant["tendency"]
            persona["personal"]["income_level"] = _income_from_level(new_wd)
            pulled = True

    # 봉합 후 잔여 경고 기록
    warnings = check_consistency(persona, ses, gap_threshold)
    meta = persona.setdefault("_match", {})
    meta["reconciled"] = pulled
    meta["warnings"] = warnings
    return persona


def _income_from_level(lv: int) -> str:
    return {1: "하", 2: "하", 3: "중하", 4: "중하", 5: "중", 6: "중",
            7: "중상", 8: "중상", 9: "상", 10: "상"}.get(lv, "중")
