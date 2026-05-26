"""
scripts/persona/llm_reconcile.py  (방식 A + LLM — 모순 보정 레이어)
====================================================================
방식 A(rank-coupling)는 **숫자(소비·이동·배달)를 BDC 통계에서 먼저 정확히 샘플**한 뒤,
같은 (구·성·연령) 셀 안에서 NVIDIA 서사를 SES 순위로 갖다 붙인다. 숫자는 정확하지만
순위 매칭만으로는 **서사 ↔ 숫자 모순**이 남는다. 예:
  - SES 높은 전문직 서사인데 그 셀 소비분위가 낮음 → "고SES 서사 ↔ 저소비"
  - 서사가 '명품·미식'인데 소비분위 2 → "고급 취미 ↔ 저소비"
  - sex_age 폴백 매칭이라 서사가 다른 자치구를 언급 → "거주지 ↔ 서사 위치 충돌"

이 모듈은:
  1) **규칙으로 모순을 싸게 사전탐지** (LLM은 모순 있는 페르소나에만 발동 → 비용 절감)
  2) **LLM이 서사를 숫자에 맞게 재서술**(봉합). 핵심 원칙:
       · 숫자(spending·behavior)는 통계적 사실 → **절대 안 바꿈**
       · 인물 서사(lifestyle)만 숫자와 모순 없게 자연스럽게 다시 씀
       · 원본 NVIDIA 서사(nvidia_persona.summary 등)는 **보존**, 융합 결과는 별도 필드
  3) LLM 없이도 동작하도록 **오프라인 stub fixer** 제공 (테스트·서버 부재 시)

방식 C(reconcile.py)와의 차이:
  - C(방식 B용): 숫자(소비분위)를 SES 쪽으로 **당김** (규칙 기반)
  - 이 모듈(방식 A용): 숫자는 그대로 두고 **서사를 LLM이 봉합** (숫자 = BDC 진실 보존)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconcile import check_consistency  # noqa: E402  (규칙 기반 SES/취미/직업 모순)

# fixer(persona, warnings) -> {"fused_lifestyle": str, "resolution": str, "consistent": bool}
Fixer = Callable[[dict, list[str]], dict]

SEOUL_GU = (
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구",
    "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구",
    "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구",
)


# ---------------------------------------------------------------------------
# 1) 모순 사전탐지 (싼 규칙 — LLM 호출 게이트)
# ---------------------------------------------------------------------------
def detect_contradictions(persona: dict, ses: float | None = None,
                          gap_threshold: float = 0.4) -> list[str]:
    """SES/취미/직업 모순(reconcile 규칙) + 거주지↔서사 위치 충돌 탐지."""
    if ses is None:
        ses = float(persona.get("_match", {}).get("nvidia_ses") or 0.5)
    warnings = list(check_consistency(persona, ses, gap_threshold))

    # A 특유: sex_age 폴백 매칭이면 서사가 다른 자치구를 언급할 수 있음
    summary = str(persona.get("nvidia_persona", {}).get("summary") or "")
    res_gu = persona.get("residence", {}).get("gu") or ""
    for gu in SEOUL_GU:
        if gu != res_gu and gu in summary:
            warnings.append(f"location_conflict:{gu}≠{res_gu}")
            break
    return warnings


# ---------------------------------------------------------------------------
# 2) LLM 보정 본체
# ---------------------------------------------------------------------------
def llm_reconcile_persona(persona: dict, *, fixer: Fixer,
                          ses: float | None = None,
                          gap_threshold: float = 0.4) -> dict:
    """모순 있으면 fixer(LLM 또는 stub)로 서사를 봉합. 숫자는 불변.

    반환: 갱신된 persona(in-place). _match 에 llm_* 메타 기록.
    """
    if ses is None:
        ses = float(persona.get("_match", {}).get("nvidia_ses") or 0.5)
    warnings = detect_contradictions(persona, ses, gap_threshold)
    meta = persona.setdefault("_match", {})
    meta["llm_contradictions"] = warnings

    if not warnings:
        meta["llm_reconciled"] = False
        return persona

    fix = fixer(persona, warnings) or {}
    fused = (fix.get("fused_lifestyle") or "").strip()
    if fused:
        # 원본 서사는 보존하고, 융합 결과를 lifestyle + nvidia_persona 에 기록
        persona["personality"]["lifestyle"] = fused[:200]
        persona["nvidia_persona"]["fused_lifestyle"] = fused[:200]
        meta["llm_reconciled"] = True
    else:
        meta["llm_reconciled"] = False
    meta["llm_resolution"] = (fix.get("resolution") or "")[:120]
    return persona


# ---------------------------------------------------------------------------
# 3) 프롬프트 빌더 (LLM-입력 필드만 사용 — nvidia_reserved 는 제외)
# ---------------------------------------------------------------------------
_SYSTEM = (
    "너는 한국 가상 인구 시뮬레이션의 '페르소나 정합성 보정기'다.\n"
    "소비·행동 수치는 통계로 확정된 사실이므로 절대 바꾸지 마라.\n"
    "인물 서사가 그 수치와 모순되면, 수치와 어긋나지 않게 서사를 자연스럽게 재서술하라.\n"
    "출력은 JSON 한 개만. 형식: "
    '{"fused_lifestyle": "2~3문장 한국어", "resolution": "무엇을 어떻게 정합화했는지 1문장", "consistent": true}'
)


def build_reconcile_prompt(persona: dict, warnings: list[str]) -> tuple[str, str]:
    nv = persona.get("nvidia_persona", {})
    sp = persona.get("spending", {})
    bh = persona.get("behavior", {})
    per = persona.get("personal", {})
    facts = {
        "거주": f"{persona.get('residence', {}).get('gu', '')} {persona.get('residence', {}).get('dong', '')}",
        "직업": per.get("job", ""),
        "소득라벨": per.get("income_level", ""),
        "소비분위(평일/주말)": f"{sp.get('weekday_spending_level')}/{sp.get('weekend_spending_level')}",
        "일소비(평일)": sp.get("daily_spending_weekday"),
        "주요소비카테고리": list((sp.get("weekday_top_categories") or {}).keys())[:4],
        "배달일수(월)": bh.get("delivery_days"),
        "이동(평일km)": bh.get("weekday_move_km"),
        "재택시간(평일h)": bh.get("home_hours_weekday"),
    }
    narrative = {
        "요약": nv.get("summary", ""),
        "취미": nv.get("hobbies", []),
        "문화배경": nv.get("cultural_background", ""),
        "혼인/가족": f"{nv.get('marital_status', '')}/{nv.get('family_type', '')}",
    }
    user = (
        "[통계로 확정된 수치 — 변경 금지]\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
        + "\n\n[NVIDIA 인물 서사 — 수치와 모순되면 이쪽을 재서술]\n"
        + json.dumps(narrative, ensure_ascii=False, indent=2)
        + "\n\n[탐지된 모순]\n- " + "\n- ".join(warnings)
        + "\n\n위 수치를 그대로 받아들이면서, 서사와 수치가 모순 없이 한 사람으로 읽히도록 "
          "fused_lifestyle 을 작성하라. 예: 고소득 서사인데 소비가 낮으면 '검소·저축 지향' 등으로 "
          "납득되게. 거주지 충돌이 있으면 현재 거주지 기준으로 서술. JSON만 출력."
    )
    return _SYSTEM, user


# ---------------------------------------------------------------------------
# 4) fixer 구현체 — (a) 실제 LLM, (b) 오프라인 stub
# ---------------------------------------------------------------------------
def make_llm_fixer(mode: str | None = None,
                   temperature: float = 0.4, max_tokens: int = 400) -> Fixer:
    """SGLang/vLLM 서버를 호출하는 fixer. openai/서버는 lazy import."""
    def _fixer(persona: dict, warnings: list[str]) -> dict:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
        from llm_client import call_chat  # noqa: E402  (lazy: openai 의존)
        system, user = build_reconcile_prompt(persona, warnings)
        resp = call_chat(mode, system, user,
                         temperature=temperature, max_tokens=max_tokens)
        text = resp if isinstance(resp, str) else (resp.choices[0].message.content or "")
        return parse_fix(text)
    return _fixer


def stub_fixer(persona: dict, warnings: list[str]) -> dict:
    """서버 없이 동작하는 결정적 fixer. 규칙 기반으로 그럴듯한 융합 서술 생성.

    실제 LLM이 아니므로 출력은 '자리표시자' 성격 — 서버 붙으면 make_llm_fixer 사용.
    """
    nv = persona.get("nvidia_persona", {})
    sp = persona.get("spending", {})
    name = (nv.get("summary", "") or "").split(" 씨", 1)[0][:6]
    lv = sp.get("weekday_spending_level") or 5
    job = persona.get("personal", {}).get("job", "")
    res_gu = persona.get("residence", {}).get("gu", "")

    if lv <= 3:
        tone = "수입 대비 알뜰하게 지출을 관리하며 검소한 일상을 보낸다"
    elif lv >= 8:
        tone = "여유 있는 소비로 취미와 외식을 적극적으로 즐긴다"
    else:
        tone = "무리하지 않는 선에서 일상 소비를 유지한다"

    hobby = (nv.get("hobbies") or ["동네 산책"])[0]
    fused = f"{name} 씨는 {res_gu}에 거주하는 {job}로(으로), {tone}. 여가로는 {hobby}을(를) 즐긴다."
    res = "[STUB] 규칙 기반 자리표시자 — 실제 LLM 서버로 재생성 필요"
    if any(w.startswith("location_conflict") for w in warnings):
        res = "[STUB] 거주지를 현재 통계 기준으로 정합화 (실제 LLM 재생성 필요)"
    return {"fused_lifestyle": fused, "resolution": res, "consistent": True}


# ---------------------------------------------------------------------------
# JSON 파싱 (LLM 출력에서 첫 {...} 추출)
# ---------------------------------------------------------------------------
def parse_fix(text: str) -> dict:
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}
