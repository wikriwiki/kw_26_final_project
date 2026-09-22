"""
scripts/persona/llm_reconcile.py  (방식 A + LLM — 전수 LLM 일관성 감수)
====================================================================
방식 A(rank-coupling)는 **숫자(소비·이동·배달)를 BDC 통계에서 먼저 정확히 샘플**한 뒤,
같은 (구·성·연령) 셀 안에서 NVIDIA 서사를 SES 순위로 갖다 붙인다. 숫자는 정확하지만
순위 매칭만으로는 **서사 ↔ 숫자 모순**이 남는다.

이 모듈은 **모든 페르소나를 LLM이 직접 검증**한다(규칙 기반 사전탐지 없음). LLM이
"이 사람이 한 명의 실제 인간으로 자연스럽게 읽히는가"를 종합 판단하고, 모순이면
**서사(lifestyle)만** 숫자에 맞게 재서술한다. 핵심 원칙:
  · 숫자(spending·behavior)는 통계적 사실 → **절대 안 바꿈**
  · 원본 NVIDIA 서사(summary 등)는 **보존**, 융합 결과는 별도 필드
  · 판정 기준(임계값·규칙)은 프롬프트에 *적지 않는다*. LLM에는 **무엇을 보고
    판단할지(정보 차원)** 만 안내 → 룰 베이스와 구분.

방식 C(reconcile.py)와의 차이:
  - C(방식 B용): 규칙으로 숫자(소비분위)를 SES 쪽으로 당김
  - 이 모듈(방식 A용): LLM이 전수 판정 후 서사를 봉합 (숫자 = BDC 진실 보존)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable

# judge(persona) -> {"consistent": bool, "issues": str,
#                    "fused_lifestyle": str, "resolution": str}
Judge = Callable[[dict], dict]


# ---------------------------------------------------------------------------
# 전수 LLM 감수 본체
# ---------------------------------------------------------------------------
def llm_audit_persona(persona: dict, *, judge: Judge) -> dict:
    """모든 페르소나를 LLM(judge)이 검증. 모순이면 서사 봉합. 숫자는 불변.

    반환: 갱신된 persona(in-place). _match 에 llm_* 메타 기록.
    """
    verdict = judge(persona) or {}
    consistent = bool(verdict.get("consistent", True))
    fused = (verdict.get("fused_lifestyle") or "").strip()

    meta = persona.setdefault("_match", {})
    meta["llm_audited"] = True
    meta["llm_consistent"] = consistent

    if not consistent and fused:
        # 원본 서사는 보존, 융합 결과를 lifestyle + nvidia_persona 에 기록
        persona["personality"]["lifestyle"] = fused[:200]
        persona["nvidia_persona"]["fused_lifestyle"] = fused[:200]
        meta["llm_reconciled"] = True
        meta["llm_issues"] = (verdict.get("issues") or "")[:200]
        meta["llm_resolution"] = (verdict.get("resolution") or "")[:120]
    else:
        meta["llm_reconciled"] = False
        if verdict.get("issues"):
            meta["llm_issues"] = str(verdict.get("issues"))[:200]
    return persona


# ---------------------------------------------------------------------------
# 프롬프트 — 판정 기준이 아니라 '무엇을 볼지'만 안내 (룰 베이스와 구분)
# ---------------------------------------------------------------------------
_SYSTEM = (
    "너는 가상 인구 페르소나의 '일관성 감수자'다. 한 페르소나가 **한 명의 실제 사람으로 "
    "자연스럽게 읽히는지** 종합적으로 판단한다.\n"
    "\n"
    "원칙:\n"
    "1) 소비·행동 수치는 통계로 확정된 사실이다. 절대 바꾸지 마라.\n"
    "2) 수치 임계값이나 기계적 규칙으로 판정하지 말고, 한 인간으로서 전체 맥락이 말이 되는지 "
    "직관적·종합적으로 판단하라.\n"
    "3) 서사가 수치와 어긋나면, 수치는 그대로 두고 **서사(lifestyle)를 자연스럽게 재서술**해 "
    "모순을 없앤다. 다만 무리한 설정 변경 없이 납득 가능한 방향으로.\n"
    "\n"
    "다음 정보들이 한 사람 안에서 서로 모순되지 않는지 두루 살펴라 "
    "(이것은 '무엇을 볼지'에 대한 안내이며, 합격/불합격 기준이 아니다):\n"
    "- 소비 수준(분위·하루 지출액·소득 라벨)과 직업·학력이 함의하는 경제적 여건\n"
    "- 소비 수준이 서사·취미에서 드러나는 씀씀이·생활양식과 어울리는지\n"
    "- 거주지(자치구·동)와 서사가 언급·암시하는 생활 반경·지역의 일치 여부\n"
    "- 행동 지표(배달 빈도·이동 거리·재택 시간·주요 소비 카테고리)와 서사의 라이프스타일\n"
    "- 생애단계·혼인·가족 구성이 위 모든 것과 자연스럽게 맞물리는지\n"
    "\n"
    "출력은 JSON 하나만:\n"
    '{"consistent": true/false, "issues": "발견한 모순 요약(없으면 빈 문자열)", '
    '"fused_lifestyle": "수치와 모순 없는 2~3문장 한국어 라이프스타일", '
    '"resolution": "어떻게 정합화했는지 1문장(모순 없으면 빈 문자열)"}'
)


def build_audit_prompt(persona: dict) -> tuple[str, str]:
    nv = persona.get("nvidia_persona", {})
    sp = persona.get("spending", {})
    bh = persona.get("behavior", {})
    per = persona.get("personal", {})
    facts = {
        "거주": f"{persona.get('residence', {}).get('gu', '')} {persona.get('residence', {}).get('dong', '')}",
        "직업": per.get("job", ""),
        "소득라벨": per.get("income_level", ""),
        "생애단계": per.get("life_stage", ""),
        "소비분위(평일/주말)": f"{sp.get('weekday_spending_level')}/{sp.get('weekend_spending_level')}",
        "일소비(평일/주말)": f"{sp.get('daily_spending_weekday')}/{sp.get('daily_spending_weekend')}",
        "주요소비카테고리": list((sp.get("weekday_top_categories") or {}).keys())[:4],
        "배달일수(월)": bh.get("delivery_days"),
        "이동(평일km)": bh.get("weekday_move_km"),
        "재택시간(평일h)": bh.get("home_hours_weekday"),
    }
    narrative = {
        "요약": nv.get("summary", ""),
        "취미": nv.get("hobbies", []),
        "문화배경": nv.get("cultural_background", ""),
        "학력": nv.get("education_level", ""),
        "혼인/가족": f"{nv.get('marital_status', '')}/{nv.get('family_type', '')}",
    }
    user = (
        "[통계로 확정된 수치 — 변경 금지]\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
        + "\n\n[NVIDIA 인물 서사 — 수치와 모순되면 이쪽을 재서술]\n"
        + json.dumps(narrative, ensure_ascii=False, indent=2)
        + "\n\n위 수치와 서사가 한 사람으로서 자연스럽게 맞물리는지 종합 판단하고, "
          "JSON으로만 답하라."
    )
    return _SYSTEM, user


# ---------------------------------------------------------------------------
# judge 구현체 — (a) 실제 LLM, (b) 오프라인 stub
# ---------------------------------------------------------------------------
def make_llm_judge(mode: str | None = None,
                   temperature: float = 0.3, max_tokens: int = 500) -> Judge:
    """SGLang/vLLM 서버를 호출하는 judge. openai/서버는 lazy import."""
    def _judge(persona: dict) -> dict:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
        from llm_client import call_chat  # noqa: E402  (lazy: openai 의존)
        system, user = build_audit_prompt(persona)
        resp = call_chat(mode, system, user,
                         temperature=temperature, max_tokens=max_tokens)
        text = resp if isinstance(resp, str) else (resp.choices[0].message.content or "")
        return parse_verdict(text)
    return _judge


def stub_judge(persona: dict) -> dict:
    """서버 없이 동작하는 결정적 judge (테스트·오프라인 전용).

    **실제 LLM이 아니다.** 데모용으로 단순 휴리스틱(소비-SES 격차)으로 모순 여부만
    흉내 내며, 출력에 `[STUB]` 표기. 운영 판정은 make_llm_judge(LLM) 사용.
    """
    sp = persona.get("spending", {})
    nv = persona.get("nvidia_persona", {})
    ses = float(persona.get("_match", {}).get("nvidia_ses") or 0.5)
    lv = sp.get("weekday_spending_level") or 5
    gap = abs((lv - 1) / 9.0 - ses)
    consistent = gap <= 0.4

    if consistent:
        return {"consistent": True, "issues": "", "fused_lifestyle": "", "resolution": ""}

    name = (nv.get("summary", "") or "").split(" 씨", 1)[0][:6]
    job = persona.get("personal", {}).get("job", "")
    res_gu = persona.get("residence", {}).get("gu", "")
    tone = ("수입 대비 알뜰하게 지출을 관리하며 검소한 일상을 보낸다" if lv <= 3
            else "여유 있는 소비로 취미와 외식을 적극적으로 즐긴다" if lv >= 8
            else "무리하지 않는 선에서 일상 소비를 유지한다")
    hobby = (nv.get("hobbies") or ["동네 산책"])[0]
    return {
        "consistent": False,
        "issues": f"[STUB] 소비-SES 격차 {gap:.2f}",
        "fused_lifestyle": f"{name} 씨는 {res_gu}에 거주하는 {job}로(으로), {tone}. "
                           f"여가로는 {hobby}을(를) 즐긴다.",
        "resolution": "[STUB] 소비 수준에 맞춰 서사 톤 정합화 (실제 LLM 재생성 필요)",
    }


# ---------------------------------------------------------------------------
# JSON 파싱 (LLM 출력에서 첫 {...} 추출)
# ---------------------------------------------------------------------------
def parse_verdict(text: str) -> dict:
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
