"""계산 결과 → 사람이 읽는 해설. LLM 이 있으면 쓰고, 없으면 결정론적 서술로 간다.

해설은 **절대로 숫자를 만들지 않는다.**
1. 프롬프트에 계산 결과 JSON 만 넣고 "여기 없는 숫자는 쓰지 말라"고 지시한다.
2. 생성된 문장의 숫자를 계산 결과의 숫자 집합과 대조한다 (`llm.numeric_guard`).
3. 하나라도 근거가 없으면 그 문단은 **버리고** 결정론적 문장으로 되돌린다.
   그리고 그 사실을 보고서에 `guard_rejected` 로 남긴다.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from . import llm as llm_module
except ImportError:  # menu.py 가 scripts/report 를 sys.path 에 직접 넣는 실행 경로
    import llm as llm_module  # type: ignore

SYSTEM = """너는 정책 시뮬레이션 결과를 설명하는 한국어 분석 보고서 작성자다.

지켜야 할 규칙:
1. 입력 JSON 에 없는 숫자를 절대 쓰지 마라. 계산하지도, 어림하지도, 반올림해 새 숫자를 만들지도 마라.
2. 인과를 단정하지 마라. 이중차분은 대조군 가정이 성립할 때만 인과로 읽을 수 있다.
3. 정책을 홍보하거나 옹호하지 마라. 효과가 작거나 음수면 그대로 쓴다.
4. 마크다운 표·제목·머리기호를 쓰지 마라. 2~4문장의 평문 문단으로만 쓴다.
5. 데이터가 없거나 검증이 실패한 항목은 "확인되지 않았다"고 명시하라.
6. 존댓말(합니다체)로 쓴다."""


def _fallback(section: str, payload: dict[str, Any]) -> str:
    """LLM 없이도 보고서가 읽히도록 하는 결정론적 서술. 숫자는 계산 결과 그대로만 쓴다."""
    if section == "overview":
        meta = payload.get("meta", {})
        period = payload.get("period", {})
        return (
            f"{meta.get('run_id')} 실행의 {meta.get('day_count')}일 구간을 분석했습니다. "
            f"정책 {meta.get('policy_id') or '미지정'}의 시행일은 {period.get('policy_from') or '미지정'}이며, "
            f"사전 {len(period.get('pre') or [])}일과 사후 {len(period.get('post') or [])}일로 나누어 비교했습니다. "
            "아래의 모든 수치는 run snapshot 의 events.jsonl 과 metrics 파일에서 직접 집계한 값입니다."
        )
    if section == "did":
        did = payload.get("did")
        if not did:
            return (
                "이 실행에서는 이중차분을 계산할 수 없었습니다. "
                f"사유: {payload.get('period', {}).get('reason') or '사전 또는 사후 기간이 부족합니다.'}"
            )
        return (
            "정책 대상 업종과 비대상 업종을 각각 처치군과 대조군으로 두고, "
            "대조군의 사전 대비 사후 성장률을 처치군에 적용해 반사실값을 만들었습니다. "
            "실제 사후값과 반사실값의 차이가 이중차분 추정치입니다. "
            "단순 전후비교에는 시장 전체의 추세가 섞여 있으므로 두 값을 함께 보아야 합니다."
        )
    if section == "categories":
        return (
            "업종별로 사전 일평균과 사후 일평균을 비교했습니다. "
            "증감률만으로는 시장 전체의 변화와 정책 효과를 구분할 수 없으므로, "
            "같은 표의 이중차분 열을 함께 확인해야 합니다."
        )
    if section == "overlay":
        return (
            "시행 전 구간과 시행 후 구간을 같은 길이로 잘라 하나의 축에 겹쳤습니다. "
            "두 곡선 사이의 면적이 소비가 달라진 크기이며, 색은 방향을 나타냅니다."
        )
    if section == "consistency":
        checks = payload.get("consistency", {})
        return (
            f"섹션 사이의 항등식 {checks.get('counts', {}).get('total', 0)}건을 다시 계산해 대조했습니다. "
            f"{checks.get('verdict', '')}"
        )
    return ""


def _prompt(section: str, payload: dict[str, Any]) -> str:
    guides = {
        "overview": "이 보고서가 무엇을 분석했고 독자가 무엇을 먼저 봐야 하는지 요약하라.",
        "did": "이중차분 결과가 무엇을 뜻하는지, 단순 전후비교와 어떻게 다른지 설명하라. 사전추세 검증 결과가 있으면 그 신뢰도도 언급하라.",
        "categories": "어떤 업종에서 금액이 늘고 줄었는지, 정책 대상 업종과 비대상 업종의 차이를 중심으로 설명하라.",
        "overlay": "시행 전후 곡선이 어떻게 달라졌는지, 어느 구간에서 차이가 벌어졌는지 설명하라.",
        "consistency": "일관성 검사 결과를 근거로 이 보고서의 수치를 신뢰할 수 있는지 판단하라.",
    }
    return (
        f"{guides.get(section, '아래 계산 결과를 설명하라.')}\n\n"
        "계산 결과 JSON (여기 있는 숫자만 사용할 것):\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}\n```"
    )


def _slim(section: str, bundle: dict[str, Any], consistency: dict[str, Any]) -> dict[str, Any]:
    """프롬프트에 넣을 최소 payload. 원본 전체를 넣으면 토큰만 태우고 정확도가 떨어진다."""
    meta = {
        key: bundle["meta"].get(key)
        for key in ("run_id", "policy_id", "policy_name", "policy_type", "day_count", "policy_from_used")
    }
    if section == "overview":
        return {
            "meta": meta,
            "period": {k: bundle["period"].get(k) for k in ("policy_from", "usable", "reason")},
            "period_days": {"pre": len(bundle["period"]["pre"]), "post": len(bundle["period"]["post"])},
            "totals": bundle["totals"],
            "mix": bundle["mix"],
            "did_absolute": (bundle.get("did") or {}).get("did_absolute"),
            "did_pct": (bundle.get("did") or {}).get("did_pct_of_counterfactual"),
            "consistency": consistency.get("counts"),
        }
    if section == "did":
        return {
            "meta": meta,
            "did": bundle.get("did"),
            "targets": bundle.get("targets"),
            "control_categories": bundle.get("control_categories"),
            "event_study": {
                "available": bundle["event_study"].get("available"),
                "reason": bundle["event_study"].get("reason"),
                "pre_points": [
                    p for p in (bundle["event_study"].get("points") or []) if p.get("rel_day", 0) < 0
                ][-5:],
            },
        }
    if section == "categories":
        return {
            "meta": meta,
            "top_increase": bundle["did_by_category"][:5],
            "top_decrease": bundle["did_by_category"][-5:],
            "categories": bundle["categories"][:10],
        }
    if section == "overlay":
        return {
            "meta": meta,
            "overlay": {
                "window_days": bundle["overlay"].get("window_days"),
                "overall": bundle["overlay"].get("overall"),
                "available": bundle["overlay"].get("available"),
                "reason": bundle["overlay"].get("reason"),
            },
        }
    if section == "consistency":
        return {"meta": meta, "consistency": consistency}
    return {"meta": meta}


def narrate_report(
    bundle: dict[str, Any],
    consistency: dict[str, Any],
    *,
    sections: tuple[str, ...] = ("overview", "did", "categories", "overlay", "consistency"),
    enabled: bool = True,
) -> dict[str, Any]:
    """섹션별 해설을 만든다. 결과에는 항상 출처(LLM/결정론)와 검증 상태가 붙는다."""
    status = llm_module.provider_status()
    narration: dict[str, Any] = {
        "llm": status,
        "used_llm": False,
        "sections": {},
        "guard_rejected": [],
        "errors": [],
    }
    allowed = llm_module.allowed_number_set(
        {
            "totals": bundle.get("totals"),
            "mix": bundle.get("mix"),
            "did": bundle.get("did"),
            "did_by_category": bundle.get("did_by_category"),
            "categories": bundle.get("categories"),
            "daily": bundle.get("daily"),
            "overlay": bundle.get("overlay"),
            "deciles": bundle.get("deciles"),
            "event_study": bundle.get("event_study"),
            "consistency": consistency,
            "meta_days": bundle.get("meta", {}).get("days"),
        }
    )
    for section in sections:
        payload = _slim(section, bundle, consistency)
        fallback = _fallback(section, {**bundle, "consistency": consistency})
        entry = {"text": fallback, "source": "deterministic", "model": None, "guard": None}
        if enabled and status.get("configured"):
            result = llm_module.complete(SYSTEM, _prompt(section, payload), load_env=False)
            if result.ok:
                ok, offenders = llm_module.numeric_guard(result.text, allowed)
                if ok:
                    entry = {
                        "text": result.text,
                        "source": f"llm:{result.provider}",
                        "model": result.model,
                        "guard": "passed",
                        "latency_ms": result.latency_ms,
                        "usage": result.usage,
                    }
                    narration["used_llm"] = True
                else:
                    entry["guard"] = "rejected"
                    entry["rejected_numbers"] = offenders[:12]
                    narration["guard_rejected"].append(section)
            else:
                entry["guard"] = "llm_error"
                narration["errors"].append({"section": section, "error": result.error})
        narration["sections"][section] = entry
    return narration
