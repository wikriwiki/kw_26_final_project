"""Stage 2 — Stage 1 이벤트 시퀀스 → 각 이벤트의 poi_id 결정.

입력:
  - Stage1Output (시간순 events)
  - DawnContext.persona (거주·직장 동 코드)
  - 각 이벤트별 candidate POI list (Cypher 사전 조회)

출력:
  Stage2Output — [{order, poi_id}, ...]. residence/workplace anchor는 home_poi/work_poi 그대로,
  pinned_poi가 있으면 그대로, 그 외는 LLM이 candidate 중 선택.

설계: docs/schedule_generation_plan/runtime_ontology.md §4.3
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dawn_context import (  # noqa: E402
    DawnContext, build_dawn_context,
    build_stage2_candidates,
    build_stage2_candidates_l1_dong,
    build_stage2_candidates_l1_district,
)
from stage1_intent import Stage1Output, call_stage1, _extract_json  # noqa: E402
from llm_client import call_chat as _llm_call  # noqa: E402

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise


class Stage2Pick(BaseModel):
    order: int
    poi_id: str


class Stage2Output(BaseModel):
    picks: list[Stage2Pick]


# =========================================================
# Helper: Stage 1 이벤트 → 행정동 코드 결정
# =========================================================
def resolve_dong(event_anchor: str, persona: dict, stats: dict | None = None) -> str | None:
    """zone:DONG_CODE → dong code 추출. invalid면 persona fallback + stats 카운트."""
    if event_anchor == "residence":
        return persona.get("home_dong_code")
    if event_anchor == "workplace":
        return persona.get("work_dong_code")
    if event_anchor.startswith("zone:"):
        dong = event_anchor.split(":", 1)[1].strip()
        # Neo4j Dong 노드 코드는 8자리 (행정안전부 표준 8자리, KOSIS 코드).
        # 5자리(district)로 와도 fallback 처리(persona 동코드 사용).
        if dong.isdigit() and len(dong) == 8:
            return dong
        # placeholder/invalid — persona fallback
        if stats is not None:
            stats["resolve_dong_placeholder_fallback"] = stats.get("resolve_dong_placeholder_fallback", 0) + 1
        if "work" in dong.lower():
            return persona.get("work_dong_code") or persona.get("home_dong_code")
        return persona.get("home_dong_code")
    return None


# =========================================================
# 각 이벤트별 candidate 수집 (commerce만, residence/workplace/직장/집 제외)
# =========================================================
INTERNAL_CATS = {"집", "직장"}  # residence/workplace anchor에서 사용. POI 미고정


def fetch_candidates_for_events(
    aid: str, events: list, persona: dict, k_per_event: int = 15,
    stats: dict | None = None,
) -> dict[int, list[dict]]:
    """이벤트별 candidate POI dict. key=order, value=list of candidate dicts.

    stats: fallback 카운트 dict (mutate). 아래 키가 누적됨:
      - resolve_dong_placeholder_fallback
      - cand_sub_match (정상 1차 sub_category 매칭)
      - cand_fallback_l1_dong (2차: 같은 dong, L1)
      - cand_fallback_l1_district (3차: 자치구, L1)
      - cand_all_empty (모든 fallback 실패)
    """
    out: dict[int, list[dict]] = {}
    s = stats if stats is not None else {}
    for i, ev in enumerate(events):
        if ev.category in INTERNAL_CATS:
            out[i] = []
            continue
        if ev.pinned_poi:
            out[i] = []
            continue
        sub_cat = ev.sub_category or _guess_sub_from_l1(ev.category)
        if sub_cat is None:
            out[i] = []
            continue
        dong_code = resolve_dong(ev.anchor, persona, stats=s)
        if not dong_code:
            out[i] = []
            continue

        # 1차: sub_category 정확 매칭
        cands = build_stage2_candidates(aid, dong_code, sub_cat, limit=k_per_event)
        if cands:
            s["cand_sub_match"] = s.get("cand_sub_match", 0) + 1
        else:
            # 2차: 같은 dong, L1 카테고리
            if ev.category and ev.category not in INTERNAL_CATS:
                cands = build_stage2_candidates_l1_dong(aid, dong_code, ev.category, limit=k_per_event)
                if cands:
                    s["cand_fallback_l1_dong"] = s.get("cand_fallback_l1_dong", 0) + 1
            # 3차: 자치구 + L1
            if not cands and ev.category:
                district_code = dong_code[:5] if dong_code and len(dong_code) >= 5 else None
                if district_code:
                    cands = build_stage2_candidates_l1_district(aid, district_code, ev.category, limit=k_per_event)
                    if cands:
                        s["cand_fallback_l1_district"] = s.get("cand_fallback_l1_district", 0) + 1
            if not cands:
                s["cand_all_empty"] = s.get("cand_all_empty", 0) + 1
        out[i] = cands
    return out


# L1 → 대표 sub 매핑 (sub_category 누락 시 fallback)
_L1_TO_SUB_DEFAULT = {
    "식사": "한식", "카페": "카페", "디저트": "베이커리", "주점": "일반주점",
    "편의점": "편의점", "마트": "슈퍼마켓", "미용": "미용실",
    "쇼핑": "의류", "여가": "노래방", "건강": "약국", "교육": "학원", "기타": "기타개인",
}


def _guess_sub_from_l1(l1: str) -> str | None:
    return _L1_TO_SUB_DEFAULT.get(l1)


# =========================================================
# 프롬프트 빌더
# =========================================================
SYSTEM_S2 = """당신은 Stage 1에서 결정된 의도 시퀀스를 받아, 각 외출 이벤트의 구체 POI를 결정하는 Daily Planner의 두 번째 단계입니다.
규칙:
- **각 이벤트는 자기 자신의 candidates 풀에서만 선택**. 다른 이벤트의 후보를 가져오지 마세요.
- order는 **0부터 시작하는 정수** (이벤트 0, 이벤트 1, …). 사용자 블록의 "### 이벤트 N" 의 N 그대로 사용.
- 후보 POI 중에서만 선택. 다른 ID 지어내지 마세요.
- 같은 카테고리에서 단골(known=true, visit_count>0, affinity 높음)을 살짝 선호하되, 가끔 새 곳도 시도 (탐색·다양성).
- 만족도(avg_satisfaction) ≤ 0.3 인 POI는 회피.
- 거리(km) 가까운 것 우선이지만, 단골 선호와 균형.
- residence/workplace/집/직장 이벤트는 결과에 포함하지 않음 (시스템이 자동으로 home/work POI 사용).
- pinned_poi가 있는 이벤트도 결과에 포함하지 않음.

출력 형식 (JSON만, order는 0-base):
{"picks": [
  {"order": 0, "poi_id": "C_xxxxxx"},
  {"order": 2, "poi_id": "C_yyyyyy"},
  ...
]}
"""


def _format_event_with_candidates(i: int, ev, cands: list[dict]) -> str:
    if not cands:
        return ""
    lines = [f"### 이벤트 {i} | {ev.time} | {ev.anchor} | {ev.category}/{ev.sub_category or _guess_sub_from_l1(ev.category)} | {ev.intent}"]
    for c in cands:
        known_mark = "★" if c["known"] else " "
        vc = c.get("visit_count") or 0
        sat = c.get("avg_satisfaction")
        sat_s = f"sat={sat:.2f}" if sat is not None else ""
        aff = c.get("affinity") or 0
        km = c.get("km")
        km_s = f"{km:.2f}km" if km is not None else ""
        lines.append(f"  {known_mark} {c['poi_id']} | {c.get('name') or '(이름없음)'} | {km_s} | 방문{vc}회 {sat_s} aff={aff:.2f}")
    return "\n".join(lines)


def build_stage2_prompt(events: list, cands_by_order: dict[int, list[dict]]) -> str:
    blocks = []
    for i, ev in enumerate(events):
        cs = cands_by_order.get(i) or []
        if not cs:
            continue
        blocks.append(_format_event_with_candidates(i, ev, cs))
    if not blocks:
        return "(외부 POI 결정 필요한 이벤트 없음)"
    body = "\n\n".join(blocks)
    return f"""다음 이벤트별 candidates 중에서 각 이벤트의 poi_id를 결정하세요.

{body}

각 이벤트의 order(0-base index)와 선택한 poi_id를 JSON으로 출력하세요. /no_think"""


# =========================================================
# LLM 호출 (SGLang/vLLM auto-detect via llm_client)
# =========================================================


def call_stage2(
    aid: str,
    stage1: Stage1Output,
    persona: dict,
    max_retry: int = 2,
    verbose: bool = False,
) -> tuple[Stage2Output, dict[int, list[dict]], dict]:
    """Stage 2 LLM 호출. (picks, 사용된 candidates, meta) 반환."""
    fb_stats: dict[str, int] = {}
    cands_by_order = fetch_candidates_for_events(aid, stage1.events, persona, stats=fb_stats)
    need_llm = any(cs for cs in cands_by_order.values())

    if not need_llm:
        # 외부 POI 결정 필요 없음 (전부 residence/workplace/pinned)
        return Stage2Output(picks=[]), cands_by_order, {"skipped": True, **fb_stats}

    user_block = build_stage2_prompt(stage1.events, cands_by_order)

    last_err = None
    for attempt in range(max_retry + 1):
        temp = 0.7 + 0.1 * attempt
        try:
            resp = _llm_call(
                None, SYSTEM_S2, user_block,
                temperature=temp, max_tokens=600,
            )
            raw = resp.choices[0].message.content
            if verbose:
                print(f"--- attempt {attempt} (temp={temp}) ---")
                print(raw[:600])
            json_str = _extract_json(raw)
            data = json.loads(json_str)
            parsed = Stage2Output.model_validate(data)

            # 후보 풀 안에 있는지 검증 — 반드시 해당 order의 candidates 안에서만 valid.
            # (이전 버그: valid_pois = 전체 cands flat → 다른 order의 POI도 통과되어 카테고리 매칭이 깨짐)
            import random as _random
            corrected_picks = []
            hallucinations = 0          # 보정 (해당 order의 cands에 없지만 cands는 존재)
            hallucinations_dropped = 0  # 드롭 (해당 order에 cands 자체 없음)
            order_mismatch = 0          # LLM이 다른 order의 POI를 가져옴 (보정 카운트에 포함)
            rng = _random.Random(hash(aid))
            # 전체 cands flat — order 추적용 (어느 다른 order에 속하는지 진단)
            poi_to_orders: dict[str, list[int]] = {}
            for ord_i, cs in cands_by_order.items():
                for c in cs:
                    poi_to_orders.setdefault(c["poi_id"], []).append(ord_i)
            for pick in parsed.picks:
                cands_for_this_order = cands_by_order.get(pick.order, [])
                valid_for_order = {c["poi_id"] for c in cands_for_this_order}
                if pick.poi_id in valid_for_order:
                    corrected_picks.append(pick)
                else:
                    # 다른 order의 cands에 있는 POI? (order 매핑 흐트러짐 진단)
                    if pick.poi_id in poi_to_orders:
                        order_mismatch += 1
                    if cands_for_this_order:
                        top = cands_for_this_order[:5]
                        chosen = rng.choice(top)["poi_id"]
                        corrected_picks.append(Stage2Pick(order=pick.order, poi_id=chosen))
                        hallucinations += 1
                    else:
                        # 해당 order에 candidates 자체 없음 — drop
                        hallucinations_dropped += 1
            parsed = Stage2Output(picks=corrected_picks)

            # 후처리: LLM이 답 안 한 외출 이벤트에 candidates 자동 fill
            picks_before_fill = len(parsed.picks)
            parsed = _fill_missing_picks(parsed, stage1.events, cands_by_order, aid=aid)
            missing_filled = len(parsed.picks) - picks_before_fill

            meta = {
                "attempt": attempt,
                "temp": temp,
                "tokens_in": resp.usage.prompt_tokens,
                "tokens_out": resp.usage.completion_tokens,
                "hallucinations_corrected": hallucinations,
                "hallucinations_dropped": hallucinations_dropped,
                "order_mismatch": order_mismatch,
                "missing_picks_filled": missing_filled,
                **fb_stats,
            }
            return parsed, cands_by_order, meta
        except Exception as e:
            last_err = e
            if verbose:
                print(f"[attempt {attempt}] failed: {e}")

    # 최종 retry 실패: LLM picks 빈 상태에서 candidates 첫 거 강제 fill
    fallback = _fill_missing_picks(Stage2Output(picks=[]), stage1.events, cands_by_order, aid=aid)
    if fallback.picks:
        return fallback, cands_by_order, {"fallback_only": True, "last_err": str(last_err)[:200]}
    raise RuntimeError(f"Stage2 failed after {max_retry+1} attempts: {last_err}")


def _fill_missing_picks(
    stage2: Stage2Output, stage1_events: list, cands_by_order: dict[int, list[dict]],
    aid: str = "",
) -> Stage2Output:
    """LLM이 picks에 안 만든 외출 이벤트에 candidates Top 5 중 random POI 자동 채움."""
    import random as _random
    picked_orders = {p.order for p in stage2.picks}
    new_picks = list(stage2.picks)
    rng = _random.Random(hash(aid) if aid else 42)
    for i, ev in enumerate(stage1_events):
        if i in picked_orders:
            continue
        if ev.category in INTERNAL_CATS or ev.pinned_poi:
            continue
        cs = cands_by_order.get(i) or []
        if not cs:
            continue
        top = cs[:5]
        chosen = rng.choice(top)["poi_id"]
        new_picks.append(Stage2Pick(order=i, poi_id=chosen))
    return Stage2Output(picks=new_picks)


# =========================================================
# Stage 1 + Stage 2 병합 → 최종 events
# =========================================================
def merge_to_final_events(stage1: Stage1Output, stage2: Stage2Output, persona: dict) -> list[dict]:
    """Stage 1 + Stage 2 picks → 최종 events with poi_id.

    카테고리 기준 우선 (anchor는 출발지 표시일 뿐):
      - pinned_poi 있으면 그대로
      - cat ∈ {집, 직장} (머무름) → anchor에 따라 home/work POI
      - cat ∈ 외출 카테고리 (식사·카페·편의점 등) → Stage 2 pick (commerce POI)
        - Stage 2 pick 누락 시 fallback으로 anchor POI 사용
    """
    pick_by_order = {p.order: p.poi_id for p in stage2.picks}
    out = []
    for i, ev in enumerate(stage1.events):
        poi_id = None
        if ev.pinned_poi:
            poi_id = ev.pinned_poi
        elif ev.category in INTERNAL_CATS:
            # 머무름 — anchor POI 사용
            if ev.anchor == "residence":
                poi_id = persona.get("home_poi_id")
            elif ev.anchor == "workplace":
                poi_id = persona.get("work_poi_id")
            else:  # cat=집/직장인데 zone: anchor — LLM 흔한 패턴, category 기준으로 처리
                if ev.category == "집":
                    poi_id = persona.get("home_poi_id")
                else:  # 직장
                    poi_id = persona.get("work_poi_id")
        else:
            # 외출 카테고리 — Stage 2 pick (anchor의 동에서 commerce POI 결정)
            poi_id = pick_by_order.get(i)
            if not poi_id:
                # Stage 2 pick 누락 시 anchor POI fallback
                if ev.anchor == "residence":
                    poi_id = persona.get("home_poi_id")
                elif ev.anchor == "workplace":
                    poi_id = persona.get("work_poi_id")

        out.append({
            "order": i,
            "time": ev.time,
            "duration_min": None,  # 다음 이벤트 시간으로 계산 또는 기본 60
            "anchor": ev.anchor,
            "category": ev.category,
            "sub_category": ev.sub_category,
            "intent": ev.intent,
            "poi_id": poi_id,
            "with_agents": ev.with_agents or [],
            "actual_satisfaction": None,
        })
    return out


# =========================================================
# CLI 테스트
# =========================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--aid", default="AGT_11110515_F_20대_001")
    ap.add_argument("--day", default="2026-05-01")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    today = date.fromisoformat(args.day)
    ctx = build_dawn_context(args.aid, today)
    s1, m1 = call_stage1(args.aid, today, ctx=ctx, verbose=args.verbose)
    print("\n=== Stage 1 ===")
    print(s1.model_dump_json(indent=2))
    print(f"\nmeta: {m1}")

    s2, cands, m2 = call_stage2(args.aid, s1, ctx.persona, verbose=args.verbose)
    print("\n=== Stage 2 ===")
    print(s2.model_dump_json(indent=2))
    print(f"\nmeta: {m2}")

    final = merge_to_final_events(s1, s2, ctx.persona)
    print("\n=== 최종 이벤트 ===")
    for e in final:
        print(f"  [{e['order']}] {e['time']} {e['anchor']:30} {e['category']:6} → {e['poi_id']} ({e['intent']})")
