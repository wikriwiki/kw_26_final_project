"""Agent 노드 적재 (agents_final.json).

정규화는 POC 최소만:
- residence.dong_code 8자리 그대로 사용 (이미 NSO)
- workplace.dong_code 길이 정상화는 LIVES_AT/WORKS_AT 단계에서 처리
- job/life_stage/lifestyle은 raw 텍스트 보존 (정규화 매핑은 별도 단계)
- spending.top_categories는 dict → ratio 보존된 raw JSON 그대로
- flat 복제 속성: p_gender·p_age_group·p_income_level·p_life_stage·pr_spending_tendency·s_daily_wd·s_daily_we·b_mobility_level

agents_final.json 스키마 (sample):
{
  "agent_id": "AGT_11110515_F_20대_001",
  "personal": {age, age_group, gender, income_level, job, life_stage},
  "spending": {daily_spending_weekday, daily_spending_weekend,
               weekday_spending_level, weekend_spending_level,
               weekend_weekday_spending_ratio,
               weekday_top_categories: {cat: ratio}, weekend_top_categories: {cat: ratio}},
  "behavior": {delivery_days, shopping_days, weekday_move_km, weekend_move_km,
               home_hours_weekday, home_hours_weekend, mobility_level},
  "personality": {spending_tendency, lifestyle},
  "residence": {dong, dong_code, gu},
  "workplace": {dong, dong_code, commute_min}  (nullable)
}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session, bulk_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENTS_JSON = PROJECT_ROOT / "data" / "neo4j_load" / "agents" / "agents_final.json"


def transform(agent: dict) -> dict:
    """raw → Neo4j 속성 dict."""
    p = agent.get("personal", {}) or {}
    sp = agent.get("spending", {}) or {}
    bh = agent.get("behavior", {}) or {}
    pr = agent.get("personality", {}) or {}
    res = agent.get("residence", {}) or {}
    wk = agent.get("workplace") or {}

    # 일부 텍스트 정리
    age_group = p.get("age_group") or ""
    job_raw = p.get("job") or ""
    life_stage = p.get("life_stage") or ""
    lifestyle = pr.get("lifestyle") or ""

    return {
        "id": agent.get("agent_id"),

        # personal (nested + flat)
        "personal_age": p.get("age"),
        "personal_age_group": age_group,
        "personal_gender": p.get("gender") or "",
        "personal_job_raw": job_raw,
        "personal_life_stage_raw": life_stage,
        "personal_income_level": p.get("income_level") or "",

        # flat 복제 (인덱싱)
        "p_gender": p.get("gender") or "",
        "p_age_group": age_group,
        "p_income_level": p.get("income_level") or "",
        "p_life_stage": life_stage,

        # spending (flat 복제 일부)
        "spending_daily_wd": sp.get("daily_spending_weekday"),
        "spending_daily_we": sp.get("daily_spending_weekend"),
        "spending_level_wd": sp.get("weekday_spending_level"),
        "spending_level_we": sp.get("weekend_spending_level"),
        "spending_we_wd_ratio": sp.get("weekend_weekday_spending_ratio"),
        "spending_top_wd_json": json.dumps(sp.get("weekday_top_categories") or {}, ensure_ascii=False),
        "spending_top_we_json": json.dumps(sp.get("weekend_top_categories") or {}, ensure_ascii=False),
        "s_daily_wd": sp.get("daily_spending_weekday"),
        "s_daily_we": sp.get("daily_spending_weekend"),

        # behavior
        "behavior_delivery_days": bh.get("delivery_days"),
        "behavior_shopping_days": bh.get("shopping_days"),
        "behavior_weekday_move_km": bh.get("weekday_move_km"),
        "behavior_weekend_move_km": bh.get("weekend_move_km"),
        "behavior_home_h_wd": bh.get("home_hours_weekday"),
        "behavior_home_h_we": bh.get("home_hours_weekend"),
        "behavior_mobility_level": bh.get("mobility_level"),
        "b_mobility_level": bh.get("mobility_level"),

        # personality
        "personality_spending_tendency": pr.get("spending_tendency") or "",
        "personality_lifestyle_raw": lifestyle,
        "pr_spending_tendency": pr.get("spending_tendency") or "",

        # anchors (raw)
        "residence_dong_code_raw": res.get("dong_code") or "",
        "residence_dong_name": res.get("dong") or "",
        "residence_gu": res.get("gu") or "",
        "workplace_dong_code_raw": (wk.get("dong_code") if wk else "") or "",
        "workplace_dong_name": (wk.get("dong") if wk else "") or "",
        "workplace_commute_min": (wk.get("commute_min") if wk else None),
    }


def main():
    print(f"[load] reading {AGENTS_JSON.name} ...")
    data = json.loads(AGENTS_JSON.read_text(encoding="utf-8"))
    print(f"  agents: {len(data)}")

    rows = [transform(a) for a in data if a.get("agent_id")]
    print(f"  valid rows: {len(rows)}")

    with driver_session() as s:
        bulk_run(s, """
            UNWIND $batch AS a
            MERGE (n:Agent {id: a.id})
            SET n += a
        """, batch=rows, batch_size=2000)
        print(f"  + Agent x {len(rows)}")

    print()
    print("[done] Agents loaded.")


if __name__ == "__main__":
    main()
