"""Dawn 컨텍스트 빌더 — 7종 고정 Cypher → 텍스트 블록.

매일 자정 각 agent에 대해 호출. 반환 dict는 Stage 1 프롬프트에 그대로 주입.

설계 출처: docs/schedule_generation_plan/runtime_ontology.md §4
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# Windows console에서 한글 + 유니코드 다이아크리틱(em-dash 등) 출력 가능하게
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))
from _common import driver_session


# =========================================================
# Persona — 60일 고정. 한 번 빌드해서 캐시 가능 (Stage 1 prefix cache).
# =========================================================
PERSONA_CYPHER = """
MATCH (a:Agent {id: $aid})-[:LIVES_AT]->(home:POI)-[:IN_DONG]->(hd:Dong)
OPTIONAL MATCH (a)-[wr:WORKS_AT]->(work:POI)-[:IN_DONG]->(wd:Dong)
RETURN
  a.id AS id,
  a.p_age_group AS age_group,
  a.p_gender AS gender,
  a.p_income_level AS income,
  a.p_life_stage AS life_stage,
  a.personal_job_raw AS job,
  a.pr_spending_tendency AS tendency,
  a.personality_lifestyle_raw AS lifestyle,
  // NVIDIA Nemotron 봉합 페르소나 필드 (load_fusion_to_neo4j.py 로 적재됨, 미적재 시 NULL)
  a.nvidia_summary AS nv_summary,
  a.nvidia_hobbies AS nv_hobbies,
  a.nvidia_cultural_background AS nv_cultural,
  a.nvidia_education_level AS nv_education,
  a.nvidia_marital_status AS nv_marital,
  a.nvidia_family_type AS nv_family,
  a.nvidia_career_goals AS nv_career,
  a.nvidia_skills AS nv_skills,
  a.s_daily_wd AS daily_wd,
  a.s_daily_we AS daily_we,
  a.spending_we_wd_ratio AS we_wd_ratio,
  a.spending_top_wd_json AS top_wd_json,
  a.spending_top_we_json AS top_we_json,
  a.behavior_delivery_days AS delivery_days,
  a.behavior_home_h_wd AS home_h_wd,
  a.behavior_home_h_we AS home_h_we,
  a.b_mobility_level AS mobility,
  hd.code AS home_dong_code, hd.name AS home_dong, home.id AS home_poi_id, home.name AS home_poi,
  wd.code AS work_dong_code, wd.name AS work_dong, work.id AS work_poi_id, work.name AS work_poi,
  wr.commute_min AS commute_min
"""


# =========================================================
# State — 어제 잔액·에너지·mood·fatigue·정책 라이프사이클
# =========================================================
STATE_CYPHER = """
MATCH (a:Agent {id: $aid})-[:HAS_STATE {day: $yesterday}]->(s:State)
RETURN s.balance AS balance, s.energy AS energy, s.mood AS mood,
       s.fatigue AS fatigue, s.yesterday_satisfaction AS yest_sat,
       s.month_spent AS month_spent, s.policy_lifecycle AS policy_lc,
       s.policy_used AS policy_used
"""


# =========================================================
# Memory Top-N (30일, importance × exp(-days/14) 정렬)
# Day 0에는 initial Memory 없으므로 Day 1엔 빈 결과 가능
# =========================================================
MEMORY_CYPHER = """
MATCH (a:Agent {id: $aid})-[:REMEMBERS]->(m:Memory)
WHERE m.day >= $today - duration({days: 30})
OPTIONAL MATCH (m)-[:ABOUT_POI]->(p:POI)-[:IN_CATEGORY]->(c:Category)
WITH m, p, c,
     duration.inDays($today, m.day).days AS days_ago,
     m.importance * exp(-toFloat(duration.inDays($today, m.day).days) / 14.0) AS score
RETURN m.type AS type, m.day AS day, m.importance AS importance,
       coalesce(m.satisfaction, 0.0) AS satisfaction,
       coalesce(m.summary, '') AS summary,
       m.source AS source,
       m.topic_type AS topic_type,
       m.topic_value AS topic_value,
       p.name AS poi_name, c.name AS category, days_ago
ORDER BY score DESC LIMIT $top_n
"""


# =========================================================
# 약속 큐 — should_inject=true AND day + target_day_offset == today
# (노션 §6 — 약속만 Plan에 자동 주입)
# Conversation 노드에 저장된 initiator_id/recipient_id로 상대 식별,
# meeting_location_hint는 자유 문자열(POI에 못 붙은 케이스 포함).
# =========================================================
APPOINTMENT_CYPHER = """
MATCH (a:Agent {id: $aid})-[part:PARTICIPATES_IN]->(c:Conversation {intent:'약속'})
WHERE c.should_inject = true
  AND c.target_day_offset IS NOT NULL
  AND date(c.day) + duration({days: c.target_day_offset}) = date($today)
OPTIONAL MATCH (c)-[:MENTIONS_POI]->(meet:POI)
WITH c, part, meet,
     CASE WHEN part.role = 'initiator' THEN c.recipient_id ELSE c.initiator_id END AS counterpart_id
RETURN c.id AS conv_id,
       c.target_time AS target_time,
       c.meeting_location_hint AS meeting_location_hint,
       meet.id AS meeting_poi_id, meet.name AS meeting_poi_name,
       collect(DISTINCT counterpart_id) AS with_agents
"""


# =========================================================
# 활성 정책/이슈 — 거주·직장 동 또는 자치구에 applied_to
# 정책 단위로 1행씩 반환 (다중 자치구 적용 정책은 districts 리스트로 묶음)
# =========================================================
POLICY_CYPHER = """
MATCH (a:Agent {id: $aid})-[:LIVES_AT|WORKS_AT]->(:POI)-[:IN_DONG]->(d:Dong)<-[:HAS_DONG]-(dist:District)
WITH a, collect(DISTINCT d) AS my_dongs, collect(DISTINCT dist) AS my_dists

// 내 동·자치구 중 한 곳이라도 적용되는 정책만 (DISTINCT로 정책 단위 reduce)
MATCH (pol:Policy)-[:applied_to]->(target)
WHERE $today >= pol.effective_from AND $today <= pol.effective_until
  AND (target IN my_dongs OR target IN my_dists)
WITH DISTINCT pol

// 정책 단위로 다시 적용 지역·대상 카테고리 펼침
OPTIONAL MATCH (pol)-[:applied_to]->(reg)
WITH pol,
     collect(DISTINCT coalesce(reg.name, '')) AS regions,
     collect(DISTINCT coalesce(reg.code, '')) AS region_codes
OPTIONAL MATCH (pol)-[:targets]->(cat:Category)
WITH pol, regions, region_codes, collect(DISTINCT cat.parent) AS target_l1s

RETURN pol.id AS id, pol.name AS name, pol.type AS type,
       pol.description AS description,
       pol.benefit_rate AS rate, pol.cap_per_agent AS cap,
       pol.effective_from AS from_, pol.effective_until AS until_,
       regions, region_codes, target_l1s
"""


# =========================================================
# 지인 풀 — KNOWS strength 정렬
# =========================================================
SOCIAL_CYPHER = """
MATCH (a:Agent {id: $aid})-[k:KNOWS]->(b:Agent)
RETURN b.id AS friend_id, k.strength AS strength, k.relation AS relation,
       b.p_age_group AS age, b.p_gender AS gender, b.personality_lifestyle_raw AS lifestyle
ORDER BY k.strength DESC LIMIT $top_n
"""


# =========================================================
# KNOWS_POI 카테고리별 요약 — Stage 1 참고용 (Stage 2 candidate 풀러 traversal은 별도)
# =========================================================
KNOWS_POI_SUMMARY_CYPHER = """
MATCH (a:Agent {id: $aid})-[kp:KNOWS_POI]->(p:POI)-[:IN_CATEGORY]->(c:Category)
RETURN c.parent AS L1, c.name AS sub,
       count(p) AS n,
       sum(CASE WHEN kp.visit_count > 0 THEN 1 ELSE 0 END) AS n_visited
ORDER BY n DESC LIMIT 30
"""


# =========================================================
# Stage 2 candidate — (행정동, 카테고리)별 POI Top-K
# Stage 1 출력의 각 이벤트마다 별도 호출
# =========================================================
STAGE2_CANDIDATE_CYPHER = """
MATCH (p:POI {type:'commerce'})-[:IN_DONG]->(:Dong {code: $dong_code})
MATCH (p)-[:IN_CATEGORY]->(c:Category {name: $sub_category})
OPTIONAL MATCH (a:Agent {id: $aid})-[kp:KNOWS_POI]->(p)
OPTIONAL MATCH (a)-[:LIVES_AT|WORKS_AT]->(anchor:POI)
WITH p, c, kp, anchor,
     CASE WHEN anchor IS NOT NULL AND p.lon IS NOT NULL THEN
       point.distance(point({longitude: p.lon, latitude: p.lat}),
                      point({longitude: anchor.lon, latitude: anchor.lat})) / 1000.0
     ELSE NULL END AS km
RETURN p.id AS poi_id, p.name AS name,
       (kp IS NOT NULL) AS known,
       coalesce(kp.visit_count, 0) AS visit_count,
       kp.avg_satisfaction AS avg_satisfaction,
       coalesce(kp.affinity, 0.0) AS affinity,
       kp.last_visit AS last_visit,
       size(coalesce(kp.recent_visit_dates, [])) AS v30,
       c.recovery_tau_days AS cat_tau,
       c.desire_drop AS cat_drop,
       c.saturation_n AS cat_sat_n,
       kp.source AS source,
       km
ORDER BY known DESC, km ASC LIMIT $limit
"""

# Fallback: sub_category 매칭 실패 시 L1 단위로 같은 dong에서 fetch
STAGE2_FALLBACK_L1_DONG_CYPHER = """
MATCH (p:POI {type:'commerce'})-[:IN_DONG]->(:Dong {code: $dong_code})
MATCH (p)-[:IN_CATEGORY]->(c:Category)
WHERE c.parent = $l1
OPTIONAL MATCH (a:Agent {id: $aid})-[kp:KNOWS_POI]->(p)
RETURN p.id AS poi_id, p.name AS name,
       (kp IS NOT NULL) AS known,
       coalesce(kp.visit_count, 0) AS visit_count,
       kp.avg_satisfaction AS avg_satisfaction,
       coalesce(kp.affinity, 0.0) AS affinity,
       kp.last_visit AS last_visit,
       size(coalesce(kp.recent_visit_dates, [])) AS v30,
       c.recovery_tau_days AS cat_tau,
       c.desire_drop AS cat_drop,
       c.saturation_n AS cat_sat_n,
       kp.source AS source,
       NULL AS km
ORDER BY known DESC LIMIT $limit
"""

# Fallback: dong에 아예 commerce POI 부족 시 자치구 단위 L1 fetch
STAGE2_FALLBACK_L1_DISTRICT_CYPHER = """
MATCH (p:POI {type:'commerce'})-[:IN_DONG]->(:Dong)<-[:HAS_DONG]-(d:District {code: $district_code})
MATCH (p)-[:IN_CATEGORY]->(c:Category)
WHERE c.parent = $l1
OPTIONAL MATCH (a:Agent {id: $aid})-[kp:KNOWS_POI]->(p)
RETURN p.id AS poi_id, p.name AS name,
       (kp IS NOT NULL) AS known,
       coalesce(kp.visit_count, 0) AS visit_count,
       kp.avg_satisfaction AS avg_satisfaction,
       coalesce(kp.affinity, 0.0) AS affinity,
       kp.last_visit AS last_visit,
       size(coalesce(kp.recent_visit_dates, [])) AS v30,
       c.recovery_tau_days AS cat_tau,
       c.desire_drop AS cat_drop,
       c.saturation_n AS cat_sat_n,
       kp.source AS source,
       NULL AS km
ORDER BY known DESC LIMIT $limit
"""


# =========================================================
# 데이터 클래스 + 포매터
# =========================================================
@dataclass
class DawnContext:
    persona: dict
    state: dict
    memory: list[dict] = field(default_factory=list)
    appointment: list[dict] = field(default_factory=list)
    policy: list[dict] = field(default_factory=list)
    social: list[dict] = field(default_factory=list)
    knows_poi_summary: list[dict] = field(default_factory=list)

    def to_prompt_blocks(self) -> dict[str, str]:
        """각 컨텍스트를 LLM 프롬프트에 넣을 텍스트 블록으로 변환."""
        # State의 policy_used JSON을 파싱해서 _format_policy에 전달
        import json as _json
        policy_used = {}
        raw = (self.state or {}).get("policy_used")
        if raw:
            try:
                policy_used = _json.loads(raw)
            except Exception:
                policy_used = {}
        return {
            "persona": _format_persona(self.persona),
            "state": _format_state(self.state),
            "memory": _format_memory(self.memory),
            "appointment": _format_appointment(self.appointment),
            "policy": _format_policy(self.policy, policy_used=policy_used),
            "social": _format_social(self.social),
            "knows_poi": _format_knows_poi(self.knows_poi_summary),
        }

    def get_policy_used(self) -> dict:
        """State에서 정책별 누적 사용액 dict 반환 (없으면 {})."""
        import json as _json
        raw = (self.state or {}).get("policy_used")
        if not raw:
            return {}
        try:
            return _json.loads(raw)
        except Exception:
            return {}


def _safe_top_cats(raw_json: str | None, k: int = 3) -> str:
    if not raw_json:
        return ""
    try:
        d = json.loads(raw_json)
    except Exception:
        return ""
    top = sorted(d.items(), key=lambda x: -x[1])[:k]
    return ", ".join(f"{k_}({int(v*100)}%)" for k_, v in top)


def _format_persona(p: dict) -> str:
    if not p:
        return "(페르소나 없음)"
    top_wd = _safe_top_cats(p.get("top_wd_json"))
    top_we = _safe_top_cats(p.get("top_we_json"))
    job = (p.get("job") or "").strip()
    lifestyle = (p.get("lifestyle") or "").strip()[:140]
    lines = [
        f"ID: {p['id']}",
        f"인구학: {p.get('age_group','')} {p.get('gender','')} / 직업: {job or '미상'} / 생애주기: {p.get('life_stage','')} / 소득: {p.get('income','')}",
        f"소비: 평일 {p.get('daily_wd',0):,}원, 주말 {p.get('daily_we',0):,}원 (주말/평일 {p.get('we_wd_ratio',1):.2f}배) / 성향: {p.get('tendency','')}",
        f"평일 Top 카테고리: {top_wd or '(없음)'}",
        f"주말 Top 카테고리: {top_we or '(없음)'}",
        f"행태: 배달 {p.get('delivery_days',0)}일/월, 평일 재택 {p.get('home_h_wd',0):.1f}h, 주말 재택 {p.get('home_h_we',0):.1f}h, 이동성 분위 {p.get('mobility',0)}",
        f"거주: {p.get('home_dong','?')} ({p.get('home_dong_code','?')}) — {p.get('home_poi','(이름없음)')}",
    ]
    if p.get("work_dong"):
        lines.append(f"직장: {p.get('work_dong','?')} ({p.get('work_dong_code','?')}) — {p.get('work_poi','(이름없음)')} / 통근 {p.get('commute_min',0)}분")
    else:
        lines.append("직장: 없음")
    if lifestyle:
        lines.append(f"라이프스타일: {lifestyle}")
    # NVIDIA 봉합 결과는 personality_lifestyle_raw 한 줄(200자)에 응축되어 있음 (가이드 §7).
    # 그 외 풍부 필드(summary/hobbies/cultural/career/skills/education/marital/family)는
    # Neo4j 에 보존되어 인터뷰·시각화·사후 분석에서 활용되지만, Stage 1 reasoning 프롬프트
    # 에는 토큰 절감을 위해 노출하지 않는다.
    return "\n".join(lines)


def _format_state(s: dict | None) -> str:
    if not s:
        return "(어제 State 없음 — Day 0 시드 누락 가능)"
    lc = s.get("policy_lc") or "{}"
    return (
        f"잔액: {s.get('balance',0):,}원 / 이번달 누적지출: {s.get('month_spent',0):,}원\n"
        f"에너지: {s.get('energy',0):.2f}, mood: {s.get('mood',0):.2f}, fatigue: {s.get('fatigue',0):.2f}\n"
        f"어제 평균 만족도: {s.get('yest_sat',0):.2f}\n"
        f"정책 라이프사이클: {lc}"
    )


def _format_memory(rows: list[dict]) -> str:
    if not rows:
        return "(최근 30일 기억 없음 — Day 0 직후 또는 신규 agent)"
    lines = []
    for r in rows:
        days = r.get("days_ago", 0)
        mtype = r["type"]
        prefix = f"{r['day']} ({days}일 전) [{mtype}]"
        cat = r.get("category") or ""

        if mtype == "visited":
            loc = r.get("poi_name") or "?"
            sat = r.get("satisfaction") or 0
            prefix += f" {loc}"
            if cat:
                prefix += f" / {cat}"
            prefix += f" 만족도 {sat:.2f}"
            summary = (r.get("summary") or "").strip()
            if summary:
                prefix += f" — {summary}"
        elif mtype == "rumor":
            # 노션 §5 — source(전달자) + topic_type:topic_value 가 핵심
            src = r.get("source") or "?"
            tt = r.get("topic_type") or "?"
            tv = r.get("topic_value") or ""
            prefix += f" {src}한테 들음 ({tt}: {tv})"
            if r.get("poi_name"):
                prefix += f" → {r['poi_name']}"
                if cat:
                    prefix += f" / {cat}"
        else:  # policy / sns / 기타
            loc = r.get("poi_name") or ""
            if loc:
                prefix += f" {loc}"
            if cat:
                prefix += f" / {cat}"
            summary = (r.get("summary") or "").strip()
            if summary:
                prefix += f" — {summary}"
        lines.append(prefix)
    return "\n".join(lines)


def _format_appointment(rows: list[dict]) -> str:
    if not rows:
        return "(오늘 예정된 약속 없음)"
    lines = []
    for r in rows:
        partners = ", ".join(r.get("with_agents") or []) or "?"
        when = r.get("target_time") or "시간 미정"
        # 매칭된 POI 우선, 없으면 LLM이 적은 자유 힌트
        poi_name = r.get("meeting_poi_name")
        poi_id = r.get("meeting_poi_id")
        if poi_name:
            where = f"{poi_name} ({poi_id})"
        elif r.get("meeting_location_hint"):
            where = f"{r['meeting_location_hint']} (POI 미매칭)"
        else:
            where = "장소 미정"
        lines.append(f"약속 {when} @ {where} / 대상: {partners}")
    return "\n".join(lines)


_POLICY_TYPE_LABEL = {
    "subsidy": "환급/쿠폰", "regulation": "규제", "facility": "시설",
    "campaign": "홍보", "tax": "세제", "transit": "교통", "environment": "환경",
}


def _format_policy(rows: list[dict], policy_used: dict[str, int] | None = None) -> str:
    """정책 컨텍스트 — 자연어 description 중심. subsidy는 잔액 노출.

    policy_used: {"P007": 87000, ...} 정책별 누적 사용액. State에서 가져옴.
    """
    if not rows:
        return "(거주·직장 동에 적용 정책 없음)"
    used = policy_used or {}
    lines = []
    for r in rows:
        type_label = _POLICY_TYPE_LABEL.get(r.get("type") or "", r.get("type") or "기타")
        regions = ", ".join([x for x in (r.get("regions") or []) if x]) or "?"
        target_l1s = r.get("target_l1s") or []
        targets = ", ".join([t for t in target_l1s if t]) if target_l1s else "(업종 비특정)"

        head = f"{r['id']} [{type_label}] {r['name']}"
        meta_parts = [f"적용지역: {regions}", f"대상업종: {targets}",
                      f"기간: {r['from_']}~{r['until_']}"]

        # subsidy(쿠폰·환급) 정책: 환급률 + 잔액 표시
        rate = r.get("rate")
        cap = r.get("cap")
        ptype = r.get("type")
        if ptype == "subsidy" and cap:
            cap_used = int(used.get(r["id"], 0))
            remaining = max(0, cap - cap_used)
            rate_s = f"{int(rate*100)}% 환급" if rate else "100% 차감"
            meta_parts.insert(0, f"{rate_s} (한도 {cap:,}원)")
            # 잔액 명시 (LLM이 보고 의사결정)
            meta_parts.append(
                f"💳 누적 사용 {cap_used:,}원 / 한도 {cap:,}원 — **남은 잔액 {remaining:,}원**"
                + (" ⚠️ 잔액 소진" if remaining == 0 else "")
            )
        elif rate:
            # cap 없는 subsidy 등
            meta_parts.insert(0, f"{int(rate*100)}% 환급")

        desc = r.get("description") or ""
        body = f"  ↳ {desc}" if desc else ""

        lines.append(head)
        lines.append("  " + " | ".join(meta_parts))
        if body:
            lines.append(body)
        lines.append("")  # 정책 간 공백
    return "\n".join(lines).rstrip()


def _format_social(rows: list[dict]) -> str:
    if not rows:
        return "(지인 없음)"
    lines = []
    for r in rows:
        ls = (r.get("lifestyle") or "")[:50]
        lines.append(
            f"{r['friend_id']} ({r.get('age','?')} {r.get('gender','?')}, {r['relation']}, 친밀도 {r['strength']:.1f})"
            + (f" — {ls}" if ls else "")
        )
    return "\n".join(lines)


def _format_knows_poi(rows: list[dict]) -> str:
    if not rows:
        return "(인지 POI 없음)"
    lines = []
    by_l1 = {}
    for r in rows:
        by_l1.setdefault(r["L1"], []).append(r)
    for l1, subs in by_l1.items():
        total = sum(s["n"] for s in subs)
        visited = sum(s["n_visited"] for s in subs)
        sub_str = ", ".join(f"{s['sub']}({s['n']})" for s in subs[:5])
        lines.append(f"{l1}: 총 {total}곳 인지 (방문경험 {visited}곳) — {sub_str}")
    return "\n".join(lines)


# =========================================================
# 메인 엔트리
# =========================================================
def build_dawn_context(
    aid: str,
    today: date,
    memory_top_n: int = 7,
    social_top_n: int = 8,
) -> DawnContext:
    """한 agent의 Dawn 컨텍스트를 7종 Cypher로 수집."""
    yesterday = today - __import__("datetime").timedelta(days=1)
    with driver_session() as s:
        persona = s.run(PERSONA_CYPHER, aid=aid).single()
        persona = dict(persona) if persona else {}

        state = s.run(STATE_CYPHER, aid=aid, yesterday=yesterday).single()
        state = dict(state) if state else {}

        memory = [dict(r) for r in s.run(MEMORY_CYPHER, aid=aid, today=today, top_n=memory_top_n)]
        appointment = [dict(r) for r in s.run(APPOINTMENT_CYPHER, aid=aid, today=today)]
        policy = [dict(r) for r in s.run(POLICY_CYPHER, aid=aid, today=today)]
        social = [dict(r) for r in s.run(SOCIAL_CYPHER, aid=aid, top_n=social_top_n)]
        knows_poi = [dict(r) for r in s.run(KNOWS_POI_SUMMARY_CYPHER, aid=aid)]

    return DawnContext(
        persona=persona, state=state, memory=memory,
        appointment=appointment, policy=policy, social=social,
        knows_poi_summary=knows_poi,
    )


def build_stage2_candidates(
    aid: str,
    dong_code: str,
    sub_category: str,
    limit: int = 30,
) -> list[dict]:
    """Stage 2 candidate POI Top-K."""
    with driver_session() as s:
        return [dict(r) for r in s.run(
            STAGE2_CANDIDATE_CYPHER,
            aid=aid, dong_code=dong_code, sub_category=sub_category, limit=limit
        )]


def build_stage2_candidates_l1_dong(
    aid: str, dong_code: str, l1: str, limit: int = 30,
) -> list[dict]:
    """Fallback: 같은 dong에서 L1 카테고리 단위로 commerce POI fetch."""
    with driver_session() as s:
        return [dict(r) for r in s.run(
            STAGE2_FALLBACK_L1_DONG_CYPHER,
            aid=aid, dong_code=dong_code, l1=l1, limit=limit
        )]


def build_stage2_candidates_l1_district(
    aid: str, district_code: str, l1: str, limit: int = 30,
) -> list[dict]:
    """Fallback: 자치구 안에서 L1 카테고리 단위로 commerce POI fetch."""
    with driver_session() as s:
        return [dict(r) for r in s.run(
            STAGE2_FALLBACK_L1_DISTRICT_CYPHER,
            aid=aid, district_code=district_code, l1=l1, limit=limit
        )]


# =========================================================
# CLI 테스트
# =========================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--aid", default="AGT_11110515_F_20대_001")
    ap.add_argument("--day", default="2026-05-01")
    args = ap.parse_args()

    today = date.fromisoformat(args.day)
    ctx = build_dawn_context(args.aid, today)
    blocks = ctx.to_prompt_blocks()

    for name, body in blocks.items():
        print(f"\n=== {name.upper()} ===")
        print(body)
