"""Dawn 컨텍스트 빌더 — 7종 고정 Cypher → 텍스트 블록.

매일 자정 각 agent에 대해 호출. 반환 dict는 Stage 1 프롬프트에 그대로 주입.

설계 출처: docs/schedule_generation_plan/runtime_ontology.md §4
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

# Windows console에서 한글 + 유니코드 다이아크리틱(em-dash 등) 출력 가능하게
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))            # mobility 등 동일 디렉토리
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neo4j_load._common import driver_session


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
  a.spending_level_wd AS spend_decile,
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
       s.policy_used AS policy_used,
       s.grant_received AS grant_received,
       s.grant_remaining AS grant_remaining
"""


# =========================================================
# Memory Top-N (30일, importance × exp(-days/14) 정렬)
# Day 0에는 initial Memory 없으므로 Day 1엔 빈 결과 가능
# =========================================================
MEMORY_CYPHER = """
MATCH (a:Agent {id: $aid})-[:REMEMBERS]->(m:Memory)
WHERE m.day >= $memory_since AND m.day < $today
WITH m, duration.inDays(m.day, $today).days AS days_ago
WITH m, days_ago,
     coalesce(m.importance, 0.0) * exp(-toFloat(days_ago) / 14.0) AS score
ORDER BY score DESC, m.day DESC, m.id
LIMIT $top_n
// Top-N을 먼저 확정한 뒤 POI/Category를 조인한다. 이전 쿼리는 최근 30일의
// 모든 Memory에 OPTIONAL MATCH를 수행해 누적 일수가 길수록 불필요한 행 확장이 컸다.
OPTIONAL MATCH (m)-[:ABOUT_POI]->(p:POI)
OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
RETURN m.id AS id, m.type AS type, m.day AS day, m.importance AS importance,
       coalesce(m.satisfaction, 0.0) AS satisfaction,
       coalesce(m.summary, '') AS summary,
       m.source AS source,
       m.topic_type AS topic_type,
       m.topic_value AS topic_value,
       p.name AS poi_name, c.name AS category, days_ago, score
ORDER BY score DESC, m.day DESC, m.id
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
       pol.poi_restricted AS poi_restricted,
       pol.effective_from AS from_, pol.effective_until AS until_,
       toString(pol.effective_from) AS effective_from, toString(pol.effective_until) AS effective_until,
       pol.income_grants AS income_grants,
       pol.excluded_income AS excluded_income,
       pol.decile_grants AS decile_grants,
       pol.excluded_deciles AS excluded_deciles,
       pol.grant_key AS grant_key,
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
ORDER BY n_visited DESC, n DESC LIMIT 20
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
       kp.last_visit AS last_visit,
       p.coupon_eligible AS coupon_eligible,
       km
ORDER BY km ASC LIMIT $limit
"""

# Fallback: sub_category 매칭 실패 시 L1 단위로 같은 dong에서 fetch
# Stage1 cat이 미매핑 세부업종이라도 Category.name으로도 매칭 시도 (parent OR name)
STAGE2_FALLBACK_L1_DONG_CYPHER = """
MATCH (p:POI {type:'commerce'})-[:IN_DONG]->(:Dong {code: $dong_code})
MATCH (p)-[:IN_CATEGORY]->(c:Category)
WHERE c.parent = $l1 OR c.name = $l1
OPTIONAL MATCH (a:Agent {id: $aid})-[kp:KNOWS_POI]->(p)
RETURN p.id AS poi_id, p.name AS name,
       (kp IS NOT NULL) AS known,
       coalesce(kp.visit_count, 0) AS visit_count,
       kp.avg_satisfaction AS avg_satisfaction,
       kp.last_visit AS last_visit,
       p.coupon_eligible AS coupon_eligible,
       NULL AS km
ORDER BY known DESC LIMIT $limit
"""

# Fallback: dong에 아예 commerce POI 부족 시 자치구 단위 L1 fetch
# Category.name도 매칭 (미매핑 세부업종 대응)
STAGE2_FALLBACK_L1_DISTRICT_CYPHER = """
MATCH (p:POI {type:'commerce'})-[:IN_DONG]->(:Dong)<-[:HAS_DONG]-(d:District {code: $district_code})
MATCH (p)-[:IN_CATEGORY]->(c:Category)
WHERE c.parent = $l1 OR c.name = $l1
OPTIONAL MATCH (a:Agent {id: $aid})-[kp:KNOWS_POI]->(p)
RETURN p.id AS poi_id, p.name AS name,
       (kp IS NOT NULL) AS known,
       coalesce(kp.visit_count, 0) AS visit_count,
       kp.avg_satisfaction AS avg_satisfaction,
       kp.last_visit AS last_visit,
       p.coupon_eligible AS coupon_eligible,
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
    # 오늘 갈 수 있는 zone 후보 (생활권 + Huff 광역상권) — Problem A
    zone_candidates: list[dict] = field(default_factory=list)
    # 기존 DB 호출에 타이머만 덧댄 진단 정보. 별도 LLM/DB 호출을 만들지 않는다.
    dawn_timing: dict = field(default_factory=dict)
    prompt_timing: dict = field(default_factory=dict)

    def to_prompt_blocks(self) -> dict[str, str]:
        """각 컨텍스트를 LLM 프롬프트에 넣을 텍스트 블록으로 변환."""
        tm: dict[str, float] = {}

        def timed(name: str, fn):
            started = time.perf_counter()
            value = fn()
            tm[name] = time.perf_counter() - started
            return value

        policy_used = timed(
            "t_policy_state_parse",
            lambda: _json_dict((self.state or {}).get("policy_used")),
        )
        blocks = {
            # 정책의 공통 사실과 개인별 상태를 분리한다. Stage1에서 공통 사실을
            # 사용자 메시지 앞쪽에 두면 SGLang prefix/radix cache가 재사용할 수 있다.
            "policy_facts": timed("t_policy_facts", lambda: _format_policy_facts(self.policy)),
            "policy": timed(
                "t_policy_status",
                lambda: _format_policy_status(
                    self.policy,
                    policy_used=policy_used,
                    persona=self.persona,
                    state=self.state,
                ),
            ),
            "persona": timed("t_persona", lambda: _format_persona(self.persona)),
            "state": timed("t_state", lambda: _format_state(self.state)),
            "memory": timed("t_memory", lambda: _format_memory(self.memory)),
            "appointment": timed("t_appointment", lambda: _format_appointment(self.appointment)),
            "social": timed("t_social", lambda: _format_social(self.social)),
            "knows_poi": timed("t_knows_poi", lambda: _format_knows_poi(self.knows_poi_summary)),
            "zones": timed("t_zones", lambda: _format_zones(self.zone_candidates)),
        }
        tm["t_total"] = sum(tm.values())
        self.prompt_timing = {k: round(v, 6) for k, v in tm.items()}
        return blocks

    def get_policy_used(self) -> dict:
        """State에서 정책별 누적 사용액 dict 반환 (없으면 {})."""
        return _json_dict((self.state or {}).get("policy_used"))

    def get_grant_remaining(self) -> dict:
        """State에서 정책별 grant 잔액 dict 반환 (grant type 정책용)."""
        return _json_dict((self.state or {}).get("grant_remaining"))


def _strip_lifestyle_first_line(lifestyle: str) -> str:
    """5줄 페르소나에서 첫 줄(성격/계획성 요약)을 제거.

    LLM에는 토큰 절감 + 행동 다양성 유도 위해 ②~⑤만 노출.
    형식이 다양해 ② 마커 우선, 없으면 줄 단위 첫 줄 제거.
    """
    if not lifestyle:
        return ""
    s = lifestyle.strip()
    for marker in ("②", "②", "**②"):
        if marker in s:
            return s[s.find(marker):].strip()
    lines = [ln for ln in s.split("\n") if ln.strip()]
    if len(lines) <= 1:
        return s
    return "\n".join(lines[1:]).strip()


def _format_persona(p: dict) -> str:
    if not p:
        return "(페르소나 없음)"
    job = (p.get("job") or "").strip()
    lifestyle = _strip_lifestyle_first_line(p.get("lifestyle") or "")[:280]
    lines = [
        f"ID: {p['id']}",
        f"인구학: {p.get('age_group','')} {p.get('gender','')} / 직업: {job or '미상'} / 생애주기: {p.get('life_stage','')} / 소득: {p.get('income','')}",
        f"소비: 평일 {(p.get('daily_wd') or 0):,}원, 주말 {(p.get('daily_we') or 0):,}원 (주말/평일 {(p.get('we_wd_ratio') or 1):.2f}배) / 성향: {p.get('tendency','')}",
        f"행태: 배달 {(p.get('delivery_days') or 0)}일/월, 평일 재택 {(p.get('home_h_wd') or 0):.1f}h, 주말 재택 {(p.get('home_h_we') or 0):.1f}h, 이동성 분위 {(p.get('mobility') or 0)}",
        f"거주: {p.get('home_dong','?')} ({p.get('home_dong_code','?')}) — {p.get('home_poi','(이름없음)')}",
    ]
    if p.get("work_dong"):
        lines.append(f"직장: {p.get('work_dong','?')} ({p.get('work_dong_code','?')}) — {p.get('work_poi','(이름없음)')} / 통근 {(p.get('commute_min') or 0)}분")
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
    # grant는 balance와 분리된 독립 지갑(grant_remaining) — 정책이 순수 자산을 오염시키지 않음.
    # windfall 감쇠 상세는 정책 카드(_format_policy)가 담당, 여기선 '별도 보유' 사실만 명시.
    bal = s.get("balance", 0) or 0
    bal_line = f"잔액(내 돈): {bal:,}원"
    rem_all = sum(int(v or 0) for v in _json_dict(s.get("grant_remaining")).values())
    if rem_all > 0:
        days_map = s.get("grant_days_since") or {}
        d_min = min(days_map.values()) if days_map else None
        if d_min is None or d_min <= 0:
            bal_line += f" · 정책지갑 잔액 {rem_all:,}원 (오늘 지급, 사용 조건은 정책 블록 참조)"
        else:
            bal_line += f" · 정책지갑 잔액 {rem_all:,}원 ({d_min}일 전 지급, 정책 블록 참조)"
    return (
        f"{bal_line} / 이번달 누적지출: {s.get('month_spent',0):,}원\n"
        f"에너지: {s.get('energy',0):.2f}, mood: {s.get('mood',0):.2f}, fatigue: {s.get('fatigue',0):.2f}\n"
        f"어제 평균 만족도: {s.get('yest_sat',0):.2f}\n"
        f"정책 라이프사이클: {lc}"
    )


def _format_memory(rows: list[dict]) -> str:
    if not rows:
        return "(최근 기억 없음 — 과거 방문·만족도·소문을 임의로 만들지 말 것)"
    lines = []
    for r in rows:
        days = r.get("days_ago", 0)
        mtype = r.get("type") or "기타"
        prefix = f"- D-{days} [{mtype}]"
        cat = r.get("category") or ""

        if mtype == "visited":
            loc = r.get("poi_name") or "장소 미상"
            sat = r.get("satisfaction") or 0
            prefix += f" {loc}"
            if cat:
                prefix += f"({cat})"
            prefix += f" 만족 {sat:.2f}"
            summary = " ".join((r.get("summary") or "").split())[:120]
            if summary:
                prefix += f" · {summary}"
        elif mtype == "rumor":
            src = r.get("source") or "출처 미상"
            tt = r.get("topic_type") or "주제"
            tv = " ".join(str(r.get("topic_value") or "").split())[:100]
            prefix += f" {src}에게 들음 · {tt}={tv}"
            if r.get("poi_name"):
                prefix += f" · {r['poi_name']}"
                if cat:
                    prefix += f"({cat})"
        else:  # policy / sns / 기타
            loc = r.get("poi_name") or ""
            if loc:
                prefix += f" {loc}"
            if cat:
                prefix += f"({cat})"
            summary = " ".join((r.get("summary") or "").split())[:120]
            if summary:
                prefix += f" · {summary}"
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
    "subsidy": "환급/쿠폰", "grant": "지원금", "regulation": "규제", "facility": "시설",
    "campaign": "홍보", "tax": "세제", "transit": "교통", "environment": "환경",
}


def _grant_amount_for(income: str, pol: dict, spend_decile=None) -> int:
    """grant 정책에서 이 agent가 받을 금액 (plan_writer 로직 재사용, lazy import).

    spend_decile: 소비 10분위. policy.grant_key='spend_decile'인 정책(예: P010)에서 사용.
    """
    try:
        from plan_writer import _grant_for_single_policy
        return _grant_for_single_policy(income, pol, spend_decile=spend_decile)
    except Exception:
        return 0


def _json_dict(raw) -> dict:
    import json as _json
    if isinstance(raw, dict):
        return raw
    try:
        return _json.loads(raw or "{}")
    except Exception:
        return {}


def _compact_regions(raw_regions: list | None) -> str:
    regions = sorted({str(x).strip() for x in (raw_regions or []) if str(x).strip()})
    if len(regions) >= 20:
        return f"서울 전역({len(regions)}개 자치구)"
    if len(regions) > 5:
        return ", ".join(regions[:5]) + f" 외 {len(regions) - 5}곳"
    return ", ".join(regions) or "지역 미상"


def _format_policy_facts(rows: list[dict]) -> str:
    """에이전트와 무관한 정책 사실.

    Stage1 사용자 프롬프트의 맨 앞에 배치해 동일 정책을 보는 에이전트끼리
    SGLang prefix/radix cache를 재사용한다. 행동 방향은 넣지 않는다.
    """
    if not rows:
        return "(활성 정책 없음)"
    lines: list[str] = []
    for r in sorted(rows, key=lambda x: str(x.get("id") or "")):
        ptype = r.get("type") or "기타"
        label = _POLICY_TYPE_LABEL.get(ptype, ptype)
        targets = [str(x) for x in (r.get("target_l1s") or []) if x]
        scope = ", ".join(targets[:8]) if targets else "업종 제한 없음"
        if len(targets) > 8:
            scope += f" 외 {len(targets) - 8}개"
        restrictions = " · [쿠폰] 표시 POI에서만 사용" if r.get("poi_restricted") else ""
        desc = " ".join(str(r.get("description") or "").split())[:280]
        lines.append(
            f"- {r.get('id')} [{label}] {r.get('name')} | "
            f"{r.get('from_')}~{r.get('until_')} | {_compact_regions(r.get('regions'))} | "
            f"{scope}{restrictions}"
        )
        if desc:
            lines.append(f"  배경: {desc}")
    return "\n".join(lines)


def _format_policy_status(
    rows: list[dict],
    policy_used: dict[str, int] | None = None,
    persona: dict | None = None,
    state: dict | None = None,
) -> str:
    """개인별 자격·잔액만 간결하게 표시한다.

    정책이 소비를 늘리거나 줄인다고 지시하지 않는다. 정책은 선택 가능한 예산과
    제약으로만 제시하고, 행동은 페르소나·필요·상황에서 내생적으로 결정하게 한다.
    """
    if not rows:
        return "(해당 없음 — 정책·지원금·바우처·쿠폰을 임의로 언급하지 말 것)"
    used = policy_used or {}
    p = persona or {}
    st = state or {}
    grant_received = _json_dict(st.get("grant_received"))
    grant_remaining = _json_dict(st.get("grant_remaining"))
    days_since = _json_dict(st.get("grant_days_since"))
    lines: list[str] = []

    for r in sorted(rows, key=lambda x: str(x.get("id") or "")):
        pid = str(r.get("id") or "")
        ptype = r.get("type")
        if ptype == "grant":
            income = str(p.get("income") or "").strip()
            decile = p.get("spend_decile")
            my_amt = _grant_amount_for(income, r, spend_decile=decile)
            uses_decile = bool(r.get("decile_grants")) or (
                (r.get("grant_key") or "income") == "spend_decile"
            )
            basis = f"소비 {int(decile)}분위" if uses_decile and decile is not None else f"소득 {income or '미상'}"
            rec = int(grant_received.get(pid, 0) or 0)
            rem = int(grant_remaining.get(pid, 0) or 0)
            d_since = days_since.get(pid)
            age = "지급 전"
            if rec > 0:
                age = "지급일" if d_since in (None, 0, "0") else f"지급 후 {int(d_since)}일"
            relative = ""
            daily = float(p.get("daily_wd") or 0)
            if rem > 0 and daily > 0:
                relative = f" · 평소 평일소비의 {rem / daily:.1f}일치"
            eligibility = "대상" if my_amt > 0 else "비대상"
            lines.append(
                f"- {pid}: {eligibility}({basis}) | 지급액 {my_amt:,}원 | "
                f"누적수령 {rec:,}원 | 정책지갑 잔액 {rem:,}원 | {age}{relative}"
            )
        elif ptype == "subsidy":
            cap = int(r.get("cap") or 0)
            spent = int(used.get(pid, 0) or 0)
            rem = max(0, cap - spent) if cap else 0
            rate = r.get("rate")
            rate_text = f"{float(rate) * 100:.0f}%" if rate is not None else "정책 정의값"
            lines.append(f"- {pid}: 환급률 {rate_text} | 누적사용 {spent:,}원 | 잔여한도 {rem:,}원")
        else:
            lines.append(f"- {pid}: 현재 적용 중")

    lines.append(
        "- 판단 원칙: 소비 필요·시점·총액·POI는 본인의 평소 습관, 자산, 일정에 따라 "
        "판단한다. 선택한 거래가 정책 사용처이면 정책지갑을 자기자금보다 먼저 결제하며, "
        "이 결제 순서는 소비 자체를 새로 만들라는 뜻이 아니다."
    )
    return "\n".join(lines)


def _format_policy(
    rows: list[dict],
    policy_used: dict[str, int] | None = None,
    persona: dict | None = None,
    state: dict | None = None,
) -> str:
    """기존 호출부 호환용: 공통 정책 사실과 개인 상태를 함께 반환."""
    return (
        _format_policy_facts(rows)
        + "\n"
        + _format_policy_status(rows, policy_used=policy_used, persona=persona, state=state)
    )


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


_ZONE_TAG = {"home": "생활권·거주", "work": "생활권·직장", "hub": "광역상권"}


def _format_zones(zones: list[dict]) -> str:
    """오늘 갈 수 있는 zone 후보 — Stage1이 외출 anchor(zone:<코드>)에 쓸 코드 목록."""
    if not zones:
        return "(zone 후보 없음 — 거주/직장 동 코드 사용)"
    lines = []
    for z in zones:
        tag = _ZONE_TAG.get(z.get("type"), "상권")
        sig = z.get("signature")
        if z.get("type") == "hub" and sig and sig != "general":
            tag = f"{tag}·{sig}"
        dist = z.get("distance_km")
        dist_s = f", {dist:.1f}km" if isinstance(dist, (int, float)) else ""
        extra = " ← 주말 나들이·여가 등에 적합" if z.get("type") == "hub" else ""
        lines.append(f"- [{tag}] {z['code']} {z.get('name','')}{dist_s}{extra}")
    lines.append("평일엔 주로 생활권, 주말·여가/쇼핑이면 광역상권도 자연스럽게 선택(거리·기분 고려).")
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
def _build_zone_candidates(persona: dict, today: date) -> list[dict]:
    """오늘 갈 수 있는 zone 후보 = 생활권(거주·직장) + Huff 광역상권(MOBILITY_WIDE).

    좌표/카탈로그 없거나 legacy 모드면 생활권만 → 기존 동작으로 자연 degrade.
    런타임 Neo4j 거리쿼리 없이 dong_centroids.json + haversine 으로 계산.
    """
    home_code = persona.get("home_dong_code")
    work_code = persona.get("work_dong_code")
    zones: list[dict] = []
    if home_code:
        zones.append({"code": home_code, "name": persona.get("home_dong") or "",
                      "type": "home", "distance_km": 0.0})
    if work_code:
        zones.append({"code": work_code, "name": persona.get("work_dong") or "",
                      "type": "work", "distance_km": None})

    if os.environ.get("MOBILITY_WIDE", "wide") == "legacy" or not home_code:
        return zones
    try:
        import mobility
        exclude = {c for c in (home_code, work_code) if c}
        day_type = "weekend" if today.weekday() >= 5 else "weekday"
        rng = random.Random(hash((persona.get("id"), today.isoformat())))
        for h in mobility.suggest_hubs(home_code, exclude, day_type,
                                       persona.get("mobility"), k=8, rng=rng,
                                       persona=persona):
            zones.append({"code": h["code"], "name": h.get("name", ""),
                          "gu": h.get("gu", ""), "type": "hub",
                          "signature": h.get("signature"),
                          "distance_km": h.get("distance_km")})
    except Exception:
        pass   # 광역 prior 실패해도 생활권만으로 진행
    return zones


_PERSONA_CACHE: dict[str, dict] = {}
_PERSONA_CACHE_LOCK = threading.Lock()
_POLICY_CACHE: dict[tuple[str, str, str], list[dict]] = {}
_POLICY_CACHE_DAY: str | None = None
_POLICY_CACHE_LOCK = threading.Lock()


def _persona_from_cache(aid: str) -> dict | None:
    if os.environ.get("DAWN_PERSONA_CACHE", "1") == "0":
        return None
    with _PERSONA_CACHE_LOCK:
        cached = _PERSONA_CACHE.get(aid)
        return dict(cached) if cached is not None else None


def _cache_persona(aid: str, persona: dict) -> None:
    if os.environ.get("DAWN_PERSONA_CACHE", "1") == "0" or not persona:
        return
    with _PERSONA_CACHE_LOCK:
        _PERSONA_CACHE.setdefault(aid, dict(persona))


def _policy_cache_key(today: date, persona: dict) -> tuple[str, str, str]:
    return (
        today.isoformat(),
        str(persona.get("home_dong_code") or ""),
        str(persona.get("work_dong_code") or ""),
    )


def _policy_from_cache(today: date, persona: dict) -> list[dict] | None:
    if os.environ.get("DAWN_POLICY_CACHE", "1") == "0":
        return None
    global _POLICY_CACHE_DAY
    key = _policy_cache_key(today, persona)
    with _POLICY_CACHE_LOCK:
        if _POLICY_CACHE_DAY != key[0]:
            _POLICY_CACHE.clear()
            _POLICY_CACHE_DAY = key[0]
        cached = _POLICY_CACHE.get(key)
        return [dict(x) for x in cached] if cached is not None else None


def _cache_policy(today: date, persona: dict, rows: list[dict]) -> None:
    if os.environ.get("DAWN_POLICY_CACHE", "1") == "0":
        return
    global _POLICY_CACHE_DAY
    key = _policy_cache_key(today, persona)
    with _POLICY_CACHE_LOCK:
        if _POLICY_CACHE_DAY != key[0]:
            _POLICY_CACHE.clear()
            _POLICY_CACHE_DAY = key[0]
        _POLICY_CACHE.setdefault(key, [dict(x) for x in rows])


def build_dawn_context(
    aid: str,
    today: date,
    memory_top_n: int = 8,
    social_top_n: int = 8,
) -> DawnContext:
    """한 agent의 Dawn 컨텍스트와 기존 호출 구간별 소요시간을 반환."""
    yesterday = today - timedelta(days=1)
    memory_since = today - timedelta(days=30)
    tm: dict[str, float | int | bool] = {}
    total_started = time.perf_counter()

    with driver_session() as s:
        started = time.perf_counter()
        persona = _persona_from_cache(aid)
        tm["persona_cache_hit"] = persona is not None
        if persona is None:
            row = s.run(PERSONA_CYPHER, aid=aid).single()
            persona = dict(row) if row else {}
            _cache_persona(aid, persona)
        tm["t_persona"] = time.perf_counter() - started

        started = time.perf_counter()
        state = s.run(STATE_CYPHER, aid=aid, yesterday=yesterday).single()
        state = dict(state) if state else {}
        tm["t_state"] = time.perf_counter() - started

        started = time.perf_counter()
        memory = [
            dict(r)
            for r in s.run(
                MEMORY_CYPHER,
                aid=aid,
                today=today,
                memory_since=memory_since,
                top_n=memory_top_n,
            )
        ]
        tm["t_memory"] = time.perf_counter() - started
        tm["n_memory_returned"] = len(memory)

        started = time.perf_counter()
        appointment = [dict(r) for r in s.run(APPOINTMENT_CYPHER, aid=aid, today=today)]
        tm["t_appointment"] = time.perf_counter() - started

        started = time.perf_counter()
        policy = _policy_from_cache(today, persona)
        tm["policy_cache_hit"] = policy is not None
        if policy is None:
            policy = [dict(r) for r in s.run(POLICY_CYPHER, aid=aid, today=today)]
            _cache_policy(today, persona, policy)
        tm["t_policy"] = time.perf_counter() - started

        started = time.perf_counter()
        social = [dict(r) for r in s.run(SOCIAL_CYPHER, aid=aid, top_n=social_top_n)]
        tm["t_social"] = time.perf_counter() - started

        started = time.perf_counter()
        knows_poi = [dict(r) for r in s.run(KNOWS_POI_SUMMARY_CYPHER, aid=aid)]
        tm["t_knows_poi"] = time.perf_counter() - started

    started = time.perf_counter()
    zones = _build_zone_candidates(persona, today)
    tm["t_zones"] = time.perf_counter() - started
    tm["t_total"] = time.perf_counter() - total_started

    return DawnContext(
        persona=persona, state=state, memory=memory,
        appointment=appointment, policy=policy, social=social,
        knows_poi_summary=knows_poi,
        zone_candidates=zones,
        dawn_timing={
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in tm.items()
        },
    )


def _run_candidates(cypher: str, session=None, **params) -> list[dict]:
    """후보 Cypher 실행. session 주어지면 재사용(권장 — agent-day당 1세션),
    없으면 단발 세션 오픈(하위호환). 세션 재사용으로 커넥션/세션 생성 오버헤드 제거."""
    if session is not None:
        return [dict(r) for r in session.run(cypher, **params)]
    with driver_session() as s:
        return [dict(r) for r in s.run(cypher, **params)]


def build_stage2_candidates(
    aid: str,
    dong_code: str,
    sub_category: str,
    limit: int = 30,
    session=None,
) -> list[dict]:
    """Stage 2 candidate POI Top-K."""
    return _run_candidates(
        STAGE2_CANDIDATE_CYPHER, session=session,
        aid=aid, dong_code=dong_code, sub_category=sub_category, limit=limit,
    )


def build_stage2_candidates_l1_dong(
    aid: str, dong_code: str, l1: str, limit: int = 30, session=None,
) -> list[dict]:
    """Fallback: 같은 dong에서 L1 카테고리 단위로 commerce POI fetch."""
    return _run_candidates(
        STAGE2_FALLBACK_L1_DONG_CYPHER, session=session,
        aid=aid, dong_code=dong_code, l1=l1, limit=limit,
    )


def build_stage2_candidates_l1_district(
    aid: str, district_code: str, l1: str, limit: int = 30, session=None,
) -> list[dict]:
    """Fallback: 자치구 안에서 L1 카테고리 단위로 commerce POI fetch."""
    return _run_candidates(
        STAGE2_FALLBACK_L1_DISTRICT_CYPHER, session=session,
        aid=aid, district_code=district_code, l1=l1, limit=limit,
    )


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
