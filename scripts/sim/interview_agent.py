"""한 에이전트와 1대1 인터뷰 — 페르소나 어시밀레이션 + 실제 시뮬 reasoning 인용.

시뮬 풀런 중 Stage 1/2/Night LLM이 INCLUDES·Conversation·Memory에 남긴
reasoning/trigger/pick_reason/pick_factor 흔적을 모두 모아 LLM에게 주입.
LLM은 "그 에이전트가 된 듯" 1인칭 자연어로 자신의 행동·심리를 설명함.

CLI:
  python scripts/sim/interview_agent.py --aid AGT_11680670_M_50대_001 \
      --days 2026-05-01,2026-05-02,2026-05-03
  → 대화형 REPL 진입. 빈 줄 입력 시 종료.

  python scripts/sim/interview_agent.py --aid <id> --question "왜 그날 점심을 두부마을찬으로?"
  → 일회성 질의응답.

옵션:
  --label positive|negative|neutral
       각 라벨 군집에서 대표 1명 자동 추출 후 인터뷰 (정책 사용액 + mood 기반).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))

from _common import driver_session  # noqa: E402
from llm_client import call_chat as _llm_call  # noqa: E402


# ═══════════════════════════════════════════════════════════════
# 인터뷰 SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════
INTERVIEW_SYSTEM = """당신은 가상 도시 시뮬레이션의 한 시민 에이전트입니다.
지금부터 면접관과 1대1 인터뷰를 합니다. 다음 페르소나의 사람처럼 1인칭으로 답하세요.

[행동 원칙]
- 답변은 페르소나(연령·직업·라이프스타일·소득)와 일관되어야 합니다. 페르소나를 벗어난 답은 금지.
- 사용자 블록의 "## 내 실제 행동·결정 흔적" 섹션의 각 이벤트에는 reasoning, trigger, pick_reason, pick_factor가 적혀 있습니다.
  - reasoning = 그 이벤트를 선택한 본인의 사고
  - trigger = appointment/rumor/policy/lifestyle/top_category/mood/none 중 하나의 결정 요인
  - pick_reason = 그 장소(POI)를 고른 사고
  - pick_factor = known/distance/satisfaction/rumor/novelty/random
  이 흔적을 **반드시 인용·확장**해서 답변하세요. 사후 지어낸 추론보다 흔적이 우선입니다.
- 약속(appointment) 때문에 갔는지, 소문(rumor) 들어서 갔는지, 정책(policy) 보고 갔는지,
  라이프스타일(lifestyle/top_category) 따라간 건지, 그날 컨디션(mood) 때문인지 **명확히 분리해서 말하세요**.
- 만족도(actual_satisfaction)·정책 사용액·기억 Memory도 자유롭게 인용하세요.

[심리·감정 표현]
- 페르소나의 소비성향(절약형/보통/소비형)과 라이프스타일에 맞는 감정 어휘를 쓰세요.
  · 절약형: "할인 받아서 기분 좋더라", "잔액 남았으니 아꼈다"
  · 가족중심: "아내·자녀와 함께 가서 좋았다", "혼자였으면 안 갔을 것"
  · 건강·운동: "체력 회복", "스트레스 풀려", "꾸준히 가는 게 중요"
  · 트렌드·문화: "분위기가 인스타용", "한 번 가보고 싶었다"
- mood·fatigue 수치도 1인칭 감정으로 변환 ("그날 좀 피곤해서…", "기분이 별로라…").

[설명 가능성 — 매우 중요]
- 면접관이 "왜?"를 물으면, **데이터 근거(만족도 수치, POI명, 정책 ID, 약속 상대 agent_id 등)를 명시 인용**하세요.
- "정책 때문이었나요?"라는 질문에는 **policy_id + cap 잔액 + benefit_rate**까지 인용.
- "그 사람이 왜 추천했어요?"에는 **Memory.summary 또는 Conversation.reasoning**을 인용.

[답변 형식]
- 자연스러운 한국어 구어체 (존댓말 권장). 1~5문장.
- 데이터 인용은 인용부호 없이 자연스럽게 ("그날 두부마을찬 만족도가 0.65였거든요").
- JSON·코드블록·메타설명 금지. 인터뷰 답변만.

[금지]
- 페르소나에 없는 정보 (성격, 가족 구성, 학력 등) 지어내기 금지.
- 시뮬에 없는 사건 만들어내기 금지.
- 추상적·모호한 답 금지 ("그냥요", "기분 따라요").
/no_think"""


# ═══════════════════════════════════════════════════════════════
# Neo4j 데이터 수집
# ═══════════════════════════════════════════════════════════════
def fetch_agent_full(aid: str, days: list[str]) -> dict:
    """한 agent의 페르소나·State·Plan·Memory·Conversation·KNOWS_POI 전체."""
    out: dict = {}
    with driver_session() as s:
        # 1. 페르소나
        r = s.run("""
            MATCH (a:Agent {id:$aid})
            OPTIONAL MATCH (a)-[:LIVES_AT]->(home:POI)-[:IN_DONG]->(hd:Dong)
            OPTIONAL MATCH (a)-[:WORKS_AT]->(work:POI)-[:IN_DONG]->(wd:Dong)
            RETURN a {.*}, home.name AS home_name, hd.name AS home_dong,
                   work.name AS work_name, wd.name AS work_dong
        """, aid=aid).single()
        if not r:
            raise SystemExit(f"agent not found: {aid}")
        out["persona"] = dict(r["a"])
        out["persona"]["home_poi_name"] = r["home_name"]
        out["persona"]["home_dong_name"] = r["home_dong"]
        out["persona"]["work_poi_name"] = r["work_name"]
        out["persona"]["work_dong_name"] = r["work_dong"]

        # 2. State (일자별)
        out["state"] = []
        for x in s.run("""
            MATCH (a:Agent {id:$aid})-[:HAS_STATE]->(st:State)
            WHERE toString(st.day) IN $days
            RETURN toString(st.day) AS day, st.balance, st.mood, st.fatigue,
                   st.yesterday_satisfaction, st.policy_used, st.month_spent
            ORDER BY day
        """, aid=aid, days=days):
            out["state"].append(dict(x))

        # 3. Plan + INCLUDES (reasoning·trigger·pick_reason 다 포함)
        out["plans"] = []
        for x in s.run("""
            MATCH (a:Agent {id:$aid})-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(poi:POI)
            WHERE toString(p.day) IN $days
            OPTIONAL MATCH (poi)-[:IN_CATEGORY]->(c:Category)
            OPTIONAL MATCH (poi)-[:IN_DONG]->(d:Dong)
            OPTIONAL MATCH (d)<-[:HAS_DONG]-(dist:District)
            RETURN toString(p.day) AS day, i.order AS ord, toString(i.time) AS time,
                   i.anchor AS anchor, i.category AS cat, i.sub_category AS sub,
                   i.intent AS intent,
                   i.reasoning AS reasoning, i.trigger AS trigger,
                   i.pick_reason AS pick_reason, i.pick_factor AS pick_factor,
                   i.actual_satisfaction AS sat, i.actual_spent AS spent,
                   poi.id AS poi_id, poi.name AS poi_name,
                   c.parent AS l1, c.name AS sub_cat_name,
                   d.name AS dong_name, dist.name AS district_name
            ORDER BY day, ord
        """, aid=aid, days=days):
            out["plans"].append(dict(x))

        # 4. Memory (visited + rumor)
        out["memories"] = []
        for x in s.run("""
            MATCH (a:Agent {id:$aid})-[:REMEMBERS]->(m:Memory)
            OPTIONAL MATCH (m)-[:ABOUT_POI]->(p:POI)
            OPTIONAL MATCH (m)-[:FROM_CONVERSATION]->(c:Conversation)
            RETURN m.type AS type, toString(m.day) AS day,
                   m.importance AS imp, m.satisfaction AS sat,
                   m.summary AS summary, m.source AS source,
                   m.topic_type AS topic_type, m.topic_value AS topic_value,
                   p.name AS poi_name,
                   c.id AS conv_id, c.intent AS conv_intent, c.reasoning AS conv_reasoning
            ORDER BY day DESC, m.importance DESC LIMIT 50
        """, aid=aid):
            out["memories"].append(dict(x))

        # 5. Conversation (양쪽 참여 모두 — 사회적 상호작용 전체)
        out["conversations"] = []
        for x in s.run("""
            MATCH (a:Agent {id:$aid})-[part:PARTICIPATES_IN]->(c:Conversation)
            OPTIONAL MATCH (c)-[:MENTIONS_POI]->(meet:POI)
            RETURN c.id AS cid, toString(c.day) AS day, c.intent AS intent,
                   part.role AS role,
                   c.initiator_id AS initiator, c.recipient_id AS recipient,
                   c.topic_type AS topic_type, c.topic_value AS topic_value,
                   c.target_day_offset AS offset, c.target_time AS target_time,
                   c.meeting_location_hint AS hint,
                   meet.name AS meeting_poi,
                   c.reasoning AS reasoning
            ORDER BY day DESC, c.id LIMIT 30
        """, aid=aid):
            out["conversations"].append(dict(x))

        # 6. KNOWS_POI (단골 Top)
        out["knows_poi"] = []
        for x in s.run("""
            MATCH (a:Agent {id:$aid})-[kp:KNOWS_POI]->(p:POI)
            WHERE kp.visit_count > 0
            OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
            RETURN p.name AS poi_name, c.parent AS l1, c.name AS sub_cat,
                   kp.visit_count AS visit, kp.avg_satisfaction AS sat,
                   kp.affinity AS affinity, kp.source AS source
            ORDER BY kp.affinity DESC LIMIT 15
        """, aid=aid):
            out["knows_poi"].append(dict(x))

    return out


# ═══════════════════════════════════════════════════════════════
# user_block 빌더
# ═══════════════════════════════════════════════════════════════
def _fmt_int(value, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        return f"{int(value):,}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def _fmt_float(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def build_user_block(data: dict, question: str,
                     policy_ctx: dict | None = None) -> str:
    p = data["persona"]
    lines = ["## 페르소나 (당신 자신)"]
    lines.append(f"- ID: {p.get('id')}")
    lines.append(f"- 인구학: {p.get('p_age_group','?')} {p.get('p_gender','?')} / "
                 f"직업: {p.get('personal_job_raw','?')} / 생애주기: {p.get('p_life_stage','?')}")
    lines.append(f"- 소득: {p.get('p_income_level','?')} / 성향: {p.get('pr_spending_tendency','?')}")
    lines.append(f"- 라이프스타일: {p.get('personality_lifestyle_raw','')}")
    lines.append(f"- 거주: {p.get('home_dong_name')} — {p.get('home_poi_name')}")
    if p.get('work_poi_name'):
        lines.append(f"- 직장: {p.get('work_dong_name')} — {p.get('work_poi_name')}")
    lines.append(f"- 평일 소비 {p.get('s_daily_wd','?')}원, 주말 {p.get('s_daily_we','?')}원")
    try:
        top_wd = json.loads(p.get('spending_top_wd_json') or '{}')
        top3 = ", ".join(f"{k}({int(v*100)}%)" for k, v in list(top_wd.items())[:3])
        lines.append(f"- 평일 Top 카테고리: {top3}")
    except Exception:
        pass

    if policy_ctx:
        lines.append("\n## 인터뷰 대상 정책")
        lines.append(f"- ID: {policy_ctx.get('id')}")
        lines.append(f"- 이름: {policy_ctx.get('name')}")
        lines.append(f"- 유형: {policy_ctx.get('type')}")
        if policy_ctx.get("effective_from"):
            lines.append(f"- 시행일: {policy_ctx.get('effective_from')}")
        if policy_ctx.get("target_districts"):
            lines.append(f"- 대상 지역: {', '.join(policy_ctx.get('target_districts') or [])}")
        target_cats = policy_ctx.get("target_cats") or policy_ctx.get("benefit_categories") or []
        if target_cats:
            lines.append(f"- 대상 업종: {', '.join(target_cats)}")
        desc = (policy_ctx.get("description") or "")[:220]
        if desc:
            lines.append(f"- 설명: {desc}")
        lines.append("- 답변할 때 정책과 직접 관련 없는 방문은 정책 효과로 말하지 마세요.")

    # State
    lines.append("\n## 일자별 컨디션·잔액")
    for st in data["state"]:
        used_summary = ""
        if st.get('policy_used'):
            try:
                pu = json.loads(st['policy_used']) if isinstance(st['policy_used'], str) else st['policy_used']
                if pu:
                    used_summary = " · 정책사용 " + ", ".join(
                        f"{k}={_fmt_int(v, '원')}" for k, v in pu.items()
                    )
            except Exception:
                pass
        lines.append(
            f"- {st['day']}: 잔액 {_fmt_int(st.get('balance'), '원')} · "
            f"mood {_fmt_float(st.get('mood'))} · "
            f"fatigue {_fmt_float(st.get('fatigue'))}{used_summary}"
        )

    # Plan + reasoning (인터뷰의 핵심 근거)
    # vLLM context 8192 한계 → 외출 이벤트만, 최근 40개로 cap, reasoning 100자 cut
    lines.append("\n## 내 실제 행동·결정 흔적 (시뮬 LLM이 그 시점에 남긴 reasoning 그대로)")
    outing_plans = [ev for ev in data["plans"]
                    if (ev.get("cat") or "") not in ("집", "직장")]
    # 최근 40개만
    outing_plans = outing_plans[-40:] if len(outing_plans) > 40 else outing_plans
    cur_day = None
    for ev in outing_plans:
        if ev["day"] != cur_day:
            cur_day = ev["day"]
            lines.append(f"\n### {cur_day}")
        sat_str = f" sat={ev['sat']:.2f}" if ev.get('sat') is not None else ""
        spent_str = f" · {ev['spent']:,}원" if (ev.get('spent') or 0) > 0 else ""
        time_str = (ev['time'] or "")[:5]
        district = ev.get("district_name")
        dong = ev.get("dong_name")
        loc = ""
        if district or dong:
            loc = f" ({district or '?'} {dong or '?'})"
        lines.append(
            f"- {time_str} {ev['cat']}/{ev.get('sub') or '-'} → "
            f"{ev.get('poi_name') or ev['poi_id']}{loc}{sat_str}{spent_str}"
        )
        r = (ev.get("reasoning") or "")[:120]
        if r:
            lines.append(f"  · 내가 생각한 것 ({ev.get('trigger','?')}): {r}")
        pr = (ev.get("pick_reason") or "")[:80]
        if pr:
            lines.append(f"  · 장소 선택 ({ev.get('pick_factor','?')}): {pr}")

    # Memory (소문 위주 — visited는 위 Plan에 이미 있음)
    rumor_mems = [m for m in data["memories"] if m["type"] == "rumor"][:10]
    if rumor_mems:
        lines.append("\n## 내가 들은 소문 (rumor Memory)")
        for m in rumor_mems:
            src = m.get('source') or '?'
            tv = (m.get('topic_value') or '')[:50]
            summary = (m.get('summary') or '')[:100]
            lines.append(f"- {m['day']} {src}한테 들음 | {tv} · {summary}")

    # Conversation (10개로 줄임)
    if data["conversations"]:
        lines.append("\n## 내 상호작용 (Conversation)")
        for c in data["conversations"][:10]:
            partner = c["recipient"] if c["role"] == "initiator" else c["initiator"]
            who = "내가 먼저" if c["role"] == "initiator" else "상대가 먼저"
            extra = ""
            if c["intent"] == "약속" and c.get("offset") is not None:
                extra = f" → D+{c['offset']} {c.get('target_time','')} @ {c.get('meeting_poi') or c.get('hint','?')}"
            line = f"- {c['day']} [{c['intent']}] {who}({partner}) · {c.get('topic_value') or '-'}{extra}"
            lines.append(line)
            r = (c.get("reasoning") or "")[:120]
            if r:
                lines.append(f"  · 사유: {r}")

    # KNOWS_POI 단골 (5개)
    if data["knows_poi"]:
        lines.append("\n## 내 단골 가게 Top 5")
        for k in data["knows_poi"][:5]:
            lines.append(
                f"- {k['poi_name']} ({k.get('l1','?')}) "
                f"방문 {_fmt_int(k.get('visit'))}회 aff {_fmt_float(k.get('affinity'))}"
            )

    # 질문
    lines.append("\n---")
    lines.append("## 면접관 질문")
    lines.append(question)
    lines.append("\n위 모든 데이터를 근거로 본인 페르소나로 1인칭 답변. JSON 금지, 자연어만. /no_think")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# LLM 호출
# ═══════════════════════════════════════════════════════════════
def ask(data: dict, question: str, temperature: float = 0.7,
        mode: str | None = None, policy_ctx: dict | None = None) -> str:
    user = build_user_block(data, question, policy_ctx=policy_ctx)
    resp = _llm_call(
        mode, INTERVIEW_SYSTEM, user,
        temperature=temperature, max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════════
# 군집 대표 추출 (긍정/부정/변화없음)
# ═══════════════════════════════════════════════════════════════
# P009 grant 정책 (소득별 현금 지원금: 중상=100k, 중=250k, 중하=450k, 하=600k, 상=제외)
# effective_from=2026-05-27 (3일 sim의 Day 3)
POLICY_DAY = "2026-05-27"


def find_label_sample(label: str, last_day: str) -> str | None:
    """label ∈ {positive, negative, neutral}.

    P009는 type=grant 정책. 소득별 현금 지원금이 잔액에 추가되고,
    LLM이 거래마다 policy_spend(쿠폰 사용액)를 자율 결정.
    INCLUDES.spent_from_policy JSON에 거래별 정책 사용액 적재됨.

    라벨링 (grant 수혜 + 사용 + 만족도 기반):
      positive: grant 대상 (income != '상') + 정책일 grant 사용 거래 ≥ 1 + 평균 만족도 상위
                → "지원금 수혜·활용·만족"
      negative: grant 대상 + 정책일 grant 사용 거래 = 0 + 만족도 하위
                → "지원금 받았지만 미사용·불만"
      neutral : grant 대상 + 정책일 grant 사용 거래 ≥ 1 + 만족도 중간
                → "지원금 사용했지만 평균적"
    """
    if label not in ("positive", "negative", "neutral"):
        return None

    # spent_from_policy JSON이 의미있는 값(`{}`·`null`·NULL 아님)을 정책 사용으로 간주
    USED_FILTER = (
        "i.spent_from_policy IS NOT NULL "
        "AND i.spent_from_policy <> '{}' "
        "AND i.spent_from_policy <> 'null'"
    )

    with driver_session() as s:
        if label == "positive":
            # grant 대상 + 정책일 grant 사용 + 만족 상위
            rows = s.run(f"""
                MATCH (a:Agent)
                WHERE a.p_income_level <> '상'
                MATCH (a)-[:HAS_PLAN {{day: date('{POLICY_DAY}')}}]->(:Plan)-[i:INCLUDES]->(:POI)
                WHERE i.actual_satisfaction IS NOT NULL
                WITH a,
                     avg(i.actual_satisfaction) AS sat,
                     sum(CASE WHEN {USED_FILTER} THEN 1 ELSE 0 END) AS policy_uses,
                     count(i) AS visits
                WHERE policy_uses >= 1 AND sat >= 0.6 AND visits >= 2
                RETURN a.id AS id ORDER BY policy_uses DESC, sat DESC LIMIT 50
            """).data()
        elif label == "negative":
            # grant 대상이지만 정책일 grant 사용 안 함 + 만족도 하위
            rows = s.run(f"""
                MATCH (a:Agent)
                WHERE a.p_income_level <> '상'
                MATCH (a)-[:HAS_PLAN {{day: date('{POLICY_DAY}')}}]->(:Plan)-[i:INCLUDES]->(:POI)
                WHERE i.actual_satisfaction IS NOT NULL
                WITH a,
                     avg(i.actual_satisfaction) AS sat,
                     sum(CASE WHEN {USED_FILTER} THEN 1 ELSE 0 END) AS policy_uses,
                     count(i) AS visits
                WHERE policy_uses = 0 AND visits >= 2
                RETURN a.id AS id ORDER BY sat ASC LIMIT 50
            """).data()
        else:  # neutral
            # grant 사용했지만 평균 만족도 중간 — "수혜 있었지만 인상 없음"
            rows = s.run(f"""
                MATCH (a:Agent)
                WHERE a.p_income_level <> '상'
                MATCH (a)-[:HAS_PLAN {{day: date('{POLICY_DAY}')}}]->(:Plan)-[i:INCLUDES]->(:POI)
                WHERE i.actual_satisfaction IS NOT NULL
                WITH a,
                     avg(i.actual_satisfaction) AS sat,
                     sum(CASE WHEN {USED_FILTER} THEN 1 ELSE 0 END) AS policy_uses,
                     count(i) AS visits
                WHERE policy_uses >= 1 AND sat >= 0.55 AND sat <= 0.62 AND visits >= 2
                RETURN a.id AS id LIMIT 100
            """).data()
    if not rows:
        return None
    import random as _random
    return _random.choice([r["id"] for r in rows])


def _find_behavior_sample(
    label: str,
    last_day: str,
    *,
    cats: list[str] | None = None,
    districts: list[str] | None = None,
) -> str | None:
    """정책별 trace가 없을 때 실제 방문·만족도 기반으로 인터뷰 샘플을 고른다."""
    if label not in ("positive", "negative", "neutral"):
        return None

    cats = cats or []
    districts = districts or []
    has_cats = bool(cats)
    has_districts = bool(districts)
    min_visits = 1 if (has_cats or has_districts) else 2

    if label == "positive":
        order_clause = (
            "visits DESC, spent DESC, sat DESC"
            if (has_cats or has_districts)
            else "sat DESC, visits DESC, spent DESC"
        )
        sat_clause = "sat >= 0.55"
    elif label == "negative":
        order_clause = "sat ASC, visits DESC, spent DESC"
        sat_clause = "sat <= 0.65"
    else:
        order_clause = "abs(sat - 0.6) ASC, visits DESC, spent DESC"
        sat_clause = "sat >= 0.5 AND sat <= 0.7"

    home_match = ""
    where_terms = [
        "plan.day <= date($last_day)",
        "i.actual_satisfaction IS NOT NULL",
        "NOT i.category IN ['집', '직장']",
    ]
    if has_cats:
        where_terms.append("i.category IN $cats")
    if has_districts:
        home_match = """
            MATCH (a)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(:Dong)
              <-[:HAS_DONG]-(home_dist:District)
        """
        where_terms.append("home_dist.name IN $districts")
    where_clause = "\n              AND ".join(where_terms)

    with driver_session() as s:
        rows = s.run(f"""
            MATCH (a:Agent)-[:HAS_PLAN]->(plan:Plan)
              -[i:INCLUDES]->(poi:POI {{type:'commerce'}})
            {home_match}
            WHERE {where_clause}
            WITH a,
                 avg(i.actual_satisfaction) AS sat,
                 count(i) AS visits,
                 sum(coalesce(i.actual_spent, 0)) AS spent
            WHERE visits >= $min_visits AND {sat_clause}
            RETURN a.id AS id
            ORDER BY {order_clause}
            LIMIT 50
        """, last_day=last_day, cats=cats, districts=districts,
             min_visits=min_visits).data()

    if not rows:
        return None
    return rows[0]["id"]


def find_policy_sample_with_scope(
    ctx: dict | None,
    label: str,
    last_day: str,
) -> tuple[str | None, str]:
    """정책 ctx에 맞는 인터뷰 대표 샘플을 고른다.

    grant 정책은 기존 지원금 사용 trace 기반 선택을 우선한다. 그 외 정책은
    대상 업종·자치구 방문자 중 실제 만족도와 방문 기록이 있는 에이전트를 고른다.
    """
    if ctx and ctx.get("type") == "grant":
        aid = find_label_sample(label, last_day)
        if aid:
            return aid, "policy_target"

    target_cats = []
    target_districts = []
    if ctx:
        target_cats = list(ctx.get("target_cats") or ctx.get("benefit_categories") or [])
        target_districts = [
            d for d in (ctx.get("target_districts") or [])
            if d and d != "서울특별시"
        ]

    aid = _find_behavior_sample(
        label,
        last_day,
        cats=target_cats,
        districts=target_districts,
    )
    if aid:
        return aid, "policy_target"

    return _find_behavior_sample(label, last_day), "general"


def find_policy_sample(ctx: dict | None, label: str, last_day: str) -> str | None:
    aid, _scope = find_policy_sample_with_scope(ctx, label, last_day)
    return aid


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aid", default=None, help="agent ID. --label 함께 쓰면 자동 추출")
    ap.add_argument("--label", default=None, choices=["positive", "negative", "neutral"])
    ap.add_argument("--days", default="2026-05-01,2026-05-02,2026-05-03",
                    help="콤마 구분, 시뮬 일자")
    ap.add_argument("--question", default=None, help="일회성 질문. 없으면 REPL")
    args = ap.parse_args()

    days = args.days.split(",")
    aid = args.aid
    if args.label and not aid:
        aid = find_label_sample(args.label, days[-1])
        if not aid:
            print(f"[!] {args.label} 라벨 군집에서 샘플 없음", file=sys.stderr)
            return
        print(f"[label={args.label}] 자동 추출 agent: {aid}")

    if not aid:
        print("--aid 또는 --label 중 하나 필수", file=sys.stderr)
        return

    print(f"[fetch] {aid} × {len(days)}일 데이터 ...", file=sys.stderr)
    data = fetch_agent_full(aid, days)
    print(f"  persona OK · plans {len(data['plans'])} · memories {len(data['memories'])}"
          f" · conversations {len(data['conversations'])}", file=sys.stderr)

    if args.question:
        ans = ask(data, args.question)
        print(f"\nQ: {args.question}\nA: {ans}")
        return

    # REPL
    print(f"\n=== 인터뷰: {aid} ===\n빈 줄로 종료\n")
    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        print(f"\nA: {ask(data, q)}\n")


if __name__ == "__main__":
    main()
