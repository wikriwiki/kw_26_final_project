"""Night Phase 2 — 의도 분류 LLM 호출 + Conversation 적재.

설계 출처: 노션 "상호작용 의도분류 로직" + "런타임 노드 5종" 매트릭스 그대로.

Memory type 정의 (노션):
  - policy = Watchdog/LangChain이 공식 정책을 그래프에 처음 등록할 때 시드 (별도 모듈)
  - rumor  = 다른 agent한테 전달받은 모든 정보 (이슈·추천 모두)
  - visited = 직접 방문 (Phase 1에서 적재)
  - sns    = SNS 노출 (별도 채널)

입력: night_interaction.select_interaction_pairs()의 출력 (선정된 쌍)
출력 (노션 매트릭스 그대로):

  공통: (a)-[:PARTICIPATES_IN {role:'initiator'}]->(c:Conversation)
        (b)-[:PARTICIPATES_IN {role:'recipient'}]->(c)

  | intent | Conversation                                | Memory{rumor} (recipient)                | Plan 주입            |
  |--------|---------------------------------------------|------------------------------------------|----------------------|
  | 약속   | ✅ should_inject + target_day_offset/time   | ❌                                       | ✅ Dawn ④가 자동 조회|
  | 이슈   | ✅ topic_type=policy/topic_value=정책ID    | ✅ + ABOUT_POLICY(policy면)              | ❌                   |
  | 추천   | ✅ topic_type=poi|category                  | ✅ + MENTIONS_POI + KNOWS_POI{rumor} MERGE | ❌                   |
  | 기타   | ✅ topic_type=none                          | ❌                                       | ❌                   |

importance(이슈·추천 공통) = urgency × 0.6 + relationship × 0.4 (노션 §9)
Memory id = MEM_RUMOR_<recipient>_D<YYYYMMDD>_<n>
"""
from __future__ import annotations

import json
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session  # noqa: E402
from llm_client import call_chat as _llm_call  # noqa: E402

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    raise

# ═══════════════════════════════════════════
# Pydantic 출력 스키마 (노션 신규 — flat topic + 확장 plan_signal)
# ═══════════════════════════════════════════
class PlanSignal(BaseModel):
    should_inject: bool = False
    target_day_offset: int | None = None        # 약속이면 D+N의 N (int)
    target_time: str | None = None              # 약속이면 "HH:MM"
    meeting_location_hint: str | None = None    # 약속이면 장소 힌트 (자유 문자열)


class IntentOutput(BaseModel):
    intent: str = Field(..., description="약속 | 이슈 | 추천 | 기타")
    initiator_id: str
    recipient_id: str
    topic_type: str = Field(..., description="policy | poi | category | none")
    topic_value: str | None = None
    plan_signal: PlanSignal
    # ───────────── 사고과정 흔적 (인터뷰용) ─────────────
    # 두 페르소나의 어떤 요소(친밀도·동선 겹침·정책 노출·라이프스타일 등)가
    # 이 의도로 이끌었는지 1~2문장.
    reasoning: str | None = Field(default=None)

    @field_validator("intent")
    @classmethod
    def _check_intent(cls, v):
        if v not in {"약속", "이슈", "추천", "기타"}:
            raise ValueError(f"invalid intent: {v}")
        return v

    @field_validator("topic_type")
    @classmethod
    def _check_topic_type(cls, v):
        if v not in {"policy", "poi", "category", "none"}:
            raise ValueError(f"invalid topic_type: {v}")
        return v


# ═══════════════════════════════════════════
# 프롬프트 (노션 §3 그대로)
# ═══════════════════════════════════════════
SYSTEM_PROMPT = """너는 Night 단계의 상호작용 의도분류기다.
입력은 이미 매칭된 두 agent 쌍의 상호작용 후보이며, 상대를 다시 선택하지 않는다.
반드시 JSON만 출력한다. 설명, 요약, 코드펜스는 금지한다.

입력 구조:
- MATCHING_ANALYSIS: interaction_score, exposure_score, relationship_score, urgency_score
- AGENT_A, AGENT_B: role, agent_id, job, lifestyle, mood, fatigue, daily_log
- persona 판단에는 agent_id, job, lifestyle, mood, fatigue만 사용한다.
- daily_log 각 줄 형식: time: HH:MM-HH:MM | dong: ... | poi: ... | category: ... | activity: ...
- POLICY_CONTEXT가 있을 수 있다. 이 블록에는 policy id, name, description, aware_a, aware_b가 포함된다.
- POLICY_CONTEXT는 정책 관련 대화 후보를 보여주는 참고 정보이며, 자동으로 "이슈"를 뜻하지 않는다.
- home, workplace, residence 같은 anchor 이벤트도 의도 판단의 단서로 활용한다.

분류 우선순위 및 규칙 (위에서부터 순서대로 적용):
1. 미래에 둘이 다시 만나거나 함께 무엇을 하자는 제안이 핵심이면 → "약속"
   - 실제 대화문은 없으므로, 같은 시간대·동선 겹침·공유 장소 경험·반복 방문·생활 패턴 적합성이 함께 보이면 미래 만남 제안을 적극 추정할 수 있다.
   - 단순한 장소 겹침만으로는 부족하지만, 반복적인 교집합과 생활 패턴 적합성이 함께 있으면 약속 가능성을 높게 본다.

2. 정책, 뉴스, 사건 전달이 핵심이면 → "이슈"
   - 정책/뉴스 노출이나 정보 비대칭이 보이면, 그 내용을 상대에게 전하는 흐름을 우선 검토한다.
   - POLICY_CONTEXT가 있더라도 자동으로 이슈로 보내지 말고, 정책 내용이 두 agent의 당일 동선, 방문 동, 방문 카테고리와 실제로 맞닿을 때만 이슈를 적극 고려한다.
   - 같은 정책이나 뉴스는 여러 agent 사이에서 반복 전파될 수 있으므로, 관련 노출이 있으면 소극적으로 기타로 보내지 말고 이슈를 적극 고려한다.

3. 장소나 행동을 권유하거나, 자신의 경험을 바탕으로 가보라고 하거나 피하는 것이 좋다고 조언하는 흐름이 핵심이면 → "추천"
   - 예: "거기 좋더라, 너도 가봐"
   - 한쪽의 장소 경험, 반복 방문, 선호 카테고리가 다른 한쪽에게 권유나 조언으로 이어질 때 추천이다.
   - 긍정적인 추천뿐 아니라, 가볍게 피하는 것이 좋겠다는 수준의 소극적 조언도 포함할 수 있다.
   - 단, 장소명·카테고리·반복 방문만으로 자동 추천하지 말고, 실제로 상대의 선택에 영향을 주는 흐름이 자연스럽게 추정될 때만 추천이다.
   - 정책/뉴스 전달이 더 핵심이면 추천이 아니라 이슈다.

4. 그 외는 → "기타"
   - 위 세 intent를 뒷받침할 구체적 단서가 부족할 때만 기타다.
   - 단순히 매칭 점수가 낮다는 이유만으로 기타로 보내지 말고, 정책/장소/약속으로 이어질 명확한 주제가 없으면 기타로 분류한다.
   - POLICY_CONTEXT가 보이더라도 대화의 핵심 주제가 불분명하면 기타로 둘 수 있다.

출력 규칙:
- initiator_id: AGENT_A의 agent_id 값
- recipient_id: AGENT_B의 agent_id 값
- topic_type:
    - policy: 정책/뉴스 관련 (예: P001)
    - poi: 특정 장소 관련 (예: C_551092)
    - category: 카테고리·업종 키워드 (예: 한식, 카페)
    - none: 기타 intent 또는 토픽 없음
- topic_value: 입력에 등장한 실제 값만 사용. none일 때는 null
- plan_signal.should_inject: 약속일 때만 true, 나머지는 false
- plan_signal.target_day_offset: 약속일 때 정수 — **D+1 ~ D+7 사이에서 자연스럽게 분포**.
    · 급한 약속(점심·당장 만남): D+1
    · 평일 저녁 회식·동료 약속: D+2 ~ D+4
    · 주말 친구·가족 약속: D+3 ~ D+5
    · 멀리 잡는 약속(생일·기념일·이벤트): D+6 ~ D+7
    실제 사회 관계처럼 D+1 한 군데로 쏠리지 않게 다양한 offset을 사용한다.
    약속은 시뮬레이션 종료일 안에 들어오는 경우만 고르지 말고, 대화 흐름상 자연스러운 날짜를 고른다.
- plan_signal.target_time: 약속일 때 "HH:MM" 형식 (예: "19:00"). 나머지는 null
- plan_signal.meeting_location_hint: 약속 장소 자유 문자열 힌트 (예: "봉피양 역삼본점"). 없거나 약속이 아니면 null

intent별 필드 매핑 규칙:
- 약속: topic_type ∈ {poi, category, none} / plan_signal 4 필드 필수
- 이슈: topic_type = policy / topic_value = 정책 ID / plan_signal 전부 null·false
- 추천: topic_type ∈ {poi, category} / plan_signal 전부 null·false
- 기타: topic_type = none / topic_value = null / plan_signal 전부 null·false

[reasoning — 인터뷰 가능성을 위한 핵심]
reasoning 필드에 **두 agent의 어떤 페르소나·일정·정책 요소가** 이 의도로 이끌었는지 1~2문장 명시한다.
- 약속 → 누가 누구에게, 왜 (예: "AGT_A는 동료 친밀도 0.6 + 같은 동 점심 자주 겹쳐서 토요일 외식 제안.")
- 이슈 → 어떤 정책·뉴스가 화제 (예: "정책 P001이 두 사람의 강남역 일대 동선과 맞닿아 보행환경 개선 이야기가 화제로 이어짐.")
- 추천 → 누가 어디를 권유 (예: "AGT_A가 어제 만족도 0.7로 만족한 두부마을찬을 AGT_B에게 추천.")
- 기타 → 일상 잡담 사유 (예: "라이프스타일·일정 모두 무관한 동선 겹침. 간단한 인사 수준.")

출력 JSON 스키마:
{
  "intent": "약속 | 이슈 | 추천 | 기타",
  "initiator_id": "<AGENT_A의 agent_id>",
  "recipient_id": "<AGENT_B의 agent_id>",
  "topic_type": "policy | poi | category | none",
  "topic_value": "<입력에서 추출한 값 또는 null>",
  "plan_signal": {
    "should_inject": false,
    "target_day_offset": null,
    "target_time": null,
    "meeting_location_hint": null
  },
  "reasoning": "<두 페르소나·일정 요소가 이 intent로 이끈 사유 1~2문장>"
}
/no_think"""


# ═══════════════════════════════════════════
# 입력 빌더 (노션 §1 입력 계약)
# ═══════════════════════════════════════════
FETCH_PAIR_INPUT_CYPHER = """
UNWIND $pairs AS pair
MATCH (a:Agent {id:pair.aid_a}), (b:Agent {id:pair.aid_b})
OPTIONAL MATCH (a)-[:HAS_STATE {day:date($d)}]->(sa:State)
OPTIONAL MATCH (b)-[:HAS_STATE {day:date($d)}]->(sb:State)
OPTIONAL MATCH (a)-[:HAS_PLAN {day:date($d)}]->(pa:Plan)
OPTIONAL MATCH (b)-[:HAS_PLAN {day:date($d)}]->(pb:Plan)
RETURN
  pair.aid_a AS aid_a, pair.aid_b AS aid_b,
  pair.score AS score, pair.exposure AS exp, pair.relationship AS rel, pair.urgency AS urg,
  a.personal_job_raw AS a_job, a.personality_lifestyle_raw AS a_life,
  sa.mood AS a_mood, sa.fatigue AS a_fatigue, sa.policy_lifecycle AS a_policy_lc,
  b.personal_job_raw AS b_job, b.personality_lifestyle_raw AS b_life,
  sb.mood AS b_mood, sb.fatigue AS b_fatigue, sb.policy_lifecycle AS b_policy_lc,
  pa.id AS plan_a_id, pb.id AS plan_b_id
"""

FETCH_EVENTS_CYPHER = """
UNWIND $plan_ids AS pid
MATCH (p:Plan {id:pid})-[i:INCLUDES]->(poi:POI)-[:IN_DONG]->(d:Dong)
RETURN pid AS plan_id, i.order AS ord, toString(i.time) AS time,
       i.anchor AS anchor, i.category AS cat, i.sub_category AS sub,
       i.intent AS intent,
       poi.id AS poi_id, poi.name AS poi_name, poi.type AS poi_type,
       d.code AS dong_code, d.name AS dong_name
ORDER BY pid, ord
"""

FETCH_POLICY_CYPHER = """
UNWIND $agent_ids AS aid
MATCH (a:Agent {id: $aid})-[:LIVES_AT|WORKS_AT]->(:POI)-[:IN_DONG]->(d:Dong)<-[:HAS_DONG]-(dist:District)
WITH aid, collect(DISTINCT d) AS my_dongs, collect(DISTINCT dist) AS my_dists
MATCH (pol:Policy)-[:applied_to]->(target)
WHERE date($d) >= pol.effective_from AND date($d) <= pol.effective_until
  AND (target IN my_dongs OR target IN my_dists)
WITH aid, DISTINCT pol
OPTIONAL MATCH (pol)-[:applied_to]->(reg)
WITH aid, pol, collect(DISTINCT coalesce(reg.code, '')) AS region_codes
OPTIONAL MATCH (pol)-[:targets]->(cat:Category)
WITH aid, pol, region_codes, collect(DISTINCT cat.parent) AS target_l1s
RETURN aid, pol.id AS id, pol.name AS name, pol.description AS description,
       region_codes, target_l1s
ORDER BY aid, id
"""


def fetch_pair_data(pairs: list[dict], day: date) -> dict:
    """각 쌍에 필요한 입력 데이터를 한 번에 fetch."""
    pair_data = {}
    plan_ids = []
    agent_ids = set()
    BATCH = 500
    with driver_session() as s:
        for i in range(0, len(pairs), BATCH):
            for r in s.run(FETCH_PAIR_INPUT_CYPHER, pairs=pairs[i:i+BATCH], d=day.isoformat()):
                key = (r["aid_a"], r["aid_b"])
                pair_data[key] = {
                    "score": r["score"], "exp": r["exp"], "rel": r["rel"], "urg": r["urg"],
                    "a": {"job": r["a_job"] or "", "life": r["a_life"] or "",
                          "mood": r["a_mood"], "fatigue": r["a_fatigue"],
                          "policy_lifecycle": r["a_policy_lc"] or "{}"},
                    "b": {"job": r["b_job"] or "", "life": r["b_life"] or "",
                          "mood": r["b_mood"], "fatigue": r["b_fatigue"],
                          "policy_lifecycle": r["b_policy_lc"] or "{}"},
                    "plan_a_id": r["plan_a_id"], "plan_b_id": r["plan_b_id"],
                    "events_a": [], "events_b": [],
                    "policies_a": [], "policies_b": [],
                }
                agent_ids.add(r["aid_a"])
                agent_ids.add(r["aid_b"])
                if r["plan_a_id"]: plan_ids.append(r["plan_a_id"])
                if r["plan_b_id"]: plan_ids.append(r["plan_b_id"])

        # 이벤트 (Plan별)
        events_by_plan = {}
        for i in range(0, len(plan_ids), BATCH):
            for r in s.run(FETCH_EVENTS_CYPHER, plan_ids=plan_ids[i:i+BATCH]):
                events_by_plan.setdefault(r["plan_id"], []).append(dict(r))

        policies_by_agent = {}
        agent_id_list = sorted(agent_ids)
        for i in range(0, len(agent_id_list), BATCH):
            for r in s.run(FETCH_POLICY_CYPHER, agent_ids=agent_id_list[i:i+BATCH], d=day.isoformat()):
                policies_by_agent.setdefault(r["aid"], []).append({
                    "id": r["id"],
                    "name": r["name"] or "",
                    "description": r["description"] or "",
                    "region_codes": [x for x in (r["region_codes"] or []) if x],
                    "target_l1s": [x for x in (r["target_l1s"] or []) if x],
                })

    # pair_data에 이벤트 매핑
    for key, d in pair_data.items():
        d["events_a"] = events_by_plan.get(d["plan_a_id"], [])
        d["events_b"] = events_by_plan.get(d["plan_b_id"], [])
        d["policies_a"] = policies_by_agent.get(key[0], [])
        d["policies_b"] = policies_by_agent.get(key[1], [])
    return pair_data


def _format_event_line(ev: dict) -> str:
    poi = ev["poi_name"] or ev["poi_id"]
    return (f"  - time: {ev['time'][:5]}~ | dong: {ev['dong_name']} | poi: {poi} | "
            f"category: {ev['cat'] or '?'} | activity: {ev['intent'] or '?'}")


def _parse_policy_lifecycle(raw) -> set[str]:
    if not raw:
        return set()
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return set()
    if not isinstance(parsed, dict):
        return set()
    return {str(k) for k, v in parsed.items() if v}


def _select_relevant_policies(data: dict) -> list[dict]:
    aware_a = _parse_policy_lifecycle(data["a"].get("policy_lifecycle"))
    aware_b = _parse_policy_lifecycle(data["b"].get("policy_lifecycle"))
    event_dongs = {
        e.get("dong_code") for e in (data.get("events_a") or []) + (data.get("events_b") or [])
        if e.get("dong_code")
    }
    event_cats = {
        e.get("cat") for e in (data.get("events_a") or []) + (data.get("events_b") or [])
        if e.get("cat")
    }
    selected = []
    seen = set()
    for pol in (data.get("policies_a") or []) + (data.get("policies_b") or []):
        pid = pol.get("id")
        if not pid or pid in seen:
            continue
        aware_a_hit = pid in aware_a
        aware_b_hit = pid in aware_b
        relevant = bool(set(pol.get("region_codes") or []) & event_dongs) or bool(
            set(pol.get("target_l1s") or []) & event_cats
        )
        if (aware_a_hit or aware_b_hit) and relevant:
            selected.append({
                "id": pid,
                "name": pol.get("name") or "",
                "description": pol.get("description") or "",
                "aware_a": aware_a_hit,
                "aware_b": aware_b_hit,
            })
            seen.add(pid)
        if len(selected) >= 2:
            break
    return selected


def _format_policy_context(data: dict) -> str:
    policies = _select_relevant_policies(data)
    if not policies:
        return ""
    lines = ["### [POLICY_CONTEXT]"]
    for pol in policies:
        desc = " ".join((pol["description"] or "").split())
        if len(desc) > 180:
            desc = desc[:177].rstrip() + "..."
        lines.append(f"- {pol['id']} | {pol['name']}")
        lines.append(f"  description: {desc}")
        lines.append(f"  aware_a: {str(pol['aware_a']).lower()}")
        lines.append(f"  aware_b: {str(pol['aware_b']).lower()}")
    return "\n".join(lines)


def build_user_block(pair_key: tuple[str, str], data: dict) -> str:
    aid_a, aid_b = pair_key
    a = data["a"]; b = data["b"]
    a_mood = a['mood'] if a['mood'] is not None else 0.5
    a_fatigue = a['fatigue'] if a['fatigue'] is not None else 0.3
    b_mood = b['mood'] if b['mood'] is not None else 0.5
    b_fatigue = b['fatigue'] if b['fatigue'] is not None else 0.3
    ev_a = "\n".join(_format_event_line(e) for e in data["events_a"]) or "  (이벤트 없음)"
    ev_b = "\n".join(_format_event_line(e) for e in data["events_b"]) or "  (이벤트 없음)"
    policy_block = _format_policy_context(data)
    policy_section = f"\n\n{policy_block}" if policy_block else ""
    return f"""### [MATCHING_ANALYSIS]
- interaction_score: {data['score']:.2f}
- exposure_score: {data['exp']:.2f}
- relationship_score: {data['rel']:.2f}
- urgency_score: {data['urg']:.2f}

### [AGENT_A]
- role: initiator
- agent_id: {aid_a}
- job: {a['job'][:60]}
- lifestyle: {(a['life'] or '')[:100]}
- mood: {a_mood:.2f}
- fatigue: {a_fatigue:.2f}
- daily_log:
{ev_a}

### [AGENT_B]
- role: recipient
- agent_id: {aid_b}
- job: {b['job'][:60]}
- lifestyle: {(b['life'] or '')[:100]}
- mood: {b_mood:.2f}
- fatigue: {b_fatigue:.2f}
- daily_log:
{ev_b}{policy_section}

위 입력만 근거로 상호작용 의도를 1개 분류하고 JSON만 출력하라."""


# ═══════════════════════════════════════════
# LLM 호출 (SGLang/vLLM auto-detect via llm_client)
# ═══════════════════════════════════════════


def _extract_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m: return m.group(1)
    s = text.find("{"); e = text.rfind("}")
    if s < 0 or e < 0: raise ValueError("no JSON found")
    return text[s:e+1]


def classify_intent(pair_key: tuple[str, str], data: dict, max_retry: int = 2) -> dict | None:
    """한 쌍에 대해 의도 분류 LLM 호출. data에 보관된 score(exp/rel/urg)를
    결과에 함께 실어 importance 계산에 사용."""
    user = build_user_block(pair_key, data)
    last_err = None
    for attempt in range(max_retry + 1):
        temp = 0.3 + 0.2 * attempt   # 의도 분류는 더 결정론적으로
        try:
            resp = _llm_call(
                None, SYSTEM_PROMPT, user,
                temperature=temp, max_tokens=600,  # reasoning 필드 추가
            )
            raw = resp.choices[0].message.content
            data_json = json.loads(_extract_json(raw))
            parsed = IntentOutput.model_validate(data_json)
            return {
                "intent": parsed.intent,
                "initiator_id": parsed.initiator_id,
                "recipient_id": parsed.recipient_id,
                "topic_type": parsed.topic_type,
                "topic_value": parsed.topic_value,
                "should_inject": parsed.plan_signal.should_inject,
                "target_day_offset": parsed.plan_signal.target_day_offset,
                "target_time": parsed.plan_signal.target_time,
                "meeting_location_hint": parsed.plan_signal.meeting_location_hint,
                "reasoning": parsed.reasoning,   # ← Conversation.reasoning + Memory.summary 로 흐름
                # 매칭 점수(importance 계산용 — 노션 §9)
                "exposure_score": data.get("exp", 0.0),
                "relationship_score": data.get("rel", 0.0),
                "urgency_score": data.get("urg", 0.0),
                "tokens_in": resp.usage.prompt_tokens,
                "tokens_out": resp.usage.completion_tokens,
                "attempt": attempt,
            }
        except Exception as e:
            last_err = e
    return {"error": str(last_err)[:200], "pair": list(pair_key)}


# ═══════════════════════════════════════════
# 적재 — 노션 §4·§5·§9·§10 그대로
# ═══════════════════════════════════════════

# Conversation 베이스 — 모든 intent 공통 (노션 §4 + PARTICIPATES_IN.role)
# day는 다른 노드(Plan/State/Memory)와 일관성 위해 date 유지
# target_day_offset만 별도 int로 (Dawn에서 date(c.day) + duration({days:offset}) 비교)
CREATE_CONVERSATION_CYPHER = """
UNWIND $rows AS r
MATCH (a:Agent {id:r.initiator})
MATCH (b:Agent {id:r.recipient})
WITH r, a, b
CREATE (c:Conversation {
  id: r.cid,
  day: date(r.day),
  intent: r.intent,
  initiator_id: r.initiator,
  recipient_id: r.recipient,
  topic_type: r.topic_type,
  topic_value: r.topic_value,
  should_inject: r.should_inject,
  target_day_offset: r.target_day_offset,
  target_time: r.target_time,
  meeting_location_hint: r.meeting_location_hint,
  // 사고과정 흔적 (인터뷰용)
  reasoning: r.reasoning
})
CREATE (a)-[:PARTICIPATES_IN {role:'initiator'}]->(c)
CREATE (b)-[:PARTICIPATES_IN {role:'recipient'}]->(c)
"""

# 이슈·추천 공통 — Memory{rumor} + REMEMBERS + FROM_CONVERSATION (노션 §5)
# id 형식: MEM_RUMOR_<recipient>_D<day>_<n>
# importance = urgency × 0.6 + relationship × 0.4 (노션 §9)
# source = initiator_id / topic_type·topic_value는 Conversation에서 복사
LINK_RUMOR_MEMORY_CYPHER = """
UNWIND $rows AS r
MATCH (c:Conversation {id:r.cid})
MATCH (b:Agent {id:r.recipient})
CREATE (m:Memory {
  id: r.mem_id,
  type: 'rumor',
  day: c.day,
  source: r.initiator,
  topic_type: c.topic_type,
  topic_value: c.topic_value,
  importance: r.importance,
  // Conversation의 reasoning을 그대로 복사 — Memory에서 직접 인용 가능
  summary: c.reasoning
})
CREATE (b)-[:REMEMBERS {day: c.day}]->(m)
CREATE (m)-[:FROM_CONVERSATION]->(c)
"""

# 추천 의도 추가 효과 — MENTIONS_POI + KNOWS_POI{rumor} MERGE + Memory.ABOUT_POI
# (recipient의 인지 풀에 신규 POI 추가)
LINK_RECOMMEND_EXTRA_CYPHER = """
UNWIND $rows AS r
MATCH (c:Conversation {id:r.cid})
MATCH (b:Agent {id:r.recipient})
MATCH (m:Memory {id:r.mem_id})
WITH r, c, b, m
OPTIONAL MATCH (poi:POI) WHERE r.topic_type = 'poi'
  AND (poi.id = r.topic_value OR poi.name = r.topic_value)
WITH r, c, b, m, poi
ORDER BY poi.id LIMIT 1
FOREACH (_ IN CASE WHEN poi IS NOT NULL THEN [1] ELSE [] END |
  MERGE (c)-[:MENTIONS_POI]->(poi)
  MERGE (b)-[kp:KNOWS_POI]->(poi)
    ON CREATE SET kp.since = c.day, kp.source = 'rumor',
                  kp.visit_count = 0, kp.affinity = 0.5
    ON MATCH  SET kp.affinity = kp.affinity + (1.0 - kp.affinity) * 0.15
  CREATE (m)-[:ABOUT_POI]->(poi)
)
"""

# 이슈 의도 추가 효과 — topic_type=policy면 Conversation → Policy 엣지
LINK_ISSUE_EXTRA_CYPHER = """
UNWIND $rows AS r
MATCH (c:Conversation {id:r.cid})
OPTIONAL MATCH (pol:Policy {id:r.topic_value}) WHERE r.topic_type = 'policy'
FOREACH (_ IN CASE WHEN pol IS NOT NULL THEN [1] ELSE [] END |
  MERGE (c)-[:ABOUT_POLICY]->(pol)
)
"""

# 약속 의도 — meeting_location_hint가 실제 POI에 매칭되면 MENTIONS_POI 추가
# (Plan 주입은 Dawn ④가 자동 조회 — 여기선 Conversation에 plan_signal만 저장됨)
LINK_APPOINTMENT_EXTRA_CYPHER = """
UNWIND $rows AS r
MATCH (c:Conversation {id:r.cid})
OPTIONAL MATCH (poi:POI) WHERE r.meeting_hint IS NOT NULL
  AND (poi.name = r.meeting_hint OR poi.id = r.meeting_hint)
WITH c, poi ORDER BY poi.id LIMIT 1
FOREACH (_ IN CASE WHEN poi IS NOT NULL THEN [1] ELSE [] END |
  MERGE (c)-[:MENTIONS_POI]->(poi)
)
"""


def write_conversations(day: date, results: list[dict]):
    """의도 분류 결과 → Conversation 노드 + intent별 후속 엣지·노드 (노션 §4·§5·§9·§10).

    intent별 처리:
      - 약속: Conversation(+ plan_signal) → meeting_hint가 POI이면 MENTIONS_POI
      - 이슈: Conversation → Memory{rumor} + REMEMBERS + FROM_CONVERSATION
              + topic_type=policy면 ABOUT_POLICY 엣지
      - 추천: Conversation → Memory{rumor} + REMEMBERS + FROM_CONVERSATION
              + MENTIONS_POI + KNOWS_POI{rumor} MERGE + Memory.ABOUT_POI
      - 기타: Conversation만 (노션 §4 — 기타는 Conversation만 적재)
    """
    base_rows, rumor_rows, recommend_rows, issue_rows, appt_rows = [], [], [], [], []

    # recipient별 Memory id 카운터 (MEM_RUMOR_<recipient>_D<day>_<n>)
    mem_seq: dict[str, int] = {}
    day_iso = day.isoformat()
    day_tag = day_iso.replace("-", "")   # D20260515 식

    for r in results:
        if "error" in r:
            continue
        intent = r["intent"]
        cid = f"conv_{uuid.uuid4().hex[:16]}"

        # Conversation 베이스 (노션 §4 — 모든 intent 공통)
        base_rows.append({
            "cid": cid, "day": day_iso,
            "intent": intent,
            "initiator": r["initiator_id"],
            "recipient": r["recipient_id"],
            "topic_type": r["topic_type"],
            "topic_value": r.get("topic_value"),
            "should_inject": bool(r.get("should_inject")),
            "target_day_offset": r.get("target_day_offset"),
            "target_time": r.get("target_time"),
            "meeting_location_hint": r.get("meeting_location_hint"),
            # 사고과정 흔적 (인터뷰 인용용)
            "reasoning": r.get("reasoning"),
        })

        # intent별 분기
        if intent in ("이슈", "추천"):
            # Memory{rumor} 생성 — 두 intent 모두 recipient에 적재 (노션 §5)
            recipient = r["recipient_id"]
            mem_seq[recipient] = mem_seq.get(recipient, 0) + 1
            mem_id = f"MEM_RUMOR_{recipient}_D{day_tag}_{mem_seq[recipient]}"

            # importance = urgency × 0.6 + relationship × 0.4 (노션 §9)
            imp = (r.get("urgency_score", 0.0) * 0.6
                   + r.get("relationship_score", 0.0) * 0.4)
            imp = round(imp, 2)

            rumor_rows.append({
                "cid": cid, "mem_id": mem_id,
                "initiator": r["initiator_id"],
                "recipient": recipient,
                "importance": imp,
            })

            if intent == "추천":
                recommend_rows.append({
                    "cid": cid, "mem_id": mem_id,
                    "recipient": recipient,
                    "topic_type": r["topic_type"],
                    "topic_value": r.get("topic_value"),
                })
            else:  # 이슈
                issue_rows.append({
                    "cid": cid,
                    "topic_type": r["topic_type"],
                    "topic_value": r.get("topic_value"),
                })
        elif intent == "약속":
            appt_rows.append({
                "cid": cid,
                "meeting_hint": r.get("meeting_location_hint"),
            })
        # "기타"는 base만 — 노션 §4

    if not base_rows:
        return {"created": 0}

    with driver_session() as s:
        s.run(CREATE_CONVERSATION_CYPHER, rows=base_rows)
        if rumor_rows:
            s.run(LINK_RUMOR_MEMORY_CYPHER, rows=rumor_rows)
        if recommend_rows:
            s.run(LINK_RECOMMEND_EXTRA_CYPHER, rows=recommend_rows)
        if issue_rows:
            s.run(LINK_ISSUE_EXTRA_CYPHER, rows=issue_rows)
        if appt_rows:
            s.run(LINK_APPOINTMENT_EXTRA_CYPHER, rows=appt_rows)

    by_intent: dict[str, int] = {}
    for row in base_rows:
        by_intent[row["intent"]] = by_intent.get(row["intent"], 0) + 1
    return {
        "created": len(base_rows),
        "by_intent": by_intent,
        "rumor_memory": len(rumor_rows),
        "recommend_extra": len(recommend_rows),
        "issue_extra": len(issue_rows),
        "appointment_extra": len(appt_rows),
    }


# ═══════════════════════════════════════════
# 메인 — 쌍 list 받아서 분류 → 적재
# ═══════════════════════════════════════════
def run_intent_classification(
    day: date,
    pairs: list[dict],
    workers: int = 16,
    verbose: bool = True,
) -> dict:
    if not pairs:
        return {"processed": 0}
    t0 = time.time()
    if verbose:
        print(f"[Intent] fetching pair data for {len(pairs)} pairs ...")
    pair_data = fetch_pair_data(pairs, day)
    if verbose:
        print(f"  pair data fetched: {len(pair_data)}")

    if verbose:
        print(f"[Intent] LLM classification with {workers} workers ...")
    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for (a, b), d in pair_data.items():
            futs[ex.submit(classify_intent, (a, b), d)] = (a, b)
        for fut in as_completed(futs):
            results.append(fut.result())
            completed += 1
            if verbose and completed % max(10, len(pair_data)//10) == 0:
                print(f"  {completed}/{len(pair_data)} ({time.time()-t0:.0f}s)")

    ok = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]
    if verbose:
        print(f"[Intent] LLM done: {len(ok)} ok, {len(err)} err ({time.time()-t0:.0f}s)")

    write_stats = write_conversations(day, ok)
    if verbose:
        print(f"[Intent] adapted: {write_stats}")
    return {"processed": len(ok), "errors": len(err),
            "write": write_stats, "elapsed": time.time()-t0,
            "samples": ok[:5]}


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    import os as _os
    DEFAULT_SIM_OUT = _os.environ.get("SIM_OUTPUT_DIR",
                                       _os.path.expanduser("~/sim_output"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2026-05-02")
    ap.add_argument("--pairs", default=None,
                    help=f"night_interaction.py가 dump한 JSON. "
                         f"기본: $SIM_OUTPUT_DIR/interactions_<day>.json "
                         f"(현재 SIM_OUTPUT_DIR={DEFAULT_SIM_OUT})")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    day = date.fromisoformat(args.day)
    pairs_path = args.pairs or f"{DEFAULT_SIM_OUT}/interactions_{args.day}.json"
    pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
    if args.limit:
        pairs = pairs[:args.limit]

    out = run_intent_classification(day, pairs, workers=args.workers)
    print(f"\n=== 결과 요약 ===")
    print(json.dumps({k: v for k, v in out.items() if k != "samples"},
                     ensure_ascii=False, indent=2))
    print(f"\n=== 샘플 5건 ===")
    for s in out.get("samples", [])[:5]:
        plan_bits = ""
        if s.get("should_inject"):
            plan_bits = (f", inject=D+{s.get('target_day_offset')}"
                         f" @ {s.get('target_time')}"
                         f" hint={s.get('meeting_location_hint')!r}")
        print(f"  {s['initiator_id']} → {s['recipient_id']}: intent={s['intent']}, "
              f"topic={s['topic_type']}:{s.get('topic_value')}{plan_bits}")
