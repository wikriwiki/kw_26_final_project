"""
룰 기반 일간 행동 엔진

LLM 주간 지침 + daily_memory → 일간 소비 행동 생성.
세그먼트별 패턴, 업종 선택, 좌표 이동, 기분/피로도 관리.
"""
import numpy as np

# ═══════════════════════════════════════════
# 세그먼트별 기본 행동 패턴
# ═══════════════════════════════════════════

SEGMENT_PATTERNS = {
    "commuter": {
        "weekday": {
            "lunch":  {"prob": 0.85, "time": (11, 13), "dong": "work", "budget": 0.30},
            "cafe":   {"prob": 0.40, "time": (14, 16), "dong": "work", "budget": 0.15},
            "dinner": {"prob": 0.25, "time": (18, 21), "dong": "any",  "budget": 0.50},
        },
        "friday": {
            "lunch":  {"prob": 0.85, "time": (11, 13), "dong": "work", "budget": 0.30},
            "cafe":   {"prob": 0.45, "time": (14, 16), "dong": "work", "budget": 0.15},
            "dinner": {"prob": 0.55, "time": (18, 23), "dong": "any",  "budget": 0.70},
        },
        "weekend": {
            "meal":     {"prob": 0.60, "time": (11, 20), "dong": "home", "budget": 0.50},
            "shopping": {"prob": 0.25, "time": (13, 18), "dong": "any",  "budget": 0.40},
        },
    },
    "resident": {
        "weekday": {
            "meal":    {"prob": 0.50, "time": (11, 20), "dong": "home", "budget": 0.40},
            "grocery": {"prob": 0.30, "time": (16, 19), "dong": "home", "budget": 0.30},
            "cafe":    {"prob": 0.20, "time": (10, 15), "dong": "home", "budget": 0.15},
        },
        "friday": {
            "meal":    {"prob": 0.55, "time": (11, 20), "dong": "home", "budget": 0.40},
            "dinner":  {"prob": 0.40, "time": (18, 22), "dong": "any",  "budget": 0.55},
        },
        "weekend": {
            "meal":     {"prob": 0.70, "time": (11, 20), "dong": "home", "budget": 0.50},
            "shopping": {"prob": 0.35, "time": (13, 18), "dong": "any",  "budget": 0.40},
        },
    },
    "weekend_visitor": {
        "weekday": {
            "meal": {"prob": 0.10, "time": (11, 20), "dong": "home", "budget": 0.30},
        },
        "friday": {
            "meal":   {"prob": 0.15, "time": (11, 20), "dong": "home", "budget": 0.30},
            "dinner": {"prob": 0.30, "time": (18, 23), "dong": "work", "budget": 0.50},
        },
        "weekend": {
            "meal":          {"prob": 0.80, "time": (11, 21), "dong": "work", "budget": 0.40},
            "cafe":          {"prob": 0.60, "time": (14, 17), "dong": "work", "budget": 0.20},
            "shopping":      {"prob": 0.50, "time": (13, 19), "dong": "work", "budget": 0.40},
            "entertainment": {"prob": 0.30, "time": (18, 23), "dong": "work", "budget": 0.30},
        },
    },
    "evening_visitor": {
        "weekday": {
            "dinner": {"prob": 0.60, "time": (18, 23), "dong": "work", "budget": 0.55},
            "bar":    {"prob": 0.25, "time": (20, 24), "dong": "work", "budget": 0.40},
        },
        "friday": {
            "dinner": {"prob": 0.75, "time": (18, 23), "dong": "work", "budget": 0.60},
            "bar":    {"prob": 0.45, "time": (20, 24), "dong": "work", "budget": 0.45},
        },
        "weekend": {
            "dinner":        {"prob": 0.75, "time": (17, 23), "dong": "work", "budget": 0.50},
            "bar":           {"prob": 0.40, "time": (20, 24), "dong": "work", "budget": 0.40},
            "entertainment": {"prob": 0.35, "time": (19, 23), "dong": "work", "budget": 0.30},
        },
    },
}

# 활동 유형 → 업종 매핑
ACTIVITY_INDUSTRIES = {
    "lunch":         ["한식", "중식", "일식", "분식", "패스트푸드"],
    "dinner":        ["한식", "중식", "일식", "치킨", "피자", "주류"],
    "meal":          ["한식", "중식", "일식", "분식", "패스트푸드", "베이커리"],
    "cafe":          ["카페", "베이커리"],
    "bar":           ["주류", "치킨"],
    "grocery":       ["슈퍼마켓", "편의점"],
    "shopping":      ["패션잡화", "전자제품"],
    "entertainment": ["문화여가", "숙박"],
}

# 업종별 건당 평균 소비금액
INDUSTRY_SPEND = {
    "한식": 12000, "중식": 15000, "일식": 22000, "카페": 6000,
    "편의점": 5000, "패스트푸드": 8000, "패션잡화": 35000, "의료": 25000,
    "주유": 60000, "슈퍼마켓": 18000, "미용": 20000, "문화여가": 15000,
    "주류": 30000, "베이커리": 7000, "치킨": 20000, "피자": 25000,
    "분식": 7000, "전자제품": 50000, "숙박": 80000, "교육": 30000,
    "디저트": 8000, "미용": 20000,
}

DAYS_OF_WEEK = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# ═══════════════════════════════════════════
# 라이프스타일 → 업종 보너스
# ═══════════════════════════════════════════

LIFESTYLE_INDUSTRY_BONUS = {
    "카페러버": {"카페": 2.0, "베이커리": 1.5, "디저트": 2.0},
    "미식가": {"한식": 1.3, "일식": 1.5, "중식": 1.3, "주류": 1.2},
    "가성비추구": {"패스트푸드": 1.5, "분식": 1.5, "편의점": 1.3},
    "건강지향": {"슈퍼마켓": 1.5, "한식": 1.3, "의료": 1.2},
    "쇼핑중독": {"패션잡화": 2.0, "전자제품": 1.5},
    "문화예술": {"문화여가": 2.0, "카페": 1.3, "숙박": 1.3},
    "집순이": {"편의점": 1.5, "슈퍼마켓": 1.3},
    "야식파": {"치킨": 1.8, "주류": 1.5, "피자": 1.5},
}

# ═══════════════════════════════════════════
# 3단계 뉴스 인지 모델
#   AWARE(2)  = 직접 인지 → 풀 boost + 핫스팟 이동
#   HEARD(1)  = 입소문 → boost × 0.4, 이동 없음
#   UNAWARE(0)= 모름 → 영향 없음
# ═══════════════════════════════════════════

AWARENESS_UNAWARE = 0
AWARENESS_HEARD = 1
AWARENESS_AWARE = 2


def _agent_awareness_level(agent, event, rng):
    """에이전트의 뉴스 인지 수준 판정 (확률적).

    Returns: 0(UNAWARE), 1(HEARD), 2(AWARE)
    """
    cat = event.get("category", "")
    age = str(agent.get("age_group", ""))
    target_demo = event.get("target_demo", [])
    sns = agent.get("sns_activity", 0.3)
    trend = agent.get("trend_sensitivity", 0.5)
    social_w = agent.get("social_influence_weight", 0.3)

    # 이미 소셜 전파로 들은 경우 (memory에 기록됨)
    aid = agent.get("agent_id", "")
    headline = event.get("headline", "")
    awareness_map = agent.get("_news_awareness", {})
    if headline in awareness_map:
        return awareness_map[headline]

    # target_demo가 지정되어 있으면 비대상 연령은 인지 확률 대폭 감소
    demo_penalty = 1.0
    if target_demo and age not in target_demo:
        demo_penalty = 0.2  # 대상 아닌 연령도 20% 확률로 볼 수 있음

    # 카테고리별 직접 인지(AWARE) 기본 확률
    if cat == "SNS_VIRAL":
        aware_prob = sns * 0.7      # SNS 활동도↑ → 볼 확률↑
    elif cat == "ENTERTAINMENT":
        interests = agent.get("interests", [])
        has_interest = any(i in interests for i in ["K-POP", "공연", "음악", "문화여가"])
        aware_prob = 0.5 if has_interest else trend * 0.3
    elif cat == "FOOD_TREND":
        aware_prob = max(trend * 0.5, sns * 0.4)
    elif cat == "REAL_ESTATE":
        aware_prob = agent.get("price_sensitivity", 0.5) * 0.4
    elif cat == "SUBSIDY":
        aware_prob = 0.35  # 정책은 모두 볼 수 있지만 확정은 아님
    elif cat == "POLICY_ANNOUNCE":
        aware_prob = 0.40
    elif cat == "SEASONAL":
        aware_prob = 0.60  # 계절 변화는 많은 사람이 체감
    elif cat == "REDEVELOPMENT":
        aware_prob = 0.45
    else:
        aware_prob = 0.30

    aware_prob *= demo_penalty

    # 판정
    roll = rng.random()
    if roll < aware_prob:
        return AWARENESS_AWARE
    elif roll < aware_prob + social_w * 0.3:  # 소셜 영향력이 높으면 입소문 들을 확률↑
        return AWARENESS_HEARD
    return AWARENESS_UNAWARE


def _get_event_modifiers(agent, event, awareness=AWARENESS_AWARE):
    """이벤트가 에이전트에게 주는 수치적 영향 (인지 수준에 따라 차등)."""
    if awareness == AWARENESS_UNAWARE:
        return {
            "spending_boost": 1.0, "explore_boost": 0.0,
            "location_pull": None, "pop_boost": 1.0,
            "affected_industries": [],
        }

    cat = event.get("category", "")
    spending_boost = event.get("spending_boost", 1.0)
    pop_boost = event.get("population_boost", 1.0)
    explore_boost = 0.0
    location_pull = None

    if cat == "SNS_VIRAL":
        sns = agent.get("sns_activity", 0.3)
        spending_boost = 1.0 + (spending_boost - 1.0) * sns
        explore_boost = 0.3 * sns
    elif cat == "ENTERTAINMENT":
        spending_boost = spending_boost
        explore_boost = 0.2
    elif cat == "REAL_ESTATE":
        ps = agent.get("price_sensitivity", 0.5)
        spending_boost = 1.0 - (1.0 - spending_boost) * ps
    elif cat == "REDEVELOPMENT":
        spending_boost = spending_boost
    elif cat == "FOOD_TREND":
        trend = agent.get("trend_sensitivity", 0.5)
        spending_boost = 1.0 + (spending_boost - 1.0) * trend
        explore_boost = 0.25 * trend

    # HEARD = 약한 반응 (boost의 40%, 이동 없음)
    if awareness == AWARENESS_HEARD:
        spending_boost = 1.0 + (spending_boost - 1.0) * 0.4
        explore_boost *= 0.3
        location_pull = None  # 입소문만 들었으므로 직접 이동 안함
    else:
        # AWARE = 풀 반응 + 핫스팟 이동
        hotspot = event.get("hotspot")
        if hotspot:
            location_pull = {
                "name": hotspot.get("name", ""),
                "dong": hotspot.get("dong", ""),
                "lat": hotspot.get("lat"),
                "lng": hotspot.get("lng"),
                "radius": hotspot.get("radius", 200),
            }

    return {
        "spending_boost": spending_boost,
        "explore_boost": explore_boost,
        "location_pull": location_pull,
        "pop_boost": pop_boost,
        "affected_industries": event.get("affected_industries", []),
    }


# ═══════════════════════════════════════════
# daily_memory 초기화 / 업데이트
# ═══════════════════════════════════════════

def init_daily_memory() -> dict:
    """에이전트별 daily_memory 초기화"""
    return {
        "recent_industries": [],        # [{day, industry, dong, satisfaction}] 최근 7일
        "consecutive_same": {},         # {industry: 연속일수}
        "exploration_log": [],          # [{day, industry, dong, result}]
        "peer_recommendations": [],     # [{from, industry, dong, day}]
        "mood": 0.6,
        "fatigue": 0.3,
    }


def update_mood_fatigue(memory: dict, day_name: str, rng) -> None:
    """기분/피로도 랜덤 워크 업데이트"""
    mood_delta = rng.normal(0, 0.1)
    if day_name == "friday":
        mood_delta += 0.15
    elif day_name == "monday":
        mood_delta -= 0.1

    memory["mood"] = float(np.clip(memory["mood"] + mood_delta, 0.1, 1.0))

    if day_name in ("saturday", "sunday"):
        memory["fatigue"] = max(0.05, memory["fatigue"] - 0.2)
    else:
        memory["fatigue"] = min(0.95, memory["fatigue"] + rng.uniform(0, 0.1))


def update_memory_after_action(memory: dict, day_idx: int, actions: list[dict]) -> None:
    """일간 행동 결과를 memory에 반영"""
    for act in actions:
        industry = act.get("industry")
        if not industry:
            continue

        # recent_industries 업데이트 (최대 7일 유지)
        memory["recent_industries"].insert(0, {
            "day": day_idx,
            "industry": industry,
            "dong": act.get("dong", ""),
            "satisfaction": act.get("satisfaction", 0.7),
        })
        if len(memory["recent_industries"]) > 7:
            memory["recent_industries"] = memory["recent_industries"][:7]

        # consecutive 업데이트
        prev = memory["consecutive_same"]
        for k in list(prev.keys()):
            if k != industry:
                prev[k] = 0
        prev[industry] = prev.get(industry, 0) + 1


# ═══════════════════════════════════════════
# 업종 선택 (daily_memory 반영)
# ═══════════════════════════════════════════

def choose_industry(
    activity_type: str,
    agent: dict,
    memory: dict,
    weekly_directive: dict,
    rng,
    closed_industries: set = None,
    new_industries: set = None,
    graph_context: dict = None,
) -> str:
    """daily_memory + LLM 지침 + 환경 변화 + GraphRAG 반영하여 업종 선택

    graph_context: GraphMemoryManager.get_rule_context() 결과
        - best_industries: 과거 만족도 높은 업종
        - avoid_industries: 과거 만족도 낮은 업종
        - peer_recommendations: 동료/이웃 추천
        - exploration_count: 새 업종 시도 횟수
        - total_experiences: 전체 경험 수
    """

    candidates = list(ACTIVITY_INDUSTRIES.get(activity_type, ["한식"]))

    # LLM preferred_industries가 있으면 앞에 추가
    preferred = weekly_directive.get("params", {}).get("preferred_industries", [])
    for ind in preferred:
        if ind in INDUSTRY_SPEND and ind not in candidates:
            candidates.insert(0, ind)

    # LLM avoid_industries 제거
    avoid = set(weekly_directive.get("params", {}).get("avoid_industries", []))
    candidates = [c for c in candidates if c not in avoid] or candidates

    # 가중치 계산
    weights = {}
    recent_set = {r["industry"] for r in memory.get("recent_industries", [])}

    # GraphRAG 데이터 준비
    graph_best = set()
    graph_avoid = set()
    graph_peer_recs = {}  # {industry: satisfaction}
    if graph_context:
        graph_best = set(graph_context.get("best_industries", []))
        graph_avoid = set(graph_context.get("avoid_industries", []))
        for rec in graph_context.get("peer_recommendations", []):
            ind = rec.get("industry")
            if ind:
                graph_peer_recs[ind] = rec.get("satisfaction", 0.7)

    for ind in candidates:
        w = 1.0

        # 어제 먹은 것 → 확률 감소
        recent = memory.get("recent_industries", [])
        if recent and recent[0].get("industry") == ind:
            w *= 0.5

        # 연속 같은 업종 회피
        consec = memory.get("consecutive_same", {}).get(ind, 0)
        if consec >= 3:
            w *= 0.05
        elif consec >= 2:
            w *= 0.3

        # 최근 안 먹은 업종 보너스
        if ind not in recent_set:
            w *= 1.3

        # 동료 추천 보너스 (daily_memory)
        for rec in memory.get("peer_recommendations", []):
            if rec.get("industry") == ind:
                w *= 1.5

        # 탐색 성공 재방문 보너스
        for exp in memory.get("exploration_log", []):
            if exp.get("industry") == ind and exp.get("result") == "good":
                w *= 1.4

        # LLM 선호 업종 보너스
        if ind in preferred:
            w *= 1.3

        # ── 라이프스타일 보너스 ──
        lifestyle = agent.get("lifestyle", "")
        ls_bonus = LIFESTYLE_INDUSTRY_BONUS.get(lifestyle, {})
        if ind in ls_bonus:
            w *= ls_bonus[ind]

        # ── GraphRAG 반영 ──
        if ind in graph_best:
            loyalty = agent.get("loyalty", 0.5)
            w *= (1.2 + 0.5 * loyalty)

        if ind in graph_avoid:
            w *= 0.15

        if ind in graph_peer_recs:
            peer_sat = graph_peer_recs[ind]
            trend = agent.get("trend_sensitivity", 0.5)
            w *= (1.3 + 0.6 * peer_sat * trend)

        # ── 환경 에이전트 반영 ──
        if closed_industries and ind in closed_industries:
            w *= 0.05
        if new_industries and ind in new_industries:
            trend = agent.get("trend_sensitivity", 0.5)
            w *= (1.0 + 0.5 * trend)

        weights[ind] = max(w, 0.01)

    # 가중 랜덤 선택
    inds = list(weights.keys())
    ws = np.array([weights[i] for i in inds], dtype=float)
    ws /= ws.sum()
    return inds[rng.choice(len(inds), p=ws)]


# ═══════════════════════════════════════════
# 일간 행동 생성
# ═══════════════════════════════════════════

def generate_daily_actions(
    agent: dict,
    memory: dict,
    weekly_directive: dict,
    day_name: str,
    day_idx: int,
    rng,
    env_context: dict = None,
    graph_context: dict = None,
) -> list[dict]:
    """하루의 소비 행동 리스트 생성

    Args:
        env_context: 환경 에이전트 컨텍스트 (population_factor, industry_changes 등)
        graph_context: GraphRAG 룰엔진 컨텍스트 (best/avoid/peer_recommendations)

    Returns: [{activity, industry, time_slot, amount, dong, type}, ...]
    """
    segment = agent["segment"]
    patterns = SEGMENT_PATTERNS.get(segment, SEGMENT_PATTERNS["commuter"])

    # 요일 유형 결정
    if day_name == "friday":
        day_type = "friday"
    elif day_name in ("saturday", "sunday"):
        day_type = "weekend"
    else:
        day_type = "weekday"

    day_patterns = patterns.get(day_type, patterns.get("weekday", {}))

    # LLM 액션 반영
    llm_action = weekly_directive.get("action", "유지")
    llm_params = weekly_directive.get("params", {})

    # 온라인전환: 소비 확률 전체 하락
    online_ratio = llm_params.get("online_ratio", 0.0)

    # 기분/피로 보정
    mood = memory.get("mood", 0.6)
    fatigue = memory.get("fatigue", 0.3)
    mood_factor = 0.7 + 0.6 * mood        # 0.7 ~ 1.3
    fatigue_factor = 1.2 - 0.4 * fatigue   # 0.8 ~ 1.2

    # ── 환경 에이전트 보정 ──
    pop_factor = 1.0
    closed_industries = set()
    new_industries = set()
    news_events = []        # 하위 호환: 문자열 리스트
    structured_events = []  # 신규: 구조화된 이벤트 딕셔너리
    policy_str = "없음"

    if env_context:
        pop_factor = env_context.get("population_factor", 1.0)
        news_events = env_context.get("events", [])
        structured_events = env_context.get("structured_events", [])
        policy_str = env_context.get("active_policy", "없음")
        for change in env_context.get("industry_changes", []):
            if change.get("closings", 0) > 0 and change.get("store_count", 1) <= 1:
                closed_industries.add(change["industry"])
            if change.get("openings", 0) > 0:
                new_industries.add(change["industry"])

    # ── 구조화된 이벤트 → 에이전트별 차등 반응 ──
    trend_sens = agent.get("trend_sensitivity", 0.5)
    news_explore_boost = 0.0
    news_spending_boost = 1.0
    active_triggers = []
    event_location_pull = None  # 이벤트 핫스팟으로 끌려갈 위치
    event_affected_industries = set()

    for evt in structured_events:
        if not isinstance(evt, dict):
            continue

        # 3단계 인지 판정
        awareness = _agent_awareness_level(agent, evt, rng)
        if awareness == AWARENESS_UNAWARE:
            continue

        # 인지 결과 저장 (소셜 전파용)
        headline = evt.get("headline", "")
        agent.setdefault("_news_awareness", {})[headline] = awareness

        mods = _get_event_modifiers(agent, evt, awareness)
        news_spending_boost *= mods["spending_boost"]
        news_explore_boost += mods["explore_boost"]
        pop_factor *= mods.get("pop_boost", 1.0)

        if mods["location_pull"] and event_location_pull is None:
            event_location_pull = mods["location_pull"]

        for ind in mods.get("affected_industries", []):
            if ind != "all":
                event_affected_industries.add(ind)

        cat = evt.get("category", "")
        headline = evt.get("headline", "")[:30]
        hotspot_name = ""
        if evt.get("hotspot"):
            hotspot_name = f"@{evt['hotspot'].get('name', '')}"
        level_tag = "[직접]" if awareness == AWARENESS_AWARE else "[입소문]"
        active_triggers.append(f"{level_tag}{cat}:{headline}{hotspot_name}")

    # 폴백: 구조화 이벤트 없으면 기존 문자열 뉴스 사용
    if not structured_events and news_events:
        news_explore_boost = 0.15 * trend_sens * len(news_events)
        for n in news_events:
            txt = n if isinstance(n, str) else str(n)
            active_triggers.append(f"뉴스:{txt[:20]}")

    # 정책 보정
    if policy_str != "없음":
        if "쿠폰" in policy_str or "할인" in policy_str:
            news_spending_boost *= (1.1 + 0.05 * trend_sens)
            active_triggers.append(f"정책:{policy_str[:20]}")
        elif "임대료" in policy_str:
            active_triggers.append(f"정책:{policy_str[:20]}")
    if new_industries:
        active_triggers.append(f"신규입점:{','.join(list(new_industries)[:2])}")
    if closed_industries:
        active_triggers.append(f"폐업:{','.join(list(closed_industries)[:2])}")

    # ── 추천 기반 위치 (peer_recommendations) ──
    recommended_dong = None
    recommended_hotspot = None
    for rec in memory.get("peer_recommendations", []):
        if rec.get("dong"):
            if rng.random() < agent.get("social_influence_weight", 0.3) * 0.6:
                recommended_dong = rec["dong"]
                if rec.get("hotspot"):
                    recommended_hotspot = rec["hotspot"]
                break

    actions = []

    for activity_name, spec in day_patterns.items():
        # 확률 판정 (mood/fatigue/online/population 보정)
        base_prob = spec["prob"]
        adjusted_prob = base_prob * mood_factor * fatigue_factor * (1 - online_ratio)
        adjusted_prob *= min(max(pop_factor, 0.8), 1.2)
        adjusted_prob += news_explore_boost
        adjusted_prob = min(adjusted_prob, 0.95)

        if rng.random() > adjusted_prob:
            continue

        # 업종 선택
        industry = choose_industry(
            activity_name, agent, memory, weekly_directive, rng,
            closed_industries=closed_industries,
            new_industries=new_industries,
            graph_context=graph_context,
        )

        # 이벤트 관련 업종이면 해당 업종으로 교체할 확률
        if event_affected_industries and rng.random() < trend_sens * 0.5:
            evt_inds = [i for i in event_affected_industries if i in INDUSTRY_SPEND]
            if evt_inds:
                industry = rng.choice(evt_inds)

        # 시간대
        t_start, t_end = spec.get("time", (12, 13))
        time_slot = int(rng.integers(t_start, t_end + 1))

        # 소비금액
        base_amount = INDUSTRY_SPEND.get(industry, 15000)
        budget_adj = llm_params.get("budget_adjustment", 1.0) * news_spending_boost
        day_factor = 1.3 if day_name == "friday" else 1.0
        amount = int(base_amount * budget_adj * day_factor * mood_factor * rng.normal(1.0, 0.15))
        amount = max(amount, 1000)

        # ── 위치 결정 (우선순위: 이벤트 핫스팟 > 추천 장소 > 기본 룰) ──
        dong_rule = spec.get("dong", "work")
        target_lat = None
        target_lng = None
        triggered_by = list(active_triggers)

        if event_location_pull and rng.random() < (trend_sens * 0.6 + agent.get("sns_activity", 0) * 0.3):
            # 이벤트 핫스팟으로 이동
            dong = event_location_pull["dong"]
            target_lat = event_location_pull.get("lat")
            target_lng = event_location_pull.get("lng")
            triggered_by.append(f"핫스팟이동:{event_location_pull.get('name', '')}")
        elif recommended_dong:
            # 추천받은 장소로 이동
            dong = recommended_dong
            if recommended_hotspot:
                target_lat = recommended_hotspot.get("lat")
                target_lng = recommended_hotspot.get("lng")
            triggered_by.append(f"추천이동:{dong}")
        elif llm_action == "전환" and llm_params.get("target_dong"):
            dong = llm_params["target_dong"]
        elif dong_rule == "work":
            dong = str(agent["adm_cd"])[:8]
        elif dong_rule == "home":
            dong = agent.get("home_adm_cd", str(agent["adm_cd"])[:8])
        else:
            if rng.random() < 0.5:
                dong = str(agent["adm_cd"])[:8]
            else:
                dong = agent.get("home_adm_cd", str(agent["adm_cd"])[:8])

        # ── 이벤트 기반 탐색: 트렌드 민감도 높으면 새 업종/장소 시도 ──
        if structured_events and rng.random() < trend_sens * 0.35:
            all_industries = list(INDUSTRY_SPEND.keys())
            current_top = set(agent.get("top_industries", []))
            new_options = [i for i in all_industries if i not in current_top]
            if new_options:
                industry = rng.choice(new_options)
                triggered_by.append(f"이벤트탐색:{industry}")
                memory.setdefault("exploration_log", []).append({
                    "day": day_idx,
                    "industry": industry,
                    "dong": dong,
                    "result": "good" if rng.random() < 0.55 else "bad",
                    "trigger": "event",
                })

        # 신규탐색 액션 처리 (LLM 지시)
        if llm_action == "신규탐색" and activity_name in ("lunch", "dinner", "meal"):
            explore_ind = llm_params.get("explore_industry")
            if explore_ind and explore_ind in INDUSTRY_SPEND:
                industry = explore_ind
                result = "good" if rng.random() < 0.6 else "bad"
                memory.setdefault("exploration_log", []).append({
                    "day": day_idx,
                    "industry": industry,
                    "dong": dong,
                    "result": result,
                    "trigger": "llm",
                })

        # 만족도
        base_sat = mood * 0.4 + rng.random() * 0.4
        if industry in new_industries:
            base_sat += 0.15
        if industry in event_affected_industries:
            base_sat += 0.1  # 이벤트 관련 업종 방문 → 만족도 보너스
        if closed_industries:
            base_sat -= 0.05
        satisfaction = float(np.clip(base_sat, 0.1, 1.0))

        action = {
            "activity": activity_name,
            "industry": industry,
            "time_slot": time_slot,
            "amount": amount,
            "dong": dong,
            "satisfaction": round(satisfaction, 2),
            "type": "외출_소비",
            "triggered_by": triggered_by,
        }
        # 핫스팟 좌표가 있으면 포함 (move_agent에서 활용)
        if target_lat is not None:
            action["target_lat"] = target_lat
            action["target_lng"] = target_lng

        actions.append(action)

    # 행동 없으면 재택
    if not actions:
        actions.append({
            "activity": "stay_home",
            "industry": None,
            "time_slot": 12,
            "amount": 0,
            "dong": agent.get("home_adm_cd", str(agent["adm_cd"])[:8]),
            "satisfaction": 0.5,
            "type": "재택",
            "triggered_by": [],
        })

    return actions


# ═══════════════════════════════════════════
# 좌표 이동
# ═══════════════════════════════════════════

def move_agent(agent: dict, actions: list[dict], dong_grids: dict, rng) -> None:
    """일간 행동 결과에 따라 에이전트 좌표 업데이트.

    에이전트의 세그먼트(페르소나)에 따라 자율적으로 이동:
    - commuter: 평일엔 직장 근처에 오래 머무름, 주말엔 집 근처
    - resident: 대부분 집 근처에서 활동
    - evening_visitor: 저녁/밤에 소비 장소에 오래 머무름
    - weekend_visitor: 주말에 먼 곳까지 탐색
    """
    home_lat = agent.get("home_lat", agent["current_lat"])
    home_lng = agent.get("home_lng", agent["current_lng"])
    segment = agent.get("segment", "commuter")

    # 마지막 외출 소비 행동 찾기
    last_outing = None
    for act in reversed(actions):
        if act["type"] == "외출_소비":
            last_outing = act
            break

    if not last_outing:
        # 재택: 집 근처에서 약간 이동
        agent["current_lat"] = home_lat + rng.normal(0, 0.001)
        agent["current_lng"] = home_lng + rng.normal(0, 0.001)
        return

    # ── 핫스팟 좌표가 있으면 우선 사용 ──
    if last_outing.get("target_lat") is not None:
        act_lat = last_outing["target_lat"] + rng.normal(0, 0.001)
        act_lng = last_outing["target_lng"] + rng.normal(0, 0.001)
    else:
        dong = last_outing["dong"]
        grids = dong_grids.get(dong, [])

        if not grids:
            agent["current_lat"] += rng.normal(0, 0.001)
            agent["current_lng"] += rng.normal(0, 0.001)
            return

        chosen = grids[rng.integers(0, len(grids))]
        act_lat = chosen["lat"] + rng.normal(0, 0.0005)
        act_lng = chosen["lng"] + rng.normal(0, 0.0005)

    time_slot = last_outing.get("time_slot", 14)

    # ── 세그먼트별 자율 체류 패턴 ──
    # stay_at_activity: 소비 장소에 머무르는 비중 (0=집, 1=소비장소)
    if segment == "commuter":
        # 직장인: 퇴근 전까지 직장 근처, 자연스럽게 귀가
        if time_slot <= 17:
            stay = 0.7 + rng.normal(0, 0.05)    # 근무 중: 직장 근처
        else:
            stay = 0.3 + rng.normal(0, 0.1)     # 퇴근 후: 점차 집 방향
    elif segment == "resident":
        # 거주민: 대부분 집 근처, 외출해도 멀리 안 감
        stay = 0.2 + rng.normal(0, 0.08)
    elif segment == "evening_visitor":
        # 저녁방문: 밤늦게까지 소비 장소에 체류
        if time_slot >= 20:
            stay = 0.8 + rng.normal(0, 0.05)    # 밤: 아직 밖
        elif time_slot >= 17:
            stay = 0.6 + rng.normal(0, 0.1)     # 저녁: 밖에서 활동 중
        else:
            stay = 0.3 + rng.normal(0, 0.1)
    elif segment == "weekend_visitor":
        # 주말방문: 주말엔 먼 곳까지, 평일엔 집 근처
        day_of_week = last_outing.get("time_slot", 14)  # fallback
        stay = 0.6 + rng.normal(0, 0.1)        # 방문 목적이니까 소비 장소 근처
    else:
        stay = 0.4 + rng.normal(0, 0.1)

    stay = float(np.clip(stay, 0.05, 0.95))

    # 현재 위치 → 집과 소비 장소 사이에서 자율 배치
    agent["current_lat"] = home_lat * (1 - stay) + act_lat * stay + rng.normal(0, 0.0003)
    agent["current_lng"] = home_lng * (1 - stay) + act_lng * stay + rng.normal(0, 0.0003)


# ═══════════════════════════════════════════
# 추천 전파
# ═══════════════════════════════════════════

def propagate_recommendations(agents: list[dict], memories: dict, decisions: dict, rng) -> None:
    """'추천' 액션을 수행한 에이전트의 추천을 주변 에이전트에 전파"""

    recommenders = []
    for agent in agents:
        decision = decisions.get(agent["agent_id"], {})
        if decision.get("action") == "추천":
            recommenders.append(agent)

    if not recommenders:
        return

    for rec_agent in recommenders:
        rec_params = decisions[rec_agent["agent_id"]].get("params", {})
        rec_industry = rec_params.get("recommend_industry")
        rec_dong = rec_params.get("recommend_dong")
        if not rec_industry:
            continue

        # social_influence_weight가 높은 주변 에이전트에게 전파
        for target in agents:
            if target["agent_id"] == rec_agent["agent_id"]:
                continue
            if rng.random() < target.get("social_influence_weight", 0.3) * 0.3:
                mem = memories.get(target["agent_id"])
                if mem:
                    rec_entry = {
                        "from": rec_agent["agent_id"],
                        "industry": rec_industry,
                        "dong": rec_dong or str(rec_agent["adm_cd"])[:8],
                        "day": 0,
                    }
                    # 핫스팟 좌표 포함 (있으면)
                    if rec_params.get("hotspot"):
                        rec_entry["hotspot"] = rec_params["hotspot"]
                    mem.setdefault("peer_recommendations", []).append(rec_entry)
                    if len(mem["peer_recommendations"]) > 5:
                        mem["peer_recommendations"] = mem["peer_recommendations"][-5:]


def propagate_news_awareness(agents: list[dict], structured_events: list, rng) -> None:
    """AWARE 에이전트가 소셜 네트워크 이웃에게 뉴스를 전파.

    매일 호출: 직접 인지(AWARE) 에이전트 → 이웃에게 HEARD로 전파.
    _news_awareness: {headline: awareness_level} 에이전트 딕셔너리에 저장.
    """
    if not structured_events:
        return

    for evt in structured_events:
        if not isinstance(evt, dict):
            continue
        headline = evt.get("headline", "")
        if not headline:
            continue

        # AWARE 에이전트 찾기
        aware_agents = []
        for a in agents:
            awareness_map = a.get("_news_awareness", {})
            if awareness_map.get(headline, 0) == AWARENESS_AWARE:
                aware_agents.append(a)

        if not aware_agents:
            continue

        # AWARE → 이웃에게 전파 (확률적)
        for aware_a in aware_agents:
            for target in agents:
                if target["agent_id"] == aware_a["agent_id"]:
                    continue
                # 이미 인지하고 있으면 스킵
                target_map = target.setdefault("_news_awareness", {})
                if target_map.get(headline, 0) >= AWARENESS_HEARD:
                    continue
                # 전파 확률: 소셜 영향력 × 0.15 (하루에 소수만 전파)
                propagate_prob = target.get("social_influence_weight", 0.3) * 0.15
                if rng.random() < propagate_prob:
                    target_map[headline] = AWARENESS_HEARD

