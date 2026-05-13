"""
Night 단계 — 상호작용 대상 선정 알고리즘

매일 시뮬레이션이 끝난 뒤 (Night 단계), 에이전트들의 페르소나 + daily log를
입력으로 받아 누가 누구와 대화할 것인지 상호작용 쌍(pair)을 수식 기반으로 결정합니다.

LLM 호출 없이 3가지 축의 점수를 가중 합산하여 높은 순으로 상호작용 대상을 매칭합니다.

InteractionScore(A, B) = w₁ · Exposure + w₂ · Relationship + w₃ · Urgency

축 1: Exposure (물리적/디지털 접점) — 같은 동 + 시간 겹침
축 2: Relationship (관계 깊이) — SocialNetwork 엣지 + 과거 대화 이력
축 3: Urgency (상호작용 필요성) — 정보 비대칭, 감정 상태
"""
from collections import defaultdict


# ═══════════════════════════════════════════
# 가중치 기본값 (config로 분리 가능)
# ═══════════════════════════════════════════

W_EXPOSURE = 0.4
W_RELATION = 0.3
W_URGENCY = 0.3


# ═══════════════════════════════════════════
# 축 1: Exposure (물리적/디지털 접점) — 0.0 ~ 1.0
# ═══════════════════════════════════════════

def calc_exposure(log_a, log_b, agent_a, agent_b):
    """오늘 하루 동안 A와 B가 얼마나 접점이 있었는지 계산.

    Args:
        log_a: 에이전트 A의 오늘 actions 리스트 [{dong, time_slot, activity, ...}, ...]
        log_b: 에이전트 B의 오늘 actions 리스트
        agent_a: 에이전트 A 프로필 dict
        agent_b: 에이전트 B 프로필 dict

    Returns:
        float: 0.0 ~ 1.0 사이의 접점 점수
    """
    co_visits = []  # 같은 동 + 겹치는 시간대 리스트

    for act_a in log_a:
        dong_a = act_a.get("dong")
        time_a = act_a.get("time_slot")
        if dong_a is None or time_a is None:
            continue

        for act_b in log_b:
            dong_b = act_b.get("dong")
            time_b = act_b.get("time_slot")
            if dong_b is None or time_b is None:
                continue

            if dong_a == dong_b:
                # 시간대 겹침 계산 (±1시간 이내면 접점)
                time_diff = abs(time_a - time_b)
                if time_diff <= 1:
                    co_visits.append({
                        "dong": dong_a,
                        "overlap": 1.0 - time_diff * 0.5,  # 동시간=1.0, 1시간차=0.5
                    })

    if not co_visits:
        return 0.0

    # 빈도(frequency) × 체류 겹침(duration proxy)
    frequency = min(len(co_visits), 5) / 5.0              # 최대 5회 → 1.0
    avg_overlap = sum(v["overlap"] for v in co_visits) / len(co_visits)

    return min(frequency * 0.6 + avg_overlap * 0.4, 1.0)


# ═══════════════════════════════════════════
# 축 2: Relationship (에이전트 간 관계도) — 0.0 ~ 1.0
# ═══════════════════════════════════════════

def calc_relationship(agent_a_id, agent_b_id, social_network, interaction_history):
    """A와 B 사이의 관계 깊이.

    Args:
        agent_a_id: 에이전트 A ID
        agent_b_id: 에이전트 B ID
        social_network: SocialNetwork 인스턴스 (graph_memory.py)
        interaction_history: dict — {(a_id, b_id): 누적 대화 횟수}

    Returns:
        float: 0.0 ~ 1.0 사이의 관계 깊이 점수
    """
    # 1) 기존 소셜 네트워크 관계
    base_relation = 0.0
    if social_network.G.has_edge(agent_a_id, agent_b_id):
        edge = social_network.G[agent_a_id][agent_b_id]
        rel_type = edge.get("type", "")
        if rel_type == "COLLEAGUE":
            base_relation = 0.6
        elif rel_type == "NEIGHBOR":
            base_relation = 0.4
        else:
            base_relation = 0.3

    # 2) 과거 대화 이력에 의한 친밀도 (intimacy)
    # interaction_history는 정렬된 키를 사용 (a,b) 또는 (b,a) 모두 체크
    pair = tuple(sorted([agent_a_id, agent_b_id]))
    past_count = interaction_history.get(pair, 0)
    intimacy = min(past_count / 10.0, 1.0)  # 10회 대화 → 친밀도 최대

    # 가중 합산
    return min(base_relation * 0.5 + intimacy * 0.5, 1.0)


# ═══════════════════════════════════════════
# 축 3: Urgency (상호작용 필요성) — 0.0 ~ 1.0
# ═══════════════════════════════════════════

def calc_urgency(agent_a, agent_b, memory_a, memory_b, policy_info_holders):
    """A↔B 간 상호작용 필요성 (비대칭 → 높은 쪽 사용).

    Args:
        agent_a: 에이전트 A 프로필 dict (agent_id, _news_awareness 등)
        agent_b: 에이전트 B 프로필 dict
        memory_a: 에이전트 A의 daily_memory (mood, fatigue 등)
        memory_b: 에이전트 B의 daily_memory
        policy_info_holders: set — 정책 정보를 알고 있는 에이전트 ID 집합

    Returns:
        float: 0.0 ~ 1.0 사이의 상호작용 필요성 점수
    """
    urgency_a = 0.0  # A가 B에게 말할 필요성
    urgency_b = 0.0  # B가 A에게 말할 필요성

    # ── 1) 정보 희소성 (Information Scarcity) ──
    # A가 정책 정보를 알고 있고 B는 모르면 → A가 B에게 전파 욕구
    a_knows_policy = agent_a["agent_id"] in policy_info_holders
    b_knows_policy = agent_b["agent_id"] in policy_info_holders

    if a_knows_policy and not b_knows_policy:
        urgency_a += 0.5
    if b_knows_policy and not a_knows_policy:
        urgency_b += 0.5

    # 뉴스 인지 차이 (AWARE인 이벤트를 상대가 모를 때)
    awareness_a = agent_a.get("_news_awareness", {})
    awareness_b = agent_b.get("_news_awareness", {})
    info_gap = 0
    for headline, level in awareness_a.items():
        if level >= 2 and awareness_b.get(headline, 0) == 0:
            info_gap += 1
    for headline, level in awareness_b.items():
        if level >= 2 and awareness_a.get(headline, 0) == 0:
            info_gap += 1
    urgency_a += min(info_gap * 0.15, 0.4)

    # ── 2) 감정 임계치 (Emotional Threshold) ──
    # mood가 극단적이면(매우 좋거나 매우 나쁘면) 표출 욕구 ↑
    mood_a = memory_a.get("mood", 0.5)
    mood_b = memory_b.get("mood", 0.5)

    # mood가 0.3 이하(우울) 또는 0.7 이상(흥분)이면 표출 욕구
    emotional_a = max(0, 0.8 - mood_a) if mood_a < 0.3 else max(0, mood_a - 0.7) * 1.5
    emotional_b = max(0, 0.8 - mood_b) if mood_b < 0.3 else max(0, mood_b - 0.7) * 1.5

    urgency_a += min(emotional_a, 0.4)
    urgency_b += min(emotional_b, 0.4)

    # 피로도가 높으면 대화 욕구 감소 (보정)
    fatigue_a = memory_a.get("fatigue", 0.3)
    fatigue_b = memory_b.get("fatigue", 0.3)
    urgency_a *= (1.0 - fatigue_a * 0.3)
    urgency_b *= (1.0 - fatigue_b * 0.3)

    # 양방향 중 높은 값 사용 (한 쪽이라도 말하고 싶으면 대화 발생)
    return min(max(urgency_a, urgency_b), 1.0)


# ═══════════════════════════════════════════
# 후보 쌍 필터링 (O(N²) 회피)
# ═══════════════════════════════════════════

def find_candidate_pairs(agents, daily_logs):
    """같은 동에 방문한 에이전트들만 후보 쌍으로 추출 + 소셜 이웃.

    Args:
        agents: 에이전트 프로필 리스트
        daily_logs: {agent_id: [action, ...]} — 오늘의 행동 로그
        social_network: SocialNetwork 인스턴스

    Returns:
        set of tuple: {(a_id, b_id), ...} 후보 쌍 (정렬된 키)
    """
    # 동별 방문 에이전트 인덱스
    dong_visitors = defaultdict(set)
    for agent in agents:
        aid = agent["agent_id"]
        for action in daily_logs.get(aid, []):
            dong = action.get("dong")
            if dong:
                dong_visitors[dong].add(aid)

    candidates = set()
    for dong, visitors in dong_visitors.items():
        visitor_list = list(visitors)
        for i in range(len(visitor_list)):
            for j in range(i + 1, len(visitor_list)):
                candidates.add(tuple(sorted([visitor_list[i], visitor_list[j]])))

    return candidates


# ═══════════════════════════════════════════
# 매칭 알고리즘
# ═══════════════════════════════════════════

def select_interaction_pairs(
    agents, daily_logs, memories, social_network,
    interaction_history, policy_info_holders,
    max_pairs_per_agent=2, threshold=0.3,
    weights=None,
):
    """Night 단계: 모든 에이전트 쌍의 점수 계산 → 높은 순으로 매칭.

    Args:
        agents: 에이전트 프로필 리스트
        daily_logs: {agent_id: [action, ...]} — 오늘의 행동 로그
        memories: {agent_id: daily_memory}
        social_network: SocialNetwork 인스턴스
        interaction_history: {(a_id, b_id): count} — 누적 대화 횟수
        policy_info_holders: set — 정책 정보 보유 에이전트 ID
        max_pairs_per_agent: 에이전트당 최대 상호작용 수
        threshold: 최소 점수 임계치
        weights: (w1, w2, w3) 가중치 튜플 (None이면 기본값 사용)

    Returns:
        list of tuples: [(agent_a_id, agent_b_id, score, breakdown), ...]
    """
    w1 = weights[0] if weights else W_EXPOSURE
    w2 = weights[1] if weights else W_RELATION
    w3 = weights[2] if weights else W_URGENCY

    agents_map = {a["agent_id"]: a for a in agents}

    # 1) 후보 쌍 필터링
    candidate_pairs = find_candidate_pairs(agents, daily_logs)

    # 2) 각 쌍별 점수 계산
    scored_pairs = []
    for a_id, b_id in candidate_pairs:
        agent_a = agents_map.get(a_id)
        agent_b = agents_map.get(b_id)
        if agent_a is None or agent_b is None:
            continue

        log_a = daily_logs.get(a_id, [])
        log_b = daily_logs.get(b_id, [])
        mem_a = memories.get(a_id, {})
        mem_b = memories.get(b_id, {})

        exposure = calc_exposure(log_a, log_b, agent_a, agent_b)
        relationship = calc_relationship(a_id, b_id, social_network, interaction_history)
        urgency = calc_urgency(agent_a, agent_b, mem_a, mem_b, policy_info_holders)

        total = w1 * exposure + w2 * relationship + w3 * urgency

        if total >= threshold:
            scored_pairs.append((a_id, b_id, total, {
                "exposure": round(exposure, 4),
                "relationship": round(relationship, 4),
                "urgency": round(urgency, 4),
            }))

    # 3) 점수 순 정렬 → 그리디 매칭
    scored_pairs.sort(key=lambda x: x[2], reverse=True)

    # 4) 에이전트당 최대 상호작용 수 제한 (그리디)
    interaction_count = defaultdict(int)
    selected = []

    for a_id, b_id, score, breakdown in scored_pairs:
        if (interaction_count[a_id] < max_pairs_per_agent and
            interaction_count[b_id] < max_pairs_per_agent):
            selected.append((a_id, b_id, score, breakdown))
            interaction_count[a_id] += 1
            interaction_count[b_id] += 1

    return selected


# ═══════════════════════════════════════════
# 대화 이력 누적
# ═══════════════════════════════════════════

def update_interaction_history(selected_pairs, interaction_history):
    """선정된 상호작용 쌍의 대화 이력을 누적.

    Args:
        selected_pairs: select_interaction_pairs 반환값
        interaction_history: {(a_id, b_id): count} — 정렬된 키 사용

    Returns:
        int: 이번에 추가된 상호작용 수
    """
    count = 0
    for a_id, b_id, score, breakdown in selected_pairs:
        pair = tuple(sorted([a_id, b_id]))
        interaction_history[pair] = interaction_history.get(pair, 0) + 1
        count += 1
    return count


# ═══════════════════════════════════════════
# 정책 정보 보유자 추출 (POLICY_ANNOUNCE 카테고리 활용)
# ═══════════════════════════════════════════

def extract_policy_info_holders(agents):
    """_news_awareness에서 POLICY_ANNOUNCE를 AWARE(2)로 인지한 에이전트 추출.

    Open Question #4 해결: _news_awareness를 활용하여 정책 정보 보유자를 판별.
    'POLICY_ANNOUNCE' 또는 'SUBSIDY' 카테고리의 뉴스를 AWARE(2) 수준으로 인지한
    에이전트를 정책 정보 보유자로 간주합니다.

    Args:
        agents: 에이전트 프로필 리스트

    Returns:
        set: 정책 정보를 알고 있는 에이전트 ID 집합
    """
    holders = set()
    # 정책 관련 키워드
    policy_keywords = ["정책", "쿠폰", "할인", "지원", "보전", "바우처", "상품권",
                        "지정", "특구", "오픈", "조성", "보행자"]

    for agent in agents:
        awareness = agent.get("_news_awareness", {})
        for headline, level in awareness.items():
            if level >= 2:  # AWARE
                # 정책 관련 키워드가 포함된 뉴스
                if any(kw in headline for kw in policy_keywords):
                    holders.add(agent["agent_id"])
                    break
    return holders
