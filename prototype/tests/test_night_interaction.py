"""
Night Interaction 모듈 단위 + 통합 테스트

테스트 항목:
1. calc_exposure: 접점 없음 → ≈0, 같은 동/같은 시간 → 높음
2. calc_relationship: 관계 없음 → 0, COLLEAGUE → 높음, 대화 이력 → 친밀도 상승
3. calc_urgency: 정책 비대칭 → 높음, 감정 극단 → 높음
4. select_interaction_pairs: mock 10명 → 결과 쌍 점수 ≥ threshold
"""
import sys
import os

# src 디렉토리를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import networkx as nx
from collections import defaultdict

from night_interaction import (
    calc_exposure,
    calc_relationship,
    calc_urgency,
    find_candidate_pairs,
    select_interaction_pairs,
    update_interaction_history,
    extract_policy_info_holders,
)


# ═══════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════

class MockSocialNetwork:
    """SocialNetwork의 최소 모킹."""
    def __init__(self):
        self.G = nx.Graph()

    def add_edge(self, a, b, **kwargs):
        self.G.add_edge(a, b, **kwargs)

    def add_node(self, n, **kwargs):
        self.G.add_node(n, **kwargs)


@pytest.fixture
def social_net():
    sn = MockSocialNetwork()
    # 3명의 에이전트: c001, c002, c003
    for i in range(1, 4):
        sn.add_node(f"consumer_{i:04d}")
    # c001 - c002: COLLEAGUE
    sn.add_edge("consumer_0001", "consumer_0002", type="COLLEAGUE", weight=0.6)
    # c002 - c003: NEIGHBOR
    sn.add_edge("consumer_0002", "consumer_0003", type="NEIGHBOR", weight=0.4)
    return sn


@pytest.fixture
def agents_3():
    return [
        {"agent_id": "consumer_0001", "segment": "commuter",
         "_news_awareness": {}, "adm_cd": "1114055000"},
        {"agent_id": "consumer_0002", "segment": "commuter",
         "_news_awareness": {}, "adm_cd": "1114055000"},
        {"agent_id": "consumer_0003", "segment": "resident",
         "_news_awareness": {}, "adm_cd": "1117010000"},
    ]


@pytest.fixture
def memories_3():
    return {
        "consumer_0001": {"mood": 0.6, "fatigue": 0.3},
        "consumer_0002": {"mood": 0.2, "fatigue": 0.1},  # 우울
        "consumer_0003": {"mood": 0.9, "fatigue": 0.5},  # 흥분
    }


# ═══════════════════════════════════════════
# calc_exposure 테스트
# ═══════════════════════════════════════════

class TestCalcExposure:
    def test_no_overlap(self):
        """접점 없으면 → 0"""
        log_a = [{"dong": "11140", "time_slot": 12}]
        log_b = [{"dong": "11170", "time_slot": 18}]
        agent_a = {}
        agent_b = {}
        score = calc_exposure(log_a, log_b, agent_a, agent_b)
        assert score == 0.0

    def test_same_dong_same_time(self):
        """같은 동 + 같은 시간 → 높은 점수"""
        log_a = [{"dong": "11140", "time_slot": 12}]
        log_b = [{"dong": "11140", "time_slot": 12}]
        agent_a = {}
        agent_b = {}
        score = calc_exposure(log_a, log_b, agent_a, agent_b)
        # frequency=1/5=0.2, overlap=1.0 → 0.2*0.6 + 1.0*0.4 = 0.52
        assert score > 0.4

    def test_same_dong_one_hour_diff(self):
        """같은 동 + 1시간 차이 → 중간 점수"""
        log_a = [{"dong": "11140", "time_slot": 12}]
        log_b = [{"dong": "11140", "time_slot": 13}]
        agent_a = {}
        agent_b = {}
        score = calc_exposure(log_a, log_b, agent_a, agent_b)
        assert score > 0.0

    def test_multiple_covisits(self):
        """여러 번 같은 동 방문 → 빈도 증가 → 점수 증가"""
        log_a = [
            {"dong": "11140", "time_slot": 10},
            {"dong": "11140", "time_slot": 12},
            {"dong": "11140", "time_slot": 14},
        ]
        log_b = [
            {"dong": "11140", "time_slot": 10},
            {"dong": "11140", "time_slot": 12},
            {"dong": "11140", "time_slot": 14},
        ]
        agent_a = {}
        agent_b = {}
        score = calc_exposure(log_a, log_b, agent_a, agent_b)
        # 최대 9개의 co_visits 발생 가능 (3x3), 대부분 ±1이내
        assert score > 0.6

    def test_empty_logs(self):
        """빈 로그 → 0"""
        score = calc_exposure([], [], {}, {})
        assert score == 0.0


# ═══════════════════════════════════════════
# calc_relationship 테스트
# ═══════════════════════════════════════════

class TestCalcRelationship:
    def test_no_relation(self, social_net):
        """관계 없는 쌍 → 0"""
        # c001과 c003은 직접 연결 없음
        score = calc_relationship("consumer_0001", "consumer_0003", social_net, {})
        assert score == 0.0

    def test_colleague(self, social_net):
        """COLLEAGUE 관계 → base_relation 0.6"""
        score = calc_relationship("consumer_0001", "consumer_0002", social_net, {})
        # base_relation=0.6, intimacy=0 → 0.6*0.5 + 0*0.5 = 0.3
        assert abs(score - 0.3) < 1e-6

    def test_neighbor(self, social_net):
        """NEIGHBOR 관계 → base_relation 0.4"""
        score = calc_relationship("consumer_0002", "consumer_0003", social_net, {})
        # base_relation=0.4, intimacy=0 → 0.4*0.5 + 0*0.5 = 0.2
        assert abs(score - 0.2) < 1e-6

    def test_with_history(self, social_net):
        """대화 이력 있으면 친밀도 상승"""
        history = {("consumer_0001", "consumer_0002"): 5}
        score = calc_relationship("consumer_0001", "consumer_0002", social_net, history)
        # base_relation=0.6, intimacy=5/10=0.5 → 0.6*0.5 + 0.5*0.5 = 0.55
        assert abs(score - 0.55) < 1e-6

    def test_max_intimacy(self, social_net):
        """대화 이력 10회 이상 → 친밀도 최대"""
        history = {("consumer_0001", "consumer_0002"): 15}
        score = calc_relationship("consumer_0001", "consumer_0002", social_net, history)
        # base_relation=0.6, intimacy=1.0 → 0.6*0.5 + 1.0*0.5 = 0.8
        assert abs(score - 0.8) < 1e-6

    def test_sorted_key_lookup(self, social_net):
        """키가 역순이어도 정렬 후 조회되는지 확인"""
        history = {("consumer_0001", "consumer_0002"): 5}
        # b, a 순으로 호출해도 정렬된 키로 검색
        score = calc_relationship("consumer_0002", "consumer_0001", social_net, history)
        assert abs(score - 0.55) < 1e-6


# ═══════════════════════════════════════════
# calc_urgency 테스트
# ═══════════════════════════════════════════

class TestCalcUrgency:
    def test_no_urgency(self):
        """특별한 상태 없음 → 낮은 urgency"""
        agent_a = {"agent_id": "a", "_news_awareness": {}}
        agent_b = {"agent_id": "b", "_news_awareness": {}}
        mem_a = {"mood": 0.5, "fatigue": 0.3}
        mem_b = {"mood": 0.5, "fatigue": 0.3}
        score = calc_urgency(agent_a, agent_b, mem_a, mem_b, set())
        assert score < 0.2

    def test_policy_asymmetry(self):
        """정책 정보 비대칭 → urgency 상승"""
        agent_a = {"agent_id": "a", "_news_awareness": {}}
        agent_b = {"agent_id": "b", "_news_awareness": {}}
        mem_a = {"mood": 0.5, "fatigue": 0.3}
        mem_b = {"mood": 0.5, "fatigue": 0.3}
        # A만 정책 정보 보유
        score = calc_urgency(agent_a, agent_b, mem_a, mem_b, {"a"})
        assert score >= 0.3  # +0.5 * (1 - 0.3*0.3) ≈ 0.455

    def test_depressed_mood(self):
        """우울한 기분 → urgency 상승"""
        agent_a = {"agent_id": "a", "_news_awareness": {}}
        agent_b = {"agent_id": "b", "_news_awareness": {}}
        mem_a = {"mood": 0.15, "fatigue": 0.1}  # 매우 우울
        mem_b = {"mood": 0.5, "fatigue": 0.3}
        score = calc_urgency(agent_a, agent_b, mem_a, mem_b, set())
        assert score > 0.3  # 우울할 때 대화 욕구

    def test_excited_mood(self):
        """흥분 상태 → urgency 상승"""
        agent_a = {"agent_id": "a", "_news_awareness": {}}
        agent_b = {"agent_id": "b", "_news_awareness": {}}
        mem_a = {"mood": 0.95, "fatigue": 0.1}  # 매우 흥분
        mem_b = {"mood": 0.5, "fatigue": 0.3}
        score = calc_urgency(agent_a, agent_b, mem_a, mem_b, set())
        assert score > 0.2  # 기분 좋으면 자랑하고 싶음

    def test_news_awareness_gap(self):
        """뉴스 인지 격차 → urgency 상승"""
        agent_a = {"agent_id": "a", "_news_awareness": {"뉴스1": 2, "뉴스2": 2}}
        agent_b = {"agent_id": "b", "_news_awareness": {}}
        mem_a = {"mood": 0.5, "fatigue": 0.1}
        mem_b = {"mood": 0.5, "fatigue": 0.1}
        score = calc_urgency(agent_a, agent_b, mem_a, mem_b, set())
        assert score > 0.15  # 정보 격차 2건 → +0.30

    def test_high_fatigue_reduces_urgency(self):
        """피로도 높으면 urgency 감소"""
        agent_a = {"agent_id": "a", "_news_awareness": {}}
        agent_b = {"agent_id": "b", "_news_awareness": {}}
        mem_low_fat = {"mood": 0.15, "fatigue": 0.1}
        mem_high_fat = {"mood": 0.15, "fatigue": 0.9}

        score_low = calc_urgency(agent_a, agent_b, mem_low_fat, {"mood": 0.5, "fatigue": 0.3}, set())
        score_high = calc_urgency(agent_a, agent_b, mem_high_fat, {"mood": 0.5, "fatigue": 0.3}, set())
        assert score_low > score_high  # 피로도 높으면 대화 욕구 감소


# ═══════════════════════════════════════════
# select_interaction_pairs 통합 테스트
# ═══════════════════════════════════════════

class TestSelectInteractionPairs:
    @pytest.fixture
    def agents_10(self):
        """10명의 mock 에이전트."""
        agents = []
        for i in range(10):
            agents.append({
                "agent_id": f"consumer_{i:04d}",
                "segment": "commuter" if i < 7 else "resident",
                "adm_cd": "1114055000" if i < 5 else "1117010000",
                "home_adm_cd": "1117010000" if i < 5 else "1114055000",
                "_news_awareness": {},
            })
        # 일부에게 정책 뉴스 인지 부여
        agents[0]["_news_awareness"] = {"소상공인 쿠폰 배포": 2}
        agents[3]["_news_awareness"] = {"야간 경제 특구 지정": 2}
        return agents

    @pytest.fixture
    def social_10(self, agents_10):
        sn = MockSocialNetwork()
        for a in agents_10:
            sn.add_node(a["agent_id"])
        # 같은 adm_cd끼리 COLLEAGUE
        for i in range(5):
            for j in range(i + 1, 5):
                sn.add_edge(agents_10[i]["agent_id"], agents_10[j]["agent_id"],
                           type="COLLEAGUE", weight=0.6)
        for i in range(5, 10):
            for j in range(i + 1, 10):
                sn.add_edge(agents_10[i]["agent_id"], agents_10[j]["agent_id"],
                           type="NEIGHBOR", weight=0.4)
        return sn

    @pytest.fixture
    def daily_logs_10(self, agents_10):
        """같은 동에서 활동하는 로그."""
        logs = {}
        for i, a in enumerate(agents_10):
            dong = str(a["adm_cd"])[:8]
            logs[a["agent_id"]] = [
                {"dong": dong, "time_slot": 12, "activity": "lunch", "type": "외출_소비"},
                {"dong": dong, "time_slot": 14, "activity": "cafe", "type": "외출_소비"},
            ]
        return logs

    @pytest.fixture
    def memories_10(self, agents_10):
        import random
        random.seed(42)
        return {
            a["agent_id"]: {
                "mood": 0.3 + random.random() * 0.5,
                "fatigue": 0.1 + random.random() * 0.4,
            }
            for a in agents_10
        }

    def test_basic_matching(self, agents_10, social_10, daily_logs_10, memories_10):
        """기본 매칭: 결과 쌍이 생성되는지 확인"""
        pairs = select_interaction_pairs(
            agents_10, daily_logs_10, memories_10, social_10,
            {}, set(), max_pairs_per_agent=2, threshold=0.1,
        )
        assert len(pairs) > 0
        # 모든 점수가 threshold 이상
        for a, b, score, breakdown in pairs:
            assert score >= 0.1
            assert "exposure" in breakdown
            assert "relationship" in breakdown
            assert "urgency" in breakdown

    def test_max_pairs_limit(self, agents_10, social_10, daily_logs_10, memories_10):
        """에이전트당 최대 상호작용 수 제한 확인"""
        pairs = select_interaction_pairs(
            agents_10, daily_logs_10, memories_10, social_10,
            {}, set(), max_pairs_per_agent=1, threshold=0.1,
        )
        # 에이전트별 매칭 횟수가 1 이하인지 확인
        count = defaultdict(int)
        for a, b, score, breakdown in pairs:
            count[a] += 1
            count[b] += 1
        for agent_id, c in count.items():
            assert c <= 1

    def test_threshold_filtering(self, agents_10, social_10, daily_logs_10, memories_10):
        """높은 threshold → 적은 결과"""
        pairs_low = select_interaction_pairs(
            agents_10, daily_logs_10, memories_10, social_10,
            {}, set(), threshold=0.1,
        )
        pairs_high = select_interaction_pairs(
            agents_10, daily_logs_10, memories_10, social_10,
            {}, set(), threshold=0.5,
        )
        assert len(pairs_low) >= len(pairs_high)


# ═══════════════════════════════════════════
# update_interaction_history 테스트
# ═══════════════════════════════════════════

class TestUpdateInteractionHistory:
    def test_update(self):
        history = {}
        pairs = [
            ("consumer_0001", "consumer_0002", 0.8, {}),
            ("consumer_0003", "consumer_0001", 0.5, {}),
        ]
        count = update_interaction_history(pairs, history)
        assert count == 2
        assert history[("consumer_0001", "consumer_0002")] == 1
        assert history[("consumer_0001", "consumer_0003")] == 1

    def test_cumulative(self):
        history = {("consumer_0001", "consumer_0002"): 3}
        pairs = [("consumer_0001", "consumer_0002", 0.8, {})]
        update_interaction_history(pairs, history)
        assert history[("consumer_0001", "consumer_0002")] == 4


# ═══════════════════════════════════════════
# extract_policy_info_holders 테스트
# ═══════════════════════════════════════════

class TestExtractPolicyInfoHolders:
    def test_no_holders(self):
        agents = [{"agent_id": "a", "_news_awareness": {"맛집 뉴스": 2}}]
        holders = extract_policy_info_holders(agents)
        assert len(holders) == 0

    def test_policy_holder(self):
        agents = [
            {"agent_id": "a", "_news_awareness": {"소상공인 쿠폰 배포": 2}},
            {"agent_id": "b", "_news_awareness": {"맛집 뉴스": 2}},
            {"agent_id": "c", "_news_awareness": {"야간 경제 특구 지정": 2}},
        ]
        holders = extract_policy_info_holders(agents)
        assert "a" in holders  # '쿠폰' 키워드
        assert "c" in holders  # '특구' 키워드
        assert "b" not in holders

    def test_heard_not_aware(self):
        """HEARD(1)는 정책 보유자로 간주 안 함"""
        agents = [{"agent_id": "a", "_news_awareness": {"할인 쿠폰": 1}}]
        holders = extract_policy_info_holders(agents)
        assert len(holders) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
