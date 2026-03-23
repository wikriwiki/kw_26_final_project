"""
GraphRAG 지식그래프 — NetworkX 기반 로컬 메모리

Seoul_BigData_PLAN.md §2단계 구현:
- KnowledgeGraph: 상권-점포-소비자-정책 관계 그래프
- EpisodicMemory: 에이전트별 행동 이력 (전체 보존)
- SocialNetwork: 에이전트 간 관계 + 입소문 전파 + 커뮤니티 탐지
- RAGRetriever: LLM 프롬프트에 삽입할 그래프 컨텍스트 검색
"""
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime


# ═══════════════════════════════════════════
# 1. KnowledgeGraph — 지식그래프
# ═══════════════════════════════════════════

class KnowledgeGraph:
    """상권-점포-업종-소비자-정책 관계 그래프.

    노드 타입: DISTRICT, INDUSTRY, CONSUMER, POLICY
    엣지 타입: CONTAINS, LIVES_IN, WORKS_IN, VISITS, TARGETS, AFFECTS
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self._district_names = {}  # dong_code → name

    # ── 초기 구축 (ETL 데이터에서) ──

    def build_from_etl(self, agents, district_profiles, industry_groups=None):
        """ETL 결과에서 지식그래프 초기 구축.

        agents: list[dict] — 에이전트 프로필
        district_profiles: DataFrame — 행정동 정보
        industry_groups: dict — {(dong, industry): {...}} — DistrictAgent에서
        """
        # 행정동 노드
        if district_profiles is not None:
            for _, row in district_profiles.iterrows():
                dong = str(row["adm_cd"])[:8]
                d_type = row.get("district_type", "HL")
                self.G.add_node(
                    f"D:{dong}", type="DISTRICT",
                    dong_code=dong, district_type=d_type,
                )

        # 업종 노드 + 업종↔행정동 관계
        if industry_groups:
            for (dong, ind), data in industry_groups.items():
                ind_id = f"I:{ind}"
                if not self.G.has_node(ind_id):
                    self.G.add_node(ind_id, type="INDUSTRY", name=ind)
                self.G.add_edge(
                    f"D:{dong}", ind_id, type="CONTAINS",
                    store_count=data.get("store_count", 0),
                )

        # 소비자 노드 + 관계
        for agent in agents:
            aid = agent["agent_id"]
            dong = str(agent.get("adm_cd", ""))[:8]
            home_dong = str(agent.get("home_adm_cd", agent.get("adm_cd", "")))[:8]

            self.G.add_node(
                f"C:{aid}", type="CONSUMER",
                segment=agent.get("segment", ""),
                gender=agent.get("gender", ""),
                age_group=agent.get("age_group", ""),
                trend_sensitivity=agent.get("trend_sensitivity", 0.5),
                loyalty=agent.get("loyalty", 0.5),
            )

            # 근무지 관계
            if self.G.has_node(f"D:{dong}"):
                self.G.add_edge(f"C:{aid}", f"D:{dong}", type="WORKS_IN")

            # 거주지 관계
            if home_dong and self.G.has_node(f"D:{home_dong}"):
                self.G.add_edge(f"C:{aid}", f"D:{home_dong}", type="LIVES_IN")

            # 선호 업종 관계
            for ind in agent.get("top_industries", []):
                ind_id = f"I:{ind}"
                if self.G.has_node(ind_id):
                    self.G.add_edge(f"C:{aid}", ind_id, type="PREFERS")

        print(f"  [GraphRAG] 초기 그래프: {self.G.number_of_nodes()} 노드, "
              f"{self.G.number_of_edges()} 엣지")

    # ── 동적 업데이트 (시뮬레이션 중) ──

    def add_policy(self, policy_name, target_dongs=None, week=0):
        """정책 노드 추가 + 대상 행정동 연결."""
        pid = f"P:{policy_name}"
        self.G.add_node(pid, type="POLICY", name=policy_name, start_week=week)
        if target_dongs:
            for dong in target_dongs:
                if self.G.has_node(f"D:{dong}"):
                    self.G.add_edge(pid, f"D:{dong}", type="TARGETS")

    def update_store_count(self, dong, industry, store_count, openings=0, closings=0):
        """점포수 변화 반영."""
        d_id = f"D:{dong}"
        i_id = f"I:{industry}"
        if self.G.has_edge(d_id, i_id):
            self.G[d_id][i_id]["store_count"] = store_count
            self.G[d_id][i_id]["recent_openings"] = openings
            self.G[d_id][i_id]["recent_closings"] = closings

    def record_visit(self, agent_id, dong, industry, satisfaction, week):
        """방문 기록을 그래프에 반영 (엣지 가중치 업데이트)."""
        c_id = f"C:{agent_id}"
        d_id = f"D:{dong}"
        i_id = f"I:{industry}"

        # 소비자 → 행정동 방문 엣지
        if self.G.has_edge(c_id, d_id):
            edge = self.G[c_id][d_id]
            edge["visit_count"] = edge.get("visit_count", 0) + 1
            edge["last_satisfaction"] = satisfaction
            edge["last_week"] = week
        elif self.G.has_node(d_id):
            self.G.add_edge(c_id, d_id, type="VISITS",
                            visit_count=1, last_satisfaction=satisfaction,
                            last_week=week)

        # 소비자 → 업종 방문 엣지
        if self.G.has_edge(c_id, i_id):
            edge = self.G[c_id][i_id]
            if edge.get("type") == "PREFERS":
                edge["visit_count"] = edge.get("visit_count", 0) + 1
                edge["last_satisfaction"] = satisfaction
            else:
                edge["visit_count"] = edge.get("visit_count", 0) + 1
                edge["last_satisfaction"] = satisfaction
        elif self.G.has_node(i_id):
            self.G.add_edge(c_id, i_id, type="VISITS",
                            visit_count=1, last_satisfaction=satisfaction)

    # ── 검색 ──

    def get_district_info(self, dong):
        """행정동 관련 정보 (업종, 점포수, 최근 변화)."""
        d_id = f"D:{dong}"
        if not self.G.has_node(d_id):
            return {}

        info = dict(self.G.nodes[d_id])
        industries = {}
        for _, target, data in self.G.out_edges(d_id, data=True):
            if data.get("type") == "CONTAINS":
                ind_name = self.G.nodes[target].get("name", target)
                industries[ind_name] = {
                    "store_count": data.get("store_count", 0),
                    "openings": data.get("recent_openings", 0),
                    "closings": data.get("recent_closings", 0),
                }
        info["industries"] = industries

        # 이 행정동을 대상으로 하는 정책
        policies = []
        for source, _, data in self.G.in_edges(d_id, data=True):
            if data.get("type") == "TARGETS":
                policies.append(self.G.nodes[source].get("name", source))
        info["active_policies"] = policies

        return info

    def get_agent_graph_context(self, agent_id):
        """에이전트 관련 그래프 이웃 정보."""
        c_id = f"C:{agent_id}"
        if not self.G.has_node(c_id):
            return {}

        context = {
            "profile": dict(self.G.nodes[c_id]),
            "work_district": None,
            "home_district": None,
            "visited_districts": [],
            "preferred_industries": [],
            "related_policies": [],
        }

        for _, target, data in self.G.out_edges(c_id, data=True):
            etype = data.get("type", "")
            if etype == "WORKS_IN":
                context["work_district"] = self.G.nodes[target].get("dong_code")
            elif etype == "LIVES_IN":
                context["home_district"] = self.G.nodes[target].get("dong_code")
            elif etype == "VISITS":
                node_data = self.G.nodes.get(target, {})
                if node_data.get("type") == "DISTRICT":
                    context["visited_districts"].append({
                        "dong": node_data.get("dong_code"),
                        "visits": data.get("visit_count", 0),
                        "last_satisfaction": data.get("last_satisfaction"),
                    })
            elif etype == "PREFERS":
                context["preferred_industries"].append(
                    self.G.nodes[target].get("name", target)
                )

        # 관련 정책 (근무지+거주지 대상)
        for dong in [context["work_district"], context["home_district"]]:
            if dong:
                d_info = self.get_district_info(dong)
                for pol in d_info.get("active_policies", []):
                    if pol not in context["related_policies"]:
                        context["related_policies"].append(pol)

        return context

    def stats(self):
        """그래프 통계."""
        type_counts = Counter(
            data.get("type", "unknown")
            for _, data in self.G.nodes(data=True)
        )
        edge_counts = Counter(
            data.get("type", "unknown")
            for _, _, data in self.G.edges(data=True)
        )
        return {"nodes": dict(type_counts), "edges": dict(edge_counts)}


# ═══════════════════════════════════════════
# 2. EpisodicMemory — 에피소딕 메모리
# ═══════════════════════════════════════════

class EpisodicMemory:
    """에이전트별 행동 이력 — 전체 보존, 검색 가능.

    7일 제한 없이 모든 에피소드를 저장하고,
    만족도 높은 경험, 특정 업종/장소 경험 등을 검색합니다.
    """

    def __init__(self):
        self._episodes = defaultdict(list)  # {agent_id: [episode, ...]}

    def record(self, agent_id, episode):
        """에피소드 기록.

        episode: {
            week, day, type, industry, dong, amount,
            satisfaction, triggered_by, ...
        }
        """
        self._episodes[agent_id].append(episode)

    def get_all(self, agent_id):
        """전체 에피소드 반환."""
        return self._episodes.get(agent_id, [])

    def get_recent(self, agent_id, n_weeks=4):
        """최근 N주 에피소드."""
        all_eps = self._episodes.get(agent_id, [])
        if not all_eps:
            return []
        max_week = max(ep.get("week", 0) for ep in all_eps)
        return [ep for ep in all_eps if ep.get("week", 0) > max_week - n_weeks]

    def get_best_experiences(self, agent_id, top_n=5):
        """만족도 가장 높았던 경험 TOP N."""
        all_eps = self._episodes.get(agent_id, [])
        outing = [ep for ep in all_eps if ep.get("type") == "외출_소비"]
        outing.sort(key=lambda x: x.get("satisfaction", 0), reverse=True)
        return outing[:top_n]

    def get_worst_experiences(self, agent_id, top_n=3):
        """만족도 가장 낮았던 경험 TOP N."""
        all_eps = self._episodes.get(agent_id, [])
        outing = [ep for ep in all_eps if ep.get("type") == "외출_소비"]
        outing.sort(key=lambda x: x.get("satisfaction", 0))
        return outing[:top_n]

    def get_industry_history(self, agent_id, industry):
        """특정 업종 방문 이력."""
        return [
            ep for ep in self._episodes.get(agent_id, [])
            if ep.get("industry") == industry
        ]

    def get_dong_history(self, agent_id, dong):
        """특정 행정동 방문 이력."""
        return [
            ep for ep in self._episodes.get(agent_id, [])
            if ep.get("dong") == dong
        ]

    def get_exploration_history(self, agent_id):
        """이전에 없던 새 업종/장소 방문 이력."""
        return [
            ep for ep in self._episodes.get(agent_id, [])
            if ep.get("is_exploration", False)
        ]

    def summary(self, agent_id):
        """에이전트 메모리 요약 통계."""
        all_eps = self._episodes.get(agent_id, [])
        if not all_eps:
            return {"total": 0}

        outings = [ep for ep in all_eps if ep.get("type") == "외출_소비"]
        industries = Counter(ep.get("industry") for ep in outings if ep.get("industry"))
        dongs = Counter(ep.get("dong") for ep in outings if ep.get("dong"))

        sats = [ep.get("satisfaction", 0.5) for ep in outings]
        return {
            "total_episodes": len(all_eps),
            "total_outings": len(outings),
            "unique_industries": len(industries),
            "top_industries": industries.most_common(5),
            "unique_dongs": len(dongs),
            "top_dongs": dongs.most_common(5),
            "avg_satisfaction": round(sum(sats) / max(len(sats), 1), 3),
        }

    def total_episodes(self):
        """전체 에피소드 수."""
        return sum(len(eps) for eps in self._episodes.values())


# ═══════════════════════════════════════════
# 3. SocialNetwork — 사회적 네트워크
# ═══════════════════════════════════════════

class SocialNetwork:
    """에이전트 간 관계 그래프 + 입소문 전파 + 커뮤니티 탐지.

    관계 유형:
    - COLLEAGUE: 같은 근무 행정동
    - NEIGHBOR: 같은 거주 행정동
    - PEER: 같은 세그먼트
    """

    def __init__(self):
        self.G = nx.Graph()  # 무방향 (social relations)
        self._recommendations = defaultdict(list)  # {agent_id: [rec, ...]}
        self._communities = {}  # {agent_id: community_id}

    def build_from_agents(self, agents, max_connections_per_type=5, rng=None):
        """에이전트 목록에서 사회적 네트워크 구축."""
        if rng is None:
            rng = np.random.default_rng()

        # 에이전트 노드 추가
        for agent in agents:
            self.G.add_node(agent["agent_id"], **{
                "segment": agent.get("segment", ""),
                "adm_cd": str(agent.get("adm_cd", ""))[:8],
                "home_adm_cd": str(agent.get("home_adm_cd", agent.get("adm_cd", "")))[:8],
            })

        # 같은 근무 행정동 → COLLEAGUE
        by_work = defaultdict(list)
        for agent in agents:
            dong = str(agent.get("adm_cd", ""))[:8]
            by_work[dong].append(agent["agent_id"])

        for dong, aids in by_work.items():
            if len(aids) > 1:
                for i, a1 in enumerate(aids):
                    # 랜덤으로 최대 N명과 연결
                    others = [a for a in aids if a != a1]
                    n_conn = min(len(others), max_connections_per_type)
                    chosen = rng.choice(others, size=n_conn, replace=False)
                    for a2 in chosen:
                        if not self.G.has_edge(a1, a2):
                            self.G.add_edge(a1, a2, type="COLLEAGUE", weight=0.6)

        # 같은 거주 행정동 → NEIGHBOR
        by_home = defaultdict(list)
        for agent in agents:
            dong = str(agent.get("home_adm_cd", agent.get("adm_cd", "")))[:8]
            by_home[dong].append(agent["agent_id"])

        for dong, aids in by_home.items():
            if len(aids) > 1:
                for a1 in aids:
                    others = [a for a in aids if a != a1]
                    n_conn = min(len(others), max_connections_per_type)
                    chosen = rng.choice(others, size=n_conn, replace=False)
                    for a2 in chosen:
                        if not self.G.has_edge(a1, a2):
                            self.G.add_edge(a1, a2, type="NEIGHBOR", weight=0.4)

        print(f"  [GraphRAG] 소셜 네트워크: {self.G.number_of_nodes()} 에이전트, "
              f"{self.G.number_of_edges()} 관계")

    def detect_communities(self):
        """Louvain 커뮤니티 탐지."""
        try:
            from community import community_louvain
            partition = community_louvain.best_partition(self.G, random_state=42)
            self._communities = partition
            n_comm = len(set(partition.values()))
            print(f"  [GraphRAG] 커뮤니티 탐지: {n_comm}개 커뮤니티")
            return partition
        except ImportError:
            # python-louvain 없으면 connected_components로 대체
            self._communities = {}
            for i, comp in enumerate(nx.connected_components(self.G)):
                for node in comp:
                    self._communities[node] = i
            return self._communities

    def propagate_recommendations(self, episodic_memory, week, rng=None):
        """입소문 전파: 만족도 높은 경험을 관계 있는 에이전트에 전달.

        조건: 만족도 0.7+ 경험 → 연결된 에이전트에게 추천
        전파 확률: 관계 가중치 × social_influence_weight
        """
        if rng is None:
            rng = np.random.default_rng()

        new_recs = 0
        for agent_id in self.G.nodes:
            # 이번 주 만족도 높은 경험
            recent = episodic_memory.get_recent(agent_id, n_weeks=1)
            great_experiences = [
                ep for ep in recent
                if ep.get("satisfaction", 0) >= 0.7 and ep.get("type") == "외출_소비"
            ]

            if not great_experiences:
                continue

            # 관계 있는 에이전트에게 전파
            for neighbor in self.G.neighbors(agent_id):
                edge = self.G[agent_id][neighbor]
                weight = edge.get("weight", 0.5)

                if rng.random() < weight * 0.5:  # 전파 확률
                    best = max(great_experiences,
                               key=lambda x: x.get("satisfaction", 0))
                    rec = {
                        "from": agent_id,
                        "industry": best.get("industry"),
                        "dong": best.get("dong"),
                        "satisfaction": best.get("satisfaction"),
                        "week": week,
                        "relation": edge.get("type", "PEER"),
                    }
                    self._recommendations[neighbor].append(rec)
                    new_recs += 1

        if new_recs > 0:
            print(f"  [GraphRAG] 입소문 전파: {new_recs}건 추천 발생")
        return new_recs

    def get_recommendations(self, agent_id, n_weeks=4):
        """에이전트가 받은 추천 (최근 N주)."""
        recs = self._recommendations.get(agent_id, [])
        if not recs:
            return []
        max_week = max(r.get("week", 0) for r in recs)
        return [r for r in recs if r.get("week", 0) > max_week - n_weeks]

    def get_community(self, agent_id):
        """에이전트가 속한 커뮤니티 ID."""
        return self._communities.get(agent_id)

    def get_community_members(self, community_id):
        """특정 커뮤니티의 멤버들."""
        return [aid for aid, cid in self._communities.items() if cid == community_id]


# ═══════════════════════════════════════════
# 4. RAGRetriever — 그래프 컨텍스트 검색
# ═══════════════════════════════════════════

class RAGRetriever:
    """LLM 프롬프트에 삽입할 그래프 컨텍스트를 조합.

    PLAN.md 311~314행 형식:
    - "을지로3가" ──(정책 시행 중)──→ "소상공인 임대료 지원"
    - "을지로 한식집A" ──(폐업)──→ "임대료 상승"
    """

    def __init__(self, knowledge_graph, episodic_memory, social_network):
        self.kg = knowledge_graph
        self.memory = episodic_memory
        self.social = social_network

    def retrieve(self, agent_id, current_week=0):
        """에이전트용 전체 그래프 컨텍스트 조합.

        Returns: str — LLM 프롬프트에 삽입할 텍스트
        """
        lines = []

        # 1. 그래프 이웃 정보
        graph_ctx = self.kg.get_agent_graph_context(agent_id)
        if not graph_ctx:
            return ""

        # 관련 정책
        for pol in graph_ctx.get("related_policies", []):
            work = graph_ctx.get("work_district", "?")
            lines.append(f'"{work}" ──(정책 시행 중)──→ "{pol}"')

        # 근무지 상권 변화
        work_dong = graph_ctx.get("work_district")
        if work_dong:
            d_info = self.kg.get_district_info(work_dong)
            for ind, data in d_info.get("industries", {}).items():
                if data.get("openings", 0) > 0:
                    lines.append(f'"{work_dong} {ind}" ──(신규 입점)──→ "{data["openings"]}곳 오픈"')
                if data.get("closings", 0) > 0:
                    lines.append(f'"{work_dong} {ind}" ──(폐업)──→ "{data["closings"]}곳 폐업"')

        # 2. 과거 경험 (에피소딕 메모리)
        best = self.memory.get_best_experiences(agent_id, top_n=3)
        for exp in best:
            w = exp.get("week", "?")
            ind = exp.get("industry", "?")
            sat = exp.get("satisfaction", 0)
            lines.append(f'"나" ──({w}주차 방문, 만족도 {sat:.1f})──→ "{ind}"')

        # 3. 동료/이웃 추천 (소셜 네트워크)
        recs = self.social.get_recommendations(agent_id, n_weeks=4)
        for rec in recs[:3]:  # 최대 3건
            rel = {"COLLEAGUE": "직장 동료", "NEIGHBOR": "동네 이웃"}.get(
                rec.get("relation"), "지인"
            )
            ind = rec.get("industry", "?")
            sat = rec.get("satisfaction", 0)
            lines.append(f'"{rel}" ──(추천, 만족도 {sat:.1f})──→ "{ind}"')

        if not lines:
            return ""

        return "\n".join(lines)

    def retrieve_for_rule_engine(self, agent_id, current_week=0):
        """룰 엔진용 구조화된 컨텍스트.

        Returns: dict with keys:
            - best_industries: 과거 만족도 높은 업종
            - avoid_industries: 과거 만족도 낮은 업종
            - peer_recommendations: 동료/이웃 추천
            - exploration_count: 새 업종 시도 횟수
        """
        best = self.memory.get_best_experiences(agent_id, top_n=5)
        worst = self.memory.get_worst_experiences(agent_id, top_n=3)
        recs = self.social.get_recommendations(agent_id, n_weeks=4)
        explores = self.memory.get_exploration_history(agent_id)

        best_industries = [ep.get("industry") for ep in best if ep.get("industry")]
        avoid_industries = [ep.get("industry") for ep in worst
                            if ep.get("industry") and ep.get("satisfaction", 0.5) < 0.3]

        peer_recs = []
        for rec in recs[:5]:
            peer_recs.append({
                "industry": rec.get("industry"),
                "dong": rec.get("dong"),
                "satisfaction": rec.get("satisfaction"),
            })

        return {
            "best_industries": best_industries,
            "avoid_industries": avoid_industries,
            "peer_recommendations": peer_recs,
            "exploration_count": len(explores),
            "total_experiences": self.memory.summary(agent_id).get("total_outings", 0),
        }


# ═══════════════════════════════════════════
# 5. GraphMemoryManager — 통합 관리
# ═══════════════════════════════════════════

class GraphMemoryManager:
    """KnowledgeGraph + EpisodicMemory + SocialNetwork + RAGRetriever 통합."""

    def __init__(self):
        self.kg = KnowledgeGraph()
        self.memory = EpisodicMemory()
        self.social = SocialNetwork()
        self.rag = RAGRetriever(self.kg, self.memory, self.social)

    def initialize(self, agents, district_profiles, industry_groups=None, rng=None):
        """ETL 데이터에서 전체 초기화."""
        self.kg.build_from_etl(agents, district_profiles, industry_groups)
        self.social.build_from_agents(agents, rng=rng)
        self.social.detect_communities()

    def record_action(self, agent_id, action, week, day):
        """에이전트 행동을 에피소딕 메모리 + 지식그래프에 기록."""
        episode = {
            "week": week,
            "day": day,
            "type": action.get("type", ""),
            "industry": action.get("industry"),
            "dong": action.get("dong"),
            "amount": action.get("amount", 0),
            "satisfaction": action.get("satisfaction", 0.5),
            "triggered_by": action.get("triggered_by", []),
            "is_exploration": action.get("is_exploration", False),
        }
        self.memory.record(agent_id, episode)

        # 지식그래프에 방문 기록
        if action.get("type") == "외출_소비" and action.get("dong") and action.get("industry"):
            self.kg.record_visit(
                agent_id, action["dong"], action["industry"],
                action.get("satisfaction", 0.5), week,
            )

    def end_of_week(self, week, rng=None):
        """주말 처리: 입소문 전파."""
        self.social.propagate_recommendations(self.memory, week, rng)

    def get_llm_context(self, agent_id, week=0):
        """LLM 프롬프트용 GraphRAG 검색 결과."""
        return self.rag.retrieve(agent_id, week)

    def get_rule_context(self, agent_id, week=0):
        """룰 엔진용 구조화된 GraphRAG 컨텍스트."""
        return self.rag.retrieve_for_rule_engine(agent_id, week)

    def stats(self):
        """전체 통계."""
        return {
            "knowledge_graph": self.kg.stats(),
            "total_episodes": self.memory.total_episodes(),
            "social_edges": self.social.G.number_of_edges(),
            "communities": len(set(self.social._communities.values()))
                            if self.social._communities else 0,
        }

    def save(self, output_dir):
        """그래프 + 메모리를 디스크에 영속화.

        저장 파일:
            output_dir/graph_knowledge.graphml  — 지식그래프
            output_dir/social_network.graphml   — 소셜 네트워크
            output_dir/episodic_memory.json     — 에피소딕 메모리
            output_dir/communities.json         — 커뮤니티 매핑
        """
        import json
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. 지식그래프 (GraphML)
        kg_path = out / "graph_knowledge.graphml"
        # GraphML은 복잡한 속성을 지원하지 않으므로 문자열로 변환
        kg_clean = self.kg.G.copy()
        for _, data in kg_clean.nodes(data=True):
            for k, v in list(data.items()):
                data[k] = str(v) if not isinstance(v, (int, float, str)) else v
        for _, _, data in kg_clean.edges(data=True):
            for k, v in list(data.items()):
                data[k] = str(v) if not isinstance(v, (int, float, str)) else v
        with open(kg_path, "wb") as f:
            nx.write_graphml(kg_clean, f)

        # 2. 소셜 네트워크 (GraphML)
        sn_path = out / "social_network.graphml"
        sn_clean = self.social.G.copy()
        for _, data in sn_clean.nodes(data=True):
            for k, v in list(data.items()):
                data[k] = str(v) if not isinstance(v, (int, float, str)) else v
        for _, _, data in sn_clean.edges(data=True):
            for k, v in list(data.items()):
                data[k] = str(v) if not isinstance(v, (int, float, str)) else v
        with open(sn_path, "wb") as f:
            nx.write_graphml(sn_clean, f)

        # 3. 에피소딕 메모리 (JSON)
        mem_path = out / "episodic_memory.json"
        mem_data = dict(self.memory._episodes)
        with open(mem_path, "w", encoding="utf-8") as f:
            json.dump(mem_data, f, ensure_ascii=False, indent=1)

        # 4. 커뮤니티 (JSON)
        comm_path = out / "communities.json"
        with open(comm_path, "w", encoding="utf-8") as f:
            json.dump(self.social._communities, f, ensure_ascii=False, indent=1)

        # 5. 추천 기록 (JSON)
        rec_path = out / "recommendations.json"
        rec_data = dict(self.social._recommendations)
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(rec_data, f, ensure_ascii=False, indent=1)

        total_size = sum(p.stat().st_size for p in out.glob("graph_*")) + \
                     sum(p.stat().st_size for p in out.glob("social_*")) + \
                     sum(p.stat().st_size for p in out.glob("episodic_*")) + \
                     sum(p.stat().st_size for p in out.glob("communities*")) + \
                     sum(p.stat().st_size for p in out.glob("recommendations*"))
        print(f"  [GraphRAG] 저장 완료: {out.name}/ ({total_size / 1024 / 1024:.1f} MB)")

    @classmethod
    def load(cls, output_dir):
        """디스크에서 그래프 + 메모리 복원."""
        import json
        from pathlib import Path

        out = Path(output_dir)
        mgr = cls()

        # 1. 지식그래프
        kg_path = out / "graph_knowledge.graphml"
        if kg_path.exists():
            mgr.kg.G = nx.read_graphml(str(kg_path))

        # 2. 소셜 네트워크
        sn_path = out / "social_network.graphml"
        if sn_path.exists():
            mgr.social.G = nx.read_graphml(str(sn_path))

        # 3. 에피소딕 메모리
        mem_path = out / "episodic_memory.json"
        if mem_path.exists():
            with open(mem_path, "r", encoding="utf-8") as f:
                mgr.memory._episodes = defaultdict(list, json.load(f))

        # 4. 커뮤니티
        comm_path = out / "communities.json"
        if comm_path.exists():
            with open(comm_path, "r", encoding="utf-8") as f:
                mgr.social._communities = json.load(f)

        # 5. 추천
        rec_path = out / "recommendations.json"
        if rec_path.exists():
            with open(rec_path, "r", encoding="utf-8") as f:
                mgr.social._recommendations = defaultdict(list, json.load(f))

        mgr.rag = RAGRetriever(mgr.kg, mgr.memory, mgr.social)
        print(f"  [GraphRAG] 로드 완료: {mgr.kg.G.number_of_nodes()} 노드, "
              f"{mgr.memory.total_episodes()} 에피소드")
        return mgr


# ═══════════════════════════════════════════
# 테스트
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import pandas as pd

    # 테스트 데이터
    test_profiles = pd.DataFrame([
        {"adm_cd": "1114055000", "district_type": "HL"},
        {"adm_cd": "1117010000", "district_type": "LH"},
    ])

    test_agents = [
        {
            "agent_id": "consumer_0001",
            "segment": "commuter",
            "gender": "남", "age_group": "30_39세",
            "adm_cd": "1114055000",
            "home_adm_cd": "1117010000",
            "top_industries": ["한식", "카페"],
            "trend_sensitivity": 0.5, "loyalty": 0.6,
        },
        {
            "agent_id": "consumer_0002",
            "segment": "commuter",
            "gender": "여", "age_group": "20_29세",
            "adm_cd": "1114055000",
            "home_adm_cd": "1114055000",
            "top_industries": ["카페", "일식"],
            "trend_sensitivity": 0.8, "loyalty": 0.3,
        },
        {
            "agent_id": "consumer_0003",
            "segment": "resident",
            "gender": "남", "age_group": "40_49세",
            "adm_cd": "1117010000",
            "home_adm_cd": "1117010000",
            "top_industries": ["한식", "편의점"],
            "trend_sensitivity": 0.2, "loyalty": 0.8,
        },
    ]

    test_industry = {
        ("11140550", "한식"): {"store_count": 15},
        ("11140550", "카페"): {"store_count": 10},
        ("11170100", "한식"): {"store_count": 8},
    }

    rng = np.random.default_rng(42)

    # 초기화
    mgr = GraphMemoryManager()
    mgr.initialize(test_agents, test_profiles, test_industry, rng=rng)

    # 행동 기록
    mgr.record_action("consumer_0001", {
        "type": "외출_소비", "industry": "한식",
        "dong": "11140550", "amount": 12000,
        "satisfaction": 0.85,
    }, week=0, day=1)

    mgr.record_action("consumer_0001", {
        "type": "외출_소비", "industry": "카페",
        "dong": "11140550", "amount": 5000,
        "satisfaction": 0.9,
    }, week=0, day=2)

    mgr.record_action("consumer_0002", {
        "type": "외출_소비", "industry": "일식",
        "dong": "11140550", "amount": 18000,
        "satisfaction": 0.75,
    }, week=0, day=1)

    # 주말 처리
    mgr.end_of_week(0, rng)

    # RAG 검색
    print("\n=== RAG Context for consumer_0001 ===")
    ctx = mgr.get_llm_context("consumer_0001", week=1)
    print(ctx)

    print("\n=== Rule Engine Context for consumer_0001 ===")
    rule_ctx = mgr.get_rule_context("consumer_0001", week=1)
    for k, v in rule_ctx.items():
        print(f"  {k}: {v}")

    print("\n=== Stats ===")
    stats = mgr.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n[OK] GraphMemoryManager test complete")
"""
Complexity: 9
Description: Complete GraphRAG implementation with NetworkX — knowledge
graph (districts, industries, consumers, policies), episodic memory (full
history per agent), social network (colleague/neighbor relations +
word-of-mouth propagation + Louvain community detection), and RAG retriever
(assembles graph context for LLM prompts in PLAN.md format).
"""
