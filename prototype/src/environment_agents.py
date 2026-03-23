"""
환경 에이전트 4종 — 상권 / 유동인구 / 정책 / 뉴스

매 라운드(=1주) Phase 1에서 순차 실행:
  PolicyAgent → DistrictAgent → PopulationAgent → NewsAgent

Seoul_BigData_PLAN.md §3-2 / DETAIL_SPEC.md §4 참조.
"""
import numpy as np
from copy import deepcopy


# ═══════════════════════════════════════════
# 1. DistrictAgent — 상권 에이전트
# ═══════════════════════════════════════════

class DistrictAgent:
    """업종 그룹 변화(개폐업, 임대료, 점포수) 관리.

    개폐업은 확률이 아닌 **소비자 행동(방문 빈도, 소비금액)** 에 의해 결정:
    - 폐업: 4주 누적 방문이 적은 업종 그룹 → stress 누적 → 임계치 초과 시 폐업
    - 입점: 4주 누적 방문이 많고 만족도 높은 행정동 → 수요가 공급을 초과하면 신규 입점

    industry_groups: {(dong_8, industry_major): {store_count, stress, demand_score, ...}}
    district_types:  {dong_8: "HH"|"HL"|"LH"|"LL"}
    """

    # stress가 이 값을 초과하면 폐업 1건 발생
    STRESS_THRESHOLD = 4.0
    # demand_score가 이 값을 초과하면 신규 입점 1건 발생
    DEMAND_THRESHOLD = 3.0

    TRANSITION_RULES = {
        "LL": ["LH"],
        "LH": ["HH", "LL"],
        "HL": ["HH", "LL"],
        "HH": ["HL"],
    }

    def __init__(self, district_profiles, industry_data=None):
        """
        district_profiles: DataFrame with adm_cd, district_type
        industry_data: optional — 기존 업종 그룹 데이터
        """
        self.district_types = {}
        self.industry_groups = {}

        if district_profiles is not None and len(district_profiles) > 0:
            for _, row in district_profiles.iterrows():
                dong = str(row["adm_cd"])[:8]
                d_type = row.get("district_type", "HL")
                self.district_types[dong] = d_type

        # 업종 그룹 초기화 (기본 업종)
        DEFAULT_INDUSTRIES = ["한식", "카페", "편의점", "중식", "일식", "패션잡화", "슈퍼마켓"]
        for dong in self.district_types:
            for ind in DEFAULT_INDUSTRIES:
                key = (dong, ind)
                self.industry_groups[key] = {
                    "store_count": np.random.randint(3, 20),
                    "estimated_revenue": np.random.randint(2000, 8000) * 10000,
                    "recent_openings": 0,
                    "recent_closings": 0,
                    "status": "active",
                    # 소비자 행동 추적
                    "weekly_visits": 0,       # 이번 주 방문 수
                    "weekly_spending": 0,     # 이번 주 소비 금액
                    "weekly_satisfaction": 0,  # 이번 주 만족도 합
                    "stress": 0.0,            # 누적 스트레스 (방문↓→증가, 방문↑→감소)
                    "demand_score": 0.0,      # 누적 수요 점수 (방문↑→증가)
                }

        self.week_events = []
        self._new_store_boost = {}  # {dong: boost from policy}

    def receive_policy_effect(self, dong, effect_type, value):
        """정책 에이전트로부터 효과 수신."""
        if effect_type == "NEW_STORE_PROBABILITY":
            self._new_store_boost[dong] = value

    def receive_consumer_activity(self, dong, industry, visits, spending, satisfaction):
        """시뮬레이션에서 소비자 행동 데이터 수신 (매주 집계 후 호출).

        Args:
            dong: 행정동 코드
            industry: 업종명
            visits: 이번 주 방문 횟수
            spending: 이번 주 총 소비금액
            satisfaction: 이번 주 평균 만족도
        """
        key = (dong, industry)
        if key in self.industry_groups:
            g = self.industry_groups[key]
            g["weekly_visits"] += visits
            g["weekly_spending"] += spending
            g["weekly_satisfaction"] += satisfaction * visits  # 가중 합

    def update_round(self, week, rng):
        """매 라운드 — 소비자 행동 기반으로 개폐업 판정."""
        self.week_events = []

        for (dong, ind), group in self.industry_groups.items():
            if group["store_count"] <= 0:
                continue

            visits = group["weekly_visits"]
            stores = group["store_count"]

            # 점포당 주간 방문 수
            visits_per_store = visits / max(stores, 1)

            # ── 스트레스 계산 (방문↓→스트레스↑) ──
            # 기준: 점포당 주 5회 방문 미만 → 스트레스 증가
            if visits_per_store < 5:
                group["stress"] += (5 - visits_per_store) * 0.3
            else:
                # 방문 충분하면 스트레스 감소
                group["stress"] = max(0, group["stress"] - 0.5)

            # 정책 부스트가 있으면 스트레스 감소
            boost = self._new_store_boost.get(dong, 0.0)
            if boost > 0:
                group["stress"] = max(0, group["stress"] - boost)

            # ── 폐업 판정: 스트레스가 임계치 초과 ──
            closings = 0
            if group["stress"] >= self.STRESS_THRESHOLD and stores > 1:
                closings = 1
                group["stress"] -= self.STRESS_THRESHOLD * 0.7  # 일부만 리셋
                group["store_count"] -= 1
                group["recent_closings"] = 1

                self.week_events.append({
                    "type": "STORE_CLOSED",
                    "dong": dong,
                    "industry": ind,
                    "count": 1,
                    "reason": f"방문 부족 (주당 {visits_per_store:.1f}회/점포)",
                    "msg": f"{dong} {ind}: 점포 1개 폐업 (방문↓)",
                })

            # ── 수요 점수 계산 (방문↑→수요↑) ──
            if visits_per_store >= 10:
                group["demand_score"] += (visits_per_store - 10) * 0.2
            else:
                group["demand_score"] = max(0, group["demand_score"] - 0.3)

            # 정책 부스트 반영
            if boost > 0:
                group["demand_score"] += boost * 0.5

            # ── 입점 판정: 수요가 임계치 초과 ──
            openings = 0
            if group["demand_score"] >= self.DEMAND_THRESHOLD:
                openings = 1
                group["demand_score"] -= self.DEMAND_THRESHOLD * 0.7
                group["store_count"] += 1
                group["recent_openings"] = 1

                self.week_events.append({
                    "type": "NEW_STORE",
                    "dong": dong,
                    "industry": ind,
                    "count": 1,
                    "reason": f"수요 과잉 (주당 {visits_per_store:.1f}회/점포)",
                    "msg": f"{dong} {ind}: 신규 1개 입점 (수요↑)",
                })

            # 기록 리셋 (다음 주를 위해)
            if closings == 0:
                group["recent_closings"] = 0
            if openings == 0:
                group["recent_openings"] = 0
            group["weekly_visits"] = 0
            group["weekly_spending"] = 0
            group["weekly_satisfaction"] = 0

        # 상권 유형 전환 판정 (10주마다)
        if week > 0 and week % 10 == 0:
            for dong, d_type in list(self.district_types.items()):
                possible = self.TRANSITION_RULES.get(d_type, [])
                if possible and rng.random() < 0.15:
                    new_type = rng.choice(possible)
                    self.district_types[dong] = new_type
                    self.week_events.append({
                        "type": "DISTRICT_TRANSITION",
                        "dong": dong,
                        "from": d_type,
                        "to": new_type,
                        "msg": f"{dong} 상권 전환: {d_type} → {new_type}",
                    })

        # 부스트 초기화 (매주 정책에서 다시 전달)
        self._new_store_boost = {}

    def get_industry_context(self, dong):
        """에이전트에게 전달할 해당 행정동 업종 변화 정보."""
        dong_groups = {k: v for k, v in self.industry_groups.items() if k[0] == dong}
        changes = []
        for (_, ind), g in dong_groups.items():
            if g["recent_closings"] > 0 or g["recent_openings"] > 0:
                changes.append({
                    "industry": ind,
                    "store_count": g["store_count"],
                    "openings": g["recent_openings"],
                    "closings": g["recent_closings"],
                })
        return {
            "district_type": self.district_types.get(dong, "unknown"),
            "industry_changes": changes,
        }


# ═══════════════════════════════════════════
# 2. PopulationAgent — 유동인구 에이전트
# ═══════════════════════════════════════════

class PopulationAgent:
    """시간대/요일별 유동인구 변화 관리.

    정책 에이전트의 POPULATION_BOOST를 수신하여 보정.
    """

    def __init__(self):
        self._base_factor = 1.0
        self._boosts = {}  # {dong: boost_value}
        self._global_change = 0.0  # -1.0 ~ +1.0 (시나리오 pop_change)

    def receive_policy_effect(self, dong, effect_type, value):
        """정책 효과 수신."""
        if effect_type == "POPULATION_BOOST":
            self._boosts[dong] = value

    def set_global_change(self, change_str):
        """시나리오의 pop_change 반영 (예: '+15%')."""
        try:
            self._global_change = float(change_str.replace("%", "").replace("+", "")) / 100
        except (ValueError, AttributeError):
            self._global_change = 0.0

    def update_round(self, week, rng):
        """매 라운드 업데이트 — 자연 변동 추가."""
        self._base_factor = 1.0 + self._global_change + rng.normal(0, 0.02)
        self._base_factor = max(0.5, min(self._base_factor, 2.0))

    def get_population_factor(self, dong, day_type="weekday"):
        """에이전트 소비 확률 보정용 유동인구 계수.

        Returns: float (1.0 = 변화 없음, >1.0 = 유동인구 증가)
        """
        boost = self._boosts.get(dong, 0.0)
        day_mod = {"weekday": 1.0, "friday": 1.1, "weekend": 0.85}.get(day_type, 1.0)
        return self._base_factor * (1.0 + boost) * day_mod

    def get_pop_change_str(self):
        """사람이 읽을 수 있는 유동인구 변화율 문자열."""
        pct = (self._base_factor - 1.0) * 100
        return f"{pct:+.1f}%"


# ═══════════════════════════════════════════
# 3. PolicyAgent — 정책 에이전트
# ═══════════════════════════════════════════

class PolicyAgent:
    """정책 생명주기: PENDING → ACTIVE → ENDED.

    누적 효과를 DistrictAgent / PopulationAgent에 전파.
    """

    # 정책 유형별 효과 규칙
    POLICY_EFFECTS = {
        "임대료 지원": {
            "target": "district",
            "effect": "NEW_STORE_PROBABILITY",
            "formula": lambda weeks: min(0.3, 0.05 * weeks),
        },
        "보행환경 개선": {
            "target": "population",
            "effect": "POPULATION_BOOST",
            "formula": lambda weeks: min(0.2, 0.03 * weeks),
        },
        "임대료 상한 규제": {
            "target": "district",
            "effect": "NEW_STORE_PROBABILITY",
            "formula": lambda weeks: max(-0.03, -0.01 * weeks),  # 신규 입점 억제
        },
        "소상공인 쿠폰": {
            "target": "population",
            "effect": "POPULATION_BOOST",
            "formula": lambda weeks: min(0.15, 0.04 * weeks),
        },
    }

    def __init__(self, scenario=None):
        self.policies = []
        self.week_events = []

        if scenario:
            for evt in scenario:
                if evt.get("type") == "policy":
                    desc = evt.get("description", "")
                    if desc == "없음":
                        continue
                    self.policies.append({
                        "name": desc,
                        "effective_week": evt.get("week", 0),
                        "duration_weeks": evt.get("duration", 12),
                        "target_dong": evt.get("target_dong"),
                        "policy_type": self._classify(desc),
                        "status": "PENDING",
                    })

    def _classify(self, name):
        """정책명 → 유형 분류."""
        for ptype in self.POLICY_EFFECTS:
            if ptype in name:
                return ptype
        if "쿠폰" in name or "할인" in name:
            return "소상공인 쿠폰"
        if "보행" in name or "조명" in name:
            return "보행환경 개선"
        if "임대" in name and "지원" in name:
            return "임대료 지원"
        if "상한" in name or "규제" in name:
            return "임대료 상한 규제"
        return "임대료 지원"  # 기본

    def update_round(self, week, district_agent, population_agent):
        """매 라운드 정책 상태 업데이트 + 효과 전파."""
        self.week_events = []

        for policy in self.policies:
            eff_week = policy["effective_week"]
            dur = policy["duration_weeks"]

            # 시작
            if week == eff_week and policy["status"] == "PENDING":
                policy["status"] = "ACTIVE"
                self.week_events.append({
                    "type": "POLICY_ACTIVE",
                    "policy_name": policy["name"],
                    "msg": f"[정책 시행] {policy['name']}",
                })

            # 진행 중 — 누적 효과 전파
            if policy["status"] == "ACTIVE":
                weeks_active = week - eff_week
                ptype = policy["policy_type"]
                rule = self.POLICY_EFFECTS.get(ptype)

                if rule:
                    value = rule["formula"](weeks_active)
                    target_dong = policy.get("target_dong")

                    if rule["target"] == "district":
                        if target_dong:
                            district_agent.receive_policy_effect(target_dong, rule["effect"], value)
                        else:
                            for dong in list(district_agent.district_types.keys())[:50]:
                                district_agent.receive_policy_effect(dong, rule["effect"], value)
                    elif rule["target"] == "population":
                        if target_dong:
                            population_agent.receive_policy_effect(target_dong, rule["effect"], value)
                        else:
                            for dong in list(district_agent.district_types.keys())[:50]:
                                population_agent.receive_policy_effect(dong, rule["effect"], value)

                    if weeks_active % 4 == 0:
                        self.week_events.append({
                            "type": "POLICY_EFFECT",
                            "policy_name": policy["name"],
                            "weeks_active": weeks_active,
                            "effect_value": round(value, 3),
                            "msg": f"[정책 {weeks_active}주차] {policy['name']}: {rule['effect']}={value:.3f}",
                        })

            # 종료
            if week >= eff_week + dur and policy["status"] == "ACTIVE":
                policy["status"] = "ENDED"
                self.week_events.append({
                    "type": "POLICY_ENDED",
                    "policy_name": policy["name"],
                    "msg": f"[정책 종료] {policy['name']}",
                })

    def get_active_policy_str(self):
        """현재 활성 정책 문자열."""
        active = [p["name"] for p in self.policies if p["status"] == "ACTIVE"]
        return ", ".join(active) if active else "없음"


# ═══════════════════════════════════════════
# 4. NewsAgent — 뉴스 에이전트
# ═══════════════════════════════════════════

class NewsAgent:
    """다양한 현실적 뉴스/이벤트 생성 에이전트.

    8개 이벤트 카테고리 + 서울 핫스팟 POI 시스템.
    뉴스를 구조화된 이벤트 딕셔너리로 반환하여
    에이전트가 카테고리별로 차등 반응할 수 있게 합니다.
    """

    DONG_NAMES = {
        "11110": "종로", "11140": "중구", "11170": "용산",
        "11200": "성동", "11215": "광진", "11230": "동대문",
        "11260": "중랑", "11290": "성북", "11305": "강북",
        "11320": "도봉", "11350": "노원", "11380": "은평",
        "11410": "서대문", "11440": "마포", "11470": "양천",
        "11500": "강서", "11530": "구로", "11545": "금천",
        "11560": "영등포", "11590": "동작", "11620": "관악",
        "11650": "서초", "11680": "강남", "11710": "송파",
        "11740": "강동",
    }

    SEG_KR = {
        "commuter": "직장인", "weekend_visitor": "주말방문객",
        "resident": "거주민", "evening_visitor": "저녁방문자",
    }

    # ── 서울 핫스팟 POI ──
    HOTSPOTS = {
        "성수_카페거리": {"dong": "11200", "lat": 37.5446, "lng": 127.0567, "radius": 200,
                         "industries": ["카페", "베이커리"], "vibe": "힙한 카페골목"},
        "을지로_노가리골목": {"dong": "11140", "lat": 37.5660, "lng": 126.9910, "radius": 150,
                            "industries": ["주류", "한식"], "vibe": "레트로 감성 골목"},
        "홍대_걷고싶은거리": {"dong": "11440", "lat": 37.5563, "lng": 126.9236, "radius": 300,
                            "industries": ["카페", "패션잡화", "문화여가"], "vibe": "거리공연+쇼핑"},
        "익선동_한옥거리": {"dong": "11110", "lat": 37.5741, "lng": 126.9878, "radius": 100,
                           "industries": ["카페", "한식", "베이커리"], "vibe": "한옥 감성"},
        "잠실_롯데타워": {"dong": "11710", "lat": 37.5136, "lng": 127.1025, "radius": 250,
                         "industries": ["패션잡화", "문화여가", "카페"], "vibe": "대형 쇼핑몰"},
        "강남역_먹자골목": {"dong": "11680", "lat": 37.4979, "lng": 127.0276, "radius": 200,
                          "industries": ["한식", "일식", "중식", "카페"], "vibe": "직장인 맛집"},
        "가로수길": {"dong": "11650", "lat": 37.5197, "lng": 127.0233, "radius": 250,
                    "industries": ["카페", "패션잡화", "베이커리"], "vibe": "감성 카페+쇼핑"},
        "여의도_IFC": {"dong": "11560", "lat": 37.5255, "lng": 126.9256, "radius": 200,
                      "industries": ["패션잡화", "문화여가", "카페"], "vibe": "직장인 쇼핑"},
        "광화문_세종로": {"dong": "11110", "lat": 37.5723, "lng": 126.9769, "radius": 200,
                        "industries": ["한식", "카페", "문화여가"], "vibe": "역사+오피스"},
        "이태원_경리단길": {"dong": "11170", "lat": 37.5375, "lng": 126.9874, "radius": 250,
                          "industries": ["주류", "카페", "일식"], "vibe": "다국적 맛집거리"},
        "망원동_망리단길": {"dong": "11440", "lat": 37.5563, "lng": 126.9089, "radius": 150,
                          "industries": ["카페", "베이커리", "한식"], "vibe": "로컬 감성"},
        "종로_종삼골목": {"dong": "11110", "lat": 37.5713, "lng": 126.9920, "radius": 100,
                        "industries": ["주류", "한식"], "vibe": "저렴한 안주골목"},
        "신촌_연세로": {"dong": "11410", "lat": 37.5593, "lng": 126.9367, "radius": 200,
                       "industries": ["패스트푸드", "카페", "주류"], "vibe": "대학가"},
        "건대입구_먹자골목": {"dong": "11215", "lat": 37.5404, "lng": 127.0698, "radius": 200,
                            "industries": ["한식", "주류", "카페"], "vibe": "대학가 맛집"},
        "여의도_벚꽃길": {"dong": "11560", "lat": 37.5258, "lng": 126.9337, "radius": 300,
                         "industries": ["카페", "편의점"], "vibe": "봄 명소"},
        "용산_HDC아이파크몰": {"dong": "11170", "lat": 37.5299, "lng": 126.9655, "radius": 200,
                              "industries": ["패션잡화", "문화여가", "카페"], "vibe": "복합몰"},
        "코엑스_삼성역": {"dong": "11680", "lat": 37.5116, "lng": 127.0595, "radius": 300,
                         "industries": ["문화여가", "카페", "패션잡화"], "vibe": "컨벤션+쇼핑"},
        "압구정_로데오": {"dong": "11680", "lat": 37.5270, "lng": 127.0388, "radius": 200,
                        "industries": ["패션잡화", "카페", "미용"], "vibe": "럭셔리 패션"},
        "북촌_한옥마을": {"dong": "11110", "lat": 37.5826, "lng": 126.9832, "radius": 200,
                         "industries": ["카페", "문화여가"], "vibe": "전통+관광"},
        "노량진_수산시장": {"dong": "11590", "lat": 37.5134, "lng": 126.9407, "radius": 100,
                          "industries": ["일식", "한식"], "vibe": "신선한 해산물"},
    }

    # ── 이벤트 카테고리 정의 ──
    EVENT_CATEGORIES = {
        "SUBSIDY": {
            "templates": [
                "{area} 일대 소상공인에게 월 {amount}만원 지원금 지급",
                "서울시, {area} 상권활성화 위해 임대료 {pct}% 보전 발표",
                "{area} 전통시장 활성화 바우처 {amount}만원 배포",
                "지역화폐 '{area}사랑상품권' 10% 할인 판매 시작",
            ],
            "affected_industries": ["all"],
            "spending_boost": 1.1,
            "population_boost": 1.05,
        },
        "REAL_ESTATE": {
            "templates": [
                "{area} 상가 매매가 전월 대비 {pct}% 상승, 임차인 부담 가중",
                "{area} 상가 공실률 역대 최저 {pct}%, 신규 입점 러시",
                "{area} 일대 재건축 기대감에 상가 프리미엄 급등",
                "{area} 월세 상승으로 소형 자영업 폐업 잇따라",
            ],
            "affected_industries": ["all"],
            "spending_boost": 0.95,
            "population_boost": 1.0,
        },
        "REDEVELOPMENT": {
            "templates": [
                "{area} 재개발 사업 확정, 향후 2년간 일부 구역 통행 제한",
                "{area} 도시재생 뉴딜사업 본격 착수, 상인 임시 이전",
                "서울시 '{area} 도시재정비 촉진지구' 지정 고시",
                "{area} 복합개발 사업 인허가 확정, 랜드마크 조성 기대",
            ],
            "affected_industries": ["all"],
            "spending_boost": 0.85,
            "population_boost": 0.8,
        },
        "SNS_VIRAL": {
            "templates": [
                "{area} '{item}' 인스타 릴스에서 화제, 조회수 {views}만",
                "틱톡 '{item}' 챌린지 확산, {area} 방문 인증 폭주",
                "유튜버 '{influencer}'가 {area} 맛집 투어 영상 공개, 조회수 {views}만",
                "{area} '{item}' SNS 바이럴, 주말 웨이팅 2시간",
                "인스타 핫플 '{area} {item}' 게시물 {views}만건 돌파",
            ],
            "affected_industries": ["카페", "베이커리", "한식", "일식"],
            "spending_boost": 1.15,
            "population_boost": 1.3,
            "target_demo": ["20_29세", "30_39세"],
        },
        "ENTERTAINMENT": {
            "templates": [
                "잠실종합운동장 {artist} 콘서트 {duration}일간, 팬 {views}만 명 예상",
                "{area}에서 '{festival}' 축제 개최, {duration}일간 진행",
                "올림픽공원 {artist} 내한공연 전석 매진, {area} 일대 교통 혼잡 예상",
                "{area} '{festival}' 페스티벌 D-{duration}, 사전 예매 오픈",
                "고척돔 {artist} 콘서트 개최, {area} 숙박·음식점 예약 급증",
            ],
            "affected_industries": ["카페", "편의점", "패션잡화", "숙박", "패스트푸드"],
            "spending_boost": 1.2,
            "population_boost": 1.5,
            "target_demo": ["20_29세", "30_39세", "10_19세"],
        },
        "SEASONAL": {
            "templates_by_month": {
                0: ["2026 새해맞이 할인 페스타 전국 백화점 동시 시행",
                    "정초 한파 기승, 실내 쇼핑몰·카페 방문객 증가"],
                1: ["설 연휴 앞두고 서울 전역 전통시장 특별 할인전",
                    "밸런타인데이 앞두고 강남·홍대 디저트 매장 특수"],
                2: ["벚꽃 개화 시즌 시작, 여의도·석촌호수 일대 인파 예상",
                    "봄맞이 카페 신메뉴 출시 러시, 딸기 디저트 전성시대"],
                3: ["서울 곳곳 봄꽃 축제 개최, 야외 나들이 수요 급증",
                    "봄나들이 시즌, 한강공원·북촌 관광객 전년 대비 20%↑"],
                4: ["어린이날·어버이날 연속 가정의 달, 외식 수요 급증 예상",
                    "초여름 날씨에 아이스 음료 매출 전년 대비 30%↑"],
                5: ["서울 낮 기온 30도 돌파, 빙수·아이스크림 매출 급등",
                    "여름 휴가철 시작, 서울 빠져나가는 인구 늘어 상권 소강기"],
                6: ["장마 시작, 우산·레인부츠 매출↑ 야외 상권 타격",
                    "폭염경보 발령, 에어컨 있는 카페·쇼핑몰로 인파 몰려"],
                7: ["역대급 폭염에 야외 활동 자제 권고, 배달 수요 폭증",
                    "여름 할인 시즌 '서머세일' 강남·명동 쇼핑몰 인파"],
                8: ["가을 축제 시즌 개막, 서울 곳곳 문화행사 풍성",
                    "선선한 날씨에 야외 테라스 카페 매출 30%↑"],
                9: ["단풍 시즌 시작, 북한산·남산 일대 방문객 급증",
                    "핼러윈 앞두고 이태원·홍대 특수 기대감 고조"],
                10: ["빼빼로데이 특수, 편의점·베이커리 매출 역대 최고",
                     "연말 분위기 시작, 크리스마스 마켓 준비 한창"],
                11: ["크리스마스·연말 모임 시즌, 레스토랑 예약 풀가동",
                     "연말 할인 시즌 '윈터세일' 전 상권 활기"],
            },
            "affected_industries": ["all"],
            "spending_boost": 1.05,
            "population_boost": 1.0,
        },
        "FOOD_TREND": {
            "templates": [
                "서울 전역 '{item}' 열풍, {area}에만 전문점 {count}곳 오픈",
                "맛집 이어 '{item}' 유행 시작, SNS 인증 열풍",
                "'{item}' 맛집 전국 확산, {area} 골목 줄 서는 가게 속출",
                "'{item}' 밀키트 출시에 오프라인 매장 동반 인기",
                "해외 유행 '{item}' 국내 상륙, {area} 1호점 웨이팅 3시간",
            ],
            "affected_industries": ["베이커리", "카페", "한식", "일식"],
            "spending_boost": 1.1,
            "population_boost": 1.15,
            "target_demo": ["20_29세", "30_39세"],
        },
        "POLICY_ANNOUNCE": {
            "templates": [
                "서울시, {area} 보행자 전용구역 지정 추진",
                "{area} 야간 경제 특구 지정, 심야영업 지원",
                "'서울 야시장' {area}에 신규 오픈, 매주 금토 운영",
                "서울시 '{area} 먹거리 거리' 조성사업 착수",
                "{area} 대규모 복합상업시설 그랜드오픈, 매장 {count}개",
            ],
            "affected_industries": ["all"],
            "spending_boost": 1.08,
            "population_boost": 1.1,
        },
    }

    # SNS 바이럴 아이템 풀
    VIRAL_ITEMS = [
        "맛집", "약과 크루아상", "흑임자 라떼", "크로플",
        "소금빵", "마라탕후루", "감자빵", "생크림 떡볶이",
        "수플레 팬케이크", "디저트 오마카세", "제주 말차 라떼",
        "크럼블 쿠키", "바스크 치즈케이크", "에그타르트", "딸기 찹쌀떡",
    ]

    # 유튜버/인플루언서 풀
    INFLUENCERS = [
        "먹보형", "하얀트리", "떵개떵", "쯔양", "히밥",
        "성시경", "백종원", "풍자", "입짧은햇님", "승우아빠",
    ]

    # 공연/행사 풀
    ARTISTS = [
        "BTS", "BLACKPINK", "NewJeans", "aespa", "IVE",
        "르세라핌", "세븐틴", "스트레이키즈", "아이유", "임영웅",
    ]
    FESTIVALS = [
        "서울 월드 뮤직 페스티벌", "서울 재즈 페스티벌", "서울 푸드 위크",
        "DDP 디자인 마켓", "한강 불꽃축제", "서울 빛초롱 축제",
        "서울 거리예술 축제", "북촌 한옥 음악회", "서울 국제 커피쇼",
    ]

    # 식품 트렌드 아이템 풀
    FOOD_TRENDS = [
        "흑임자 라떼", "크럼블 쿠키", "할매니얼 간식", "제로슈거 디저트",
        "비건 버거", "수제 맥주", "오마카세", "탕후루 2.0",
        "단백질 도시락", "저탄고지 빵", "모찌 도넛", "말차 붐",
    ]

    def __init__(self, scenario=None):
        self.current_news = []      # 구조화된 이벤트 딕셔너리 리스트
        self._prev_state = {}

    def _dong_to_name(self, dong_code):
        prefix = str(dong_code)[:5]
        return self.DONG_NAMES.get(prefix, f"{dong_code} 일대")

    def _find_hotspot(self, dong=None, industries=None):
        """동 코드 또는 업종에 매칭되는 핫스팟 찾기."""
        best = None
        for name, hs in self.HOTSPOTS.items():
            if dong and not hs["dong"].startswith(str(dong)[:5]):
                continue
            if industries:
                overlap = set(hs["industries"]) & set(industries)
                if not overlap:
                    continue
            best = {"name": name, **hs}
            break
        return best

    def update_round(self, week, district_events, district_agent=None,
                     population_agent=None, policy_agent=None, rng=None,
                     graph_mgr=None, use_llm=True):
        """매 라운드: 에이전트가 자율적으로 다양한 뉴스 이벤트 생성.

        1순위: LLM 기반 (시뮬레이션 상태 분석 → 뉴스 추론)
        2순위: 규칙 기반 (카테고리별 확률적 이벤트 생성)
        """
        self.current_news = []

        # ── 컨텍스트 수집 ──
        context = self._build_context(
            week, district_events, district_agent,
            population_agent, policy_agent, graph_mgr,
        )

        # ── 1) LLM 뉴스 생성 시도 ──
        if use_llm:
            llm_news = self._generate_news_llm(context, week)
            if llm_news:
                self.current_news = llm_news
                return

        # ── 2) 규칙 기반: 다양한 이벤트 카테고리에서 자율 생성 ──
        self._generate_diverse_news(context, week, district_events,
                                    population_agent, policy_agent, rng)

        # 최소 1건 보장
        if not self.current_news:
            self.current_news.append({
                "headline": "이번 주 서울 상권 안정적 흐름 유지, 특이 사항 없음",
                "category": "SEASONAL",
                "target_area": None,
                "affected_industries": [],
                "spending_boost": 1.0,
                "population_boost": 1.0,
                "target_demo": [],
                "hotspot": None,
                "duration_weeks": 1,
            })

    def _generate_diverse_news(self, context, week, district_events,
                               population_agent, policy_agent, rng):
        """규칙 기반: 다양한 카테고리에서 현실적 뉴스 생성."""
        if rng is None:
            return

        month = (week // 4) % 12

        # ── 계절 뉴스 (매주 높은 확률로 1건) ──
        seasonal_templates = self.EVENT_CATEGORIES["SEASONAL"]["templates_by_month"]
        if month in seasonal_templates and rng.random() < 0.5:
            headline = rng.choice(seasonal_templates[month])
            self.current_news.append({
                "headline": headline,
                "category": "SEASONAL",
                "target_area": None,
                "affected_industries": ["all"],
                "spending_boost": self.EVENT_CATEGORIES["SEASONAL"]["spending_boost"],
                "population_boost": 1.0,
                "target_demo": [],
                "hotspot": None,
                "duration_weeks": 1,
            })

        # ── SNS 바이럴 (30% 확률) ──
        if rng.random() < 0.30:
            item = rng.choice(self.VIRAL_ITEMS)
            hs_names = list(self.HOTSPOTS.keys())
            hs_key = rng.choice(hs_names)
            hotspot = {"name": hs_key, **self.HOTSPOTS[hs_key]}
            area = self._dong_to_name(hotspot["dong"])
            views = rng.integers(50, 800)

            templates = self.EVENT_CATEGORIES["SNS_VIRAL"]["templates"]
            tmpl = rng.choice(templates)
            influencer = rng.choice(self.INFLUENCERS)
            headline = tmpl.format(area=area, item=item, views=views, influencer=influencer)

            self.current_news.append({
                "headline": headline,
                "category": "SNS_VIRAL",
                "target_area": hotspot["dong"],
                "affected_industries": self.EVENT_CATEGORIES["SNS_VIRAL"]["affected_industries"],
                "spending_boost": self.EVENT_CATEGORIES["SNS_VIRAL"]["spending_boost"],
                "population_boost": self.EVENT_CATEGORIES["SNS_VIRAL"]["population_boost"],
                "target_demo": self.EVENT_CATEGORIES["SNS_VIRAL"]["target_demo"],
                "hotspot": hotspot,
                "duration_weeks": rng.integers(1, 3),
            })

        # ── 음식 트렌드 (20% 확률) ──
        if rng.random() < 0.20:
            item = rng.choice(self.FOOD_TRENDS)
            hs_names = list(self.HOTSPOTS.keys())
            hs_key = rng.choice(hs_names)
            hotspot = {"name": hs_key, **self.HOTSPOTS[hs_key]}
            area = self._dong_to_name(hotspot["dong"])
            count = rng.integers(3, 15)

            templates = self.EVENT_CATEGORIES["FOOD_TREND"]["templates"]
            tmpl = rng.choice(templates)
            headline = tmpl.format(area=area, item=item, count=count)

            self.current_news.append({
                "headline": headline,
                "category": "FOOD_TREND",
                "target_area": hotspot["dong"],
                "affected_industries": self.EVENT_CATEGORIES["FOOD_TREND"]["affected_industries"],
                "spending_boost": self.EVENT_CATEGORIES["FOOD_TREND"]["spending_boost"],
                "population_boost": self.EVENT_CATEGORIES["FOOD_TREND"]["population_boost"],
                "target_demo": self.EVENT_CATEGORIES["FOOD_TREND"].get("target_demo", []),
                "hotspot": hotspot,
                "duration_weeks": rng.integers(2, 5),
            })

        # ── 공연/행사 (15% 확률) ──
        if rng.random() < 0.15:
            artist = rng.choice(self.ARTISTS)
            festival = rng.choice(self.FESTIVALS)
            duration = rng.integers(1, 4)
            views = rng.integers(3, 20)
            # 주로 잠실/올림픽공원/고척 근처
            ent_hotspots = ["잠실_롯데타워", "코엑스_삼성역"]
            hs_key = rng.choice(ent_hotspots)
            hotspot = {"name": hs_key, **self.HOTSPOTS[hs_key]}
            area = self._dong_to_name(hotspot["dong"])

            templates = self.EVENT_CATEGORIES["ENTERTAINMENT"]["templates"]
            tmpl = rng.choice(templates)
            headline = tmpl.format(area=area, artist=artist, festival=festival,
                                   duration=duration, views=views)

            self.current_news.append({
                "headline": headline,
                "category": "ENTERTAINMENT",
                "target_area": hotspot["dong"],
                "affected_industries": self.EVENT_CATEGORIES["ENTERTAINMENT"]["affected_industries"],
                "spending_boost": self.EVENT_CATEGORIES["ENTERTAINMENT"]["spending_boost"],
                "population_boost": self.EVENT_CATEGORIES["ENTERTAINMENT"]["population_boost"],
                "target_demo": self.EVENT_CATEGORIES["ENTERTAINMENT"].get("target_demo", []),
                "hotspot": hotspot,
                "duration_weeks": duration,
            })

        # ── 부동산 이슈 (10% 확률) ──
        if rng.random() < 0.10:
            dong_codes = list(self.DONG_NAMES.keys())
            dong = rng.choice(dong_codes)
            area = self.DONG_NAMES[dong]
            pct = round(rng.uniform(3, 12), 1)

            templates = self.EVENT_CATEGORIES["REAL_ESTATE"]["templates"]
            tmpl = rng.choice(templates)
            headline = tmpl.format(area=area, pct=pct)

            self.current_news.append({
                "headline": headline,
                "category": "REAL_ESTATE",
                "target_area": dong,
                "affected_industries": ["all"],
                "spending_boost": self.EVENT_CATEGORIES["REAL_ESTATE"]["spending_boost"],
                "population_boost": 1.0,
                "target_demo": [],
                "hotspot": None,
                "duration_weeks": rng.integers(2, 6),
            })

        # ── 정책 이벤트 전달 ──
        if policy_agent:
            for evt in policy_agent.week_events:
                if evt["type"] == "POLICY_ACTIVE":
                    self.current_news.append({
                        "headline": f"서울시 '{evt['policy_name']}' 정책 시행",
                        "category": "SUBSIDY",
                        "target_area": None,
                        "affected_industries": ["all"],
                        "spending_boost": 1.08,
                        "population_boost": 1.05,
                        "target_demo": [],
                        "hotspot": None,
                        "duration_weeks": 4,
                    })

        # ── 상권 변화 (폐업/입점 — 여전히 기록하되 주요 뉴스가 아님) ──
        from collections import Counter
        dong_openings = Counter()
        dong_closings = Counter()
        for evt in district_events:
            if evt["type"] == "NEW_STORE":
                dong_openings[evt["dong"]] += evt.get("count", 1)
            elif evt["type"] == "STORE_CLOSED":
                dong_closings[evt["dong"]] += evt.get("count", 1)

        for dong, count in dong_openings.most_common(1):
            if count >= 3:
                area = self._dong_to_name(dong)
                self.current_news.append({
                    "headline": f"{area} 일대 신규 점포 {count}곳 동시 오픈, 상권 활기",
                    "category": "POLICY_ANNOUNCE",
                    "target_area": dong,
                    "affected_industries": ["all"],
                    "spending_boost": 1.05,
                    "population_boost": 1.05,
                    "target_demo": [],
                    "hotspot": self._find_hotspot(dong=dong),
                    "duration_weeks": 1,
                })

        # 최대 3건까지
        self.current_news = self.current_news[:3]

    def _build_context(self, week, district_events, district_agent,
                       population_agent, policy_agent, graph_mgr):
        """LLM에 전달할 시뮬레이션 상태 요약 — 인과관계 데이터 포함."""
        from collections import Counter, defaultdict

        ctx = {"week": week}

        # ── 상권 변화 + 인과 데이터 ──
        openings, closings = [], []
        ind_open = Counter()
        ind_close = Counter()
        dong_closings = Counter()   # 동별 폐업 수
        dong_openings = Counter()   # 동별 입점 수
        for evt in district_events:
            if evt["type"] == "NEW_STORE":
                area = self._dong_to_name(evt["dong"])
                openings.append(f"{area} {evt['industry']}")
                ind_open[evt["industry"]] += 1
                dong_openings[evt["dong"]] += 1
            elif evt["type"] == "STORE_CLOSED":
                area = self._dong_to_name(evt["dong"])
                closings.append(f"{area} {evt['industry']}")
                ind_close[evt["industry"]] += 1
                dong_closings[evt["dong"]] += 1
        ctx["openings"] = openings[:5]
        ctx["closings"] = closings[:5]
        ctx["hot_industries"] = dict(ind_open.most_common(3))
        ctx["declining_industries"] = dict(ind_close.most_common(3))

        # ── 인과 데이터: 위기 지역 + 활황 지역 ──
        crisis_areas = []
        for dong, cnt in dong_closings.most_common(3):
            if cnt >= 2:
                crisis_areas.append({
                    "dong": dong,
                    "area_name": self._dong_to_name(dong),
                    "closings": cnt,
                    "reason": "방문객 감소로 폐업 집중",
                })
        ctx["crisis_areas"] = crisis_areas

        boom_areas = []
        for dong, cnt in dong_openings.most_common(3):
            if cnt >= 1:
                boom_areas.append({
                    "dong": dong,
                    "area_name": self._dong_to_name(dong),
                    "openings": cnt,
                })
        ctx["boom_areas"] = boom_areas

        # ── 상권 스트레스 분석 (DistrictAgent 내부 데이터 활용) ──
        if district_agent:
            stressed_stores = []
            for (dong, ind), data in district_agent.industry_groups.items():
                stress = data.get("stress", 0)
                if stress > 2.0:
                    stressed_stores.append({
                        "dong": dong,
                        "area": self._dong_to_name(dong),
                        "industry": ind,
                        "stress": round(stress, 1),
                    })
            stressed_stores.sort(key=lambda x: x["stress"], reverse=True)
            ctx["stressed_stores"] = stressed_stores[:5]

        # 유동인구
        if population_agent:
            ctx["pop_change"] = population_agent.get_pop_change_str()

        # 정책
        if policy_agent:
            ctx["policy"] = policy_agent.get_active_policy_str()
            ctx["policy_events"] = [
                e.get("msg", "") for e in policy_agent.week_events
            ]

        # ── 커뮤니티 소비 패턴 (GraphRAG) ──
        if graph_mgr and graph_mgr.social._communities:
            communities = []
            comm_data = defaultdict(lambda: {
                "members": 0, "spending": 0, "sats": [],
                "industries": Counter(), "segments": Counter(),
            })

            for aid, episodes in graph_mgr.memory._episodes.items():
                cid = graph_mgr.social.get_community(aid)
                if cid is None:
                    continue
                cd = comm_data[cid]
                cd["members"] += 1
                recent = episodes[-7:] if len(episodes) > 7 else episodes
                for ep in recent:
                    cd["spending"] += ep.get("amount", 0)
                    if ep.get("satisfaction"):
                        cd["sats"].append(ep["satisfaction"])
                    if ep.get("industry"):
                        cd["industries"][ep["industry"]] += 1

            sorted_comms = sorted(comm_data.items(),
                                  key=lambda x: x[1]["members"], reverse=True)[:5]
            for cid, cd in sorted_comms:
                n = max(cd["members"], 1)
                avg_sat = round(np.mean(cd["sats"]), 2) if cd["sats"] else 0.5
                top_inds = [k for k, _ in cd["industries"].most_common(3)]
                communities.append({
                    "id": cid,
                    "members": cd["members"],
                    "avg_spending": round(cd["spending"] / n),
                    "satisfaction": avg_sat,
                    "top_industries": top_inds,
                })
            ctx["communities"] = communities

        # 계절
        month = (week // 4) % 12
        seasons = {0: "겨울(1월)", 1: "겨울(2월)", 2: "봄(3월)", 3: "봄(4월)",
                   4: "봄(5월)", 5: "여름(6월)", 6: "여름(7월)", 7: "여름(8월)",
                   8: "가을(9월)", 9: "가을(10월)", 10: "가을(11월)", 11: "겨울(12월)"}
        ctx["season"] = seasons.get(month, "")

        # ── 사용 가능한 핫스팟 리스트 ──
        ctx["hotspots"] = list(self.HOTSPOTS.keys())

        return ctx


    def _generate_news_llm(self, context, week):
        """LLM으로 시뮬레이션 상태 기반 뉴스 생성."""
        import requests, json

        try:
            from llm_client import OLLAMA_URL, MODEL_NAME
            r = requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=3)
            if r.status_code != 200:
                return None
        except Exception:
            return None

        # 컨텍스트 텍스트 구성
        lines = [f"현재 시점: 시뮬레이션 {week}주차, 계절: {context.get('season', '?')}"]

        if context.get("openings"):
            lines.append(f"신규 입점: {', '.join(context['openings'][:3])}")
        if context.get("closings"):
            lines.append(f"폐업: {', '.join(context['closings'][:3])}")
        if context.get("hot_industries"):
            lines.append(f"성장 업종: {context['hot_industries']}")
        if context.get("declining_industries"):
            lines.append(f"침체 업종: {context['declining_industries']}")
        if context.get("pop_change"):
            lines.append(f"유동인구 변화: {context['pop_change']}")
        if context.get("policy") and context["policy"] != "없음":
            lines.append(f"현재 정책: {context['policy']}")

        # 커뮤니티 패턴
        communities = context.get("communities", [])
        if communities:
            lines.append("\n[소비자 커뮤니티 동향]")
            for c in communities[:3]:
                lines.append(
                    f"- 그룹{c['id']}({c['members']}명): "
                    f"만족도 {c['satisfaction']}, "
                    f"주요업종 {','.join(c['top_industries'][:3])}"
                )

        ctx_text = "\n".join(lines)

        # 이벤트 카테고리 안내
        categories = ", ".join(self.EVENT_CATEGORIES.keys())

        prompt = f"""당신은 서울 지역 경제/상권 전문 기자입니다.
아래 데이터를 분석하여 이번 주 서울 상권 관련 뉴스 헤드라인을 1~3건 생성하세요.

[이번 주 상황]
{ctx_text}

[규칙]
- 실제 한국 뉴스처럼 자연스러운 헤드라인을 작성하세요.
- 다음 카테고리 중에서 선택하세요: {categories}
- 단순히 폐업/입점만 다루지 말고, SNS 바이럴, 식품 트렌드, 공연, 부동산, 정책 등 다양한 주제를 다루세요.
- 변화가 미미하면 1건만, 큰 변화가 있으면 2~3건 생성하세요.
- JSON 배열로 반환: [{{"headline": "...", "category": "..."}}]
- /no_think"""

        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.8, "top_p": 0.9, "num_predict": 400},
            }
            resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()

            if "<think>" in raw:
                raw = raw.split("</think>")[-1].strip()
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            if "[" in raw:
                raw = raw[raw.index("["):raw.rindex("]") + 1]

            news_list = json.loads(raw)
            result = []
            for item in news_list[:3]:
                if isinstance(item, str):
                    # 단순 문자열 → 구조화
                    result.append(self._string_to_event(item))
                elif isinstance(item, dict):
                    cat = item.get("category", "SEASONAL")
                    cat_def = self.EVENT_CATEGORIES.get(cat, {})
                    result.append({
                        "headline": item.get("headline", str(item)),
                        "category": cat,
                        "target_area": None,
                        "affected_industries": cat_def.get("affected_industries", []),
                        "spending_boost": cat_def.get("spending_boost", 1.0),
                        "population_boost": cat_def.get("population_boost", 1.0),
                        "target_demo": cat_def.get("target_demo", []),
                        "hotspot": None,
                        "duration_weeks": 1,
                    })
            return result if result else None

        except Exception:
            pass

        return None

    def _string_to_event(self, text):
        """문자열 뉴스 → 구조화된 이벤트 딕셔너리 변환."""
        cat = "SEASONAL"  # 기본
        for keyword, category in [
            ("바이럴", "SNS_VIRAL"), ("인스타", "SNS_VIRAL"), ("틱톡", "SNS_VIRAL"),
            ("유튜브", "SNS_VIRAL"), ("SNS", "SNS_VIRAL"),
            ("콘서트", "ENTERTAINMENT"), ("축제", "ENTERTAINMENT"), ("공연", "ENTERTAINMENT"),
            ("매매가", "REAL_ESTATE"), ("공실", "REAL_ESTATE"), ("임대료", "REAL_ESTATE"),
            ("재개발", "REDEVELOPMENT"), ("도시재생", "REDEVELOPMENT"),
            ("지원금", "SUBSIDY"), ("보조금", "SUBSIDY"), ("바우처", "SUBSIDY"),
            ("트렌드", "FOOD_TREND"), ("열풍", "FOOD_TREND"), ("유행", "FOOD_TREND"),
            ("특구", "POLICY_ANNOUNCE"), ("보행자", "POLICY_ANNOUNCE"), ("오픈", "POLICY_ANNOUNCE"),
        ]:
            if keyword in text:
                cat = category
                break

        cat_def = self.EVENT_CATEGORIES.get(cat, {})
        return {
            "headline": text,
            "category": cat,
            "target_area": None,
            "affected_industries": cat_def.get("affected_industries", []),
            "spending_boost": cat_def.get("spending_boost", 1.0),
            "population_boost": cat_def.get("population_boost", 1.0),
            "target_demo": cat_def.get("target_demo", []),
            "hotspot": None,
            "duration_weeks": 1,
        }

    def get_news(self):
        """구조화된 뉴스 이벤트 리스트 반환."""
        return self.current_news

    def get_news_headlines(self):
        """헤드라인 문자열 리스트 반환 (하위 호환용)."""
        return [n["headline"] if isinstance(n, dict) else n for n in self.current_news]

# ═══════════════════════════════════════════
# 5. EnvironmentManager — 4종 통합 관리
# ═══════════════════════════════════════════

class EnvironmentManager:
    """4종 환경 에이전트 오케스트레이션.

    advance_week() 순서: Policy → District → Population → News
    """

    def __init__(self, district_profiles, scenario=None):
        self.district_agent = DistrictAgent(district_profiles)
        self.population_agent = PopulationAgent()
        self.policy_agent = PolicyAgent(scenario)
        self.news_agent = NewsAgent(scenario)

        # 시나리오에서 pop_change 이벤트 추출
        self._pop_changes = {}
        if scenario:
            for evt in scenario:
                if evt.get("type") == "pop_change":
                    self._pop_changes[evt.get("week", 0)] = evt.get("value", "0%")

    def advance_week(self, week, rng, graph_mgr=None, use_llm=True):
        """순서: Policy → District → Population → News."""
        # Pop change 시나리오 적용
        if week in self._pop_changes:
            self.population_agent.set_global_change(self._pop_changes[week])

        # 1. 정책 → 효과 전파
        self.policy_agent.update_round(
            week, self.district_agent, self.population_agent
        )
        # 2. 상권 상태 업데이트
        self.district_agent.update_round(week, rng)
        # 3. 유동인구 업데이트
        self.population_agent.update_round(week, rng)
        # 4. 뉴스: LLM + GraphRAG 커뮤니티 기반 추론
        self.news_agent.update_round(
            week, self.district_agent.week_events,
            district_agent=self.district_agent,
            population_agent=self.population_agent,
            policy_agent=self.policy_agent,
            rng=rng,
            graph_mgr=graph_mgr,
            use_llm=use_llm,
        )

        # 로그 출력
        for evt in self.policy_agent.week_events:
            print(f"  {evt['msg']}")
        for evt in self.district_agent.week_events[:5]:
            print(f"  [District] {evt['msg']}")
        for news in self.news_agent.get_news():
            headline = news["headline"] if isinstance(news, dict) else news
            category = news.get("category", "?") if isinstance(news, dict) else "?"
            hotspot = ""
            if isinstance(news, dict) and news.get("hotspot"):
                hotspot = f" @ {news['hotspot']['name']}"
            print(f"  [News:{category}] {headline}{hotspot}")

    def get_context(self, agent):
        """에이전트별 맞춤 환경 컨텍스트 반환.

        Returns dict for LLM prompt / rule_engine.
        구조화된 이벤트와 핫스팟 정보를 포함합니다.
        """
        dong = str(agent.get("adm_cd", ""))[:8]
        day_type = "weekday"  # 주간 컨텍스트이므로 기본값

        industry_ctx = self.district_agent.get_industry_context(dong)
        pop_factor = self.population_agent.get_population_factor(dong, day_type)

        # 구조화된 뉴스 이벤트
        structured_events = self.news_agent.get_news()
        # 하위 호환용 문자열 리스트
        events_text = self.news_agent.get_news_headlines()

        return {
            "district_type": industry_ctx["district_type"],
            "industry_changes": industry_ctx["industry_changes"],
            "population_factor": pop_factor,
            "floating_pop_change": self.population_agent.get_pop_change_str(),
            "active_policy": self.policy_agent.get_active_policy_str(),
            "events": events_text,    # 하위 호환 (문자열 리스트)
            "structured_events": structured_events,  # 신규: 구조화된 이벤트
        }

    def get_state_for_llm(self, agent):
        """LLM 프롬프트에 전달할 환경 상태 (기존 호환)."""
        ctx = self.get_context(agent)
        return {
            "district_type": ctx["district_type"],
            "floating_pop_change": ctx["floating_pop_change"],
            "active_policy": ctx["active_policy"],
            "events": ctx["events"],
        }

    def get_week_summary(self):
        """주간 환경 요약 (리포트용)."""
        return {
            "policy_events": self.policy_agent.week_events,
            "district_events": self.district_agent.week_events,
            "news": self.news_agent.get_news_headlines(),
            "news_structured": self.news_agent.get_news(),
            "active_policy": self.policy_agent.get_active_policy_str(),
            "pop_change": self.population_agent.get_pop_change_str(),
        }


# ═══════════════════════════════════════════
# 테스트
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import pandas as pd

    # 테스트용 district_profiles
    test_data = pd.DataFrame([
        {"adm_cd": "1114055000", "district_type": "LL"},
        {"adm_cd": "1117010000", "district_type": "LH"},
        {"adm_cd": "1121560000", "district_type": "HH"},
    ])

    test_scenario = [
        {"week": 1, "type": "policy", "description": "소상공인 임대료 30% 지원", "duration": 12},
        {"week": 2, "type": "news", "category": "SNS_VIRAL",
         "description": "성수동 '맛집' 틱톡 챌린지 확산",
         "target_dong": "11200", "affected_industries": ["카페", "베이커리"]},
        {"week": 4, "type": "news", "category": "ENTERTAINMENT",
         "description": "잠실 BTS 콘서트 3일간, 팬 15만 명",
         "target_dong": "11710"},
        {"week": 6, "type": "pop_change", "value": "+15%"},
    ]

    rng = np.random.default_rng(42)
    env = EnvironmentManager(test_data, test_scenario)

    for w in range(8):
        print(f"\n{'='*50}")
        print(f"Week {w}")
        print(f"{'='*50}")
        env.advance_week(w, rng)

        # 에이전트 컨텍스트 테스트
        test_agent = {"adm_cd": "1114055000", "segment": "commuter",
                      "age_group": "30_39세", "sns_activity": 0.6}
        ctx = env.get_context(test_agent)
        print(f"  Context: policy={ctx['active_policy']}, "
              f"pop={ctx['floating_pop_change']}, "
              f"type={ctx['district_type']}, "
              f"changes={len(ctx['industry_changes'])}")
        for se in ctx.get("structured_events", []):
            print(f"    Event: [{se['category']}] {se['headline'][:50]}"
                  f" | boost={se['spending_boost']}"
                  f" | hotspot={se.get('hotspot', {}).get('name', 'N/A') if se.get('hotspot') else 'N/A'}")

    print("\n[OK] EnvironmentManager test complete")

