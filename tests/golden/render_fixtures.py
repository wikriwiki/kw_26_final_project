# -*- coding: utf-8 -*-
"""프롬프트 렌더 고정 입력 (DB·GPU 불필요).

정책/환경 채널 분리 리팩터링이 **렌더링된 프롬프트 바이트를 바꾸지 않는지**
증명하기 위한 결정론적 픽스처. 값은 실제 P010 런의 형태를 본떴으나
특정 에이전트의 실데이터가 아니다.
"""
from __future__ import annotations
from datetime import date

# ── P010 정책 행 (Neo4j 조회 결과 형태) ────────────────────────────────
P010_ROW = {
    "id": "P010",
    "type": "grant",
    "name": "민생회복 소비쿠폰 1차",
    "from_": "2025-07-21",
    "until_": "2025-11-30",
    "regions": ["서울특별시"],
    "target_l1s": [],
    "poi_restricted": True,
    "grant_key": "spend_decile",
    "decile_grants": {"1": 400000, "2": 300000, "3": 150000, "4": 150000,
                      "5": 150000, "6": 150000, "7": 150000, "8": 150000,
                      "9": 150000, "10": 150000},
    "description": (
        "정부가 소상공인·자영업자 지원과 지역경제 활성화를 위해 전 국민에게 "
        "민생회복 소비쿠폰을 지급합니다. 지급액은 대상 계층별로 달라 일반 국민 15만원, "
        "차상위·한부모가족 30만원, 기초생활수급자 40만원입니다. 서울 시뮬레이션에서는 "
        "보유한 최소 계층 단위인 소비 10분위에 대응하여 소비 1분위 40만원, 2분위 30만원, "
        "3~10분위 15만원을 지급합니다. 쿠폰은 지역사랑상품권 가맹점 또는 연 매출액 "
        "30억원 이하 매장([쿠폰] 표시 매장)에서만 사용할 수 있고, 대형마트·백화점·"
        "기업형 슈퍼마켓 등에서는 사용할 수 없습니다. 사용기한은 2025년 11월 30일까지이며 "
        "미사용분은 자동 소멸됩니다."
    ),
}

PERSONAS = [
    {
        "id": "A0001", "job": "사무직", "income": "중",
        "daily_wd": 42000, "daily_we": 51000, "spend_decile": 5,
        "tendency": "실속형", "lifestyle": "출퇴근 후 동네에서 저녁을 해결하는 편",
        "work_dong": "역삼1동", "cat_ratio_wd": {"식사": 0.42, "카페": 0.18, "마트": 0.20},
    },
    {
        "id": "A0002", "job": "무직", "income": "하",
        "daily_wd": 12000, "daily_we": 15000, "spend_decile": 1,
        "tendency": "절약형", "lifestyle": "집 근처를 벗어나지 않는 생활",
        "work_dong": None, "cat_ratio_wd": {"마트": 0.5, "식사": 0.3},
    },
]

STATES = [
    {"balance": 1_404_344, "month_spent": 812_000, "policy_used": {},
     "grant_received": {"P010": 150000}, "grant_remaining": {"P010": 150000},
     "grant_days_since": {"P010": 0}, "policy_lc": '{"P010": "S3"}',
     "mood": 0.62, "energy": 0.70, "fatigue": 0.30, "yest_sat": 3.80},
    {"balance": 247_941, "month_spent": 1_650_000, "policy_used": {"P010": 62000},
     "grant_received": {"P010": 400000}, "grant_remaining": {"P010": 338000},
     "grant_days_since": {"P010": 4}, "policy_lc": '{"P010": "S4"}',
     "mood": 0.41, "energy": 0.45, "fatigue": 0.62, "yest_sat": 3.10},
]

MEMORY = [
    {"days_ago": 1, "poi_name": "동네국수", "category": "식사", "satisfaction": 4.2,
     "summary": "9200원 · [배고픔] 퇴근길에 간단히", "source": "visited"},
    {"days_ago": 3, "poi_name": "우리마트", "category": "마트", "satisfaction": 3.5,
     "summary": "31000원 · [생필품] 주말 장보기", "source": "visited"},
    {"days_ago": 2, "poi_name": "블루샵", "category": "카페", "satisfaction": 4.0,
     "summary": "소문: 쿠폰 쓸 수 있다더라", "source": "rumor"},
]

APPOINTMENT = [
    {"target_time": "19:00", "with_agents": ["A0777"], "meeting_poi_name": "삼겹이네",
     "meeting_poi_id": "P77", "meeting_location_hint": "역삼1동", "topic_type": "추천",
     "topic_value": "저녁"},
]

SOCIAL = [
    {"friend_id": "A0777", "age": 34, "gender": "M", "relation": "직장동료",
     "strength": 0.72, "lifestyle": "퇴근 후 한 잔 하는 걸 좋아함"},
    {"friend_id": "A0912", "age": 41, "gender": "F", "relation": "이웃",
     "strength": 0.45, "lifestyle": "주말마다 동네 산책"},
]

KNOWS_POI = [
    {"L1": "식사", "sub": "한식", "n": 12, "n_visited": 6},
    {"L1": "식사", "sub": "분식", "n": 5, "n_visited": 2},
    {"L1": "마트", "sub": "슈퍼마켓", "n": 7, "n_visited": 4},
    {"L1": "카페", "sub": "커피전문점", "n": 9, "n_visited": 3},
]

ZONES = [
    {"code": "1168064000", "name": "역삼1동", "type": "work", "distance_km": 0.0},
    {"code": "1156054000", "name": "신길3동", "type": "home", "distance_km": 0.0},
    {"code": "1168010100", "name": "강남역 일대", "type": "hub",
     "distance_km": 7.4, "signature": "office"},
]

DATES = [date(2025, 7, 21), date(2025, 7, 26), date(2025, 11, 29)]


def cases():
    """(케이스명, persona, state, policy_rows) 목록."""
    out = []
    for i, (p, s) in enumerate(zip(PERSONAS, STATES)):
        out.append((f"agent{i}_policy", p, s, [P010_ROW]))
        out.append((f"agent{i}_nopolicy", p,
                    {**s, "grant_received": {}, "grant_remaining": {},
                     "grant_days_since": {}, "policy_lc": "{}"}, []))
    return out
