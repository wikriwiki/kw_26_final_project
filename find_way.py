import requests

# 교통 상태 코드 → 한글
TRAFFIC_STATE = {
    0: "정보없음",
    1: "원활",
    2: "서행",
    3: "지체",
    4: "정체",
    6: "통행불가"
}

def format_distance(meters):
    """거리를 보기 좋게 변환"""
    if meters >= 1000:
        return f"{meters / 1000:.1f}km"
    return f"{meters}m"

def format_duration(seconds):
    """소요시간을 보기 좋게 변환"""
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}시간 {m}분"
    elif seconds >= 60:
        return f"{seconds // 60}분"
    return f"{seconds}초"

def get_kakao_directions(origin, destination):
    # 1. REST API 키 설정
    rest_api_key = "2549a68999b11b10b32aedafd368797a"
    url = "https://apis-navi.kakaomobility.com/v1/directions"

    # 2. 파라미터 설정 (출발지, 목적지 좌표)
    # 형식: "경도,위도"
    params = {
        "origin": origin,
        "destination": destination,
        "priority": "RECOMMEND",  # 추천 경로
        "car_fuel": "GASOLINE",
        "car_hipass": "false"
    }

    headers = {
        "Authorization": f"KakaoAK {rest_api_key}",
        "Content-Type": "application/json"
    }

    # 3. 요청 및 응답
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        route = data['routes'][0]

        # result_code 확인 (0이면 정상)
        result_code = route.get('result_code', -1)
        if result_code != 0:
            result_msg = route.get('result_msg', '알 수 없는 오류')
            print(f"⚠️  경로 탐색 주의 (code: {result_code}): {result_msg}")

            if 'summary' not in route:
                print("📍 DISTANCE 우선순위로 재시도합니다...")
                params["priority"] = "DISTANCE"
                response = requests.get(url, params=params, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    route = data['routes'][0]
                    result_code = route.get('result_code', -1)
                    if result_code != 0 and 'summary' not in route:
                        print(f"❌ 재시도도 실패 (code: {result_code}): {route.get('result_msg', '')}")
                        return None

        # summary 존재 여부 최종 확인
        if 'summary' not in route:
            print("❌ 경로 정보(summary)를 가져올 수 없습니다.")
            return None

        summary = route['summary']
        distance = summary['distance']
        duration = summary['duration']
        fare = summary.get('fare', {})
        taxi_fare = fare.get('taxi', 0)
        toll_fare = fare.get('toll', 0)

        # ===== 전체 요약 출력 =====
        print("=" * 50)
        print("🚗 카카오 내비게이션 경로 안내")
        print("=" * 50)
        print(f"📍 출발: {origin}")
        print(f"📍 도착: {destination}")
        print(f"📏 총 거리: {format_distance(distance)}")
        print(f"⏱️  예상 소요 시간: {format_duration(duration)}")
        print(f"💰 예상 택시비: {taxi_fare:,}원")
        if toll_fare > 0:
            print(f"🛣️  통행료: {toll_fare:,}원")
        else:
            print(f"🛣️  통행료: 없음")
        print("=" * 50)

        # ===== 상세 도로 정보 출력 =====
        sections = route.get('sections', [])
        if sections:
            roads = sections[0].get('roads', [])
            guides = sections[0].get('guides', [])

            # 경유 도로 정보
            print("\n📋 경유 도로 정보:")
            print("-" * 50)
            for i, road in enumerate(roads, 1):
                road_name = road.get('name', '') or '(이름 없는 도로)'
                road_dist = road.get('distance', 0)
                road_dur = road.get('duration', 0)
                speed = road.get('traffic_speed', 0)
                state_code = road.get('traffic_state', 0)
                state = TRAFFIC_STATE.get(state_code, "정보없음")

                print(f"  {i:2d}. {road_name}")
                print(f"      거리: {format_distance(road_dist)} | "
                      f"소요: {format_duration(road_dur)} | "
                      f"속도: {speed:.0f}km/h | "
                      f"교통: {state}")

            # 턴바이턴 안내
            print(f"\n🧭 턴바이턴 안내 ({len(guides)}단계):")
            print("-" * 50)
            for i, guide in enumerate(guides, 1):
                guidance = guide.get('guidance', '')
                name = guide.get('name', '')
                dist = guide.get('distance', 0)
                dur = guide.get('duration', 0)

                label = f"{name} - {guidance}" if name else guidance
                if dist > 0:
                    print(f"  {i:2d}. {label}")
                    print(f"      → {format_distance(dist)} 이동 ({format_duration(dur)})")
                else:
                    print(f"  {i:2d}. {label}")

        print("\n" + "=" * 50)
        print("✅ 경로 안내 완료")
        print("=" * 50)

        return data
    else:
        print(f"❌ HTTP Error: {response.status_code}")
        print(response.text)
        return None

# 실행 예시 (서울역 -> 강남역)
get_kakao_directions('126.9706,37.5546', '127.0276,37.4979')