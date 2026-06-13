import requests
import json

# api키가 맞지 않음!

def get_tmap_transit_route():
    # 1. 설정 정보
    url = "https://apis.openapi.sk.com/transit/routes"
    app_key = "zstcw4o3NE4rkB7fU0QQh9Gq5gBfpvLAesfwVNVg"
    
    headers = {
        "accept": "application/json",
        "appKey": app_key,
        "content-type": "application/json"
    }

    # 2. 요청 파라미터 (출발지: 서울역 / 목적지: 롯데월드)
    payload = {
        "startX": "126.9723",
        "startY": "37.5559",
        "endX": "127.1009",
        "endY": "37.5113",
        "lang": 0,         # 0: 한국어
        "format": "json",
        "count": 3         # 추천 경로 3개까지
    }

    try:
        # 3. API 호출
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status() # 에러 발생 시 예외 처리
        
        data = response.json()
        
        # 4. 결과 출력
        if "metaData" in data and "plan" in data["metaData"]:
            itineraries = data["metaData"]["plan"]["itineraries"]
            
            print(f"--- 총 {len(itineraries)}개의 경로를 찾았습니다 ---\n")
            
            for i, route in enumerate(itineraries):
                total_time = route['totalTime'] // 60  # 초 -> 분
                total_fare = route['fare']['regular']['totalFare']
                transfer_count = route['transferCount']
                
                print(f"[{i+1}번 추천 경로]")
                print(f"- 총 소요시간: {total_time}분")
                print(f"- 총 요금: {total_fare}원")
                print(f"- 환승 횟수: {transfer_count}회")
                
                # 구간별 상세 정보 (선택 사항)
                for leg in route['legs']:
                    mode = leg['mode'] # WALKING, BUS, SUBWAY 등
                    section_time = leg['sectionTime'] // 60
                    name = leg.get('route', leg.get('start', {}).get('name', '이동'))
                    print(f"  └ {mode}: {name} ({section_time}분)")
                print("-" * 30)
        else:
            print("경로를 찾을 수 없습니다.")

    except requests.exceptions.RequestException as e:
        print(f"API 호출 중 오류 발생: {e}")

if __name__ == "__main__":
    get_tmap_transit_route()