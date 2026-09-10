"""Kakao Map POI 스크레이퍼 (별점·리뷰·메뉴).

엔드포인트 (리버스 엔지니어링 결과):
  - 매칭: https://dapi.kakao.com/v2/local/search/keyword.json (공식, REST API key 필요)
  - 디테일: https://place-api.map.kakao.com/places/panel3/{place_id} (비공식, no auth)
  - 리뷰페이지: https://place-api.map.kakao.com/places/tab/reviews/kakaomap (비공식)

박사급 anti-bot:
  - curl_cffi chrome120 TLS 지문 위장
  - 세션 워밍업 (SPA 페이지 GET → JSESSIONID 획득)
  - Poisson 분포 페이싱 (uniform 금지)
  - 4xx/5xx 지수 백오프 + jitter
  - 일일 쿼터 추적
"""
