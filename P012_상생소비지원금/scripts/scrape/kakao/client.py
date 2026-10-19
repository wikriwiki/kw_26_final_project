"""Kakao 무인증 크롤 클라이언트.

엔드포인트 (전부 비공식, no auth):
  - 검색: search.map.kakao.com/mapsearch/map.daum?q={query}  → place[].confirmid
  - 디테일: place-api.map.kakao.com/places/panel3/{confirmid}  → 별점·리뷰·메뉴 등
"""
from __future__ import annotations

from typing import Optional

from curl_cffi import requests
from rapidfuzz import fuzz

from .pacing import DailyQuota, Pacer
from .session import PANEL3_BASE, UA, make_session, panel3_headers, warmup

SEARCH_URL = "https://search.map.kakao.com/mapsearch/map.daum"


import math
import re


def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


_PAREN = re.compile(r"\s*\([^)]*\)\s*")
_TAIL_BRANCH = re.compile(r"\s*(직영점|본점|지점|매장)$")


def _strip_branch_tail(name: str) -> str:
    """'스타벅스 강남R점' → '스타벅스 강남R' (마지막 한 글자 '점' 제거)."""
    n = _PAREN.sub(" ", name).strip()
    n = _TAIL_BRANCH.sub("", n).strip()
    if n.endswith("점") and len(n) > 2:
        n = n[:-1].rstrip()
    return n


def _brand_only(name: str) -> str:
    """첫 토큰만 — '맘스터치 청담점' → '맘스터치'."""
    n = _PAREN.sub(" ", name).strip()
    return n.split()[0] if n else n


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


class KakaoClient:
    def __init__(self, *, mean_pace: float = 0.7,
                 daily_limit: int = 30_000):
        self.session = make_session(restore=True)
        self._warmed = False
        self.pacer = Pacer(mean_interval=mean_pace)
        self.quota = DailyQuota(limit_per_day=daily_limit)

    def _ensure_warm(self):
        if not self._warmed:
            self._warmed = warmup(self.session)

    # ---- 검색 (무인증) ----
    def search(self, *, query: str, lat: float = None, lng: float = None,
               radius: int = 500) -> list[dict]:
        """키워드 검색. place[] 반환. confirmid = kakao place_id."""
        self._ensure_warm()
        params = {"q": query}
        if lat is not None and lng is not None:
            params.update({"lat": lat, "lng": lng, "radius": radius,
                          "msFlag": "A"})
        try:
            r = self.session.get(
                SEARCH_URL, params=params,
                headers={"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9",
                         "Referer": "https://map.kakao.com/"},
                timeout=10,
            )
        except Exception as e:
            print(f"[search] EXC: {type(e).__name__} {e}")
            self.pacer.report(0)
            return []
        self.pacer.report(r.status_code)
        self.quota.hit()
        if r.status_code != 200:
            return []
        try:
            return (r.json() or {}).get("place", []) or []
        except Exception:
            return []

    def best_match(self, *, poi_name: str, lat: float = None, lng: float = None,
                   name_threshold: int = 55) -> Optional[dict]:
        """Cascade 매칭. 100% 매칭률 목표:
          Pass 1: 원본 이름 + 좌표 bias 500m (sim>=55 또는 거리<=100m)
          Pass 2: 브랜치 suffix 제거 + 좌표 bias 1km
          Pass 3: 브랜드(첫 토큰)만 + 좌표 bias 200m → 거리 최소 채택
          Pass 4: 브랜드만 + 좌표 bias 1km → 거리<=200m && sim>=40 채택
          Pass 5: 좌표 없이 원본 이름 — sim>=70만 채택
        """
        def _score(cands: list, target: str, max_dist: float = None,
                   min_sim: int = name_threshold) -> Optional[dict]:
            if not cands:
                return None
            t = _norm(target)
            scored = []
            for c in cands:
                sim = fuzz.token_sort_ratio(t, _norm(c.get("name", "")))
                dist = None
                if lat is not None and lng is not None:
                    try:
                        dist = _haversine_m(lat, lng,
                                            float(c.get("lat", 0)),
                                            float(c.get("lon", 0)))
                    except (ValueError, TypeError):
                        pass
                scored.append((sim, dist, c))
            # 거리 후보가 있으면 거리 가까운 쪽 + 유사도 보너스
            scored.sort(key=lambda x: (-x[0], x[1] or 999999))
            for sim, dist, c in scored:
                if max_dist is not None and dist is not None and dist > max_dist:
                    continue
                if sim >= min_sim:
                    return c
                if dist is not None and dist <= 50:  # 거리 50m 이내면 이름 미달도 채택
                    return c
            return None

        # Pass 1: 원본 + 좌표 500m
        cands = self.search(query=poi_name, lat=lat, lng=lng, radius=500)
        m = _score(cands, poi_name, max_dist=500, min_sim=name_threshold)
        if m:
            return m

        # Pass 2: 가지치기 이름 + 좌표 1km
        stripped = _strip_branch_tail(poi_name)
        if stripped and stripped != poi_name:
            cands = self.search(query=stripped, lat=lat, lng=lng, radius=1000)
            m = _score(cands, stripped, max_dist=1000, min_sim=name_threshold)
            if m:
                return m

        # Pass 3: 브랜드만 + 좌표 200m (가장 가까운 거 채택)
        brand = _brand_only(poi_name)
        if brand and brand != poi_name:
            cands = self.search(query=brand, lat=lat, lng=lng, radius=200)
            m = _score(cands, brand, max_dist=200, min_sim=30)
            if m:
                return m

        # Pass 4: 브랜드 + 1km, sim>=40 + 거리<=200m
        if brand and lat is not None:
            cands = self.search(query=brand, lat=lat, lng=lng, radius=1000)
            m = _score(cands, brand, max_dist=200, min_sim=40)
            if m:
                return m

        # Pass 5: 좌표 없이 원본 — sim>=70만
        if lat is None or not cands:
            cands = self.search(query=poi_name)
            m = _score(cands, poi_name, max_dist=None, min_sim=70)
            if m:
                return m

        return None

    # ---- 디테일 (무인증) ----
    def panel3(self, confirmid: str) -> Optional[dict]:
        """패널3. 별점·리뷰·메뉴·방문자통계·영업시간 풀데이터."""
        self._ensure_warm()
        url = f"{PANEL3_BASE}/{confirmid}"
        try:
            r = self.session.get(url, headers=panel3_headers(confirmid),
                                 timeout=15)
        except Exception as e:
            print(f"[panel3] {confirmid} EXC: {type(e).__name__} {e}")
            self.pacer.report(0)
            return None
        self.pacer.report(r.status_code)
        self.quota.hit()
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except Exception:
            return None
