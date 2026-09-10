"""Kakao 스크레이퍼 세션. curl_cffi chrome120 임퍼소네이션 + 쿠키 영속화."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from curl_cffi import requests

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"

PANEL3_BASE = "https://place-api.map.kakao.com/places/panel3"
REVIEWS_TAB = "https://place-api.map.kakao.com/places/tab/reviews/kakaomap"
WARMUP_URL = "https://place.map.kakao.com/m/{pid}"

SESSION_DIR = Path("C:/Users/Administrator/naver_crawl/session")
SESSION_DIR.mkdir(parents=True, exist_ok=True)
COOKIE_PATH = SESSION_DIR / "kakao_cookies.pkl"


def make_session(*, restore: bool = True) -> requests.Session:
    """워밍업된 chrome120 세션. 쿠키 복원/재사용."""
    s = requests.Session(impersonate="chrome120")
    if restore and COOKIE_PATH.exists():
        try:
            with COOKIE_PATH.open("rb") as f:
                jar = pickle.load(f)
            for k, v in jar.items():
                s.cookies.set(k, v)
        except Exception:
            pass
    return s


def save_cookies(session: requests.Session) -> None:
    try:
        jar = dict(session.cookies)
        with COOKIE_PATH.open("wb") as f:
            pickle.dump(jar, f)
    except Exception:
        pass


def warmup(session: requests.Session, place_id: str = "12108448") -> bool:
    """SPA 페이지 GET → JSESSIONID 받기. 5초 timeout, 1회만 시도."""
    try:
        r = session.get(
            WARMUP_URL.format(pid=place_id),
            headers={"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9"},
            timeout=10,
        )
        ok = r.status_code == 200 and len(session.cookies) > 0
        if ok:
            save_cookies(session)
        return ok
    except Exception:
        return False


def panel3_headers(place_id: str) -> dict:
    """panel3 호출용 표준 헤더."""
    return {
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
        "Origin": "https://place.map.kakao.com",
        "Referer": f"https://place.map.kakao.com/{place_id}",
        "PF": "web",
        "Sec-Fetch-Site": "same-site",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
    }
