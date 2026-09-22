"""panel3 응답 → 평탄화된 dict 추출. 누락 필드는 None."""
from __future__ import annotations

from typing import Any


def _g(d: dict, *path, default=None):
    """안전한 nested get."""
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
        if cur is None:
            return default
    return cur


def extract_summary(panel: dict) -> dict:
    """별점·리뷰수·메뉴 등 핵심만 평탄화."""
    s = panel.get("summary", {}) or {}
    km = panel.get("kakaomap_review", {}) or {}
    score_set = km.get("score_set") or {}
    menu = panel.get("menu", {}) or {}
    visitor = panel.get("visitor", {}) or {}
    open_h = panel.get("open_hours", {}) or {}
    ai = panel.get("ai_mate", {}) or {}

    return {
        "place_id": s.get("confirm_id") or _g(s, "meta", "confirm_id"),
        "name": s.get("name"),
        "category": _g(s, "category", "name"),
        "category_path": [_g(s, "category", f"name{i}") for i in (1, 2, 3, 4)],
        "address_road": _g(s, "address", "road"),
        "address_old": _g(s, "address", "jibun") or _g(s, "address", "disp"),
        "address_region": [r.get("name") for r in (s.get("regions") or [])],
        "lat": _g(s, "point", "lat"),
        "lon": _g(s, "point", "lon"),
        # 별점·리뷰 (스키마: kakaomap_review.score_set.{average_score,review_count,total_score})
        "rating": _g(km, "score_set", "average_score"),
        "rating_count": _g(km, "score_set", "review_count"),
        "rating_total_score": _g(km, "score_set", "total_score"),
        "rating_strength_counts": _g(km, "score_set", "strength_counts"),
        "review_count_blog": _g(panel, "blog_review", "review_count"),
        "photo_count": _g(km, "score_set", "photo_count"),
        # 메뉴
        "menus": menu.get("menus") or [],
        "menu_count": len(menu.get("menus") or []),
        # 방문자 통계 (요일별 평균)
        "visitor_weekly_avg": visitor.get("weekly_uv_average"),
        "visitor_by_day": {
            d: visitor.get(f"{d}_uv")
            for d in ("monday", "tuesday", "wednesday", "thursday", "friday",
                      "saturday", "sunday")
        },
        # 영업
        "open_status": open_h.get("headline"),
        "week_hours": open_h.get("week_from_today"),
        # AI
        "ai_summary": ai.get("summary"),
        "ai_price_level": ai.get("price_level"),
        # tags
        "panel_card_tags": panel.get("panel_card_tags") or [],
        "panel_tab_tags": panel.get("panel_tab_tags") or [],
    }


def extract_reviews(panel: dict, *, limit: int = 5) -> list[dict]:
    """첫 페이지 리뷰 텍스트 추출. 더 많이는 페이지네이션 호출 필요."""
    km = panel.get("kakaomap_review", {}) or {}
    out = []
    for rv in (km.get("reviews") or [])[:limit]:
        out.append({
            "review_id": rv.get("review_id") or rv.get("id"),
            "star_rating": rv.get("star_rating"),
            "contents": rv.get("contents") or rv.get("content"),
            "date": rv.get("date") or rv.get("registered_at"),
            "user_nickname": _g(rv, "meta", "owner", "name") or _g(rv, "user", "nickname"),
            "thumb_up": rv.get("thumb_up_count") or rv.get("thumb_up"),
        })
    return out
