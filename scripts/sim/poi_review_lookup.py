"""카카오 리뷰·별점 lookup — Stage 2에서 LLM이 선택적으로 요청 시 사용.

소상공인 csv POI id (Neo4j: `C_<상가업소번호>`)
→ 카카오 매칭 DB poi_id (`COM_<상가업소번호>`) 변환 후
   panel3_raw JSON에서 별점·리뷰만 추출.

가게가 매칭 안 됐거나 별점 없으면 None.
"""
from __future__ import annotations

import functools
import json
import sqlite3
from pathlib import Path

DB_PATH = Path("C:/Users/Administrator/naver_crawl/sqlite/kakao_enrich.db")


@functools.lru_cache(maxsize=1)
def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def neo4j_poi_id_to_com_id(neo4j_id: str) -> str | None:
    """C_MA01... → COM_MA01... 변환. C_ prefix 아니면 None."""
    if not neo4j_id:
        return None
    if neo4j_id.startswith("C_"):
        return "COM_" + neo4j_id[2:]
    return None


def lookup_review(neo4j_poi_id: str, *, max_reviews: int = 3) -> dict | None:
    """단일 POI 별점·리뷰 조회. 매칭 없거나 panel3 없으면 None.

    반환 필드:
      rating: 평균 별점 (None 가능)
      rating_count: 리뷰 수
      reviews: [{star, contents (≤120자)}, ...] 최대 max_reviews
      category: 카카오 카테고리
    """
    com_id = neo4j_poi_id_to_com_id(neo4j_poi_id)
    if not com_id:
        return None
    row = _conn().execute(
        "SELECT s.kakao_pid, p.raw_json FROM poi_status s "
        "JOIN panel3_raw p ON s.kakao_pid = p.kakao_pid "
        "WHERE s.poi_id = ? AND s.status = 'fetched' LIMIT 1",
        (com_id,),
    ).fetchone()
    if not row:
        return None
    try:
        panel = json.loads(row["raw_json"])
    except (json.JSONDecodeError, TypeError):
        return None
    summary = panel.get("summary") or {}
    km = panel.get("kakaomap_review") or {}
    score = km.get("score_set") or {}
    reviews_raw = km.get("reviews") or []
    cat = (summary.get("category") or {}).get("name")

    rating = score.get("average_score")
    rcount = score.get("review_count") or 0
    if rating in (None, 0) and rcount == 0 and not reviews_raw:
        return None  # 데이터 없음 — None 반환

    out_reviews = []
    for rv in reviews_raw[:max_reviews]:
        contents = rv.get("contents") or rv.get("content") or ""
        out_reviews.append({
            "star": rv.get("star_rating"),
            "contents": contents[:120],
        })
    return {
        "rating": rating,
        "rating_count": rcount,
        "reviews": out_reviews,
        "category": cat,
    }


def lookup_reviews_batch(neo4j_poi_ids: list[str], *,
                         max_reviews: int = 3) -> dict[str, dict]:
    """여러 POI 일괄 조회. 매칭 안 된 POI는 결과에서 제외."""
    out = {}
    for pid in neo4j_poi_ids:
        r = lookup_review(pid, max_reviews=max_reviews)
        if r:
            out[pid] = r
    return out


def format_review_block(neo4j_poi_id: str, info: dict) -> str:
    """LLM prompt 첨부용 포맷팅."""
    parts = [f"  {neo4j_poi_id}"]
    rating = info.get("rating")
    rcount = info.get("rating_count") or 0
    if rating:
        parts.append(f"★{rating:.1f} ({rcount}리뷰)")
    cat = info.get("category")
    if cat:
        parts.append(f"[{cat}]")
    line1 = " | ".join(parts)
    review_lines = []
    for rv in info.get("reviews") or []:
        star = rv.get("star")
        contents = rv.get("contents") or ""
        if star is not None or contents:
            star_s = f"★{star}" if star is not None else ""
            review_lines.append(f"    {star_s} {contents}")
    return line1 + ("\n" + "\n".join(review_lines) if review_lines else "")
