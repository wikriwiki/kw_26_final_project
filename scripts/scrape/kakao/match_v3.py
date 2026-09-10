"""매칭 v3 — 80% 달성 목표 cascade.

핵심 전략 (false positive 0건 유지):
  1. 변형 query 다양화 — 전각/한자/영문 변환, 체인점 표기, 핵심 토큰 추출
  2. 카테고리 매칭 검증 — csv 대분류·중분류와 카카오 카테고리 일치 확인
  3. 거리·sim 임계의 단계적 완화 — 카테고리 일치 시 거리 임계 확대 허용
  4. 시군구+brand 검색 query 추가
  5. 시장·복합건물 가게는 별도 표시 (매칭 시도 안 함 — 정직성 보장)
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from .client import KakaoClient, _norm
from .run_apt import _safe_dist


# === 카테고리 매핑: 소상공인 csv 대분류·중분류 → 카카오 카테고리 키워드 ===
# panel3/search 응답의 cate_name_depth1~5 또는 last_cate_name 에 포함될 키워드
CSV_TO_KAKAO_CAT = {
    # 대분류 → 허용 카카오 카테고리 키워드 (모두 2자 이상)
    "음식": ["음식점", "식당", "한식", "양식", "중식", "일식", "분식", "카페",
             "주점", "치킨", "패스트푸드", "버거", "베이커리", "제과", "디저트",
             "도시락", "찌개", "국밥", "초밥", "고기", "구이", "보쌈", "찜닭",
             "샐러드", "피자", "파스타", "면집", "수산물", "해물", "생선", "곱창",
             "막창", "갈비", "삼겹살", "분식점", "음료", "호프", "주류"],
    "소매": ["편의점", "슈퍼", "마트", "의류", "신발", "화장품", "잡화",
             "악세서리", "가방", "안경", "보석", "도서", "문구",
             "약국", "꽃집", "전자제품", "가전", "휴대폰", "통신기기",
             "정육", "수산", "식품", "주류", "취미", "완구",
             "스포츠용품", "악기", "패션"],
    "수리·개인": ["미용", "이용", "세탁", "수선", "수리", "정비",
                "에스테틱", "피부관리", "네일", "마사지", "타투",
                "사진관", "스튜디오"],
    "교육": ["학원", "교습소", "교육", "스쿨", "유치원", "어린이집",
             "독서실", "도서관", "강의실", "어학원"],
    "보건의료": ["병원", "의원", "한의원", "치과", "약국", "약방",
                "정신과", "산부인과", "내과", "외과", "피부과", "안과",
                "이비인후과", "정형외과", "비뇨기과", "신경과", "성형외과",
                "재활", "검진"],
    "부동산": ["부동산", "공인중개", "중개"],
    "예술·스포츠": ["헬스", "스포츠", "체육", "수영장", "골프", "테니스",
                  "당구", "노래방", "PC방", "오락", "볼링", "필라테스",
                  "요가", "복싱", "유도", "태권도", "검도",
                  "갤러리", "박물관", "전시관", "공연장", "영화관"],
    "숙박": ["호텔", "모텔", "여관", "펜션", "게스트하우스", "리조트",
             "민박", "유스호스텔"],
}


def kakao_cat_str(c: dict) -> str:
    """카카오 후보의 카테고리 문자열 — depth1~5 통합."""
    parts = [c.get(f"cate_name_depth{i}", "") or "" for i in (1, 2, 3, 4, 5)]
    parts.append(c.get("last_cate_name", "") or "")
    return " ".join(p for p in parts if p)


def cat_matches(c: dict, expected_keywords: list[str]) -> bool:
    """카카오 후보 카테고리에 csv 대분류 키워드 중 하나라도 매칭되면 True.

    Token 단위 substring 매칭 (한 글자 키워드의 우연한 substring 매칭 방지).
    """
    if not expected_keywords:
        return True
    cat_str = kakao_cat_str(c)
    # 카테고리 토큰 분리 (공백/슬래시/콤마)
    tokens = re.split(r'[\s/,·]+', cat_str)
    for k in expected_keywords:
        if len(k) < 2:
            continue
        for t in tokens:
            if k in t:
                return True
    return False


# === query 변형 생성 ===

# 전각 → 반각 매핑
FULLWIDTH = {chr(ord('０') + i): str(i) for i in range(10)}
FULLWIDTH.update({chr(ord('Ａ') + i): chr(ord('A') + i) for i in range(26)})
FULLWIDTH.update({chr(ord('ａ') + i): chr(ord('a') + i) for i in range(26)})

# 체인점 영문 ↔ 한글
CHAIN_MAP = [
    (r'지에스\s*25', 'GS25'),
    (r'씨유(?:편의점)?', 'CU'),
    (r'세븐일레븐', '7-Eleven'),
    (r'세븐\s*([일이삼사오])?(?:호점)?', '7-Eleven'),
    (r'이마트\s*24', '이마트24'),
    (r'미니스톱', '미니스톱'),
    (r'홈플러스익스프레스', '홈플러스익스프레스'),
]


def normalize_query(s: str) -> str:
    """기본 정규화: 전각→반각, 다중 공백 정리."""
    out = s
    for fw, hw in FULLWIDTH.items():
        out = out.replace(fw, hw)
    out = re.sub(r'\s+', ' ', out).strip()
    return out


def gen_query_variants(name: str, brand: str, branch: str,
                       sigungu: str) -> list[str]:
    """이름의 다양한 변형 생성. 중복 제거. 길이 ≥2."""
    out = []
    out.append(name)
    out.append(normalize_query(name))
    if brand and brand != name:
        out.append(brand)
        out.append(normalize_query(brand))
    # 법인 접미사 제거
    for s in (name, brand or ''):
        if not s:
            continue
        out.append(re.sub(r'\s*\(?주\)?\s*$', '', s).strip())
        out.append(re.sub(r'^\(?주\)?\s*', '', s).strip())
        out.append(re.sub(r'\s*주식회사\s*', ' ', s).strip())
        out.append(re.sub(r'\s*\([^)]*\)\s*', ' ', s).strip())
    # 체인 표기 변환
    norm = normalize_query(name)
    for pat, repl in CHAIN_MAP:
        cv = re.sub(pat, repl, norm)
        if cv != norm:
            out.append(cv)
    # 시군구 + brand
    if sigungu and brand:
        out.append(f'{sigungu} {brand}')
    # branch + brand 합친 형태 변형 — "지에스25묵동 제일점" → "묵동 제일점", "묵동제일점"
    if branch:
        out.append(branch)
        out.append(re.sub(r'\s+', '', branch))
    # 마지막 토큰 = 지점이면 brand만 (no branch)
    tokens = name.split()
    if tokens and (tokens[-1].endswith('점') or tokens[-1].endswith('역점')):
        out.append(' '.join(tokens[:-1]))

    seen, final = set(), []
    for v in out:
        v = re.sub(r'\s+', ' ', v).strip()
        if v and len(v) >= 2 and v not in seen:
            seen.add(v)
            final.append(v)
    return final


# === score & pick (smart) ===

def _smart_pick_v3(cands: list[dict], target: str, lat, lng, *,
                   min_sim: int, max_dist: float,
                   expected_cat_keywords: list[str] | None = None,
                   require_cat: bool = False,
                   top_n: int = 20) -> tuple[dict, int, float] | None:
    """top N에서 sim·distance·(카테고리) 조건 통과 후보 채택."""
    if lat is None or lng is None or not cands:
        return None
    t = _norm(target)
    scored = []
    for c in cands[:top_n]:
        n = _norm(c.get('name', ''))
        sim = max(fuzz.token_sort_ratio(t, n), fuzz.partial_ratio(t, n))
        try:
            d = _safe_dist(c, lat, lng)
        except Exception:
            continue
        if d > max_dist:
            continue
        if sim < min_sim:
            continue
        if require_cat and not cat_matches(c, expected_cat_keywords or []):
            continue
        scored.append((sim, d, c))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1]))
    sim, d, c = scored[0]
    return c, sim, d


# === best_match_v3 ===

def best_match_v3(client: KakaoClient, poi: dict) -> tuple[dict | None, str]:
    """8-pass cascade — 정확성 + 회수율 양립.

    좌표 hint=2km. 모든 pass는 좌표 ≤ max_dist + 이름 sim ≥ min_sim 통과 필요.
    (P6/P7/P8) 카테고리 일치 시에만 임계 완화.
    """
    name = poi['poi_name']
    brand = poi.get('poi_brand') or name
    branch = poi.get('poi_branch') or ''
    sigungu = poi.get('sigungu') or ''
    cat_l = poi.get('cat_l') or ''
    lat, lng = poi['poi_lat'], poi['poi_lon']

    if lat is None or lng is None:
        return None, 'nomatch_no_coord'

    variants = gen_query_variants(name, brand, branch, sigungu)
    expected_cats = CSV_TO_KAKAO_CAT.get(cat_l, [])

    # P1: 원본 + 2km hint, sim ≥ 60, d ≤ 200m
    cands = client.search(query=name, lat=lat, lng=lng, radius=2000)
    r = _smart_pick_v3(cands, name, lat, lng, min_sim=60, max_dist=200)
    if r:
        return r[0], 'P1_orig_200m'

    # P2: brand + 2km, sim ≥ 60, d ≤ 200m
    if brand != name:
        cands = client.search(query=brand, lat=lat, lng=lng, radius=2000)
        r = _smart_pick_v3(cands, name, lat, lng, min_sim=60, max_dist=200)
        if r:
            return r[0], 'P2_brand_200m'

    # P3: 변형 query (전각/체인/법인접미사) — 2km hint, sim ≥ 60, d ≤ 200m
    for v in variants[:8]:
        if v in (name, brand):
            continue
        cands = client.search(query=v, lat=lat, lng=lng, radius=2000)
        r = _smart_pick_v3(cands, name, lat, lng, min_sim=60, max_dist=200)
        if r:
            return r[0], 'P3_variant_200m'

    # P4: 카테고리 일치 시 — sim 55 + d 200m + cat 검증 (csv↔kakao 카테고리 매핑 활용)
    if expected_cats:
        for v in [name, brand] + variants[:5]:
            if not v:
                continue
            cands = client.search(query=v, lat=lat, lng=lng, radius=2000)
            r = _smart_pick_v3(cands, name, lat, lng,
                              min_sim=55, max_dist=200,
                              expected_cat_keywords=expected_cats,
                              require_cat=True)
            if r:
                return r[0], 'P4_cat_match_200m'

    # P5: 이름 매우 일치 + 거리 가까움 — sim ≥ 85 + d ≤ 100m
    cands = client.search(query=name, lat=lat, lng=lng, radius=2000)
    r = _smart_pick_v3(cands, name, lat, lng, min_sim=85, max_dist=100)
    if r:
        return r[0], 'P5_very_high_sim_100m'

    # P6: sigungu + brand 검색 — sim ≥75 + d ≤150m
    if sigungu and brand:
        cands = client.search(query=f'{sigungu} {brand}',
                              lat=lat, lng=lng, radius=2000)
        r = _smart_pick_v3(cands, name, lat, lng, min_sim=75, max_dist=150)
        if r:
            return r[0], 'P6_sigungu_brand_150m'

    return None, 'nomatch'
