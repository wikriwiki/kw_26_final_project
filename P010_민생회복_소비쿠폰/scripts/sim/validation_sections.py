"""4일 baseline 검증 6종 — V1~V6.

V1. 공간 분포 (자치구별 매출 비율) — 코사인 + JSD + MAPE
V2. 시간대 활동 분포 (6구간) — 코사인 + JSD + 피크 일치
V3. 업종 분포 (대분류 매핑) — 코사인 + Spearman rank
V4. 인구통계×업종 cross (연령×업종, 성별×업종) — 코사인
V5. 주중/주말 비율 — 비율 비교 + MAPE
V6. 소득-소비 탄력성 — 피어슨 ρ (시뮬 한계 정직 보고)

held-out 비교 데이터: 서울시 상권분석서비스 추정매출 2025년 1분기 (data/seoul_commerce/)
"""
import sys, csv, math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))
from _common import driver_session

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SEOUL_CSV = "data/seoul_commerce/서울시 상권분석서비스(추정매출-행정동)_2025년.csv"


# ─────────────────────────────────────────────────────
# 통계 유틸
# ─────────────────────────────────────────────────────
def _normalize(v):
    """list → 합 1로 normalize."""
    s = sum(v)
    return [x / s for x in v] if s > 0 else [0] * len(v)


def cosine(a, b):
    """코사인 유사도. 0~1."""
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def jsd(p, q):
    """Jensen-Shannon divergence. 0~ln(2)≈0.693."""
    p = _normalize(p); q = _normalize(q)
    m = [(x + y) / 2 for x, y in zip(p, q)]

    def kl(a, b):
        s = 0
        for x, y in zip(a, b):
            if x > 0 and y > 0:
                s += x * math.log(x / y)
        return s

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def mape(actual, predicted):
    """MAPE — actual 0 회피."""
    pairs = [(a, b) for a, b in zip(actual, predicted) if a > 0]
    if not pairs:
        return 0.0
    return 100 * sum(abs(a - b) / a for a, b in pairs) / len(pairs)


def spearman(a, b):
    """Spearman rank correlation."""
    n = len(a)
    if n < 2:
        return 0.0
    ra = sorted(range(n), key=lambda i: a[i])
    rb = sorted(range(n), key=lambda i: b[i])
    rank_a = [0] * n; rank_b = [0] * n
    for rank, idx in enumerate(ra): rank_a[idx] = rank
    for rank, idx in enumerate(rb): rank_b[idx] = rank
    d2 = sum((rank_a[i] - rank_b[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def gini(values):
    """Gini 계수 — 분포 불평등(집중도). 0=완전 균등, 1=완전 집중."""
    vals = [v for v in values if v > 0]
    n = len(vals)
    if n == 0:
        return 0.0
    sorted_v = sorted(vals)
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(sorted_v, 1):
        cum += i * v
    return (2 * cum) / (n * total) - (n + 1) / n


def pearson(a, b):
    n = len(a)
    if n < 2: return 0.0
    ma = sum(a) / n; mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    if da == 0 or db == 0: return 0.0
    return num / (da * db)


# ─────────────────────────────────────────────────────
# 업종 매핑 — 서울 데이터 + 시뮬 LLM 카테고리 → 대분류 10개
# ─────────────────────────────────────────────────────
DAEBUNRYU = [
    "식음료", "편의점/소매", "의류/잡화", "의료", "교육·학원",
    "문화·여가", "미용·서비스", "운송·주유", "숙박", "기타",
]


def map_seoul_industry(name: str) -> str:
    """서울 상권분석 서비스_업종_코드_명 → 대분류 10개 (보강판)."""
    n = name or ""
    # 식음료
    if any(k in n for k in ["음식점", "한식", "양식", "일식", "중식", "분식", "패스트푸드",
                             "카페", "커피", "제과", "베이커리", "주점", "호프", "치킨"]):
        return "식음료"
    # 편의점/소매 (식품 도소매 포함)
    if any(k in n for k in ["편의점", "슈퍼마켓", "수퍼마켓", "할인점", "농수산물",
                             "청과", "수산물판매", "반찬", "육류판매", "미곡", "식료품", "식품"]):
        return "편의점/소매"
    # 의류/잡화 (전자·서적·문구·스포츠용품 포함)
    if any(k in n for k in ["의류", "신발", "가방", "잡화", "가전", "가구", "스포츠용품",
                             "안경", "시계", "악세사리", "화장품", "컴퓨터", "핸드폰",
                             "조명", "문구", "서적", "운동/경기용품", "꽃", "식물"]):
        return "의류/잡화"
    # 의료
    if any(k in n for k in ["병원", "의원", "한의원", "약국", "치과", "의약품", "의료기기"]):
        return "의료"
    # 교육
    if any(k in n for k in ["학원", "교습소", "교육"]):
        return "교육·학원"
    # 문화/여가
    if any(k in n for k in ["영화", "노래방", "PC방", "당구장", "볼링", "골프", "헬스",
                             "스포츠", "여행", "여가"]):
        return "문화·여가"
    # 미용/서비스
    if any(k in n for k in ["미용실", "이용원", "네일", "피부관리", "세탁"]):
        return "미용·서비스"
    # 운송/주유
    if any(k in n for k in ["주유소", "세차", "주차", "자동차수리"]):
        return "운송·주유"
    # 숙박
    if any(k in n for k in ["숙박", "모텔", "호텔"]):
        return "숙박"
    return "기타"


def map_sim_category(cat: str, sub: str = "") -> str:
    """시뮬 LLM cat·sub_category → 대분류 10개 (보강판).

    LLM이 cat에 상위 분류명("식당"·"쇼핑"·"여가" 등)을 자주 출력하므로
    sub_category까지 함께 확인. 우선순위: 명확한 hit → cat 기반 → sub 기반.
    """
    c = (cat or "").strip()
    sb = (sub or "").strip()
    cs = c + " " + sb  # 둘 다 검사용 합산

    if c in ("집", "직장"):
        return None  # commerce 아님 — 제외

    # 식음료 — 식당·카페·치킨·디저트·주점
    if c in ("식당", "카페", "음식", "외식"):
        return "식음료"
    if any(k in cs for k in ["식사", "한식", "양식", "일식", "중식", "분식", "패스트푸드",
                               "카페", "커피", "디저트", "제과", "베이커리", "주점", "술집",
                               "기타요식", "치킨", "피자", "햄버거", "패밀리"]):
        return "식음료"

    # 편의점/소매 (식품 소매 포함)
    if c in ("식료품", "마트") or sb in ("마트", "식료품"):
        return "편의점/소매"
    if any(k in cs for k in ["편의점", "슈퍼", "할인점", "농수산", "청과", "수산",
                               "반찬", "육류", "미곡"]):
        return "편의점/소매"

    # 의류/잡화 — 의류·생활잡화·전자·가구
    if c == "쇼핑":
        # 쇼핑 안에서 마트·식료품은 위에서 잡힘 — 여기서 의류/잡화로
        if any(k in sb for k in ["의류", "신발", "가방", "잡화", "생활용품", "가전", "가구",
                                   "악세", "화장품", "컴퓨터", "핸드폰", "조명", "문구", "서적",
                                   "스포츠", "운동", "꽃", "식물", "기타상품"]):
            return "의류/잡화"
        return "의류/잡화"  # 쇼핑 default
    if any(k in cs for k in ["의류", "신발", "가방", "잡화", "생활잡화", "가전", "가구",
                               "악세", "화장품", "컴퓨터", "핸드폰", "서적", "문구"]):
        return "의류/잡화"

    # 의료
    if c in ("건강", "의료"):
        return "의료"
    if any(k in cs for k in ["병원", "의원", "약국", "치과", "건강", "한의원", "의약"]):
        return "의료"

    # 교육
    if c == "교육" or any(k in cs for k in ["학원", "교육", "교습"]):
        return "교육·학원"

    # 문화·여가 — cat="여가" 전체 + 키워드
    if c in ("여가", "오락", "문화", "취미"):
        return "문화·여가"
    if any(k in cs for k in ["영화", "노래방", "PC방", "볼링", "골프", "헬스", "스포츠",
                               "운동", "취미", "여행", "공연", "전시"]):
        return "문화·여가"

    # 미용·서비스
    if any(k in cs for k in ["미용", "이용", "네일", "피부", "세탁"]):
        return "미용·서비스"

    # 운송·주유
    if any(k in cs for k in ["주유", "세차", "주차", "자동차", "수리"]):
        return "운송·주유"

    # 숙박
    if any(k in cs for k in ["숙박", "호텔", "모텔", "펜션"]):
        return "숙박"

    return "기타"


# ─────────────────────────────────────────────────────
# 서울 실측 데이터 로드 — 1분기만 사용 (held-out)
# ─────────────────────────────────────────────────────
def load_seoul_real():
    """서울 상권분석 1분기 데이터 → 집계.

    Returns dict with:
      - by_gu: 자치구별 매출 총액 (행정동 코드 앞5자리)
      - by_time6: 시간대 6구간 매출 비율
      - by_daebunryu: 대분류 10개 매출 비율
      - weekday_total, weekend_total
      - by_age_daebunryu: 연령대 × 대분류 매출
      - by_gender_daebunryu: 성별 × 대분류 매출
    """
    GU_CODE = {
        "11110": "종로구", "11140": "중구", "11170": "용산구",
        "11200": "성동구", "11215": "광진구", "11230": "동대문구",
        "11260": "중랑구", "11290": "성북구", "11305": "강북구",
        "11320": "도봉구", "11350": "노원구", "11380": "은평구",
        "11410": "서대문구", "11440": "마포구", "11470": "양천구",
        "11500": "강서구", "11530": "구로구", "11545": "금천구",
        "11560": "영등포구", "11590": "동작구", "11620": "관악구",
        "11650": "서초구", "11680": "강남구", "11710": "송파구",
        "11740": "강동구",
    }
    by_gu = defaultdict(float)
    by_time6 = [0.0] * 6
    by_daebunryu = defaultdict(float)
    weekday_total = 0.0; weekend_total = 0.0
    by_age_db = defaultdict(lambda: defaultdict(float))
    by_gender_db = defaultdict(lambda: defaultdict(float))

    age_cols = [
        ("10대", "연령대_10_매출_금액"),
        ("20대", "연령대_20_매출_금액"),
        ("30대", "연령대_30_매출_금액"),
        ("40대", "연령대_40_매출_금액"),
        ("50대", "연령대_50_매출_금액"),
        ("60대이상", "연령대_60_이상_매출_금액"),
    ]
    time_cols = [
        "시간대_00~06_매출_금액",
        "시간대_06~11_매출_금액",
        "시간대_11~14_매출_금액",
        "시간대_14~17_매출_금액",
        "시간대_17~21_매출_금액",
        "시간대_21~24_매출_금액",
    ]
    USE_QUARTER = "20251"
    with open(SEOUL_CSV, encoding="cp949") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("기준_년분기_코드") != USE_QUARTER:
                continue
            dong = row.get("행정동_코드", "")
            if not dong or not dong.startswith("11"):
                continue
            gu = GU_CODE.get(dong[:5])
            sales = float(row.get("당월_매출_금액") or 0)
            if sales <= 0:
                continue
            db = map_seoul_industry(row.get("서비스_업종_코드_명", ""))
            if gu: by_gu[gu] += sales
            by_daebunryu[db] += sales
            for i, c in enumerate(time_cols):
                by_time6[i] += float(row.get(c) or 0)
            weekday_total += float(row.get("주중_매출_금액") or 0)
            weekend_total += float(row.get("주말_매출_금액") or 0)
            # 성별·연령 × 대분류
            by_gender_db["남"][db] += float(row.get("남성_매출_금액") or 0)
            by_gender_db["여"][db] += float(row.get("여성_매출_금액") or 0)
            for age, col in age_cols:
                by_age_db[age][db] += float(row.get(col) or 0)

    return {
        "by_gu": dict(by_gu),
        "by_time6": by_time6,
        "by_daebunryu": dict(by_daebunryu),
        "weekday_total": weekday_total,
        "weekend_total": weekend_total,
        "by_age_daebunryu": {k: dict(v) for k, v in by_age_db.items()},
        "by_gender_daebunryu": {k: dict(v) for k, v in by_gender_db.items()},
    }


# ─────────────────────────────────────────────────────
# 시뮬 데이터 로드 — Day 1~N commerce
# ─────────────────────────────────────────────────────
def load_sim_data(days_strs):
    """시뮬 4일 INCLUDES commerce 거래 집계."""
    GU_CODE_TO_NAME = {
        "11110": "종로구", "11140": "중구", "11170": "용산구",
        "11200": "성동구", "11215": "광진구", "11230": "동대문구",
        "11260": "중랑구", "11290": "성북구", "11305": "강북구",
        "11320": "도봉구", "11350": "노원구", "11380": "은평구",
        "11410": "서대문구", "11440": "마포구", "11470": "양천구",
        "11500": "강서구", "11530": "구로구", "11545": "금천구",
        "11560": "영등포구", "11590": "동작구", "11620": "관악구",
        "11650": "서초구", "11680": "강남구", "11710": "송파구",
        "11740": "강동구",
    }
    by_gu = defaultdict(float)
    by_time6 = [0.0] * 6
    by_daebunryu = defaultdict(float)
    weekday_total = 0.0; weekend_total = 0.0
    by_age_db = defaultdict(lambda: defaultdict(float))
    by_gender_db = defaultdict(lambda: defaultdict(float))
    # 페르소나별 소득-소비
    income_spend = defaultdict(lambda: {"agents": set(), "total": 0.0})
    person_spend = defaultdict(float)  # aid → 누적 소비
    person_income_level = {}  # aid → spending_level_wd (1~10)
    person_daily_wd = {}  # aid → 페르소나 daily_wd (budget proxy)

    with driver_session() as s:
        # 페르소나 소득 분위 + daily_wd
        for r in s.run("MATCH (a:Agent) RETURN a.id AS id, a.spending_level_wd AS lvl, a.s_daily_wd AS wd, a.p_age_group AS age, a.p_gender AS g, a.residence_dong_code_raw AS dong"):
            person_income_level[r['id']] = r['lvl']
            person_daily_wd[r['id']] = r['wd']
        # 모든 commerce INCLUDES (4일)
        for r in s.run("""
            MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(poi:POI {type:'commerce'})
            WHERE toString(p.day) IN $days AND i.actual_spent > 0
            RETURN a.id AS aid, a.residence_gu AS gu, a.p_age_group AS age, a.p_gender AS g,
                   i.category AS cat, i.sub_category AS sub, i.actual_spent AS sp,
                   i.time AS t, p.day AS day
        """, days=days_strs):
            aid = r['aid']; sp = float(r['sp'] or 0); cat = r['cat']; sub = r['sub']; gu = r['gu']
            db = map_sim_category(cat, sub)
            if db is None:
                continue
            if gu: by_gu[gu] += sp
            by_daebunryu[db] += sp
            # 시간대 6구간
            t = r['t']
            hour = t.hour if hasattr(t, 'hour') else int(str(t).split(':')[0])
            if hour < 6: by_time6[0] += sp
            elif hour < 11: by_time6[1] += sp
            elif hour < 14: by_time6[2] += sp
            elif hour < 17: by_time6[3] += sp
            elif hour < 21: by_time6[4] += sp
            else: by_time6[5] += sp
            # 주중/주말
            from datetime import date as _date
            d = _date.fromisoformat(str(r['day']))
            if d.weekday() >= 5: weekend_total += sp
            else: weekday_total += sp
            # 연령·성별 × 대분류
            age = r['age']; gender = r['g']
            if age: by_age_db[age][db] += sp
            if gender:
                gender_kr = "남" if gender == "M" else "여"
                by_gender_db[gender_kr][db] += sp
            # 소득-소비
            person_spend[aid] += sp

    # income_level → 소비 매핑 + 페르소나 daily_wd budget
    inc_pairs = []  # (level, sim_total_spend)
    inc_budget_pairs = []  # (level, daily_wd budget)
    for aid, sp in person_spend.items():
        lvl = person_income_level.get(aid)
        wd = person_daily_wd.get(aid)
        if lvl is not None and lvl > 0:
            inc_pairs.append((lvl, sp))
            if wd:
                inc_budget_pairs.append((lvl, wd))
    return {
        "by_gu": dict(by_gu),
        "by_time6": by_time6,
        "by_daebunryu": dict(by_daebunryu),
        "weekday_total": weekday_total,
        "weekend_total": weekend_total,
        "by_age_daebunryu": {k: dict(v) for k, v in by_age_db.items()},
        "by_gender_daebunryu": {k: dict(v) for k, v in by_gender_db.items()},
        "income_spend_pairs": inc_pairs,
        "income_budget_pairs": inc_budget_pairs,
    }


# ─────────────────────────────────────────────────────
# 6개 검증 함수 + 차트
# ─────────────────────────────────────────────────────
def run_validation(start, days, out_dir: Path) -> dict:
    """6개 검증 일괄 실행 → 결과 dict + 차트 fig."""
    from datetime import date, timedelta
    out_dir.mkdir(parents=True, exist_ok=True)
    days_strs = [(start + timedelta(days=i)).isoformat() for i in range(days)]

    print("[V] 서울 실측 데이터 로드 (held-out)...", file=sys.stderr)
    real = load_seoul_real()
    print("[V] 시뮬 데이터 로드 ...", file=sys.stderr)
    sim = load_sim_data(days_strs)

    results = {}

    # V1. 공간 분포 (자치구 25개)
    gu_list = sorted(set(real["by_gu"].keys()) | set(sim["by_gu"].keys()))
    real_gu = [real["by_gu"].get(g, 0) for g in gu_list]
    sim_gu = [sim["by_gu"].get(g, 0) for g in gu_list]
    real_gu_n = _normalize(real_gu)
    sim_gu_n = _normalize(sim_gu)
    results["V1"] = {
        "name": "공간 분포 (자치구 25개)",
        "cosine": cosine(real_gu_n, sim_gu_n),
        "jsd": jsd(real_gu_n, sim_gu_n),
        "mape": mape(real_gu_n, sim_gu_n),
        "real_gini": gini(real_gu),
        "sim_gini": gini(sim_gu),
        "real_top3": sorted(zip(gu_list, real_gu_n), key=lambda x: -x[1])[:3],
        "sim_top3": sorted(zip(gu_list, sim_gu_n), key=lambda x: -x[1])[:3],
    }

    # V2. 시간대 6구간
    real_t = _normalize(real["by_time6"])
    sim_t = _normalize(sim["by_time6"])
    time_labels = ["00~06", "06~11", "11~14", "14~17", "17~21", "21~24"]
    results["V2"] = {
        "name": "시간대 활동 분포 (6구간)",
        "cosine": cosine(real_t, sim_t),
        "jsd": jsd(real_t, sim_t),
        "peak_real": time_labels[real_t.index(max(real_t))],
        "peak_sim": time_labels[sim_t.index(max(sim_t))],
        "real": real_t, "sim": sim_t, "labels": time_labels,
    }

    # V3. 업종 분포 (대분류 10개)
    real_db = [real["by_daebunryu"].get(d, 0) for d in DAEBUNRYU]
    sim_db = [sim["by_daebunryu"].get(d, 0) for d in DAEBUNRYU]
    real_db_n = _normalize(real_db)
    sim_db_n = _normalize(sim_db)
    results["V3"] = {
        "name": "업종 분포 (대분류 10개)",
        "cosine": cosine(real_db_n, sim_db_n),
        "jsd": jsd(real_db_n, sim_db_n),
        "spearman": spearman(real_db, sim_db),
        "labels": DAEBUNRYU, "real": real_db_n, "sim": sim_db_n,
    }

    # V4. 연령 × 대분류 cross
    age_list = ["10대", "20대", "30대", "40대", "50대", "60대이상"]
    sim_age_map = {"10대":"10대","20대":"20대","30대":"30대","40대":"40대","50대":"50대","60대":"60대이상","70대이상":"60대이상"}
    real_age_mat = []
    sim_age_mat = []
    for age in age_list:
        real_row = [real["by_age_daebunryu"].get(age, {}).get(d, 0) for d in DAEBUNRYU]
        # 시뮬에서 70대이상→60대이상 합산
        sim_row = [0.0] * len(DAEBUNRYU)
        for sim_age, target_age in sim_age_map.items():
            if target_age != age: continue
            row = sim["by_age_daebunryu"].get(sim_age, {})
            for j, d in enumerate(DAEBUNRYU):
                sim_row[j] += row.get(d, 0)
        real_age_mat.append(_normalize(real_row))
        sim_age_mat.append(_normalize(sim_row))
    # 각 연령대 row 코사인 평균
    cos_per_age = [cosine(r, s) for r, s in zip(real_age_mat, sim_age_mat) if sum(r) > 0 and sum(s) > 0]
    results["V4_age"] = {
        "name": "연령×업종 cross 분포",
        "cosine_avg": sum(cos_per_age) / len(cos_per_age) if cos_per_age else 0,
        "per_age_cos": list(zip(age_list, [cosine(r, s) if sum(r) > 0 and sum(s) > 0 else 0
                                            for r, s in zip(real_age_mat, sim_age_mat)])),
    }
    # 성별 × 대분류
    real_m = _normalize([real["by_gender_daebunryu"].get("남", {}).get(d, 0) for d in DAEBUNRYU])
    sim_m = _normalize([sim["by_gender_daebunryu"].get("남", {}).get(d, 0) for d in DAEBUNRYU])
    real_f = _normalize([real["by_gender_daebunryu"].get("여", {}).get(d, 0) for d in DAEBUNRYU])
    sim_f = _normalize([sim["by_gender_daebunryu"].get("여", {}).get(d, 0) for d in DAEBUNRYU])
    results["V4_gender"] = {
        "name": "성별×업종 cross 분포",
        "cosine_male": cosine(real_m, sim_m),
        "cosine_female": cosine(real_f, sim_f),
    }

    # V5. 주중/주말 비율
    real_wd_pct = real["weekday_total"] / (real["weekday_total"] + real["weekend_total"]) * 100
    sim_wd_pct = sim["weekday_total"] / (sim["weekday_total"] + sim["weekend_total"]) * 100 if (sim["weekday_total"] + sim["weekend_total"]) > 0 else 0
    # 평일 1일 평균 / 주말 1일 평균 ratio
    real_ratio = (real["weekday_total"] / 5) / (real["weekend_total"] / 2) if real["weekend_total"] > 0 else 0
    sim_weekday_days = sum(1 for d in days_strs if _to_date(d).weekday() < 5)
    sim_weekend_days = sum(1 for d in days_strs if _to_date(d).weekday() >= 5)
    sim_ratio = (sim["weekday_total"] / sim_weekday_days) / (sim["weekend_total"] / sim_weekend_days) if sim_weekend_days > 0 and sim["weekend_total"] > 0 else 0
    results["V5"] = {
        "name": "주중/주말 비율",
        "real_weekday_pct": real_wd_pct,
        "real_weekend_pct": 100 - real_wd_pct,
        "sim_weekday_pct": sim_wd_pct,
        "sim_weekend_pct": 100 - sim_wd_pct,
        "real_daily_ratio": real_ratio,
        "sim_daily_ratio": sim_ratio,
        "sim_weekday_days": sim_weekday_days,
        "sim_weekend_days": sim_weekend_days,
        "mape": abs(real_wd_pct - sim_wd_pct) / real_wd_pct * 100 if real_wd_pct > 0 else 0,
        "note": f"baseline {days}일 (평일 {sim_weekday_days}일 + 주말 {sim_weekend_days}일)",
    }

    # V6. 페르소나 소비분위 적합도 (Persona Fidelity)
    # 정확한 명칭: 페르소나 소비분위(1~10) × 시뮬 실제 commerce 소비액의 피어슨 상관.
    # 경제학의 "소득 탄력성"이 아님 (그건 종단 측정 필요). cross-sectional 분위-실소비 적합도.
    if sim["income_spend_pairs"]:
        levels = [p[0] for p in sim["income_spend_pairs"]]
        spends = [p[1] for p in sim["income_spend_pairs"]]
        rho = pearson(levels, spends)
        # 분위별 평균 1인당 일소비
        lvl_avg = defaultdict(list)
        for lvl, sp in sim["income_spend_pairs"]:
            lvl_avg[lvl].append(sp / days)
        lvl_avg = {k: sum(v) / len(v) for k, v in lvl_avg.items()}
    else:
        rho = 0; lvl_avg = {}
    # 페르소나 daily_wd budget 분위별 평균
    budget_avg = {}
    if sim.get("income_budget_pairs"):
        bd = defaultdict(list)
        for lvl, wd in sim["income_budget_pairs"]:
            bd[lvl].append(wd)
        budget_avg = {k: sum(v)/len(v) for k, v in bd.items()}
    # 분위 1·10 ratio + budget fidelity
    sim_ratio = lvl_avg.get(10, 0) / lvl_avg.get(1, 1) if lvl_avg.get(1) else 0
    budget_ratio = budget_avg.get(10, 0) / budget_avg.get(1, 1) if budget_avg.get(1) else 0
    # budget fidelity: 분위별 (시뮬 실소비 / 페르소나 daily_wd budget)
    fidelity = {}
    for lvl in sorted(set(lvl_avg.keys()) & set(budget_avg.keys())):
        if budget_avg[lvl] > 0:
            fidelity[lvl] = lvl_avg[lvl] / budget_avg[lvl] * 100
    results["V6"] = {
        "name": "페르소나 소비분위 적합도 (Persona Fidelity)",
        "rho": rho,
        "lvl_avg": dict(sorted(lvl_avg.items())),
        "budget_avg": dict(sorted(budget_avg.items())),
        "sim_q1_q10_ratio": sim_ratio,
        "budget_q1_q10_ratio": budget_ratio,
        "fidelity_pct": fidelity,  # 분위별 budget 활용률
        "note": "ρ는 분위와 실소비의 선형 상관. ρ≥0.7 강함, 0.4~0.7 중간, <0.4 약함. (경제학 '탄력성'이 아닌 cross-sectional 적합도)",
    }

    # 연령 간 ratio — 각 대분류별 "20대/60대이상" 점유율 ratio
    # 1.0보다 크면 20대가 더 많이 쓰는 업종, 1.0보다 작으면 60대가 더 많이 쓰는 업종.
    age_20 = "20대"; age_60 = "60대이상"
    real_20_db = [real["by_age_daebunryu"].get(age_20, {}).get(d, 0) for d in DAEBUNRYU]
    real_60_db = [real["by_age_daebunryu"].get(age_60, {}).get(d, 0) for d in DAEBUNRYU]
    sim_20_db = [sim["by_age_daebunryu"].get(age_20, {}).get(d, 0) for d in DAEBUNRYU]
    sim_60_db_raw = [0.0] * len(DAEBUNRYU)
    for age_src in ("60대", "70대이상"):
        row = sim["by_age_daebunryu"].get(age_src, {})
        for j, d in enumerate(DAEBUNRYU):
            sim_60_db_raw[j] += row.get(d, 0)
    real_20_n = _normalize(real_20_db); real_60_n = _normalize(real_60_db)
    sim_20_n = _normalize(sim_20_db); sim_60_n = _normalize(sim_60_db_raw)
    real_ratio = [(r20 / r60) if r60 > 0 else None for r20, r60 in zip(real_20_n, real_60_n)]
    sim_ratio = [(s20 / s60) if s60 > 0 else None for s20, s60 in zip(sim_20_n, sim_60_n)]
    age_ratio_pairs = []  # [(daebunryu, real_ratio, sim_ratio, agree)]
    for d, rr, sr in zip(DAEBUNRYU, real_ratio, sim_ratio):
        if rr is None or sr is None:
            agree = None
        else:
            # 같은 방향(>1 vs >1, <1 vs <1)이면 일치
            agree = (rr > 1) == (sr > 1)
        age_ratio_pairs.append((d, rr, sr, agree))
    # 부분 코사인 — None 빼고
    rr_valid = [(rr, sr) for _, rr, sr, _ in age_ratio_pairs if rr is not None and sr is not None and rr > 0 and sr > 0]
    if rr_valid:
        rr_v = [x[0] for x in rr_valid]; sr_v = [x[1] for x in rr_valid]
        age_ratio_cos = cosine(rr_v, sr_v)
    else:
        age_ratio_cos = 0.0
    direction_agree = sum(1 for _, _, _, a in age_ratio_pairs if a is True)
    direction_total = sum(1 for _, _, _, a in age_ratio_pairs if a is not None)
    results["V_age_ratio"] = {
        "name": "연령 간 ratio (20대/60대이상)",
        "pairs": age_ratio_pairs,
        "cosine": age_ratio_cos,
        "direction_agree": direction_agree,
        "direction_total": direction_total,
    }

    # 차트 — 6 panel (적정 사이즈, 축 글씨 ↑)
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12})
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    # V1. 자치구
    ax = axes[0, 0]
    idx = sorted(range(len(gu_list)), key=lambda i: -real_gu_n[i])[:10]
    g_show = [gu_list[i] for i in idx]
    ax.barh(range(10), [real_gu_n[i] for i in idx], color="#4361ee", alpha=0.7, label="실측")
    ax.barh(range(10), [sim_gu_n[i] for i in idx], color="#f72585", alpha=0.7, label="시뮬")
    ax.set_yticks(range(10)); ax.set_yticklabels(g_show)
    ax.invert_yaxis()
    ax.set_title(f"V1. 자치구 분포 cos={results['V1']['cosine']:.3f}")
    ax.legend()
    # V2. 시간대
    ax = axes[0, 1]
    x = range(6)
    ax.bar([i - 0.2 for i in x], real_t, width=0.4, color="#4361ee", label="실측")
    ax.bar([i + 0.2 for i in x], sim_t, width=0.4, color="#f72585", label="시뮬")
    ax.set_xticks(x); ax.set_xticklabels(time_labels, rotation=30)
    ax.set_title(f"V2. 시간대 cos={results['V2']['cosine']:.3f}")
    ax.legend()
    # V3. 대분류
    ax = axes[0, 2]
    x = range(len(DAEBUNRYU))
    ax.bar([i - 0.2 for i in x], real_db_n, width=0.4, color="#4361ee", label="실측")
    ax.bar([i + 0.2 for i in x], sim_db_n, width=0.4, color="#f72585", label="시뮬")
    ax.set_xticks(x); ax.set_xticklabels(DAEBUNRYU, rotation=45, ha='right', fontsize=8)
    ax.set_title(f"V3. 업종 분포 cos={results['V3']['cosine']:.3f}")
    ax.legend()
    # V4. 연령×업종 cosine per age
    ax = axes[1, 0]
    age_lbls = [x[0] for x in results["V4_age"]["per_age_cos"]]
    age_cos = [x[1] for x in results["V4_age"]["per_age_cos"]]
    ax.bar(age_lbls, age_cos, color="#3a0ca3")
    ax.axhline(y=0.9, color="green", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_title(f"V4. 연령×업종 코사인 (avg={results['V4_age']['cosine_avg']:.3f})")
    # V5. 주중/주말 — 4-bar (실측 평일·주말 + 시뮬 평일·주말)
    ax = axes[1, 1]
    labels = ["실측 평일", "실측 주말", "시뮬 평일", "시뮬 주말"]
    vals = [real_wd_pct, 100 - real_wd_pct, sim_wd_pct, 100 - sim_wd_pct]
    colors = ["#4361ee", "#a3c4ff", "#f72585", "#ffc6e0"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 110)
    ax.set_ylabel("매출 비율 (%)")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=9)
    ax.set_title(f"V5. 주중/주말 매출 비율")
    # V6. 소득 분위별 평균
    ax = axes[1, 2]
    if lvl_avg:
        lvls = sorted(lvl_avg.keys())
        ax.bar(lvls, [lvl_avg[l] for l in lvls], color="#ffb703")
        ax.set_xlabel("소비분위 (1~10)")
        ax.set_ylabel("1인당 일평균 commerce (원)")
        ax.set_title(f"V6. 소득-소비 탄력성 ρ={rho:.3f}")
    else:
        ax.text(0.5, 0.5, "no data", ha='center')

    plt.tight_layout(pad=2.5)
    fname = "fig6_validation.png"
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()

    # 차트 보조 — V3 업종 차이 + 연령간 ratio (2 panel)
    fig2, axb = plt.subplots(1, 2, figsize=(18, 7))
    # 왼쪽: V3 업종별 실측-시뮬 차이 (시뮬 - 실측)
    diffs = [(s - r) * 100 for r, s in zip(results["V3"]["real"], results["V3"]["sim"])]
    colors = ['#f72585' if d > 0 else '#4361ee' for d in diffs]
    axb[0].barh(DAEBUNRYU, diffs, color=colors)
    axb[0].axvline(x=0, color='gray', linewidth=1)
    axb[0].set_title("V3 업종 분포 차이 (시뮬 - 실측, %p)\n양수=시뮬 과잉, 음수=시뮬 결손", fontsize=15)
    axb[0].set_xlabel("차이 (%p)", fontsize=13)
    for i, d in enumerate(diffs):
        axb[0].text(d + (0.5 if d >= 0 else -0.5), i, f"{d:+.1f}", va='center',
                    ha='left' if d >= 0 else 'right', fontsize=11)
    # 오른쪽: 연령간 ratio (20대 / 60대이상) - 실측 vs 시뮬
    ratio_lbls = [d for d, rr, sr, _ in age_ratio_pairs if rr is not None and sr is not None]
    real_r = [rr for d, rr, sr, _ in age_ratio_pairs if rr is not None and sr is not None]
    sim_r = [sr for d, rr, sr, _ in age_ratio_pairs if rr is not None and sr is not None]
    x = range(len(ratio_lbls))
    axb[1].bar([i - 0.2 for i in x], real_r, width=0.4, color="#4361ee", label="실측")
    axb[1].bar([i + 0.2 for i in x], sim_r, width=0.4, color="#f72585", label="시뮬")
    axb[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.6)
    axb[1].set_xticks(list(x))
    axb[1].set_xticklabels(ratio_lbls, rotation=40, ha='right', fontsize=12)
    axb[1].set_title(f"연령 간 ratio: 20대 점유 / 60대이상 점유\n(cos {age_ratio_cos:.3f}, 방향 일치 {direction_agree}/{direction_total})", fontsize=15)
    axb[1].set_ylabel("ratio (1.0=같음, >1=20대 우세, <1=60대 우세)", fontsize=13)
    axb[1].legend(fontsize=12)
    plt.tight_layout(pad=2.5)
    fname_b = "fig7_v3_age_ratio.png"
    plt.savefig(out_dir / fname_b, dpi=150, bbox_inches="tight")
    plt.close()

    results["fig"] = fname
    results["fig_v3_age_ratio"] = fname_b
    return results


def _to_date(s: str):
    from datetime import date as _d
    return _d.fromisoformat(s)
