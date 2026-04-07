"""
preprocess_join.py
======================
Reads raw / synthetic datasets from `synthetic_data/raw_samples` (or
`synthetic_data/full_synth` for the synthetic pipeline) and the code
mapping tables from `mapping_code/`.

Performs outer/left joins according to `docs/data_join.md`:
  - Level 1  joined_persona_base.csv   [adm8, gender, age]
  - Level 2  joined_dong_context.csv   [adm8] only
  - Unjoined & ref files              copied verbatim from raw_samples

Missing values are explicitly preserved as empty strings (no imputation).

Output directories:
  joinData/original/   ← raw_samples pipeline
  joinData/synthetic/  ← full_synth pipeline
"""

import csv
import shutil
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths  (MAPPING_DIR is fixed; SAMPLE_DIR / OUT_DIR are set per pipeline)
# ---------------------------------------------------------------------------
MAPPING_DIR = Path("data/mapping")
SAMPLE_DIR: Path = None  # set by run_pipeline()
OUT_DIR: Path = None      # set by run_pipeline()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def clean_gender(g):
    g = str(g).strip()
    if g in ("남", "M", "1", "male"):
        return "M"
    if g in ("여", "F", "2", "female"):
        return "F"
    return "U"

def clean_age(s):
    """Normalise the many age formats into a canonical set:
       20세미만, 20대, 30대, 40대, 50대, 60대, 70대이상
    """
    s = str(s).strip()
    # Handle "30_39세" → "30"
    if "_" in s:
        s = s.split("_")[0]
    if "이상" in s:
        return "70대이상"
    if "대" in s:
        if "70" in s:
            return "70대이상"
        return s.replace("세", "")
    if "미만" in s:
        return "20세미만"
    # Remove trailing "세"
    s = s.replace("세", "")
    if s.isdigit():
        a = int(s)
        if len(s) == 4:
            a = int(s[:2])       # 4044 → 40
        if a < 20:
            return "20세미만"
        if a < 30:
            return "20대"
        if a < 40:
            return "30대"
        if a < 50:
            return "40대"
        if a < 60:
            return "50대"
        if a < 70:
            return "60대"
        return "70대이상"
    return "U"

def csv_read(path, enc=None):
    """Read a CSV, trying several encodings automatically."""
    if not path.exists():
        print(f"  ⚠ File not found: {path}")
        return [], []
    encodings = [enc] if enc else ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, errors="strict") as f:
                reader = csv.reader(f)
                try:
                    h = [c.strip().strip('"').strip("\ufeff") for c in next(reader)]
                except StopIteration:
                    return [], []
                rows = [row for row in reader if row]
            return h, rows
        except UnicodeDecodeError:
            continue
    # Fallback
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        try:
            h = [c.strip().strip('"').strip("\ufeff") for c in next(reader)]
        except StopIteration:
            return [], []
        rows = [row for row in reader if row]
    return h, rows

def csv_write(path, headers, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

# ---------------------------------------------------------------------------
# Telecom-29 column discovery
# ---------------------------------------------------------------------------
# Maps a *keyword* (without spaces) → short metric name.
# We want only the **평균** (average / 합계 / 사용일수 / ...) columns,
# explicitly excluding '4분위수', '25%', '50%', '75%', and '미추정'.
_TEL_METRIC_MAP = {
    "야간상주지변경횟수평균":       "tel_night_move",
    "주간상주지변경횟수평균":       "tel_day_move",
    "평균출근소요시간평균":         "tel_commute_time",
    "평균근무시간평균":             "tel_work_time",
    "소액결재사용횟수평균":         "tel_micropay_cnt",
    "소액결재사용금액평균":         "tel_micropay_amt",
    "SNS사용횟수":                 "tel_sns_cnt",          # col 39, no "평균" suffix
    "평균통화량":                   "tel_call_amt",
    "평균문자량":                   "tel_text_amt",
    "평균통화대상자수":             "tel_call_users",
    "평균문자대상자수":             "tel_text_users",
    "데이터사용량":                 "tel_data_usage",       # col 59
    "평일총이동횟수":               "tel_wd_move_cnt",      # col 64, NOT "미추정"
    "휴일총이동횟수평균":           "tel_we_move_cnt",      # col 69
    "집추정위치평일총체류시간":     "tel_home_wd_time",     # col 74
    "집추정위치휴일총체류시간":     "tel_home_we_time",     # col 79
    "평일총이동거리합계":           "tel_wd_move_dist",     # col 84
    "휴일총이동거리합계":           "tel_we_move_dist",     # col 89
    "지하철이동일수합계":           "tel_subway_days",      # col 94
    "게임서비스사용일수":           "tel_game_days",        # col 99
    "금융서비스사용일수":           "tel_finance_days",     # col 104
    "쇼핑서비스사용일수":           "tel_shopping_days",    # col 109
    "동영상/방송서비스사용일수":    "tel_video_days",       # col 114
    "유튜브사용일수":               "tel_youtube_days",     # col 119
    "넷플릭스사용일수":             "tel_netflix_days",     # col 124
    "배달서비스사용일수":           "tel_delivery_days",    # col 129
    "배달_브랜드서비스사용일수":    "tel_delivery_brand_days",
    "배달_식재료서비스사용일수":    "tel_delivery_grocery_days",
    "최근3개월내요금연체비율":      "tel_overdue_ratio",
}

def find_tel_col_indices(headers):
    """Return {short_name: col_index} for the 29 target telecom metrics."""
    result = {}
    for i, h in enumerate(headers):
        h_clean = h.replace(" ", "")
        # Skip quartile / unestimated columns
        if any(q in h_clean for q in ["25%", "50%", "75%", "4분위수", "미추정"]):
            continue
        for pattern, short in _TEL_METRIC_MAP.items():
            if pattern in h_clean and short not in result:
                result[short] = i
    return result

# ---------------------------------------------------------------------------
# Step 1 – Code Mappings
# ---------------------------------------------------------------------------
def get_mappings():
    """Load stat7→adm8 mapping from code_mapping_mopas_nso.csv.
    Columns: 행안부_8자리[0], 행안부_10자리[1], 통계청_7자리[2], 자치구[3], 행정동명[4]
    """
    print("[1] Loading Code Mappings …")
    stat7_to_adm8 = {}
    h, rows = csv_read(MAPPING_DIR / "code_mapping_mopas_nso.csv")
    for r in rows:
        if len(r) > 2 and r[2].strip() and r[0].strip():
            stat7_to_adm8[r[2].strip()] = r[0].strip()[:8]
    print(f"    Loaded {len(stat7_to_adm8)} stat7→adm8 mappings")
    return stat7_to_adm8

# ---------------------------------------------------------------------------
# Step 2 – Joined Persona Base  (Level 1 join by [adm8, gender, age])
# ---------------------------------------------------------------------------
def merge_demographics(stat7_to_adm8):
    print("[2] Processing & Joining Demographic Data (Level 1) …")

    # ── 2-a: Telecom 29 (master base) ────────────────────────────
    # telecom_29.csv lives in the SAMPLE_DIR (raw_samples or full_synth)
    h_tel, r_tel = csv_read(SAMPLE_DIR / "telecom_29.csv")
    tel_cols = find_tel_col_indices(h_tel)
    tel_attr_names = sorted(tel_cols.keys())
    print(f"    telecom_29: {len(r_tel)} rows, {len(tel_cols)}/{len(_TEL_METRIC_MAP)} metrics matched")

    # Accumulate weighted sums  (value * pop) per (adm8, gender, age)
    def _init():
        return {"pop": 0.0, "count": 0, **{k: 0.0 for k in tel_attr_names}}

    tel_agg = defaultdict(_init)
    gu_dong = {}

    for r in r_tel:
        if len(r) < 6:
            continue
        st7 = r[0].strip()
        adm8 = stat7_to_adm8.get(st7)
        if not adm8:
            continue

        gu, dong = r[1].strip(), r[2].strip()
        gen = clean_gender(r[3])
        age = clean_age(r[4])
        pop = safe_float(r[5]) or 0.0

        key = (adm8, gen, age)
        gu_dong[adm8] = (gu, dong)

        t = tel_agg[key]
        t["pop"] += pop
        t["count"] += 1

        for attr in tel_attr_names:
            idx = tel_cols[attr]
            val = safe_float(r[idx]) if idx < len(r) else None
            if val is not None:
                t[attr] += val * pop

    # Compute weighted average, clamping negative values to 0
    base_data = {}
    for k, v in tel_agg.items():
        pop_total = v["pop"]
        d = {
            "adm_cd_8": k[0],
            "gu": gu_dong.get(k[0], ("", ""))[0],
            "dong": gu_dong.get(k[0], ("", ""))[1],
            "gender": k[1],
            "age_grp": k[2],
            "tel_pop": pop_total,
            # placeholders for joined columns
            "b079_card_amt": None,
            "b009_weekday_flow": None,
            "b009_weekend_flow": None,
            "b063_sb_amt": None,
        }
        for attr in tel_attr_names:
            raw = (v[attr] / pop_total) if pop_total > 0 else 0.0
            # Physical counts/durations cannot be negative
            d[attr] = max(0.0, raw)
        base_data[k] = d

    print(f"    Base personas: {len(base_data)} unique (adm8, gender, age) combos")

    # ── 2-b: B079-1 카드 결제액 ──────────────────────────────────
    # 7.서울시 내국인 성별 연령대별(행정동별).csv
    # Cols: 0=기준일자, 1=가맹점행정동코드(10자리), 2=개인법인구분,
    #        3=성별, 4=연령대, 5=업종대분류, 6=카드이용금액계, 7=카드이용건수계
    h791, r791 = csv_read(SAMPLE_DIR / "7.서울시 내국인 성별 연령대별(행정동별).csv")
    b791_dict = defaultdict(float)
    if h791:
        for r in r791:
            if len(r) < 7:
                continue
            a8 = r[1].strip()[:8]
            gen = clean_gender(r[3])
            age = clean_age(r[4])
            amt = safe_float(r[6]) or 0.0
            b791_dict[(a8, gen, age)] += amt
        print(f"    B079-1 카드결제: {len(b791_dict)} unique keys from {len(r791)} rows")

    # ── 2-c: B009-2 KT 유동인구 ──────────────────────────────────
    # KT 월별 성연령대별 거주지별 유동인구.csv
    # Cols: 0=셀id, 1=x좌표, 2=y좌표, 3=성별, 4=연령대,
    #        5=주중보행인구수, 6=주말보행인구수, 7=거주지코드, 8=행정동코드(8자리), 9=기준년월
    h092, r092 = csv_read(SAMPLE_DIR / "KT 월별 성연령대별 거주지별 유동인구.csv")
    b092_dict = defaultdict(lambda: [0.0, 0.0])
    if h092:
        for r in r092:
            if len(r) < 9:
                continue
            a8 = r[8].strip()[:8]
            gen = clean_gender(r[3])
            age = clean_age(r[4])
            wd = safe_float(r[5]) or 0.0
            we = safe_float(r[6]) or 0.0
            b092_dict[(a8, gen, age)][0] += wd
            b092_dict[(a8, gen, age)][1] += we
        print(f"    B009-2 유동인구: {len(b092_dict)} unique keys from {len(r092)} rows")

    # ── 2-d: B063-2 집계구 성별연령대별 결제 ─────────────────────
    # 내국인(집계구) 성별연령대별.csv
    # Cols: 0=가맹점집계구코드(13자리), 1=내국인업종코드, 2=기준년월, 3=일별,
    #        4=개인법인구분, 5=성별(SEX_CCD), 6=연령대별(AGE_GB), 7=카드이용금액계, 8=카드이용건수
    h632, r632 = csv_read(SAMPLE_DIR / "내국인(집계구) 성별연령대별.csv")
    b632_dict = defaultdict(float)
    if h632:
        for r in r632:
            if len(r) < 8:
                continue
            cen = r[0].strip()
            # Census 13→ first 7 digits = stat7 code → adm8
            a8 = stat7_to_adm8.get(cen[:7])
            if not a8:
                continue
            gen = clean_gender(r[5])
            age = clean_age(r[6])
            amt = safe_float(r[7]) or 0.0
            b632_dict[(a8, gen, age)] += amt
        print(f"    B063-2 집계구결제: {len(b632_dict)} unique keys from {len(r632)} rows")

    # ── Apply Left Joins ─────────────────────────────────────────
    out_headers = ["adm_cd_8", "gu", "dong", "gender", "age_grp", "tel_pop"]
    out_headers.extend(tel_attr_names)
    out_headers.extend(["b079_card_amt", "b009_weekday_flow", "b009_weekend_flow", "b063_sb_amt"])

    out_rows = []
    join_stats = {"b079": 0, "b009": 0, "b063": 0}

    for k, v in base_data.items():
        if k in b791_dict:
            v["b079_card_amt"] = b791_dict[k]
            join_stats["b079"] += 1
        if k in b092_dict:
            v["b009_weekday_flow"], v["b009_weekend_flow"] = b092_dict[k]
            join_stats["b009"] += 1
        if k in b632_dict:
            v["b063_sb_amt"] = b632_dict[k]
            join_stats["b063"] += 1

        row = [v.get(h, "") if v.get(h) is not None else "" for h in out_headers]
        out_rows.append(row)

    total = len(base_data)
    print(f"    Join rates: b079={join_stats['b079']}/{total} ({join_stats['b079']/max(1,total)*100:.1f}%), "
          f"b009={join_stats['b009']}/{total} ({join_stats['b009']/max(1,total)*100:.1f}%), "
          f"b063={join_stats['b063']}/{total} ({join_stats['b063']/max(1,total)*100:.1f}%)")

    out_path = OUT_DIR / "joined_persona_base.csv"
    csv_write(out_path, out_headers, out_rows)
    print(f"    → Saved: {out_path.name}  ({len(out_rows)} rows, {len(out_headers)} cols)")
    return base_data

# ---------------------------------------------------------------------------
# Step 3 – Joined Dong Context  (Level 2 join by [adm8] only)
# ---------------------------------------------------------------------------
def merge_context(base_adms, stat7_to_adm8):
    print("[3] Processing & Joining Context Data (Level 2) …")

    ctx_data = {adm: {"adm_cd_8": adm} for adm in base_adms}

    # ── B069: 상권발달 개별지수 ───────────────────────────────────
    # Cols: 0=일자, 1=행정동코드(8자리), 2=매출지수, 3=인프라지수,
    #        4=가맹점지수, 5=인구지수, 6=금융지수
    h69, r69 = csv_read(SAMPLE_DIR / "행정동별 상권발달 개별지수.csv")
    idx_cnt_69 = 0
    if h69:
        for r in r69:
            if len(r) < 7:
                continue
            a8 = r[1].strip()[:8]
            if a8 in ctx_data and "b069_sales" not in ctx_data[a8]:
                ctx_data[a8]["b069_sales"] = safe_float(r[2])
                ctx_data[a8]["b069_infra"] = safe_float(r[3])
                ctx_data[a8]["b069_store"] = safe_float(r[4])
                ctx_data[a8]["b069_pop"]   = safe_float(r[5])
                ctx_data[a8]["b069_finance"] = safe_float(r[6])
                idx_cnt_69 += 1
        print(f"    B069 상권지수: {idx_cnt_69}/{len(base_adms)} dongs matched")

    # ── B079-2: 유입지별 (행정동별) ───────────────────────────────
    # 8.서울시 내국인의 개인카드 기준 유입지별(행정동별).csv
    # Cols: 0=기준일자, 1=가맹점행정동코드(10자리), 2=고객주소광역시도,
    #        3=고객주소시군구, 4=업종대분류, 5=카드이용금액계, 6=카드이용건수계
    h792, r792 = csv_read(SAMPLE_DIR / "8.서울시 내국인의 개인카드 기준 유입지별(행정동별).csv")
    if h792:
        agg792 = defaultdict(lambda: {"seoul": 0.0, "other": 0.0})
        for r in r792:
            if len(r) < 6:
                continue
            a8 = r[1].strip()[:8]
            if a8 not in ctx_data:
                continue
            amt = safe_float(r[5]) or 0.0
            key = "seoul" if "서울" in r[2] else "other"
            agg792[a8][key] += amt
        for a8, v in agg792.items():
            ctx_data[a8]["b079_2_inflow_seoul"] = v["seoul"]
            ctx_data[a8]["b079_2_inflow_other"] = v["other"]
        print(f"    B079-2 유입지(행정동): {len(agg792)}/{len(base_adms)} dongs matched")

    # ── B063 유입지별 (집계구) ─────────────────────────────────────
    # 내국인(집계구) 유입지별.csv
    # Cols: 0=가맹점집계구코드(13자리), 1=내국인업종코드, 2=기준연월, 3=일별,
    #        4=고객주소광역시(SIDO), 5=고객주소시군구(SGG), 6=카드이용금액계, 7=카드이용건수
    h63in, r63in = csv_read(SAMPLE_DIR / "내국인(집계구) 유입지별.csv")
    if h63in:
        agg63in = defaultdict(lambda: {"seoul": 0.0, "other": 0.0})
        for r in r63in:
            if len(r) < 7:
                continue
            cen = r[0].strip()
            a8 = stat7_to_adm8.get(cen[:7])
            if not a8 or a8 not in ctx_data:
                continue
            amt = safe_float(r[6]) or 0.0
            key = "seoul" if "서울" in r[4] else "other"
            agg63in[a8][key] += amt
        for a8, v in agg63in.items():
            ctx_data[a8]["b063_inflow_seoul"] = v["seoul"]
            ctx_data[a8]["b063_inflow_other"] = v["other"]
        print(f"    B063 유입지(집계구): {len(agg63in)}/{len(base_adms)} dongs matched")

    # ── Write ─────────────────────────────────────────────────────
    out_headers = [
        "adm_cd_8",
        "b069_sales", "b069_infra", "b069_store", "b069_pop", "b069_finance",
        "b079_2_inflow_seoul", "b079_2_inflow_other",
        "b063_inflow_seoul", "b063_inflow_other",
    ]
    out_rows = []
    for adm in sorted(ctx_data.keys()):
        v = ctx_data[adm]
        row = [v.get(h, "") if v.get(h) is not None else "" for h in out_headers]
        out_rows.append(row)

    out_path = OUT_DIR / "joined_dong_context.csv"
    csv_write(out_path, out_headers, out_rows)
    print(f"    → Saved: {out_path.name}  ({len(out_rows)} rows, {len(out_headers)} cols)")

# ---------------------------------------------------------------------------
# Step 4 – Copy unjoined / reference files
# ---------------------------------------------------------------------------
def copy_unjoined_and_ref():
    """Copy unjoined distribution files and reference mapping files.
    
    IMPORTANT: Unjoined files are ALWAYS copied from raw_samples to preserve
    the original distribution, regardless of whether this is the original or
    synthetic pipeline.  Reference mapping files come from mapping_code/.
    """
    print("[4] Copying Unjoined / Reference Files …")

    # Always use raw_samples for unjoined files (preserve original distribution)
    raw_dir = Path("data/raw")

    unjoined_files = [
        ("블록별 성별연령대별 카드소비패턴.csv",           "unjoined_block_consumption.csv"),
        ("내국인(블록) 일자별시간대별.csv",                "unjoined_block_timeslot.csv"),
        ("PURPOSE_250M_202403.csv",                       "unjoined_mobility_purpose.csv"),
        ("KT 월별 시간대별 성연령대별 유동인구.csv",       "unjoined_temporal_activity.csv"),
        ("2.서울시민의 일별 시간대별(행정동).csv",          "unjoined_time_sales.csv"),
    ]
    for src_name, dst_name in unjoined_files:
        src = raw_dir / src_name
        if src.exists():
            shutil.copy(src, OUT_DIR / dst_name)
            print(f"    Copied (raw): {src_name} → {dst_name}")

    ref_files = [
        (MAPPING_DIR / "신한카드 내국인 63업종 코드.csv",  "ref_industry_code_63.csv"),
        (MAPPING_DIR / "카드소비 업종코드.csv",            "ref_industry_code_ss.csv"),
        (MAPPING_DIR / "code_mapping_mopas_nso.csv",      "ref_mopas_nso.csv"),
        (MAPPING_DIR / "adm_code_mapping.csv",            "ref_adm_code.csv"),
    ]
    for src, dst_name in ref_files:
        if src.exists():
            shutil.copy(src, OUT_DIR / dst_name)
            print(f"    Copied (ref): {src.name} → {dst_name}")

# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline(use_synthetic: bool):
    global SAMPLE_DIR, OUT_DIR

    if use_synthetic:
        SAMPLE_DIR = Path("data/synthetic")
        OUT_DIR = Path("output/synthetic")
    else:
        SAMPLE_DIR = Path("data/raw")
        OUT_DIR = Path("output/original")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mode_str = "SYNTHETIC (full_synth)" if use_synthetic else "ORIGINAL (raw_samples)"
    print("=" * 60)
    print(f"  PIPELINE: {mode_str}")
    print("=" * 60)

    stat7_map = get_mappings()
    base_data = merge_demographics(stat7_map)
    base_adms = {v["adm_cd_8"] for v in base_data.values()}
    merge_context(base_adms, stat7_map)
    copy_unjoined_and_ref()

    print(f"\n  [{mode_str}] Pipeline complete!  → {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    # Usage: python preprocess_join.py [original|synthetic|both]
    mode = "both"
    for arg in sys.argv[1:]:
        if arg in ("original", "synthetic", "both"):
            mode = arg

    if mode in ("original", "both"):
        run_pipeline(False)
    if mode in ("synthetic", "both"):
        if mode == "both":
            print("\n")
        run_pipeline(True)

    print(f"\n[OK] Pipeline ({mode}) completed successfully!")
