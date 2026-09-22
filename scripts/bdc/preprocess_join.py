"""
preprocess_join.py
======================
Reads raw / synthetic datasets and code-mapping tables, then performs the
joins defined in `docs/data_join.md`:

  - Level 1  joined_persona_base.csv   [adm8, gender, age]
  - Level 2  joined_dong_context.csv   [adm8] only
  - Unjoined & ref files              copied verbatim

File discovery is delegated to `file_discovery.py`, which handles the
Big Data Campus distribution quirks (zip archives, nested folders, mixed
.csv/.txt, mixed delimiters, Korean encodings).  Column lookups use
keyword matching (`find_col`) rather than hardcoded indices, so minor
schema changes in the source files don't break the pipeline.

Missing values are explicitly preserved as empty strings (no imputation).

Output directories:
  output/original/    ← raw-sample pipeline
  output/synthetic/   ← synthetic-pipeline
"""

import csv
import shutil
import sys
import time
from pathlib import Path

# 이 파일은 scripts/bdc/ 안에 있음 — 프로젝트 루트는 두 단계 위
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from collections import defaultdict
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

# Windows 기본 cp949 콘솔에서 ⚠/→ 등 유니코드 문자를 인코딩하지 못해
# UnicodeEncodeError 가 터지는 것을 방지. utf-8 단말/파일이면 정상 출력.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# 진행 로그 주기 (행 기준). 대용량 스트리밍 도중 정지/속도 체감용.
PROGRESS_EVERY = 2_000_000

from file_discovery import (
    prepare_raw_dir,
    find_dataset,
    smart_read,
    smart_read_many,
    smart_stream_many,
    find_col,
    peek_header,
    open_binary,
    DATASETS,
)

# Lazy import — workers import pandas themselves to avoid cost in main
# process if the pandas-free paths are ever taken.
import pandas as pd

# Pandas chunk size for the large B009/B079/B042 files.  Tuned so each
# chunk fits comfortably in RAM (≈ chunksize × ~12 cols × ~30B ≈ 360MB).
# Larger = less Python overhead, more memory per worker.
# Override with env PD_CHUNKSIZE (e.g. 6000000 on high-RAM VDIs) — bigger
# chunks reduce the per-chunk Python loop overhead substantially.
import os as _os
PD_CHUNKSIZE = int(_os.environ.get("PD_CHUNKSIZE", 3_000_000))

# ---------------------------------------------------------------------------
# Paths  (MAPPING_DIR is fixed; SAMPLE_DIR / OUT_DIR / RAW_INDEX per pipeline)
# ---------------------------------------------------------------------------
MAPPING_DIR = PROJECT_ROOT / "data/mapping"
SAMPLE_DIR: Path = None    # set by run_pipeline()
OUT_DIR: Path = None       # set by run_pipeline()
RAW_INDEX: list = None     # file_discovery index built in run_pipeline()

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
    """Backward-compatible wrapper delegating to file_discovery.smart_read."""
    return smart_read(Path(path))

def csv_write(path, headers, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def read_dataset(dataset_id):
    """Look up `dataset_id` in the current RAW_INDEX and return (header, rows).

    Concatenates if the dataset spans multiple files (e.g. monthly splits).
    Returns ([], []) and warns if no file matches — callers treat this as
    "missing data, skip this section".
    """
    if RAW_INDEX is None:
        raise RuntimeError("RAW_INDEX not initialized; call prepare_raw_dir first")
    paths = find_dataset(RAW_INDEX, dataset_id)
    if not paths:
        print(f"    [dataset] ⚠ no file matched for '{dataset_id}'")
        return [], []
    return smart_read_many(paths)

def stream_dataset(dataset_id):
    """Streaming version of read_dataset — returns (header, row_generator).

    Rows are yielded one at a time, so memory usage is independent of
    dataset size.  Returns ([], iter([])) if no file matches.
    """
    if RAW_INDEX is None:
        raise RuntimeError("RAW_INDEX not initialized; call prepare_raw_dir first")
    paths = find_dataset(RAW_INDEX, dataset_id)
    if not paths:
        print(f"    [dataset] ⚠ no file matched for '{dataset_id}'")
        return [], iter([])
    return smart_stream_many(paths)

# ---------------------------------------------------------------------------
# Telecom-29 column discovery
# ---------------------------------------------------------------------------
# Maps a *keyword* (space-stripped) → short metric name.  We want only the
# **평균 / 합계 / 사용일수** columns, explicitly excluding '4분위수',
# '25%', '50%', '75%', and '미추정'.  Keyword matching is positional-
# agnostic so the actual column index in the raw file does not matter.
_TEL_METRIC_MAP = {
    "야간상주지변경횟수평균":       "tel_night_move",
    "주간상주지변경횟수평균":       "tel_day_move",
    "평균출근소요시간평균":         "tel_commute_time",
    "평균근무시간평균":             "tel_work_time",
    "소액결재사용횟수평균":         "tel_micropay_cnt",
    "소액결재사용금액평균":         "tel_micropay_amt",
    "SNS사용횟수":                 "tel_sns_cnt",
    "평균통화량":                   "tel_call_amt",
    "평균문자량":                   "tel_text_amt",
    "평균통화대상자수":             "tel_call_users",
    "평균문자대상자수":             "tel_text_users",
    "데이터사용량":                 "tel_data_usage",
    "평일총이동횟수":               "tel_wd_move_cnt",
    "휴일총이동횟수평균":           "tel_we_move_cnt",
    "집추정위치평일총체류시간":     "tel_home_wd_time",
    "집추정위치휴일총체류시간":     "tel_home_we_time",
    "평일총이동거리합계":           "tel_wd_move_dist",
    "휴일총이동거리합계":           "tel_we_move_dist",
    "지하철이동일수합계":           "tel_subway_days",
    "게임서비스사용일수":           "tel_game_days",
    "금융서비스사용일수":           "tel_finance_days",
    "쇼핑서비스사용일수":           "tel_shopping_days",
    "동영상/방송서비스사용일수":    "tel_video_days",
    "유튜브사용일수":               "tel_youtube_days",
    "넷플릭스사용일수":             "tel_netflix_days",
    "배달서비스사용일수":           "tel_delivery_days",
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
# Pandas-based parallel workers (5× faster than Python csv.reader loops).
# Strategy:
#   1) peek_header()  → detects encoding/delimiter + returns cleaned header
#   2) pd.read_csv    → C engine, usecols only, dtype=str, chunksize=PD_CHUNKSIZE
#   3) vectorize      → str.strip + str[:8] for adm8, map() for gen/age, to_numeric for amt
#   4) per-chunk groupby + per-worker final groupby → dict output
#
# clean_gender/clean_age vectorized helpers below use map over *unique* values
# in a chunk, so the complex Python logic still applies but runs once per
# distinct input — usually <50 calls per chunk instead of N rows.
# ---------------------------------------------------------------------------
def _vec_gender(s: pd.Series) -> pd.Series:
    """Vectorized clean_gender via unique-value cache."""
    s = s.astype(str).str.strip()
    uniq = s.unique().tolist()
    return s.map({v: clean_gender(v) for v in uniq})

def _vec_age(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    uniq = s.unique().tolist()
    return s.map({v: clean_age(v) for v in uniq})

def _open_for_pandas(entry):
    """Return (file-like, cleanup_callable) usable by pd.read_csv.

    Zip members are streamed via ZipFile.open (forward-only, chunk-by-chunk
    decompression). Previously we did zf.read(member) which loads the entire
    decompressed CSV into a BytesIO — fine for small files, fatal for B078
    where each monthly zip is ~9GB compressed / ~25GB decompressed. A worker
    opening such a zip would need 25GB RAM, which OOMs on 16GB hosts.
    Plain files open in binary mode with a 1MB buffer.
    """
    if entry.is_zip_member:
        import zipfile
        zf = zipfile.ZipFile(entry.zip_path, "r")
        fh = zf.open(entry.member, "r")
        def _cleanup():
            try:
                fh.close()
            finally:
                zf.close()
        return fh, _cleanup
    # 1MB buffer — default is 8KB, which causes excessive syscalls on
    # network-mounted BDC storage. Universal safe win for multi-GB CSVs.
    fh = open(entry.path, "rb", buffering=1 << 20)
    return fh, lambda: fh.close()

def _iter_chunks(entry, usecols, encoding, delim, dtype=None, na_filter=None):
    """Yield pandas DataFrame chunks from a file or zip member.

    Uses C engine. Default dtype=str (we control numeric parsing in vectorized
    steps, consistent with the old safe_float semantics). Callers may pass a
    dtype dict (e.g. {col: "float32"}) to let the C parser do the numeric
    conversion in one pass — useful for wide numeric tables like B009
    time_flow where a separate pd.to_numeric step dominates CPU.
    """
    if dtype is None:
        dtype = str
    if na_filter is None:
        # float dtype needs NaN detection; str dtype can skip it for speed.
        na_filter = not (dtype is str or dtype == str)
    fh, cleanup = _open_for_pandas(entry)
    try:
        reader = pd.read_csv(
            fh,
            encoding=encoding,
            sep=delim,
            dtype=dtype,
            usecols=usecols,
            chunksize=PD_CHUNKSIZE,
            engine="c",
            na_filter=na_filter,
            on_bad_lines="skip",
            skipinitialspace=True,
            quotechar='"',
        )
        for chunk in reader:
            yield chunk
    finally:
        cleanup()


# --- B009 resi_flow (자치구별 파일) -----------------------------------------
def _process_b009_resi_file(entry):
    """Aggregate one resi_자치구 file into {(adm8, gen, age): [wd, we]}."""
    header, enc, delim = peek_header(entry)
    if not header:
        return {}, 0, entry
    try:
        c_gender = find_col(header, "성별", "GNDR", "SEX")
        c_age    = find_col(header, "연령대", "AGE")
        c_wd     = find_col(header, "주중보행", "WKDY_FLPOP")
        c_we     = find_col(header, "주말보행", "WKND_FLPOP")
        c_admi   = find_col(header, "행정동코드", "ADMI_CD", exclude=["INFLOW", "거주지"])
    except KeyError as e:
        print(f"    [B009] skip {entry.name}: {e}", flush=True)
        return {}, 0, entry

    col_gender, col_age, col_wd, col_we, col_admi = (
        header[c_gender], header[c_age], header[c_wd], header[c_we], header[c_admi]
    )
    usecols = [col_gender, col_age, col_wd, col_we, col_admi]

    t0 = time.monotonic()
    partials = []
    count = 0
    for chunk in _iter_chunks(entry, usecols, enc, delim):
        count += len(chunk)
        a8 = chunk[col_admi].astype(str).str.strip().str.strip('"').str[:8]
        gen = _vec_gender(chunk[col_gender])
        age = _vec_age(chunk[col_age])
        wd  = pd.to_numeric(chunk[col_wd], errors="coerce").fillna(0.0)
        we  = pd.to_numeric(chunk[col_we], errors="coerce").fillna(0.0)
        tmp = pd.DataFrame({"a8": a8, "gen": gen, "age": age, "wd": wd, "we": we})
        g = tmp.groupby(["a8", "gen", "age"], sort=False, as_index=False)[["wd", "we"]].sum()
        partials.append(g)
        if count >= PROGRESS_EVERY and (count // PROGRESS_EVERY) != ((count - len(chunk)) // PROGRESS_EVERY):
            rate = count / max(time.monotonic() - t0, 1e-6)
            print(f"      [B009/{entry.name}] {count:,} rows ({rate:,.0f}/s)", flush=True)

    if partials:
        merged = pd.concat(partials, ignore_index=True)
        merged = merged.groupby(["a8", "gen", "age"], sort=False, as_index=False)[["wd", "we"]].sum()
        local = {
            (r.a8, r.gen, r.age): [float(r.wd), float(r.we)]
            for r in merged.itertuples(index=False)
        }
    else:
        local = {}

    elapsed = time.monotonic() - t0
    print(f"      [B009/{entry.name}] DONE: {count:,} rows in {elapsed:.1f}s "
          f"({count/max(elapsed,1e-6):,.0f}/s)", flush=True)
    return local, count, entry


def _agg_amount_by_demo(paths, dataset_tag, adm_keywords, stat7_to_adm8=None):
    """Generic pandas-based (adm8, gen, age) → sum(amount) aggregator.

    Used by B079-07 (keyed on 가맹점행정동) and B042 demo (keyed on
    TOT_REG_CD with stat7→adm8 translation).
    """
    paths = list(paths)
    if not paths:
        return {}, 0
    header, enc, delim = peek_header(paths[0])
    if not header:
        return {}, 0
    try:
        c_adm    = find_col(header, *adm_keywords)
        c_gender = find_col(header, "성별", "SEX_CCD", "SEX")
        c_age    = find_col(header, "연령대", "AGE_GB", "AGE")
        c_amt    = find_col(header, "카드이용금액", "AMT_CORR")
    except KeyError as e:
        print(f"    [{dataset_tag}] skip: {e}", flush=True)
        return {}, 0

    col_adm, col_gender, col_age, col_amt = (
        header[c_adm], header[c_gender], header[c_age], header[c_amt]
    )
    usecols = [col_adm, col_gender, col_age, col_amt]

    t0 = time.monotonic()
    partials = []
    count = 0
    for entry in paths:
        for chunk in _iter_chunks(entry, usecols, enc, delim):
            count += len(chunk)
            raw_adm = chunk[col_adm].astype(str).str.strip().str.strip('"')
            if stat7_to_adm8 is None:
                # adm8 = first 8 chars of 행정동코드
                a8 = raw_adm.str[:8]
            else:
                # B042: 집계구13자리 → stat7 (앞7자리) → adm8
                a8 = raw_adm.str[:7].map(stat7_to_adm8)
            gen = _vec_gender(chunk[col_gender])
            age = _vec_age(chunk[col_age])
            amt = pd.to_numeric(chunk[col_amt], errors="coerce").fillna(0.0)
            tmp = pd.DataFrame({"a8": a8, "gen": gen, "age": age, "amt": amt})
            tmp = tmp[tmp["a8"].notna() & (tmp["a8"] != "")]
            if tmp.empty:
                continue
            g = tmp.groupby(["a8", "gen", "age"], sort=False, as_index=False)["amt"].sum()
            partials.append(g)
            if count >= PROGRESS_EVERY and (count // PROGRESS_EVERY) != ((count - len(chunk)) // PROGRESS_EVERY):
                rate = count / max(time.monotonic() - t0, 1e-6)
                print(f"      [{dataset_tag}] {count:,} rows ({rate:,.0f}/s)", flush=True)

    if partials:
        merged = pd.concat(partials, ignore_index=True)
        merged = merged.groupby(["a8", "gen", "age"], sort=False, as_index=False)["amt"].sum()
        out = {(r.a8, r.gen, r.age): float(r.amt) for r in merged.itertuples(index=False)}
    else:
        out = {}
    print(f"      [{dataset_tag}] DONE: {count:,} rows in {time.monotonic()-t0:.0f}s", flush=True)
    return out, count


def _agg_b079_07(paths):
    """B079-07 서울시민 성별/연령대별 카드매출."""
    return _agg_amount_by_demo(paths, "B079-07", ("가맹점행정동", "ADSTRD"))


def _agg_b042_totreg_demo(paths, stat7_to_adm8):
    """B042 내국인(집계구) 성별연령대별 결제."""
    return _agg_amount_by_demo(paths, "B042 demo", ("TOT_REG_CD", "집계구코드"),
                               stat7_to_adm8=stat7_to_adm8)


def _agg_sido_inflow(paths, dataset_tag, adm_keywords, ctx_keys, stat7_to_adm8=None):
    """Generic inflow aggregator: (adm8) → {seoul, other}: sum(amount).

    For B079-08 (행정동 단위) and B042 inflow (집계구→adm8).
    """
    paths = list(paths)
    if not paths:
        return {}, 0
    header, enc, delim = peek_header(paths[0])
    if not header:
        return {}, 0
    try:
        c_adm  = find_col(header, *adm_keywords)
        c_sido = find_col(header, "광역시", "SIDO")
        c_amt  = find_col(header, "카드이용금액", "AMT_CORR")
    except KeyError as e:
        print(f"    [{dataset_tag}] skip: {e}", flush=True)
        return {}, 0

    col_adm, col_sido, col_amt = header[c_adm], header[c_sido], header[c_amt]
    usecols = [col_adm, col_sido, col_amt]

    t0 = time.monotonic()
    partials = []
    count = 0
    for entry in paths:
        for chunk in _iter_chunks(entry, usecols, enc, delim):
            count += len(chunk)
            raw_adm = chunk[col_adm].astype(str).str.strip().str.strip('"')
            if stat7_to_adm8 is None:
                a8 = raw_adm.str[:8]
            else:
                a8 = raw_adm.str[:7].map(stat7_to_adm8)
            amt = pd.to_numeric(chunk[col_amt], errors="coerce").fillna(0.0)
            sido = chunk[col_sido].astype(str)
            key = pd.Series("other", index=sido.index)
            key[sido.str.contains("서울", na=False)] = "seoul"
            tmp = pd.DataFrame({"a8": a8, "key": key, "amt": amt})
            tmp = tmp[tmp["a8"].notna() & (tmp["a8"] != "") & tmp["a8"].isin(ctx_keys)]
            if tmp.empty:
                continue
            g = tmp.groupby(["a8", "key"], sort=False, as_index=False)["amt"].sum()
            partials.append(g)
            if count >= PROGRESS_EVERY and (count // PROGRESS_EVERY) != ((count - len(chunk)) // PROGRESS_EVERY):
                rate = count / max(time.monotonic() - t0, 1e-6)
                print(f"      [{dataset_tag}] {count:,} rows ({rate:,.0f}/s)", flush=True)

    out = {}
    if partials:
        merged = pd.concat(partials, ignore_index=True)
        merged = merged.groupby(["a8", "key"], sort=False, as_index=False)["amt"].sum()
        for r in merged.itertuples(index=False):
            d = out.setdefault(r.a8, {"seoul": 0.0, "other": 0.0})
            d[r.key] = float(r.amt)
    print(f"      [{dataset_tag}] DONE: {count:,} rows in {time.monotonic()-t0:.0f}s", flush=True)
    return out, count


def _agg_b079_08(paths, ctx_keys):
    """B079-08 서울시민 유입지별 (행정동 단위)."""
    return _agg_sido_inflow(paths, "B079-08", ("가맹점행정동", "ADSTRD"), ctx_keys)


def _agg_b042_totreg_inflow(paths, stat7_to_adm8, ctx_keys):
    """B042 내국인(집계구) 유입지별."""
    return _agg_sido_inflow(paths, "B042 inflow", ("TOT_REG_CD", "집계구코드"),
                            ctx_keys, stat7_to_adm8=stat7_to_adm8)

# ---------------------------------------------------------------------------
# Step 2 – Joined Persona Base  (Level 1 join by [adm8, gender, age])
# ---------------------------------------------------------------------------
def merge_demographics(stat7_to_adm8):
    print("[2] Processing & Joining Demographic Data (Level 1) …")

    # ── 2-a: Telecom 29 (master base) ────────────────────────────
    h_tel, r_tel = stream_dataset("telecom_29")
    if not h_tel:
        raise RuntimeError("telecom_29 master dataset not found — cannot build base")
    tel_cols = find_tel_col_indices(h_tel)
    tel_attr_names = sorted(tel_cols.keys())
    # Key demographic columns (resilient to reordering / renaming)
    t_stat7   = find_col(h_tel, "행정동코드", "ADM_DONG_CD")
    t_gu      = find_col(h_tel, "자치구")
    t_dong    = find_col(h_tel, "행정동", exclude=["코드"])
    t_gender  = find_col(h_tel, "성별", "SEX")
    t_age     = find_col(h_tel, "연령대", "AGE")
    t_pop     = find_col(h_tel, "총인구수", "인구수", "POPULATION")

    # Accumulate weighted sums  (value * pop) per (adm8, gender, age)
    def _init():
        return {"pop": 0.0, "count": 0, **{k: 0.0 for k in tel_attr_names}}

    tel_agg = defaultdict(_init)
    gu_dong = {}

    tel_row_count = 0
    _t0 = time.monotonic()
    for r in r_tel:
        tel_row_count += 1
        if tel_row_count % PROGRESS_EVERY == 0:
            rate = tel_row_count / max(time.monotonic() - _t0, 1e-6)
            print(f"      [telecom_29] {tel_row_count:,} rows ({rate:,.0f}/s)", flush=True)
        if len(r) <= max(t_stat7, t_pop):
            continue
        st7 = r[t_stat7].strip()
        adm8 = stat7_to_adm8.get(st7)
        if not adm8:
            continue

        gu, dong = r[t_gu].strip(), r[t_dong].strip()
        gen = clean_gender(r[t_gender])
        age = clean_age(r[t_age])
        pop = safe_float(r[t_pop]) or 0.0

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

    print(f"    telecom_29: {tel_row_count} rows, "
          f"{len(tel_cols)}/{len(_TEL_METRIC_MAP)} metrics matched")

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
            "b042_card_amt": None,
        }
        for attr in tel_attr_names:
            raw = (v[attr] / pop_total) if pop_total > 0 else 0.0
            # Physical counts/durations cannot be negative
            d[attr] = max(0.0, raw)
        base_data[k] = d

    print(f"    Base personas: {len(base_data)} unique (adm8, gender, age) combos")

    # ── 2-b / 2-c / 2-d: B079-07 + B009 resi + B042 demo — 병렬 실행 ──
    # B079-07, B042 demo, B009 (자치구별 multi-file) 는 서로 독립적이므로
    # ProcessPoolExecutor 로 데이터셋 레벨 동시 실행. B009 는 내부적으로도
    # 자치구별 per-file 병렬화 돼 있으므로 중첩 Pool 을 피하기 위해 여기서는
    # per-file 작업을 top-level executor 에 submit 하는 방식으로 통합.
    b791_paths = find_dataset(RAW_INDEX, "b079_07_demo") if RAW_INDEX else []
    b092_paths = find_dataset(RAW_INDEX, "b009_resi_flow") if RAW_INDEX else []
    b042_paths = find_dataset(RAW_INDEX, "b042_totreg_demo") if RAW_INDEX else []

    b791_dict = defaultdict(float)
    b092_dict = defaultdict(lambda: [0.0, 0.0])
    b042_dict = defaultdict(float)

    n_workers = max(1, min(cpu_count(), 12))
    t_par = time.monotonic()
    print(f"    [병렬] B079-07 + B009 resi ({len(b092_paths)} files) + B042 demo "
          f"→ {n_workers} workers", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut_map = {}
        # B009: submit one job per 자치구 파일
        for p in b092_paths:
            fut = ex.submit(_process_b009_resi_file, p)
            fut_map[fut] = ("b009", p)
        # B079-07: 단일 agg 작업 (파일 여러 개면 내부에서 concat 스트리밍)
        if b791_paths:
            fut = ex.submit(_agg_b079_07, b791_paths)
            fut_map[fut] = ("b079_07", None)
        else:
            print(f"    [dataset] ⚠ no file matched for 'b079_07_demo'")
        # B042 demo: 단일 agg 작업
        if b042_paths:
            fut = ex.submit(_agg_b042_totreg_demo, b042_paths, stat7_to_adm8)
            fut_map[fut] = ("b042_demo", None)
        else:
            print(f"    [dataset] ⚠ no file matched for 'b042_totreg_demo'")

        b009_done = 0
        b009_rows = 0
        for fut in as_completed(fut_map):
            tag, p = fut_map[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"    [병렬] {tag} 실패: {e}", flush=True)
                continue
            if tag == "b009":
                part, cnt, entry = result
                for k, (wd, we) in part.items():
                    b092_dict[k][0] += wd
                    b092_dict[k][1] += we
                b009_rows += cnt
                b009_done += 1
                elapsed = time.monotonic() - t_par
                remaining = (elapsed / b009_done) * (len(b092_paths) - b009_done) if b009_done else 0
                print(f"    [B009 progress] {b009_done}/{len(b092_paths)} files "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)", flush=True)
            elif tag == "b079_07":
                part, cnt = result
                for k, v in part.items():
                    b791_dict[k] += v
                print(f"    B079-07 카드결제: {len(part)} unique keys from {cnt:,} rows "
                      f"({time.monotonic()-t_par:.0f}s)")
            elif tag == "b042_demo":
                part, cnt = result
                for k, v in part.items():
                    b042_dict[k] += v
                print(f"    B042 집계구결제: {len(part)} unique keys from {cnt:,} rows "
                      f"({time.monotonic()-t_par:.0f}s)")

    if b092_paths:
        print(f"    B009 유동인구: {len(b092_dict)} unique keys from {b009_rows:,} rows "
              f"in {time.monotonic()-t_par:.0f}s")

    # ── Apply Left Joins ─────────────────────────────────────────
    out_headers = ["adm_cd_8", "gu", "dong", "gender", "age_grp", "tel_pop"]
    out_headers.extend(tel_attr_names)
    out_headers.extend(["b079_card_amt", "b009_weekday_flow", "b009_weekend_flow", "b042_card_amt"])

    out_rows = []
    join_stats = {"b079": 0, "b009": 0, "b042": 0}

    for k, v in base_data.items():
        if k in b791_dict:
            v["b079_card_amt"] = b791_dict[k]
            join_stats["b079"] += 1
        if k in b092_dict:
            v["b009_weekday_flow"], v["b009_weekend_flow"] = b092_dict[k]
            join_stats["b009"] += 1
        if k in b042_dict:
            v["b042_card_amt"] = b042_dict[k]
            join_stats["b042"] += 1

        row = [v.get(h, "") if v.get(h) is not None else "" for h in out_headers]
        out_rows.append(row)

    total = len(base_data)
    print(f"    Join rates: b079={join_stats['b079']}/{total} ({join_stats['b079']/max(1,total)*100:.1f}%), "
          f"b009={join_stats['b009']}/{total} ({join_stats['b009']/max(1,total)*100:.1f}%), "
          f"b042={join_stats['b042']}/{total} ({join_stats['b042']/max(1,total)*100:.1f}%)")

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
    h69, r69 = stream_dataset("b069_dong_idx")
    idx_cnt_69 = 0
    if h69:
        c_adm    = find_col(h69, "ADSTRD_CD", "행정동코드")
        c_sales  = find_col(h69, "SALES", "매출")
        c_infra  = find_col(h69, "INFRASTRUCTURE", "인프라")
        c_store  = find_col(h69, "STORE", "가맹점")
        c_pop    = find_col(h69, "POPULATION", "인구")
        c_deposit= find_col(h69, "DEPOSIT", "금융")
        _t0, _cnt = time.monotonic(), 0
        for r in r69:
            _cnt += 1
            if _cnt % PROGRESS_EVERY == 0:
                print(f"      [B069] {_cnt:,} rows ({_cnt/max(time.monotonic()-_t0,1e-6):,.0f}/s)", flush=True)
            if len(r) <= c_deposit:
                continue
            a8 = r[c_adm].strip().strip('"')[:8]
            if a8 in ctx_data and "b069_sales" not in ctx_data[a8]:
                ctx_data[a8]["b069_sales"]   = safe_float(r[c_sales])
                ctx_data[a8]["b069_infra"]   = safe_float(r[c_infra])
                ctx_data[a8]["b069_store"]   = safe_float(r[c_store])
                ctx_data[a8]["b069_pop"]     = safe_float(r[c_pop])
                ctx_data[a8]["b069_finance"] = safe_float(r[c_deposit])
                idx_cnt_69 += 1
        print(f"    B069 상권지수: {idx_cnt_69}/{len(base_adms)} dongs matched")

    # ── B079-08 + B042 inflow — 병렬 실행 ─────────────────────────
    b792_paths = find_dataset(RAW_INDEX, "b079_08_inflow") if RAW_INDEX else []
    b42in_paths = find_dataset(RAW_INDEX, "b042_totreg_inflow") if RAW_INDEX else []
    ctx_keys = set(ctx_data.keys())

    t_par2 = time.monotonic()
    print(f"    [병렬] B079-08 + B042 inflow → 2 workers", flush=True)
    with ProcessPoolExecutor(max_workers=2) as ex:
        fut_map2 = {}
        if b792_paths:
            fut_map2[ex.submit(_agg_b079_08, b792_paths, ctx_keys)] = "b079_08"
        else:
            print(f"    [dataset] ⚠ no file matched for 'b079_08_inflow'")
        if b42in_paths:
            fut_map2[ex.submit(_agg_b042_totreg_inflow, b42in_paths, stat7_to_adm8, ctx_keys)] = "b042_inflow"
        else:
            print(f"    [dataset] ⚠ no file matched for 'b042_totreg_inflow'")

        for fut in as_completed(fut_map2):
            tag = fut_map2[fut]
            try:
                agg, n = fut.result()
            except Exception as e:
                print(f"    [병렬] {tag} 실패: {e}", flush=True)
                continue
            if tag == "b079_08":
                for a8, v in agg.items():
                    ctx_data[a8]["b079_2_inflow_seoul"] = v["seoul"]
                    ctx_data[a8]["b079_2_inflow_other"] = v["other"]
                print(f"    B079-08 유입지(행정동): {len(agg)}/{len(base_adms)} dongs matched "
                      f"({time.monotonic()-t_par2:.0f}s)")
            elif tag == "b042_inflow":
                for a8, v in agg.items():
                    ctx_data[a8]["b042_inflow_seoul"] = v["seoul"]
                    ctx_data[a8]["b042_inflow_other"] = v["other"]
                print(f"    B042 유입지(집계구): {len(agg)}/{len(base_adms)} dongs matched "
                      f"({time.monotonic()-t_par2:.0f}s)")

    # ── Write ─────────────────────────────────────────────────────
    out_headers = [
        "adm_cd_8",
        "b069_sales", "b069_infra", "b069_store", "b069_pop", "b069_finance",
        "b079_2_inflow_seoul", "b079_2_inflow_other",
        "b042_inflow_seoul", "b042_inflow_other",
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
    """Copy reference mapping files only.

    이전에는 10GB 규모의 unjoined_*.csv 사본도 생성했지만, 디스크 공간
    부족(450GB 원본 + 출력) 으로 제거했다. analyze_stats.py 는 data/raw 의
    원본을 file_discovery 인덱스로 직접 읽는다.
    """
    print("[4] Copying Reference Files …")

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
    global SAMPLE_DIR, OUT_DIR, RAW_INDEX

    if use_synthetic:
        SAMPLE_DIR = PROJECT_ROOT / "data/synthetic"
        OUT_DIR = PROJECT_ROOT / "output/synthetic"
    else:
        SAMPLE_DIR = PROJECT_ROOT / "data/raw"
        OUT_DIR = PROJECT_ROOT / "output/original"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mode_str = "SYNTHETIC (full_synth)" if use_synthetic else "ORIGINAL (raw_samples)"
    print("=" * 60)
    print(f"  PIPELINE: {mode_str}")
    print("=" * 60)

    # One-shot: unzip everything under SAMPLE_DIR and build a file index.
    # Subsequent reads go through file_discovery.find_dataset().
    RAW_INDEX = prepare_raw_dir(SAMPLE_DIR)

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
