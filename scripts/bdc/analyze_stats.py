"""
analyze_stats.py
======================
Reads joined & unjoined datasets from `output/synthetic/` (or `output/original/`)
and produces comprehensive statistical profiles for LLM-based agent generation.
All input files are read from the output directory — no direct `data/raw/` access.

Output (JSON) → output/stats/
  1. agent_profiles.json       – Per (adm8, gender, age) detailed statistics
  2. dong_context.json         – Per-dong commercial/infrastructure context
  3. workplace_flow.json       – Residence→workplace probability distributions
  4. global_distributions.json – Temporal, consumption, movement patterns
  5. agent_allocation.json     – How many agents to generate per combo (for 3000 target)
  7. workplace_population.json – Per-dong workplace population by gender×age

Usage:
  python analyze_stats.py                  # default: synthetic pipeline
  python analyze_stats.py --source original
"""

import csv
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 이 파일은 scripts/bdc/ 안에 있음 — 프로젝트 루트는 두 단계 위
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from collections import defaultdict

import numpy as np
import pandas as pd

from file_discovery import (
    smart_read,
    smart_read_many,
    smart_stream_many,
    find_col,
    find_dataset,
    prepare_raw_dir,
    peek_header,
)
from preprocess_join import _iter_chunks, _vec_gender, _vec_age, PD_CHUNKSIZE

# ---------------------------------------------------------------------------
# Raw-data access (replaces the old output/unjoined_*.csv copies)
# ---------------------------------------------------------------------------
# Previously preprocess_join.py produced ~380GB of unjoined_*.csv copies in
# output/.  With 450GB raw + 50GB free that is infeasible, so the copies are
# gone and analyze_stats reads data/raw directly via file_discovery.
_RAW_INDEX = None

def _ensure_raw_index():
    global _RAW_INDEX
    if _RAW_INDEX is None:
        _RAW_INDEX = prepare_raw_dir(PROJECT_ROOT / "data/raw", verbose=False)
    return _RAW_INDEX

def stream_raw(dataset_id):
    """Return (header, row_generator) for a BDC dataset, from data/raw.

    Replaces csv_read(out_dir / "unjoined_*.csv").  Rows are streamed so
    memory stays O(1) even for multi-GB datasets.  Returns ([], iter([])) if
    the dataset is not present (callers fall back gracefully).
    """
    idx = _ensure_raw_index()
    paths = find_dataset(idx, dataset_id)
    if not paths:
        print(f"    [raw] ⚠ no file matched for '{dataset_id}'")
        return [], iter([])
    return smart_stream_many(paths)


def stream_chunks_raw(dataset_id, col_specs):
    """Stream a raw BDC dataset as pandas DataFrame chunks (C parser).

    col_specs: dict of {logical_name: dict(keywords=[...], exclude=[...]?, required=bool?)}
    Each chunk is yielded with columns renamed to logical names.  Optional
    (required=False) logical names that can't be matched are silently dropped
    from the yielded chunk (caller checks presence).  Missing required names
    cause the file to be skipped with a warning.
    """
    idx = _ensure_raw_index()
    paths = find_dataset(idx, dataset_id)
    if not paths:
        print(f"    [raw] ⚠ no file matched for '{dataset_id}'")
        return
    for entry in paths:
        header, enc, delim = peek_header(entry)
        if not header:
            continue
        col_map = {}
        skip = False
        for logical, spec in col_specs.items():
            kws = spec.get("keywords", [])
            excl = spec.get("exclude")
            req = spec.get("required", True)
            idx_col = find_col(header, *kws, required=False, exclude=excl)
            if idx_col is None:
                if req:
                    print(f"    [raw] ⚠ {entry.name}: required col '{logical}' not found")
                    skip = True
                    break
                continue
            col_map[header[idx_col]] = logical
        if skip:
            continue
        usecols = list(col_map.keys())
        if not usecols:
            continue
        for chunk in _iter_chunks(entry, usecols, enc, delim):
            chunk = chunk.rename(columns=col_map)
            yield chunk

# ---------------------------------------------------------------------------
# Parallel per-file workers (top-level for ProcessPoolExecutor picklability)
# ---------------------------------------------------------------------------
# Cap workers at 4 so peak memory stays in budget on the 16GB BDC machine
# (each worker holds ~1 PD_CHUNKSIZE worth of pandas DataFrame ≈ 250-500MB).
_PARALLEL_CAP = 8

def _agg_workers(n_files):
    return max(1, min(n_files, os.cpu_count() or 4, _PARALLEL_CAP))


def _mem_aware_workers(n_files, mb_per_worker=500, headroom_mb=2048):
    """Cap worker count so each worker has ~mb_per_worker RAM budget.

    Priority:
      1. env ANALYZE_STATS_MEM_GB  (manual override — total RAM available for the job)
      2. psutil.virtual_memory().available  (live available RAM)
      3. fallback _PARALLEL_CAP

    After the budget cap, also cap by CPU, file count, and _PARALLEL_CAP.
    headroom_mb is left for the parent process + OS.
    """
    try:
        mem_gb_env = os.environ.get("ANALYZE_STATS_MEM_GB")
        if mem_gb_env:
            avail_mb = float(mem_gb_env) * 1024.0
        else:
            import psutil  # type: ignore
            avail_mb = psutil.virtual_memory().available / (1024.0 * 1024.0)
        budget_mb = max(1024.0, avail_mb - headroom_mb)
        by_mem = max(1, int(budget_mb // mb_per_worker))
    except Exception:
        by_mem = _PARALLEL_CAP
    cpu = os.cpu_count() or 4
    return max(1, min(n_files, cpu, _PARALLEL_CAP, by_mem))


def _worker_b079_07(entry):
    """Per-file aggregation of b079_07_demo. Returns (by_key, by_demo, n_rows)."""
    import time as _t
    _t0 = _t.monotonic()
    header, enc, delim = peek_header(entry)
    if not header:
        return {}, {}, 0
    col_map = {}
    for log, kws in [
        ("adm",    ["가맹점행정동", "ADSTRD"]),
        ("gender", ["성별", "SEX"]),
        ("age",    ["연령대", "AGE"]),
        ("ind",    ["업종대분류", "업종", "UPJONG"]),
        ("amt",    ["카드이용금액", "AMT_CORR"]),
    ]:
        i = find_col(header, *kws, required=False)
        if i is None:
            return {}, {}, 0
        col_map[header[i]] = log
    usecols = list(col_map.keys())
    by_key = defaultdict(float)
    by_demo = defaultdict(float)
    n_rows = 0
    for chunk in _iter_chunks(entry, usecols, enc, delim):
        chunk = chunk.rename(columns=col_map)
        n_rows += len(chunk)
        chunk["adm"] = chunk["adm"].astype(str).str.strip().str[:8]
        chunk["gender"] = _vec_gender(chunk["gender"])
        chunk["age"] = _vec_age(chunk["age"])
        chunk["ind"] = chunk["ind"].astype(str).str.strip()
        chunk["amt"] = pd.to_numeric(chunk["amt"], errors="coerce").fillna(0.0)
        chunk = chunk[(chunk["gender"] != "U") & (chunk["age"] != "U") & (chunk["amt"] > 0)]
        if chunk.empty:
            continue
        for k, v in chunk.groupby(["adm", "gender", "age", "ind"], sort=False)["amt"].sum().items():
            by_key[k] += float(v)
        for k, v in chunk.groupby(["gender", "age", "ind"], sort=False)["amt"].sum().items():
            by_demo[k] += float(v)
    print(f"      [b079_07/{entry.name}] DONE {n_rows:,} rows in "
          f"{_t.monotonic()-_t0:.1f}s", flush=True)
    return dict(by_key), dict(by_demo), n_rows


def _worker_b079_02(entry):
    """Per-file aggregation of b079_02_timeslot.
    Returns (dong_total, dong_wd, dong_we, weekday_days, weekend_days,
             unique_dates, hourly_totals, dong_hourly,
             hourly_wd_totals, hourly_we_totals,
             dong_hourly_wd, dong_hourly_we).
    """
    import datetime, time as _t
    _t0 = _t.monotonic()
    header, enc, delim = peek_header(entry)
    empty = ({}, {}, {}, set(), set(), set(), {}, {}, {}, {}, {}, {})
    if not header:
        return empty
    tail_cols = header[3:]
    logical_actuals = {}
    for log, kws in [
        ("date", ["기준일자", "기준년월", "일자", "YMD"]),
        ("dong", ["행정동코드", "고객행정동", "ADSTRD"]),
        ("ind",  ["업종대분류", "업종", "UPJONG"]),
        ("amt",  ["카드이용금액", "금액", "AMT"]),
    ]:
        i = find_col(header, *kws, required=False)
        if i is not None:
            logical_actuals[log] = header[i]
    needed = set(logical_actuals.values()) | set(tail_cols)
    usecols = [c for c in header if c in needed]
    if not usecols:
        return empty

    dong_wd = defaultdict(lambda: defaultdict(float))
    dong_we = defaultdict(lambda: defaultdict(float))
    dong_total = defaultdict(lambda: defaultdict(float))
    weekday_days = set()
    weekend_days = set()
    unique_dates = set()
    hourly_totals = defaultdict(float)
    hourly_wd_totals = defaultdict(float)
    hourly_we_totals = defaultdict(float)
    dong_hourly = defaultdict(lambda: defaultdict(float))
    dong_hourly_wd = defaultdict(lambda: defaultdict(float))
    dong_hourly_we = defaultdict(lambda: defaultdict(float))

    has_dong_col = "dong" in logical_actuals
    has_date_col = "date" in logical_actuals

    def _is_we(ds):
        try:
            return datetime.datetime.strptime(ds[:8], "%Y%m%d").weekday() >= 5
        except ValueError:
            return False

    for chunk in _iter_chunks(entry, usecols, enc, delim):
        # 동별 시간대 합산을 위해 dong 컬럼을 먼저 준비
        dong_col_series = None
        if has_dong_col and logical_actuals["dong"] in chunk.columns:
            dong_col_series = chunk[logical_actuals["dong"]].astype(str).str.strip().str[:8]

        # 평일/주말 마스크 (tail_cols 집계에서 재사용)
        is_we_mask = None
        if has_date_col and logical_actuals["date"] in chunk.columns:
            date_s = chunk[logical_actuals["date"]].astype(str).str.strip().str.replace(".0", "", regex=False)
            is_we_mask = date_s.map(_is_we)

        for c_name in tail_cols:
            if c_name in chunk.columns:
                vals = pd.to_numeric(chunk[c_name], errors="coerce").fillna(0.0)
                s = float(vals.sum(skipna=True))
                if s:
                    hourly_totals[c_name] += s
                # 평일/주말 분리 (date 컬럼 있는 경우만)
                if is_we_mask is not None:
                    s_we = float(vals[is_we_mask].sum(skipna=True))
                    s_wd = float(vals[~is_we_mask].sum(skipna=True))
                    if s_wd:
                        hourly_wd_totals[c_name] += s_wd
                    if s_we:
                        hourly_we_totals[c_name] += s_we
                # 동별 시간대 합산
                if dong_col_series is not None:
                    for dong_code, amt in vals.groupby(dong_col_series).sum().items():
                        if dong_code and amt > 0:
                            dong_hourly[dong_code][c_name] += float(amt)
                    if is_we_mask is not None:
                        wd_vals = vals.where(~is_we_mask, 0.0)
                        we_vals = vals.where(is_we_mask, 0.0)
                        for dong_code, amt in wd_vals.groupby(dong_col_series).sum().items():
                            if dong_code and amt > 0:
                                dong_hourly_wd[dong_code][c_name] += float(amt)
                        for dong_code, amt in we_vals.groupby(dong_col_series).sum().items():
                            if dong_code and amt > 0:
                                dong_hourly_we[dong_code][c_name] += float(amt)

        if not all(logical_actuals.get(k) in chunk.columns
                   for k in ("date", "dong", "ind", "amt")):
            continue
        c = chunk[[logical_actuals["date"], logical_actuals["dong"],
                   logical_actuals["ind"], logical_actuals["amt"]]].copy()
        c.columns = ["date", "dong", "ind", "amt"]
        c["date"] = c["date"].astype(str).str.strip().str.replace(".0", "", regex=False)
        unique_dates.update(c["date"].unique())
        c["dong"] = c["dong"].astype(str).str.strip().str[:8]
        c["ind"]  = c["ind"].astype(str).str.strip()
        c["amt"]  = pd.to_numeric(c["amt"], errors="coerce").fillna(0.0)
        c = c[(c["dong"] != "") & (c["ind"] != "") & (c["amt"] > 0)]
        if c.empty:
            continue
        for (d, i), v in c.groupby(["dong", "ind"], sort=False)["amt"].sum().items():
            dong_total[d][i] += float(v)
        c["_is_we"] = c["date"].map(_is_we)
        wd_chunk = c[~c["_is_we"]]
        we_chunk = c[c["_is_we"]]
        if not wd_chunk.empty:
            weekday_days.update(wd_chunk["date"].unique())
            for (d, i), v in wd_chunk.groupby(["dong", "ind"], sort=False)["amt"].sum().items():
                dong_wd[d][i] += float(v)
        if not we_chunk.empty:
            weekend_days.update(we_chunk["date"].unique())
            for (d, i), v in we_chunk.groupby(["dong", "ind"], sort=False)["amt"].sum().items():
                dong_we[d][i] += float(v)
    print(f"      [b079_02/{entry.name}] DONE in {_t.monotonic()-_t0:.1f}s "
          f"(wd_days={len(weekday_days)}, we_days={len(weekend_days)})", flush=True)
    return ({d: dict(v) for d, v in dong_total.items()},
            {d: dict(v) for d, v in dong_wd.items()},
            {d: dict(v) for d, v in dong_we.items()},
            weekday_days, weekend_days, unique_dates, dict(hourly_totals),
            {d: dict(v) for d, v in dong_hourly.items()},
            dict(hourly_wd_totals), dict(hourly_we_totals),
            {d: dict(v) for d, v in dong_hourly_wd.items()},
            {d: dict(v) for d, v in dong_hourly_we.items()})


def _worker_workplace_flow(entry):
    """Per-file aggregation for workplace_flow.
    Returns (flow {res:{obs:wd}}, residence_dongs_set, has_resi_bool, any_chunk_bool).
    """
    import time as _t
    _t0 = _t.monotonic()
    header, enc, delim = peek_header(entry)
    if not header:
        return {}, set(), False, False
    col_map = {}
    has_resi = False
    for log, kws, excl, req in [
        ("resi", ["INFLOW_ADMIN_CD", "거주지"], None, False),
        ("obs",  ["행정동코드", "ADMI_CD"], ["INFLOW", "거주지"], True),
        ("wd",   ["주중보행", "WKDY_FLPOP"], None, True),
    ]:
        i = find_col(header, *kws, required=False, exclude=excl)
        if i is None:
            if req:
                return {}, set(), False, False
            continue
        col_map[header[i]] = log
        if log == "resi":
            has_resi = True
    usecols = list(col_map.keys())
    if not usecols:
        return {}, set(), False, False

    flow = defaultdict(lambda: defaultdict(float))
    residence_dongs = set()
    any_chunk = False
    for chunk in _iter_chunks(entry, usecols, enc, delim):
        any_chunk = True
        chunk = chunk.rename(columns=col_map)
        if "resi" not in chunk.columns:
            continue
        chunk["resi"] = chunk["resi"].astype(str).str.strip().str[:8]
        chunk["obs"]  = chunk["obs"].astype(str).str.strip().str[:8]
        chunk["wd"]   = pd.to_numeric(chunk["wd"], errors="coerce").fillna(0.0)
        nonempty = chunk[chunk["resi"] != ""]
        if not nonempty.empty:
            residence_dongs.update(nonempty["resi"].unique())
        valid = chunk[(chunk["resi"] != "") & (chunk["obs"] != "") & (chunk["wd"] > 0)]
        if valid.empty:
            continue
        for (res, obs), v in valid.groupby(["resi", "obs"], sort=False)["wd"].sum().items():
            flow[res][obs] += float(v)
    print(f"      [wflow/{entry.name}] DONE in {_t.monotonic()-_t0:.1f}s", flush=True)
    return ({k: dict(v) for k, v in flow.items()}, residence_dongs, has_resi, any_chunk)


def _clean_bt(s: pd.Series) -> pd.Series:
    """Strip whitespace, double-quotes, and backticks from a string Series.

    BDC B042/B069 files wrap values in backticks (e.g. `1234567`) which are
    not handled by pandas' quotechar.  Must be called before numeric parsing,
    gender/age mapping, or code lookups.
    """
    return s.astype(str).str.strip().str.strip('"').str.strip('`').str.strip()


def _worker_consumption_detail(args):
    """Per-file aggregation for consumption_detail.
    args is (entry, stat7_to_adm8, sb_to_name) — single-arg form for executor.map.
    Returns (agg {(a8,gen,age,daytype,ind): amt}, weekday_dates, weekend_dates,
             n_rows, skipped, any_chunk_bool).
    """
    import datetime, time as _t
    _t0 = _t.monotonic()
    entry, stat7_to_adm8, sb_to_name = args
    header, enc, delim = peek_header(entry)
    if not header:
        return {}, set(), set(), 0, 0, False
    col_map = {}
    for log, kws in [
        ("cen",    ["TOT_REG_CD", "집계구코드", "가맹점집계구"]),
        ("sb",     ["SB_CODE", "업종코드", "내국인업종"]),
        ("date",   ["TS_YMD", "일별", "기준일"]),
        ("gender", ["SEX_CCD", "성별"]),
        ("age",    ["AGE_GB", "연령대"]),
        ("amt",    ["AMT_CORR", "카드이용금액"]),
    ]:
        i = find_col(header, *kws, required=False)
        if i is None:
            return {}, set(), set(), 0, 0, False
        col_map[header[i]] = log
    usecols = list(col_map.keys())

    def _is_we(ds):
        try:
            return datetime.datetime.strptime(ds[:8], "%Y%m%d").weekday() >= 5
        except ValueError:
            return False

    weekday_dates = set()
    weekend_dates = set()
    agg = defaultdict(float)
    n_rows = 0
    skipped = 0
    any_chunk = False

    for chunk in _iter_chunks(entry, usecols, enc, delim):
        any_chunk = True
        chunk = chunk.rename(columns=col_map)
        n_rows += len(chunk)
        chunk["cen"] = _clean_bt(chunk["cen"])
        chunk["a8"] = chunk["cen"].str[:7].map(stat7_to_adm8)
        skipped += int(chunk["a8"].isna().sum())
        chunk = chunk[chunk["a8"].notna()]
        if chunk.empty:
            continue
        chunk["gender"] = _vec_gender(_clean_bt(chunk["gender"]))
        chunk["age"] = _vec_age(_clean_bt(chunk["age"]))
        chunk["amt"] = pd.to_numeric(_clean_bt(chunk["amt"]), errors="coerce").fillna(0.0)
        chunk = chunk[(chunk["gender"] != "U") & (chunk["age"] != "U") & (chunk["amt"] > 0)]
        if chunk.empty:
            continue
        chunk["sb_code"] = _clean_bt(chunk["sb"]).str.upper()
        chunk["ind_name"] = chunk["sb_code"].map(sb_to_name).fillna(chunk["sb_code"])
        chunk["date_s"] = _clean_bt(chunk["date"]).str[:8]
        chunk["_is_we"] = chunk["date_s"].map(_is_we)
        chunk["daytype"] = chunk["_is_we"].map({True: "weekend", False: "weekday"})
        weekday_dates.update(chunk.loc[~chunk["_is_we"], "date_s"].unique())
        weekend_dates.update(chunk.loc[chunk["_is_we"], "date_s"].unique())
        for (a8, gen, age, daytype, ind), v in chunk.groupby(
            ["a8", "gender", "age", "daytype", "ind_name"], sort=False
        )["amt"].sum().items():
            agg[(a8, gen, age, daytype, ind)] += float(v)
    print(f"      [cons/{entry.name}] DONE {n_rows:,} rows in "
          f"{_t.monotonic()-_t0:.1f}s", flush=True)
    return dict(agg), weekday_dates, weekend_dates, n_rows, skipped, any_chunk


# ---------------------------------------------------------------------------
# Cached pandas aggregators (avoid re-reading the same raw dataset twice)
# ---------------------------------------------------------------------------
_B079_07_CACHE = None   # ({(adm8,gen,age,ind): amt}, {(gen,age,ind): amt})
_B079_02_CACHE = None   # (dong_data, hourly_totals, hourly_wd_totals, hourly_we_totals)

def _compute_b079_07():
    """Parallel per-file aggregation of b079_07_demo.

    Returns two dicts:
      by_key:  (adm8, gen, age, ind) -> total amt  (for convert_profiles_to_export)
      by_demo: (gen, age, ind)       -> total amt  (for global_distributions 4-e)
    """
    paths = find_dataset(_ensure_raw_index(), "b079_07_demo")
    if not paths:
        print("    [raw] ⚠ no file matched for 'b079_07_demo'")
        return {}, {}

    by_key = defaultdict(float)
    by_demo = defaultdict(float)

    def _merge(pk, pd_):
        for k, v in pk.items():
            by_key[k] += v
        for k, v in pd_.items():
            by_demo[k] += v

    if len(paths) == 1:
        pk, pd_, _ = _worker_b079_07(paths[0])
        _merge(pk, pd_)
    else:
        with ProcessPoolExecutor(max_workers=_agg_workers(len(paths))) as ex:
            for f in as_completed([ex.submit(_worker_b079_07, e) for e in paths]):
                pk, pd_, _ = f.result()
                _merge(pk, pd_)
    return dict(by_key), dict(by_demo)


def _get_b079_07():
    global _B079_07_CACHE
    if _B079_07_CACHE is None:
        _B079_07_CACHE = _compute_b079_07()
    return _B079_07_CACHE


def _compute_b079_02():
    """Parallel per-file aggregation of b079_02_timeslot.

    Returns:
      dong_data: {dong: {
          "wd_ind": {ind: amt}, "we_ind": {ind: amt},
          "total_ind": {ind: amt}, "n_wd": int, "n_we": int,
          "has_multi": bool,
          "hourly":    {col: amt}, "hourly_wd": {col: amt}, "hourly_we": {col: amt}
      }}
      hourly_totals:    {col_name: total}   — legacy total (wd+we mixed)
      hourly_wd_totals: {col_name: weekday total}
      hourly_we_totals: {col_name: weekend total}
    """
    paths = find_dataset(_ensure_raw_index(), "b079_02_timeslot")
    if not paths:
        print("    [raw] ⚠ no file matched for 'b079_02_timeslot'")
        return {}, {}, {}, {}

    dong_wd = defaultdict(lambda: defaultdict(float))
    dong_we = defaultdict(lambda: defaultdict(float))
    dong_total = defaultdict(lambda: defaultdict(float))
    weekday_days = set()
    weekend_days = set()
    unique_dates = set()
    hourly_totals = defaultdict(float)
    hourly_wd_totals = defaultdict(float)
    hourly_we_totals = defaultdict(float)
    dong_hourly = defaultdict(lambda: defaultdict(float))
    dong_hourly_wd = defaultdict(lambda: defaultdict(float))
    dong_hourly_we = defaultdict(lambda: defaultdict(float))

    def _merge(p_total, p_wd, p_we, p_wdays, p_wedays, p_udates, p_hourly,
               p_dong_hourly, p_hourly_wd, p_hourly_we,
               p_dong_hourly_wd, p_dong_hourly_we):
        for d, m in p_total.items():
            for i, v in m.items():
                dong_total[d][i] += v
        for d, m in p_wd.items():
            for i, v in m.items():
                dong_wd[d][i] += v
        for d, m in p_we.items():
            for i, v in m.items():
                dong_we[d][i] += v
        weekday_days.update(p_wdays)
        weekend_days.update(p_wedays)
        unique_dates.update(p_udates)
        for k, v in p_hourly.items():
            hourly_totals[k] += v
        for k, v in p_hourly_wd.items():
            hourly_wd_totals[k] += v
        for k, v in p_hourly_we.items():
            hourly_we_totals[k] += v
        for d, slots in p_dong_hourly.items():
            for slot, v in slots.items():
                dong_hourly[d][slot] += v
        for d, slots in p_dong_hourly_wd.items():
            for slot, v in slots.items():
                dong_hourly_wd[d][slot] += v
        for d, slots in p_dong_hourly_we.items():
            for slot, v in slots.items():
                dong_hourly_we[d][slot] += v

    if len(paths) == 1:
        _merge(*_worker_b079_02(paths[0]))
    else:
        with ProcessPoolExecutor(max_workers=_agg_workers(len(paths))) as ex:
            for f in as_completed([ex.submit(_worker_b079_02, e) for e in paths]):
                _merge(*f.result())

    has_multi = len(unique_dates) > 1
    all_dongs = (set(dong_total) | set(dong_wd) | set(dong_we) |
                 set(dong_hourly) | set(dong_hourly_wd) | set(dong_hourly_we))
    dong_data = {}
    for d in all_dongs:
        dong_data[d] = {
            "total_ind": dict(dong_total.get(d, {})),
            "wd_ind":    dict(dong_wd.get(d, {})),
            "we_ind":    dict(dong_we.get(d, {})),
            "n_wd":      max(1, len(weekday_days)),
            "n_we":      max(1, len(weekend_days)),
            "has_multi": has_multi,
            "hourly":    dict(dong_hourly.get(d, {})),
            "hourly_wd": dict(dong_hourly_wd.get(d, {})),
            "hourly_we": dict(dong_hourly_we.get(d, {})),
        }
    return (dong_data, dict(hourly_totals),
            dict(hourly_wd_totals), dict(hourly_we_totals))


def _get_b079_02():
    global _B079_02_CACHE
    if _B079_02_CACHE is None:
        _B079_02_CACHE = _compute_b079_02()
    return _B079_02_CACHE


# ---------------------------------------------------------------------------
# Byte-range chunk parallelism for 4-b/4-c/4-d/4-f (plain CSV files only).
#
# B009/B063/B042/B078 are single-file datasets on BDC, so file-level
# parallelism leaves 3 of 4 cores idle. Splitting one file into N disjoint
# byte ranges at line boundaries lets N worker processes parse concurrently
# — CSV parsing + numeric conversion scale ~linearly with core count.
#
# Fallback: for zip members we can't random-seek, so we run a single
# streaming worker (_worker_4X) as before.
# ---------------------------------------------------------------------------
class _BoundedReader:
    """File-like wrapper that stops reading at `end` byte offset.

    Passed to pd.read_csv so each worker parses only its slice without
    loading the full range into RAM.
    """
    def __init__(self, path, start, end):
        self.fh = open(path, 'rb', buffering=1 << 20)
        self.fh.seek(start)
        self.end = end

    def read(self, size=-1):
        remaining = self.end - self.fh.tell()
        if remaining <= 0:
            return b''
        if size < 0 or size > remaining:
            return self.fh.read(remaining)
        return self.fh.read(size)

    def readline(self, size=-1):
        remaining = self.end - self.fh.tell()
        if remaining <= 0:
            return b''
        if size < 0 or size > remaining:
            return self.fh.readline(remaining + 1)
        return self.fh.readline(size)

    def readable(self):
        return True

    def close(self):
        self.fh.close()


def _split_byte_ranges(path, n_workers):
    """Return [(start, end), ...] ranges covering the data rows (header
    excluded), each starting at a line boundary so no row is split.
    """
    path = str(path)
    size = os.path.getsize(path)
    with open(path, 'rb', buffering=1 << 20) as f:
        f.readline()  # skip header
        header_end = f.tell()
    # Don't bother splitting tiny files
    if n_workers <= 1 or (size - header_end) < 50 * 1024 * 1024:
        return [(header_end, size)]

    ranges = []
    with open(path, 'rb', buffering=1 << 20) as f:
        data_size = size - header_end
        chunk_size = data_size // n_workers
        start = header_end
        for i in range(n_workers - 1):
            tentative = header_end + (i + 1) * chunk_size
            if tentative >= size:
                break
            f.seek(tentative)
            f.readline()  # advance to next line boundary
            actual_end = f.tell()
            if actual_end > start:
                ranges.append((start, actual_end))
            start = actual_end
        if start < size:
            ranges.append((start, size))
    return ranges


def _iter_range_chunks(path, start, end, header, usecols, encoding, delim,
                       dtype=None, na_filter=True):
    """Yield pandas chunks from byte range [start, end) of a plain CSV.
    Caller must have split ranges at line boundaries after the header row.
    """
    if dtype is None:
        dtype = str
    bounded = _BoundedReader(str(path), start, end)
    try:
        reader = pd.read_csv(
            bounded,
            encoding=encoding,
            sep=delim,
            header=None,
            names=header,
            usecols=usecols,
            dtype=dtype,
            na_filter=na_filter,
            chunksize=PD_CHUNKSIZE,
            engine="c",
            on_bad_lines="skip",
            skipinitialspace=True,
            quotechar='"',
        )
        for chunk in reader:
            yield chunk
    finally:
        bounded.close()


def _chunk_workers(n_ranges, mb_per_worker):
    """How many concurrent range workers to run. Bounded by CPU, range count,
    and RAM budget (ANALYZE_STATS_MEM_GB or psutil.available)."""
    try:
        mem_gb_env = os.environ.get("ANALYZE_STATS_MEM_GB")
        if mem_gb_env:
            avail_mb = float(mem_gb_env) * 1024.0
        else:
            import psutil  # type: ignore
            avail_mb = psutil.virtual_memory().available / (1024.0 * 1024.0)
        budget_mb = max(1024.0, avail_mb - 2048.0)
        by_mem = max(1, int(budget_mb // mb_per_worker))
    except Exception:
        by_mem = _PARALLEL_CAP
    cpu = os.cpu_count() or 4
    return max(1, min(n_ranges, cpu, _PARALLEL_CAP, by_mem))


# ---------------------------------------------------------------------------
# Core chunk-processing logic — shared between entry-based workers (zip
# fallback) and range-based workers (byte-range parallel). Takes an iterable
# of pandas chunks and returns the substage's partial aggregate.
# ---------------------------------------------------------------------------
def _process_4b_chunks(chunks_iter, time_col_name, demo_cols):
    col_to_demo = {c: f"{g}_{a}" for c, g, a in demo_cols}
    time_dist = defaultdict(lambda: defaultdict(float))
    n_rows = 0
    for chunk in chunks_iter:
        n_rows += len(chunk)
        demo_names = [c for c in col_to_demo if c in chunk.columns]
        if not demo_names:
            continue
        chunk[demo_names] = chunk[demo_names].fillna(0.0)
        chunk[time_col_name] = (
            chunk[time_col_name].astype(str).str.strip().astype("category")
        )
        sub = chunk.groupby(time_col_name, sort=False, observed=True)[demo_names].sum()
        if sub.empty:
            continue
        for col_name in demo_names:
            col = sub[col_name]
            if not col.any():
                continue
            demo_key = col_to_demo[col_name]
            target = time_dist[demo_key]
            for t, v in col.items():
                if v:
                    target[t] += float(v)
    return {k: dict(v) for k, v in time_dist.items()}, n_rows


def _process_4c_chunks(chunks_iter, purpose_col, count_col):
    purpose_counts = defaultdict(float)
    n_rows = 0
    for chunk in chunks_iter:
        n_rows += len(chunk)
        chunk[purpose_col] = chunk[purpose_col].astype(str).str.strip().astype("category")
        if count_col and count_col in chunk.columns:
            chunk["_cnt"] = chunk[count_col].fillna(1.0)
            chunk.loc[chunk["_cnt"] == 0, "_cnt"] = 1.0
        else:
            chunk["_cnt"] = 1.0
        g = chunk.groupby(purpose_col, sort=False, observed=True)["_cnt"].sum()
        for p, v in g.items():
            purpose_counts[p] += float(v)
    return dict(purpose_counts), n_rows


def _process_4d_chunks(chunks_iter, c_gen_name, c_age_name, value_cols):
    row_counts = defaultdict(lambda: defaultdict(int))
    n_rows = 0
    for chunk in chunks_iter:
        n_rows += len(chunk)
        # B063 gender/age values are backtick-wrapped (like B042/B069); strip
        # before mapping or _vec_gender/_vec_age fall through to "U" / default.
        chunk["_gen"] = _vec_gender(_clean_bt(chunk[c_gen_name]))
        chunk["_age"] = _vec_age(_clean_bt(chunk[c_age_name]))
        chunk["_demo"] = (chunk["_gen"] + "_" + chunk["_age"]).astype("category")
        present_vals = [c for c in value_cols if c in chunk.columns]
        if not present_vals:
            continue
        counts_df = chunk.groupby("_demo", sort=False, observed=True)[present_vals].count()
        for demo_key, row in counts_df.iterrows():
            bucket = row_counts[str(demo_key)]
            for col_name, n in row.items():
                if n:
                    # B063 headers are backtick-wrapped too, so the raw column
                    # name would serialize as "`카드이용건수계`". Strip to get
                    # clean JSON keys like "카드이용건수계".
                    clean_key = str(col_name).strip().strip('"').strip('`').strip()
                    bucket[clean_key] += int(n)
    return {k: dict(v) for k, v in row_counts.items()}, n_rows


def _process_4f_chunks(chunks_iter, d_name, a_name):
    weekend_names = {"토요일", "일요일", "토", "일", "Saturday", "Sunday"}
    wd_amt = 0.0; we_amt = 0.0
    wd_cnt = 0;  we_cnt = 0
    n_rows = 0
    for chunk in chunks_iter:
        n_rows += len(chunk)
        # B042 day-of-week values are backtick-wrapped; without stripping,
        # d.isin({"토요일",...}) would match nothing and all rows fall into
        # weekday, skewing the ratio.
        d = _clean_bt(chunk[d_name])
        valid = d != ""
        if not valid.any():
            continue
        d = d[valid].astype("category")
        # B042 values are backtick-wrapped (`1234567`), so we read a_name as
        # str to avoid float cast errors, then clean + convert here.
        a_raw = chunk.loc[valid, a_name]
        if a_raw.dtype == object:
            a = pd.to_numeric(_clean_bt(a_raw), errors="coerce").fillna(0.0)
        else:
            a = a_raw.fillna(0.0)
        is_we = d.isin(weekend_names)
        we_amt += float(a[is_we].sum())
        we_cnt += int(is_we.sum())
        wd_mask = ~is_we
        wd_amt += float(a[wd_mask].sum())
        wd_cnt += int(wd_mask.sum())
    return wd_amt, we_amt, wd_cnt, we_cnt, n_rows


# ---------------------------------------------------------------------------
# Workers for global_distributions sub-stages (4-b, 4-c, 4-d, 4-f).
# Entry-based: supports plain files AND zip members (single-stream fallback).
# Range-based (*_range): plain files only, used for byte-range parallelism.
# Module-top-level for Windows spawn pickling.
# ---------------------------------------------------------------------------
def _worker_4b(args):
    """Entry-based 4-b worker (zip-safe, single-stream).

    Used as the zip-member fallback — for plain files, main() dispatches
    to _worker_4b_range for byte-range parallelism.
    """
    entry, time_col_name, demo_cols, usecols = args
    import time as _t
    _t0 = _t.monotonic()
    eh, e_enc, e_delim = peek_header(entry)
    if not eh:
        return {}, entry.name, 0
    present = [c for c in usecols if c in eh]
    if time_col_name not in present:
        return {}, entry.name, 0
    demo_col_names = [c for c, _, _ in demo_cols]
    dtype_map = {c: "float32" for c in demo_col_names if c in present}
    dtype_map[time_col_name] = "str"
    chunks_iter = _iter_chunks(
        entry, present, e_enc, e_delim,
        dtype=dtype_map, na_filter=True,
    )
    result, n_rows = _process_4b_chunks(chunks_iter, time_col_name, demo_cols)
    print(f"      [4-b/{entry.name}] DONE {n_rows:,} rows in "
          f"{_t.monotonic()-_t0:.1f}s", flush=True)
    return result, entry.name, n_rows


def _worker_4b_range(args):
    """Range-based 4-b worker (plain file byte-range)."""
    path, start, end, header, encoding, delim, time_col_name, demo_cols, usecols = args
    import time as _t
    _t0 = _t.monotonic()
    demo_col_names = [c for c, _, _ in demo_cols]
    dtype_map = {c: "float32" for c in demo_col_names if c in usecols}
    dtype_map[time_col_name] = "str"
    chunks_iter = _iter_range_chunks(
        path, start, end, header, usecols, encoding, delim,
        dtype=dtype_map, na_filter=True,
    )
    result, n_rows = _process_4b_chunks(chunks_iter, time_col_name, demo_cols)
    print(f"      [4-b/range {start // (1024*1024)}-{end // (1024*1024)}MB] "
          f"DONE {n_rows:,} rows in {_t.monotonic()-_t0:.1f}s", flush=True)
    return result, n_rows


def _worker_4c(args):
    """Entry-based 4-c (zip fallback)."""
    entry, purpose_col, count_col, usecols = args
    import time as _t
    _t0 = _t.monotonic()
    eh, e_enc, e_delim = peek_header(entry)
    if not eh or purpose_col not in eh:
        return {}, entry.name, 0
    present = [c for c in usecols if c in eh]
    dtype_map = {purpose_col: "str"}
    if count_col and count_col in present:
        dtype_map[count_col] = "float32"
    chunks_iter = _iter_chunks(
        entry, present, e_enc, e_delim,
        dtype=dtype_map, na_filter=True,
    )
    result, n_rows = _process_4c_chunks(chunks_iter, purpose_col, count_col)
    print(f"      [4-c/{entry.name}] DONE {n_rows:,} rows in "
          f"{_t.monotonic()-_t0:.1f}s", flush=True)
    return result, entry.name, n_rows


def _worker_4c_range(args):
    """Range-based 4-c (plain file byte-range)."""
    path, start, end, header, encoding, delim, purpose_col, count_col, usecols = args
    import time as _t
    _t0 = _t.monotonic()
    dtype_map = {purpose_col: "str"}
    if count_col and count_col in usecols:
        dtype_map[count_col] = "float32"
    chunks_iter = _iter_range_chunks(
        path, start, end, header, usecols, encoding, delim,
        dtype=dtype_map, na_filter=True,
    )
    result, n_rows = _process_4c_chunks(chunks_iter, purpose_col, count_col)
    print(f"      [4-c/range {start // (1024*1024)}-{end // (1024*1024)}MB] "
          f"DONE {n_rows:,} rows in {_t.monotonic()-_t0:.1f}s", flush=True)
    return result, n_rows


def _worker_4d(args):
    """Entry-based 4-d (zip fallback)."""
    entry, c_gen_name, c_age_name, value_cols, usecols = args
    import time as _t
    _t0 = _t.monotonic()
    eh, e_enc, e_delim = peek_header(entry)
    if not eh:
        return {}, entry.name, 0
    present = [c for c in usecols if c in eh]
    if c_gen_name not in present or c_age_name not in present:
        return {}, entry.name, 0
    # B063 values are backtick-wrapped (e.g. `1038400`) like B042/B069, so
    # float32 parsing fails. _process_4d_chunks only needs .count() on
    # value columns, so read as str — na_filter still turns empty cells
    # into NaN and backtick-wrapped cells are counted as populated.
    dtype_map = {c: "str" for c in value_cols if c in present}
    dtype_map[c_gen_name] = "str"
    dtype_map[c_age_name] = "str"
    chunks_iter = _iter_chunks(
        entry, present, e_enc, e_delim,
        dtype=dtype_map, na_filter=True,
    )
    result, n_rows = _process_4d_chunks(chunks_iter, c_gen_name, c_age_name, value_cols)
    print(f"      [4-d/{entry.name}] DONE {n_rows:,} rows in "
          f"{_t.monotonic()-_t0:.1f}s", flush=True)
    return result, entry.name, n_rows


def _worker_4d_range(args):
    """Range-based 4-d (plain file byte-range)."""
    path, start, end, header, encoding, delim, c_gen_name, c_age_name, value_cols, usecols = args
    import time as _t
    _t0 = _t.monotonic()
    # See _worker_4d comment: B063 values are backtick-wrapped; str dtype
    # sidesteps float cast errors while .count() still works correctly.
    dtype_map = {c: "str" for c in value_cols if c in usecols}
    dtype_map[c_gen_name] = "str"
    dtype_map[c_age_name] = "str"
    chunks_iter = _iter_range_chunks(
        path, start, end, header, usecols, encoding, delim,
        dtype=dtype_map, na_filter=True,
    )
    result, n_rows = _process_4d_chunks(chunks_iter, c_gen_name, c_age_name, value_cols)
    print(f"      [4-d/range {start // (1024*1024)}-{end // (1024*1024)}MB] "
          f"DONE {n_rows:,} rows in {_t.monotonic()-_t0:.1f}s", flush=True)
    return result, n_rows


def _worker_4f(args):
    """Entry-based 4-f (zip fallback)."""
    entry, d_name, a_name = args
    import time as _t
    _t0 = _t.monotonic()
    eh, e_enc, e_delim = peek_header(entry)
    if not eh or d_name not in eh or a_name not in eh:
        return 0.0, 0.0, 0, 0, entry.name, 0
    # B042 values are backtick-wrapped; read a_name as str and let
    # _process_4f_chunks strip backticks before numeric conversion.
    dtype_map = {d_name: "str", a_name: "str"}
    chunks_iter = _iter_chunks(
        entry, [d_name, a_name], e_enc, e_delim,
        dtype=dtype_map, na_filter=True,
    )
    wd_a, we_a, wd_c, we_c, n_rows = _process_4f_chunks(chunks_iter, d_name, a_name)
    print(f"      [4-f/{entry.name}] DONE {n_rows:,} rows in "
          f"{_t.monotonic()-_t0:.1f}s", flush=True)
    return wd_a, we_a, wd_c, we_c, entry.name, n_rows


def _worker_4f_range(args):
    """Range-based 4-f (plain file byte-range)."""
    path, start, end, header, encoding, delim, d_name, a_name = args
    import time as _t
    _t0 = _t.monotonic()
    # See _worker_4f: B042 is backtick-wrapped, so read a_name as str.
    dtype_map = {d_name: "str", a_name: "str"}
    usecols = [d_name, a_name]
    chunks_iter = _iter_range_chunks(
        path, start, end, header, usecols, encoding, delim,
        dtype=dtype_map, na_filter=True,
    )
    wd_a, we_a, wd_c, we_c, n_rows = _process_4f_chunks(chunks_iter, d_name, a_name)
    print(f"      [4-f/range {start // (1024*1024)}-{end // (1024*1024)}MB] "
          f"DONE {n_rows:,} rows in {_t.monotonic()-_t0:.1f}s", flush=True)
    return wd_a, we_a, wd_c, we_c, n_rows


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_AGENTS = 15000

def parse_args():
    """
    Flags:
      [original|synthetic]        source pipeline (default: synthetic)
      --number4 | --stage 4       run stage 4 → 5 → 6 → 7 (resume from stage 4).
                                  assumes stages 1/2/3 (+3b/3c/3d) already done.
      --number5 | --stage 5       only run stage 5 (agent_allocation)
      --number6 | --stage 6       only run stage 6 (aggregate_stats)
      --number7 | --stage 7       only run stage 7 (convert_profiles_to_export)
      --stage=N | --number=N      shorthand form
      --stage 1,4                 (future) run multiple stages
    """
    import re as _re
    _shorthand = _re.compile(r"^--(?:number|stage)(\d+)$")
    source = "synthetic"
    only_stage = None
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        m = _shorthand.match(arg)
        if m:
            only_stage = int(m.group(1))
        elif arg.startswith("--stage=") or arg.startswith("--number="):
            val = arg.split("=", 1)[1]
            if val.isdigit():
                only_stage = int(val)
        elif arg in ("--stage", "--number") and i + 1 < len(argv):
            val = argv[i + 1]
            if val.isdigit():
                only_stage = int(val)
            i += 1
        elif arg in ("--source", "-s"):
            pass
        elif arg in ("original", "synthetic"):
            source = arg
        i += 1
    return source, only_stage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def csv_read(path, enc=None):
    """Backward-compatible wrapper delegating to file_discovery.smart_read."""
    return smart_read(Path(path))

def compute_stats(values):
    """Compute mean, std, min, max for a list of floats."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
    n = len(values)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    return {
        "mean": round(m, 4),
        "std": round(math.sqrt(var), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "count": n,
    }

def compute_decile(values_dict):
    """Assign 10-quantile ranks (1=lowest 10%, 10=highest 10%) using
    numpy.percentile edges so each decile spans a real quantile range.

    Returns (assignments, boundaries):
      - assignments: {key: decile_rank(1~10)}
      - boundaries: {"1": [quantile_lo, quantile_hi], ...}
        (decile d spans the (d-1)*10% to d*10% quantile edges)
    """
    if not values_dict:
        return {}, {}
    keys = list(values_dict.keys())
    vals = np.array([values_dict[k] for k in keys], dtype=float)
    n = len(vals)

    # Quantile edges: 0%, 10%, 20%, ..., 100% (11 points, 10 intervals)
    edges = np.percentile(vals, np.arange(0, 101, 10))

    # np.searchsorted: right=True gives 1-indexed decile; clip 1~10
    ranks = np.searchsorted(edges[1:-1], vals, side="right") + 1
    ranks = np.clip(ranks, 1, 10).astype(int)
    assignments = {k: int(r) for k, r in zip(keys, ranks)}

    boundaries = {}
    for d in range(1, 11):
        lo = float(edges[d - 1])
        hi = float(edges[d])
        boundaries[str(d)] = [round(lo, 2), round(hi, 2)]
    return assignments, boundaries

def clean_gender(g):
    g = str(g).strip()
    if g in ("남", "M", "1", "male"):
        return "M"
    if g in ("여", "F", "2", "female"):
        return "F"
    return "U"

def clean_age(s):
    s = str(s).strip()
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
    s = s.replace("세", "")
    if s.isdigit():
        a = int(s)
        if len(s) == 4:
            a = int(s[:2])
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

# ---------------------------------------------------------------------------
# 1. Agent Profiles – Per (adm8, gender, age) statistics
# ---------------------------------------------------------------------------
def build_agent_profiles(out_dir):
    print("[1] Building agent profiles …")
    jpath = out_dir / "joined_persona_base.csv"
    h, rows = csv_read(jpath)
    if not h:
        print(f"    [warn] joined_persona_base.csv not found or empty — "
              f"looked for: {jpath.resolve()} (exists={jpath.exists()})",
              flush=True)
        return {}, {}

    profiles = {}
    pop_weights = {}

    # Identify column groups
    tel_cols = [c for c in h if c.startswith("tel_") and c != "tel_pop"]
    join_cols = ["b079_card_amt", "b009_weekday_flow", "b009_weekend_flow", "b042_card_amt"]

    for row in rows:
        if len(row) < len(h):
            continue
        d = dict(zip(h, row))
        key = f"{d['adm_cd_8']}_{d['gender']}_{d['age_grp']}"
        pop = safe_float(d.get("tel_pop")) or 0.0

        profile = {
            "location": {
                "adm_cd_8": d["adm_cd_8"],
                "gu": d.get("gu", ""),
                "dong": d.get("dong", ""),
            },
            "demographics": {
                "gender": d["gender"],
                "age_grp": d["age_grp"],
                "population": pop,
            },
            "telecom": {},
            "consumption": {},
            "mobility": {},
        }

        # Telecom metrics
        for col in tel_cols:
            v = safe_float(d.get(col))
            if v is not None:
                profile["telecom"][col] = round(v, 4)

        # Joined consumption/mobility (raw 값만; per-capita 계산은 convert 단계에서)
        for col in join_cols:
            v = safe_float(d.get(col))
            if v is not None:
                if "flow" in col:
                    profile["mobility"][col] = round(v, 4)
                else:
                    profile["consumption"][col] = round(v, 2)

        profiles[key] = profile
        pop_weights[key] = pop

    print(f"    {len(profiles)} unique profiles loaded")
    return profiles, pop_weights

# ---------------------------------------------------------------------------
# 1-b. Convert agent profiles to export format (응용집계)
# ---------------------------------------------------------------------------
def convert_profiles_to_export(profiles, consumption_detail, out_dir):
    """Convert raw agent profiles to 응용집계 format:
    - consumption: weekday_spending_level, weekend_spending_level, industry_ratio
    - mobility:    mobility_level (10분위), weekend_weekday_ratio

    spending_level 은 tel_pop 기반이 아닌 **b042 ÷ b009 유동인구** 기반
    (평일/주말 별도). 값은 이미 `build_consumption_detail()` 에서 계산된
    것을 주입한다.
    """
    print("[1b] Converting profiles to export format (응용집계) …")

    # --- Step 1: Mobility 10분위 (총 유동인구 기준; raw b009 값) ---
    mobility_vals = {}
    mobility_raw = {}
    for key, p in profiles.items():
        wd = p["mobility"].get("b009_weekday_flow", 0) or 0
        we = p["mobility"].get("b009_weekend_flow", 0) or 0
        if wd + we > 0:
            mobility_vals[key] = wd + we
            mobility_raw[key] = (wd, we)
    mobility_deciles, mobility_boundaries = compute_decile(mobility_vals)

    # --- Step 2: Industry ratio per (adm8, gender, age) from B079-07 ---
    industry_ratios = {}
    by_key, _ = _get_b079_07()
    ind_totals = defaultdict(lambda: defaultdict(float))
    for (adm8, gen, age, ind), amt in by_key.items():
        ind_totals[f"{adm8}_{gen}_{age}"][ind] += amt
    for k, industries in ind_totals.items():
        total = sum(industries.values())
        if total > 0:
            industry_ratios[k] = {
                ind: round(v / total, 4)
                for ind, v in sorted(industries.items(), key=lambda x: -x[1])
            }

    print(f"    Mobility deciles: {len(mobility_deciles)}, "
          f"Industry ratios: {len(industry_ratios)} groups")

    # --- Step 3: Replace raw values in profiles ---
    for key, p in profiles.items():
        new_consumption = {}
        det = consumption_detail.get(key) if consumption_detail else None
        if det:
            if "weekday_spending_level" in det:
                new_consumption["weekday_spending_level"] = det["weekday_spending_level"]
            if "weekend_spending_level" in det:
                new_consumption["weekend_spending_level"] = det["weekend_spending_level"]
        if key in industry_ratios:
            new_consumption["industry_ratio"] = industry_ratios[key]
        p["consumption"] = new_consumption

        new_mobility = {}
        if key in mobility_deciles:
            new_mobility["mobility_level"] = mobility_deciles[key]
        if key in mobility_raw:
            wd, we = mobility_raw[key]
            if wd > 0:
                new_mobility["weekend_weekday_ratio"] = round(we / wd, 4)
        p["mobility"] = new_mobility

    return profiles

# ---------------------------------------------------------------------------
# 2. Dong Context
# ---------------------------------------------------------------------------
def build_dong_context(out_dir):
    print("[2] Building dong context …")
    h, rows = csv_read(out_dir / "joined_dong_context.csv")
    if not h:
        return {}

    context = {}
    for row in rows:
        d = dict(zip(h, row))
        adm8 = d["adm_cd_8"]
        ctx = {}

        # b069 fields — keep as-is (already exportable data)
        for col in h:
            if col.startswith("b069_"):
                v = safe_float(d.get(col))
                if v is not None:
                    ctx[col] = round(v, 4)

        # b079_2 inflow — convert to 서울유입비율 (응용집계)
        seoul_079 = safe_float(d.get("b079_2_inflow_seoul")) or 0
        other_079 = safe_float(d.get("b079_2_inflow_other")) or 0
        total_079 = seoul_079 + other_079
        if total_079 > 0:
            ctx["b079_seoul_inflow_ratio"] = round(seoul_079 / total_079, 4)

        # b042 inflow — convert to 서울유입비율 (응용집계)
        seoul_042 = safe_float(d.get("b042_inflow_seoul")) or 0
        other_042 = safe_float(d.get("b042_inflow_other")) or 0
        total_042 = seoul_042 + other_042
        if total_042 > 0:
            ctx["b042_seoul_inflow_ratio"] = round(seoul_042 / total_042, 4)

        context[adm8] = ctx

    print(f"    {len(context)} dongs with context data")
    return context

# ---------------------------------------------------------------------------
# 3. Workplace Flow – Residence→Workplace probability from KT data
# ---------------------------------------------------------------------------
def build_workplace_flow(out_dir, workplace_pop=None):
    print("[3] Building workplace flow probabilities …")

    paths = find_dataset(_ensure_raw_index(), "b009_resi_flow")
    if not paths:
        print("    [raw] ⚠ no file matched for 'b009_resi_flow'")
        return {}

    flow = defaultdict(lambda: defaultdict(float))
    all_residence_dongs = set()
    any_chunk = False
    has_resi = False

    def _merge(p_flow, p_resi_dongs, p_has_resi, p_any):
        nonlocal any_chunk, has_resi
        if p_any:
            any_chunk = True
        if p_has_resi:
            has_resi = True
        all_residence_dongs.update(p_resi_dongs)
        for res, dests in p_flow.items():
            for obs, v in dests.items():
                flow[res][obs] += v

    print(f"    Files: {len(paths)} (parallel workers={_agg_workers(len(paths))})")
    if len(paths) == 1:
        _merge(*_worker_workplace_flow(paths[0]))
    else:
        with ProcessPoolExecutor(max_workers=_agg_workers(len(paths))) as ex:
            for f in as_completed([ex.submit(_worker_workplace_flow, e) for e in paths]):
                _merge(*f.result())

    if not any_chunk:
        return {}
    if not has_resi:
        print("    ⚠ 거주지코드 column not found, skipping workplace flow")
        return {}

    # Convert to probability distributions (top 10 per dong)
    workplace_flow = {}
    for res_dong, destinations in flow.items():
        total = sum(destinations.values())
        if total <= 0:
            continue
        sorted_dests = sorted(destinations.items(), key=lambda x: -x[1])
        top = sorted_dests[:10]
        # Include "기타" for remaining probability
        top_total = sum(v for _, v in top)
        other_pct = round((total - top_total) / total, 4) if total > top_total else 0

        probs = []
        for dong, cnt in top:
            probs.append({"dong": dong, "probability": round(cnt / total, 4)})
        if other_pct > 0.001:
            probs.append({"dong": "기타", "probability": other_pct})

        workplace_flow[res_dong] = probs

    kt_count = len(workplace_flow)
    print(f"    {kt_count} residence dongs with KT flow data")

    # Fallback: fill missing dongs using workplace_population distribution
    if workplace_pop:
        # Build a global workplace probability distribution from workplace_pop
        global_wp = {}
        for dong_code, info in workplace_pop.items():
            t = info.get("total", 0)
            if t > 0:
                global_wp[dong_code] = t
        total_wp = sum(global_wp.values())

        if total_wp > 0:
            # Sort by total descending, take top 20 as fallback destinations
            sorted_wp = sorted(global_wp.items(), key=lambda x: -x[1])
            top_wp = sorted_wp[:20]
            top_wp_total = sum(v for _, v in top_wp)
            other_wp = round((total_wp - top_wp_total) / total_wp, 4)

            fallback_probs = []
            for dong_code, cnt in top_wp:
                fallback_probs.append({
                    "dong": dong_code,
                    "probability": round(cnt / total_wp, 4)
                })
            if other_wp > 0.001:
                fallback_probs.append({"dong": "기타", "probability": other_wp})

            # Collect all known dongs (from KT + workplace_pop keys)
            all_dongs = all_residence_dongs | set(workplace_pop.keys())
            fallback_count = 0
            for dong in all_dongs:
                if dong not in workplace_flow:
                    workplace_flow[dong] = fallback_probs
                    fallback_count += 1

            print(f"    {fallback_count} dongs filled with workplace_pop fallback")

    print(f"    {len(workplace_flow)} total dongs with workplace flow")
    return workplace_flow


# ---------------------------------------------------------------------------
# 3-b. Workplace Population – Per-dong workers by gender×age
# ---------------------------------------------------------------------------
def build_workplace_population(out_dir):
    """Read 서울시 상권분석서비스(직장인구-행정동).csv and extract per-dong
    workplace population by gender × age group.
    Uses only the latest quarter."""
    print("[3b] Building workplace population …")
    # workplace_pop 은 소규모(~1600행)라 한 번에 로드해도 됨.
    idx = _ensure_raw_index()
    paths = find_dataset(idx, "workplace_pop")
    if not paths:
        print("    [raw] ⚠ no file matched for 'workplace_pop'")
        return {}
    h, rows = smart_read_many(paths)
    if not h:
        return {}

    c_quarter   = find_col(h, "년분기", "기준", required=False) or 0
    c_dong      = find_col(h, "행정동_코드", "행정동코드", exclude=["명"])
    c_dong_name = find_col(h, "행정동_코드_명", "행정동명", required=False)
    c_total     = find_col(h, "총_직장_인구", "총직장인구")
    c_male      = find_col(h, "남성_직장_인구", "남성직장인구")
    c_female    = find_col(h, "여성_직장_인구", "여성직장인구")

    # Find gender×age columns dynamically by pattern matching
    age_map = {"10": "20세미만", "20": "20대", "30": "30대",
               "40": "40대", "50": "50대", "60": "60대"}
    ga_cols = []  # list of (col_index, "M_20대" or "F_30대" etc.)
    for i, col in enumerate(h):
        norm = col.replace(" ", "").replace("_", "")
        for age_key, age_label in age_map.items():
            if f"남성연령대{age_key}" in norm and "직장" in norm:
                ga_cols.append((i, f"M_{age_label}"))
            elif f"여성연령대{age_key}" in norm and "직장" in norm:
                ga_cols.append((i, f"F_{age_label}"))

    # Find the latest quarter
    quarters = set()
    for r in rows:
        if r and len(r) > c_quarter:
            quarters.add(r[c_quarter].strip())
    if not quarters:
        return {}
    latest_q = max(quarters)
    print(f"    Using latest quarter: {latest_q}")

    result = {}
    for r in rows:
        if len(r) <= max(c_total, c_male, c_female):
            continue
        if r[c_quarter].strip() != latest_q:
            continue

        dong_code = r[c_dong].strip()[:8]
        dong_name = r[c_dong_name].strip() if c_dong_name is not None and len(r) > c_dong_name else ""
        total_pop = safe_float(r[c_total]) or 0

        by_gender_age = {}
        for col_idx, ga_key in ga_cols:
            if col_idx < len(r):
                v = safe_float(r[col_idx]) or 0
                if v > 0:
                    by_gender_age[ga_key] = int(v)

        result[dong_code] = {
            "dong_name": dong_name,
            "total": int(total_pop),
            "male": int(safe_float(r[c_male]) or 0),
            "female": int(safe_float(r[c_female]) or 0),
            "by_gender_age": by_gender_age,
        }

    print(f"    {len(result)} dongs with workplace population data")
    return result


# ---------------------------------------------------------------------------
# 3-c. Consumption Detail – (adm8, gender, age, weekday/weekend, industry) avg
# ---------------------------------------------------------------------------
def build_consumption_detail(out_dir):
    """Read 내국인(집계구) 성별연령대별 and compute daily average spending
    per (행정동, 성별, 나이대, 평일/주말, 업종).

    Uses TS_YMD to derive weekday/weekend, maps 집계구→행정동 via stat7→adm8,
    and translates SB codes to industry names.

    Output structure per key:
      "11110530_M_20대": {
        "weekday": { "한식": 1234.5, "커피전문점": 567.8, ... },
        "weekend": { "한식": 890.1, ... },
        "weekday_total": 5000.0,
        "weekend_total": 3000.0
      }
    """
    import datetime

    print("[3c-detail] Building consumption detail (adm8×gender×age×daytype×industry) …")

    # Load stat7→adm8 mapping (from ref file in output)
    h_map, r_map = csv_read(out_dir / "ref_mopas_nso.csv")
    stat7_to_adm8 = {}
    for r in r_map:
        if len(r) > 2 and r[2].strip() and r[0].strip():
            stat7_to_adm8[r[2].strip()] = r[0].strip()[:8]

    # Load SB code→name mapping (from ref file in output)
    sb_to_name = {}
    h_sb, r_sb = csv_read(out_dir / "ref_industry_code_63.csv")
    if h_sb:
        for r in r_sb:
            if len(r) >= 4:
                code = r[3].strip().upper()
                name = r[2].strip()
                sb_to_name[code] = name

    # Read 내국인(집계구) 성별연령대별 — parallel per-file pandas aggregation
    paths = find_dataset(_ensure_raw_index(), "b042_totreg_demo")
    if not paths:
        print("    ⚠ File not found or empty")
        return {}

    weekday_dates = set()
    weekend_dates = set()
    agg = defaultdict(float)
    n_rows = 0
    skipped = 0
    any_chunk = False

    def _merge(p_agg, p_wd, p_we, p_n, p_skip, p_any):
        nonlocal n_rows, skipped, any_chunk
        if p_any:
            any_chunk = True
        n_rows += p_n
        skipped += p_skip
        weekday_dates.update(p_wd)
        weekend_dates.update(p_we)
        for k, v in p_agg.items():
            agg[k] += v

    print(f"    Files: {len(paths)} (parallel workers={_agg_workers(len(paths))})")
    args_list = [(e, stat7_to_adm8, sb_to_name) for e in paths]
    if len(paths) == 1:
        _merge(*_worker_consumption_detail(args_list[0]))
    else:
        with ProcessPoolExecutor(max_workers=_agg_workers(len(paths))) as ex:
            for f in as_completed([ex.submit(_worker_consumption_detail, a) for a in args_list]):
                _merge(*f.result())

    if not any_chunk:
        print("    ⚠ File not found or empty")
        return {}

    n_wd = max(1, len(weekday_dates))
    n_we = max(1, len(weekend_dates))

    print(f"    Rows processed: {n_rows}, skipped (no mapping): {skipped}")
    print(f"    Unique weekday dates: {n_wd}, weekend dates: {n_we}")
    print(f"    SB codes mapped: {len(sb_to_name)}")

    # Build result grouped by (adm8_gender_age) — 응용집계: 비율 + 범주
    # First pass: compute totals per (key, daytype)
    group_daytype_totals = defaultdict(lambda: {"weekday": 0.0, "weekend": 0.0})
    group_industry = defaultdict(lambda: {"weekday": defaultdict(float), "weekend": defaultdict(float)})

    for (a8, gen, age, daytype, ind_name), total_amt in agg.items():
        key = f"{a8}_{gen}_{age}"
        group_daytype_totals[key][daytype] += total_amt
        group_industry[key][daytype][ind_name] += total_amt

    # joined_persona_base.csv 에서 b009 유동인구 로드 — per-capita 분모.
    # tel_pop 은 "해당 동 통신 가입자" 이므로 실제 소비 주체(그 동에 있던 사람)
    # 추정에 부적절. b009_weekday_flow / b009_weekend_flow (12개월 누적 인·일)
    # 를 분모로 쓰면 분자(b042 주중/주말 총 소비액, 12개월 누적) 와 기간 상쇄 →
    # "1 인·일당 평균 소비액" 이 된다.
    h_pb, r_pb = csv_read(out_dir / "joined_persona_base.csv")
    b009_wd_flow = {}  # key → b009_weekday_flow (12개월 누적)
    b009_we_flow = {}
    if h_pb:
        idx_adm = h_pb.index("adm_cd_8")
        idx_gen = h_pb.index("gender")
        idx_age = h_pb.index("age_grp")
        idx_wd  = h_pb.index("b009_weekday_flow") if "b009_weekday_flow" in h_pb else None
        idx_we  = h_pb.index("b009_weekend_flow") if "b009_weekend_flow" in h_pb else None
        for row in r_pb:
            if len(row) < len(h_pb):
                continue
            k = f"{row[idx_adm]}_{row[idx_gen]}_{row[idx_age]}"
            if idx_wd is not None:
                v = safe_float(row[idx_wd])
                if v and v > 0:
                    b009_wd_flow[k] = v
            if idx_we is not None:
                v = safe_float(row[idx_we])
                if v and v > 0:
                    b009_we_flow[k] = v

    # Second pass: convert to ratios + per-capita
    result = {}
    weekday_pp_for_decile = {}  # key → weekday 1인·일당 소비
    weekend_pp_for_decile = {}

    for key in group_industry:
        entry = {}

        for daytype in ["weekday", "weekend"]:
            dt_total = group_daytype_totals[key][daytype]
            if dt_total > 0:
                ratios = {
                    ind: round(amt / dt_total, 4)
                    for ind, amt in group_industry[key][daytype].items()
                }
                entry[f"{daytype}_ratio"] = dict(
                    sorted(ratios.items(), key=lambda x: -x[1])
                )
            else:
                entry[f"{daytype}_ratio"] = {}

        wd_total = group_daytype_totals[key]["weekday"]
        we_total = group_daytype_totals[key]["weekend"]

        # 일평균 소비비 (n_wd, n_we 는 모든 키 공통) → 이 비율만 비교에 의미 있음
        if wd_total > 0 and we_total > 0:
            wd_avg = wd_total / n_wd
            we_avg = we_total / n_we
            entry["weekend_weekday_spending_ratio"] = round(we_avg / wd_avg, 4)

        # 1 인·일당 평균 소비액 — 분자/분모 둘 다 12개월 누적이라 기간 상쇄
        wd_flow = b009_wd_flow.get(key)
        we_flow = b009_we_flow.get(key)
        if wd_total > 0 and wd_flow:
            weekday_pp_for_decile[key] = wd_total / wd_flow
        if we_total > 0 and we_flow:
            weekend_pp_for_decile[key] = we_total / we_flow

        result[key] = entry

    # Compute weekday/weekend spending_level (10분위) separately
    wd_deciles, wd_bounds = compute_decile(weekday_pp_for_decile)
    we_deciles, we_bounds = compute_decile(weekend_pp_for_decile)
    for key, d in wd_deciles.items():
        result[key]["weekday_spending_level"] = d
    for key, d in we_deciles.items():
        result[key]["weekend_spending_level"] = d
    print(f"    per-capita 분모 적용: weekday {len(weekday_pp_for_decile)} keys, "
          f"weekend {len(weekend_pp_for_decile)} keys")

    # Add metadata
    result["_meta"] = {
        "n_weekday_dates": n_wd,
        "n_weekend_dates": n_we,
        "description": "업종별 소비비중·per-capita 평일/주말 분위 "
                       "(행정동×성별×나이대) — 응용집계",
        "unit": "ratio=비율(합=1.0), spending_level=10분위(1~10)",
        "per_capita_source": "분자 b042 card_amt (주중/주말 날짜 실제분리) ÷ 분모 b009_weekday/weekend_flow",
    }

    print(f"    {len(result) - 1} unique (adm8, gender, age) groups with consumption detail")
    return result


# ---------------------------------------------------------------------------
# 3-d. Dong Consumption – Per-dong industry spending, weekday vs weekend
# ---------------------------------------------------------------------------
def build_dong_consumption(out_dir):
    """Read unjoined_time_sales.csv (2.서울시민의 일별 시간대별) and compute
    per-dong industry spending breakdown with weekday/weekend split and
    per-dong hourly consumption ratio.

    Output structure per dong:
      { "industry_ratio": { "한식": 0.35, ... },
        "weekend_to_weekday": 0.8,
        "industry_weekday_ratio": { ... },
        "industry_weekend_ratio": { ... },
        "hourly_consumption_ratio": { "00~02시": 0.02, "12~14시": 0.18, ... } }
    """
    print("[3c] Building dong consumption patterns …")
    dong_data, _, _, _ = _get_b079_02()  # cached single-pass pandas aggregate
    if not dong_data:
        return {}

    # Check if multi-day from first dong (all share same flag)
    first = next(iter(dong_data.values()))
    has_multi_dates = first.get("has_multi", False)
    n_wd = first.get("n_wd", 1)
    n_we = first.get("n_we", 1)
    print(f"    Multi-day: {has_multi_dates}")

    result = {}
    hourly_count = 0
    for dong, info in dong_data.items():
        industries = info["total_ind"]
        total = sum(industries.values())
        if total <= 0:
            continue

        entry = {
            "industry_ratio": {
                k: round(v / total, 4)
                for k, v in sorted(industries.items(), key=lambda x: -x[1])
            },
        }

        if has_multi_dates and info["wd_ind"] and info["we_ind"]:
            wd_ind = info["wd_ind"]
            we_ind = info["we_ind"]
            wd_total = sum(wd_ind.values())
            we_total = sum(we_ind.values())
            wd_avg = wd_total / n_wd
            we_avg = we_total / n_we

            entry["weekend_to_weekday"] = round(we_avg / max(1, wd_avg), 4)

            if wd_total > 0:
                entry["industry_weekday_ratio"] = {
                    k: round(v / wd_total, 4)
                    for k, v in sorted(wd_ind.items(), key=lambda x: -x[1])
                }
            if we_total > 0:
                entry["industry_weekend_ratio"] = {
                    k: round(v / we_total, 4)
                    for k, v in sorted(we_ind.items(), key=lambda x: -x[1])
                }

        # 동별 시간대 소비 비율 (평일/주말 분리 + 기존 total 유지)
        hourly_raw = info.get("hourly", {})
        hourly_total = sum(hourly_raw.values())
        if hourly_total > 0:
            entry["hourly_consumption_ratio"] = {
                k: round(v / hourly_total, 4)
                for k, v in sorted(hourly_raw.items())
            }
            hourly_count += 1
        hourly_wd_raw = info.get("hourly_wd", {})
        hourly_wd_total = sum(hourly_wd_raw.values())
        if hourly_wd_total > 0:
            entry["hourly_consumption_weekday_ratio"] = {
                k: round(v / hourly_wd_total, 4)
                for k, v in sorted(hourly_wd_raw.items())
            }
        hourly_we_raw = info.get("hourly_we", {})
        hourly_we_total = sum(hourly_we_raw.values())
        if hourly_we_total > 0:
            entry["hourly_consumption_weekend_ratio"] = {
                k: round(v / hourly_we_total, 4)
                for k, v in sorted(hourly_we_raw.items())
            }

        result[dong] = entry

    print(f"    {len(result)} dongs with consumption patterns")
    print(f"    {hourly_count} dongs with hourly consumption ratio")
    if has_multi_dates:
        print(f"    Weekday days: {n_wd}, Weekend days: {n_we}")
    return result

# ---------------------------------------------------------------------------
# 4. Global Distributions (from unjoined/reference data)
# ---------------------------------------------------------------------------
def build_global_distributions(out_dir, save_cb=None):
    """Build global distribution JSON with sub-stage incremental saves.

    save_cb: optional callable (dict) -> None invoked after every sub-stage.
             Use it to write the partial `global_distributions.json` so that
             a mid-run kill doesn't lose hours of work.
    """
    import time as _t
    t_all = _t.monotonic()
    print("[4] Building global distributions …", flush=True)

    # Limit BLAS threads in spawned worker processes to prevent oversubscription
    # (4 workers × N BLAS threads each → cache thrashing). Parent's numpy is
    # already loaded, so this only affects children launched after this point.
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_k] = "1"

    distributions = {}

    # Resume from a prior partial run. build_global_distributions writes the
    # accumulating JSON after every sub-stage (_flush), so if a crash happens
    # mid-run, the next invocation can pick up from the first missing key.
    # Set ANALYZE_STATS_FORCE_REBUILD=1 to ignore the cache and start fresh.
    resume_path = PROJECT_ROOT / "output/stats/global_distributions.json"
    if resume_path.exists() and os.environ.get("ANALYZE_STATS_FORCE_REBUILD") != "1":
        try:
            with open(resume_path, "r", encoding="utf-8") as _rf:
                distributions = json.load(_rf)
            print(f"    [resume] loaded {resume_path} — keys: "
                  f"{sorted(distributions.keys())}", flush=True)
        except Exception as e:
            print(f"    [resume] WARN could not load {resume_path}: {e} — "
                  f"starting fresh", flush=True)
            distributions = {}

    def _need(key):
        """Return True if the sub-stage owning `key` still needs to run."""
        if key in distributions:
            print(f"    [resume] skip — '{key}' already present", flush=True)
            return False
        return True

    def _flush(tag):
        if save_cb is not None:
            try:
                save_cb(distributions)
            except Exception as e:
                print(f"    [incremental save] WARN {tag}: {e}", flush=True)

    # ---- 4-a: Hourly totals from cached b079_02 ----------------------------
    # b079_02 is recomputed from raw in a fresh process (the cache is
    # in-memory only). Unlike b079_07, nothing else in --number4 mode reuses
    # it, so skipping when already computed is a real saving.
    if _need("hourly_consumption"):
        print(f"[4-a] Hourly totals (cached b079_02) …", flush=True)
        t0 = _t.monotonic()
        _, hourly_totals, hourly_wd_totals, hourly_we_totals = _get_b079_02()
        if hourly_totals:
            total = sum(hourly_totals.values())
            if total > 0:
                distributions["hourly_consumption"] = {
                    k: round(v / total, 4) for k, v in sorted(hourly_totals.items())
                }
        if hourly_wd_totals:
            total_wd = sum(hourly_wd_totals.values())
            if total_wd > 0:
                distributions["hourly_consumption_weekday"] = {
                    k: round(v / total_wd, 4) for k, v in sorted(hourly_wd_totals.items())
                }
        if hourly_we_totals:
            total_we = sum(hourly_we_totals.values())
            if total_we > 0:
                distributions["hourly_consumption_weekend"] = {
                    k: round(v / total_we, 4) for k, v in sorted(hourly_we_totals.items())
                }
        if hourly_totals:
            print(f"    Temporal: {len(hourly_totals)} time slots "
                  f"(wd={len(hourly_wd_totals)}, we={len(hourly_we_totals)}) "
                  f"[{_t.monotonic()-t0:.1f}s]", flush=True)
    _flush("4-a")

    idx = _ensure_raw_index()

    # ---- 4-b: Temporal activity by demo (B009 time_flow) -------------------
    # Plain single file → byte-range parallel (4 workers on one CSV).
    # Zip member → single-stream fallback. Multi-file → file-level parallel.
    print(f"[4-b] Temporal activity by demo (B009 time_flow) …", flush=True)
    t0 = _t.monotonic()
    ta_paths = find_dataset(idx, "b009_time_flow")
    if ta_paths and _need("temporal_activity_by_demo"):
        header, enc, delim = peek_header(ta_paths[0])
        c_time_idx = find_col(header, "시간대", required=False) if header else None
        demo_cols = []
        if header:
            for i, col in enumerate(header):
                norm = col.replace(" ", "").replace("_", "").lower()
                if norm.startswith(("남자", "male")):
                    gen = "M"
                elif norm.startswith(("여자", "female")):
                    gen = "F"
                else:
                    continue
                digits = re.findall(r"\d+", col)
                age = digits[0] if digits else "70"
                demo_cols.append((col, gen, age))

        if c_time_idx is not None and demo_cols:
            time_col_name = header[c_time_idx]
            usecols = [time_col_name] + [c for c, _, _ in demo_cols]
            time_dist = defaultdict(lambda: defaultdict(float))

            def _merge_4b(d):
                for dk, slots in d.items():
                    for t, v in slots.items():
                        time_dist[dk][t] += v

            entry = ta_paths[0]
            if len(ta_paths) == 1 and not entry.is_zip_member:
                n_req = min(4, os.cpu_count() or 4)
                ranges = _split_byte_ranges(entry.path, n_req)
                n_workers = _chunk_workers(len(ranges), mb_per_worker=1500)
                print(f"    [4-b] plain file → {len(ranges)} byte-ranges × "
                      f"{n_workers} workers", flush=True)
                args_list = [
                    (str(entry.path), s, e, header, enc, delim,
                     time_col_name, demo_cols, usecols)
                    for s, e in ranges
                ]
                if n_workers == 1 or len(ranges) == 1:
                    for a in args_list:
                        d, _n = _worker_4b_range(a)
                        _merge_4b(d)
                else:
                    with ProcessPoolExecutor(max_workers=n_workers) as ex:
                        for f in as_completed([ex.submit(_worker_4b_range, a) for a in args_list]):
                            d, _n = f.result()
                            _merge_4b(d)
            else:
                n_workers = _mem_aware_workers(len(ta_paths), mb_per_worker=1500)
                print(f"    [4-b] {len(ta_paths)} files × {n_workers} workers "
                      f"(mem-aware, zip/multi-file)", flush=True)
                args_list = [(e, time_col_name, demo_cols, usecols) for e in ta_paths]
                if n_workers == 1 or len(ta_paths) == 1:
                    for a in args_list:
                        d, _name, _n = _worker_4b(a)
                        _merge_4b(d)
                else:
                    with ProcessPoolExecutor(max_workers=n_workers) as ex:
                        for f in as_completed([ex.submit(_worker_4b, a) for a in args_list]):
                            d, _name, _n = f.result()
                            _merge_4b(d)

            temporal_ratios = {}
            for demo_key, cols in time_dist.items():
                total = sum(cols.values())
                if total > 0:
                    temporal_ratios[demo_key] = {
                        k: round(v / total, 4) for k, v in cols.items()
                    }
            if temporal_ratios:
                distributions["temporal_activity_by_demo"] = temporal_ratios
                print(f"    Activity by demo: {len(temporal_ratios)} groups (ratio) "
                      f"[{_t.monotonic()-t0:.1f}s]", flush=True)
    _flush("4-b")

    # ---- 4-c: Movement purpose (B078 PURPOSE_250M, parallel) ---------------
    print(f"[4-c] Movement purpose (B078 PURPOSE_250M) …", flush=True)
    t0 = _t.monotonic()
    mp_paths = find_dataset(idx, "b078_purpose")
    if mp_paths and _need("movement_purpose"):
        mp_header, mp_enc, mp_delim = peek_header(mp_paths[0])
        purpose_col = None
        count_col = None
        if mp_header:
            for c in mp_header:
                c_low = c.lower()
                if "목적" in c or "purpose" in c_low:
                    purpose_col = c
                # narrower count-col match to avoid 좌표수/격자수 false positives
                if ("인구수" in c or "이동자수" in c or "통행량" in c
                        or "count" in c_low or "cnt" in c_low):
                    count_col = c
        if purpose_col:
            usecols = [purpose_col] + ([count_col] if count_col else [])
            purpose_counts = defaultdict(float)

            def _merge_4c(d):
                for p, v in d.items():
                    purpose_counts[p] += v

            entry = mp_paths[0]
            if len(mp_paths) == 1 and not entry.is_zip_member:
                n_req = min(4, os.cpu_count() or 4)
                ranges = _split_byte_ranges(entry.path, n_req)
                n_workers = _chunk_workers(len(ranges), mb_per_worker=400)
                print(f"    [4-c] plain file → {len(ranges)} byte-ranges × "
                      f"{n_workers} workers (count_col={count_col!r})", flush=True)
                args_list = [
                    (str(entry.path), s, e, mp_header, mp_enc, mp_delim,
                     purpose_col, count_col, usecols)
                    for s, e in ranges
                ]
                if n_workers == 1 or len(ranges) == 1:
                    for a in args_list:
                        d, _n = _worker_4c_range(a)
                        _merge_4c(d)
                else:
                    with ProcessPoolExecutor(max_workers=n_workers) as ex:
                        for f in as_completed([ex.submit(_worker_4c_range, a) for a in args_list]):
                            d, _n = f.result()
                            _merge_4c(d)
            else:
                n_workers = _mem_aware_workers(len(mp_paths), mb_per_worker=400)
                print(f"    [4-c] {len(mp_paths)} files × {n_workers} workers "
                      f"(mem-aware, zip/multi-file, count_col={count_col!r})", flush=True)
                args_list = [(e, purpose_col, count_col, usecols) for e in mp_paths]
                if n_workers == 1 or len(mp_paths) == 1:
                    for a in args_list:
                        d, _name, _n = _worker_4c(a)
                        _merge_4c(d)
                else:
                    with ProcessPoolExecutor(max_workers=n_workers) as ex:
                        for f in as_completed([ex.submit(_worker_4c, a) for a in args_list]):
                            d, _name, _n = f.result()
                            _merge_4c(d)
            total = sum(purpose_counts.values())
            if total > 0:
                distributions["movement_purpose"] = {
                    k: round(v / total, 4)
                    for k, v in sorted(purpose_counts.items(), key=lambda x: -x[1])
                }
                print(f"    Movement purposes: {len(purpose_counts)} categories "
                      f"[{_t.monotonic()-t0:.1f}s]", flush=True)
    _flush("4-c")

    # ---- 4-d: Block consumption decile (B063, parallel) --------------------
    print(f"[4-d] Block consumption (B063 블록 성연령) …", flush=True)
    t0 = _t.monotonic()
    bc_paths = find_dataset(idx, "b063_block_demo")
    if bc_paths and _need("block_consumption_by_demo"):
        bc_header, bc_enc, bc_delim = peek_header(bc_paths[0])
        c_gen_idx = find_col(bc_header, "성별", "SEX", required=False) if bc_header else None
        c_age_idx = find_col(bc_header, "연령대", "AGE", required=False) if bc_header else None
        value_cols = []
        if bc_header:
            value_cols = [
                h for h in bc_header
                if any(kw in h for kw in ("금액", "건수", "AMT", "CNT"))
            ]
        if c_gen_idx is not None and c_age_idx is not None and value_cols:
            c_gen_name = bc_header[c_gen_idx]
            c_age_name = bc_header[c_age_idx]
            usecols = [c_gen_name, c_age_name] + value_cols
            row_counts = defaultdict(lambda: defaultdict(int))

            def _merge_4d(d):
                for demo_key, cols in d.items():
                    for col, n in cols.items():
                        row_counts[demo_key][col] += n

            entry = bc_paths[0]
            if len(bc_paths) == 1 and not entry.is_zip_member:
                n_req = min(4, os.cpu_count() or 4)
                ranges = _split_byte_ranges(entry.path, n_req)
                n_workers = _chunk_workers(len(ranges), mb_per_worker=800)
                print(f"    [4-d] plain file → {len(ranges)} byte-ranges × "
                      f"{n_workers} workers ({len(value_cols)} value cols)", flush=True)
                args_list = [
                    (str(entry.path), s, e, bc_header, bc_enc, bc_delim,
                     c_gen_name, c_age_name, value_cols, usecols)
                    for s, e in ranges
                ]
                if n_workers == 1 or len(ranges) == 1:
                    for a in args_list:
                        d, _n = _worker_4d_range(a)
                        _merge_4d(d)
                else:
                    with ProcessPoolExecutor(max_workers=n_workers) as ex:
                        for f in as_completed([ex.submit(_worker_4d_range, a) for a in args_list]):
                            d, _n = f.result()
                            _merge_4d(d)
            else:
                n_workers = _mem_aware_workers(len(bc_paths), mb_per_worker=800)
                print(f"    [4-d] {len(bc_paths)} files × {n_workers} workers "
                      f"(mem-aware, zip/multi-file, {len(value_cols)} value cols)", flush=True)
                args_list = [(e, c_gen_name, c_age_name, value_cols, usecols) for e in bc_paths]
                if n_workers == 1 or len(bc_paths) == 1:
                    for a in args_list:
                        d, _name, _n = _worker_4d(a)
                        _merge_4d(d)
                else:
                    with ProcessPoolExecutor(max_workers=n_workers) as ex:
                        for f in as_completed([ex.submit(_worker_4d, a) for a in args_list]):
                            d, _name, _n = f.result()
                            _merge_4d(d)

            block_deciles = {}
            for demo_key, cols in row_counts.items():
                demo_result = {}
                for col, n in cols.items():
                    if n <= 0:
                        continue
                    idx_arr = np.arange(n, dtype=np.int64)
                    deciles_arr = np.minimum(10, (idx_arr * 10) // n + 1)
                    counts = np.bincount(deciles_arr, minlength=11)
                    dist = {str(d): int(counts[d]) for d in range(1, 11) if counts[d] > 0}
                    demo_result[col] = {"decile_distribution": dist}
                block_deciles[demo_key] = demo_result
            if block_deciles:
                distributions["block_consumption_by_demo"] = block_deciles
                print(f"    Block consumption: {len(block_deciles)} demo groups (decile) "
                      f"[{_t.monotonic()-t0:.1f}s]", flush=True)
    _flush("4-d")

    # ---- 4-e: Industry spending breakdown (cached b079_07) -----------------
    print(f"[4-e] Industry spending (cached b079_07) …", flush=True)
    t0 = _t.monotonic()
    _, by_demo = _get_b079_07()
    if by_demo:
        industry_spend = defaultdict(lambda: defaultdict(float))
        for (gen, age, ind), amt in by_demo.items():
            industry_spend[f"{gen}_{age}"][ind] += amt
        industry_dist = {}
        for demo_key, industries in industry_spend.items():
            total = sum(industries.values())
            if total > 0:
                industry_dist[demo_key] = {
                    k: round(v / total, 4)
                    for k, v in sorted(industries.items(), key=lambda x: -x[1])
                }
        if industry_dist:
            distributions["industry_spending_ratio"] = industry_dist
            print(f"    Industry spending: {len(industry_dist)} demo groups "
                  f"[{_t.monotonic()-t0:.1f}s]", flush=True)
    _flush("4-e")

    # ---- 4-f: Weekday/Weekend spending ratio (B042 blck timeslot, parallel) -
    print(f"[4-f] Weekday/Weekend ratio (B042 블록 일자별 시간대별) …", flush=True)
    t0 = _t.monotonic()
    ww_paths = find_dataset(idx, "b042_blck_timeslot")
    if ww_paths and _need("weekday_weekend_spending"):
        ww_header, ww_enc, ww_delim = peek_header(ww_paths[0])
        d_idx = find_col(ww_header, "요일", "DAW", required=False) if ww_header else None
        a_idx = find_col(ww_header, "금액", "AMT", required=False) if ww_header else None
        if d_idx is not None and a_idx is not None:
            d_name = ww_header[d_idx]
            a_name = ww_header[a_idx]
            weekday_amt = 0.0; weekend_amt = 0.0
            weekday_cnt = 0;   weekend_cnt = 0

            entry = ww_paths[0]
            if len(ww_paths) == 1 and not entry.is_zip_member:
                n_req = min(4, os.cpu_count() or 4)
                ranges = _split_byte_ranges(entry.path, n_req)
                n_workers = _chunk_workers(len(ranges), mb_per_worker=300)
                print(f"    [4-f] plain file → {len(ranges)} byte-ranges × "
                      f"{n_workers} workers", flush=True)
                args_list = [
                    (str(entry.path), s, e, ww_header, ww_enc, ww_delim, d_name, a_name)
                    for s, e in ranges
                ]
                if n_workers == 1 or len(ranges) == 1:
                    for a in args_list:
                        wd_a, we_a, wd_c, we_c, _n = _worker_4f_range(a)
                        weekday_amt += wd_a; weekend_amt += we_a
                        weekday_cnt += wd_c; weekend_cnt += we_c
                else:
                    with ProcessPoolExecutor(max_workers=n_workers) as ex:
                        for f in as_completed([ex.submit(_worker_4f_range, a) for a in args_list]):
                            wd_a, we_a, wd_c, we_c, _n = f.result()
                            weekday_amt += wd_a; weekend_amt += we_a
                            weekday_cnt += wd_c; weekend_cnt += we_c
            else:
                n_workers = _mem_aware_workers(len(ww_paths), mb_per_worker=300)
                print(f"    [4-f] {len(ww_paths)} files × {n_workers} workers "
                      f"(mem-aware, zip/multi-file)", flush=True)
                args_list = [(e, d_name, a_name) for e in ww_paths]
                if n_workers == 1 or len(ww_paths) == 1:
                    for a in args_list:
                        wd_a, we_a, wd_c, we_c, _name, _n = _worker_4f(a)
                        weekday_amt += wd_a; weekend_amt += we_a
                        weekday_cnt += wd_c; weekend_cnt += we_c
                else:
                    with ProcessPoolExecutor(max_workers=n_workers) as ex:
                        for f in as_completed([ex.submit(_worker_4f, a) for a in args_list]):
                            wd_a, we_a, wd_c, we_c, _name, _n = f.result()
                            weekday_amt += wd_a; weekend_amt += we_a
                            weekday_cnt += wd_c; weekend_cnt += we_c

            if weekday_cnt > 0 and weekend_cnt > 0:
                wd_avg = weekday_amt / weekday_cnt
                we_avg = weekend_amt / weekend_cnt
                total_avg = (weekday_amt + weekend_amt) / (weekday_cnt + weekend_cnt)
                distributions["weekday_weekend_spending"] = {
                    "weekday_ratio": round(wd_avg / total_avg, 4),
                    "weekend_ratio": round(we_avg / total_avg, 4),
                    "weekend_to_weekday": round(we_avg / wd_avg, 4),
                    "_note": "weekend_to_weekday < 1 means weekend spending is lower"
                }
                print(f"    Weekday/Weekend: ratio={we_avg/wd_avg:.3f} "
                      f"[{_t.monotonic()-t0:.1f}s]", flush=True)
    _flush("4-f")

    print(f"[4] global_distributions DONE in {_t.monotonic()-t_all:.1f}s", flush=True)
    return distributions

# ---------------------------------------------------------------------------
# 5. Agent Allocation – proportional to population
# ---------------------------------------------------------------------------
def compute_allocation(pop_weights, target=TARGET_AGENTS):
    print(f"[5] Computing agent allocation for {target} agents …")
    total_pop = sum(pop_weights.values())
    if total_pop <= 0:
        return {}

    allocation = {}
    assigned = 0
    # Proportional allocation, minimum 1 for non-zero populations
    raw_alloc = {}
    for key, pop in pop_weights.items():
        raw = (pop / total_pop) * target
        raw_alloc[key] = raw

    # Round and ensure minimum 1 for non-zero, adjust to hit target
    for key, raw in sorted(raw_alloc.items(), key=lambda x: -x[1]):
        n = max(1, round(raw))
        allocation[key] = n
        assigned += n

    # Adjust to exactly match target
    diff = assigned - target
    if diff != 0:
        # Adjust the largest allocations
        sorted_keys = sorted(allocation.keys(), key=lambda k: -allocation[k])
        for key in sorted_keys:
            if diff == 0:
                break
            if diff > 0 and allocation[key] > 1:
                allocation[key] -= 1
                diff -= 1
            elif diff < 0:
                allocation[key] += 1
                diff += 1

    print(f"    Allocated {sum(allocation.values())} agents across {len(allocation)} combos")
    print(f"    Range: {min(allocation.values())}-{max(allocation.values())} agents per combo")
    return allocation

# ---------------------------------------------------------------------------
# 6. Aggregate Stats  –  (gender, age) 그룹별 요약 통계
# ---------------------------------------------------------------------------
def build_aggregate_stats(profiles):
    """Build summary statistics per (gender, age) group for LLM context."""
    print("[6] Building aggregate statistics per (gender, age) …")
    group_data = defaultdict(lambda: defaultdict(list))

    for key, profile in profiles.items():
        gen = profile["demographics"]["gender"]
        age = profile["demographics"]["age_grp"]
        group_key = f"{gen}_{age}"

        # telecom 메트릭만 포함 (이미 반출 가능 데이터 기반)
        # consumption/mobility는 응용집계 변환 후 범주이므로 mean/std 불필요
        for metric, val in profile["telecom"].items():
            group_data[group_key][metric].append(val)

    agg_stats = {}
    for group_key, metrics in group_data.items():
        agg_stats[group_key] = {
            metric: compute_stats(vals) for metric, vals in metrics.items()
        }

    print(f"    {len(agg_stats)} demographic groups summarized")
    return agg_stats

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    source, only_stage = parse_args()
    out_dir = PROJECT_ROOT / f"output/{source}"
    stats_dir = PROJECT_ROOT / "output/stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    def save_json(data, filename):
        path = stats_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        size_kb = path.stat().st_size / 1024
        print(f"    → {filename} ({size_kb:.1f} KB)", flush=True)

    print("=" * 55)
    print(f"  Statistical Analysis (source: {source})")
    if only_stage is not None:
        if only_stage == 4:
            print(f"  Mode: --stage 4 (resume: run 4 → 5 → 6 → 7)")
        else:
            print(f"  Mode: --stage {only_stage} (only this stage)")
    print("=" * 55, flush=True)

    # ---- Stage 4 → 5 → 6 → 7 (--number4 / --stage 4) ----------------------
    # Resume from stage 4 when stages 1/2/3 (and 3b/3c/3d) have already
    # produced their JSON. Runs 4 (heavy), then cheap 5/6/7 using cached
    # consumption_detail.json + freshly-rebuilt profiles.
    if only_stage == 4:
        # 4. Global distributions (heavy — hours)
        global_dist = build_global_distributions(
            out_dir,
            save_cb=lambda d: save_json(d, "global_distributions.json"),
        )
        save_json(global_dist, "global_distributions.json")

        # 5 & 6 & 7 shared dependency: profiles + pop_weights rebuilt once.
        print(f"\n[post-4] Rebuilding profiles for stages 5/6/7 …", flush=True)
        profiles, pop_weights = build_agent_profiles(out_dir)

        # 5. Agent allocation
        allocation = compute_allocation(pop_weights)
        save_json(allocation, "agent_allocation.json")

        # 6. Aggregate stats (raw profiles required → before export conversion)
        agg_stats = build_aggregate_stats(profiles)
        save_json(agg_stats, "aggregate_stats.json")

        # 7. Convert profiles to export format (응용집계) — needs consumption_detail.
        cd_path = stats_dir / "consumption_detail.json"
        if cd_path.exists():
            print(f"[3c-detail] Loading cached consumption_detail.json …", flush=True)
            with open(cd_path, "r", encoding="utf-8") as f:
                consumption_detail = json.load(f)
        else:
            print(f"[3c-detail] consumption_detail.json not found → rebuilding …", flush=True)
            consumption_detail = build_consumption_detail(out_dir)
            save_json(consumption_detail, "consumption_detail.json")
        profiles = convert_profiles_to_export(profiles, consumption_detail, out_dir)
        save_json(profiles, "agent_profiles.json")

        print(f"\n{'='*55}")
        print(f"  [OK] Stages 4 → 5 → 6 → 7 Complete")
        print(f"{'='*55}", flush=True)
        return

    # ---- Stage 5 only (--number5 / --stage 5) -----------------------------
    # Rebuild agent_allocation.json. Needs pop_weights → rerun stage 1
    # (joined_persona_base.csv read only; cheap).
    if only_stage == 5:
        _, pop_weights = build_agent_profiles(out_dir)
        allocation = compute_allocation(pop_weights)
        save_json(allocation, "agent_allocation.json")
        print(f"\n{'='*55}")
        print(f"  [OK] Stage 5 Complete")
        print(f"{'='*55}", flush=True)
        return

    # ---- Stage 6 only (--number6 / --stage 6) -----------------------------
    # Rebuild aggregate_stats.json. Needs raw profiles → rerun stage 1.
    if only_stage == 6:
        profiles, _ = build_agent_profiles(out_dir)
        agg_stats = build_aggregate_stats(profiles)
        save_json(agg_stats, "aggregate_stats.json")
        print(f"\n{'='*55}")
        print(f"  [OK] Stage 6 Complete")
        print(f"{'='*55}", flush=True)
        return

    # ---- Stage 7 only (--number7 / --stage 7) -----------------------------
    # Rebuild agent_profiles.json 응용집계 포맷. Needs raw profiles + consumption_detail.
    # consumption_detail: load cached JSON if present (heavy to rebuild), else rerun 3c.
    if only_stage == 7:
        profiles, _ = build_agent_profiles(out_dir)
        cd_path = stats_dir / "consumption_detail.json"
        if cd_path.exists():
            print(f"[3c-detail] Loading cached consumption_detail.json …")
            with open(cd_path, "r", encoding="utf-8") as f:
                consumption_detail = json.load(f)
        else:
            print(f"[3c-detail] consumption_detail.json not found → rebuilding …")
            consumption_detail = build_consumption_detail(out_dir)
            save_json(consumption_detail, "consumption_detail.json")
        profiles = convert_profiles_to_export(profiles, consumption_detail, out_dir)
        save_json(profiles, "agent_profiles.json")
        print(f"\n{'='*55}")
        print(f"  [OK] Stage 7 Complete")
        print(f"{'='*55}", flush=True)
        return

    # 1. Agent profiles
    profiles, pop_weights = build_agent_profiles(out_dir)
    save_json(profiles, "agent_profiles.json")

    # 2. Dong context
    dong_context = build_dong_context(out_dir)
    save_json(dong_context, "dong_context.json")

    # 3b. Workplace population
    workplace_pop = build_workplace_population(out_dir)
    save_json(workplace_pop, "workplace_population.json")

    # 5. Agent allocation (fast — 위로 이동: 빠른 insurance)
    allocation = compute_allocation(pop_weights)
    save_json(allocation, "agent_allocation.json")

    # 6. Aggregate stats by (gender, age) (fast — raw profiles 필요)
    agg_stats = build_aggregate_stats(profiles)
    save_json(agg_stats, "aggregate_stats.json")

    # 3c. Consumption detail (adm8 × gender × age × weekday/weekend × industry)
    #     → 6시 컷오프 대비 가장 가치 높은 무거운 섹션을 먼저 확보
    consumption_detail = build_consumption_detail(out_dir)
    save_json(consumption_detail, "consumption_detail.json")

    # 7. Convert agent profiles to export format (응용집계)
    #    consumption_detail 에서 계산된 평일/주말 spending_level 을 주입.
    #    → agent_profiles.json 을 최종 export 포맷으로 덮어쓴다.
    profiles = convert_profiles_to_export(profiles, consumption_detail, out_dir)
    save_json(profiles, "agent_profiles.json")

    # 3. Workplace flow (enhanced with workplace_pop fallback)
    workplace_flow = build_workplace_flow(out_dir, workplace_pop)
    save_json(workplace_flow, "workplace_flow.json")

    # 3d. Dong consumption patterns (uses _get_b079_02 cache)
    dong_consumption = build_dong_consumption(out_dir)
    save_json(dong_consumption, "dong_consumption.json")

    # 4. Global distributions (parallel 4-b/4-c/4-d/4-f + incremental save)
    global_dist = build_global_distributions(
        out_dir,
        save_cb=lambda d: save_json(d, "global_distributions.json"),
    )
    save_json(global_dist, "global_distributions.json")

    # --- Summary ---
    print(f"\n{'='*55}")
    print(f"  [OK] Analysis Complete")
    print(f"{'='*55}")
    print(f"  Profiles:     {len(profiles)} unique (adm8, gender, age)")
    print(f"  Dongs:        {len(dong_context)} with context")
    print(f"  Workplace:    {len(workplace_flow)} dongs with flow data")
    print(f"  WorkplacePop: {len(workplace_pop)} dongs with population data")
    print(f"  ConsumpDetail:{max(0, len(consumption_detail) - 1)} (adm8,gender,age) consumption details")
    print(f"  DongConsump:  {len(dong_consumption)} dongs with consumption patterns")
    print(f"  Allocation:   {sum(allocation.values())} agents → {len(allocation)} combos")
    print(f"  Output:       {stats_dir}/")
    print(flush=True)


if __name__ == "__main__":
    main()
