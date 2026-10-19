"""서울 실제 인구 분포에 맞춘 명부를 만든다 — **소득은 맞추지 않는다.**

    py -3 tools/freeze_demographic_matched_cohort.py \
        --frame output/population_matching_20260928/policy_population_targets_P012_202109.json \
        --candidates output/population_frame_sources_20260928/graph_full_population_code_projection_20260928.json \
        --citizens 1000 --out output/cohort/p012_main_1000.json

## 왜 기존 도구를 안 쓰는가

`tools/freeze_population_matched_cohort.py` 는 성별·연령·행정동·**소득** 네 주변분포를
맞추고, 소득이 공식 원자료로 검증되지 않으면 진행하지 않는다(fail-closed). 우리
후보 파일은 `income_verified: false` 이고 소득 티어가
`unverified_generated_income_tier` 다. 그래서 그 관문을 **낮추지 않고**, 맞출 수 있는
세 분포만 맞추는 별도 도구를 쓴다. 소득이 맞지 않았다는 사실을 산출물에 적는다.

## 무엇을 맞추는가

    성별        행안부 2021-09 서울 주민등록 비율
    연령대      같은 자료 (프레임이 **20세 이상**이라 10대는 범위 밖 — 제외한다)
    행정동      같은 자료 (426개)

동 안에서는 프레임의 **결합 칸**(동 x 성별 x 연령대)을 쓴다. 주변분포만 맞추는 것보다
강하다 — 다만 후보가 얇아 칸을 다 채우지 못하면 **채운 척하지 않고 부족분을 적는다.**

## 배분 방식 두 가지 (`--allocation`)

    coarse_first     (기본, P012 명부) 성별x연령대 10칸을 먼저 정확히 맞추고, 동은 그 안에서
                     큰 나머지로 나눈다. 동마다 1명이 안 되는 칸이 많아 **작은 동이 통째로
                     빠진다** (2,000명에서 425개 동 중 325개, 빠진 동 인구 13.7%).
    dong_controlled  표를 **두 방향으로 동시에** 반올림한다(통제 반올림), 두 단계로:
                       1) 구 x (성별,연령대) 250칸 — 구 합계와 (성별,연령대) 합계를 동시에
                       2) 구마다 동 x (성별,연령대) — 1)의 칸 합계와 동 합계를 동시에
                     모든 칸이 기대치의 내림 또는 올림이다. 어느 칸을 올릴지는 최대 흐름으로
                     고르고, 탐색 순서를 씨앗으로 섞어 코드 순서가 성별·연령을 몰지 않게 한다
                     (섞지 않으면 앞 번호 구에 여성 20대가 몰렸다 — 측정: 구x성x연령 칸 오차 0.84%p).
                     후보가 없는 칸은 **같은 구** 같은 (성별,연령대)에서 꺼내고 그 수를 적는다.

## 모르는 것을 지어내지 않는다

  · 후보에 없는 동(빈 동)은 **대체하지 않고** 부족분으로 적는다.
  · 칸이 비면 같은 동의 다른 칸에서 꺼내고, 그 횟수를 `substitutions` 에 적는다.
  · 소득 분포는 맞추지 않는다 — `income_calibrated: false` 를 산출물에 박는다.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# 후보의 연령대 어휘 -> 프레임의 어휘. 프레임이 20세 이상이라 10대는 범위 밖이다.
AGE_TO_FRAME = {"20대": "20대", "30대": "30대", "40대": "40대", "50대": "50대",
                "60대": "60세 이상", "70대이상": "60세 이상"}
OUT_OF_SCOPE = {"10대"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dong8(code: str) -> str:
    """프레임의 10자리 법정동 코드를 후보의 8자리로. 1114055000 -> 11140550."""
    c = str(code)
    return c[:8] if len(c) == 10 and c.endswith("00") else c


def largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """비례 배분에서 반올림 손실을 큰 나머지 순으로 되돌린다 — 합이 정확히 total."""
    raw = {k: w * total for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    left = total - sum(out.values())
    for k in sorted(raw, key=lambda x: (-(raw[x] - out[x]), x))[:max(0, left)]:
        out[k] += 1
    return out


def controlled_round(cells: dict[tuple[str, str], float],
                     row_tot: dict[str, int], col_tot: dict[str, int],
                     rnd: random.Random | None = None) -> tuple[dict, int]:
    """2차원 표의 통제 반올림. cells[(행,열)] = 기대 인원(실수).

    각 칸 = 내림 또는 올림, 행 합 = row_tot, 열 합 = col_tot. 어느 칸을 올릴지는
    (출발 -> 행 [남은 몫] -> 열 [소수부가 있는 칸마다 1] -> 도착 [남은 몫]) 최대 흐름으로 고른다.
    흐름이 모자라면 기대치가 0 인 칸도 1 까지 열고, 그렇게 연 칸 수를 돌려준다.
    rnd 를 주면 행과 칸의 탐색 순서를 섞는다 — 코드 순서가 어느 칸을 올릴지 정하지 않게.
    """
    import math
    out = {k: int(math.floor(v)) for k, v in cells.items()}
    rows = sorted(row_tot)
    cols = sorted(col_tot)
    if rnd is not None:
        rnd.shuffle(rows)
    r_need = {r: row_tot[r] - sum(out.get((r, c), 0) for c in cols) for r in rows}
    c_need = {c: col_tot[c] - sum(out.get((r, c), 0) for r in rows) for c in cols}
    if min(r_need.values(), default=0) < 0 or min(c_need.values(), default=0) < 0:
        raise SystemExit("통제 반올림: 내림 합이 목표보다 크다 — 목표가 반올림이 아니다")
    if sum(r_need.values()) != sum(c_need.values()):
        raise SystemExit("통제 반올림: 행 합과 열 합이 다르다")

    def flow(edges: set) -> dict:
        used: dict = {}
        rleft, cleft = dict(r_need), dict(c_need)
        adj = {r: sorted(c for (rr, c) in edges if rr == r) for r in rows}
        if rnd is not None:
            for r in rows:
                rnd.shuffle(adj[r])
        while True:
            # 너비 우선 증가 경로: 남은 행 -> (안 쓴 칸) 열 -> (쓴 칸을 되돌려) 행 -> ... -> 남은 열
            prev: dict = {}
            queue = [("r", r) for r in rows if rleft[r] > 0]
            for q in queue:
                prev[q] = None
            end = None
            i = 0
            while i < len(queue) and end is None:
                kind, x = queue[i]
                i += 1
                if kind == "r":
                    for c in adj[x]:
                        if used.get((x, c), 0) == 0 and ("c", c) not in prev:
                            prev[("c", c)] = (kind, x)
                            if cleft[c] > 0:
                                end = ("c", c)
                                break
                            queue.append(("c", c))
                else:
                    for r in rows:
                        if used.get((r, x), 0) == 1 and ("r", r) not in prev:
                            prev[("r", r)] = (kind, x)
                            queue.append(("r", r))
            if end is None:
                return used
            node = end
            cleft[end[1]] -= 1
            while prev[node] is not None:
                p = prev[node]
                if node[0] == "c":
                    used[(p[1], node[1])] = 1
                else:
                    used[(node[1], p[1])] = 0
                node = p
            rleft[node[1]] -= 1

    frac = {k for k, v in cells.items() if v - math.floor(v) > 1e-12}
    used = flow(frac)
    opened = 0
    if sum(used.values()) < sum(r_need.values()):
        used = flow({(r, c) for r in rows for c in cols})
        opened = sum(1 for k, v in used.items() if v and k not in frac)
        if sum(used.values()) < sum(r_need.values()):
            raise SystemExit("통제 반올림: 행 합과 열 합을 동시에 맞출 수 없다")
    for k, v in used.items():
        if v:
            out[k] = out.get(k, 0) + 1
    return out, opened


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--citizens", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--eligible-ids", default=None,
                    help="그래프가 실제로 가진 에이전트 id 목록(JSON 배열). 주면 이 안에서만 "
                         "뽑는다 — 후보 목록이 다른 그래프에서 만들어졌을 수 있다.")
    ap.add_argument("--allocation", choices=["coarse_first", "dong_controlled"],
                    default="coarse_first",
                    help="coarse_first: P012 명부 방식(기본). dong_controlled: 동 x (성별,연령대) "
                         "통제 반올림 — 작은 동이 빠지지 않는다.")
    ap.add_argument("--max-margin-error", type=float, default=0.02,
                    help="주변분포 최대 절대오차. 넘으면 쓰지 않고 멈춘다.")
    a = ap.parse_args()

    fp, cp = Path(a.frame), Path(a.candidates)
    frame = json.loads(fp.read_text(encoding="utf-8"))
    cand = json.loads(cp.read_text(encoding="utf-8"))
    if frame.get("schema") != "population_calibration_frame_v1":
        raise SystemExit("frame schema unsupported: %s" % frame.get("schema"))
    tg = frame["targets"]

    # 후보 — 범위 밖 연령과 **그래프에 없는 사람**을 먼저 걷어낸다.
    # 후보 파일은 다른 시점의 그래프에서 만들어졌을 수 있다. 실제로 1,000명 중
    # 9명이 지금 그래프에 거주지가 없었고, 시뮬의 명부 관문이 그것을 잡았다.
    eligible = None
    if a.eligible_ids:
        eligible = set(json.loads(Path(a.eligible_ids).read_text(encoding="utf-8")))
    pool: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    dropped = not_in_graph = 0
    for ag in cand["agents"]:
        band = ag.get("age_band")
        if band in OUT_OF_SCOPE:
            dropped += 1
            continue
        if eligible is not None and str(ag["aid"]) not in eligible:
            not_in_graph += 1
            continue
        fb = AGE_TO_FRAME.get(band)
        if fb is None:
            raise SystemExit("연령대를 프레임 어휘로 옮길 수 없다: %r" % band)
        pool[(str(ag["raw_residence_code"]), str(ag["sex"]), fb)].append(str(ag["aid"]))
    for v in pool.values():
        v.sort()
    have_dong = {k[0] for k in pool}

    # [배분 순서가 결과를 정한다]
    # 동마다 2~3명을 큰 나머지로 뽑으면 **가장 큰 칸만 계속 뽑혀** 연령이 쏠린다
    # (측정: 60세 이상 28.2% -> 64.6%). 그래서 거친 분포를 먼저 정확히 맞추고,
    # 동은 그 안에서 조건부로 나눠 **나머지를 흡수하게** 한다. 동은 칸이 426개라
    # 원래 오차가 작다(측정 0.0037).
    sexp = {k: float(v) for k, v in tg["sex"]["proportions"].items()}
    agep = {k: float(v) for k, v in tg["age_band"]["proportions"].items()}
    coarse = {(sx, ab): sexp[sx] * agep[ab] for sx in sexp for ab in agep}
    tot_c = sum(coarse.values()) or 1.0
    want_coarse = largest_remainder({k: v / tot_c for k, v in coarse.items()}, a.citizens)

    # 프레임의 결합 칸에서 (성별,연령대) 안의 동 분포를 읽는다
    joint: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in frame.get("demographic_joint_counts") or []:
        ab = row.get("age_band")
        if ab not in agep:
            continue
        joint[(str(row["sex"]), ab)][dong8(row["admin_dong"])] += int(row.get("count") or 0)

    rnd = random.Random(a.seed)
    picked: list[str] = []
    short_cell: dict[str, int] = {}
    subs = 0
    subs_same_gu = 0
    opened = 0
    if a.allocation == "dong_controlled":
        # 동 x (성별,연령대) 결합 칸에서 기대 인원을 만들고 두 방향으로 동시에 반올림한다.
        jt: dict[tuple[str, str], float] = defaultdict(float)
        for (sx, ab), dc in joint.items():
            for dong, cnt in dc.items():
                jt[(dong, sx + "|" + ab)] += cnt
        jsum = sum(jt.values()) or 1.0
        cells = {k: a.citizens * v / jsum for k, v in jt.items()}
        # 1단 — 구 x (성별,연령대)
        gcells: dict[tuple[str, str], float] = defaultdict(float)
        for (d, c), v in cells.items():
            gcells[(d[:5], c)] += v
        gsum: dict[str, float] = defaultdict(float)
        csum: dict[str, float] = defaultdict(float)
        for (g, c), v in gcells.items():
            gsum[g] += v
            csum[c] += v
        g_tot = largest_remainder({k: v / a.citizens for k, v in gsum.items()}, a.citizens)
        col_tot = largest_remainder({k: v / a.citizens for k, v in csum.items()}, a.citizens)
        galloc, opened = controlled_round(dict(gcells), g_tot, col_tot, rnd)
        # 2단 — 구마다 동 x (성별,연령대). 칸 합계는 1단 결과, 동 합계는 그 안의 큰 나머지
        alloc: dict[tuple[str, str], int] = {}
        for g in sorted(g_tot):
            sub = {k: v for k, v in cells.items() if k[0][:5] == g}
            ccol: dict[str, float] = defaultdict(float)
            for (d, c), v in sub.items():
                ccol[c] += v
            want_c = {c: galloc.get((g, c), 0) for c in ccol}
            if sum(want_c.values()) != g_tot[g]:
                raise SystemExit("2단 배분: 구 %s 의 칸 합계가 1단과 다르다" % g)
            e = {(d, c): (want_c[c] * v / ccol[c] if ccol[c] > 0 else 0.0)
                 for (d, c), v in sub.items()}
            drow: dict[str, float] = defaultdict(float)
            for (d, c), v in e.items():
                drow[d] += v
            gt = sum(want_c.values())
            d_tot = (largest_remainder({k: v / gt for k, v in drow.items()}, gt)
                     if gt > 0 else {k: 0 for k in drow})
            sa, op2 = controlled_round(e, d_tot, want_c, rnd)
            opened += op2
            alloc.update(sa)
        want_coarse = {tuple(c.split("|")): n for c, n in col_tot.items()}
        used_ids: set[str] = set()
        short: list[tuple[str, str, str, int]] = []
        for (dong, c), n in sorted(alloc.items()):
            if n <= 0:
                continue
            sx, ab = c.split("|")
            avail = [x for x in pool.get((dong, sx, ab), []) if x not in used_ids]
            rnd.shuffle(avail)
            got = avail[:n]
            used_ids.update(got)
            picked.extend(got)
            if len(got) < n:
                short.append((dong, sx, ab, n - len(got)))
        # 후보가 없어 못 채운 몫은 같은 구 -> 서울 전체 순으로, 같은 (성별,연령대) 안에서 꺼낸다
        for dong, sx, ab, k in short:
            for scope in ("gu", "seoul"):
                if k <= 0:
                    break
                rest = [x for key, v in pool.items()
                        if key[1] == sx and key[2] == ab
                        and (scope == "seoul" or key[0][:5] == dong[:5])
                        for x in v if x not in used_ids]
                rnd.shuffle(rest)
                add = rest[:k]
                used_ids.update(add)
                picked.extend(add)
                subs += len(add)
                if scope == "gu":
                    subs_same_gu += len(add)
                k -= len(add)
            if k > 0:
                key = "%s/%s" % (sx, ab)
                short_cell[key] = short_cell.get(key, 0) + k
    for (sx, ab), want in (sorted(want_coarse.items()) if a.allocation == "coarse_first" else []):
        if want <= 0:
            continue
        dcnt = joint.get((sx, ab)) or Counter()
        tot = sum(dcnt.values())
        if tot > 0:
            per = largest_remainder({k: v / tot for k, v in dcnt.items()}, want)
        else:
            per = {}
        taken: list[str] = []
        for dong, n in sorted(per.items()):
            avail = list(pool.get((dong, sx, ab), []))
            rnd.shuffle(avail)
            taken.extend(avail[:n])
        # 동이 비어 못 채운 만큼은 **같은 (성별,연령대)** 안의 다른 동에서 꺼낸다.
        # 거친 분포는 정확히 지키고, 동 오차만 조금 커진다 — 그 횟수를 적는다.
        if len(taken) < want:
            used = set(taken)
            rest = [x for k, v in pool.items() if k[1] == sx and k[2] == ab
                    for x in v if x not in used]
            rnd.shuffle(rest)
            add = rest[:want - len(taken)]
            subs += len(add)
            taken.extend(add)
        if len(taken) < want:
            short_cell["%s/%s" % (sx, ab)] = want - len(taken)
        picked.extend(taken)
    short_dong = short_cell

    if len(picked) != len(set(picked)):
        raise SystemExit("명부에 중복이 있다")

    # 주변분포 오차를 잰다 — 넘으면 쓰지 않는다
    by_aid = {}
    for (dong, sex, band), v in pool.items():
        for x in v:
            by_aid[x] = (dong, sex, band)
    n = len(picked) or 1
    got_sex = Counter(by_aid[x][1] for x in picked)
    got_age = Counter(by_aid[x][2] for x in picked)
    got_dong = Counter(by_aid[x][0] for x in picked)
    err = {}
    err["sex"] = {k: round(got_sex[k] / n - float(v), 5)
                  for k, v in tg["sex"]["proportions"].items()}
    err["age_band"] = {k: round(got_age[k] / n - float(v), 5)
                       for k, v in tg["age_band"]["proportions"].items()}
    dmax = max((abs(got_dong[dong8(k)] / n - float(v))
                for k, v in tg["admin_dong"]["proportions"].items()), default=0.0)
    gu_tgt: dict[str, float] = defaultdict(float)
    for k, v in tg["admin_dong"]["proportions"].items():
        gu_tgt[dong8(k)[:5]] += float(v)
    got_gu = Counter(d[:5] for d in got_dong.elements())
    gmax = max((abs(got_gu[g] / n - v) for g, v in gu_tgt.items()), default=0.0)
    empty_dong_share = sum(float(v) for k, v in tg["admin_dong"]["proportions"].items()
                           if got_dong[dong8(k)] == 0)
    worst = max(max(abs(x) for x in err["sex"].values()),
                max(abs(x) for x in err["age_band"].values()))

    print("# 인구 분포 맞춘 명부 — 소득은 맞추지 않았다")
    print()
    print("  프레임  %s  (%s · %s · 동 %d개 · %s명)"
          % (fp.name, frame.get("reference_period") or frame.get("reference_year"),
             frame.get("age_scope"), frame.get("administrative_dong_count"),
             "{:,}".format(frame.get("resident_count") or 0)))
    print("  후보    %s  (%d명 중 범위 밖 %d명 · 그래프에 없음 %d명 제외)"
          % (cp.name, len(cand["agents"]), dropped, not_in_graph))
    print("  뽑음    **%d명** / 요청 %d명" % (len(picked), a.citizens))
    print()
    print("  %-10s %10s %10s %9s" % ("분포", "목표", "표본", "오차"))
    for k, v in sorted(tg["sex"]["proportions"].items()):
        print("  성별 %-5s %9.4f %10.4f %+9.4f" % (k, float(v), got_sex[k] / n, err["sex"][k]))
    for k in ("20대", "30대", "40대", "50대", "60세 이상"):
        v = float(tg["age_band"]["proportions"][k])
        print("  %-10s %9.4f %10.4f %+9.4f" % (k, v, got_age[k] / n, err["age_band"][k]))
    print("  행정동     동 %d개 중 %d개에 배정 · 최대 동별 오차 %.5f"
          % (len(tg["admin_dong"]["proportions"]), len(got_dong), dmax))
    print("  사람이 없는 동의 인구 몫 %.3f · 자치구 %d/%d개 · 최대 구별 오차 %.5f"
          % (empty_dong_share, len(got_gu), len(gu_tgt), gmax))
    print("  배분 방식 %s" % a.allocation)
    print()
    print("  주변분포 최대 오차 **%.5f** (문턱 %.3f)" % (worst, a.max_margin_error))
    if short_dong:
        print("  후보가 없어 못 채운 (성별/연령대) 칸 **%d개 · %d명**"
              % (len(short_dong), sum(short_dong.values())))
    if subs:
        print("  같은 (성별/연령대) 안의 다른 동에서 꺼낸 횟수 **%d** (빈 동이 있다)" % subs)
    if a.allocation == "dong_controlled":
        print("  그중 같은 구 안에서 %d · 기대치 0 칸을 연 횟수 %d" % (subs_same_gu, opened))
    print("  소득 분포는 **맞추지 않았다** — 후보의 소득이 공식 원자료로 검증되지 않았다")

    if worst > a.max_margin_error:
        raise SystemExit("주변분포 오차 %.5f 가 문턱 %.3f 를 넘는다 — 명부를 쓰지 않는다"
                         % (worst, a.max_margin_error))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    io.open(out, "w", encoding="utf-8", newline="\n").write(
        json.dumps(sorted(picked), ensure_ascii=False, indent=1))
    io.open(str(out) + ".manifest.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps({
            "schema": "demographic_matched_cohort_v1",
            "citizens": len(picked), "requested": a.citizens, "seed": a.seed,
            "matched_fields": ["sex", "age_band", "admin_dong"],
            "income_calibrated": False,
            "income_note": ("후보의 소득이 공식 원자료로 검증되지 않아(income_verified=false) "
                            "소득 분포는 맞추지 않았다. 보고서에 명시할 것."),
            "age_scope": frame.get("age_scope"),
            "out_of_scope_dropped": dropped,
           "not_in_graph_dropped": not_in_graph,
           "eligible_ids": a.eligible_ids,
            "allocation": a.allocation,
            "margin_error": err, "dong_max_error": round(dmax, 6),
            "gu_max_error": round(gmax, 6),
            "empty_dong_population_share": round(empty_dong_share, 6),
            "dongs_with_people": len(got_dong),
            "worst_margin_error": round(worst, 6),
            "unfilled_coarse_cells": short_dong, "cross_dong_substitutions": subs,
            "cross_dong_substitutions_same_gu": subs_same_gu if a.allocation == "dong_controlled" else None,
            "zero_expectation_cells_opened": opened if a.allocation == "dong_controlled" else None,
            "frame": {"path": a.frame, "sha256": sha256(fp)},
            "candidates": {"path": a.candidates, "sha256": sha256(cp)},
            "roster_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        }, ensure_ascii=False, indent=1))
    print()
    print("→ %s" % out)
    print("→ %s.manifest.json" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
