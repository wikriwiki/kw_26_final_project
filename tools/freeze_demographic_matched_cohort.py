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
    for (sx, ab), want in sorted(want_coarse.items()):
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
    print()
    print("  주변분포 최대 오차 **%.5f** (문턱 %.3f)" % (worst, a.max_margin_error))
    if short_dong:
        print("  후보가 없어 못 채운 (성별/연령대) 칸 **%d개 · %d명**"
              % (len(short_dong), sum(short_dong.values())))
    if subs:
        print("  같은 (성별/연령대) 안의 다른 동에서 꺼낸 횟수 **%d** (빈 동이 있다)" % subs)
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
            "margin_error": err, "dong_max_error": round(dmax, 6),
            "worst_margin_error": round(worst, 6),
            "unfilled_coarse_cells": short_dong, "cross_dong_substitutions": subs,
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
