"""P013 1차 긴급재난지원금 — 같은 사람의 지원금 있음/없음 두 시뮬레이션을 KDI 정답지와 맞댄다.

    python scripts/report/score_p013_two_arm.py --dir <채점 폴더> --anchor-map <aid→기준액 json> \
        --json-out <out.json>

채점 폴더에는 on.policy.ledger.jsonl, off.policy.ledger.jsonl (export_policy_daily_ledger.py) 가 있다.
지표와 실측 수치는 data/experiments/P013_indicator_contract.json 에서만 읽는다.

## 판정 규칙 (P012 와 같다)
- 부호 확실성: 사람을 다시 뽑아(지원금 있음·없음 짝을 함께) 2,000번 계산했을 때 부호가 그대로인 비율.
  97.5% 이상이어야 방향을 말한다.
- 쌍체 부호검정: 사람 단위로 오른 사람과 내린 사람의 수(동점은 버린다).
- 수준: 실측 값(또는 범위)이 시뮬레이션 95% 구간과 겹치는가.

## 정책 전 날짜
지급일(정책 파일 effective_from) 전 날짜의 두 시뮬레이션 격차(D4)를 먼저 낸다. 같은 사람이라도
생성이 흔들리면 이 격차가 생기고, 그만큼은 지원금 뒤 격차도 잡음일 수 있다. 정책 뒤 효과는 그대로(1차)
적고, 정책 전 격차를 뺀 값을 민감도로 함께 적는다.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_p012_two_arm import sign_test_p  # noqa: E402
from score_p013_usage import load_dossier, score_usage  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "data/experiments/P013_indicator_contract.json"
POLICY = ROOT / "data/experiments/P013_v53_policy_20260926.json"
CERTAIN = 0.975
BOOT = 2000
SEED = 20261003

# 정답지 그림 3 주석의 업종 정의를 우리 소분류로 옮긴다. 정의에 없는 소분류는 넣지 않는다
# (예: 화장품·시계·생활용품은 '잡화'로 볼 수도 있으나 원문이 열거하지 않았다).
GROUPS = {
    "준내구재": ["가구", "문구", "서점", "안경", "의류"],
    "필수재": ["슈퍼마켓", "편의점", "식료품", "정육", "청과", "수산"],
    "대면서비스": ["미용실", "네일", "피부관리", "욕탕·신체관리", "스포츠", "헬스장",
              "노래방", "유원지·오락", "당구", "PC방", "볼링"],
    "음식업": ["한식", "일식", "중식", "양식", "아시안", "기타식사", "분식", "치킨", "피자",
             "구내식당·뷔페", "베이커리", "카페"],
}
FACE = ("대면서비스", "음식업")
NONFACE = ("준내구재", "필수재")


def load_ledger(path: Path):
    rows = defaultdict(dict)
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["aid"]][r["day"]] = r
    return rows


def boot_weights(n: int, b: int = BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    out = []
    done = 0
    while done < b:
        k = min(250, b - done)
        out.append(rng.multinomial(n, np.full(n, 1.0 / n), size=k).astype(np.float64))
        done += k
    return np.vstack(out)


def ratio_stats(off: np.ndarray, on: np.ndarray, W: np.ndarray):
    """(증가율 %, lo, hi, 부호확실성) — 짝을 함께 재표집."""
    so, sn = off.sum(), on.sum()
    if so <= 0:
        return None, None, None, None
    base = 100 * (sn / so - 1)
    bo, bn = W @ off, W @ on
    ok = bo > 0
    vals = np.sort(100 * (bn[ok] / bo[ok] - 1))
    if len(vals) == 0:
        return base, None, None, None
    same = float(np.mean((vals > 0) == (base > 0)))
    return base, float(vals[int(len(vals) * 0.025)]), float(vals[int(len(vals) * 0.975)]), same


def per_grant_stats(diff: np.ndarray, grant: np.ndarray, W: np.ndarray):
    """(Σ차이 / Σ지원금 %, lo, hi, 부호확실성)."""
    g = grant.sum()
    if g <= 0:
        return None, None, None, None
    base = 100 * diff.sum() / g
    bd, bg = W @ diff, W @ grant
    vals = np.sort(100 * bd / bg)
    same = float(np.mean((vals > 0) == (base > 0)))
    return base, float(vals[int(len(vals) * 0.025)]), float(vals[int(len(vals) * 0.975)]), same


def gap_stats(oa, na, ob, nb, W):
    """(A 증가율 − B 증가율 %p, lo, hi, 부호확실성)."""
    if oa.sum() <= 0 or ob.sum() <= 0:
        return None, None, None, None
    base = 100 * (na.sum() / oa.sum() - 1) - 100 * (nb.sum() / ob.sum() - 1)
    boa, bna, bob, bnb = W @ oa, W @ na, W @ ob, W @ nb
    ok = (boa > 0) & (bob > 0)
    vals = np.sort(100 * (bna[ok] / boa[ok] - 1) - 100 * (bnb[ok] / bob[ok] - 1))
    same = float(np.mean((vals > 0) == (base > 0)))
    return base, float(vals[int(len(vals) * 0.025)]), float(vals[int(len(vals) * 0.975)]), same


def signs(off, on):
    up = int(np.sum(on > off)); dn = int(np.sum(on < off))
    return up, dn, int(len(off) - up - dn)


def verdict(base, lo, hi, conf, truth_sign=1, truth_range=None):
    if base is None or conf is None:
        return "관측없음"
    if conf < CERTAIN or base == 0:
        return "판정불가"
    same = (base > 0) == (truth_sign > 0)
    lvl = ""
    if truth_range is not None and lo is not None:
        a, b = truth_range
        lvl = " · 수준 안" if (lo <= b and hi >= a) else " · 수준 밖"
    return ("일치" if same else "불일치") + lvl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--anchor-map", default="", help="{aid: 하루 소비 기준액} — 소득분위(H1·H2)용, 정책 전 속성")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--policy-file", default=str(POLICY), help="런에 쓴 정책 파일(지급일·배정액)")
    ap.add_argument("--dossier-on", default="", help="지원금 있는 쪽 기억 모음(dossier.jsonl) — U4·H3 용")
    ap.add_argument("--dossier-off", default="", help="지원금 없는 쪽 기억 모음 — U4 용")
    a = ap.parse_args()
    d = Path(a.dir)
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    pol = json.loads(Path(a.policy_file).read_text(encoding="utf-8"))
    pay = date.fromisoformat(pol["effective_from"])

    on, off = load_ledger(d / "on.policy.ledger.jsonl"), load_ledger(d / "off.policy.ledger.jsonl")
    aids = sorted(set(on) & set(off))
    days = sorted(set(next(iter(on.values())).keys()))
    if any(set(on[x]) != set(days) or set(off[x]) != set(days) for x in aids):
        raise SystemExit("두 시뮬레이션의 날짜가 사람마다 같지 않다 — 대조 무효")
    pre = [x for x in days if date.fromisoformat(x) < pay]
    post = [x for x in days if date.fromisoformat(x) >= pay]
    n = len(aids)
    W = boot_weights(n)

    def series(arm, dset, key, subs=None):
        src = on if arm == "on" else off
        out = np.zeros(n)
        for i, x in enumerate(aids):
            for dy in dset:
                r = src[x][dy]
                if key == "total":
                    out[i] += r["offline_spent"] + r["online_spent"]
                elif key == "eligible":
                    out[i] += r["eligible_offline_spent"]
                elif key == "excluded":
                    out[i] += r["offline_spent"] + r["online_spent"] - r["eligible_offline_spent"]
                elif key == "group":
                    eb = r.get("eligible_by_sub") or {}
                    out[i] += sum(eb.get(s, 0) for s in subs)
        return out

    grant_recv = np.array([max(on[x][dy]["grant_received_cumulative"] for dy in days) for x in aids], dtype=float)
    # 분모는 **배정된 지원금**이다 — 정답지의 26.2~36.1%·주별 비율은 받은 시점과 무관하게 투입재원 전체로
    # 나눈 값이다. 지급을 실제 일정대로 나누면 창 끝까지 못 받은 사람이 있으므로 받은 금액으로 나누면 부풀려진다.
    _alloc = sorted({int(v) for v in (pol.get("decile_grants") or {}).values()})
    if len(_alloc) != 1:
        raise SystemExit("분위별 지급액이 같지 않다 — 배정액 분모를 사람마다 계산하도록 고쳐야 한다")
    grant = np.full(len(aids), float(_alloc[0]))
    grant_off = sum(max(off[x][dy]["grant_received_cumulative"] for dy in days) for x in aids)
    spent_grant = np.array([sum(on[x][dy]["grant_spent_today"] for dy in days) for x in aids], dtype=float)

    rows, P = [], print
    P("# P013 — 같은 사람 %d명 · 지원금 있음 vs 없음" % n)
    P("  날짜 %s ~ %s · 지급 전 %d일 · 지급 뒤 %d일 (지급일 %s)" % (days[0], days[-1], len(pre), len(post), pay))
    P("  지급: 배정 %s원 (1인 %s원) · 창 안에 받은 사람 %d명 (%.1f%%) · 받은 금액 %s원 · 지원금 없음 %s원 · 받은 금액 중 사용 %.1f%%"
      % ("{:,.0f}".format(grant.sum()), "{:,.0f}".format(grant[0] if n else 0), int(np.sum(grant_recv > 0)),
         100 * np.mean(grant_recv > 0) if n else 0, "{:,.0f}".format(grant_recv.sum()), "{:,.0f}".format(grant_off),
         100 * spent_grant.sum() / grant_recv.sum() if grant_recv.sum() else 0))
    gate = {"grant_paid": bool(grant_recv.sum() > 0 and grant_off == 0), "pre_days": len(pre), "post_days": len(post),
            "received_share": float(np.mean(grant_recv > 0)) if n else 0.0}
    P()

    # D4 — 정책 전 격차(잡음 관문)
    for key, nm in (("total", "총지출"), ("eligible", "사용처 지출")):
        if not pre:
            break
        o, nn = series("off", pre, key), series("on", pre, key)
        b, lo, hi, cf = ratio_stats(o, nn, W)
        up, dn, eq = signs(o, nn)
        rows.append({"id": "D4", "name": "정책 전 격차 — " + nm, "sim": b, "ci": [lo, hi], "sign_conf": cf,
                     "sign": [up, dn, eq], "sign_test_p": sign_test_p(up, dn),
                     "status": "잡음 관문: 0 이 구간 안" if (lo is not None and lo <= 0 <= hi) else "잡음 관문: 0 이 구간 밖"})
        P("  D4  정책 전 %s 격차 %+.2f%% (95%% %+.1f ~ %+.1f) · 부호확실 %.1f%% · %s"
          % (nm, b, lo, hi, 100 * cf, rows[-1]["status"]))
    gate["pre_gap_ok"] = all(r["status"].endswith("구간 안") for r in rows if r["id"] == "D4")

    # D2 — 사용처 추가 지출 / 지원금 (지급 뒤 누적), 주별
    ind = {i["id"]: i for i in c["indicators"]}
    truth_weeks = ind["D2"]["truth_by_weeks"]
    weeks = [post[k:k + 7] for k in range(0, len(post), 7)]
    full_weeks = [w for w in weeks if len(w) == 7]
    cum = []
    eo_all, en_all = series("off", post, "eligible"), series("on", post, "eligible")
    for k in range(1, len(full_weeks) + 1):
        dset = [x for w in full_weeks[:k] for x in w]
        diff = series("on", dset, "eligible") - series("off", dset, "eligible")
        b, lo, hi, cf = per_grant_stats(diff, grant, W)
        tr = truth_weeks.get("%d주" % k) or truth_weeks.get("%d주(5/11~)" % k)
        st = verdict(b, lo, hi, cf, 1, tr)
        rows.append({"id": "D2", "name": "사용처 추가 지출 / 지원금 — %d주 누적" % k, "sim": b, "ci": [lo, hi],
                     "sign_conf": cf, "truth_range": tr, "status": st, "weeks": k})
        cum.append(b)
        P("  D2  %d주 누적 사용처 추가 지출 / 지원금 %+.2f%% (95%% %+.2f ~ %+.2f) · 실측 %s%% · %s"
          % (k, b, lo, hi, tr, st))
        if pre:
            po, pn = series("off", pre, "eligible"), series("on", pre, "eligible")
            per_day_gap = (pn - po) / len(pre)
            adj = diff - per_day_gap * len(dset)
            ab, alo, ahi, acf = per_grant_stats(adj, grant, W)
            P("      (민감도: 정책 전 하루 격차를 뺀 값 %+.2f%%, 95%% %+.2f ~ %+.2f)" % (ab, alo, ahi))
            rows[-1]["pre_adjusted"] = [ab, alo, ahi, acf]
    if not full_weeks:
        P("  D2  지급 뒤 7일이 채워지지 않았다 — 주 단위 실측과 맞대지 않는다")
    # 지급 뒤 전체(주가 다 차지 않아도) 값 — 참고
    b, lo, hi, cf = per_grant_stats(en_all - eo_all, grant, W)
    rows.append({"id": "D2*", "name": "사용처 추가 지출 / 지원금 — 지급 뒤 전 기간(%d일)" % len(post),
                 "sim": b, "ci": [lo, hi], "sign_conf": cf, "status": verdict(b, lo, hi, cf)})
    P("  D2* 지급 뒤 %d일 전체 %+.2f%% (95%% %+.2f ~ %+.2f) · 부호확실 %.1f%%" % (len(post), b, lo, hi, 100 * (cf or 0)))

    # D3 — 주별 모양
    if len(full_weeks) >= 2:
        w1 = series("on", full_weeks[0], "eligible") - series("off", full_weeks[0], "eligible")
        w2 = series("on", full_weeks[1], "eligible") - series("off", full_weeks[1], "eligible")
        diff = w2 - w1
        bd = W @ diff
        cf = float(np.mean((bd > 0) == (diff.sum() > 0)))
        rows.append({"id": "D3", "name": "2주차 효과 > 1주차 효과", "sim": float(diff.sum()) / n,
                     "sign_conf": cf, "status": ("판정불가" if cf < CERTAIN else ("일치" if diff.sum() > 0 else "불일치"))})
        P("  D3  2주차 − 1주차 효과 1인 %+.0f원 · 부호확실 %.1f%% · %s" % (diff.sum() / n, 100 * cf, rows[-1]["status"]))
    else:
        rows.append({"id": "D3", "name": "주별 모양", "status": "미측정", "why": "지급 뒤 2주 이상 돌려야 잰다"})

    # 총지출·제외업종(참고)
    for key, nm in (("total", "총지출(지급 뒤)"), ("eligible", "사용처 지출(지급 뒤)"), ("excluded", "사용처 밖 지출(지급 뒤)")):
        o, nn = series("off", post, key), series("on", post, key)
        b, lo, hi, cf = ratio_stats(o, nn, W)
        up, dn, eq = signs(o, nn)
        rows.append({"id": "T-" + key, "name": nm, "sim": b, "ci": [lo, hi], "sign_conf": cf,
                     "sign": [up, dn, eq], "sign_test_p": sign_test_p(up, dn), "status": verdict(b, lo, hi, cf)})
        P("  참고 %s %+.2f%% (95%% %+.1f ~ %+.1f) · 부호확실 %.1f%% · 오름:내림 %d:%d"
          % (nm, b, lo, hi, 100 * (cf or 0), up, dn))

    # S1~S4 업종 묶음, S5 순위, S6 대면/비대면
    P()
    grp = {}
    for gi, (gid, gname) in enumerate((("S1", "준내구재"), ("S2", "필수재"), ("S3", "대면서비스"), ("S4", "음식업"))):
        o, nn = series("off", post, "group", GROUPS[gname]), series("on", post, "group", GROUPS[gname])
        b, lo, hi, cf = ratio_stats(o, nn, W)
        t = ind[gid]["truth"]
        st = verdict(b, lo, hi, cf, 1, [t, t] if t is not None else None)
        grp[gname] = (o, nn, b)
        rows.append({"id": gid, "name": gname, "sim": b, "ci": [lo, hi], "sign_conf": cf, "truth": t,
                     "status": st, "off_total": float(o.sum()), "people_with_spend": int(np.sum((o + nn) > 0))})
        P("  %s  %-6s %+.2f%% (95%% %+.1f ~ %+.1f) · 실측 %+.1f%%p · 부호확실 %.1f%% · %s · 지출 있는 사람 %d"
          % (gid, gname, b if b is not None else float("nan"), lo if lo is not None else float("nan"),
             hi if hi is not None else float("nan"), t, 100 * (cf or 0), st, rows[-1]["people_with_spend"]))
    order_truth = ["준내구재", "필수재", "대면서비스", "음식업"]
    sim_vals = [grp[g][2] for g in order_truth]
    if all(v is not None for v in sim_vals):
        ranks = sorted(order_truth, key=lambda g: -grp[g][2])
        rows.append({"id": "S5", "name": "업종 순위", "sim_order": ranks, "truth_order": order_truth,
                     "status": "일치" if ranks == order_truth else "순서 다름"})
        P("  S5  시뮬 순서 %s · 실측 순서 %s" % (" > ".join(ranks), " > ".join(order_truth)))
    oa = grp["준내구재"][0] + grp["필수재"][0]; na = grp["준내구재"][1] + grp["필수재"][1]
    ob = grp["대면서비스"][0] + grp["음식업"][0]; nb = grp["대면서비스"][1] + grp["음식업"][1]
    b, lo, hi, cf = gap_stats(oa, na, ob, nb, W)
    st = verdict(b, lo, hi, cf, 1)
    rows.append({"id": "S6", "name": "비대면 업종 − 대면 업종 효과 격차", "sim": b, "ci": [lo, hi],
                 "sign_conf": cf, "truth_gap": 6.1, "status": st})
    P("  S6  (준내구재+필수재) − (대면서비스+음식업) %+.2f%%p (95%% %+.1f ~ %+.1f) · 부호확실 %.1f%% · %s"
      % (b, lo, hi, 100 * (cf or 0), st))

    # U — 지원금 사용 기전(행안부·서울연구원·KDI 연구 Ⅱ). 실측·대응표는 계약서에서만 읽는다.
    P()
    sec = d / "on.sector.ledger.jsonl"
    urows = score_usage(on_pol=on, off_pol=off, on_sec=load_ledger(sec) if sec.is_file() else None, aids=aids,
                        post=post, grant=grant, W=W, contract=c,
                        dossier_on=load_dossier(a.dossier_on) if a.dossier_on else None,
                        dossier_off=load_dossier(a.dossier_off) if a.dossier_off else None)
    for r in urows:
        if r.get("sim") is None:
            P("  %-3s %s · %s%s" % (r["id"], r["name"], r["status"], (" — " + r["why"]) if r.get("why") else ""))
        elif r["id"] == "U1":
            P("  U1  %d주 누적 소진 %.1f%% (95%% %.1f ~ %.1f) · 서울 실측 %s%% · 전국 %s%% · %s"
              % (r["weeks"], r["sim"], r["ci"][0], r["ci"][1], r["truth"], r["truth_national"], r["status"]))
        elif r["id"] == "U2":
            P("  U2  사용처 구성 유사도 %.1f%% (95%% %.1f ~ %.1f, 상한 %.1f) · 상위 시뮬 %s / 실측 %s · %s"
              % (r["sim"], r["ci"][0], r["ci"][1], r["ceiling"], ">".join(r["top_sim"]), ">".join(r["top_truth"]), r["status"]))
        elif r["id"] == "U4":
            P("  U4  %s %.1f%% (95%% %.1f ~ %.1f) vs 일반 %.1f%% · 실측 %.1f vs %.1f · 부호확실 %.1f%% · %s"
              % (r["name"], r["sim"], r["ci"][0], r["ci"][1], r["sim_general"], r["truth"], r["truth_general"], 100 * r["sign_conf"], r["status"]))
        else:
            P("  %-3s %s %+.3f%s · %s" % (r["id"], r["name"], r["sim"],
                                       (" (95%% %+.2f ~ %+.2f)" % tuple(r["ci"])) if r.get("ci") and r["ci"][0] is not None else "", r["status"]))
    rows.extend(urows)

    # H1·H2 소득분위(정책 전 기준액)
    if a.anchor_map:
        am = json.loads(Path(a.anchor_map).read_text(encoding="utf-8"))
        have = [x for x in aids if (am.get(x) or 0) > 0]
        order = sorted(have, key=lambda x: am[x])
        q = {x: 1 + (i * 5) // len(order) for i, x in enumerate(order)}
        idx = {x: i for i, x in enumerate(aids)}
        eff_total = series("on", post, "total") - series("off", post, "total")
        per_q = {k: float(np.mean([eff_total[idx[x]] for x in have if q[x] == k])) for k in range(1, 6)}
        top = max(per_q, key=per_q.get)
        mono = all(per_q[k] >= per_q[k + 1] for k in range(1, 5)) or all(per_q[k] <= per_q[k + 1] for k in range(1, 5))
        rows.append({"id": "H1", "name": "소득분위별 1인 효과(원)", "by_quintile": per_q,
                     "status": ("일치" if top == 1 and not mono else ("1분위 최대·단조" if top == 1 else "1분위가 최대가 아님")),
                     "n_quintiled": len(have)})
        P()
        P("  H1  분위별 1인 효과(지급 뒤 총지출, 원): %s · %s"
          % (" · ".join("%d분위 %+.0f" % (k, v) for k, v in per_q.items()), rows[-1]["status"]))
        if full_weeks:
            w = full_weeks[0]
            e1 = series("on", w, "total") - series("off", w, "total")
            lo_ = np.mean([e1[idx[x]] for x in have if q[x] <= 2]); hi_ = np.mean([e1[idx[x]] for x in have if q[x] >= 4])
            rows.append({"id": "H2", "name": "지급 첫 주 1·2분위 효과 > 4·5분위", "low": float(lo_), "high": float(hi_),
                         "status": "일치" if lo_ > hi_ else "불일치"})
            P("  H2  첫 주 1인 효과 1·2분위 %+.0f원 vs 4·5분위 %+.0f원 · %s" % (lo_, hi_, rows[-1]["status"]))
    else:
        rows.append({"id": "H1", "status": "미측정", "why": "--anchor-map 이 없다"})

    if a.json_out:
        io.open(a.json_out, "w", encoding="utf-8", newline="\n").write(json.dumps(
            {"n_agents": n, "days": days, "pre_days": pre, "post_days": post, "gate": gate,
             "grant_allocated": float(grant.sum()), "grant_received": float(grant_recv.sum()),
             "grant_spent": float(spent_grant.sum()),
             "groups": GROUPS, "rows": rows}, ensure_ascii=False, indent=1, default=float))
        P("→ %s" % a.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
