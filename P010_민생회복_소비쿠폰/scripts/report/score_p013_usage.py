"""P013 지원금 사용 기전(U1~U4)과 참고 지표(H3, R1) — 행안부·서울연구원·KDI 연구 Ⅱ 실측과 맞댄다.

score_p013_two_arm.py 가 부른다. 실측 수치와 업종 대응표는 지표 계약서에서만 읽는다
(data/experiments/P013_indicator_contract.json — 시뮬레이션 결과를 보기 전 2026-10-03 에 고정).

  U1 서울 주별 누적 소진율 — 지원금으로 낸 금액 누적 / 배정 지원금 (정책 원장 grant_spent_today)
  U2 지원금 사용처 업종 구성 — 업종 원장 funded_by_sub 를 행안부 업종으로 묶어 같은 주들의 구성과 맞댄다
  U3 업종 매출 중 지원금 비율의 업종 순서 — Σfunded_by_sub / Σby_sub, 스피어만
  U4 지원금이 집 가까이에서 쓰이는가 — 기억 모음(dossier) 계획 항목의 POI 동 코드 대 거주 동 코드
  H3 (참고) 65세 미만 vs 이상의 지원금 대비 추가 소비
  R1 (참고) 업종별 효과 순위 대 KDI 연구 Ⅱ 전후 증감률 차이 순위

모든 구간은 사람 단위 재표집(W: 재표집 가중치 행렬, 지원금 있음·없음 짝을 함께)으로 낸다.
"""
from __future__ import annotations

import io
import json
from collections import defaultdict

import numpy as np

CERTAIN = 0.975


def week_days(post: list[str]) -> list[list[str]]:
    """지급일부터 7일씩 — 온전한 주만."""
    return [w for w in (post[k:k + 7] for k in range(0, len(post), 7)) if len(w) == 7]


def ranks(x: np.ndarray) -> np.ndarray:
    """평균 순위(동점 평균). 마지막 축 기준."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, axis=-1, kind="mergesort")
    r = np.empty_like(x)
    np.put_along_axis(r, order, np.arange(x.shape[-1], dtype=float) + 1, axis=-1)
    if x.ndim == 1:
        for v in np.unique(x):
            m = x == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return np.stack([ranks(row) for row in x])


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _ci(vals: np.ndarray):
    v = np.sort(vals[np.isfinite(vals)])
    if len(v) == 0:
        return None, None
    return float(v[int(len(v) * 0.025)]), float(v[min(len(v) - 1, int(len(v) * 0.975))])


def cat_matrix(ledger, aids, days, key, mapping, rest=None):
    """사람 x 업종 금액 행렬. rest 가 있으면 대응표 밖 소분류를 그 이름의 칸에 모은다."""
    cats = list(mapping) + ([rest] if rest else [])
    sub2cat = {s: k for k, subs in mapping.items() for s in subs}
    m = np.zeros((len(aids), len(cats)))
    idx = {k: j for j, k in enumerate(cats)}
    for i, x in enumerate(aids):
        for dy in days:
            for s, v in ((ledger[x][dy].get(key) or {}).items()):
                k = sub2cat.get(s, rest)
                if k is not None and v:
                    m[i, idx[k]] += v
    return cats, m


def u1(on_pol, aids, post, grant, W, ind):
    rows = []
    truth = list(ind["truth_by_weeks"].values())
    spent_by_day = {dy: np.array([on_pol[x][dy]["grant_spent_today"] for x in aids], dtype=float) for dy in post}
    cum = np.zeros(len(aids))
    for k, w in enumerate(week_days(post), start=1):
        for dy in w:
            cum = cum + spent_by_day[dy]
        base = 100 * cum.sum() / grant.sum()
        lo, hi = _ci(100 * (W @ cum) / (W @ grant))
        t = truth[k - 1] if k - 1 < len(truth) else None
        st = "실측 없음" if t is None else ("수준 안" if lo <= t <= hi else ("수준 밖 · 시뮬 높음" if base > t else "수준 밖 · 시뮬 낮음"))
        nat = list(ind["cross_check_national"]["값"].values())
        rows.append({"id": "U1", "name": "지원금 누적 소진율 — %d주" % k, "weeks": k, "sim": base, "ci": [lo, hi],
                     "truth": t, "truth_national": nat[k - 1] if k - 1 < len(nat) else None, "status": st})
    return rows


def u2(on_sec, aids, post, W, ind):
    weeks = week_days(post)
    if not weeks:
        return [{"id": "U2", "name": ind["name"], "status": "미측정", "why": "지급 뒤 온전한 주가 없다"}]
    mapping = ind["map_sim_sub_to_category"]
    amounts = ind["truth_weekly_amounts_억원"]
    k = min(len(weeks), len(ind["truth_weeks"]))
    days = [d for w in weeks[:k] for d in w]
    cats, m = cat_matrix(on_sec, aids, days, "funded_by_sub", mapping, rest="기타")
    t = np.array([sum(amounts[c][:k]) for c in cats], dtype=float)
    t = t / t.sum()
    s = m.sum(axis=0)
    if s.sum() <= 0:
        return [{"id": "U2", "name": ind["name"], "status": "관측없음", "why": "지원금으로 낸 금액이 없다"}]
    p = s / s.sum()
    sim = 1 - 0.5 * np.abs(p - t).sum()
    bs = W @ m
    bp = bs / bs.sum(axis=1, keepdims=True)
    lo, hi = _ci(1 - 0.5 * np.abs(bp - t).sum(axis=1))
    ceiling = 1 - sum(t[j] for j, c in enumerate(cats) if c in mapping and not mapping[c])
    top_sim = [cats[j] for j in np.argsort(-p)[:3]]
    top_truth = [cats[j] for j in np.argsort(-t)[:3]]
    st = "일치" if set(top_sim[:2]) == set(top_truth[:2]) else "불일치"
    return [{"id": "U2", "name": "지원금 사용처 업종 구성 — 지급 뒤 %d주" % k, "weeks": k, "sim": 100 * sim,
             "ci": [100 * lo, 100 * hi], "ceiling": 100 * ceiling, "status": st,
             "share_sim": {c: round(100 * float(p[j]), 2) for j, c in enumerate(cats)},
             "share_truth": {c: round(100 * float(t[j]), 2) for j, c in enumerate(cats)},
             "top_sim": top_sim, "top_truth": top_truth}]


def u3(on_sec, aids, post, W, ind):
    weeks = week_days(post)
    mapping = ind["map_sim_sub_to_category"]
    truth = ind["truth_ratio_pct_weeks"]
    rows = []
    for k, w in enumerate(weeks[:len(ind["truth_weeks"])], start=1):
        cats, fm = cat_matrix(on_sec, aids, w, "funded_by_sub", mapping)
        _, tm = cat_matrix(on_sec, aids, w, "by_sub", mapping)
        keep = [j for j in range(len(cats)) if tm[:, j].sum() > 0]
        if len(keep) < 5:
            rows.append({"id": "U3", "name": ind["name"] + " — %d주" % k, "weeks": k, "status": "관측없음",
                         "why": "지출이 있는 업종이 5개 미만"})
            continue
        cats = [cats[j] for j in keep]
        fm, tm = fm[:, keep], tm[:, keep]
        tv = np.array([truth[c][k - 1] for c in cats], dtype=float)
        ratio = fm.sum(0) / tm.sum(0)
        rho = spearman(ratio, tv)
        bf, bt = W @ fm, W @ tm
        with np.errstate(divide="ignore", invalid="ignore"):
            br = np.where(bt > 0, bf / bt, np.nan)
        rr = np.array([spearman(row[np.isfinite(row)], tv[np.isfinite(row)]) if np.isfinite(row).sum() >= 5 else np.nan
                       for row in br], dtype=float)
        rr = rr[np.isfinite(rr)]
        pos = float(np.mean(rr > 0)) if len(rr) else 0.0
        lo, hi = _ci(rr)
        st = "일치" if (rho or 0) > 0 and pos >= CERTAIN else ("불일치" if (rho or 0) < 0 and (1 - pos) >= CERTAIN else "판정불가")
        rows.append({"id": "U3", "name": ind["name"] + " — %d주" % k, "weeks": k, "sim": rho, "ci": [lo, hi],
                     "sign_conf": pos if (rho or 0) > 0 else 1 - pos, "status": st, "n_categories": len(cats),
                     "ratio_sim": {c: round(100 * float(ratio[j]), 1) for j, c in enumerate(cats)},
                     "ratio_truth": {c: float(tv[j]) for j, c in enumerate(cats)}})
    return rows or [{"id": "U3", "name": ind["name"], "status": "미측정", "why": "지급 뒤 온전한 주가 없다"}]


def _num(v):
    if isinstance(v, bool):
        return 0
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0


def _funded(v, pid):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return 0
    return _num((v or {}).get(pid, 0)) if isinstance(v, dict) else 0


def load_dossier(path):
    out = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["aid"]] = r
    return out


def home_dong(aid: str) -> str:
    """에이전트 id 'AGT_<행정동 8자리>_…' 의 거주 동."""
    parts = aid.split("_")
    if len(parts) < 2 or not parts[1].isdigit() or len(parts[1]) != 8:
        raise ValueError("거주 동 코드를 읽을 수 없다: %s" % aid)
    return parts[1]


def locality_vectors(dossier, aids, days, pid, funded_only):
    """사람마다 (같은 동 금액, 같은 자치구 금액, 전체 금액). funded_only 면 지원금으로 낸 몫만."""
    dset = set(days)
    out = np.zeros((len(aids), 3))
    for i, x in enumerate(aids):
        h = home_dong(x)
        for plan in (dossier.get(x) or {}).get("plans") or []:
            if plan.get("day") not in dset:
                continue
            for it in plan.get("items") or []:
                amt = _funded(it.get("spent_from_policy"), pid) if funded_only else _num(it.get("actual_spent"))
                if amt <= 0:
                    continue
                dc = str(it.get("dong_code") or "")
                out[i, 2] += amt
                if dc == h:
                    out[i, 0] += amt
                if dc[:5] == h[:5]:
                    out[i, 1] += amt
    return out


def u4(dos_on, dos_off, aids, post, W, ind, pid="P013"):
    g = locality_vectors(dos_on, aids, post, pid, True)
    o = locality_vectors(dos_off, aids, post, pid, False)
    if g[:, 2].sum() <= 0 or o[:, 2].sum() <= 0:
        return [{"id": "U4", "name": ind["name"], "status": "관측없음", "why": "지원금 결제 또는 일반 지출이 없다"}]
    rows = []
    tr = ind["truth"]
    bg, bo = W @ g, W @ o
    for j, (lab, tg, tg_off) in enumerate((("같은 동", tr["같은 동: 지원금"], tr["같은 동: 일반 카드결제"]),
                                           ("같은 자치구", tr["같은 자치구: 지원금"], tr["같은 자치구: 일반 카드결제"]))):
        sg = 100 * g[:, j].sum() / g[:, 2].sum()
        so = 100 * o[:, j].sum() / o[:, 2].sum()
        dg = 100 * bg[:, j] / bg[:, 2]
        do = 100 * bo[:, j] / bo[:, 2]
        gap = dg - do
        conf = float(np.mean(gap > 0)) if sg - so > 0 else float(np.mean(gap < 0))
        lo, hi = _ci(dg)
        st = ("판정불가" if conf < CERTAIN else ("일치" if sg > so else "불일치"))
        st += " · 수준 안" if lo <= tg <= hi else " · 수준 밖"
        rows.append({"id": "U4", "name": "지원금의 %s 몫 vs 일반 소비" % lab, "sim": sg, "ci": [lo, hi], "sim_general": so,
                     "ci_general": list(_ci(do)), "truth": tg, "truth_general": tg_off, "sign_conf": conf, "status": st})
    return rows


def h3(on_pol, off_pol, dos_on, aids, post, grant, W):
    age = np.array([_num(((dos_on.get(x) or {}).get("profile") or {}).get("age")) for x in aids], dtype=float)
    eff = np.zeros(len(aids))
    for i, x in enumerate(aids):
        for dy in post:
            a, b = on_pol[x][dy], off_pol[x][dy]
            eff[i] += (a["offline_spent"] + a["online_spent"]) - (b["offline_spent"] + b["online_spent"])
    out = {}
    for lab, m in (("65세 미만", (age > 0) & (age < 65)), ("65세 이상", age >= 65)):
        if m.sum() == 0:
            return [{"id": "H3", "name": "(참고) 연령별 지원금 대비 추가 소비", "status": "관측없음"}]
        out[lab] = (float(eff[m].sum() / grant[m].sum()), int(m.sum()))
    young, old = (age > 0) & (age < 65), age >= 65
    bd = (W[:, young] @ eff[young]) / (W[:, young] @ grant[young]) - (W[:, old] @ eff[old]) / (W[:, old] @ grant[old])
    base = out["65세 미만"][0] - out["65세 이상"][0]
    conf = float(np.mean((bd > 0) == (base > 0)))
    st = "판정불가" if conf < CERTAIN else ("같은 방향" if base > 0 else "반대 방향")
    return [{"id": "H3", "name": "(참고) 지원금 대비 추가 총지출 — 65세 미만 vs 이상", "by_age": out,
             "sim": 100 * base, "sign_conf": conf, "status": st, "scored": False}]


def r1(on_pol, off_pol, aids, post, W, ind):
    mapping = ind["map_sim_sub_to_category"]
    _, mo = cat_matrix(off_pol, aids, post, "eligible_by_sub", mapping)
    cats, mn = cat_matrix(on_pol, aids, post, "eligible_by_sub", mapping)
    keep = [j for j in range(len(cats)) if mo[:, j].sum() > 0]
    if len(keep) < 5:
        return [{"id": "R1", "name": ind["name"], "status": "관측없음", "scored": False}]
    cats = [cats[j] for j in keep]
    eff = 100 * (mn[:, keep].sum(0) / mo[:, keep].sum(0) - 1)
    tv = np.array([ind["truth_pp"][c] for c in cats], dtype=float)
    rho = spearman(eff, tv)
    return [{"id": "R1", "name": ind["name"], "sim": rho, "n_categories": len(cats), "scored": False,
             "status": "참고(판정 안 함)", "effect_sim": {c: round(float(eff[j]), 1) for j, c in enumerate(cats)}}]


def score_usage(*, on_pol, off_pol, on_sec, aids, post, grant, W, contract, dossier_on=None, dossier_off=None):
    ind = {i["id"]: i for i in contract["indicators"]}
    rows = []
    rows += u1(on_pol, aids, post, grant, W, ind["U1"])
    if on_sec is not None:
        rows += u2(on_sec, aids, post, W, ind["U2"])
        rows += u3(on_sec, aids, post, W, ind["U3"])
    else:
        rows += [{"id": k, "name": ind[k]["name"], "status": "미측정", "why": "업종 원장(on.sector.ledger.jsonl)이 없다"} for k in ("U2", "U3")]
    if dossier_on is not None and dossier_off is not None:
        rows += u4(dossier_on, dossier_off, aids, post, W, ind["U4"])
        rows += h3(on_pol, off_pol, dossier_on, aids, post, grant, W)
    else:
        rows += [{"id": "U4", "name": ind["U4"]["name"], "status": "미측정", "why": "기억 모음(dossier)을 주지 않았다"}]
    rows += r1(on_pol, off_pol, aids, post, W, ind["R1"])
    return rows
