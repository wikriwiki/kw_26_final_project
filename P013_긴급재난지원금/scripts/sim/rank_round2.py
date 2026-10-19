# -*- coding: utf-8 -*-
"""2차 후보 A단계 판정 — 사전등록한 기준을 기계로 적용한다.

판정을 손으로 하면 결과를 보고 기준이 흔들린다. 그래서 기준을 코드로 굳혀
두고 결과 JSON 만 먹인다. 기준은 `scoring_table.json` 의 ROUND2_PREREG 에
2026-09-18 에 적어 둔 것과 같다.

    ① 결함 ① — P012_HIGH 런의 P012-1 부호가 + 여야 한다
    ② 결함 ② — LOCAL_VOUCHER 런의 LV-3 부호가 − 여야 한다
    ③ 실격   — v5 가 통과한 영가설(LV-1 총소비 동등성 밴드)을 깨면 제외.
               **같은 n 끼리만 대조한다** — 표본이 다르면 밴드 폭과 CI 폭이
               함께 달라져 공정한 비교가 아니다.
    ④ 스코어 — 둘 다 충족 > 하나 충족 > 전체 적중 수 합계

    python scripts/sim/rank_round2.py /data/stage7
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CANDS = ("v5", "v7", "v8", "v9")


def load(d: Path, tag: str) -> dict | None:
    f = d / f"{tag}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def ind(res: dict | None, iid: str) -> dict | None:
    if not res:
        return None
    for r in res.get("results") or []:
        if r.get("id") == iid:
            return r
    return None


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/stage7")
    rows = []
    for c in CANDS:
        thr = load(d, f"{c}_thr105")
        gu = load(d, f"{c}_gu")
        p1, lv1, lv3 = ind(thr, "P012-1"), ind(gu, "LV-1"), ind(gu, "LV-3")
        hits = ((thr or {}).get("hits") or 0) + ((gu or {}).get("hits") or 0)
        rows.append({
            "cand": c, "thr": thr, "gu": gu,
            "p1": p1, "lv1": lv1, "lv3": lv3, "hits": hits,
            "ready": thr is not None and gu is not None,
        })

    print("=" * 78)
    print("2차 A단계 — 사전등록 기준 적용")
    print("=" * 78)
    print(f"{'후보':<6}{'P012-1':>10}{'기준①':>7}{'LV-3':>10}{'기준②':>7}"
          f"{'LV-1 밴드':>11}{'적중':>6}  상태")
    for r in rows:
        if not r["ready"]:
            have = [k for k in ("thr", "gu") if r[k] is not None]
            print(f"{r['cand']:<6}{'—':>10}{'—':>7}{'—':>10}{'—':>7}"
                  f"{'—':>11}{'—':>6}  미완({'없음' if not have else '+'.join(have)})")
            continue
        p1m = r["p1"]["mean"] if r["p1"] else None
        lv3m = r["lv3"]["mean"] if r["lv3"] else None
        c1 = (p1m is not None and p1m > 0)
        c2 = (lv3m is not None and lv3m < 0)
        # 실격 — LV-1 동등성 밴드
        lv1 = r["lv1"]
        band_ok = None
        if lv1 and isinstance(lv1.get("base"), (int, float)) and lv1.get("ci"):
            band = 0.10 * abs(lv1["base"])
            lo, hi = lv1["ci"]
            band_ok = (-band <= lo and hi <= band)
        r.update(c1=c1, c2=c2, band_ok=band_ok)
        print(f"{r['cand']:<6}{p1m:>+10,.0f}{'O' if c1 else 'X':>7}"
              f"{lv3m:>+10.4f}{'O' if c2 else 'X':>7}"
              f"{('통과' if band_ok else '초과') if band_ok is not None else '—':>11}"
              f"{r['hits']:>6}  완료")

    ready = [r for r in rows if r["ready"]]
    if len(ready) < len(CANDS):
        print()
        print(f"판정 보류 — {len(ready)}/{len(CANDS)} 후보만 두 런을 마쳤다.")
        print("v5 의 두 런이 없으면 실격 규칙의 기준선이 없어 아무도 탈락시킬 수 없다.")
        return 0

    base = next((r for r in ready if r["cand"] == "v5"), None)
    print()
    if base is None or base["band_ok"] is None:
        print("v5 결과가 없어 실격 규칙을 적용할 수 없다.")
        return 0
    if base["band_ok"]:
        dq = [r["cand"] for r in ready if r["cand"] != "v5" and r["band_ok"] is False]
        print(f"실격 규칙 발동 — v5 는 같은 n 에서 LV-1 밴드를 통과했다. 탈락: {dq or '없음'}")
    else:
        dq = []
        print("실격 규칙 미발동 — v5 도 같은 n 에서 LV-1 밴드를 깼다. "
              "기준선이 통과하지 못했으므로 이 규칙으로는 아무도 탈락시키지 않는다.")

    pool = [r for r in ready if r["cand"] not in dq]
    pool.sort(key=lambda r: (-(int(r["c1"]) + int(r["c2"])), -r["hits"], r["cand"]))
    print()
    print("순위 (둘 다 충족 > 하나 충족 > 적중 수)")
    for i, r in enumerate(pool, 1):
        met = int(r["c1"]) + int(r["c2"])
        print(f"  {i}. {r['cand']}  기준 {met}/2, 적중 {r['hits']}")
    win = pool[0]["cand"] if pool else None
    print()
    if win == "v5":
        print("→ **v5 를 그대로 둔다.** 추가 문장이 v5 를 이기지 못했다. 그것도 결과다.")
    else:
        print(f"→ A단계 승자: **{win}**. B단계에서 v5 와 훈련 정책 4개 n=500 으로 붙인다.")
        print("   2차 승자는 홀드아웃 검증을 받지 못한다는 사실을 결과와 함께 적는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
