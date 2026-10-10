# 정책 없는 쪽 장보기 기본률 대 페르소나(BDC) 소비 구성(읽기 전용). python grocery_base.py <dossier.jsonl> [...]
import json, sys, collections, statistics as st
GRO = {"슈퍼마켓", "식료품", "청과", "정육", "수산", "장보기"}
for path in sys.argv[1:]:
    n = 0; pdays = 0; trips = 0; gro_amt = 0; tot_amt = 0; share_p = []; share_s = []; keys = collections.Counter()
    for l in open(path, encoding="utf-8"):
        r = json.loads(l); n += 1
        prof = r.get("profile") or {}
        try:
            top = json.loads(prof.get("spending_top_wd_json") or "{}")
        except Exception:
            top = {}
        if isinstance(top, list):
            top = {x.get("category") or x.get("name"): x.get("share") or x.get("ratio") or x.get("pct") for x in top if isinstance(x, dict)}
        for k in top: keys[k] += 1
        mart = sum(float(v or 0) for k, v in top.items() if k and any(s in k for s in ("마트", "슈퍼", "식료", "농", "축산", "수산", "청과", "정육", "유통")))
        g = t = 0
        for p in r.get("plans") or []:
            pdays += 1
            for i in p.get("items") or []:
                a = float(i.get("actual_spent") or 0)
                if a <= 0: continue
                t += a
                if i.get("sub_category") in GRO or i.get("category") == "마트":
                    g += a; trips += 1
        gro_amt += g; tot_amt += t
        if t > 0: share_s.append(g / t); share_p.append(mart if mart <= 1 else mart / 100)
    print(f"== {path}: {n}명 · 사람-날 {pdays}")
    print(f"  장보기 결제 {trips}건 = 사람-날당 {trips/max(1,pdays):.3f}건(주당 {7*trips/max(1,pdays):.2f}건) · 지출 중 장보기 몫 {100*gro_amt/max(1,tot_amt):.1f}%")
    if share_p:
        print(f"  페르소나 소비 구성의 마트·식료 몫 평균 {100*st.mean(share_p):.1f}% (중앙 {100*st.median(share_p):.1f}%) vs 시뮬 개인 장보기 몫 평균 {100*st.mean(share_s):.1f}% (중앙 {100*st.median(share_s):.1f}%)")
    print("  소비 구성 키(상위):", keys.most_common(25))
