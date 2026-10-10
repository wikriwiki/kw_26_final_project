# P016 시험 전면 검수(읽기 전용) — 보존된 기억 모음(dossier)만 읽는다. 2026-10-11.
#   python p016_full_audit.py <run dir> <정책 파일> <정책 시작일>
# 숫자는 '비율과 건수'로만 낸다. 시험 규모가 작아 방향·크기는 읽지 않는다(출력 점검).
import json, re, sys, collections, statistics as st

run, polp, start = sys.argv[1], sys.argv[2], sys.argv[3]
pol = json.load(open(polp, encoding="utf-8"))
sec = pol["sectors"]["농축산물"]
subs = set(sec["eligibility"]["include"]["subs"])
rx = re.compile(sec["eligibility"]["include"]["name_regex"])
chains = [(c, re.compile(p)) for c, p in sec["chains"].items()]
WALLET = re.compile(r"지갑|policy_spend|정책 ?잔액|지원금으로|쿠폰을 ?(쓰|사용)|할인권")
PRESUB = re.compile(r"할인 ?(후|된) ?(가격|금액)|할인가로|할인 적용가|20% ?(뺀|제외한)|할인을 반영한 금액")
POLICY_WORD = re.compile(r"농할|P016|농축산물 할인|20% ?할인")


def load(p):
    return {json.loads(l)["aid"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def chain_of(name):
    return next((c for c, p in chains if p.search(name or "")), None) if rx.match(name or "") else None


out = {}
for arm in ("on", "off"):
    D = load(f"{run}/{arm}/dossier.jsonl")
    c = collections.Counter()
    shares = []; part_amt = []; reasons_wallet = []; reasons_presub = []; confused = []; leak = []
    for aid, r in D.items():
        disc_by_day = collections.defaultdict(float)
        for p in r.get("plans") or []:
            pre = p["day"] < start
            for it in p.get("items") or []:
                amt = float(it.get("actual_spent") or 0)
                txt = " ".join(str(it.get(k) or "") for k in ("pick_reason", "reasoning", "menu"))
                if POLICY_WORD.search(txt) and (arm == "off" or pre):
                    leak.append((arm, p["day"], aid[-14:], it.get("poi"), txt[:120]))
                if amt <= 0:
                    continue
                gro = it.get("poi_sub") in subs
                elig = gro and bool(rx.match(it.get("poi") or ""))
                if gro and not pre:
                    c["장보기"] += 1
                    c["장보기_참여"] += elig
                    if it.get("produce_spent") is None:
                        c["농축산물_답없음"] += 1
                    else:
                        shares.append(min(1.0, float(it["produce_spent"]) / amt))
                    if elig:
                        part_amt.append(amt)
                    if arm == "on":
                        if WALLET.search(txt):
                            reasons_wallet.append((aid[-14:], it.get("poi"), txt[:140]))
                        if PRESUB.search(txt):
                            reasons_presub.append((aid[-14:], it.get("poi"), txt[:140]))
                        if not elig and re.search(r"농할|할인", it.get("pick_reason") or ""):
                            confused.append((aid[-14:], it.get("poi"), (it.get("pick_reason") or "")[:140]))
                d = float(it.get("discount_total") or 0)
                if d > 0:
                    c["할인_결제"] += 1
                    disc_by_day[p["day"]] += d
        # 할인 결제마다 기억이 남았나(그날 기억 중 '결제 할인' 문장)
        mem_disc = collections.Counter(m["day"] for m in (r.get("memories") or []) if "결제 할인" in (m.get("summary") or ""))
        for day, v in disc_by_day.items():
            c["할인_있는_사람날"] += 1
            c["할인_기억_있음"] += mem_disc.get(day, 0) > 0
        # 다음 날 상태의 체인별 누적 = 실제 할인 합
        if arm == "on":
            tot_chain = collections.defaultdict(float)
            for p in sorted(r.get("plans") or [], key=lambda x: x["day"]):
                for it in p.get("items") or []:
                    if float(it.get("discount_total") or 0) > 0:
                        tot_chain[chain_of(it.get("poi"))] += float(it["discount_total"])
            last = sorted(r.get("states") or [], key=lambda s: s["day"])[-1:] if r.get("states") else []
            if last and tot_chain:
                try:
                    used = json.loads(last[0].get("policy_used") or "{}")
                except Exception:
                    used = {}
                ok = all(abs(float(used.get(f"P016:{ch}", 0)) - v) < 1 for ch, v in tot_chain.items())
                c["상태누적_검사"] += 1
                c["상태누적_일치"] += ok
                if not ok:
                    confused.append((aid[-14:], "상태 누적 불일치", json.dumps({"used": used, "sum": dict(tot_chain)}, ensure_ascii=False)))
    out[arm] = {"counts": dict(c),
                "농축산물_몫_중앙": round(st.median(shares), 3) if shares else None,
                "농축산물_몫_하위10_상위10": [round(sorted(shares)[int(.1 * len(shares))], 2), round(sorted(shares)[int(.9 * len(shares)) - 1], 2)] if len(shares) >= 10 else None,
                "참여매장_결제_중앙(원)": st.median(part_amt) if part_amt else None,
                "지갑으로_오해": reasons_wallet, "할인_미리_뺌_흔적": reasons_presub, "비참여_매장을_할인으로_착각": confused,
                "정책_누출": leak}
for arm, v in out.items():
    print(f"===== {arm}")
    print("  건수:", json.dumps(v["counts"], ensure_ascii=False))
    for k in ("농축산물_몫_중앙", "농축산물_몫_하위10_상위10", "참여매장_결제_중앙(원)"):
        print(f"  {k}: {v[k]}")
    for k in ("지갑으로_오해", "할인_미리_뺌_흔적", "비참여_매장을_할인으로_착각", "정책_누출"):
        print(f"  {k}: {len(v[k])}건")
        for x in v[k][:6]:
            print("     ", x)
json.dump(out, open(f"{run}/p016_full_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
