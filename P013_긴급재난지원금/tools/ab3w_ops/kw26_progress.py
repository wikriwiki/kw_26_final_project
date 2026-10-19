# 본런 속도·결과 요약 (읽기 전용). 2026-10-06~
#   python3 /data/ab3w/kw26_progress.py p012_main p013_main
# 속도: 지난 호출 이후 늘어난 처리 인원으로 분당 인원·이 날 남은 시간 추정(상태 파일 /data/ab3w/.kw26_progress_state.json).
# 결과: 날별 1인 소비·외출 수·업종 비중. 정책 기간이면 같은 날 정책 있음 - 없음 차이와 정책 돈.
import collections, datetime as dt, glob, json, os, statistics as st, sys

STATE = "/data/ab3w/.kw26_progress_state.json"
now = dt.datetime.now()
try:
    prev = json.load(open(STATE))
except Exception:
    prev = {}
cur = {"at": now.isoformat(), "counts": {}}


def load(f):
    rows = []
    for l in open(f, encoding="utf-8"):
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
    return rows


def summarize(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    spend, outs, cat = [], [], collections.Counter()
    pol = 0
    for r in ok:
        rec = [e for e in (r.get("execution_receipts") or []) if e.get("kind") == "purchase_receipt"]
        amt = sum(e.get("amount") or 0 for e in rec) + (r.get("cm_online_total") or 0)
        spend.append(amt)
        outs.append(len(rec))
        for e in rec:
            cat[e.get("category") or "?"] += e.get("amount") or 0
        cat["방문외(동별고정)"] += r.get("cm_online_total") or 0
        pol += (r.get("policy_spend_today") or 0) + (r.get("instant_discount_today") or 0)
    tot = sum(cat.values()) or 1
    top = ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in cat.most_common(5))
    return {"n": len(ok), "bad": len(rows) - len(ok), "spend": st.mean(spend) if spend else 0,
            "outs": st.mean(outs) if outs else 0, "top": top, "pol": pol, "cat": cat}


for run in sys.argv[1:]:
    B = f"/data/ab3w/{run}"
    print(f"===== {run}")
    try:
        tail = open(f"{B}/orchestrate.log", encoding="utf-8").read().strip().splitlines()[-1]
        print("  단계:", tail[27:140])
    except Exception:
        pass
    by_day = collections.defaultdict(dict)
    for arm in ("pre", "on", "off"):
        for f in sorted(glob.glob(f"{B}/{arm}/metrics/day_*.jsonl")):
            day = os.path.basename(f)[4:14]
            rows = load(f)
            s = summarize(rows)
            by_day[day][arm] = s
            key = f"{run}|{arm}|{day}"
            cur["counts"][key] = s["n"]
            rate = ""
            if key in prev.get("counts", {}) and prev.get("at"):
                dmin = (now - dt.datetime.fromisoformat(prev["at"])).total_seconds() / 60
                d = s["n"] - prev["counts"][key]
                if dmin > 1 and 0 < s["n"] < 2000:
                    pm = d / dmin
                    left = (2000 - s["n"]) / pm if pm > 0 else float("inf")
                    rate = f" | 지난 {dmin:.0f}분 {d}명(분당 {pm:.1f}명), 이 날 남은 약 {left:.0f}분+밤 단계"
            wk = "월화수목금토일"[dt.date.fromisoformat(day).weekday()]
            done = "끝" if s["n"] >= 2000 else "진행"
            print(f"  [{arm}] {day}({wk}) {done} {s['n']}/2000 실패 {s['bad']} · 1인 {s['spend']:,.0f}원 · 외출 {s['outs']:.2f}건"
                  f" · 정책돈 {s['pol']:,.0f}원 · 업종 {s['top']}{rate}")
    for day, arms in sorted(by_day.items()):
        if "on" in arms and "off" in arms and arms["on"]["n"] and arms["off"]["n"]:
            a, b = arms["on"], arms["off"]
            diff = 100 * (a["spend"] - b["spend"]) / b["spend"] if b["spend"] else 0
            note = "" if a["n"] == b["n"] == 2000 else f" (처리 인원 {a['n']}/{b['n']} — 끝나기 전 값은 사람 구성이 달라 참고만)"
            print(f"  ◆ {day} 정책 있음 - 없음: 1인 {a['spend']-b['spend']:+,.0f}원 ({diff:+.1f}%), 외출 {a['outs']-b['outs']:+.2f}건, 정책돈 {a['pol']:,.0f}원{note}")

json.dump(cur, open(STATE, "w"))
