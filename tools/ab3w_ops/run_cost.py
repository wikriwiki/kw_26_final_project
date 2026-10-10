#!/usr/bin/env python3
"""정책별 시뮬레이션 시간·계산량 기록 (2026-10-09, 읽기 전용).

한 런 폴더(/data/ab3w/<run>)에서:
- 벽시계 시간: orchestrate.log 의 단계 표지(=== n ...)와 각 날 '시도 1' 시각, 마지막 '=== 끝'.
- 날별 처리 시간: <arm>/summary.json 의 elapsed_sec(사람 처리 + 밤 단계).
- 계산량: <arm>/metrics/day_*.jsonl 의 사람-날별 tokens_in·tokens_out·LLM 호출 수·LLM 시간.
값은 기록에 있는 것만 쓴다. 가격(원)은 넣지 않는다 — 실제 결제액은 사용자가 준다.
"""
import glob, json, os, re, sys, datetime as dt, collections

def ts(line):
    m = re.match(r"\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d)\]", line)
    return dt.datetime.fromisoformat(m.group(1)) if m else None

def main(run):
    B = f"/data/ab3w/{run}"
    log = open(f"{B}/orchestrate.log", encoding="utf-8").read().splitlines()
    stamps = [(ts(l), l) for l in log if ts(l)]
    start, last = stamps[0][0], stamps[-1][0]
    end = next((t for t, l in reversed(stamps) if "=== 끝" in l), None)
    stages = [(t, l.split("] ", 1)[1][:60]) for t, l in stamps if "] === " in l]
    out = {"run": run, "시작": start.isoformat(), "끝": end.isoformat() if end else None,
           "벽시계_시간": round(((end or last) - start).total_seconds() / 3600, 2), "끝났나": bool(end),
           "단계": [{"시각": t.isoformat(), "표지": s} for t, s in stages]}
    arms = {}
    for arm in ("pre", "on", "off"):
        tok_in = tok_out = calls = 0; t_llm = 0.0; n = 0; elapsed = 0.0
        for f in sorted(glob.glob(f"{B}/{arm}/metrics/day_*.jsonl")):
            for l in open(f, encoding="utf-8"):
                m = json.loads(l); n += 1
                tok_in += m.get("tokens_in") or 0; tok_out += m.get("tokens_out") or 0
                for k in ("s1_timing", "s2_timing"):
                    calls += (m.get(k) or {}).get("n_llm_calls") or 0
                    t_llm += (m.get(k) or {}).get("t_llm") or 0
                elapsed += m.get("elapsed") or 0
        days = []
        sp = f"{B}/{arm}/summary.json"
        if os.path.exists(sp):
            for d in json.load(open(sp, encoding="utf-8")).get("summary", []):
                days.append({"날": d["day"], "처리_시간": round(d.get("elapsed_sec", 0) / 3600, 2),
                             "밤_단계_초": round(d.get("night2_elapsed_sec", 0))})
        arms[arm] = {"사람_날": n, "입력_토큰": tok_in, "출력_토큰": tok_out, "LLM_호출": calls,
                     "LLM_대기_합_시간": round(t_llm / 3600, 1), "날별": days}
    out["갈래"] = arms
    out["합계"] = {k: sum(a[k] for a in arms.values()) for k in ("사람_날", "입력_토큰", "출력_토큰", "LLM_호출")}
    out["한계"] = ["밤 단계(대화 생성) 토큰은 metrics 에 없어 합계에서 빠진다 — 날별 밤_단계_초만 있다.",
                 "세 런이 GPU 를 함께 썼으므로 GPU 시간은 런별 토큰 몫으로 나눠야 한다(별도 계산).",
                 "멈춘 런·실패한 첫 시작은 이 폴더 밖(…_stopped_*, …_failed_*)에 있어 따로 센다."]
    return out

if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps(main(r), ensure_ascii=False, indent=1))
