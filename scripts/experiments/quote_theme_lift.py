"""실제 이용객 발언(유림 님 조사, 기사 원문) ↔ 시뮬 시민 답변: 주제-입장 일치.

실제 발언의 입장 · 주제는 발언 내용과 유림 님 메모로 정했다(아래 표). 시뮬은 BQ2 인터뷰 답변(만 19세 이상, 시행 전 + 후)의
눈가림 주제 분류(theme_bq2.py)를 쓴다. 그 주제를 말한 비율이 같은 입장 답변에서 반대 입장 답변보다 높으면 '일치'.
같은 입장 쪽 비율이 1% 미만이면 시뮬이 그 이유를 사실상 말하지 않는 것이므로 보고에서는 불일치로 센다(reported_hit).
  python quote_theme_lift.py bq2_run
"""
import json, io, sys
from pathlib import Path

D = Path(sys.argv[1])
TH = {"health": "간접흡연·건강", "clean": "쾌적함·냄새", "family": "가족·아이", "smoker_burden": "흡연자 불편",
      "rights": "흡연 권리·과한 규제", "business": "영업 우려", "fairness": "형평", "habit": "금연·흡연량 조절"}
QUOTES = [  # (발언, 입장, 주제들, 메모)
    ("함께 당구장에 온 뒤로부터 가족 간에 대화가 생겼다", None, [], "찬반 발언이 아님(유림 님 메모) — 입장 판단 불가"),
    ("놀면서 담배를 피우는 최후의 보루였는데 아쉽다", "반대", ["smoker_burden"], "흡연 이용객"),
    ("실내에서 문 닫고 피우는 것까지 안 된다니 너무하다", "반대", ["smoker_burden", "rights"], "흡연 이용객"),
    ("점점 흡연자들의 설 자리가 없어져서 착잡한 느낌", "반대", ["smoker_burden"], "흡연 이용객"),
    ("당구장은 흡연자들의 마지막 보루나 마찬가지였는데 아쉽다", "반대", ["smoker_burden"], "흡연 이용객"),
    ("이젠 조절할 수 있어서 좋은 것 같다", "찬성", ["habit"], "흡연 이용객"),
    ("흡연자도 담배 냄새는 싫어한다", "찬성", ["clean"], "흡연 이용객"),
    ("간접흡연으로 폐암에 걸리는 것 아닌가", "찬성", ["health"], "비흡연 이용객"),
    ("이제는 어느 당구장이든 갈 수 있어서 좋다", "찬성", ["clean"], "비흡연 이용객"),
    ("부부 동반 회원이 급격히 늘었다", None, [], "송파당구연맹 관계자 발언, 이용객 아님 — 입장 판단 불가"),
]

sim = []
for ph in ("pre", "post"):
    ans = {json.loads(l)["aid"]: json.loads(l) for l in io.open(D / f"{ph}.jsonl", encoding="utf-8") if l.strip()}
    th = {json.loads(l)["aid"]: json.loads(l)["themes"] for l in io.open(D / f"themes_{ph}.jsonl", encoding="utf-8")}
    for aid, r in ans.items():
        if r.get("answer") and "_10대_" not in aid and th.get(aid) is not None and r["answer"]["choice"] in ("찬성", "반대"):
            sim.append(dict(stance=r["answer"]["choice"], smoking=r["smoking_status"], themes=th[aid]))
share = lambda pool, k: 100 * sum(k in s["themes"] for s in pool) / len(pool)
by = {st: [s for s in sim if s["stance"] == st] for st in ("찬성", "반대")}

rows = []
for text, st, ths, note in QUOTES:
    if not st:
        rows.append(dict(text=text, stance=None, note=note)); continue
    other = "반대" if st == "찬성" else "찬성"
    for k in ths:
        a, b = share(by[st], k), share(by[other], k)
        rows.append(dict(text=text, stance=st, theme=TH[k], same=round(a, 1), other=round(b, 1), hit=a > b, reported_hit=a > b and a >= 1.0, note=note))
judged = [r for r in rows if r["stance"]]
smk = [s for s in by["찬성"] if s["smoking"] == "smoker"]
res = dict(n_sim={k: len(v) for k, v in by.items()}, hit=sum(r["hit"] for r in judged), reported_hit=sum(r["reported_hit"] for r in judged), total=len(judged),
           n_smoker_support=len(smk), smoker_support_themes={TH[k]: round(share(smk, k), 1) for k in TH}, rows=rows)
(D / "quote_theme_lift.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
for r in rows:
    print(f"[{r['stance'] or '판단 불가'}] {r['text']}" + (f" | {r['theme']}: 같은 입장 {r['same']}% vs 반대 입장 {r['other']}% → {'일치' if r['reported_hit'] else '불일치'}" if r["stance"] else f" — {r['note']}"))
print(f"주제-입장 일치 {res['reported_hit']}/{res['total']} (단순 비교 {res['hit']}/{res['total']})  시뮬 답변 {res['n_sim']}")
print("시뮬 흡연자 찬성", len(smk), res["smoker_support_themes"])
