"""실제 이용객 발언마다 가장 비슷한 시뮬 시민 답변 하나를 고른다(사람이 고르지 않는다).

후보 = 같은 입장 · 그 발언의 주제를 말한 답변(theme_bq2.py 분류) · 발언자와 같은 흡연 여부, 만 19세 이상, 시행 전 + 후.
후보 중 실제 발언과 글자 2-gram 자카드 유사도가 가장 높은 답을 고른다. 동점이면 aid 순. 앞 발언에 이미 고른 답은 다시 고르지 않는다.
  python pick_ai_quotes.py bq2_run
"""
import json, io, re, sys
from pathlib import Path

D = Path(sys.argv[1])
QUOTES = [  # (발언, 입장, 주제들, 발언자 흡연 여부, 발언자 설명) — quote_theme_lift.py 와 같은 판정.
    # 유림 님 조사에는 이용객 · 구만 있고 흡연 여부는 없다. 흡연 여부는 발언 내용으로 추정했고, 드러나지 않으면 None(흡연 여부로 거르지 않음).
    ("놀면서 담배를 피우는 최후의 보루였는데 아쉽다", "반대", ["smoker_burden"], "smoker", "이용객 · 종로구 · 흡연자로 추정"),
    ("실내에서 문 닫고 피우는 것까지 안 된다니 너무하다", "반대", ["smoker_burden", "rights"], "smoker", "이용객 · 양천구 · 흡연자로 추정"),
    ("점점 흡연자들의 설 자리가 없어져서 착잡한 느낌", "반대", ["smoker_burden"], "smoker", "이용객 · 중구 · 흡연자로 추정"),
    ("당구장은 흡연자들의 마지막 보루나 마찬가지였는데 아쉽다", "반대", ["smoker_burden"], "smoker", "이용객 · 서울·인천 · 흡연자로 추정"),
    ("간접흡연으로 폐암에 걸리는 것 아닌가", "찬성", ["health"], "non_smoker", "이용객 · 중구 · 비흡연자로 추정"),
    ("흡연자도 담배 냄새는 싫어한다", "찬성", ["clean"], None, "이용객 · 서울·인천 · 흡연 여부 불명"),
    ("이제는 어느 당구장이든 갈 수 있어서 좋다", "찬성", ["clean"], "non_smoker", "이용객 · 서울·인천 · 비흡연자로 추정"),
    ("이젠 조절할 수 있어서 좋은 것 같다", "찬성", ["habit"], "smoker", "이용객 · 중구 · 흡연자로 추정"),
]
SMK = {"smoker": "흡연", "non_smoker": "비흡연"}


def field(block, name):
    m = re.search(rf"- {name}: ([^\n]*)", block)
    return m.group(1).strip() if m else ""


def persona(r):
    b = r.get("user_block", "")
    sex = {"F": "여성", "M": "남성"}.get(field(b, "성별"), field(b, "성별"))
    fac = " · ".join(n for n, k in (("당구장", "billiard"), ("골프연습장", "golf")) if r.get(k))
    return f"{field(b, '나이')}세 {sex} · {field(b, '직업')} · {field(b, '거주 구')} {field(b, '거주 동')} · {SMK.get(r['smoking_status'], '흡연 여부 모름')} · {fac} 이용"


grams = lambda s: {s[i:i + 2] for i in range(len(s) - 1)} - {"  "}
jac = lambda a, b: len(grams(a) & grams(b)) / max(1, len(grams(a) | grams(b)))
sim = []
for ph in ("pre", "post"):
    th = {json.loads(l)["aid"]: json.loads(l)["themes"] for l in io.open(D / f"themes_{ph}.jsonl", encoding="utf-8")}
    for l in io.open(D / f"{ph}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("answer") and "_10대_" not in r["aid"] and th.get(r["aid"]):
            sim.append(dict(r, themes=th[r["aid"]], phase=ph))

out = []; used = set()
for text, st, ths, smk, who in QUOTES:
    pool = [s for s in sim if s["answer"]["choice"] == st and set(ths) & set(s["themes"])]
    same = [s for s in pool if s["smoking_status"] == smk] if smk else []
    cand = [s for s in (same or pool) if s["aid"] + s["phase"] not in used] or (same or pool)
    best = max(sorted(cand, key=lambda s: s["aid"]), key=lambda s: jac(text, s["answer"]["reason"]), default=None)
    if best:
        used.add(best["aid"] + best["phase"])
    out.append(dict(text=text, speaker=who, stance=st, n_candidates=len(cand), smoking_matched=bool(same) if smk else None,
                    ai=best and dict(aid=best["aid"], phase=best["phase"], reason=best["answer"]["reason"], persona=persona(best),
                                     similarity=round(jac(text, best["answer"]["reason"]), 3))))
(D / "ai_quote_pairs.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for o in out:
    a = o["ai"]
    print(f"[{o['stance']}] {o['text']} ({o['speaker']})  후보 {o['n_candidates']}{'' if o['smoking_matched'] else ' (흡연 여부로 거르지 않음)'}")
    print(f"   → {a['reason']}  [{a['persona']} · {a['phase']}]" if a else "   → 없음")
