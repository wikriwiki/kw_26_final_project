"""[폐기 — 쓰지 않음] 실제 이용객 발언 ↔ 시뮬 답변을 AI 심판으로 비교하려던 시도. 심판이 반대 입장 답에도 높은 점수를 줘(4.67 vs 4.89)
구별을 못 하므로 결과를 쓰지 않는다. 채택한 방법은 quote_theme_lift.py. 기록으로만 남긴다.

실제 이용객 발언(유림 님 조사, 기사 원문) ↔ 시뮬 시민 인터뷰 답변 비교.

1) 실제 발언과 시뮬 답변을 같은 분류기로 입장 · 주제 분류(분류기는 출처를 모른다)
2) 실제 발언마다, 같은 입장의 시뮬 답변 중 주제가 겹치는 후보 15개를 고르고 K-EXAONE 심판이 가장 닮은 답을 골라
   의미 유사도 1~5 점을 매긴다. 대조: 반대 입장 시뮬 답변 후보에서 같은 절차.
"""
import json, io, re, time, random, concurrent.futures as cf
from pathlib import Path
from urllib import request

HERE = Path(__file__).parent; D = HERE / "bq2_run"
env = dict(l.strip().split("=", 1) for l in open(HERE / "friendli.env", encoding="utf-8") if "=" in l)
QUOTES = [  # (출처, 지역, 발언) — 노션 「신유림_금연정책 설문 인터뷰 조사」
    ("조선일보 2018 현장 취재", "서초구", "함께 당구장에 온 뒤로부터 가족 간에 대화가 생겼다"),
    ("동아일보 시행 첫날", "종로구", "놀면서 담배를 피우는 최후의 보루였는데 아쉽다"),
    ("동아일보 시행 첫날", "양천구", "실내에서 문 닫고 피우는 것까지 안 된다니 너무하다"),
    ("아시아투데이 현장 취재", "중구", "점점 흡연자들의 설 자리가 없어져서 착잡한 느낌"),
    ("아시아투데이 현장 취재", "중구", "이젠 조절할 수 있어서 좋은 것 같다"),
    ("아시아투데이 현장 취재", "중구", "간접흡연으로 폐암에 걸리는 것 아닌가"),
    ("세계일보 시행 첫 주", "서울·인천", "이제는 어느 당구장이든 갈 수 있어서 좋다"),
    ("세계일보 시행 첫 주", "서울·인천", "흡연자도 담배 냄새는 싫어한다"),
    ("세계일보 시행 첫 주", "서울·인천", "당구장은 흡연자들의 마지막 보루나 마찬가지였는데 아쉽다"),
]
THEMES = ["health", "clean", "family", "smoker_burden", "rights", "business", "fairness", "habit"]
TH_KO = {"health": "간접흡연·건강", "clean": "쾌적함·냄새", "family": "가족·아이·동행", "smoker_burden": "흡연자 불편·흡연할 곳 감소",
         "rights": "흡연자 권리·과한 규제", "business": "영업·매출·이용 감소 우려", "fairness": "형평·사회 흐름", "habit": "금연·흡연량 조절에 도움"}


def chat(system, user, max_tokens=200):
    body = json.dumps({"model": env["INTERVIEW_MODEL"], "temperature": 0.0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    for k in range(4):
        try:
            req = request.Request(env["INTERVIEW_BASE_URL"].rstrip("/") + "/chat/completions", data=body,
                                  headers={"Content-Type": "application/json", "Authorization": "Bearer " + env["INTERVIEW_API_KEY"]})
            with request.urlopen(req, timeout=120) as r:
                t = json.loads(r.read())["choices"][0]["message"]["content"]
            return json.loads(re.search(r"\{.*\}", t, re.S).group(0))
        except Exception:
            time.sleep(4 * (k + 1))
    return None


CODE_SYS = ("다음은 '당구장·실내 골프연습장 금연구역 지정'에 대한 한 사람의 발언이다. 발언에 드러난 입장과 이유 주제를 고른다.\n"
            + "\n".join(f"- {k}: {v}" for k, v in TH_KO.items())
            + '\nJSON만 반환: {"stance":"찬성|반대|중립", ' + ", ".join(f'"{k}":0' for k in THEMES) + "}  (주제는 나타나면 1)")
JUDGE_SYS = ("실제 사람의 발언 하나와 후보 답변 목록이 있다. 실제 발언과 의미(입장과 이유)가 가장 비슷한 후보를 하나 고르고, "
             "의미 유사도를 1~5로 매긴다(5 = 같은 입장·같은 이유, 3 = 입장은 같고 이유가 일부 겹침, 1 = 무관하거나 반대). "
             'JSON만 반환: {"best": 후보 번호, "score": 1~5}')


def code(text):
    j = chat(CODE_SYS, text) or {}
    return dict(stance=j.get("stance"), themes=[k for k in THEMES if str(j.get(k, 0)) in ("1", "True", "true")])


# 시뮬 답변(만 19세 이상, 시행 전·후)
sim = []
for ph in ("pre", "post"):
    ans = {json.loads(l)["aid"]: json.loads(l) for l in io.open(D / f"{ph}.jsonl", encoding="utf-8") if l.strip()}
    th = {json.loads(l)["aid"]: json.loads(l)["themes"] for l in io.open(D / f"themes_{ph}.jsonl", encoding="utf-8")}
    for aid, r in ans.items():
        if r.get("answer") and "_10대_" not in aid and th.get(aid) is not None:
            sim.append(dict(aid=aid, phase=ph, smoking=r["smoking_status"], stance=r["answer"]["choice"], reason=r["answer"]["reason"], themes=th[aid]))
rnd = random.Random(20261021)


def judge(quote, pool):
    qth = set(quote["themes"])
    ranked = sorted(pool, key=lambda s: (-len(qth & set(s["themes"])) / max(1, len(qth | set(s["themes"]))), rnd.random()))[:15]
    lst = "\n".join(f"{i + 1}. {s['reason']}" for i, s in enumerate(ranked))
    j = chat(JUDGE_SYS, f"[실제 발언] {quote['text']}\n\n[후보]\n{lst}", 60) or {}
    try:
        b = int(j.get("best")) - 1; sc = int(j.get("score"))
        return dict(score=sc, best=ranked[b])
    except Exception:
        return dict(score=None, best=None)


quotes = []
for src, gu, text in QUOTES:
    c = code(text); quotes.append(dict(src=src, gu=gu, text=text, **c))


def one(q):
    st = q["stance"] if q["stance"] in ("찬성", "반대") else None
    if not st:
        return dict(q, matched=None, control=None)
    same = [s for s in sim if s["stance"] == st]; opp = [s for s in sim if s["stance"] != st and s["stance"] in ("찬성", "반대")]
    return dict(q, matched=judge(q, same), control=judge(q, opp))


with cf.ThreadPoolExecutor(4) as ex:
    res = list(ex.map(one, quotes))
(D / "quotes_vs_sim.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
ms = [r["matched"]["score"] for r in res if r.get("matched") and r["matched"]["score"]]
cs = [r["control"]["score"] for r in res if r.get("control") and r["control"]["score"]]
# 주제 재현: 실제 발언의 주제가 같은 입장 시뮬 답변에 한 번이라도 나오는가
recall = []
for r in res:
    st = r["stance"]
    if st not in ("찬성", "반대"):
        continue
    pool = [s for s in sim if s["stance"] == st]
    for t in r["themes"]:
        recall.append(any(t in s["themes"] for s in pool))
for r in res:
    m = r.get("matched") or {}
    print(f"[{r['stance']}] {r['text']}  주제 {r['themes']}")
    if m.get("best"):
        print(f"   ↔ 시뮬({m['best']['smoking']}, {m['best']['phase']}, {m['best']['stance']}) {m['score']}점: {m['best']['reason'][:110]}")
    if r.get("control") and r["control"].get("score"):
        print(f"   대조(반대 입장 시뮬) {r['control']['score']}점")
print("평균 유사도 같은 입장", round(sum(ms) / len(ms), 2), f"(n={len(ms)})", " 반대 입장 대조", round(sum(cs) / len(cs), 2) if cs else None,
      " 4점 이상 비율", round(100 * sum(x >= 4 for x in ms) / len(ms)), "%", " 주제 재현", f"{sum(recall)}/{len(recall)}")

# ---- 강제 선택: 같은 입장 최선 답 vs 반대 입장 최선 답, 순서를 바꿔 두 번 ----
FC_SYS = ("실제 사람의 발언 하나와 답변 두 개(A, B)가 있다. 실제 발언과 입장(찬성·반대)과 이유가 더 비슷한 답변을 고른다. "
          'JSON만 반환: {"closer":"A|B"}')
fc = []
for r in res:
    m, c = r.get("matched") or {}, r.get("control") or {}
    if not (m.get("best") and c.get("best")):
        continue
    wins = 0.0
    for order in (0, 1):
        a, b = (m["best"], c["best"]) if order == 0 else (c["best"], m["best"])
        j = chat(FC_SYS, f"[실제 발언] {r['text']}\n\nA. {a['reason']}\n\nB. {b['reason']}", 20) or {}
        pick = str(j.get("closer", "")).strip().upper()[:1]
        wins += 0.5 if (pick == "A") == (order == 0) else 0.0
    fc.append(dict(text=r["text"], wins=wins))
    print(f"강제 선택 {wins:.1f}/1.0  {r['text']}")
acc = sum(x["wins"] for x in fc) / len(fc)
print(f"강제 선택 정확도 {acc * 100:.0f}% (n={len(fc)}, 우연 50%)")
(D / "quotes_forced_choice.json").write_text(json.dumps(dict(items=fc, accuracy=acc), ensure_ascii=False, indent=1), encoding="utf-8")
