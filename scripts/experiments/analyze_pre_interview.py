"""사전 인터뷰 응답 집계: 입장 분포, 흡연 여부 · 시설 이용 여부별 차이, 기록 인용 정확도, 지어낸 경험 점검."""
import json, re, sys, collections, math, io
from pathlib import Path

src = Path(sys.argv[1]); out = Path(sys.argv[2])
rows = [json.loads(l) for l in io.open(src, encoding="utf-8") if l.strip()]
ok = [r for r in rows if r.get("response") and isinstance(r["response"], dict) and r["response"].get("stance")]
ST = ["support", "oppose", "mixed", "neutral", "uncertain", "unknown"]
KO = {"support": "찬성", "oppose": "반대", "mixed": "양면적", "neutral": "중립", "uncertain": "판단 유보", "unknown": "잘 모름"}


def dist(sel):
    c = collections.Counter(r["response"]["stance"] for r in sel)
    n = len(sel)
    return {"n": n, **{KO[s]: round(100 * c.get(s, 0) / n, 1) if n else None for s in ST}}


def grp(r):
    user = "이용자" if (r.get("facility_visits_14d") or 0) > 0 else "비이용자"
    sm = {"smoker": "흡연자", "non_smoker": "비흡연자"}.get(r.get("smoking_status"), "흡연 여부 모름")
    return sm, user


norm = lambda s: re.sub(r"\s+", " ", str(s or "")).strip()
cite_total = cite_ok = 0
for r in ok:
    ub = norm(r["user_block"])
    for c in r["response"].get("cited_records") or []:
        c2 = norm(c).strip("\"'“”‘’…. ")
        if len(c2) < 6:
            continue
        cite_total += 1
        cite_ok += c2 in ub or c2[:40] in ub
# 기록과 모순: 이용 기록이 있는데 '이용 경험이 없다', 기록이 없는데 '자주 이용/다녔다'
NOEXP = re.compile(r"(이용|방문)\s*(한\s*)?(경험|적)이\s*없|이용하지\s*않|가\s*본\s*적이\s*없")
CLAIM = re.compile(r"(자주|종종|매주|주말마다)\s*(이용|방문|다니|가)|이용해\s*왔|다녀\s*왔|즐겨\s*(이용|찾)")
users = [r for r in ok if (r.get("facility_visits_14d") or 0) > 0]
nonusers = [r for r in ok if (r.get("facility_visits_14d") or 0) == 0]
contra_user = [r["aid"] for r in users if NOEXP.search(r["response"].get("answer", ""))]
claim_nonuser = [r["aid"] for r in nonusers if CLAIM.search(r["response"].get("answer", ""))]
mentions_situation = sum(1 for r in ok if re.search(r"흡연|비흡연|이용|방문|골프|당구|직업|가족|건강|운동|여가", r["response"].get("answer", "")))

by = collections.defaultdict(list)
for r in ok:
    sm, us = grp(r)
    by[(sm, us)].append(r); by[(sm, "전체")].append(r); by[("전체", us)].append(r)
visits = collections.defaultdict(collections.Counter)
for r in users:
    visits[grp(r)[0]][r["response"].get("would_change_visits")] += 1

res = dict(n_rows=len(rows), n_parsed=len(ok), parse_rate=round(100 * len(ok) / max(1, len(rows)), 1),
           errors=collections.Counter(r.get("error") for r in rows if r.get("error")),
           overall=dist(ok), by_group={f"{a}|{b}": dist(v) for (a, b), v in sorted(by.items())},
           visits_change_among_users={k: dict(v) for k, v in visits.items()},
           citation=dict(n=cite_total, exact_rate=round(100 * cite_ok / max(1, cite_total), 1)),
           contradiction_users_claim_no_experience=dict(n=len(contra_user), of=len(users), rate=round(100 * len(contra_user) / max(1, len(users)), 1), ids=contra_user[:20]),
           fabricated_frequent_use_nonusers=dict(n=len(claim_nonuser), of=len(nonusers), rate=round(100 * len(claim_nonuser) / max(1, len(nonusers)), 1), ids=claim_nonuser[:20]),
           answer_mentions_own_situation_rate=round(100 * mentions_situation / max(1, len(ok)), 1),
           mean_confidence=round(sum(float(r["response"].get("confidence") or 0) for r in ok) / max(1, len(ok)), 2),
           tokens=dict(prompt=sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows),
                       completion=sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)))
# 대표 응답: 집단마다 가장 흔한 입장의 응답 하나
examples = []
for key in [("흡연자", "이용자"), ("비흡연자", "이용자"), ("흡연자", "비이용자"), ("비흡연자", "비이용자")]:
    sel = by.get(key, [])
    if not sel:
        continue
    for want in ("oppose", "support", "mixed"):
        cand = [r for r in sel if r["response"]["stance"] == want]
        if cand:
            r = max(cand, key=lambda r: len(r["response"].get("answer", "")))
            examples.append(dict(group="|".join(key), aid=r["aid"], stance=KO[want], visits=r.get("facility_visits_14d"),
                                 answer=r["response"]["answer"], reasons=r["response"].get("key_reasons"),
                                 would_change_visits=r["response"].get("would_change_visits")))
            break
res["examples"] = examples
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "examples"}, ensure_ascii=False, indent=1))
