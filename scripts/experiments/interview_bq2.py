"""금연구역 이용객 설문(보고서 표 55, 질문 BQ2)과 같은 형식의 시민 인터뷰 — K-EXAONE(Friendli) 사용.

시행 전(2017-12-02 그래프)과 시행 후(정책 있음 세계 2017-12-16 그래프)에, 그 시점까지 당구장·골프연습장을
이용한 시민에게 '찬성 · 반대 · 모름' 중 하나를 고르게 한다. 실제 설문은 이용객 600명(당구장 400 · 골프 200).
답변은 시뮬레이션에 다시 넣지 않는다.

  python interview_bq2.py --phase pre|post --bolt bolt://localhost:17701 --pw-file pw15.txt --cohort cohort.json --out DIR
환경: friendli.env (INTERVIEW_BASE_URL · INTERVIEW_MODEL · INTERVIEW_API_KEY) — 키는 출력하지 않는다.
"""
import argparse, concurrent.futures as cf, json, re, time, hashlib
from pathlib import Path
from urllib import request, error
from neo4j import GraphDatabase

PHASE = {"pre": dict(as_of="2017-12-02", status="이 규칙은 내일(2017년 12월 3일)부터 시행될 예정입니다."),
         "post": dict(as_of="2017-12-16", status="이 규칙은 2017년 12월 3일부터 시행되어 2주째 적용되고 있습니다.")}
DAYS_FROM = "2017-11-19"
FAC_RE = re.compile(r"당구|골프")
QUESTION = ("당구장과 실내 골프연습장(스크린골프장 포함)의 실내를 금연구역으로 지정하는 정책에 대해 "
            "어떻게 생각하십니까? 찬성, 반대, 모름 가운데 하나를 고르고, 그렇게 생각하는 이유를 1~2문장으로 말해 주세요.")
SYSTEM = """당신은 아래 프로필과 기록을 가진 서울 시민입니다. 설문 조사원의 질문에 이 시민 본인으로서 솔직하게 답합니다.
기록에 없는 경험이나 가족·건강 정보를 지어내지 않습니다.
반드시 JSON만 반환합니다: {"choice":"찬성|반대|모름","reason":"이유 1~2문장"}"""
QSHA = hashlib.sha256((SYSTEM + QUESTION).encode("utf-8")).hexdigest()
PROFILE = [("personal_age", "나이"), ("personal_gender", "성별"), ("personal_job_raw", "직업"), ("personal_income_level", "소득 수준"),
           ("p_life_stage", "생애 단계"), ("residence_gu", "거주 구"), ("residence_dong_name", "거주 동"),
           ("personality_lifestyle_raw", "생활 방식"), ("nvidia_hobbies", "취미")]


def fetch(s, aid, as_of):
    a = s.run("MATCH (a:Agent {id:$aid}) RETURN a{.*} AS a", aid=aid).single()
    if not a:
        return None
    raw = a["a"]
    prof = {lab: raw.get(k) for k, lab in PROFILE if raw.get(k) not in (None, "")}
    ev = [dict(r) for r in s.run("""
        MATCH (a:Agent {id:$aid})-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(poi:POI)
        WHERE toString(p.day) >= $d0 AND toString(p.day) <= $d1 AND coalesce(i.actual_spent,0) > 0
        OPTIONAL MATCH (poi)-[:IN_CATEGORY]->(c:Category)
        RETURN toString(p.day) AS day, coalesce(c.name, i.sub_category) AS sub, poi.name AS poi, i.actual_spent AS spent,
               i.actual_satisfaction AS sat ORDER BY day""", aid=aid, d0=DAYS_FROM, d1=as_of)]
    return prof, ev


def kind(name, sub):
    t = f"{name or ''}{sub or ''}"
    return "billiard" if "당구" in t else "golf" if "골프" in t else None


def block(prof, ev, smoking, phase):
    fac = [e for e in ev if kind(e["poi"], e["sub"])]
    other = {}
    for e in ev:
        if not kind(e["poi"], e["sub"]):
            other[e["sub"]] = other.get(e["sub"], 0) + 1
    L = ["[프로필]"] + [f"- {k}: {v}" for k, v in prof.items()]
    L.append(f"- 흡연 여부: {'흡연자' if smoking == 'smoker' else '비흡연자' if smoking == 'non_smoker' else '알 수 없음'}")
    L.append(f"\n[최근 당구장·골프연습장 이용 ({DAYS_FROM} ~ {PHASE[phase]['as_of']})]")
    L += [f"- {e['day']} {e['poi']} {int(e['spent']):,}원" for e in fac] or ["- 없음"]
    L.append("\n[그 밖의 외출 결제] " + ", ".join(f"{k} {v}회" for k, v in sorted(other.items(), key=lambda x: -x[1])[:8]))
    L.append(f"\n[오늘] {PHASE[phase]['as_of']}. {PHASE[phase]['status']}")
    L.append(f"\n[질문] {QUESTION}")
    return "\n".join(L), fac


def chat(env, ub, retries=4):
    body = json.dumps({"model": env["INTERVIEW_MODEL"], "temperature": 0.7, "max_tokens": 300,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ub}]}).encode()
    for k in range(retries):
        try:
            req = request.Request(env["INTERVIEW_BASE_URL"].rstrip("/") + "/chat/completions", data=body,
                                  headers={"Content-Type": "application/json", "Authorization": "Bearer " + env["INTERVIEW_API_KEY"]})
            with request.urlopen(req, timeout=120) as r:
                j = json.loads(r.read())
            return j["choices"][0]["message"]["content"], j.get("usage", {})
        except error.HTTPError as e:
            if e.code in (401, 403):
                raise
            time.sleep(5 * (k + 1))
        except Exception:
            time.sleep(5 * (k + 1))
    raise RuntimeError("llm failed")


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        j = json.loads(m.group(0)) if m else None
    except Exception:
        j = None
    if not j:
        return None
    c = str(j.get("choice", "")).strip()
    c = "찬성" if "찬성" in c else "반대" if "반대" in c else "모름" if "모름" in c else None
    return {"choice": c, "reason": j.get("reason")} if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pre", "post"], required=True); ap.add_argument("--bolt", required=True)
    ap.add_argument("--pw-file", required=True); ap.add_argument("--cohort", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    env = dict(l.strip().split("=", 1) for l in open(Path(__file__).with_name("friendli.env"), encoding="utf-8") if "=" in l)
    pw = Path(a.pw_file).read_text().strip()
    cohort = json.loads(Path(a.cohort).read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); path = out / f"{a.phase}.jsonl"
    done = {json.loads(l)["aid"] for l in path.open(encoding="utf-8")} if path.exists() else set()
    drv = GraphDatabase.driver(a.bolt, auth=("neo4j", pw))
    as_of = PHASE[a.phase]["as_of"]
    # 이용객만: 그 시점까지 당구장·골프연습장에 결제한 시민
    users = []
    with drv.session(default_access_mode="READ") as s:
        for p in cohort:
            got = fetch(s, p["id"], as_of)
            if not got:
                continue
            prof, ev = got
            ub, fac = block(prof, ev, p["smoking_status"], a.phase)
            if fac:
                kinds = [kind(e["poi"], e["sub"]) for e in fac]
                users.append(dict(aid=p["id"], smoking_status=p["smoking_status"], ub=ub, n_fac=len(fac),
                                  billiard=kinds.count("billiard"), golf=kinds.count("golf")))
    drv.close()
    if a.limit:
        users = users[:a.limit]
    todo = [u for u in users if u["aid"] not in done]
    print(f"phase {a.phase} · users {len(users)} · todo {len(todo)} · qsha {QSHA[:12]}", flush=True)

    def one(u):
        try:
            text, usage = chat(env, u["ub"])
        except Exception as exc:
            return dict(aid=u["aid"], error=type(exc).__name__)
        return dict(aid=u["aid"], smoking_status=u["smoking_status"], billiard=u["billiard"], golf=u["golf"],
                    n_fac=u["n_fac"], answer=parse(text), raw=text, usage=usage, user_block=u["ub"], phase=a.phase, qsha=QSHA)
    t0 = time.time(); n = 0
    with path.open("a", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(one, todo):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); n += 1
            if n % 50 == 0:
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("done", n, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
