"""인터뷰 이유 문장을 주제로 분류한다(복수 선택). 분류기는 군집(흡연 여부·구·성별)을 보지 않는다(눈가림).
  python theme_bq2.py bq2_run pre|post
"""
import json, sys, io, re, time, concurrent.futures as cf
from pathlib import Path
from urllib import request

THEMES = {"health": "간접흡연·건강 보호", "clean": "쾌적함·냄새·공기", "family": "가족·아이·동행",
          "smoker_burden": "흡연자 불편·흡연할 곳 감소", "rights": "흡연자 권리·과한 규제", "business": "영업·매출·이용 감소 우려",
          "fairness": "다른 실내시설과의 형평·사회 흐름", "habit": "금연·흡연량 조절에 도움"}
SYS = ("다음은 '당구장·실내 골프연습장 금연구역 지정' 설문에 대한 한 시민의 이유 문장이다. 문장에 실제로 나타난 이유를 아래 주제에서 모두 고른다.\n"
       + "\n".join(f"- {k}: {v}" for k, v in THEMES.items())
       + '\n각 주제가 문장에 나타나면 1, 아니면 0으로 표시한다. JSON만 반환: {' + ', '.join(f'"{k}": 0' for k in THEMES) + '}')
D = Path(sys.argv[1]); ph = sys.argv[2]
env = dict(l.strip().split("=", 1) for l in open(Path(__file__).with_name("friendli.env"), encoding="utf-8") if "=" in l)
rows = [json.loads(l) for l in io.open(D / f"{ph}.jsonl", encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("answer")]


def code(r):
    body = json.dumps({"model": env["INTERVIEW_MODEL"], "temperature": 0.0, "max_tokens": 80, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": str(r["answer"].get("reason") or "")}]}).encode()
    for k in range(4):
        try:
            req = request.Request(env["INTERVIEW_BASE_URL"].rstrip("/") + "/chat/completions", data=body,
                                  headers={"Content-Type": "application/json", "Authorization": "Bearer " + env["INTERVIEW_API_KEY"]})
            with request.urlopen(req, timeout=120) as resp:
                t = json.loads(resp.read())["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", t, re.S)
            j = json.loads(m.group(0)) if m else {}
            th = [k for k in THEMES if str(j.get(k, 0)).strip() in ("1", "True", "true")]
            return dict(aid=r["aid"], themes=th)
        except Exception:
            time.sleep(4 * (k + 1))
    return dict(aid=r["aid"], themes=None)


with cf.ThreadPoolExecutor(8) as ex, open(D / f"themes_{ph}.jsonl", "w", encoding="utf-8") as f:
    for x in ex.map(code, rows):
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("done", len(rows))
