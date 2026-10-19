"""인터뷰 답변 본문이 규칙에 어느 쪽으로 기우는지 따로 분류한다(본문 코딩, 응답은 바꾸지 않는다).

입장 칸(stance)은 '나와 직접 관련 있는가'로만 골라져 거의 모두 '중립'이 됐다. 본문에는 이점 · 부담 서술이
남아 있으므로, 본문만 읽고 기울기를 고른다. 흡연 여부 · 이용 여부는 분류기에 보여 주지 않는다(눈가림).
  python code_answer_lean.py responses.jsonl out.jsonl [--workers 48]
"""
import json, sys, os, concurrent.futures as cf
from urllib import request

SYS = """다음은 한 시민의 인터뷰 답변이다. 질문은 '당구장·실내 골프연습장 이용 공간을 금연구역으로 지정하는 규칙'에 대한 입장이었다.
답변 본문만 읽고, 이 시민이 규칙에 대해 드러내는 기울기를 하나 고른다. 입장 칸에 '중립'이라 적었더라도 본문의 서술로 판단한다.
- favorable: 규칙의 이점(쾌적함, 건강, 간접흡연 감소 등)을 긍정적으로 언급하고 부담은 없거나 작게 본다
- unfavorable: 규칙의 부담(흡연 불편, 이용 감소 등)을 더 크게 본다
- mixed: 이점과 부담을 비슷한 비중으로 함께 말한다
- indifferent: 이점이나 부담에 대한 평가 없이 자신과 무관하다고만 말한다
'긍정적이거나 부정적인 입장은 없다'처럼 부정하는 문장은 이점 언급으로 보지 않는다.
JSON만 반환: {"lean":"favorable|unfavorable|mixed|indifferent","evidence":"판단 근거가 된 본문 구절 그대로"}"""


def chat(text, base, model):
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 200, "response_format": {"type": "json_object"},
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": text}]}).encode()
    req = request.Request(base.rstrip("/") + "/chat/completions", data=body, headers={"content-type": "application/json"})
    with request.urlopen(req, timeout=300) as r:
        return json.loads(json.loads(r.read())["choices"][0]["message"]["content"])


def main():
    src, dst = sys.argv[1], sys.argv[2]; workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 48
    base = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"); model = os.environ.get("LLM_MODEL", "LGAI-EXAONE/EXAONE-4.5-33B-AWQ")
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("response")]

    def one(r):
        resp = r["response"]
        text = "답변: " + resp.get("answer", "") + "\n이유: " + " / ".join(resp.get("key_reasons") or [])
        try:
            c = chat(text, base, model)
        except Exception as exc:
            c = {"error": type(exc).__name__}
        return dict(aid=r["aid"], smoking_status=r["smoking_status"], facility_visits_14d=r["facility_visits_14d"],
                    stance=resp.get("stance"), lean=c.get("lean"), evidence=c.get("evidence"), error=c.get("error"))
    with open(dst, "w", encoding="utf-8") as f, cf.ThreadPoolExecutor(workers) as ex:
        for x in ex.map(one, rows):
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("done", len(rows))


if __name__ == "__main__":
    main()
