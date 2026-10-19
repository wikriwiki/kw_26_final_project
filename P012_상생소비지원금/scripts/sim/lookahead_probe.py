# -*- coding: utf-8 -*-
"""직접 프로빙 — 익명화된 정책 블록만 보여 주고 모델이 그것을 식별하는지 묻는다.

Sarkar & Vafa (ICML 2025) 는 "시점 경계를 지키라"는 지시만으로는 누출이 막히지
않는다고 보고한다. 그렇다면 남는 길은 **누출이 실제로 있는지 직접 재는 것**이다.

돌리기 전에 읽는 법을 정해 둔다.
  ① 모델이 실제 정책 이름을 대면  → 식별 성공. 암기·누출이 살아 있다는 뜻이고
     우리 수치에 그 사실을 붙여 보고해야 한다.
  ② 대지 못하면                  → 암기 가설이 약해진다. 증명은 아니다 —
     이름을 모르면서 결과만 외웠을 수도 있어 결과 회상도 따로 묻는다.
  ③ 가짜 정책(P090)을 실재한다고 답하면 → 작화다. 그 경우 ①의 '식별'도
     작화일 수 있으므로 프로빙 자체의 신뢰도를 깎아야 한다.
가짜 정책이 대조군이다. 실제 정책만 맞히고 가짜는 모른다고 해야 식별이 진짜다.
"""
import json, os, sys, urllib.request
from datetime import date
sys.path.insert(0, "/data/repo/scripts/sim")
os.environ["EXP_POLICY_ANONYMOUS"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
from dawn_context import _format_policy_facts

BASE = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("LLM_MODEL", "")
_CORE = {"id","name","type","description","announce_date","effective_from","effective_until",
 "benefit_rate","cap_per_agent","threshold_ratio","eligible_marker","income_grants",
 "excluded_income","decile_grants","excluded_deciles","grant_key","poi_restricted","notes",
 "target_districts","benefit_categories","render_mode","sections","_notes"}

def row(pid):
    p = json.load(open(f"/data/repo/data/neo4j_load/policies/{pid}.json", encoding="utf-8"))
    ef, eu = p["effective_from"], p["effective_until"]
    d = lambda s: date(*map(int, s.split("-")))
    return {"id": p["id"], "name": p["name"], "type": p["type"], "description": p.get("description"),
            "rate": p.get("benefit_rate"), "cap": p.get("cap_per_agent"),
            "threshold_ratio": p.get("threshold_ratio"), "eligible_marker": p.get("eligible_marker"),
            "mech_params": json.dumps({k: v for k, v in p.items() if k not in _CORE}, ensure_ascii=False),
            "poi_restricted": p.get("poi_restricted"), "from_": d(ef), "until_": d(eu),
            "effective_from": ef, "effective_until": eu, "income_grants": {}, "excluded_income": [],
            "decile_grants": {}, "excluded_deciles": [], "grant_key": p.get("grant_key"),
            "regions": p.get("target_districts") or ["서울특별시"], "region_codes": ["11"], "target_l1s": []}

def ask(msg, temp=0.7):
    # 시뮬과 같은 호출 조건 — EXAONE-4.5 는 기본이 thinking ON 이라 끄지 않으면
    # 추론 서문이 토큰을 다 먹고 답이 잘린다(llm_client._extra_body_for 와 동일).
    body = {"model": MODEL, "messages": [{"role": "user", "content": msg}],
            "temperature": temp, "max_tokens": 500,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()

Q_ID = ("아래는 어떤 소비 지원 정책의 설명이다. 이 정책이 실제로 시행된 적이 있는 정책인지,\n"
        "있다면 정확한 이름·시행 국가·시행 연월을 답하라. 모르면 반드시 \"모름\" 이라고만 답하라.\n"
        "추측해서 지어내지 마라.\n\n형식:\n실재여부: 실재 / 모름\n이름:\n국가:\n시행연월:\n\n"
        "--- 정책 설명 ---\n{blk}")
Q_OUT = ("아래 정책이 시행된 뒤 소비가 어떻게 변했는지 아는가? 측정된 결과를 아는 경우에만\n"
         "수치와 함께 답하고, 모르면 반드시 \"모름\" 이라고만 답하라. 추측하지 마라.\n\n"
         "--- 정책 설명 ---\n{blk}")

TRIALS = int(os.environ.get("PROBE_TRIALS", "5"))
POLS = sys.argv[1:] or ["P012", "P015", "P013", "P014", "P090"]
out = {}
for pid in POLS:
    blk = _format_policy_facts([row(pid)])
    print("=" * 74); print(f"[{pid}] 에이전트가 보는 블록")
    print(blk); print()
    rec = {"block": blk, "id_answers": [], "out_answers": []}
    for i in range(TRIALS):
        rec["id_answers"].append(ask(Q_ID.format(blk=blk)))
        rec["out_answers"].append(ask(Q_OUT.format(blk=blk)))
    for tag, key in (("식별", "id_answers"), ("결과회상", "out_answers")):
        print(f"--- {tag} {TRIALS}회 ---")
        for i, a in enumerate(rec[key], 1):
            print(f"  [{i}] " + " / ".join(a.split("\n"))[:230])
    print()
    out[pid] = rec
json.dump(out, open("/data/probe/probe_raw.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("원문 저장: /data/probe/probe_raw.json")
