"""정책 있음 쪽에서 사람마다 신청 여부를 모델이 정한다.

2026-10-08 사용자: "페르소나별로 응답하게 해야해. 정책에 무관심한 페르소나가 있을 수도 있으니까."
P012 본런 첫날(같은 사람 486명) 93% 가 정책을 이유로 들었고 나이·소득과 관계없이 88~96% 였다.
실제 상생소비지원금은 신청한 사람에게만 적용됐고, 카드를 가진 성인 4,317만 명 중 1,566만 명(36.3%)이
신청했다(KDI 2022.9 표 3 총신청건수 15,658,152). 이 숫자는 입력에 넣지 않고 결과 대조에만 쓴다.

모델은 이 사람의 페르소나 사실과 그 사람이 실제로 보게 될 정책 사실 블록, 정답지 원문에 적힌 신청 절차만
보고 (안다, 신청한다, 이유)를 답한다.
[2026-10-08 시험 1] 200명 중 '모름' 122명의 53명이 '입력에 정보가 없어 판단할 수 없다'를 근거로 들었다 — '입력에 없는
뉴스를 만들지 말라'는 문장이 페르소나 대신 판단을 정했다. 정답지 원문의 안내 사실(9/27 콜센터·전용 웹페이지)을 넣고
'기록 없음 = 알 수 없음'으로 바꿨다. 신청률을 맞추려는 고침이 아니다(시험 1 신청률 39%). 형식이 어긋나면 온도를 올려 다시 묻고, 세 번 모두 어긋나면 멈춘다
(대체값 없음). 결과는 Agent.policy_enrolled(목록)와 Agent.enroll_<정책> (JSON 문자열)에 쓴다.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(ROOT / "scripts"))

import dawn_context as dc  # noqa: E402
from llm_client import call_chat  # noqa: E402
from neo4j_load._common import driver_session  # noqa: E402

# 정답지(KDI 2022.9) <Box2> 신청방법과 각주 20(5부제)을 옮긴 사실. 행동 방향은 넣지 않는다.
APPLY_FACTS = {
    "P012": (
        "- 2021년 9월 27일부터 통합 콜센터와 전용 웹페이지로 사업 내용을 안내했다.\n"
        "- 이 제도는 신청한 사람에게만 적용된다. 신청하지 않으면 캐시백이 없다.\n"
        "- 신청: 카드사 홈페이지·모바일웹·앱, 고객센터·콜센터, 카드사 연계 은행 영업점에서 본인 확인 → 참여 동의 → "
        "신청 정보 입력. 신청 결과와 본인의 2분기 평균 사용액은 2영업일 안에 따로 알려 준다.\n"
        "- 시행 첫 주는 출생연도 끝자리 요일제다(10/1 1·6, 10/5 2·7, 10/6 3·8, 10/7 4·9, 10/8 5·0). "
        "신청 기간은 11월 30일까지다."
    ),
}

SYSTEM = (
    "너는 서울 시민 한 사람의 판단을 대신하는 시뮬레이션의 일부다. 오늘 아래 제도가 시행된다.\n"
    "이 사람이 (1) 이 제도가 있다는 것을 아는지, (2) 신청하는지를 이 사람의 나이·직업·생활 방식·"
    "카드 사용과 소비 습관에 비추어 판단한다.\n"
    "- 사람마다 다르다. 이런 제도를 챙기는 사람도 있고, 모르고 지나가는 사람도, 알아도 신청하지 않는 사람도 있다.\n"
    "- 판단 근거는 입력에 있는 이 사람의 사실(나이·직업·생활·카드 사용·소비 성향)에서 찾는다. 입력에 없는 구체적인 "
    "대화·사건을 만들어 내지 않는다. 입력에 이 사람이 안내를 봤다는 기록이 없다는 것은 '알 수 없음'이지 '모른다'는 "
    "뜻이 아니다 — 이 사람의 생활로 판단한다.\n"
    "- 출력은 JSON 하나: {\"knows\": true 또는 false, \"applies\": true 또는 false, "
    "\"reason\": \"이 사람의 어떤 사실 때문인지 한두 문장\"}. applies 가 true 면 knows 도 true 다."
)

# 판(variant)별 문구. 실측 신청률 숫자는 어느 판에도 넣지 않는다(사용자 2026-10-08: 36% 같은 노골적인 수치 금지).
SYSTEM_V1 = (
    "너는 서울 시민 한 사람의 판단을 대신하는 시뮬레이션의 일부다. 오늘 아래 제도가 시행된다.\n"
    "이 사람이 (1) 이 제도가 있다는 것을 아는지, (2) 신청하는지를 이 사람의 나이·직업·생활 방식·"
    "카드 사용과 소비 습관에 비추어 판단한다.\n"
    "- 사람마다 다르다. 이런 제도를 챙기는 사람도 있고, 모르고 지나가는 사람도, 알아도 신청하지 않는 사람도 있다.\n"
    "- 판단 근거는 입력에 있는 이 사람의 사실에서 찾는다. 입력에 없는 대화·뉴스·기억을 만들어 내지 않는다.\n"
    "- 출력은 JSON 하나: {\"knows\": true 또는 false, \"applies\": true 또는 false, "
    "\"reason\": \"이 사람의 어떤 사실 때문인지 한두 문장\"}. applies 가 true 면 knows 도 true 다."
)
# v3: 실제로 신청이 적었던 구조를 사실로만 보인다 — 신청 수고(본인 확인 방법, 정답지 Box2)와, 평소보다 더 써야
# 생기는 혜택을 그 사람의 씀씀이(정책 전 주 적립 업종 결제, 시뮬 자신의 기록)로 따져 보게 한다.
SYSTEM_V3 = (
    "너는 서울 시민 한 사람의 판단을 대신하는 시뮬레이션의 일부다. 오늘 아래 제도가 시행된다.\n"
    "이 사람이 이 제도를 알게 되었는지, 그리고 신청하는지를 판단한다.\n"
    "- 이 사람 입장에서 판단한다: 나이·직업·생활 방식·카드 사용과 소비 습관, 신청에 드는 수고, "
    "그리고 이 사람의 평소 씀씀이로 실제 받게 될 캐시백이 얼마일지.\n"
    "- 입력에 없는 구체적인 대화·사건을 만들어 내지 않는다.\n"
    "- 출력은 JSON 하나: {\"knows\": true 또는 false, \"applies\": true 또는 false, "
    "\"reason\": \"이 사람의 어떤 사실 때문인지 한두 문장\"}. applies 가 true 면 knows 도 true 다."
)
HASSLE_FACTS = {
    "P012": ("- 본인 확인: 온라인은 공동인증서·휴대전화 인증·카드 인증, 콜센터는 주민등록번호 뒷자리·카드 비밀번호, "
             "영업점은 직접 방문.\n"),
}


def _own_spend_line(persona: dict) -> str:
    b = persona.get("sangsaeng_base_daily")
    if not b or float(b) <= 0:
        raise ValueError(f"정책 전 주 적립 업종 결제 기록이 없다: {persona.get('id')}")
    b = int(round(float(b)))
    return (f"- 이 사람의 지난 한 주(9/24~9/30) 적립 업종 카드 결제: 하루 평균 약 {b:,}원(7일이면 약 {b * 7:,}원). "
            "본인의 2분기 평균 사용액은 신청한 뒤에 알려 준다.\n")


def _benefit_arith_line(persona: dict, rows: list[dict]) -> str:
    """제도 산식을 이 사람의 씀씀이에 그대로 적용한 결과(행동 방향 없음). 문턱은 신청자가 받게 될 개인 상태와
    같은 계산(dawn_context._sangsaeng_monthly_anchor × threshold_ratio)이다."""
    r = rows[0]
    ratio, rate, cap = float(r["threshold_ratio"]), float(r["rate"]), int(r["cap"])
    threshold = int(round(dc._sangsaeng_monthly_anchor(persona) * ratio))
    per10k = int(round(10000 * rate))
    return (f"- 계산(제도 산식): 2분기 사용액이 지난 한 주 씀씀이와 같다면 이번 실적 기간 문턱은 약 {threshold:,}원이다. "
            f"평소만큼 쓰면 캐시백은 거의 없고, 문턱을 넘긴 뒤 1만원을 더 쓸 때마다 {per10k:,}원(최대 {cap:,}원).\n")


# [2026-10-08 문구 시험, 명부 10명마다 1명 = 200명, 샌드박스, 실측 신청률 숫자는 어느 판에도 없음]
#   v1(첫 문구)            45.5% — '모름'의 절반이 '입력에 정보가 없어서'(페르소나 아닌 지시문이 판단)
#   v2(안내 사실+기록없음≠모름) 92%  (앞 200명)
#   v3(수고·본인 씀씀이)     93%  — 혜택 크기를 스스로 계산하지 않음
#   v4(v3+제도 산식 결과)    55% / 반복 48.5%, 같은 답 117/200(우연 100) — 온도 0.7 에선 사람별 답이 흔들림
#   v4 온도 0              47% / 42.5%, 같은 답 167/200(우연 101). 나이 40대 89%·70대 8%, 소득 하 14%·상 71%
# → v4·온도 0 으로 고정. 실측(카드 소지 성인 36.3%)에 더 맞추려는 반복은 하지 않는다(사용자 요청: 조금이라도 비슷하게,
#   단 수치를 프롬프트에 쓰지 않는다).
VARIANT = "v4"
TEMP = 0.0


def _prompt(variant: str, persona: dict, rows: list[dict], pid: str, day: dt.date) -> tuple[str, str]:
    facts = APPLY_FACTS[pid]
    if variant == "v1":
        system = SYSTEM_V1
        facts = facts.replace("- 2021년 9월 27일부터 통합 콜센터와 전용 웹페이지로 사업 내용을 안내했다.\n", "")
    elif variant == "v2":
        system = SYSTEM
    elif variant == "v3":
        system = SYSTEM_V3
        facts = facts + "\n" + HASSLE_FACTS[pid] + _own_spend_line(persona)
    elif variant == "v4":
        system = SYSTEM_V3
        facts = facts + "\n" + HASSLE_FACTS[pid] + _own_spend_line(persona) + _benefit_arith_line(persona, rows)
    else:
        raise ValueError(variant)
    user = (f"[이 사람]\n{dc._format_persona(persona)}\n\n"
            f"[오늘 {day.isoformat()} 시행되는 제도 — 사실]\n{dc._format_policy_facts(rows)}\n"
            f"{facts.rstrip()}\n\nJSON 하나만 출력한다.")
    return system, user


SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "enrollment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "knows": {"type": "boolean"},
                "applies": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["knows", "applies", "reason"],
            "additionalProperties": False,
        },
    },
}


def _parse(text: str) -> dict:
    d = json.loads(text)
    if set(d) != {"knows", "applies", "reason"}:
        raise ValueError(f"keys {sorted(d)}")
    if not isinstance(d["knows"], bool) or not isinstance(d["applies"], bool):
        raise ValueError("knows/applies 가 참·거짓이 아니다")
    if d["applies"] and not d["knows"]:
        raise ValueError("모르는데 신청한다")
    reason = " ".join(str(d["reason"] or "").split())
    if len(reason) < 5:
        raise ValueError("이유가 비었다")
    return {"knows": d["knows"], "applies": d["applies"], "reason": reason}


def decide(aid: str, pid: str, day: dt.date) -> dict:
    with driver_session() as s:
        row = s.run(dc.PERSONA_CYPHER, aid=aid).single()
        if row is None:
            raise ValueError(f"페르소나 없음: {aid}")
        persona = dict(row)
        rows = [dict(r) for r in s.run(dc.POLICY_CYPHER, aid=aid, today=day) if r["id"] == pid]
    if not rows:
        return {"aid": aid, "exposed": False, "knows": False, "applies": False,
                "reason": "정책 지역 밖", "attempts": 0}
    system, user = _prompt(VARIANT, persona, rows, pid, day)
    errors = []
    for attempt, temp in enumerate((TEMP, TEMP + 0.1, TEMP + 0.2), start=1):
        resp = call_chat(None, system, user, temperature=temp, max_tokens=400, response_format=SCHEMA)
        text = resp.choices[0].message.content or ""
        try:
            out = _parse(text)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{attempt}: {exc}")
            continue
        out.update({"aid": aid, "exposed": True, "attempts": attempt,
                    "age": persona.get("age_group"), "income": persona.get("income"),
                    "gender": persona.get("gender"), "job": persona.get("job")})
        return out
    raise RuntimeError(f"{aid} 신청 판단 세 번 모두 형식 어긋남: {errors}")


WRITE = """
UNWIND $rows AS r
MATCH (a:Agent {id: r.aid})
SET a.policy_enrolled = [x IN coalesce(a.policy_enrolled, []) WHERE x <> $pid]
                        + CASE WHEN r.applies THEN [$pid] ELSE [] END
SET a[$prop] = r.json
"""


def main() -> int:
    global VARIANT, TEMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-id", required=True)
    ap.add_argument("--roster", required=True)
    ap.add_argument("--day", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-write", action="store_true", help="그래프에 쓰지 않고 파일만")
    ap.add_argument("--stride", type=int, default=1, help="시험용: 명부에서 N명마다 한 명")
    ap.add_argument("--variant", default=VARIANT, choices=("v1", "v2", "v3", "v4"))
    ap.add_argument("--dump-prompt", action="store_true", help="첫 사람의 문구를 출력")
    ap.add_argument("--temp", type=float, default=TEMP)
    a = ap.parse_args()
    VARIANT = a.variant
    TEMP = a.temp
    pid = a.policy_id
    if not re.fullmatch(r"P\d{3}", pid) or pid not in APPLY_FACTS:
        raise SystemExit(f"신청 절차 사실이 없는 정책: {pid}")
    day = dt.date.fromisoformat(a.day)
    roster = json.load(open(a.roster, encoding="utf-8"))
    aids = [r if isinstance(r, str) else r.get("id") or r.get("aid") for r in roster]
    aids = aids[:: max(1, a.stride)]
    if a.limit:
        aids = aids[: a.limit]
    if a.dump_prompt:
        with driver_session() as s:
            persona = dict(s.run(dc.PERSONA_CYPHER, aid=aids[0]).single())
            rows = [dict(r) for r in s.run(dc.POLICY_CYPHER, aid=aids[0], today=day) if r["id"] == pid]
        system, user = _prompt(VARIANT, persona, rows, pid, day)
        print("=== SYSTEM\n" + system + "\n=== USER\n" + user)
        return 0
    results = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(decide, aid, pid, day): aid for aid in aids}
        for i, f in enumerate(cf.as_completed(futs), 1):
            results.append(f.result())   # 실패하면 여기서 멈춘다
            if i % 200 == 0:
                print(f"  {i}/{len(aids)}", flush=True)
    results.sort(key=lambda r: r["aid"])
    exposed = [r for r in results if r["exposed"]]
    n_app = sum(r["applies"] for r in exposed)
    n_know = sum(r["knows"] for r in exposed)
    summary = {"policy_id": pid, "variant": VARIANT, "day": day.isoformat(), "people": len(results), "exposed": len(exposed),
               "knows": n_know, "applies": n_app,
               "applies_rate": round(n_app / max(1, len(exposed)), 4),
               "measured_reference": "성인 카드 소지자 4,317만 명 중 1,566만 명 신청(36.3%, 10/1~11/30), "
                                     "10/17 까지 1,401만 명(32.5%). 대조만 하고 조정하지 않는다.",
               "by": {}}
    for key in ("age", "income", "gender"):
        g = collections.defaultdict(lambda: [0, 0])
        for r in exposed:
            g[str(r.get(key))][0] += 1
            g[str(r.get(key))][1] += r["applies"]
        summary["by"][key] = {k: {"n": v[0], "applies_rate": round(v[1] / v[0], 3)} for k, v in sorted(g.items())}
    summary["attempts"] = dict(collections.Counter(r["attempts"] for r in results))
    Path(a.out).write_text(json.dumps({"summary": summary, "people": results}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    if not a.no_write:
        rows = [{"aid": r["aid"], "applies": r["applies"],
                 "json": json.dumps({k: r[k] for k in ("knows", "applies", "reason")}, ensure_ascii=False)}
                for r in results]
        with driver_session() as s:
            s.run(WRITE, rows=rows, pid=pid, prop=f"enroll_{pid}").consume()
            n = s.run("MATCH (a:Agent) WHERE $pid IN coalesce(a.policy_enrolled, []) RETURN count(a) AS n",
                      pid=pid).single()["n"]
        if n != n_app:
            raise SystemExit(f"그래프 신청자 {n} ≠ 판단 신청자 {n_app}")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
