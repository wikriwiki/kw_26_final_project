"""쌓인 기억으로 1대1 인터뷰를 한다 — **같은 사람의 두 팔을 나란히 놓고.**

    # 맥락만 만들고 확인 (모델 없이, 어디서나 실행 가능)
    python scripts/report/interview_agents.py --dir data/experiments/p012t_dossier \
        --per-cell 1 --dry-run --out output/interviews/p012t

    # 실제 인터뷰 (서버에서, LLM 필요)
    python scripts/report/interview_agents.py --dir <dossier dir> --per-cell 2 \
        --out output/interviews/p012m

## 왜 이것이 우리만 할 수 있는 것인가

집계는 "총소비가 몇 % 올랐다"까지만 말한다. 우리는 **같은 사람을 정책 없음/있음으로
두 번 살게 했으므로**, 한 사람에게 "당신은 무엇을 바꿨나"를 물으면서 **바꾸지 않은
자신**을 옆에 둘 수 있다. 실측 자료로는 불가능하다 — 같은 사람의 반사실이 없다.

## 뽑는 규칙 — 고르지 않는다

반응이 큰 사람만 뽑으면 이야기가 저절로 맞는다. 그래서 **반응 크기 5분위 x 소비수준
3분단**으로 칸을 만들고 칸마다 같은 수를 뽑는다. **반응이 0 인 칸과 음수인 칸을
반드시 포함한다.** 뽑힌 사람 목록과 칸을 산출물에 적는다.

## 맥락 — dossier 에 있는 것만

프로필, 날짜별 지출·적립·문턱 상태, 그날 간 가게와 **본인이 적은 선택 이유**,
지갑·기분. 두 팔을 날짜로 맞춰 나란히 넣는다. 원장에 없는 수치는 넣지 않는다.

## 환각 검사 — 답에 있는 숫자가 원장에 있는가

답변에서 숫자를 모두 뽑아 dossier 의 숫자 집합과 맞춘다. 맞지 않는 숫자는
`unverified_numbers` 로 남긴다. **답변은 자료가 아니다** — 자료는 원장이고, 인터뷰는
그 원장이 어떤 기전으로 만들어졌는지 사람의 말로 읽는 것이다. 보고서는 둘을 나란히
싣고, 검증되지 않은 숫자가 있으면 그 인터뷰에 표시를 남긴다.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]

QUESTIONS = [
    ("awareness", "이번 달에 카드 쓰는 방식에 영향을 준 제도나 소식이 있었나요? 있었다면 "
                  "언제 어떻게 알게 되었는지 기억나는 대로 말해 주세요."),
    ("changed", "지난 한 주 동안 평소와 달라진 지출이 있었나요? 있었다면 어떤 것이었고, "
                "없었다면 왜 그대로였는지 말해 주세요."),
    ("where", "평소보다 더 쓴 곳이 있다면 어디였고, 왜 하필 그곳이었나요?"),
    ("threshold", "'얼마 이상 써야 돌려받는다'는 기준을 의식했나요? 의식했다면 그것이 "
                  "어떤 판단을 바꿨는지, 아니라면 왜 신경 쓰지 않았는지 말해 주세요."),
    ("forgone", "쓰고 싶었지만 못 쓴 것이 있었나요? 무엇이 걸림돌이었나요?"),
    ("wallet", "지갑 사정이 지출 판단에 어떻게 작용했나요?"),
]

SYSTEM = (
    "당신은 서울에 사는 한 시민입니다. 아래 '내 기록'에 적힌 것만 근거로 1인칭으로 "
    "답하십시오.\n"
    "규칙:\n"
    "- 기록에 없는 금액·가게·날짜를 만들지 마십시오. 기억나지 않으면 '기억나지 않는다'고 "
    "하십시오.\n"
    "- 정책의 효과가 얼마인지 평가하거나 통계를 말하지 마십시오. 당신이 겪은 일만 "
    "말하십시오.\n"
    "- 짧게, 한 질문에 3~5문장으로 답하십시오.\n"
)


def load(path: Path) -> dict[str, dict]:
    out = {}
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["aid"]] = r
    return out


def nums_in(obj) -> set[int]:
    """dossier 한 사람 안의 모든 정수 — 환각 검사의 기준 집합."""
    seen: set[int] = set()

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            v = int(round(float(o)))
            if abs(v) >= 1:
                seen.add(v)
            return
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            for m in re.findall(r"-?\d[\d,]*", o):
                try:
                    seen.add(int(m.replace(",", "")))
                except ValueError:
                    pass

    walk(obj)
    return seen


def unverified(answer: str, allowed: set[int]) -> list[int]:
    """답변의 숫자 중 원장에 없는 것. 10 이하와 연도는 뺀다(문장 속 흔한 수)."""
    bad = []
    for m in re.findall(r"-?\d[\d,]*", answer or ""):
        try:
            v = int(m.replace(",", ""))
        except ValueError:
            continue
        if abs(v) <= 10 or 1900 <= v <= 2100:
            continue
        if v in allowed:
            continue
        # 만원 단위로 적는 습관 — 1,000 배수도 원장에 있으면 통과로 본다
        if any(abs(v * k - a) <= max(1, a * 0.005) for k in (1, 10, 100, 1000, 10000)
               for a in allowed if a):
            continue
        bad.append(v)
    return sorted(set(bad))


def day_lines(rec: dict, limit_items: int = 4) -> list[str]:
    """날짜별 한 줄 — 지출·적립·지갑·그날 간 가게와 본인이 적은 이유."""
    st_by_day = {s["day"]: s for s in rec.get("states") or []}
    out = []
    for p in rec.get("plans") or []:
        d = p["day"]
        items = [i for i in (p.get("items") or []) if float(i.get("actual_spent") or 0) > 0]
        items.sort(key=lambda i: -float(i.get("actual_spent") or 0))
        spent = sum(float(i.get("actual_spent") or 0) for i in items)
        s = st_by_day.get(d) or {}
        head = "  %s (%s) 지출 %s원" % (d, p.get("day_type") or "", "{:,.0f}".format(spent))
        if s.get("balance") is not None:
            head += " · 잔액 %s원" % "{:,.0f}".format(float(s["balance"]))
        if s.get("sangsaeng_month_spent") is not None:
            head += " · 적립대상 이달 누적 %s원" % "{:,.0f}".format(float(s["sangsaeng_month_spent"]))
        out.append(head)
        for i in items[:limit_items]:
            out.append("      - %s %s원 · %s" % (
                i.get("poi") or i.get("category") or "?",
                "{:,.0f}".format(float(i.get("actual_spent") or 0)),
                (i.get("pick_reason") or i.get("reasoning") or "")[:110]))
    return out


def context_for(on: dict, off: dict) -> str:
    """두 팔을 나란히 놓은 '내 기록'. 우리가 **딱지를 붙이지는 않는다.**

    '정책이 있던 주' 라고 적어 주면 답이 그 방향으로 쏠리므로 가·나 로만 제시한다.
    다만 기록 자체에 정책이 드러난다 — 그 주의 선택 이유에 본인이 "[적립] 업종이라
    골랐다" 고 적어 두었기 때문이다. **그것을 지우지 않는다.** 지우면 기억을 위조하는
    것이고, 애초에 정책이 그 사람에게 닿았다는 사실 자체가 읽어야 할 것이다.

    평소 지출 기준선은 넣지 않는다. 앵커(s_daily_wd)는 보정 전 눈금이라 실제 지출의
    2.4배이고, 그것을 '평소' 로 적으면 본인이 평소보다 훨씬 덜 쓴 것으로 읽는다.
    **가 주가 곧 그 사람의 평소다** — 반사실이 바로 옆에 있다.
    """
    p = on.get("profile") or {}
    head = ["[나]",
            "  %s세 %s · %s · %s 거주 · 소득 %s · 성향 %s" % (
                p.get("age"), "여성" if p.get("gender") == "F" else "남성",
                p.get("job") or "-", p.get("residence_dong") or "-",
                p.get("income_level") or "-", p.get("spending_tendency") or "-"),
            "  생활 %s" % (p.get("lifestyle") or "-"),
            ""]
    body = ["[내 기록 · 주 가]"] + day_lines(off) + ["", "[내 기록 · 주 나]"] + day_lines(on)
    mem = []
    for r, tag in ((off, "가"), (on, "나")):
        for m in (r.get("memories") or [])[:8]:
            if m.get("summary"):
                mem.append("  (%s) %s %s" % (tag, m.get("day"), m["summary"][:150]))
    if mem:
        body += ["", "[그때 내가 남긴 메모]"] + mem
    return "\n".join(head + body)


def cells(on: dict[str, dict], off: dict[str, dict]) -> tuple[dict, dict]:
    """(칸 -> 사람 목록, 사람 -> 지표). 반응 5분위 x 소비수준 3분단."""
    common = sorted(set(on) & set(off))
    resp, lvl = {}, {}
    for a in common:
        o = float((off[a].get("totals") or {}).get("actual_spent") or 0)
        n = float((on[a].get("totals") or {}).get("actual_spent") or 0)
        resp[a] = (n - o) / o if o > 0 else 0.0
        try:
            lvl[a] = int((on[a].get("profile") or {}).get("spending_level_wd") or 0)
        except (TypeError, ValueError):
            lvl[a] = 0
    rs = sorted(resp.values())

    def q(v):
        if not rs:
            return 0
        k = sum(1 for x in rs if x < v)
        return min(4, int(5 * k / max(1, len(rs))))

    ls = sorted(lvl.values())

    def t(v):
        if not ls:
            return 0
        k = sum(1 for x in ls if x < v)
        return min(2, int(3 * k / max(1, len(ls))))

    table = defaultdict(list)
    for a in common:
        table[(q(resp[a]), t(lvl[a]))].append(a)
    return table, {"resp": resp, "lvl": lvl}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="on.dossier.jsonl / off.dossier.jsonl 이 있는 곳")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--dry-run", action="store_true", help="맥락만 만들고 모델을 부르지 않는다")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--limit-agents", type=int, default=0, help="앞에서 N명만 (배선 확인용)")
    ap.add_argument("--limit-questions", type=int, default=0, help="앞에서 N문항만 (배선 확인용)")
    a = ap.parse_args()

    d = Path(a.dir)
    on, off = load(d / "on.dossier.jsonl"), load(d / "off.dossier.jsonl")
    table, met = cells(on, off)
    import random
    rnd = random.Random(a.seed)
    picked = []
    for cell in sorted(table):
        ids = sorted(table[cell])
        rnd.shuffle(ids)
        for x in ids[:a.per_cell]:
            picked.append((cell, x))

    print("# 1대1 인터뷰 — 같은 사람의 두 팔")
    print()
    print("  dossier %s · 짝지은 시민 %d명" % (a.dir, len(set(on) & set(off))))
    print("  칸 = 반응 5분위 x 소비수준 3분단 · 칸마다 %d명" % a.per_cell)
    print("  뽑힌 사람 **%d명** (반응 0·음수 칸을 포함한다)" % len(picked))
    print()
    print("  %-10s %-34s %10s %6s" % ("칸", "사람", "반응", "수준"))
    for cell, x in picked:
        print("  %-10s %-34s %+9.1f%% %6d" % (str(cell), x, 100 * met["resp"][x], met["lvl"][x]))

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    client = None
    if not a.dry_run:
        sys.path.insert(0, str(ROOT / "scripts" / "sim"))
        from llm_client import call_chat            # noqa: E402
        client = call_chat

    if a.limit_agents:
        picked = picked[:a.limit_agents]
        print()
        print("  ** 배선 확인 모드 — %d명만 인터뷰한다 **" % len(picked))
    qlist = QUESTIONS[:a.limit_questions] if a.limit_questions else QUESTIONS

    rows = []
    for cell, aid in picked:
        ctx = context_for(on[aid], off[aid])
        allowed = nums_in(on[aid]) | nums_in(off[aid])
        rec = {"aid": aid, "cell": list(cell), "response": met["resp"][aid],
               "spending_level": met["lvl"][aid],
               "totals": {"on": (on[aid].get("totals") or {}),
                          "off": (off[aid].get("totals") or {})},
               "context_chars": len(ctx), "answers": []}
        for key, q in qlist:
            if client is None:
                rec["answers"].append({"q": key, "a": None, "unverified_numbers": []})
                continue
            # call_chat 은 응답 객체를 돌려준다(usage·choices 를 쓰기 위해).
            resp = client(os.environ.get("LLM_MODE"), SYSTEM + "\n" + ctx, q,
                          temperature=0.7, max_tokens=a.max_tokens)
            txt = ((resp.choices[0].message.content or "") if getattr(resp, "choices", None)
                   else str(resp or "")).strip()
            rec["answers"].append({"q": key, "a": txt,
                                   "unverified_numbers": unverified(txt, allowed)})
        rec["unverified_total"] = sum(len(x["unverified_numbers"]) for x in rec["answers"])
        rows.append(rec)
        io.open(outdir / ("context_%s.txt" % aid), "w", encoding="utf-8", newline="\n").write(ctx)

    io.open(outdir / "interviews.jsonl", "w", encoding="utf-8", newline="\n").write(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    bad = [r for r in rows if r["unverified_total"]]
    print()
    print("  맥락 %d건 저장 · 인터뷰 %s"
          % (len(rows), "건너뜀(--dry-run)" if a.dry_run else "%d건" % len(rows)))
    if not a.dry_run:
        print("  **원장에 없는 숫자를 말한 인터뷰 %d건** (있으면 보고서에 표시한다)" % len(bad))
        for r in bad:
            print("     %s  %s" % (r["aid"], [x["unverified_numbers"]
                                              for x in r["answers"] if x["unverified_numbers"]]))
    print()
    print("→ %s/interviews.jsonl" % a.out)
    print("→ %s/context_<aid>.txt" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
