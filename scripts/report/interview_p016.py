"""P016(농할 농축산물 할인) 1대1 인터뷰 — 결제 기록과 기억만으로 묻고, 같은 사람의 정책 없는 하루로 답을 검증한다.

    # 맥락만 만들어 확인(모델 없이)
    python scripts/report/interview_p016.py --dir <dossier dir> --out <out> --per-cell 2 --dry-run
    # 실제 인터뷰(LLM 필요)
    python scripts/report/interview_p016.py --dir <dossier dir> --out <out> --per-cell 2

## 무엇이 P012 인터뷰(interview_agents.py)와 다른가
1. **정책 있는 쪽의 그 사람만 인터뷰한다.** 실제 인터뷰처럼 자기 기록만 보고 답한다. 정책 없는 쪽(반사실)은
   보여 주지 않고, 답을 검증하는 데만 쓴다.
2. **맥락은 행사 기간(정책 시작일 이후)의 기록이 먼저다.** 날짜별 가게·산 것·금액·그중 농축산물 금액·결제 할인을
   적고, 기억은 장보기·할인 기억을 먼저 고른다(예전 도구는 앞 8건이라 정책 전 주 기억만 들어갔다).
3. **반사실 질문의 답을 같은 사람·같은 날의 정책 없는 하루와 맞댄다.** "할인이 없었다면?"에
   A(같은 곳에서 비슷하게) / B(다른 가게에서) / C(덜 사거나 안 샀다) / D(모르겠다) 중 하나를 고르게 하고,
   정책 없는 쪽 같은 날의 실제 장보기로 A/B/C 를 정해 일치율을 낸다 — 미시 유효성 지표.
4. **환각 검사**: 답의 숫자(interview_agents.unverified)와 가게 이름이 기록에 있는가.

## 뽑는 칸 — 고르지 않는다
(행사 기간에 결제 할인을 받았나: 예/아니오) × (소비수준 3분단). 칸마다 같은 수. 할인을 안 받은 사람도 반드시 묻는다.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from interview_agents import load, nums_in, unverified  # noqa: E402

GROCERY = {"슈퍼마켓", "식료품", "청과", "정육", "수산", "장보기", "마트"}
CHAIN_RE = re.compile(r"^(이마트(?!24| ?에브리데이)|롯데마트|롯데쇼핑롯데마트|.*하나로(마트|클럽)|GS ?더 ?프레시|지에스더프레시|GS ?수퍼)")

QUESTIONS = [
    ("awareness", "7월 말부터 8월 초 사이에 장을 볼 때 알게 된 할인 행사나 소식이 있었나요? 있었다면 무엇이었고 "
                  "어떻게 알게 되었는지, 없었다면 '없었다'고 말해 주세요."),
    ("store", "그 기간에 장을 본 가게를 고른 이유는 무엇이었나요? 평소에 가던 곳과 같았나요?"),
    ("discount", "장을 보면서 결제할 때 할인을 받은 적이 있나요? 있다면 언제, 어느 가게에서, 얼마였는지 기록에 있는 대로 "
                 "말해 주세요. 없다면 왜 없었는지 말해 주세요."),
    ("counterfactual", "그 기간에 장을 본 날 중 하루를 떠올려 보세요. 결제할 때 받은 할인이 없었다면 그날 장보기를 어떻게 했을 것 같나요? "
                       "먼저 A(같은 가게에서 비슷한 만큼 샀다) / B(다른 가게에서 샀다) / C(덜 샀거나 사지 않았다) / D(모르겠다) 중 "
                       "하나를 '선택: A' 처럼 적고, 어느 날 이야기인지와 이유를 말해 주세요."),
    ("extra", "할인 때문에 원래 계획보다 더 산 농축산물(과일·채소·고기 등)이 있었나요? 무엇을 얼마나 샀는지 말해 주세요."),
    ("savings", "할인으로 아낀 돈이 있었다면 그 돈이 다른 지출에 영향을 줬나요?"),
]

SYSTEM = (
    "당신은 서울에 사는 한 시민입니다. 아래 '내 기록'에 적힌 것만 근거로 1인칭으로 답하십시오.\n"
    "규칙:\n"
    "- 기록에 없는 금액·가게·날짜를 만들지 마십시오. 기억나지 않으면 '기억나지 않는다'고 하십시오.\n"
    "- 정책의 효과를 평가하거나 통계를 말하지 마십시오. 당신이 겪은 일만 말하십시오.\n"
    "- 짧게, 한 질문에 3~5문장으로 답하십시오.\n"
)


def _money(v) -> int:
    try:
        return int(round(float(v or 0)))
    except (TypeError, ValueError):
        return 0


def _is_grocery(item: dict) -> bool:
    return (item.get("sub_category") in GROCERY or item.get("category") == "마트"
            or (item.get("produce_spent") is not None))


def window_days(rec: dict, start: str) -> list[dict]:
    return [p for p in (rec.get("plans") or []) if str(p.get("day")) >= start]


def day_lines(rec: dict, start: str, limit_items: int = 5) -> list[str]:
    st_by_day = {s["day"]: s for s in rec.get("states") or []}
    out = []
    for p in window_days(rec, start):
        d = p["day"]
        items = [i for i in (p.get("items") or []) if _money(i.get("actual_spent")) > 0]
        items.sort(key=lambda i: (not _is_grocery(i), -_money(i.get("actual_spent"))))
        spent = sum(_money(i.get("actual_spent")) for i in items)
        disc = sum(_money(i.get("discount_total")) for i in items)
        s = st_by_day.get(d) or {}
        head = f"  {d} ({p.get('day_type') or ''}) 지출 {spent:,}원"
        if disc:
            head += f" · 결제 할인 {disc:,}원(낸 돈 {spent - disc:,}원)"
        if s.get("balance") is not None:
            head += f" · 잔액 {_money(s['balance']):,}원"
        out.append(head)
        for i in items[:limit_items]:
            extra = []
            if i.get("produce_spent") is not None and _money(i.get("produce_spent")) > 0:
                extra.append(f"그중 농축산물 {_money(i['produce_spent']):,}원")
            if _money(i.get("discount_total")):
                extra.append(f"결제 할인 {_money(i['discount_total']):,}원")
            out.append("      - %s %s원%s · %s · %s" % (
                i.get("poi") or i.get("category") or "?", f"{_money(i.get('actual_spent')):,}",
                f" ({', '.join(extra)})" if extra else "", (i.get("menu") or "")[:40],
                (i.get("pick_reason") or i.get("reasoning") or "")[:100]))
    return out


def memory_lines(rec: dict, start: str, k: int = 12) -> list[str]:
    mems = [m for m in (rec.get("memories") or []) if str(m.get("day")) >= start and m.get("summary")]
    # 장보기·할인 기억 먼저, 그다음 날짜순
    mems.sort(key=lambda m: (-(1 if _money(m.get("discount_total")) else 0),
                             -(1 if (m.get("category") in GROCERY or m.get("produce_spent")) else 0),
                             str(m.get("day"))))
    return [f"  {m['day']} {(m.get('store') or '')} {m['summary'][:160]}" for m in mems[:k]]


def known_days(on: dict, start: str) -> list[str]:
    """그날 아침 이 행사를 알고 있었던 날(상태의 policy_lifecycle 에 P016 이 켜진 날). 시뮬에서 정책을 아는 길은 아침 소식뿐이다."""
    out = []
    for s in on.get("states") or []:
        if str(s.get("day")) < start:
            continue
        try:
            lc = json.loads(s.get("policy_lifecycle") or "{}")
        except (TypeError, ValueError):
            lc = {}
        if lc.get("P016"):
            out.append(str(s["day"]))
    return out


def context_for(on: dict, start: str) -> str:
    p = on.get("profile") or {}
    head = ["[나]",
            "  %s세 %s · %s · %s 거주 · 소득 %s · 성향 %s" % (
                p.get("age"), "여성" if p.get("gender") == "F" else "남성", p.get("job") or "-",
                p.get("residence_dong") or "-", p.get("income_level") or "-", p.get("spending_tendency") or "-"),
            "  생활 %s" % (p.get("lifestyle") or "-"), ""]
    kd = known_days(on, start)
    # [2026-10-11] 시험 인터뷰에서 기록이 없으니 '직원 안내·안내판' 같은 경로를 지어냈다. 시뮬에서 실제로 알게 된 길(아침 소식)과 날짜를 적는다.
    news = ([f"[내가 알고 있던 소식]", f"  {kd[0][5:7].lstrip('0')}월 {kd[0][8:].lstrip('0')}일부터 아침에 보는 소식으로 '대한민국 농할갑시다' 행사를 알고 있었다 "
             "— 이마트·롯데마트·농협하나로마트·GS더프레시 매장에서 국산 신선 농축산물 값의 20%를 결제할 때 깎아 주고, 유통업체마다 1인 최대 1만원.", ""]
            if kd else ["[내가 알고 있던 소식]", "  이 기간에 들은 할인 행사 소식은 기록에 없다.", ""])
    body = news + ["[내 기록 — 날짜별 지출]"] + day_lines(on, start)
    mem = memory_lines(on, start)
    if mem:
        body += ["", "[그때 내가 남긴 메모]"] + mem
    return "\n".join(head + body)


def discounted_days(on: dict, start: str) -> list[str]:
    return [p["day"] for p in window_days(on, start)
            if any(_money(i.get("discount_total")) for i in (p.get("items") or []))]


def off_behavior(on: dict, off: dict, day: str) -> str | None:
    """같은 날 정책 없는 쪽의 실제 장보기로 반사실 답(A/B/C)을 정한다. 정책 있는 쪽 그날 할인 결제가 기준."""
    on_items = [i for p in window_days(on, day) if p["day"] == day for i in (p.get("items") or [])
                if _money(i.get("discount_total"))]
    if not on_items:
        return None
    ref = max(on_items, key=lambda i: _money(i.get("actual_spent")))
    off_items = [i for p in (off.get("plans") or []) if p["day"] == day for i in (p.get("items") or [])
                 if _money(i.get("actual_spent")) > 0 and _is_grocery(i)]
    if not off_items:
        return "C"
    same = [i for i in off_items if (i.get("poi_id") and i.get("poi_id") == ref.get("poi_id"))
            or (CHAIN_RE.search(i.get("poi") or "") and CHAIN_RE.search(ref.get("poi") or "")
                and (i.get("poi") or "")[:3] == (ref.get("poi") or "")[:3])]
    on_amt = _money(ref.get("actual_spent"))
    off_amt = sum(_money(i.get("actual_spent")) for i in off_items)
    if same and off_amt >= 0.7 * on_amt:
        return "A"
    if off_amt < 0.7 * on_amt:
        return "C"
    return "B"


def off_behavior_window(on: dict, off: dict, start: str) -> str | None:
    """행사 기간 전체로 본 반사실. 할인은 장보는 날을 당기거나 미룰 수 있어(10명 시험: 할인받은 날 정책 없는 쪽은
    그날 장을 안 보고 다른 날 봤다) 같은 날만 맞대면 C 로 기운다. 기간 합으로 한 번 더 본다."""
    on_items = [i for p in window_days(on, start) for i in (p.get("items") or []) if _money(i.get("discount_total"))]
    if not on_items:
        return None
    on_amt = sum(_money(i.get("actual_spent")) for i in on_items)
    ids = {i.get("poi_id") for i in on_items if i.get("poi_id")}
    heads = {(i.get("poi") or "")[:3] for i in on_items if CHAIN_RE.search(i.get("poi") or "")}
    off_gro = [i for p in window_days(off, start) for i in (p.get("items") or [])
               if _money(i.get("actual_spent")) > 0 and _is_grocery(i)]
    off_amt = sum(_money(i.get("actual_spent")) for i in off_gro)
    same_amt = sum(_money(i.get("actual_spent")) for i in off_gro
                   if i.get("poi_id") in ids or (CHAIN_RE.search(i.get("poi") or "") and (i.get("poi") or "")[:3] in heads))
    if same_amt >= 0.7 * on_amt:
        return "A"
    if off_amt < 0.7 * on_amt:
        return "C"
    return "B"


# 답에 '결제 할인을 받았다'는 주장이 있는가(문장 단위, 같은 문장에 부정이 있으면 주장 아님). 어림 규칙 — 표시만 하고 사람이 읽는다.
_CLAIM_POS = re.compile(r"할인[^.!?\n]{0,20}?(받았|받은|받아|적용받|적용됐|적용되었)")
_CLAIM_NEG = re.compile(r"(적(은|이)?\s*없|받지\s*않|받지\s*못|못\s*받|없었|없습니다|없다|기억나지\s*않|있지\s*않|않다|불확실|확실하지|확인(되지|할 수)|여부)")


def claims_discount(ans: str) -> bool:
    for s in re.split(r"(?<=[.!?])\s+|\n", ans or ""):
        if _CLAIM_POS.search(s) and not _CLAIM_NEG.search(s):
            return True
    return False


def stated_choice(ans: str) -> str | None:
    m = re.search(r"선택\s*[:：]?\s*([ABCD])", ans or "")
    return m.group(1) if m else None


def stated_day(ans: str, days: list[str]) -> str | None:
    for d in days:
        mm, dd = d[5:7].lstrip("0"), d[8:10].lstrip("0")
        if d in (ans or "") or f"{mm}월 {dd}일" in (ans or "") or f"{mm}/{dd}" in (ans or ""):
            return d
    return None


def stores_in(rec: dict) -> set[str]:
    return {i.get("poi") for p in rec.get("plans") or [] for i in (p.get("items") or []) if i.get("poi")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2020-07-30", help="정책 시작일(이 날부터의 기록을 맥락에 넣는다)")
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--aid", action="append", default=[])
    a = ap.parse_args()
    d = Path(a.dir)
    # 실행기 보존 결과는 on/dossier.jsonl, 손 보존 사본은 on.dossier.jsonl 로 둔 적이 있다 — 둘 다 받는다.
    _dos = lambda arm: d / arm / "dossier.jsonl" if (d / arm / "dossier.jsonl").exists() else d / f"{arm}.dossier.jsonl"
    on, off = load(_dos("on")), load(_dos("off"))
    common = sorted(set(on) & set(off))
    lvl = {x: int((on[x].get("profile") or {}).get("spending_level_wd") or 0) for x in common}
    ls = sorted(lvl.values())
    tier = {x: min(2, int(3 * sum(1 for v in ls if v < lvl[x]) / max(1, len(ls)))) for x in common}
    table = defaultdict(list)
    for x in common:
        table[(bool(discounted_days(on[x], a.start)), tier[x])].append(x)
    rnd = random.Random(a.seed)
    picked = []
    for cell in sorted(table):
        ids = sorted(table[cell]); rnd.shuffle(ids)
        picked += [(cell, x) for x in ids[:a.per_cell]]
    if a.aid:
        picked = [(("지정",), x) for x in a.aid]
    print(f"# P016 1대1 인터뷰 · 짝지은 시민 {len(common)}명 · 뽑힌 사람 {len(picked)}명")
    for cell in sorted(table):
        print(f"  칸 할인받음={cell[0]} 소비수준={cell[1]} : {len(table[cell])}명")
    client = None
    if not a.dry_run:
        sys.path.insert(0, str(ROOT / "scripts" / "sim"))
        from llm_client import call_chat  # noqa: E402
        client = call_chat
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell, aid in picked:
        # 에이전트 기록에는 정책 ID(P016)가 그대로 남아 있다 — 사람이 쓰는 말(행사 이름)로만 바꿔 보여 준다(숫자·사실은 그대로).
        ctx = context_for(on[aid], a.start).replace("P016", "농할 행사")
        allowed = nums_in(on[aid])
        ddays = discounted_days(on[aid], a.start)
        rec = {"aid": aid, "cell": list(cell), "discounted_days": ddays, "context_chars": len(ctx), "answers": []}
        for key, q in QUESTIONS:
            if key == "counterfactual" and not ddays:
                continue  # 할인을 받지 않은 사람에게 '받은 할인이 없었다면'은 성립하지 않는다
            if client is None:
                rec["answers"].append({"q": key, "a": None}); continue
            resp = client(os.environ.get("LLM_MODE"), SYSTEM + "\n" + ctx, q, temperature=0.7, max_tokens=a.max_tokens)
            txt = ((resp.choices[0].message.content or "") if getattr(resp, "choices", None) else str(resp or "")).strip()
            item = {"q": key, "a": txt, "unverified_numbers": unverified(txt, allowed)}
            if key == "counterfactual":
                ch = stated_choice(txt); dd = stated_day(txt, ddays) or (ddays[0] if ddays else None)
                truth = off_behavior(on[aid], off[aid], dd) if dd else None
                item.update({"stated": ch, "day": dd, "off_arm": truth,
                             "agree": (ch == truth) if (ch in "ABC" and truth) else None})
                tw = off_behavior_window(on[aid], off[aid], a.start)
                item.update({"off_arm_window": tw, "agree_window": (ch == tw) if (ch in "ABC" and tw) else None})
            rec["answers"].append(item)
        said = [x for x in rec["answers"] if x["q"] in ("awareness", "discount", "savings") and x.get("a")]
        rec["claims_discount"] = any(claims_discount(x["a"]) for x in said) if said else None
        rec["claim_mismatch"] = (rec["claims_discount"] != bool(ddays)) if said else None
        rec["unverified_total"] = sum(len(x.get("unverified_numbers") or []) for x in rec["answers"])
        rows.append(rec)
        io.open(outdir / f"context_{aid}.txt", "w", encoding="utf-8", newline="\n").write(ctx)
    io.open(outdir / "interviews_p016.jsonl", "w", encoding="utf-8", newline="\n").write(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    if not a.dry_run:
        cf = [x for r in rows for x in r["answers"] if x["q"] == "counterfactual" and x.get("agree") is not None]
        print(f"  반사실 답과 정책 없는 쪽 실제 행동 일치 {sum(1 for x in cf if x['agree'])}/{len(cf)}")
        cw = [x for r in rows for x in r["answers"] if x["q"] == "counterfactual" and x.get("agree_window") is not None]
        print(f"  반사실 답과 정책 없는 쪽 행사 기간 전체 행동 일치 {sum(1 for x in cw if x['agree_window'])}/{len(cw)}")
        print(f"  할인 받음 여부를 기록과 다르게 말한 인터뷰 {sum(1 for r in rows if r.get('claim_mismatch'))}/{len(rows)}")
        print(f"  원장에 없는 숫자를 말한 인터뷰 {sum(1 for r in rows if r['unverified_total'])}/{len(rows)}")
    print(f"→ {a.out}/interviews_p016.jsonl · context_<aid>.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
