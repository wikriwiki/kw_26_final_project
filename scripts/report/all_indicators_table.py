"""**모든** 검증지표를 정책별로 실측과 나란히 놓는다 — 하나도 빼지 않고.

    python scripts/report/all_indicators_table.py

## 왜 전부인가

핵심 하나만 보면 **나머지가 어디서 막혔는지 안 보인다.** 정책이 유효한지는 지표
하나가 아니라 지표 묶음이 말한다 — 총소비는 맞았는데 업종 구성이 틀렸다면 그것도
알아야 고칠 수 있다.

## `error_budget.py` 와 무엇이 다른가

`error_budget` 은 **더할 수 있는 것만** 센다(같은 단위·같은 창). 그래서 값이 있어도
빠진다. 이 표는 **빼지 않는다.** 대신 왜 그냥 더할 수 없는지를 줄마다 적는다.

    대조가능   실측·시뮬이 같은 단위로 있다              -> 오차를 낸다
    단위다름   둘 다 있는데 단위·창이 다르다             -> 값만 병기하고 오차는 비운다
    정의확인   추정량·기간·모집단·분모 정합 확인이 남았다    -> 값만 병기하고 오차는 비운다
    동등성검증 실측 크기 대신 등록한 무효과 밴드로 평가한다   -> 기존 CI·판정을 보존한다
    방향만     시뮬은 있는데 정답지에 수치가 없다          -> 부호로만 판정
    없음       시뮬 값이 없다                          -> **메워야 할 자리**

## 값은 어디서 오나

`data/experiments/scoring_table.json` 의 `result_*` 블록. 지표마다 `pct·mean·ci·n`
이 들어 있다. 블록이 여러 개면 **같은 프롬프트 안에서 표본이 가장 큰 것**을 읽는다 —
후보를 가로질러 고르면 기각된 후보의 수를 정답지와 맞대게 된다.

순위지표(`expect=rank`)는 크기가 `got` 문자열에만 있다("A +3.1% vs B -7.7%").
거기서 **간격**을 뽑아 실측 간격과 맞댄다.
"""
from __future__ import annotations

import argparse
from datetime import date
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORING = ROOT / "data/experiments/scoring_table.json"
DASH = str.maketrans({"−": "-", "–": "-", "—": "-"})

# 기각된 후보의 수를 정답지와 맞대면 안 된다. 선택은 v5 다.
REJECTED = ("v45", "v51", "v7", "v8", "v9")

# 정책별 핵심 지표 — 표에 ★ 로 표시만 한다(무엇을 먼저 볼지 알려 주려고).
CORE = {"P010": "P010-1", "P012": "P012-1", "EMERGENCY_2020": "EM-3",
        "DISTANCING_2020": "DS-1", "LOCAL_VOUCHER": "LV-2",
        "SECTOR_VOUCHER_2020": "HO-1", "P016": "C1", "GATHERING_2020": "GA-3"}

# 단위가 실측과 맞지 않는 지표 — 값은 보여 주고 오차는 비운다.
UNIT_NOTE = {
    "P010-1": "자기보고 신규소비 비율. 시민 200명 재표집 CI; 조사·표본·기간 및 원장 출처 확인 필요",
    "P012-4": "실측 47,880원은 10~11월 수령자 평균. 10월 수령자당 약 47,828원; 옛 시뮬 값은 전체 시민·10월 일부 기간",
    "P012-3": "정답지가 '> 0' 이라 크기가 없다",
    "P012-6": "실측 21.0%는 10~11월 수령자 중 상한 도달률. 10월 표는 약 20.87%; 옛 시뮬 값은 전체 시민·10월 일부 기간",
    "PT-1": "무효과는 동등성 밴드·CI로 평가. 실측 크기 부재는 단위 불일치가 아님",
    "PT-2": "무효과는 동등성 밴드·CI로 평가. 실측 크기 부재는 단위 불일치가 아님",
    "LV-1": "실측이 '영향 미미' — 수치가 아니다",
    "DS-6": "상권 유형 자료가 없다(외부 자료 필요)",
}


def truth_of(desc, explicit=None):
    """실측 — **`실측` 이라고 적힌 자리에서만** 읽는다.

    자유 서술에서 숫자를 주워 오면 정책 파라미터를 결과로 읽는다("20% 할인" 의 20%).
    """
    srcs = []
    if explicit:
        srcs.append(str(explicit))
    if desc:
        m = re.search(r"\(실측\s*([^)]*)\)", str(desc))
        if m:
            srcs.append(m.group(1))
    for t in srcs:
        t = t.translate(DASH)
        if "원" in t and "%" not in t:
            m = re.search(r"([\d,]+)\s*원", t)
            if m:
                return float(m.group(1).replace(",", "")), "원"
        m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(%p|%)", t)
        if m:
            return float(m.group(1)), m.group(2)
        m = re.search(r"([+-]?\d*\.\d+)", t)
        if m:
            return float(m.group(1)), ""
    return None, None


def truth_gap(desc, explicit=None):
    """순위 실측의 **간격** — `(실측 +10.8%p vs +3.6%p)` 에서 7.2."""
    for src in (explicit, desc):
        if not src:
            continue
        t = str(src).translate(DASH)
        m = re.search(r"\(실측\s*([^)]*)\)", t) or re.search(r"(.*)", t)
        nums = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", m.group(1))
        if len(nums) >= 2:
            return float(nums[0]) - float(nums[1])
    return None


def sim_gap(entry):
    """순위 시뮬의 간격 — `got` 문자열 "A +3.1% vs B -7.7%" 에서 10.8."""
    t = str((entry or {}).get("got") or "").translate(DASH)
    nums = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", t)
    if len(nums) >= 2:
        return float(nums[0]) - float(nums[1])
    return None


def from_note(entry):
    """두 팔 설계의 결과는 `note` 문자열에만 있다. (상태, 값, 단위) 를 돌려준다.

    P015(8대 쿠폰)는 기준선 팔과 정책 팔을 따로 돌리고 그 **차이**가 결과다.
    숫자 필드는 기준선 블록에만 있어서, 그것을 읽으면 **무정책 팔의 수를 정책
    결과로 보고한다** — 실제로 HO-1 을 +16.1%(기준선)로 적었다. 진짜 값은
    "기준선 +16.1% → 정책 +37.6% · 차 +21.5%p" 처럼 note 에 있다.

    그리고 note 는 숫자만 담지 않는다. **"없음"·"무효"·"관측부족"·"밴드 안" 은
    전혀 다른 상태다** — 값을 못 내는 것과, 냈는데 자가 틀린 것과, 관측이 모자란
    것을 한 칸에 뭉개면 무엇을 고쳐야 하는지가 사라진다. 억지로 숫자를 뽑지 않는다.
    """
    t = str((entry or {}).get("note") or "").translate(DASH)
    if not t:
        return None
    if "무효" in t:
        return "무효", None, None
    if "관측" in t and ("너무 적" in t or "부족" in t):
        return "관측부족", None, None
    m = re.search(r"차\s*([+-]?\d+(?:\.\d+)?)\s*%p", t)
    if m:
        return "값", float(m.group(1)), "%p"
    m = re.search(r"정책\s*([+-]?\d+(?:\.\d+)?)\s*%", t)
    if m:
        return "값", float(m.group(1)), "%"
    m = re.search(r"차\s*([+-]?[\d,]+)(?!\s*%)", t)
    if m:
        return "값", float(m.group(1).replace(",", "")), "원"
    if "밴드" in t:
        return "밴드안", None, None
    return None


def best_block(pol, iid):
    """**기준선 블록은 결과로 읽지 않는다.** note 가 있는 블록을 먼저 본다."""
    noted, best = None, None
    for bk, bv in pol.items():
        if not isinstance(bv, dict) or any(r in bk for r in REJECTED):
            continue
        e = bv.get(iid)
        if not isinstance(e, dict):
            continue
        if "baseline" in bk:
            continue          # note 역시 무정책 팔을 정책 효과로 바꾸지 못한다
        if from_note(e) is not None and noted is None:
            noted = (bk, e)
        if not any(isinstance(e.get(k), (int, float)) for k in ("pct", "mean")) \
                and sim_gap(e) is None:
            continue
        if best is None or (e.get("n") or 0) > (best[1].get("n") or 0):
            best = (bk, e)
    return noted or best


def classify(ind, entry, tv, tu, suspect=False):
    """상태와 (오차, 시뮬표시). suspect 면 오차를 내지 않는다."""
    iid, expect = ind["id"], ind.get("expect")
    audit_status = (ind.get("empirical_audit") or {}).get("comparison")
    suspect = suspect or audit_status == "different_estimand"
    # **지표 정의가 스스로 '측정 불가' 라고 말하면 그것이 맞다.**
    # DS-6 의 결과 블록에는 "관측부족" 으로 적혀 있지만, 지표 desc 는 "그래프에
    # hub_type 이 0건" 이라고 한다. 둘은 다르다 — 표본을 늘려 되는 것과 자료를
    # 구해야 되는 것을 한 칸에 두면 무엇을 할지가 사라진다.
    d = str(ind.get("desc") or "")
    if "측정 불가" in d or ind.get("simulation_cannot_produce"):
        return "자료없음", None, "-"
    if entry is None:
        return "없음", None, "-"
    nt = from_note(entry)
    if nt:
        kind, v, u = nt
        if kind != "값":
            return kind, None, "-"
        shown = "%+.2f%s" % (v, u)
        if suspect:
            return "다른자", None, shown
        if tv is None:
            return "방향만", None, shown
        return ("대조가능", abs(v - tv), shown) if u == tu else ("단위다름", None, shown)
    if expect == "rank":
        g, tg = sim_gap(entry), truth_gap(ind.get("desc"), entry.get("실측"))
        shown = "간격 %+.1f%%p" % g if g is not None else "-"
        if g is None:
            return "없음", None, "-"
        if suspect:
            return "다른자", None, shown
        if tg is None:
            return "방향만", None, shown
        return "대조가능", abs(g - tg), shown
    pct, mean = entry.get("pct"), entry.get("mean")
    if isinstance(pct, (int, float)):
        shown = "%+.2f%%" % pct
        if audit_status == "unverified_source":
            return "원문미확인", None, shown
        if suspect:
            return "다른자", None, shown
        if tv is None:
            return "방향만", None, shown
        if tu == "%":
            return "대조가능", abs(pct - tv), shown
        return "단위다름", None, shown
    if isinstance(mean, (int, float)):
        shown = "%.4g" % mean
        if iid in ("P012-3", "P012-6"):
            shown = "%.2f%%" % (100 * mean)
        elif iid == "P012-4":
            shown += "원"
        if suspect:
            return "다른자", None, shown
        if audit_status == "unverified_source":
            return "원문미확인", None, shown
        if iid in ("P010-1", "P012-4", "P012-6"):
            return "정의확인", None, shown
        if tv is None:
            return ("동등성검증" if expect == "0" else "방향만"), None, shown
        # 단위 일치는 필요조건이다. 무차원이라는 이유만으로 같은 추정량은 아니다.
        if tv is not None and tu == "":
            return "정의확인", None, shown
        return "단위다름", None, shown
    return "없음", None, "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="")
    ap.add_argument("--md-out", default="", help="정책별 38개 지표를 Markdown 표로 저장")
    a = ap.parse_args()
    sc = json.loads(SCORING.read_text(encoding="utf-8"))

    # 다른 자로 잰 것은 오차에 넣지 않는다 — sign_scoreboard 가 그 목록을 갖고 있다.
    import importlib.util
    _p = ROOT / "scripts" / "report" / "sign_scoreboard.py"
    _s = importlib.util.spec_from_file_location("sign_scoreboard", _p)
    _m = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(_m)
    SUSPECT = getattr(_m, "SUSPECT", {})

    print("# 모든 검증지표 — 정책별로, 빼지 않고")
    print()
    tally, rows = {}, []
    for pk, pol in sc.items():
        if not isinstance(pol, dict) or "indicators" not in pol:
            continue
        print("## %s" % pk)
        print("   %-8s %-5s %11s %13s %7s %9s  %s"
              % ("지표", "기대", "실측", "시뮬", "n", "오차", "상태 / 단서"))
        for ind in pol["indicators"]:
            iid = ind["id"]
            hit = best_block(pol, iid)
            bk, entry = hit if hit else (None, None)
            tv, tu = truth_of(ind.get("desc"), (entry or {}).get("실측"))
            audit = ind.get("empirical_audit") or {}
            if "reported_value" in audit:
                tv, tu = audit["reported_value"], audit["reported_unit"]
            if ind.get("expect") == "rank":
                tg = audit.get("reported_gap")
                if tg is not None:
                    tshow = "간격 %+.1f%s" % (tg, audit["reported_unit"])
                else:
                    tg = truth_gap(ind.get("desc"), (entry or {}).get("실측"))
                    tshow = "간격 %+.1f%%p" % tg if tg is not None else "없음"
            else:
                if tv is None:
                    tshow = "없음"
                elif tu == "원":
                    tshow = f"{tv:+,.0f}원"
                elif tu == "log-point":
                    tshow = "%+.4g log-point" % tv
                else:
                    tshow = "%+.4g%s" % (tv, tu)
            if audit.get("comparison") == "unverified_source":
                tshow = "출처미확인(" + tshow + ")"
            sus = (pk, bk, iid) in SUSPECT
            st, err, shown = classify(ind, entry, tv, tu, sus)
            tally[st] = tally.get(st, 0) + 1
            note = audit.get("reason") \
                or (SUSPECT.get((pk, bk, iid)) if sus else None) \
                or UNIT_NOTE.get(iid) or (bk[:30] if bk else "채점된 런이 없다")
            print("   %-8s %-5s %11s %13s %7s %9s  %s%s"
                  % (("★" if CORE.get(pk) == iid else " ") + iid,
                     str(ind.get("expect")), tshow, shown,
                     str((entry or {}).get("n") or "-"),
                     # 무차원 지표의 오차에 %p 를 붙이면 거짓이 된다
                     ("-" if err is None else
                      ("%.2f%%p" % err if tu in ("%", "%p") else "%.4f" % err)),
                     st, "  " + note))
            rows.append({"policy": pk, "id": iid, "expect": ind.get("expect"),
                         "truth": tv, "truth_unit": tu, "truth_display": tshow,
                         "shown": shown,
                         "n": (entry or {}).get("n"), "err": err,
                         "status": st, "block": bk, "note": note,
                         "metric": ind.get("metric"),
                         "ci": (entry or {}).get("ci"),
                         "registered_hit": (entry or {}).get("hit")})
        print()

    print("## 합계 — 지표 %d개" % len(rows))
    for st in ("대조가능", "원문미확인", "정의확인", "동등성검증", "단위다름", "다른자", "방향만", "관측부족", "무효", "밴드안", "자료없음", "없음"):
        print("   %-8s %3d개" % (st, tally.get(st, 0)))
    # %p 오차만 더한다. 무차원 비율(MPC)의 오차를 섞으면 총합이 뜻을 잃는다.
    pp = [r["err"] for r in rows
          if r["err"] is not None and r.get("truth_unit") in ("%", "%p")]
    other = [r for r in rows
             if r["err"] is not None and r.get("truth_unit") not in ("%", "%p")]
    if pp:
        print()
        print("   %%p 로 맞댈 수 있는 %d개의 오차 합 **%.2f%%p** · 평균 %.2f%%p"
              % (len(pp), sum(pp), sum(pp) / len(pp)))
    for r in other:
        print("   %s %s 는 무차원이라 따로 — 실측 %.4g · 시뮬 %s · 오차 **%.4f**"
              % (r["policy"], r["id"], r["truth"], r["shown"], r["err"]))
    print()
    print("   '없음' 은 시뮬 값이 아예 없는 자리다 — 메워야 한다.")
    print("   '정의확인' 은 단위뿐 아니라 추정량·기간·분모·출처를 확인해야 한다.")
    print("   '대조가능' 은 기존 분류다. 단위 일치만으로 외적 타당성이 검증되지는 않는다.")
    print("   registered_hit 는 기존 등록 시험의 판정이며 실측 크기 일치 판정이 아니다.")

    if a.json_out:
        io.open(a.json_out, "w", encoding="utf-8", newline="\n").write(
            json.dumps({"rows": rows, "tally": tally}, ensure_ascii=False, indent=1))
        print()
        print("→ %s" % a.json_out)
    if a.md_out:
        lines = [f"# 정책별 검증지표 비교 현황 ({date.today().isoformat()})", "",
                 "채점표의 모든 지표를 표시한다. '대조가능'은 단위 기준의 잠정 분류이며 추정량·모집단·기간의 완전한 정합을 보증하지 않는다. 진행 중인 런의 최종 결과는 포함하지 않는다.", "",
                 "`n`의 단위는 결과마다 다르다. P010은 시민 200명과 반복 시민-일 1,971개다. 등록된 hit는 실측 크기 일치 판정이 아니다.", "",
                 "| 정책 | 지표 | 실측 | 시뮬 | n | 상태 | 확인 사항 |",
                 "|---|---|---:|---:|---:|---|---|"]
        for r in rows:
            values = [r["policy"], r["id"], r["truth_display"], r["shown"],
                      str(r["n"]) if r["n"] is not None else "-", r["status"], r["note"]]
            lines.append("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ")
                                          for v in values) + " |")
        lines += ["", "상태별 개수: " + ", ".join(f"{k} {v}" for k, v in tally.items()) + ".",
                  "", "원문과 시뮬 출력의 정의 대조: [P010](P010_EMPIRICAL_ALIGNMENT_20260926.md), [P012](P012_EMPIRICAL_ALIGNMENT_20260926.md), [P016](P016_EMPIRICAL_ALIGNMENT_20260926.md)."]
        Path(a.md_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("→ %s" % a.md_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
