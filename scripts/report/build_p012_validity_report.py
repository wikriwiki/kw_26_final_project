"""P012 유효성 보고서를 한 장으로 묶는다 — 맞춘 것과 못 맞춘 것을 같은 표에.

    python scripts/report/build_p012_validity_report.py \
        --score <dir>/score/score_full.json \
        --roster-manifest <dir>/roster.manifest.json \
        --dossier-manifest <dir>/on/dossier.jsonl.manifest.json \
        --interviews output/interviews/p012m/interviews.jsonl \
        --out output/report/P012_VALIDITY.md

## 이 보고서가 지켜야 하는 것

1. **못 맞댄 지표를 지우지 않는다.** 20개 전부 행을 갖고, 대조 불가면 이유를 적는다.
2. **자를 병기한다.** 창이 31일이 아니면 크기를 맞댈 수 없다는 사실을 표 위에 적는다.
   눈금이 옮겨진 항(총액·제외분)은 옛 읽기를 SUSPECT 로 표시한다.
3. **인터뷰 답변을 자료로 쓰지 않는다.** 원장 숫자를 옆에 놓고, 원장에 없는 숫자를
   말한 인터뷰에는 표시를 남긴다.
4. **보존 증거를 싣는다.** 명부 주변분포 오차, dossier 완전성, 덤프 검증.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "data/experiments/P012_indicator_contract.json"

# 창이 짧아도 맞댈 수 있는 것 — 무차원이거나 부호만 보는 것
WINDOW_SAFE = {"K11", "K12", "K15", "K17", "K19"}


def load(p: str | None):
    if not p:
        return None
    try:
        t = Path(p).read_text(encoding="utf-8")
    except OSError:
        return None
    if p.endswith(".jsonl"):
        return [json.loads(x) for x in t.splitlines() if x.strip()]
    return json.loads(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True)
    ap.add_argument("--roster-manifest", default="")
    ap.add_argument("--dossier-manifest", default="")
    ap.add_argument("--interviews", default="")
    ap.add_argument("--window-days", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sc = load(a.score)
    if not sc:
        raise SystemExit("score json을 읽을 수 없다: %s" % a.score)
    rows = {r["id"]: r for r in sc["rows"]}
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    rman, dman = load(a.roster_manifest), load(a.dossier_manifest)
    iv = load(a.interviews) or []
    days = a.window_days or int(sc.get("window_days") or 0)

    L: list[str] = []
    w = L.append
    w("# P012 상생소비지원금 — 시뮬레이션 유효성")
    w("")
    w("짝지은 시민 **%s명** · 창 %s" % ("{:,}".format(sc["n_agents"]), sc.get("window") or "-"))
    w("")
    w("정책 없음/있음 두 팔을 **같은 사람**에게 돌린 반사실 대조다. 원문(KDI)은 수급/미수급")
    w("**다른 사람**을 맞대어 평행추세가정이 필요했는데, 이 설계는 그 가정이 없다.")
    w("")

    if days and days != 31:
        w("## 먼저 읽어야 할 것 — 창이 한 달이 아니다")
        w("")
        w("실측 수치는 **10월 한 달 누적**이다. 이 런은 **%d일**이다. 비율 지표의 **크기**는" % days)
        w("맞댈 수 없다 — 단위 문제가 아니라 **시점 프로파일** 문제다. 증가율은 이미 무차원이라")
        w("효과가 달 안에서 고르다면 창과 무관해야 하지만, 파일럿 원장을 잘라 재면 같은 런에서")
        w("7일 +18.9% / 14일 +15.5% / 21일 +13.8% / 31일 +11.3% 로 움직인다(효과가 감쇠한다).")
        w("")
        w("실제 정책의 **주별** 프로파일이 없으므로 보정할 수 없다. 우리 모델의 감쇠로 보정하면")
        w("우리 모델의 채점 기준을 우리 모델로 조정하는 순환이 된다. 그래서 보정하지 않는다.")
        w("")
        w("**이 창에서 온전히 맞댈 수 있는 것**: 무차원·부호 지표 — %s."
          % ", ".join(sorted(WINDOW_SAFE)))
        w("흐름량(K13 1인당 캐시백·K20)은 균등가정을 명시하고 창 길이로 환산한다.")
        w("")

    w("## 지표 20개 — 하나도 빼지 않는다")
    w("")
    w("| 지표 | 이름 | 실측 | 시뮬 | 95% 구간 | 부호확실 | 쌍체p | 판정 | 창에 걸리나 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for ind in c["indicators"]:
        iid = ind["id"]
        r = rows.get(iid) or {}
        # 실측의 자는 계약서가 들고 있다. %가 아닌 것을 %처럼 적으면 읽는 이가
        # 맞댈 수 없는 두 수를 나란히 놓는다(K13 47,880원을 +47880.00 으로 적은 사례).
        unit = ind.get("unit") or "%"
        tv = ind.get("truth")
        if tv is None:
            tv = ind.get("truth_gap")
            if tv is not None:
                unit = ind.get("unit") or "%p"
        if not isinstance(tv, (int, float)):
            tv_s = "없음"
        elif unit in ("%", "%p"):
            tv_s = "%+.2f%s" % (tv, unit)
        else:
            tv_s = "{:,}{}".format(int(tv), unit)
        sim = r.get("sim")
        su = r.get("unit") or ("%p" if iid in ("K11", "K12") else unit)
        if not isinstance(sim, (int, float)):
            sim_s = "-"
        elif su in ("%", "%p"):
            sim_s = "%+.2f%s" % (sim, su)
        else:
            sim_s = "{:,.0f}{}".format(sim, su)
        ci = r.get("ci") or [None, None]
        ci_s = ("%.1f ~ %+.1f" % (ci[0], ci[1])) if ci[0] is not None else "-"
        conf = r.get("sign_conf")
        p = r.get("sign_test_p")
        st = r.get("status") or "-"
        # 자 1 을 못 넘은 행에 '수준 안' 을 적지 않는다. 구간이 -84~+437 처럼 넓으면
        # 실측이 그 안에 있는 것은 정보가 아니다.
        decided = (r.get("sign_conf") or 0) >= 0.975
        if decided and r.get("truth_in_band") is True:
            st += " · 수준 안"
        elif decided and r.get("truth_in_band") is False:
            st += " · 수준 밖"
        if r.get("why"):
            st = r["why"][:60]
        if r.get("status") == "이질성":
            if r.get("dir_total"):
                sim_s = "방향 %s/%s 칸" % (r.get("dir_ok"), r.get("dir_total"))
            elif r.get("spread") is not None:
                sim_s = "퍼짐 %.1f%%p (영분포 p=%s)" % (
                    r["spread"], ("%.3f" % r["p"]) if r.get("p") is not None else "-")
        safe = "무관" if iid in WINDOW_SAFE else ("걸린다" if days and days != 31 else "-")
        w("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            iid, ind["name"][:26], tv_s, sim_s, ci_s,
            ("%.1f%%" % (100 * conf)) if conf is not None else "-",
            ("%.3f" % p) if p is not None else "-", st, safe))
    w("")
    t = sc.get("tally") or {}
    w("합계 — 방향 일치 **%s** · 불일치 %s · 판정불가 %s · 대조불가 %s (자 1·2 모두 통과 %s)"
      % (t.get("일치"), t.get("불일치"), t.get("판정불가"), t.get("대조불가"), t.get("두자통과")))
    w("")
    w("판정의 자 셋: **부호확실성 ≥ 97.5%** (부트스트랩에서 부호가 안 뒤집힌다) ·")
    w("**쌍체부호검정 p < 0.05** (사람 단위로도 쏠린다, 동점 공개) ·")
    w("**수준** (실측이 시뮬 95% 구간 안). 자 1을 못 넘으면 부호가 맞아도 적중으로 세지 않는다.")
    w("")

    w("## 눈금 — 옮겨진 것을 적는다")
    w("")
    w("적립 몫을 상수에서 풀면서(`EXP_ELIGIBLE_CHANNEL`) **적립분은 항등**이지만 총액과")
    w("제외분이 2.4배 작아졌다. 옛 상수가 앵커 과대 2.40배 교정을 함께 안고 있었기 때문이다.")
    w("따라서 옛 런의 총액·제외분·재정배수 값은 이 표와 **같은 표에 놓을 수 없다**(SUSPECT).")
    w("")

    if rman or dman:
        w("## 보존 — 시뮬 뒤 남은 것")
        w("")
        w("| 요구 | 증거 |")
        w("|---|---|")
        if rman:
            w("| 명부가 서울 분포에 맞나 | 주변분포 최대오차 **%.5f** (문턱 %s) · 소득교정 **%s** |"
              % (rman.get("worst_margin_error", -1), 0.02,
                 "안 함" if rman.get("income_calibrated") is False else rman.get("income_calibrated")))
            w("| 범위 밖·그래프에 없는 사람 | 연령 범위 밖 %s명 · 그래프에 없음 %s명 제외 |"
              % (rman.get("out_of_scope_dropped"), rman.get("not_in_graph_dropped")))
        if dman:
            tt = dman.get("totals") or {}
            full = tt.get("states") == dman.get("agents", 0) * dman.get("expected_days", 0)
            w("| 기억·스케줄·지출 | 기억 **%s**건 · 계획항목 %s건 · 상태 %s건 (%s) |"
              % ("{:,}".format(tt.get("memories", 0)), "{:,}".format(tt.get("plan_items", 0)),
                 "{:,}".format(tt.get("states", 0)),
                 "명부x일수와 일치 — 손실 없음" if full else "**불일치 — 확인 필요**"))
            w("| 상태 날짜 빠짐 | %s |"
              % ("없다" if not dman.get("state_day_gaps") else
                 "**%d명** — manifest 에 기록" % len(dman["state_day_gaps"])))
        w("")

    if iv:
        bad = [r for r in iv if r.get("unverified_total")]
        w("## 1대1 인터뷰 — 같은 사람의 반사실 옆에서")
        w("")
        w("반응 5분위 × 소비수준 3분단으로 칸을 만들어 칸마다 뽑았다. **반응이 0·음수인 칸을**")
        w("**포함한다** — 반응 큰 사람만 뽑으면 이야기가 저절로 맞는다.")
        w("")
        w("인터뷰 **%d건** · 원장에 없는 숫자를 말한 인터뷰 **%d건**" % (len(iv), len(bad)))
        w("")
        w("답변은 모델이 쓴 글이고 **자료가 아니다.** 자료는 원장이며, 인터뷰는 그 원장이 어떤")
        w("기전으로 만들어졌는지 사람의 말로 읽는 것이다. 그래서 원장 숫자를 옆에 둔다.")
        w("")
        for r in sorted(iv, key=lambda x: x.get("response", 0)):
            on = (r.get("totals") or {}).get("on") or {}
            off = (r.get("totals") or {}).get("off") or {}
            w("### %s · 칸 %s · 반응 %+.1f%%" % (r["aid"], r.get("cell"), 100 * r.get("response", 0)))
            w("")
            w("원장: 정책 없음 **%s원** → 있음 **%s원** · 기억 %s건"
              % ("{:,.0f}".format(float(off.get("actual_spent") or 0)),
                 "{:,.0f}".format(float(on.get("actual_spent") or 0)),
                 on.get("memories")))
            if r.get("unverified_total"):
                w("")
                w("> **주의**: 이 인터뷰에 원장에 없는 숫자가 있다 — %s"
                  % [x["unverified_numbers"] for x in r["answers"] if x["unverified_numbers"]])
            w("")
            for x in r.get("answers") or []:
                if not x.get("a"):
                    continue
                w("**%s** — %s" % (x["q"], x["a"].replace("\n", " ").strip()[:600]))
                w("")

    w("## 맞대지 못한 것 — 이유와 함께")
    w("")
    for ind in c["indicators"]:
        r = rows.get(ind["id"]) or {}
        if r.get("status") in ("일치", "불일치"):
            continue
        why = r.get("why") or {
            "판정불가": "구간이 0 을 지난다 — 표본 %s명 필요" % (r.get("needed_n") or "?"),
            "수준": "자가 달라 방향 셈에서 뺐다",
            "이질성": "원문이 방향만 적었다 — 방향으로만 판정",
        }.get(r.get("status"), r.get("status") or "미구현")
        w("- **%s %s** — %s" % (ind["id"], ind["name"][:34], why))
    w("")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    io.open(out, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("지표 %d개 · 인터뷰 %d건 · %d줄" % (len(c["indicators"]), len(iv), len(L)))
    print("→ %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
