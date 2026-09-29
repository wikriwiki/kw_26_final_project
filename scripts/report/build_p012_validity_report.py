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
ARM_LABEL = {"on": "지원금 있음", "off": "지원금 없음"}

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
    ap.add_argument("--preservation", default="", help="verify_p012_preservation.py 의 json")
    ap.add_argument("--restore-dir", default="", help="verify_graph_restore 의 결과 디렉터리")
    ap.add_argument("--interview-check", default="", help="interview_agents.py --check-all 출력 파일")
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
    w("시민 **%s명** · 시뮬레이션 날짜 %s" % ("{:,}".format(sc["n_agents"]), sc.get("window") or "-"))
    w("")
    w("## 비교 설계 — 같은 사람, 같은 날짜를 두 번")
    w("")
    w("같은 시민으로 **같은 날짜**를 두 번 시뮬레이션했다.")
    w("")
    w("1. **상생소비지원금이 있는** 시뮬레이션")
    w("2. **상생소비지원금이 없는** 시뮬레이션")
    w("")
    w("사람·날짜·환경·지갑·성격이 모두 같고 지원금이 있고 없고만 다르다. 그래서 둘의 소비")
    w("차이가 곧 지원금의 효과이고, 이 차이를 KDI 실측과 맞댄다.")
    w("")
    w("### 지원금이 없는 쪽도 왜 9월이 아니라 같은 10월인가")
    w("")
    w("알고 싶은 것은 \"10월에 지원금이 있었을 때\"와 \"**같은 10월에 지원금이 없었다면**\"의")
    w("차이다. 날짜까지 같아야 남는 차이를 지원금 때문이라고 말할 수 있다. 9월을 쓰면 달이")
    w("달라서 생기는 차이가 지원금 효과로 잘못 잡힌다.")
    w("")
    w("- **요일 구성**: 2021년 10월 1일은 금요일, 9월 1일은 수요일이다. 7일 안의 주말 수와")
    w("  위치가 달라지고, 주말 소비는 평일과 크게 다르다.")
    w("- **다른 정책**: 2021년 9월 첫 주는 상생 국민지원금(1인 25만원) 신청이 시작된 때다")
    w("  (9월 6일). 9월을 기준으로 삼으면 '지원금이 없는 7일'이 다른 지원금이 막 풀리는 7일이 된다.")
    w("- **시기**: 9월 초는 추석(9월 20~22일)을 앞둔 때다.")
    w("")
    w("### KDI 는 왜 9월을 썼나")
    w("")
    w("KDI 는 지원금을 받은 사람과 받지 않은 사람, 즉 **서로 다른 사람**을 비교했다. 두 집단은")
    w("원래부터 소비가 달랐으므로(정책 직전 9월에 6.95% 차이 — 지표 K16), 9월 차이를 먼저 재서")
    w("빼야 했다. 이 시뮬레이션은 **같은 사람**을 두 번 돌리므로 원래 차이가 처음부터 0 이다.")
    w("그래서 9월을 따로 잴 필요가 없고, 같은 10월끼리 비교하는 편이 더 깨끗하다. 한 사람을 두 번")
    w("살게 하는 비교는 현실 자료로는 불가능하고 시뮬레이션만 할 수 있다.")
    w("")

    if days and days != 31:
        w("## 한계 — 시뮬레이션 기간")
        w("")
        w("GPU 시간이 유한해서 시뮬레이션 기간을 **%d일**로 잡았다. 실측은 10월 한 달 누적이다." % days)
        w("정책 효과는 달을 지나며 줄어들므로 기간이 짧으면 더 크게 잡힌다(파일럿 원장: 같은")
        w("런에서 7일 +18.9% / 31일 +11.3%). 실제 정책이 주마다 어떻게 변했는지 자료가 없어 보정할 수")
        w("없다. 우리 모델이 만든 감소 추세로 보정하면, 채점받는 모델로 채점 기준을 고치는 셈이 된다.")
        w("**그래서 크기는 판정에서 뺀다.**")
        w("")
        w("판정은 기간 길이에 영향받지 않는 것으로만 한다 — 방향(부호), 순위, 단위 없는 비율(%s)."
          % ", ".join(sorted(WINDOW_SAFE)))
        w("금액·인원 지표(K13·K20)는 한 달이 고르다고 가정하고 기간 길이만큼 나눠 비교한다.")
        w("")
    # [방향을 머리기사로]
    # 창이 짧으면 크기는 못 맞댄다. 그때 남는 것이 방향이고, 그것이 성적이다.
    # 자 1(부호확실성 >= 97.5%)을 넘은 지표만 센다 — 못 넘은 것은 잡음이다.
    judged = [r for r in rows.values() if r.get("status") in ("일치", "불일치")]
    hit = [r for r in judged if r["status"] == "일치"]
    undec = [r for r in rows.values() if r.get("status") == "판정불가"]
    w("## 방향 — 이 기간·인원에서 판정 가능한 것")
    w("")
    if judged:
        w("판정 가능한 **%d개** 중 방향 적중 **%d개 = %.0f%%**"
          % (len(judged), len(hit), 100 * len(hit) / len(judged)))
    else:
        w("이 날짜 범위·인원에서 부호확실성 97.5% 를 넘은 지표가 없다 — 방향을 주장하지 않는다.")
    w("")
    for r in sorted(judged, key=lambda x: (x["status"] != "일치", x["id"])):
        w("- **%s %s** — 실측 %+.2f · 시뮬 %+.2f · 부호확실 %.1f%% · 쌍체p %s → **%s**"
          % (r["id"], (r.get("name") or "")[:26],
             r.get("truth") if r.get("truth") is not None else float("nan"),
             r.get("sim") if r.get("sim") is not None else float("nan"),
             100 * (r.get("sign_conf") or 0),
             ("%.3f" % r["sign_test_p"]) if r.get("sign_test_p") is not None else "-",
             "방향 일치" if r["status"] == "일치" else "방향 불일치"))
    if undec:
        w("")
        w("판정 불가 **%d개** — 부호확실성이 97.5%% 에 못 미친다. 부호가 맞아도 세지 않는다."
          % len(undec))
        w("각 지표가 판정 가능해지는 표본은 표의 `필요n` 이 아니라 채점 출력에 있다.")
    w("")
    w("## 지표 20개 — 하나도 빼지 않는다")
    w("")
    w("| 지표 | 이름 | 실측 | 시뮬 | 95% 구간 | 부호확실 | 쌍체p | 판정 | 기간에 따라 달라지나 |")
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
        if r.get("truth_window") is not None and r.get("window_days"):
            # 금액 지표는 시뮬 기간 길이에 맞춘 기준과 맞댄다 — 두 수를 함께 보인다.
            tv_s = "{:,.0f}{} → {}일 기준 **{:,.0f}{}**".format(
                tv, unit, r["window_days"], r["truth_window"], unit)
        elif not isinstance(tv, (int, float)):
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
        if r.get("ratio") is not None and r.get("truth_window") is not None:
            st = "실측의 %.2f배" % r["ratio"]
        if r.get("why"):
            st = r["why"][:60]
        if r.get("status") == "이질성":
            if r.get("proxy") == "household_composition":
                sim_s = "다인 %+.2f%% · 소 %+.2f%% (격차 %+.2f%%p, 제외 %d명)" % (
                    r["large_pct"], r["small_pct"], r["gap"], r["n_excluded"])
                st = ("구성 대리 · " + ("방향 일치" if r.get("dir_ok") else
                      ("방향 불일치" if r.get("dir_total") else "판정불가")))
            elif r.get("dir_total"):
                sim_s = "방향 %s/%s 칸" % (r.get("dir_ok"), r.get("dir_total"))
            elif r.get("spread") is not None:
                sim_s = "퍼짐 %.1f%%p (영분포 p=%s)" % (
                    r["spread"], ("%.3f" % r["p"]) if r.get("p") is not None else "-")
        safe = "아니다" if iid in WINDOW_SAFE else ("달라진다" if days and days != 31 else "-")
        w("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            iid, ind["name"][:26], tv_s, sim_s, ci_s,
            ("%.1f%%" % (100 * conf)) if conf is not None else "-",
            ("%.3f" % p) if p is not None else "-", st, safe))
    w("")
    t = sc.get("tally") or {}
    w("합계 — 방향 일치 **%s** · 불일치 %s · 판정불가 %s · 대조불가 %s (부호확실성·사람 단위 쏠림 둘 다 통과 %s)"
      % (t.get("일치"), t.get("불일치"), t.get("판정불가"), t.get("대조불가"), t.get("두자통과")))
    w("")
    w("판정 기준 세 가지: **부호확실성 ≥ 97.5%** (다시 뽑아 계산해도 부호가 안 뒤집힌다) ·")
    w("**쌍체부호검정 p < 0.05** (사람 단위로도 쏠린다, 동점 공개) ·")
    w("**수준** (실측이 시뮬 95% 구간 안). 부호확실성을 못 넘으면 부호가 맞아도 적중으로 세지 않는다.")
    w("")

    w("## 눈금 — 옮겨진 것을 적는다")
    w("")
    w("적립 몫을 상수에서 풀면서(`EXP_ELIGIBLE_CHANNEL`) **적립분은 항등**이지만 총액과")
    w("제외분이 2.4배 작아졌다. 옛 상수가 앵커 과대 2.40배 교정을 함께 안고 있었기 때문이다.")
    w("따라서 옛 런의 총액·제외분·재정배수 값은 이 표와 **같은 표에 놓을 수 없다**(SUSPECT).")
    w("")

    pres = load(a.preservation) if a.preservation else None
    if pres:
        w("## 보존 — 검사로 확인한 것")
        w("")
        w("검증지표 비교는 아래 세 가지가 지켜졌을 때만 믿을 수 있다. 각각 **검사**로 확인했다.")
        w("")
        w("| 시뮬레이션 | 그래프 | 메모리·1대1 인터뷰 | 업종 원장 | 캐시백 원장 |")
        w("|---|---|---|---|---|")
        for arm in ("on", "off"):
            x = (pres.get("arms") or {}).get(arm) or {}
            def cell(k):
                v = x.get(k) or {}
                return ("통과 — " if v.get("ok") else "**실패** — ") + " · ".join(v.get("notes") or [])[:90]
            w("| %s | %s | %s | %s | %s |" % (ARM_LABEL[arm], cell("그래프"), cell("메모리·인터뷰"),
                                               cell("업종 원장"), cell("캐시백 원장")))
        w("")
        w("결론: **%s**" % ("세 가지 모두 보존됐다" if pres.get("ok") else "보존 실패 — 아래 비교를 믿을 수 없다"))
        w("")
    rdir = Path(a.restore_dir) if a.restore_dir else None
    if rdir and rdir.is_dir():
        w("### 덤프를 실제로 복원해 대조했다")
        w("")
        w("체크섬은 파일이 그대로라는 것만 말한다. 그래서 두 시뮬레이션의 덤프를 각각 **복원**하고, 기억 수")
        w("분포 전체에서 고르게 뽑은 에이전트의 기억·계획항목·상태 날짜를 dossier 와 맞댔다.")
        w("")
        for arm in ("on", "off"):
            r = load(str(rdir / ("%s.json" % arm)))
            if r:
                w("- %s: 대조 %d명 · 불일치 **%d명**" % (ARM_LABEL[arm], len(r.get("sampled") or []),
                                                    len(r.get("mismatch") or [])))
        w("")
    ic = Path(a.interview_check) if a.interview_check else None
    if ic and ic.is_file():
        w("### 1대1 인터뷰 가능성 — 명부 전원")
        w("")
        for line in ic.read_text(encoding="utf-8").splitlines():
            if not line.startswith("#") and ("가능" in line or "기억 0건" in line or "불가" in line):
                w("- " + line.strip())
        w("")
        w("아무 에이전트나 지목해 인터뷰할 수 있다: `interview_agents.py --aid <id>`.")
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
