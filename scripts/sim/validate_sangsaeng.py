# -*- coding: utf-8 -*-
"""상생 백테스트 채점기 (G5) — OFF/ON 집계 → §6 벡터 + §5.4 쌍체검정 + 판정.

입력: aggregate_period.py 가 만든 off.json / on.json (에이전트별 적립/제외/KDI 집계).
비교: 같은 관측창의 같은 에이전트 → 쌍체차 d_i = B_i(ON) − A_i(OFF).

채점 (docs/POLICY_BACKTEST_SANGSAENG.md §6):
  C2 (필수, 축)      : 적립(=사실상 총소비) B−A 쌍체검정, 평균>0 이며 p<0.05.
  C1 (C2 해석 보증)  : 적립 B−A 가 제외(시계·귀금속) B−A 보다 유의하게 큼.
                        제외가 0~소폭 + 는 정답(경보 아님), 적립만큼 튀면 경보.
  D1 (보조, 탐색)    : KDI 8분류 B−A 순위 — 가전·가구 상위군 / 학원·이·미용 하위군?
  D2 (탐색)          : 소비분위별 적립 B−A + 분위별 KDI 쏠림.
  반응 로그(§4.5 ③) : ON 런 trigger='policy' 자기보고 비율 (진단, 게이트 아님).

통계: scipy 있으면 t검정, 없으면 대표본 정규근사(n=2~3천이면 사실상 동일) + 부호검정.

사용:
  python scripts/sim/validate_sangsaeng.py --off off.json --on on.json --out-dir output/sim/report
  python scripts/sim/validate_sangsaeng.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ALPHA = 0.05
# C1 경보 임계: 제외 B−A 평균이 적립 B−A 평균의 이 비율 이상이고 유의하게 +면 '일반 수요효과' 의심.
C1_ALARM_RATIO = 0.5


# =========================================================
# 통계 헬퍼 (순수 파이썬 — 대표본 정규근사, scipy 있으면 정밀 t)
# =========================================================
def _normal_sf(z: float) -> float:
    """P(Z > z), 표준정규 상단꼬리."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _t_sf(t: float, df: int) -> float:
    """t분포 상단꼬리 P(T>t). scipy 있으면 정밀, 없으면 정규근사(df 큼)."""
    try:
        from scipy import stats  # type: ignore
        return float(stats.t.sf(t, df))
    except Exception:
        return _normal_sf(t)


def paired_test(diffs: list[float]) -> dict:
    """쌍체차 리스트 → 평균>0 단측검정 (t + 부호검정).

    반환: n, mean, sd, se, t, p_t(단측), n_pos, n_neg, p_sign(단측), significant.
    """
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "se": 0.0, "t": 0.0,
                "p_t": 1.0, "n_pos": 0, "n_neg": 0, "p_sign": 1.0, "significant": False}
    mean = sum(diffs) / n
    if n >= 2:
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    else:
        var = 0.0
    sd = math.sqrt(var)
    se = sd / math.sqrt(n) if n > 0 else 0.0
    if se > 0:
        t = mean / se
        p_t = _t_sf(t, n - 1)
    else:
        # 분산 0: 전원 동일 방향이면 유의, 전원 0이면 무효
        t = float("inf") if mean > 0 else (float("-inf") if mean < 0 else 0.0)
        p_t = 0.0 if mean > 0 else 1.0
    # 부호검정 (0 제외, 정규근사 이항)
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    n_eff = n_pos + n_neg
    if n_eff == 0:
        p_sign = 1.0
    else:
        z = (n_pos - n_eff / 2.0) / math.sqrt(n_eff / 4.0)
        p_sign = _normal_sf(z)
    significant = (mean > 0) and (p_t < ALPHA)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t,
            "p_t": p_t, "n_pos": n_pos, "n_neg": n_neg, "p_sign": p_sign,
            "significant": significant}


# =========================================================
# 채점 로직 (순수 — off/on 집계 dict 입력)
# =========================================================
def _common_agents(off: dict, on: dict) -> list[str]:
    return sorted(set(off) & set(on))


def _diffs(off: dict, on: dict, aids: list[str], metric: str) -> list[float]:
    return [float(on[a].get(metric, 0) or 0) - float(off[a].get(metric, 0) or 0) for a in aids]


def score(off_agents: dict, on_agents: dict, on_meta: dict | None = None) -> dict:
    """off/on 에이전트 집계 dict → 채점 결과 dict."""
    aids = _common_agents(off_agents, on_agents)
    out: dict = {"n_common_agents": len(aids)}

    # ── C2: 적립(=사실상 총소비) 쌍체검정 ──
    elig_diffs = _diffs(off_agents, on_agents, aids, "eligible_spent")
    c2 = paired_test(elig_diffs)
    out["C2"] = c2
    out["C2_pass"] = bool(c2["significant"])

    # ── C1: 적립 vs 제외 3-arm ──
    # 게이트는 종전대로 excluded_luxury 하나로 판정한다(호환). vice·nonconsumption 은
    # 보조 지표 — 세 arm 은 '캐시백에 반응할 수 있는 정도'가 달라 함께 보면 해석이 는다:
    #   luxury(명품)·vice(유흥·사행)  : 돈이 생기면 갈 수도 있음 → 반응 가능
    #   nonconsumption(부동산·법무·세무): 캐시백 때문에 세무사를 찾지 않음 → 순수 위약
    mean_elig = c2["mean"]

    def _arm(metric: str) -> dict:
        d = _diffs(off_agents, on_agents, aids, metric)
        t = paired_test(d)
        g = paired_test([e - x for e, x in zip(elig_diffs, d)])
        r = (t["mean"] / mean_elig) if mean_elig > 0 else (float("inf") if t["mean"] > 0 else 0.0)
        return {
            **t,
            "gap_eligible_minus_excluded": g,
            "ratio_excluded_over_eligible": r,
            "alarm": bool(t["significant"] and mean_elig > 0 and r >= C1_ALARM_RATIO),
        }

    arms = {
        "excluded_luxury": _arm("excluded_luxury_spent"),
        "excluded_vice": _arm("excluded_vice_spent"),
        "excluded_nonconsumption": _arm("excluded_nonconsumption_spent"),
    }
    lux = arms["excluded_luxury"]
    gap = lux["gap_eligible_minus_excluded"]
    out["C1"] = {
        "eligible": {"mean": mean_elig, "p_t": c2["p_t"]},
        "arms": arms,
        # 하위호환 (게이트 arm 을 최상위에도 유지)
        "excluded_luxury": lux,
        "gap_eligible_minus_excluded": gap,
        "ratio_excluded_over_eligible": lux["ratio_excluded_over_eligible"],
        "alarm": lux["alarm"],
        # 위약군이 반응하면 명품보다 강한 경고 — 게이트는 아니고 해석 플래그
        "placebo_alarm": arms["excluded_nonconsumption"]["alarm"],
        "alarm_arms": [k for k, v in arms.items() if v["alarm"]],
    }
    # C1 통과 = 적립이 제외(명품)보다 유의하게 큼 그리고 경보 아님
    out["C1_pass"] = bool(gap["significant"] and not lux["alarm"])

    # ── D1: KDI 8분류 B−A 순위 (집계 총합) ──
    kdi_off: dict[str, float] = {}
    kdi_on: dict[str, float] = {}
    for a in aids:
        for k, v in (off_agents[a].get("by_kdi") or {}).items():
            kdi_off[k] = kdi_off.get(k, 0) + v
        for k, v in (on_agents[a].get("by_kdi") or {}).items():
            kdi_on[k] = kdi_on.get(k, 0) + v
    kdi_keys = set(kdi_off) | set(kdi_on)
    kdi_diff = {k: kdi_on.get(k, 0) - kdi_off.get(k, 0) for k in kdi_keys}
    kdi_ranked = sorted(kdi_diff.items(), key=lambda kv: kv[1], reverse=True)
    out["D1"] = {"ranked": kdi_ranked}
    # 탐색적 확인: 가전·가구가 상위 절반, 학원·이·미용이 하위 절반?
    order = [k for k, _ in kdi_ranked]
    n_k = len(order)
    top_half = set(order[: max(1, n_k // 2)])
    out["D1_note"] = {
        "가전·가구_상위군": ("가전·가구" in top_half) if "가전·가구" in kdi_diff else None,
        "학원_하위군": ("학원" not in top_half) if "학원" in kdi_diff else None,
        "이·미용_하위군": ("이·미용" not in top_half) if "이·미용" in kdi_diff else None,
    }

    # ── D2: 소비분위별 적립 B−A 평균 (탐색) ──
    by_decile: dict[int, list[float]] = {}
    for a, d in zip(aids, elig_diffs):
        dec = off_agents[a].get("spend_decile") or on_agents[a].get("spend_decile")
        if dec is None:
            continue
        by_decile.setdefault(int(dec), []).append(d)
    out["D2"] = {str(k): {"n": len(v), "mean_eligible_diff": (sum(v) / len(v) if v else 0.0)}
                 for k, v in sorted(by_decile.items())}

    # ── 반응 로그 (§4.5 ③): ON 런 trigger='policy' 비율 ──
    tot_events = sum(int(on_agents[a].get("n_events", 0) or 0) for a in aids)
    tot_trig = sum(int(on_agents[a].get("n_policy_trigger", 0) or 0) for a in aids)
    out["policy_trigger"] = {
        "n_events": tot_events,
        "n_policy_trigger": tot_trig,
        "ratio": (tot_trig / tot_events) if tot_events else 0.0,
    }

    # ── 종합 판정 (§11 성공 기준) ──
    if out["C2_pass"] and out["C1_pass"]:
        verdict = "PASS"          # 1차 성공: 총소비 방향+유의 + 적립≫제외
    elif out["C2_pass"] and not out["C1_pass"]:
        if out["C1"]["alarm"]:
            verdict = "PASS_C2_ONLY_ALARM"   # 총소비는 늘었으나 제외도 같이 튐 → '일반 반응성'
        else:
            verdict = "PASS_C2_ONLY"         # 총소비 늘고 제외 경보는 없으나 적립≫제외 유의성 부족
    elif (out["C2"]["mean"] > 0) and not out["C2_pass"]:
        verdict = "DIRECTION_ONLY"           # 방향은 + 이나 노이즈와 구분 불가 (§5.4)
    else:
        verdict = "NULL"                     # 무반응/역방향 (§8-7, 유효한 결과)
    out["verdict"] = verdict
    return out


# =========================================================
# 리포트 렌더
# =========================================================
_VERDICT_KO = {
    "PASS": "1차 성공 — 총소비 방향+유의(C2) 및 적립≫제외(C1) 확인",
    "PASS_C2_ONLY": "부분 성공 — C2 통과, 단 적립≫제외 유의성 부족(경보는 없음)",
    "PASS_C2_ONLY_ALARM": "주의 — C2는 통과했으나 제외도 함께 튐(일반 반응성 의심, 신뢰도↓)",
    "DIRECTION_ONLY": "판정 보류 — 방향은 +이나 노이즈와 구분 불가(§5.4)",
    "NULL": "무반응/역방향 — 유효한 (부정적) 결과(§8-7)",
}


def render_markdown(res: dict, off_meta: dict, on_meta: dict) -> str:
    L: list[str] = []
    L.append("# 상생소비지원금 백테스트 채점 리포트 (SANGSAENG_BACKTEST)")
    L.append("")
    L.append(f"- 생성: {datetime.now(timezone.utc).isoformat()}")
    L.append(f"- OFF(A런): {off_meta.get('start')}~{off_meta.get('end')} · ON(B런): {on_meta.get('start')}~{on_meta.get('end')}")
    L.append(f"- 공통 에이전트: {res['n_common_agents']:,}명 · 유의수준 α={ALPHA}")
    L.append("")
    L.append(f"## 종합 판정: **{res['verdict']}**")
    L.append(f"> {_VERDICT_KO.get(res['verdict'], res['verdict'])}")
    L.append("")

    c2 = res["C2"]
    L.append("## C2 (필수) — 적립=사실상 총소비 B−A 쌍체검정")
    L.append(f"- 평균 B−A = **{c2['mean']:,.0f}원** / n={c2['n']:,} / t={c2['t']:.2f} / "
             f"p(단측 t)={c2['p_t']:.4g} / 부호검정 p={c2['p_sign']:.4g} (+{c2['n_pos']}/−{c2['n_neg']})")
    L.append(f"- 판정: {'✅ 통과 (평균>0 이며 p<α)' if res['C2_pass'] else '❌ 미통과'}")
    L.append("- 앵커(크기 대조 안 함, 부호·방향만): 1인 가구 +11.16% / 서울 +14.57% / 전국 +11.25%")
    L.append("")

    c1 = res["C1"]
    L.append("## C1 (C2 해석 보증) — 적립 ≫ 제외")
    L.append(f"- 적립 평균 B−A = {c1['eligible']['mean']:,.0f}원")
    L.append("")
    L.append("| 제외 arm | 성격 | 평균 B−A | p | 제외/적립 | (적립−제외) p | 경보 |")
    L.append("|---|---|---:|---:|---:|---:|:--:|")
    for key, label, nature in (
        ("excluded_luxury", "명품(시계·귀금속)", "반응 가능"),
        ("excluded_vice", "유흥·사행", "반응 가능"),
        ("excluded_nonconsumption", "비소비(부동산·법무·세무)", "**반응 원천 차단**"),
    ):
        a = (c1.get("arms") or {}).get(key)
        if not a:
            continue
        star = " ★게이트" if key == "excluded_luxury" else ""
        L.append(f"| {label}{star} | {nature} | {a['mean']:,.0f}원 | {a['p_t']:.4g} | "
                 f"{a['ratio_excluded_over_eligible']:.3f} | "
                 f"{a['gap_eligible_minus_excluded']['p_t']:.4g} | "
                 f"{'⚠️' if a['alarm'] else '—'} |")
    L.append("")
    L.append(f"- 판정(게이트=명품 arm): "
             f"{'✅ 적립이 제외보다 유의하게 큼' if res['C1_pass'] else '❌ 부등식 미확인'}")
    if c1.get("placebo_alarm"):
        L.append("- ⚠️ **위약군(비소비) 경보** — 캐시백에 반응할 수 없는 업종까지 늘었다면 "
                 "정책 반응이 아니라 런 간 일반 변동일 가능성이 크다. 명품 경보보다 강한 신호.")
    elif c1.get("alarm_arms"):
        L.append(f"- 경보 arm: {', '.join(c1['alarm_arms'])} (위약군은 정상)")
    L.append("- 정답(표4-7): 적립 총결제 +20.82% vs 제외 +2.85~5.35% (적립이 4~7배)")
    L.append("")

    L.append("## D1 (보조·탐색) — KDI 8분류 B−A 순위")
    for i, (k, v) in enumerate(res["D1"]["ranked"], 1):
        L.append(f"  {i}. {k}: {v:,.0f}원")
    note = res["D1_note"]
    L.append(f"- 관찰: 가전·가구 상위군={note['가전·가구_상위군']} / 학원 하위군={note['학원_하위군']} / "
             f"이·미용 하위군={note['이·미용_하위군']} (정답: 가전·가구 최대 +36.23%)")
    L.append("")

    L.append("## D2 (탐색) — 소비분위별 적립 B−A 평균")
    for dec, v in res["D2"].items():
        L.append(f"  - {dec}분위: n={v['n']:,} · 평균 {v['mean_eligible_diff']:,.0f}원")
    L.append("")

    pt = res["policy_trigger"]
    L.append("## 반응 로그 (§4.5 ③, 진단) — ON 런 trigger='policy' 비율")
    L.append(f"- {pt['n_policy_trigger']:,} / {pt['n_events']:,} 이벤트 = {pt['ratio']*100:.2f}%")
    L.append("")
    L.append("## 한계 (§8 요약)")
    L.append("- 크기 대조 안 함(시간지평·기대효과 한정).")
    L.append("- 제외군은 정책 제외 리스트의 **부분집합**이다 — 대형마트·백화점·면세점·온라인몰·"
             "신차·실외골프장은 모집단(소상공인 상가 DB)에 원천 부재. 성인업소도 포착 불가.")
    L.append("- 비소비 arm 은 정책의 '비소비성 지출'(보험료·세금 등 거래유형)과 개념이 다르다. "
             "'반응할 수 없는 업종'이라는 통계적 위약군으로 해석할 것.")
    L.append("- LLM 샘플링 비결정성 잔존(§5.4로 완화). 무반응/방향만+ 도 각각 유효 결과로 구분 기록.")
    return "\n".join(L)


# =========================================================
# 로드 / 메인
# =========================================================
def _load(path: Path) -> tuple[dict, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("agents", {}), raw.get("meta", {})


def _selftest() -> None:
    # 시나리오1: ON에서 적립만 크게↑, 제외(명품) 거의 그대로 → C2 pass, C1 pass, 경보 없음
    def mk(elig, lux, kdi=None, dec=5, vice=None, noncon=None):
        vice = lux // 4 if vice is None else vice
        noncon = lux // 4 if noncon is None else noncon
        return {"eligible_spent": elig, "excluded_luxury_spent": lux,
                "excluded_vice_spent": vice, "excluded_nonconsumption_spent": noncon,
                "excluded_other_spent": 0,
                "total_spent": elig + lux + vice + noncon, "by_kdi": kdi or {}, "n_events": 3,
                "n_policy_trigger": 1, "spend_decile": dec}
    off = {f"A{i}": mk(100000, 20000, {"요식": 60000, "가전·가구": 40000}) for i in range(200)}
    on = {f"A{i}": mk(130000 + (i % 5) * 1000, 21000 + (i % 3) * 200,
                      {"요식": 70000, "가전·가구": 60000}) for i in range(200)}
    r1 = score(off, on)
    assert r1["C2_pass"], r1["C2"]
    assert r1["C1_pass"], r1["C1"]
    assert not r1["C1"]["alarm"], r1["C1"]
    assert r1["verdict"] == "PASS", r1["verdict"]
    # D1: 가전·가구 diff(20000/agent) > 요식 diff(10000/agent) → 가전·가구 상위
    assert r1["D1"]["ranked"][0][0] == "가전·가구", r1["D1"]["ranked"]
    assert set(r1["C1"]["arms"]) == {"excluded_luxury", "excluded_vice",
                                     "excluded_nonconsumption"}, r1["C1"]["arms"]
    assert not r1["C1"]["placebo_alarm"], r1["C1"]
    print("  ✔ 시나리오1: C2 pass / C1 pass / 경보없음 / verdict=PASS / D1 가전·가구 상위")

    # 시나리오2: 적립도↑ 제외(명품)도 비슷하게↑ → 경보
    on2 = {f"A{i}": mk(130000, 50000, {"요식": 70000, "가전·가구": 60000}) for i in range(200)}
    r2 = score(off, on2)
    assert r2["C2_pass"], r2["C2"]
    assert r2["C1"]["alarm"], r2["C1"]
    assert r2["verdict"] == "PASS_C2_ONLY_ALARM", r2["verdict"]
    print("  ✔ 시나리오2: 제외도 함께 튐 → 경보 / verdict=PASS_C2_ONLY_ALARM")

    # 시나리오3: 변화 없음(노이즈만, 평균≈0) → DIRECTION_ONLY 또는 NULL
    import random
    rng = random.Random(0)
    on3 = {f"A{i}": mk(100000 + rng.randint(-5000, 5000), 20000 + rng.randint(-2000, 2000)) for i in range(200)}
    r3 = score(off, on3)
    assert not r3["C2_pass"], r3["C2"]
    assert r3["verdict"] in ("DIRECTION_ONLY", "NULL"), r3["verdict"]
    print(f"  ✔ 시나리오3: 노이즈만 → C2 미통과 / verdict={r3['verdict']}")

    # 시나리오4: 위약군(비소비)까지 같이 튐 → placebo_alarm
    #   캐시백에 반응할 수 없는 업종이 늘었다면 정책 반응이 아니라 런 간 일반 변동이다.
    #   off 의 비소비는 5,000 → on 25,000 (diff 20,000 = 적립 diff 30,000의 0.67 ≥ 임계 0.5)
    on4 = {f"A{i}": mk(130000, 21000, {"요식": 70000, "가전·가구": 60000},
                       noncon=25000) for i in range(200)}
    r4 = score(off, on4)
    assert r4["C1"]["placebo_alarm"], r4["C1"]["arms"]["excluded_nonconsumption"]
    assert "excluded_nonconsumption" in r4["C1"]["alarm_arms"], r4["C1"]["alarm_arms"]
    assert not r4["C1"]["alarm"], "게이트(명품) arm 은 경보 아님 — 게이트 불변 확인"
    print("  ✔ 시나리오4: 위약군만 튐 → placebo_alarm / 게이트(명품) 판정은 불변")

    # 하위호환: 구 포맷(vice/nonconsumption 키 없음)도 그대로 채점
    old_off = {f"A{i}": {"eligible_spent": 100000, "excluded_luxury_spent": 20000,
                         "by_kdi": {}, "n_events": 1, "n_policy_trigger": 0,
                         "spend_decile": 5} for i in range(200)}
    old_on = {f"A{i}": {"eligible_spent": 130000, "excluded_luxury_spent": 21000,
                        "by_kdi": {}, "n_events": 1, "n_policy_trigger": 0,
                        "spend_decile": 5} for i in range(200)}
    r5 = score(old_off, old_on)
    assert r5["C2_pass"] and r5["C1_pass"], r5["verdict"]
    assert r5["C1"]["arms"]["excluded_vice"]["mean"] == 0.0
    print("  ✔ 하위호환: 구 집계 포맷(3-arm 키 없음)도 정상 채점")

    # 렌더 스모크
    md = render_markdown(r1, {"start": "a", "end": "b"}, {"start": "c", "end": "d"})
    assert "SANGSAENG_BACKTEST" in md and "C2" in md
    assert "유흥·사행" in md and "비소비" in md, "C1 3-arm 표가 렌더돼야 한다"
    md4 = render_markdown(r4, {"start": "a", "end": "b"}, {"start": "c", "end": "d"})
    assert "위약군(비소비) 경보" in md4
    print("  ✔ 리포트 렌더 정상 (C1 3-arm 표 · 위약 경보 문구)")
    print("validate_sangsaeng SELFTEST ALL OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", type=Path, help="OFF(A런) 집계 JSON")
    ap.add_argument("--on", type=Path, help="ON(B런) 집계 JSON")
    ap.add_argument("--out-dir", type=Path, default=Path("output/sim/report"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.off or not args.on:
        ap.error("--off 와 --on 은 필수 (또는 --selftest)")

    off_agents, off_meta = _load(args.off)
    on_agents, on_meta = _load(args.on)
    res = score(off_agents, on_agents, on_meta)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(res, off_meta, on_meta)
    (args.out_dir / "SANGSAENG_BACKTEST.md").write_text(md, encoding="utf-8")
    (args.out_dir / "SANGSAENG_BACKTEST.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    print(f"\n→ {args.out_dir/'SANGSAENG_BACKTEST.md'} / .json 저장")


if __name__ == "__main__":
    main()
