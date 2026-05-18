"""1일치 풀런 결과 빠른 점검 (baseline · sanity check 용).

다음 항목을 단일 markdown으로 정리:
  1. 기본 통계 — 성공률·err·시간·토큰
  2. Fallback 카운터 (fb_*)
  3. trigger 분포 (reasoning 적재 검증)
  4. reasoning / pick_reason 적재율
  5. Plan / INCLUDES / 외출 비율
  6. Night Phase 1 — visited Memory + KNOWS_POI 갱신
  7. Night Phase 2 — Conversation by intent + rumor Memory
  8. 평균 만족도 (전체 / trigger별 / 카테고리별)
  9. 환각 검출 (POI 노드 유효성)
  10. 샘플 reasoning 5건 (LLM 출력 품질 육안 점검)

CLI:
  python scripts/sim/day_health_check.py --day 2026-05-01 \\
      --out docs/DAY_HEALTH_2026-05-01.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))

from _common import driver_session  # noqa: E402

METRICS_DIR = Path(os.environ.get("SIM_OUTPUT_DIR",
                                  os.path.expanduser("~/sim_output"))) / "metrics"


def load_metrics(day: str) -> list[dict]:
    p = METRICS_DIR / f"day_{day}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2026-05-01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    day = args.day
    rows = load_metrics(day)
    ok = [r for r in rows if r.get("status") == "ok"]
    err = [r for r in rows if r.get("status") != "ok"]
    total = len(rows)
    out_path = Path(args.out or f"docs/DAY_HEALTH_{day}.md").resolve()

    L: list[str] = []
    L.append(f"# 1일치 풀런 점검 — {day}")
    L.append("")
    L.append(f"**파일**: `{METRICS_DIR}/day_{day}.jsonl`")
    L.append("")

    # 1. 기본 통계
    if not rows:
        L.append("⚠️ **metrics jsonl 없음 — 시뮬 미실행 또는 아직 진행 중**")
    else:
        succ = len(ok) / max(total, 1) * 100
        L.append("## 1. 기본 통계")
        L.append("")
        L.append("| 항목 | 값 |")
        L.append("|---|---:|")
        L.append(f"| 총 처리 agent | {total:,} |")
        L.append(f"| 성공 | {len(ok):,} ({succ:.2f}%) |")
        L.append(f"| 실패 | {len(err):,} ({(100-succ):.2f}%) |")
        if ok:
            tot_in = sum(r.get("tokens_in", 0) for r in ok)
            tot_out = sum(r.get("tokens_out", 0) for r in ok)
            tot_el = sum(r.get("elapsed", 0) for r in ok)
            L.append(f"| 평균 elapsed/agent | {tot_el/len(ok):.1f}s |")
            L.append(f"| 총 tokens_in | {tot_in:,} ({tot_in/len(ok):,.0f}/agent) |")
            L.append(f"| 총 tokens_out | {tot_out:,} ({tot_out/len(ok):,.0f}/agent) |")
            sats = [r["avg_sat"] for r in ok if r.get("avg_sat") is not None]
            if sats:
                L.append(f"| 평균 만족도 | {sum(sats)/len(sats):.3f} |")
            s1r = sum(max(0, r.get("s1_attempts", 1)-1) for r in ok)
            s2r = sum(max(0, r.get("s2_attempts", 0)-1) for r in ok)
            L.append(f"| Stage 1 retry | {s1r:,} ({s1r/len(ok)*100:.1f}%) |")
            L.append(f"| Stage 2 retry | {s2r:,} |")
        L.append("")

    # 2. Fallback (fb_*)
    if ok:
        fb_fields = ["fb_resolve_dong", "fb_cand_sub_match", "fb_cand_l1_dong",
                     "fb_cand_l1_district", "fb_cand_all_empty",
                     "fb_hallucinations_corrected", "fb_hallucinations_dropped",
                     "fb_order_mismatch", "fb_missing_picks_filled"]
        fb = {k: sum(r.get(k, 0) for r in ok) for k in fb_fields}
        n_inc = sum(r.get("n_includes", 0) for r in ok)
        cand_calls = (fb["fb_cand_sub_match"] + fb["fb_cand_l1_dong"]
                      + fb["fb_cand_l1_district"] + fb["fb_cand_all_empty"])
        L.append("## 2. Stage 2 Fallback")
        L.append("")
        L.append(f"외출 이벤트 (cand_calls): **{cand_calls:,}** / INCLUDES total **{n_inc:,}**")
        L.append("")
        L.append("| 단계 | 건수 | 비율 |")
        L.append("|---|---:|---:|")
        for k, label in [
            ("fb_cand_sub_match", "1차 sub 매칭"),
            ("fb_cand_l1_dong", "2차 L1 dong fallback"),
            ("fb_cand_l1_district", "3차 L1 district"),
            ("fb_cand_all_empty", "4차 모두 비어 있음 (드롭)"),
        ]:
            pct = fb[k] / max(cand_calls, 1) * 100
            L.append(f"| {label} | {fb[k]:,} | {pct:.2f}% |")
        L.append("")
        L.append(f"- 환각 자동 보정 (random Top-5): **{fb['fb_hallucinations_corrected']:,}** "
                 f"({fb['fb_hallucinations_corrected']/max(n_inc,1)*1000:.2f}‰)")
        L.append(f"- 환각 드롭 (후보 자체 없음): **{fb['fb_hallucinations_dropped']:,}** "
                 f"({fb['fb_hallucinations_dropped']/max(n_inc,1)*1000:.2f}‰)")
        L.append(f"- order_mismatch (다른 order POI): **{fb['fb_order_mismatch']:,}** "
                 f"({fb['fb_order_mismatch']/max(cand_calls,1)*100:.2f}% of cand_calls)")
        L.append(f"- missing_picks_filled: **{fb['fb_missing_picks_filled']:,}** "
                 f"({fb['fb_missing_picks_filled']/max(n_inc,1)*1000:.2f}‰)")
        L.append(f"- resolve_dong placeholder: **{fb['fb_resolve_dong']:,}** "
                 f"(0이어야 정상)")
        L.append("")

    # 3·4·5·6·7·8·9·10 — Neo4j 조회
    with driver_session() as s:
        # 3. trigger 분포 + reasoning 적재율
        L.append("## 3. trigger 분포 + reasoning 적재율 (Stage 1)")
        L.append("")
        r = s.run("""
            MATCH (a:Agent)-[:HAS_PLAN {day: date($d)}]->(:Plan)-[i:INCLUDES]->()
            WHERE NOT i.category IN ['집','직장']
            RETURN
              count(*) AS total,
              sum(CASE WHEN i.reasoning IS NOT NULL THEN 1 ELSE 0 END) AS has_r,
              sum(CASE WHEN i.trigger IS NOT NULL THEN 1 ELSE 0 END) AS has_t,
              sum(CASE WHEN i.pick_reason IS NOT NULL THEN 1 ELSE 0 END) AS has_p
        """, d=day).single()
        if r and r["total"] > 0:
            L.append(f"외출 INCLUDES: **{r['total']:,}**")
            L.append("")
            L.append(f"- reasoning 적재율: **{r['has_r']/r['total']*100:.1f}%** ({r['has_r']:,})")
            L.append(f"- trigger 적재율: **{r['has_t']/r['total']*100:.1f}%** ({r['has_t']:,})")
            L.append(f"- pick_reason 적재율: **{r['has_p']/r['total']*100:.1f}%** ({r['has_p']:,})")
            L.append("")
            L.append("**trigger 분포**:")
            L.append("")
            L.append("| trigger | 건수 | 비율 |")
            L.append("|---|---:|---:|")
            for x in s.run("""
                MATCH (:Plan {day: date($d)})-[i:INCLUDES]->()
                WHERE i.trigger IS NOT NULL AND NOT i.category IN ['집','직장']
                RETURN i.trigger AS t, count(*) AS n ORDER BY n DESC
            """, d=day).data():
                pct = x["n"] / r["total"] * 100
                L.append(f"| {x['t']} | {x['n']:,} | {pct:.2f}% |")
            L.append("")

        # 5. Plan / INCLUDES / 외출 비율
        r2 = s.run("""
            MATCH (:Agent)-[:HAS_PLAN {day: date($d)}]->(:Plan)-[i:INCLUDES]->()
            RETURN count(DISTINCT i) AS n_inc,
                   sum(CASE WHEN i.category IN ['집','직장'] THEN 1 ELSE 0 END) AS internal,
                   sum(CASE WHEN NOT i.category IN ['집','직장'] THEN 1 ELSE 0 END) AS outing
        """, d=day).single()
        n_plan = s.run("""
            MATCH (a:Agent)-[:HAS_PLAN {day: date($d)}]->(p:Plan) RETURN count(p) AS n
        """, d=day).single()["n"]
        L.append("## 4. Plan / INCLUDES / 외출 비율")
        L.append("")
        L.append(f"- Plan 노드: **{n_plan:,}**")
        L.append(f"- INCLUDES 엣지: **{r2['n_inc']:,}**")
        L.append(f"- 외출 이벤트: **{r2['outing']:,}** "
                 f"({r2['outing']/max(r2['n_inc'],1)*100:.1f}% of INCLUDES)")
        L.append(f"- 내부 (집/직장): {r2['internal']:,}")
        L.append("")

        # 9. 환각
        h = s.run("""
            MATCH (:Agent)-[:HAS_PLAN {day: date($d)}]->(:Plan)-[i:INCLUDES]->(p)
            RETURN count(p) AS n_inc, count(CASE WHEN p:POI THEN 1 END) AS n_poi
        """, d=day).single()
        L.append("## 5. 환각 검출 (모든 poi_id가 :POI 노드인지)")
        L.append("")
        L.append(f"- INCLUDES: {h['n_inc']:,} / valid POI: {h['n_poi']:,} → "
                 f"환각 **{h['n_inc']-h['n_poi']}건** "
                 f"({'✅ 0건' if h['n_inc']==h['n_poi'] else '⚠️ 비정상'})")
        L.append("")

        # 8. trigger별 평균 만족도
        L.append("## 6. trigger별 평균 만족도")
        L.append("")
        L.append("| trigger | 평균 만족도 | n |")
        L.append("|---|---:|---:|")
        for x in s.run("""
            MATCH (:Plan {day: date($d)})-[i:INCLUDES]->()
            WHERE i.actual_satisfaction IS NOT NULL AND i.trigger IS NOT NULL
              AND NOT i.category IN ['집','직장']
            RETURN i.trigger AS t, avg(i.actual_satisfaction) AS s, count(*) AS n
            ORDER BY s DESC
        """, d=day).data():
            L.append(f"| {x['t']} | {x['s']:.3f} | {x['n']:,} |")
        L.append("")

        # 6. Night Phase 1 - visited Memory + KNOWS_POI
        # visited Memory는 다음 날 새벽에 적재됨. day d의 visited는 (d+1) 또는 그 이상
        n_vis = s.run("""
            MATCH (m:Memory {type:'visited', day: date($d)}) RETURN count(m) AS n
        """, d=day).single()["n"]
        n_kp = s.run("""
            MATCH ()-[kp:KNOWS_POI]->() WHERE kp.last_visit = date($d) RETURN count(kp) AS n
        """, d=day).single()["n"]
        L.append("## 7. Night Phase 1 — visited Memory + KNOWS_POI")
        L.append("")
        L.append(f"- visited Memory (day={day}): **{n_vis:,}** "
                 f"{'(다음 날 새벽에 적재 — 0이면 아직 미처리)' if n_vis == 0 else ''}")
        L.append(f"- KNOWS_POI 갱신 (last_visit={day}): **{n_kp:,}**")
        L.append("")

        # 7. Night Phase 2 - Conversation by intent
        L.append("## 8. Night Phase 2 — Conversation 분포")
        L.append("")
        L.append("| intent | n |")
        L.append("|---|---:|")
        total_conv = 0
        for x in s.run("""
            MATCH (c:Conversation {day: date($d)})
            RETURN c.intent AS i, count(*) AS n ORDER BY i
        """, d=day).data():
            L.append(f"| {x['i']} | {x['n']:,} |")
            total_conv += x["n"]
        L.append(f"| **합계** | **{total_conv:,}** |")
        L.append("")
        rumor_n = s.run("""
            MATCH (m:Memory {type:'rumor', day: date($d)}) RETURN count(m) AS n
        """, d=day).single()["n"]
        L.append(f"- rumor Memory 적재: **{rumor_n:,}** (추천+이슈 합과 일치해야 정상)")
        L.append("")

        # 10. 샘플 reasoning 5건
        L.append("## 9. 샘플 reasoning (LLM 출력 품질 육안 점검)")
        L.append("")
        L.append("**Stage 1 reasoning (5건):**")
        L.append("")
        for x in s.run("""
            MATCH (a:Agent)-[:HAS_PLAN {day: date($d)}]->(:Plan)-[i:INCLUDES]->(poi:POI)
            WHERE i.reasoning IS NOT NULL AND NOT i.category IN ['집','직장']
            RETURN a.id AS aid, i.time AS time, i.category AS cat,
                   i.trigger AS trigger, i.reasoning AS reasoning,
                   poi.name AS poi_name
            ORDER BY rand() LIMIT 5
        """, d=day).data():
            L.append(f"- `{x['aid']}` @ {str(x['time'])[:5]} | "
                     f"**{x['cat']}** → {x['poi_name']} | trigger=`{x['trigger']}`")
            L.append(f"  > {x['reasoning']}")
            L.append("")

        L.append("**Stage 2 pick_reason (3건):**")
        L.append("")
        for x in s.run("""
            MATCH (a:Agent)-[:HAS_PLAN {day: date($d)}]->(:Plan)-[i:INCLUDES]->(poi:POI)
            WHERE i.pick_reason IS NOT NULL
            RETURN a.id AS aid, i.category AS cat, i.pick_factor AS factor,
                   i.pick_reason AS reason, poi.name AS poi_name
            ORDER BY rand() LIMIT 3
        """, d=day).data():
            L.append(f"- `{x['aid']}` | **{x['cat']}** → {x['poi_name']} | "
                     f"factor=`{x['factor']}`")
            L.append(f"  > {x['reason']}")
            L.append("")

        L.append("**Night Phase 2 Conversation reasoning (3건):**")
        L.append("")
        for x in s.run("""
            MATCH (c:Conversation {day: date($d)})
            WHERE c.reasoning IS NOT NULL
            RETURN c.intent AS i, c.initiator_id AS a, c.recipient_id AS b,
                   c.reasoning AS r ORDER BY rand() LIMIT 3
        """, d=day).data():
            L.append(f"- [{x['i']}] `{x['a']}` ↔ `{x['b']}`")
            L.append(f"  > {x['r']}")
            L.append("")

    # 결론
    L.append("---")
    L.append("")
    L.append("## 진단 결과 요약")
    L.append("")
    if not rows:
        L.append("⚠️ 데이터 없음 — 시뮬 결과 미적재")
    else:
        flags = []
        if ok and len(ok) / max(total, 1) >= 0.99:
            flags.append("✅ 성공률 99%+ 정상")
        else:
            flags.append(f"⚠️ 성공률 {len(ok)/max(total,1)*100:.1f}%")
        if h["n_inc"] == h["n_poi"]:
            flags.append("✅ 환각 0건")
        else:
            flags.append(f"⚠️ 환각 {h['n_inc']-h['n_poi']:,}건")
        for f in flags:
            L.append(f"- {f}")
    L.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"→ {out_path}", file=sys.stderr)
    print("\n".join(L))


if __name__ == "__main__":
    main()
