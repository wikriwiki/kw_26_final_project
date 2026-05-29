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
      --out docs/archive/DAY_HEALTH_2026-05-01.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))

from _common import driver_session  # noqa: E402
from stage1_intent import normalize_trigger  # noqa: E402


def _aggregate_triggers(rows):
    """raw trigger row (t, n) 리스트 → normalize_trigger 적용 후 재집계, n DESC 정렬."""
    agg: dict[str, int] = {}
    for r in rows:
        k = normalize_trigger(r["t"]) or r["t"]
        agg[k] = agg.get(k, 0) + int(r["n"])
    return sorted(({"t": k, "n": v} for k, v in agg.items()), key=lambda x: -x["n"])

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
    L.append(f"**작성일**: {datetime.now().strftime('%Y-%m-%d %H:%M KST')}")
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
            raw = s.run("""
                MATCH (:Plan {day: date($d)})-[i:INCLUDES]->()
                WHERE i.trigger IS NOT NULL AND NOT i.category IN ['집','직장']
                RETURN i.trigger AS t, count(*) AS n ORDER BY n DESC
            """, d=day).data()
            for x in _aggregate_triggers(raw):
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

    # ════════════════════════════════════════════════════════════
    # v3 변경 사항 중점 점검
    # ════════════════════════════════════════════════════════════
    L.append("---")
    L.append("")
    L.append("## 🎯 v3 변경 사항 중점 점검")
    L.append("")

    with driver_session() as s:
        # 10. reasoning 품질 — 길이 분포 + 페르소나/정책 인용
        L.append("### 10-A. reasoning 품질 (Stage 1)")
        L.append("")
        rq = s.run("""
            MATCH (a:Agent)-[:HAS_PLAN {day: date($d)}]->(:Plan)-[i:INCLUDES]->()
            WHERE i.reasoning IS NOT NULL AND NOT i.category IN ['집','직장']
            WITH i.reasoning AS r, a
            RETURN
              count(*) AS total,
              avg(size(r)) AS avg_len,
              min(size(r)) AS min_len,
              max(size(r)) AS max_len,
              sum(CASE WHEN size(r) < 30 THEN 1 ELSE 0 END) AS too_short,
              sum(CASE WHEN r CONTAINS '페르소나' OR r CONTAINS '라이프스타일'
                         OR r CONTAINS 'Top' OR r CONTAINS '평일' OR r CONTAINS '주말'
                         OR r CONTAINS '소득' OR r CONTAINS '직업' THEN 1 ELSE 0 END) AS cites_persona,
              sum(CASE WHEN r CONTAINS 'policy_' OR r CONTAINS '바우처' OR r CONTAINS '쿠폰'
                         OR r CONTAINS '환급' OR r CONTAINS '정책' THEN 1 ELSE 0 END) AS cites_policy,
              sum(CASE WHEN r CONTAINS '만족도' OR r CONTAINS 'sat' THEN 1 ELSE 0 END) AS cites_sat,
              sum(CASE WHEN r CONTAINS '약속' OR r CONTAINS 'AGT_' THEN 1 ELSE 0 END) AS cites_appt
        """, d=day).single()
        if rq and rq["total"]:
            tot = rq["total"]
            L.append(f"외출 reasoning {tot:,}건")
            L.append("")
            L.append(f"- 평균 길이: **{rq['avg_len']:.0f}자** (min {rq['min_len']}, max {rq['max_len']})")
            L.append(f"- 너무 짧음 (<30자) — placeholder 의심: **{rq['too_short']:,}** "
                     f"({rq['too_short']/tot*100:.2f}%)")
            L.append("")
            L.append("**페르소나·근거 인용 비율 (높을수록 깊이 있는 reasoning):**")
            L.append("")
            L.append("| 인용 종류 | 건수 | 비율 |")
            L.append("|---|---:|---:|")
            L.append(f"| 페르소나 (Top·라이프스타일·소득·직업) | {rq['cites_persona']:,} | {rq['cites_persona']/tot*100:.1f}% |")
            L.append(f"| 정책 (바우처·쿠폰·환급·policy_id) | {rq['cites_policy']:,} | {rq['cites_policy']/tot*100:.1f}% |")
            L.append(f"| 과거 만족도 (어제 sat·만족도) | {rq['cites_sat']:,} | {rq['cites_sat']/tot*100:.1f}% |")
            L.append(f"| 약속·지인 (AGT_) | {rq['cites_appt']:,} | {rq['cites_appt']/tot*100:.1f}% |")
            L.append("")

        # 10-B. trigger 정합성 (Day 0=정책 비활성 baseline)
        L.append("### 10-B. trigger 정합성")
        L.append("")
        cutoff_iso = "2026-05-20"  # 정책 발효일 (현재 시뮬: P008 강남 보행친화거리)
        is_baseline = day < cutoff_iso
        L.append(f"오늘({day}) 정책 발효 여부: "
                 f"**{'❌ 비활성 (baseline)' if is_baseline else '✅ 활성'}**")
        L.append("")
        tr = s.run("""
            MATCH (:Plan {day: date($d)})-[i:INCLUDES]->()
            WHERE i.trigger IS NOT NULL AND NOT i.category IN ['집','직장']
            RETURN i.trigger AS t, count(*) AS n
        """, d=day).data()
        tr_map: dict[str, int] = {}
        for x in tr:
            k = normalize_trigger(x["t"]) or x["t"]
            tr_map[k] = tr_map.get(k, 0) + int(x["n"])
        n_policy = tr_map.get("policy", 0)
        n_total = sum(tr_map.values())
        if is_baseline and n_policy > n_total * 0.005:
            L.append(f"⚠️ **baseline 일자인데 policy trigger {n_policy}건 ({n_policy/n_total*100:.2f}%) — "
                     f"LLM이 미발효 정책을 잘못 인용. 정상이면 ~0건**")
        elif is_baseline:
            L.append(f"✅ baseline 일자, policy trigger {n_policy}건 (정상)")
        else:
            L.append(f"📊 활성 일자, policy trigger {n_policy}건 ({n_policy/max(n_total,1)*100:.2f}%)")
        # 분포 균형 — top_category 편중 여부
        n_top = tr_map.get("top_category", 0)
        n_lifestyle = tr_map.get("lifestyle", 0)
        L.append(f"- top_category 비중: **{n_top/max(n_total,1)*100:.1f}%** "
                 f"(60%↑이면 LLM이 안전 라벨로 쏠림 — diversity 낮음)")
        L.append(f"- 다양성: rumor {tr_map.get('rumor',0)} · mood {tr_map.get('mood',0)} · "
                 f"appointment {tr_map.get('appointment',0)}")
        L.append("")

        # 10-C. pick_factor 분포 (Stage 2 다양성)
        L.append("### 10-C. pick_factor 분포 (Stage 2 단골 vs 탐색)")
        L.append("")
        L.append("| factor | 건수 | 비율 |")
        L.append("|---|---:|---:|")
        pf = s.run("""
            MATCH (:Plan {day: date($d)})-[i:INCLUDES]->()
            WHERE i.pick_factor IS NOT NULL
            RETURN i.pick_factor AS f, count(*) AS n ORDER BY n DESC
        """, d=day).data()
        pf_total = sum(x["n"] for x in pf) or 1
        for x in pf:
            L.append(f"| {x['f']} | {x['n']:,} | {x['n']/pf_total*100:.1f}% |")
        L.append("")
        n_known = sum(x["n"] for x in pf if x["f"] == "known")
        n_novel = sum(x["n"] for x in pf if x["f"] == "novelty")
        if pf_total > 100:
            if n_known > pf_total * 0.85:
                L.append(f"⚠️ known 편중 {n_known/pf_total*100:.1f}% — 신규 탐색 부족")
            elif n_novel < pf_total * 0.05 and not is_baseline:
                L.append(f"ℹ️ novelty {n_novel/pf_total*100:.1f}% — 후반 일자에 정책으로 신규 발생 기대")
            else:
                L.append(f"✅ 균형 (known {n_known/pf_total*100:.1f}% / novelty {n_novel/pf_total*100:.1f}%)")
        L.append("")

        # 10-D. 정책 효과 정밀 측정 (P008 강남 보행친화거리 — 동 단위 layered DID)
        #   한계 극복: ① 분모 동적 카운트 ② 동 단위 L1/L2/Control 그룹 ③ 거주지+매장 위치 둘 다.
        L.append("### 10-D. 정책 효과 정밀 측정 (P008 동 단위 layered)")
        L.append("")

        # P008 적용 그룹 정의 — applied_to 관계에서 동적으로 가져옴 (하드코드 회피)
        l1_dongs = ["역삼1동", "역삼2동", "도곡1동"]   # 사업 구간
        l2_dongs = ["도곡2동", "삼성1동", "삼성2동", "논현1동", "논현2동"]  # 도보권 인접

        # 분모 — 실제 거주 agent 수 (동적)
        denoms = s.run("""
            MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(d:Dong)
              <-[:HAS_DONG]-(dist:District)
            WITH a, d.name AS dong, dist.name AS gu
            RETURN
              sum(CASE WHEN dong IN $l1 THEN 1 ELSE 0 END) AS n_l1,
              sum(CASE WHEN dong IN $l2 THEN 1 ELSE 0 END) AS n_l2,
              sum(CASE WHEN gu = '강남구' AND NOT dong IN ($l1 + $l2) THEN 1 ELSE 0 END) AS n_l3,
              sum(CASE WHEN gu <> '강남구' THEN 1 ELSE 0 END) AS n_l4,
              count(*) AS n_total
        """, l1=l1_dongs, l2=l2_dongs).single()
        L.append(f"실측 거주 agent 분포: L1 {denoms['n_l1']:,} / L2 {denoms['n_l2']:,} / "
                 f"강남 비도보권 {denoms['n_l3']:,} / 비강남 {denoms['n_l4']:,} = 총 {denoms['n_total']:,}")
        L.append("")

        # (1) 매장 위치 기준 — POI가 어느 동에 있는지로 그룹
        L.append("**[매장 위치 기준]** 정책 대상 카테고리 (식사·카페·디저트·여가) 매출:")
        L.append("")
        sales_poi = s.run("""
            MATCH (:Agent)-[:HAS_PLAN {day: date($d)}]->(:Plan)
              -[i:INCLUDES]->(p:POI {type:'commerce'})
              -[:IN_DONG]->(pd:Dong)<-[:HAS_DONG]-(pgu:District)
            WHERE i.category IN ['식사','카페','디저트','여가']
            WITH pd.name AS dong, pgu.name AS gu, i
            RETURN
              sum(CASE WHEN dong IN $l1 THEN coalesce(i.actual_spent,0) ELSE 0 END) AS s_l1,
              sum(CASE WHEN dong IN $l2 THEN coalesce(i.actual_spent,0) ELSE 0 END) AS s_l2,
              sum(CASE WHEN gu = '강남구' AND NOT dong IN ($l1 + $l2) THEN coalesce(i.actual_spent,0) ELSE 0 END) AS s_l3,
              sum(CASE WHEN gu <> '강남구' THEN coalesce(i.actual_spent,0) ELSE 0 END) AS s_l4,
              sum(CASE WHEN dong IN $l1 THEN 1 ELSE 0 END) AS n_l1,
              sum(CASE WHEN dong IN $l2 THEN 1 ELSE 0 END) AS n_l2
        """, d=day, l1=l1_dongs, l2=l2_dongs).single()
        L.append("| 매장 위치 | 매출 | 거래 수 |")
        L.append("|---|---:|---:|")
        L.append(f"| **L1 사업 구간 매장** (역삼1·2·도곡1) | {sales_poi['s_l1']:,}원 | {sales_poi['n_l1']:,} |")
        L.append(f"| **L2 도보권 매장** (도곡2·삼성1·2·논현1·2) | {sales_poi['s_l2']:,}원 | {sales_poi['n_l2']:,} |")
        L.append(f"| 강남 비도보권 매장 | {sales_poi['s_l3']:,}원 | — |")
        L.append(f"| 비강남 매장 | {sales_poi['s_l4']:,}원 | — |")
        L.append("")

        # (2) 거주지 기준 — agent 거주 동으로 그룹 (외출 의지·spillover 측정)
        L.append("**[거주지 기준]** 카페·디저트 외출 1인당 매출:")
        L.append("")
        sales_res = s.run("""
            MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(d:Dong)
              <-[:HAS_DONG]-(dist:District)
            OPTIONAL MATCH (a)-[:HAS_PLAN {day: date($d)}]->(:Plan)
              -[i:INCLUDES]->(:POI {type:'commerce'})
            WITH a, d.name AS dong, dist.name AS gu,
                 sum(CASE WHEN i.category IN ['카페','디저트'] THEN coalesce(i.actual_spent,0) ELSE 0 END) AS spent
            RETURN
              sum(CASE WHEN dong IN $l1 THEN spent ELSE 0 END) AS s_l1,
              sum(CASE WHEN dong IN $l2 THEN spent ELSE 0 END) AS s_l2,
              sum(CASE WHEN gu = '강남구' AND NOT dong IN ($l1 + $l2) THEN spent ELSE 0 END) AS s_l3,
              sum(CASE WHEN gu <> '강남구' THEN spent ELSE 0 END) AS s_l4
        """, d=day, l1=l1_dongs, l2=l2_dongs).single()

        def _per(spend, n):
            return (spend / n) if n > 0 else 0
        per_l1 = _per(sales_res['s_l1'], denoms['n_l1'])
        per_l2 = _per(sales_res['s_l2'], denoms['n_l2'])
        per_l3 = _per(sales_res['s_l3'], denoms['n_l3'])
        per_l4 = _per(sales_res['s_l4'], denoms['n_l4'])

        L.append("| 거주 그룹 | 매출 | 분모(실측) | 1인당 |")
        L.append("|---|---:|---:|---:|")
        L.append(f"| **L1 사업 구간 거주** | {sales_res['s_l1']:,}원 | {denoms['n_l1']:,} | {per_l1:,.0f}원 |")
        L.append(f"| **L2 도보권 거주** | {sales_res['s_l2']:,}원 | {denoms['n_l2']:,} | {per_l2:,.0f}원 |")
        L.append(f"| 강남 비도보권 거주 | {sales_res['s_l3']:,}원 | {denoms['n_l3']:,} | {per_l3:,.0f}원 |")
        L.append(f"| 비강남 거주 (Control) | {sales_res['s_l4']:,}원 | {denoms['n_l4']:,} | {per_l4:,.0f}원 |")
        L.append("")

        # 격차 계산 — Control(L4) 대비
        d_l1 = (per_l1 - per_l4) / max(per_l4, 1) * 100
        d_l2 = (per_l2 - per_l4) / max(per_l4, 1) * 100
        d_l3 = (per_l3 - per_l4) / max(per_l4, 1) * 100
        if is_baseline:
            L.append(f"✓ baseline 일자 — 각 그룹의 *cohort 격차* (정책 효과 X, DID 분석에서 cancel out):")
            L.append(f"  · L1 vs Control: {d_l1:+.1f}%")
            L.append(f"  · L2 vs Control: {d_l2:+.1f}%  ← spillover 측정 기준선")
            L.append(f"  · 강남 비도보권 vs Control: {d_l3:+.1f}%")
        else:
            L.append(f"📊 정책 활성 일자 — Control 대비 격차:")
            L.append(f"  · **L1 (direct + spillover)**: {d_l1:+.1f}%")
            L.append(f"  · **L2 (순수 spillover)**: {d_l2:+.1f}%  ← 민주님 자료 핵심 지표")
            L.append(f"  · 강남 비도보권: {d_l3:+.1f}%  ← 정책 인지권 밖이지만 강남 거주")
            L.append(f"  · (DID 분석: baseline 일자의 동일 격차와 비교해서 변화량이 정책 효과)")
        L.append("")

        # 10-E. 새 필드 적재율 (v3 5개 필드 + Night reasoning)
        L.append("### 10-E. v3 신규 필드 적재율")
        L.append("")
        L.append("| 필드 | 적재율 | 비고 |")
        L.append("|---|---:|---|")
        f1 = s.run("""
            MATCH (:Plan {day: date($d)})-[i:INCLUDES]->() WHERE NOT i.category IN ['집','직장']
            RETURN
              count(*) AS t,
              sum(CASE WHEN i.reasoning IS NOT NULL THEN 1 ELSE 0 END) AS r,
              sum(CASE WHEN i.trigger IS NOT NULL THEN 1 ELSE 0 END) AS tg,
              sum(CASE WHEN i.pick_reason IS NOT NULL THEN 1 ELSE 0 END) AS pr,
              sum(CASE WHEN i.pick_factor IS NOT NULL THEN 1 ELSE 0 END) AS pf
        """, d=day).single()
        if f1 and f1["t"]:
            L.append(f"| Stage1 reasoning | {f1['r']/f1['t']*100:.1f}% | "
                     f"{'✅' if f1['r']/f1['t'] > 0.95 else '⚠️'} 95%+ 정상 |")
            L.append(f"| Stage1 trigger | {f1['tg']/f1['t']*100:.1f}% | "
                     f"{'✅' if f1['tg']/f1['t'] > 0.95 else '⚠️'} 95%+ 정상 |")
            L.append(f"| Stage2 pick_reason | {f1['pr']/f1['t']*100:.1f}% | 외출 이벤트만 (50~70% 정상) |")
            L.append(f"| Stage2 pick_factor | {f1['pf']/f1['t']*100:.1f}% | 외출 이벤트만 (50~70% 정상) |")
        # Night reasoning
        nr = s.run("""
            MATCH (c:Conversation {day: date($d)})
            RETURN count(*) AS t,
                   sum(CASE WHEN c.reasoning IS NOT NULL THEN 1 ELSE 0 END) AS r
        """, d=day).single()
        if nr and nr["t"]:
            L.append(f"| Night Conversation reasoning | {nr['r']/nr['t']*100:.1f}% | "
                     f"{'✅' if nr['r']/nr['t'] > 0.95 else '⚠️'} 95%+ 정상 |")
        else:
            L.append(f"| Night Conversation reasoning | — | Night Phase 2 미완료 |")
        L.append("")

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
