"""
analyze_repeat_visits.py
========================
시뮬 결과의 POI 반복 방문 패턴 측정 — 같은 날 분할 효과, desire 곡선 효과 검증용.

6개 지표를 Neo4j Plan/INCLUDES 에서 계산해 markdown + json 으로 출력:

  1. unique_poi_per_day        : agent×day 별 unique POI / commerce 이벤트 비율
  2. same_day_repeat_rate      : 하루에 같은 POI 2회+ 픽한 agent 비율 (풀 분할 효과 직접)
  3. revisit_interval_dist     : 같은 (agent, POI) 재방문 일수 분포 (desire 곡선 효과)
  4. top_poi_concentration     : 상위 10/50 POI 가 차지하는 방문 비율 (단골 쏠림)
  5. category_diversity        : 기간 내 agent 별 unique sub_category 수 분포
  6. district_event_share      : 자치구별 commerce 이벤트 평균 (정책 효과 참고)

사용:
  python -m scripts.sim.analyze_repeat_visits --start 2026-05-01 --days 3
  python -m scripts.sim.analyze_repeat_visits --start 2026-05-01 --days 3 \\
      --out-dir output/sim/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# 데이터 모델 (Neo4j 결과 → in-memory)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlanEvent:
    aid: str
    day: date
    poi_id: str
    l1: str | None
    sub: str | None
    district: str | None    # POI 가 속한 자치구 코드 (5자리)


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------
FETCH_EVENTS_CYPHER = """
MATCH (a:Agent)-[hp:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(poi:POI)
WHERE hp.day >= date($start) AND hp.day < date($end)
  AND i.category IS NOT NULL
  AND NOT i.category IN ['집', '직장']
  AND poi.id IS NOT NULL
OPTIONAL MATCH (poi)-[:IN_DONG]->(:Dong)<-[:HAS_DONG]-(dist:District)
RETURN a.id AS aid, hp.day AS day, poi.id AS poi_id,
       i.category AS l1, i.sub_category AS sub,
       dist.code AS district
"""


def fetch_plan_events(start: date, days: int) -> list[PlanEvent]:
    """Neo4j 에서 기간 내 모든 commerce 이벤트 수집."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))
    from _common import driver_session  # type: ignore

    end = start + timedelta(days=days)
    out: list[PlanEvent] = []
    with driver_session() as s:
        for r in s.run(FETCH_EVENTS_CYPHER, start=start.isoformat(), end=end.isoformat()):
            d = r["day"]
            # neo4j.time.Date → datetime.date
            if hasattr(d, "to_native"):
                d = d.to_native()
            out.append(PlanEvent(
                aid=r["aid"], day=d, poi_id=r["poi_id"],
                l1=r["l1"], sub=r["sub"], district=r["district"],
            ))
    return out


# ---------------------------------------------------------------------------
# 지표 계산 — 순수 함수 (단위 테스트 가능)
# ---------------------------------------------------------------------------
def compute_unique_poi_per_day(events: list[PlanEvent]) -> dict:
    """agent×day 별 unique POI / 총 이벤트 비율."""
    by_aid_day: dict[tuple[str, date], list[str]] = defaultdict(list)
    for e in events:
        by_aid_day[(e.aid, e.day)].append(e.poi_id)

    ratios: list[float] = []
    for pois in by_aid_day.values():
        if not pois:
            continue
        ratios.append(len(set(pois)) / len(pois))

    if not ratios:
        return {"n": 0, "mean": None, "median": None}
    ratios.sort()
    return {
        "n": len(ratios),
        "mean": round(sum(ratios) / len(ratios), 4),
        "median": round(ratios[len(ratios) // 2], 4),
        "p25": round(ratios[len(ratios) // 4], 4),
        "p75": round(ratios[3 * len(ratios) // 4], 4),
    }


def compute_same_day_repeat_rate(events: list[PlanEvent]) -> dict:
    """하루에 같은 POI 2회 이상 픽한 (agent, day) 비율."""
    by_aid_day: dict[tuple[str, date], list[str]] = defaultdict(list)
    for e in events:
        by_aid_day[(e.aid, e.day)].append(e.poi_id)

    total = 0
    repeats = 0
    for pois in by_aid_day.values():
        if len(pois) < 2:
            continue
        total += 1
        if len(set(pois)) < len(pois):
            repeats += 1

    return {
        "agent_days_with_2plus_commerce": total,
        "with_repeat": repeats,
        "rate": round(repeats / total, 4) if total else None,
    }


def compute_revisit_intervals(events: list[PlanEvent]) -> dict:
    """같은 (agent, POI) 재방문 일수 간격."""
    by_aid_poi: dict[tuple[str, str], list[date]] = defaultdict(list)
    for e in events:
        by_aid_poi[(e.aid, e.poi_id)].append(e.day)

    intervals: list[int] = []
    for days_list in by_aid_poi.values():
        if len(days_list) < 2:
            continue
        days_list.sort()
        for i in range(1, len(days_list)):
            intervals.append((days_list[i] - days_list[i - 1]).days)

    if not intervals:
        return {"n": 0}
    intervals.sort()
    buckets = Counter()
    for d in intervals:
        if d == 0:           # 같은 날 (제외 — same_day_repeat 와 중복)
            continue
        elif d == 1:
            buckets["1일"] += 1
        elif d <= 3:
            buckets["2-3일"] += 1
        elif d <= 7:
            buckets["4-7일"] += 1
        elif d <= 14:
            buckets["8-14일"] += 1
        else:
            buckets["15일+"] += 1
    nz = [d for d in intervals if d > 0]
    return {
        "n": len(nz),
        "median_days": nz[len(nz) // 2] if nz else None,
        "mean_days": round(sum(nz) / len(nz), 2) if nz else None,
        "buckets": dict(buckets),
    }


def compute_top_poi_concentration(events: list[PlanEvent]) -> dict:
    """상위 N개 POI 가 차지하는 전체 commerce 방문 비율."""
    counts = Counter(e.poi_id for e in events)
    total = sum(counts.values())
    if total == 0:
        return {"total": 0}
    sorted_counts = sorted(counts.values(), reverse=True)
    return {
        "total_visits": total,
        "unique_pois": len(counts),
        "top10_share": round(sum(sorted_counts[:10]) / total, 4),
        "top50_share": round(sum(sorted_counts[:50]) / total, 4),
        "top100_share": round(sum(sorted_counts[:100]) / total, 4),
    }


def compute_category_diversity(events: list[PlanEvent]) -> dict:
    """agent 별 기간 내 unique sub_category 수 분포."""
    by_aid: dict[str, set[str]] = defaultdict(set)
    for e in events:
        if e.sub:
            by_aid[e.aid].add(e.sub)

    counts = sorted(len(s) for s in by_aid.values())
    if not counts:
        return {"n": 0}
    return {
        "n_agents": len(counts),
        "mean_unique_subs": round(sum(counts) / len(counts), 2),
        "median": counts[len(counts) // 2],
        "p25": counts[len(counts) // 4],
        "p75": counts[3 * len(counts) // 4],
    }


def compute_district_event_share(events: list[PlanEvent]) -> dict:
    """자치구별 commerce 이벤트 평균 (agent 1인당)."""
    by_dist_aid: dict[str, set[str]] = defaultdict(set)
    by_dist_count: Counter = Counter()
    for e in events:
        if e.district:
            by_dist_count[e.district] += 1
            by_dist_aid[e.district].add(e.aid)

    out = {}
    for dist, n in sorted(by_dist_count.items(), key=lambda x: -x[1]):
        n_agents = len(by_dist_aid[dist])
        out[dist] = {
            "events": n,
            "agents": n_agents,
            "events_per_agent": round(n / n_agents, 2) if n_agents else None,
        }
    return out


# ---------------------------------------------------------------------------
# 출력 포맷
# ---------------------------------------------------------------------------
def _ascii_bar(value: int, max_value: int, width: int = 30) -> str:
    if max_value <= 0:
        return ""
    n = int(value / max_value * width)
    return "█" * n + "·" * (width - n)


def render_markdown(metrics: dict, start: date, days: int) -> str:
    lines = [
        f"# POI 반복 방문 분석 — {start.isoformat()} ~ {(start + timedelta(days=days-1)).isoformat()} ({days}일)",
        "",
    ]

    # 1. unique_poi_per_day
    u = metrics["unique_poi_per_day"]
    lines.append("## 1. agent×day 별 unique POI 비율")
    lines.append("")
    lines.append("`unique_POI / 총_commerce_이벤트` — 1.0 = 모두 다른 가게, 0.5 = 절반 반복")
    lines.append("")
    if u["n"]:
        lines.append(f"- n (agent-days) = {u['n']:,}")
        lines.append(f"- mean = **{u['mean']:.3f}**")
        lines.append(f"- median = {u['median']:.3f}")
        lines.append(f"- p25 / p75 = {u['p25']:.3f} / {u['p75']:.3f}")
    else:
        lines.append("- (데이터 없음)")
    lines.append("")

    # 2. same_day_repeat
    s = metrics["same_day_repeat_rate"]
    lines.append("## 2. 같은 날 같은 POI 반복 비율 — 풀 분할 효과 직접 측정")
    lines.append("")
    if s.get("rate") is not None:
        lines.append(f"- 2개 이상 commerce 이벤트 있는 agent-day: {s['agent_days_with_2plus_commerce']:,}")
        lines.append(f"- 그 중 같은 POI 2회+ 등장: {s['with_repeat']:,}")
        lines.append(f"- **반복률 = {s['rate']*100:.2f}%**")
        if s['rate'] == 0:
            lines.append("  - ✅ 풀 분할이 완벽 작동 (이상적인 값)")
        elif s['rate'] < 0.05:
            lines.append("  - ✅ 거의 0 — 풀 분할 효과적")
        elif s['rate'] < 0.2:
            lines.append("  - ⚠️ 일부 케이스에서 발생 (fallback 풀 작거나 LLM 오류)")
        else:
            lines.append("  - ❌ 풀 분할이 충분치 않음 — 원인 파악 필요")
    else:
        lines.append("- (해당 agent-day 없음)")
    lines.append("")

    # 3. revisit_intervals
    r = metrics["revisit_interval_dist"]
    lines.append("## 3. 같은 (agent, POI) 재방문 일수 분포 — desire 곡선 효과")
    lines.append("")
    if r.get("n"):
        lines.append(f"- 재방문 pair 수: {r['n']:,}")
        lines.append(f"- 중앙값 / 평균: {r['median_days']}일 / {r['mean_days']}일")
        lines.append("")
        lines.append("```")
        buckets = r.get("buckets", {})
        m = max(buckets.values()) if buckets else 1
        for label in ["1일", "2-3일", "4-7일", "8-14일", "15일+"]:
            v = buckets.get(label, 0)
            lines.append(f"  {label:>7}: {_ascii_bar(v, m)} {v:,}")
        lines.append("```")
        lines.append("")
        lines.append("- 1일 / 2-3일 비율이 높을수록 어제 간 곳 또 가는 패턴 강함")
        lines.append("  → desire 곡선 적용 후 8-14일 / 15일+ 로 분포 이동 기대")
    else:
        lines.append("- (재방문 pair 없음 — 기간 짧거나 단골 적음)")
    lines.append("")

    # 4. top concentration
    t = metrics["top_poi_concentration"]
    lines.append("## 4. 상위 POI 쏠림 — 단골 집중도")
    lines.append("")
    if t.get("total_visits"):
        lines.append(f"- 총 commerce 방문: {t['total_visits']:,}")
        lines.append(f"- unique POI: {t['unique_pois']:,}")
        lines.append(f"- 상위 10 POI 점유율 = **{t['top10_share']*100:.1f}%**")
        lines.append(f"- 상위 50 POI 점유율 = {t['top50_share']*100:.1f}%")
        lines.append(f"- 상위 100 POI 점유율 = {t['top100_share']*100:.1f}%")
    lines.append("")

    # 5. category diversity
    c = metrics["category_diversity"]
    lines.append("## 5. agent 별 기간 내 unique sub_category 수")
    lines.append("")
    if c.get("n_agents"):
        lines.append(f"- agent 수 = {c['n_agents']:,}")
        lines.append(f"- 평균 unique sub_category = **{c['mean_unique_subs']}**")
        lines.append(f"- median / p25 / p75 = {c['median']} / {c['p25']} / {c['p75']}")
    lines.append("")

    # 6. district share
    d = metrics["district_event_share"]
    if d:
        lines.append("## 6. 자치구별 commerce 이벤트 (참고)")
        lines.append("")
        lines.append("| 자치구 | events | agents | events/agent |")
        lines.append("|--------|-------:|-------:|-------------:|")
        for code, info in list(d.items())[:25]:
            lines.append(f"| {code} | {info['events']:,} | {info['agents']:,} | {info['events_per_agent']} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def compute_all_metrics(events: list[PlanEvent]) -> dict:
    return {
        "unique_poi_per_day": compute_unique_poi_per_day(events),
        "same_day_repeat_rate": compute_same_day_repeat_rate(events),
        "revisit_interval_dist": compute_revisit_intervals(events),
        "top_poi_concentration": compute_top_poi_concentration(events),
        "category_diversity": compute_category_diversity(events),
        "district_event_share": compute_district_event_share(events),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="POI repeat-visit analysis from Neo4j Plans")
    parser.add_argument("--start", required=True, help="시뮬 시작일 YYYY-MM-DD")
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--out-dir", default="output/sim/analysis",
                        help="md/json 출력 디렉토리 (기본: output/sim/analysis)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fetch] Plan events {start} +{args.days}d ...")
    events = fetch_plan_events(start, args.days)
    print(f"  → {len(events):,} commerce events")
    if not events:
        print("  ⚠️ 데이터 없음 — Neo4j 에 Plan/INCLUDES 가 있는지 확인")
        return 2

    metrics = compute_all_metrics(events)
    metrics["meta"] = {
        "start": args.start, "days": args.days,
        "total_events": len(events),
    }

    stem = f"repeat_visits_{args.start}_{args.days}d"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2,
                                    default=str), encoding="utf-8")
    md_path.write_text(render_markdown(metrics, start, args.days), encoding="utf-8")

    print(f"[write] {md_path}")
    print(f"[write] {json_path}")
    print()
    # 핵심 지표 stdout 요약
    print("=" * 50)
    print(f"unique_poi_per_day mean : {metrics['unique_poi_per_day']['mean']}")
    print(f"same_day_repeat_rate    : {metrics['same_day_repeat_rate'].get('rate')}")
    print(f"top10_poi_share         : {metrics['top_poi_concentration'].get('top10_share')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
