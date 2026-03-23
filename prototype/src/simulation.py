"""
시뮬레이션 엔진 — Hybrid D (v2)

24주 × 7일 시뮬레이션 루프.
- 주간: Qwen3-32B (Ollama)로 전략 결정
- 일간: 룰 엔진으로 행동 생성 + 좌표 이동
- 환경: 4종 에이전트(상권/유동인구/정책/뉴스) 매주 업데이트
- 리포트: 매주 구조화된 분석 보고서 생성
- 지도: 일간 좌표 로그 → 부드러운 애니메이션 HTML

Usage:
    python simulation.py                          # 기본 (300명, 24주)
    python simulation.py --agents 100 --weeks 4   # 커스텀
    python simulation.py --no-llm                  # LLM 없이 룰 기반만
"""
import argparse
import json
import time
import numpy as np
from pathlib import Path
from copy import deepcopy

from config import OUTPUT_DIR
from etl_transform import run_etl
from geo_utils import build_grid_map, build_dong_to_grids, assign_coordinates_to_agents
from rule_engine import (
    DAYS_OF_WEEK,
    init_daily_memory,
    update_mood_fatigue,
    generate_daily_actions,
    move_agent,
    update_memory_after_action,
    propagate_recommendations,
    propagate_news_awareness,
)
from llm_client import decide_weekly_batch, _fallback_decision
from visualize_map import save_map
from environment_agents import EnvironmentManager
from report_agent import generate_weekly_report
from animate_map import generate_animation_html
from graph_memory import GraphMemoryManager


# ═══════════════════════════════════════════
# 주간 요약 집계
# ═══════════════════════════════════════════

def summarize_week(agent_id: str, week_log: list[list[dict]]) -> dict:
    """에이전트의 1주일 행동 로그 → 주간 요약"""
    visited_dongs = set()
    industries = set()
    total_spent = 0
    satisfactions = []

    for day_actions in week_log:
        for act in day_actions:
            if act.get("dong"):
                visited_dongs.add(act["dong"])
            if act.get("industry"):
                industries.add(act["industry"])
            total_spent += act.get("amount", 0)
            if act.get("satisfaction"):
                satisfactions.append(act["satisfaction"])

    return {
        "visited_dongs": list(visited_dongs),
        "industries": list(industries),
        "total_spent": total_spent,
        "satisfaction": round(np.mean(satisfactions), 2) if satisfactions else 0.5,
    }


# ═══════════════════════════════════════════
# 시뮬레이션 메인 루프
# ═══════════════════════════════════════════

def run_simulation(
    total_agents: int = 300,
    total_weeks: int = 24,
    use_llm: bool = True,
    scenario: list[dict] = None,
    seed: int = 42,
    save_snapshots: bool = True,
    snapshot_interval: int = 1,   # 1일마다 좌표 저장 (애니메이션용)
) -> dict:
    """Hybrid D 시뮬레이션 실행

    Args:
        total_agents: 에이전트 수
        total_weeks: 시뮬레이션 주 수
        use_llm: True → Qwen3 호출, False → 전부 룰 기반
        scenario: 주차별 이벤트 리스트 [{week, type, description, ...}]
        seed: 랜덤 시드
        save_snapshots: 스냅샷 저장 여부
        snapshot_interval: N일마다 좌표 스냅샷 저장 (1=매일)

    Returns:
        {agents, weekly_stats, daily_positions, total_days, total_spending, reports}
    """
    rng = np.random.default_rng(seed)

    # ── 1. ETL + 좌표 할당 ──
    print("=" * 60)
    print("Simulation Setup")
    print("=" * 60)

    etl_result = run_etl(total_agents)
    agents = etl_result["agents"]
    district_profiles = etl_result["district_profiles"]

    grid_map = build_grid_map()
    dong_grids = build_dong_to_grids(grid_map)
    agents = assign_coordinates_to_agents(agents, dong_grids, kt_od=etl_result["kt_od"], seed=seed)

    # ── 2. 초기화 ──
    env = EnvironmentManager(district_profiles, scenario)
    memories = {a["agent_id"]: init_daily_memory() for a in agents}
    weekly_summaries = {}  # {agent_id: last_week_summary}
    weekly_decisions = {}  # {agent_id: llm_decision}

    # ── GraphRAG 초기화 ──
    graph_mgr = GraphMemoryManager()
    graph_mgr.initialize(
        agents, district_profiles,
        industry_groups=env.district_agent.industry_groups,
        rng=rng,
    )

    # 결과 저장
    daily_positions = []    # [(day_idx, agents_snapshot)] — 애니메이션용
    weekly_stats = []       # [{week, total_spending, actions_count, ...}]
    reports = []            # [report_dict] — 주간 리포트

    total_day = 0
    total_spending = 0
    prev_report = None

    print(f"\n{'=' * 60}")
    print(f"Simulation Start: {total_agents} agents, {total_weeks} weeks")
    print(f"LLM: {'Qwen3-32B (Ollama)' if use_llm else 'Disabled (rule-based only)'}")
    print(f"Environment: DistrictAgent + PopulationAgent + PolicyAgent + NewsAgent")
    print(f"{'=' * 60}\n")

    sim_start = time.time()

    # ── 3. 시뮬레이션 루프 ──
    for week in range(total_weeks):
        week_start = time.time()

        # ── Phase 1: 환경 에이전트 업데이트 ──
        print(f"\n  ── Week {week:2d}/{total_weeks} ──")
        env.advance_week(week, rng, graph_mgr=graph_mgr, use_llm=use_llm)

        # ── Phase 2: 주간 LLM 결정 ──
        if use_llm:
            # 에이전트별 맞춤 환경 컨텍스트 + GraphRAG
            env_state = env.get_state_for_llm(agents[0])  # 대표 상태
            # GraphRAG 컨텍스트를 env_state에 주입
            for agent in agents:
                aid = agent["agent_id"]
                graph_ctx = graph_mgr.get_llm_context(aid, week)
                if graph_ctx:
                    agent["_graph_rag_context"] = graph_ctx
            weekly_decisions = decide_weekly_batch(
                agents, env_state, weekly_summaries, max_workers=4,
            )
        else:
            # LLM 없이 기본 결정 (GraphRAG 컨텍스트는 룰 엔진에 직접 전달)
            weekly_decisions = {a["agent_id"]: _fallback_decision() for a in agents}

        # 추천 전파
        propagate_recommendations(agents, memories, weekly_decisions, rng)

        week_spending = 0
        week_actions_count = 0
        week_logs = {a["agent_id"]: [] for a in agents}

        # 주간 시작 시 뉴스 인지 상태 초기화 (새 주 = 새 뉴스)
        for a in agents:
            a["_news_awareness"] = {}

        # ── Phase 3: 일간 루프 (7일) ──
        for day_in_week in range(7):
            day_name = DAYS_OF_WEEK[day_in_week]

            # 요일 유형
            if day_name == "friday":
                day_type = "friday"
            elif day_name in ("saturday", "sunday"):
                day_type = "weekend"
            else:
                day_type = "weekday"

            for agent in agents:
                aid = agent["agent_id"]
                memory = memories[aid]
                directive = weekly_decisions.get(aid, _fallback_decision())

                # 환경 컨텍스트 (에이전트별 유동인구 보정)
                env_ctx = env.get_context(agent)
                pop_factor = env_ctx.get("population_factor", 1.0)

                # GraphRAG 룰엔진 컨텍스트 (주 1회 캐시)
                graph_ctx = agent.get("_graph_rule_context")
                if day_in_week == 0 or graph_ctx is None:
                    graph_ctx = graph_mgr.get_rule_context(aid, week)
                    agent["_graph_rule_context"] = graph_ctx

                # 기분/피로 업데이트
                update_mood_fatigue(memory, day_name, rng)

                # 일간 행동 생성 (env_context + graph_context 전달)
                actions = generate_daily_actions(
                    agent, memory, directive, day_name, total_day, rng,
                    env_context=env_ctx,
                    graph_context=graph_ctx,
                )

                # 좌표 이동
                move_agent(agent, actions, dong_grids, rng)

                # memory 업데이트
                update_memory_after_action(memory, total_day, actions)

                # GraphRAG 에피소드 기록
                for act in actions:
                    graph_mgr.record_action(aid, act, week, day_in_week)

                # 집계
                day_spent = sum(a.get("amount", 0) for a in actions)
                week_spending += day_spent
                total_spending += day_spent
                week_actions_count += len([a for a in actions if a["type"] == "외출_소비"])
                week_logs[aid].append(actions)

            # ── 행동별 좌표 스냅샷 (의미있는 이동 시각화) ──
            if save_snapshots and total_day % snapshot_interval == 0:
                # 각 에이전트의 마지막 외출 행동 정보 수집
                agent_last_actions = {}
                for a in agents:
                    aid_snap = a["agent_id"]
                    acts = week_logs.get(aid_snap, [[]])
                    today_acts = acts[-1] if acts else []
                    outings = [ac for ac in today_acts if ac.get("type") == "외출_소비"]
                    if outings:
                        last = outings[-1]
                        agent_last_actions[aid_snap] = {
                            "industry": last.get("industry", ""),
                            "amount": last.get("amount", 0),
                            "satisfaction": last.get("satisfaction", 0.5),
                            "time_slot": last.get("time_slot", 12),
                            "dong": last.get("dong", ""),
                            "triggered_by": last.get("triggered_by", [])[:3],
                        }

                snapshot = []
                for a in agents:
                    snap = {
                        "agent_id": a["agent_id"],
                        "segment": a["segment"],
                        "current_lat": a["current_lat"],
                        "current_lng": a["current_lng"],
                        "lifestyle": a.get("lifestyle", ""),
                        "age_group": a.get("age_group", ""),
                    }
                    act_info = agent_last_actions.get(a["agent_id"])
                    if act_info:
                        snap.update(act_info)
                    snapshot.append(snap)
                daily_positions.append((total_day, snapshot))

            total_day += 1

            # 매일: 뉴스 인지 소셜 전파
            structured_events = env.news_agent.get_news() if hasattr(env, 'news_agent') else []
            propagate_news_awareness(agents, structured_events, rng)

        # ── Phase 4: 주간 집계 + GraphRAG 주말 처리 ──
        graph_mgr.end_of_week(week, rng)  # 입소문 전파

        # 소비자 활동 → DistrictAgent에 전달 (다음 주 개폐업 판정에 사용)
        from collections import Counter
        dong_ind_visits = Counter()
        dong_ind_spending = Counter()
        dong_ind_satisfaction = Counter()
        dong_ind_count = Counter()
        for aid, day_logs in week_logs.items():
            for day_actions in day_logs:
                for act in day_actions:
                    if act.get("type") == "외출_소비" and act.get("dong") and act.get("industry"):
                        key = (act["dong"], act["industry"])
                        dong_ind_visits[key] += 1
                        dong_ind_spending[key] += act.get("amount", 0)
                        dong_ind_satisfaction[key] += act.get("satisfaction", 0.5)
                        dong_ind_count[key] += 1
        for (dong, ind), visits in dong_ind_visits.items():
            avg_sat = dong_ind_satisfaction[(dong, ind)] / max(dong_ind_count[(dong, ind)], 1)
            env.district_agent.receive_consumer_activity(
                dong, ind, visits, dong_ind_spending[(dong, ind)], avg_sat,
            )

        weekly_summaries = {}
        for agent in agents:
            aid = agent["agent_id"]
            weekly_summaries[aid] = summarize_week(aid, week_logs[aid])

        week_elapsed = time.time() - week_start
        env_summary = env.get_week_summary()

        ws = {
            "week": week,
            "total_spending": week_spending,
            "actions_count": week_actions_count,
            "avg_spending_per_agent": round(week_spending / total_agents),
            "events": env_summary.get("news", []),
            "policy": env_summary.get("active_policy", "없음"),
        }
        weekly_stats.append(ws)

        # ── Phase 5: 주간 리포트 생성 ──
        report = generate_weekly_report(
            week, agents, week_logs, env_summary, prev_report,
            graph_mgr=graph_mgr,
        )
        reports.append(report)
        prev_report = report

        # 진행 상황 출력
        print(
            f"  Week {week:2d}/{total_weeks} | "
            f"Spending: {week_spending:>12,}won | "
            f"Actions: {week_actions_count:>4d} | "
            f"Avg/agent: {week_spending // total_agents:>7,}won | "
            f"{week_elapsed:.1f}s"
        )

    sim_elapsed = time.time() - sim_start

    # ── 4. 결과 정리 ──
    # GraphRAG 영속화
    graph_mgr.save(str(OUTPUT_DIR))
    graph_stats = graph_mgr.stats()

    print(f"\n{'=' * 60}")
    print(f"Simulation Complete")
    print(f"  Total days: {total_day}")
    print(f"  Total spending: {total_spending:,}won")
    print(f"  Avg spending/agent/week: {total_spending // total_agents // total_weeks:,}won")
    print(f"  Snapshots saved: {len(daily_positions)}")
    print(f"  Reports generated: {len(reports)}")
    print(f"  GraphRAG: {graph_stats['total_episodes']} episodes, "
          f"{graph_stats['social_edges']} social edges, "
          f"{graph_stats['communities']} communities")
    print(f"  Elapsed: {sim_elapsed:.1f}s")
    print(f"{'=' * 60}")

    return {
        "agents": agents,
        "weekly_stats": weekly_stats,
        "daily_positions": daily_positions,
        "total_days": total_day,
        "total_spending": total_spending,
        "dong_grids": dong_grids,
        "reports": reports,
    }


# ═══════════════════════════════════════════
# 결과 저장 + 시각화
# ═══════════════════════════════════════════

def save_results(result: dict) -> None:
    """시뮬레이션 결과를 파일로 저장"""

    # 1. 주간 통계 JSON
    stats_path = OUTPUT_DIR / "weekly_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(result["weekly_stats"], f, ensure_ascii=False, indent=2)
    print(f"[save] Weekly stats -> {stats_path}")

    # 2. 최종 에이전트 상태 지도
    save_map(result["agents"], "agent_map_final.html", title="Final Position")

    # 3. 초기 에이전트 지도 (첫 스냅샷)
    positions = result.get("daily_positions", [])
    if positions:
        day0, snap0 = positions[0]
        _save_snapshot_map(snap0, result["agents"], "agent_map_initial.html", "Day 0")

    # 4. 부드러운 애니메이션 HTML
    if positions:
        generate_animation_html(
            positions,
            weekly_stats=result["weekly_stats"],
            output_name="simulation_animation.html",
        )

    # 5. 소비 트렌드 요약
    print_spending_trend(result["weekly_stats"])


def _save_snapshot_map(snapshot: list[dict], agents: list[dict], filename: str, title: str) -> None:
    """스냅샷 데이터로 에이전트 위치를 갱신 후 지도 저장"""
    snap_map = {s["agent_id"]: s for s in snapshot}
    temp_agents = []
    for a in agents:
        a_copy = dict(a)
        snap = snap_map.get(a["agent_id"])
        if snap:
            a_copy["current_lat"] = snap.get("current_lat", snap.get("lat"))
            a_copy["current_lng"] = snap.get("current_lng", snap.get("lng"))
        temp_agents.append(a_copy)

    save_map(temp_agents, filename, title=title)


def print_spending_trend(weekly_stats: list[dict]) -> None:
    """주간 소비 트렌드 텍스트 출력"""
    print(f"\n--- Weekly Spending Trend ---")
    for ws in weekly_stats:
        bar_len = ws["avg_spending_per_agent"] // 10000
        bar = "█" * min(bar_len, 40)
        print(f"  W{ws['week']:02d}: {ws['avg_spending_per_agent']:>7,}won |{bar}")
        if ws.get("events"):
            print(f"        └─ {', '.join(ws['events'])}")


# ═══════════════════════════════════════════
# 데모 시나리오
# ═══════════════════════════════════════════

DEMO_SCENARIO = [
    # ── 외부 매크로 요인만 정의 (정책, 유동인구 변화) ──
    # 뉴스 이벤트는 NewsAgent가 시뮬레이션 상태/계절/랜덤에 따라 자율 생성합니다.

    # Week 3: 소상공인 지원 정책 시작
    {"week": 3, "type": "policy", "description": "서울시 소상공인 5천원 할인 쿠폰 배포", "duration": 8},
    # Week 7: 봄맞이 유동인구 증가
    {"week": 7, "type": "pop_change", "value": "+10%"},
    # Week 14: 여름 휴가철 유동인구 감소
    {"week": 14, "type": "pop_change", "value": "-8%"},
    # Week 18: 가을 축제 시즌 유동인구 증가
    {"week": 18, "type": "pop_change", "value": "+12%"},
]


# ═══════════════════════════════════════════
# CLI 진입점
# ═══════════════════════════════════════════

def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description="Consumer Behavior Simulation (Hybrid D v2)")
    parser.add_argument("--agents", type=int, default=300, help="Number of agents")
    parser.add_argument("--weeks", type=int, default=24, help="Simulation weeks")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (rule-based only)")
    parser.add_argument("--no-scenario", action="store_true", help="Run without demo scenario")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--snapshot-interval", type=int, default=1, help="Save daily position every N days (1=every day)")
    args = parser.parse_args()

    scenario = None if args.no_scenario else DEMO_SCENARIO

    result = run_simulation(
        total_agents=args.agents,
        total_weeks=args.weeks,
        use_llm=not args.no_llm,
        scenario=scenario,
        seed=args.seed,
        snapshot_interval=args.snapshot_interval,
    )

    save_results(result)


if __name__ == "__main__":
    main()
