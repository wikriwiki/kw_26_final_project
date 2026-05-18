"""풀런 결과를 인터랙티브 HTML 시각화용 JSON으로 export.

agents: 자치구별 표본 추출 (기본: 강남 300 + 중구 300 + 종로 300 = 900명)
프레임: 24h × N day timeframe
산출물 (기본): G:/내 드라이브/Kw/final_project/output/sim/visualization/{agents,timeline,memories,events}.json
            VIZ_OUT_DIR 환경변수로 override 가능

CLI:
  python export_visualization.py                              # 기본 5/01~5/03
  python export_visualization.py --start 2026-05-01 --days 3 # 명시
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))
from _common import driver_session  # noqa: E402

# 출력 경로: 환경변수 우선, 없으면 프로젝트 루트의 output/sim/visualization (공유용 G: 보관)
OUT_DIR = Path(os.environ.get(
    "VIZ_OUT_DIR",
    str(Path(__file__).resolve().parents[2] / "output" / "sim" / "visualization")
))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 기본값 — CLI로 override 가능
DEFAULT_START = "2026-05-01"
DEFAULT_DAYS = 3
# 서울 25 자치구 균등 표본 (자치구당 200명 = 5,000명).
# 사용자가 viz에서 자치구 필터로 골라볼 수 있음.
SAMPLE = {
    "11110": 200,  # 종로구
    "11140": 200,  # 중구
    "11170": 200,  # 용산구
    "11200": 200,  # 성동구
    "11215": 200,  # 광진구
    "11230": 200,  # 동대문구
    "11260": 200,  # 중랑구
    "11290": 200,  # 성북구
    "11305": 200,  # 강북구
    "11320": 200,  # 도봉구
    "11350": 200,  # 노원구
    "11380": 200,  # 은평구
    "11410": 200,  # 서대문구
    "11440": 200,  # 마포구
    "11470": 200,  # 양천구
    "11500": 200,  # 강서구
    "11530": 200,  # 구로구
    "11545": 200,  # 금천구
    "11560": 200,  # 영등포구
    "11590": 200,  # 동작구
    "11620": 200,  # 관악구
    "11650": 200,  # 서초구
    "11680": 200,  # 강남구
    "11710": 200,  # 송파구
    "11740": 200,  # 강동구
}


def fetch_agents(s):
    """자치구별 표본 추출 — 다양성 위해 income·age 분포 고려."""
    agents = []
    for dist_code, limit in SAMPLE.items():
        rows = s.run("""
            MATCH (a:Agent)-[:LIVES_AT]->(home:POI)-[:IN_DONG]->(hd:Dong)<-[:HAS_DONG]-(dist:District {code:$dc})
            OPTIONAL MATCH (a)-[wr:WORKS_AT]->(work:POI)
            WITH a, dist, hd, home, work, wr,
                 // 다양성을 위해 ID 끝 3글자로 정렬
                 substring(a.id, size(a.id) - 3, 3) AS suffix
            RETURN
              a.id AS id,
              a.p_age_group AS age, a.p_gender AS gender,
              a.p_income_level AS income, a.p_life_stage AS life_stage,
              a.personal_job_raw AS job, a.pr_spending_tendency AS tendency,
              a.s_daily_wd AS daily_wd, a.s_daily_we AS daily_we,
              a.spending_top_wd_json AS top_wd,
              a.personality_lifestyle_raw AS lifestyle,
              dist.name AS district, dist.code AS dist_code,
              hd.name AS home_dong, home.id AS home_poi_id,
              home.name AS home_poi_name, home.lon AS home_lon, home.lat AS home_lat,
              work.id AS work_poi_id, work.name AS work_poi_name,
              work.lon AS work_lon, work.lat AS work_lat,
              wr.commute_min AS commute
            ORDER BY suffix
            LIMIT $limit
        """, dc=dist_code, limit=limit)
        for r in rows:
            d = dict(r)
            # numbers
            for k in ["daily_wd", "daily_we", "home_lon", "home_lat",
                      "work_lon", "work_lat", "commute"]:
                if d.get(k) is None:
                    d[k] = None
            agents.append(d)
        print(f"  {dist_code}: {limit} agents")
    return agents


def fetch_timeline(s, agent_ids: list[str]):
    """각 agent의 Day별 INCLUDES 이벤트 → 시간 프레임 구성.

    프레임 = (day_idx 0..2) × (hour 0..23) = 72개
    각 프레임에서 agent는 그 시간 시점의 활동 POI에 위치 (가장 가까운 이전 이벤트의 POI).
    """
    print(f"[timeline] fetching for {len(agent_ids)} agents × {len(DAYS)} days ...")
    # 모든 agent × day의 INCLUDES 한 번에 쿼리 (DAYS list 동적 처리)
    rows = s.run("""
        MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(poi:POI)
        WHERE a.id IN $aids AND toString(p.day) IN $days
        OPTIONAL MATCH (poi)-[:IN_CATEGORY]->(c:Category)
        RETURN a.id AS aid, p.day AS day, i.order AS ord,
               toString(i.time) AS time, i.anchor AS anchor,
               i.category AS cat, i.sub_category AS sub_cat,
               i.intent AS intent, i.actual_satisfaction AS sat,
               i.actual_spent AS spent,
               poi.id AS poi_id, poi.name AS poi_name,
               poi.lon AS lon, poi.lat AS lat, poi.type AS poi_type,
               c.parent AS l1
    """, aids=agent_ids, days=DAYS)

    # agent별 시간순 이벤트 list
    by_agent = defaultdict(list)
    for r in rows:
        by_agent[r["aid"]].append({
            "day": str(r["day"]),
            "ord": r["ord"],
            "time": r["time"][:5] if r["time"] else "00:00",  # HH:MM
            "anchor": r["anchor"], "cat": r["cat"], "sub": r["sub_cat"],
            "intent": r["intent"], "sat": r["sat"],
            "spent": r["spent"] or 0,
            "poi_id": r["poi_id"], "poi_name": r["poi_name"] or "",
            "lon": r["lon"], "lat": r["lat"],
            "poi_type": r["poi_type"], "l1": r["l1"],
        })
    # 시간순 정렬
    for aid in by_agent:
        by_agent[aid].sort(key=lambda e: (e["day"], e["ord"]))

    print(f"  events fetched for {len(by_agent)} agents")

    # 72 프레임 생성: 각 (day_idx, hour) 시점에 각 agent의 위치
    frames = []
    for day_idx, day_str in enumerate(DAYS):
        for hour in range(0, 24):
            frame_agents = []
            for aid, events in by_agent.items():
                # 해당 day의 hour까지의 마지막 이벤트
                day_events = [e for e in events if e["day"] == day_str]
                # hour시간까지 발생한 이벤트 중 가장 늦은 것 (HH:MM의 시 부분)
                current = None
                for e in day_events:
                    eh = int(e["time"][:2])
                    if eh <= hour:
                        current = e
                    else:
                        break
                if current is None and day_idx > 0:
                    # 어제 마지막 이벤트로 fallback (집)
                    prev_day_events = [e for e in events if e["day"] == DAYS[day_idx - 1]]
                    if prev_day_events:
                        current = prev_day_events[-1]
                if current and current.get("lon") and current.get("lat"):
                    frame_agents.append({
                        "id": aid,
                        "lon": current["lon"], "lat": current["lat"],
                        "cat": current.get("cat") or "집",
                        "l1": current.get("l1"),
                        "intent": current["intent"],
                        "sat": current["sat"],
                        "spent": current.get("spent") or 0,
                        "anchor": current["anchor"],
                    })
            frames.append({
                "day_idx": day_idx, "day": day_str, "hour": hour,
                "label": f"Day {day_idx+1} {day_str} {hour:02d}:00",
                "agents": frame_agents,
            })
        print(f"  Day {day_idx+1} ({day_str}) frames built")
    return frames, by_agent


def fetch_memories(s, agent_ids: list[str]):
    """각 agent의 모든 Memory(type별) + KNOWS_POI Top.

    구조: {aid: {memories: [{type, day, ...}], visited: [...], knows_poi: [...], state: {...}}}
    """
    print(f"[memories] fetching ...")
    all_memories = defaultdict(list)
    # 모든 Memory type (visited / rumor / sns / policy)
    # ABOUT_POI 엣지가 없는 케이스도 포함 (rumor 일부, policy 등)
    for r in s.run("""
        MATCH (a:Agent)-[:REMEMBERS]->(m:Memory)
        WHERE a.id IN $aids
        OPTIONAL MATCH (m)-[:ABOUT_POI]->(p:POI)
        OPTIONAL MATCH (m)-[:FROM_CONVERSATION]->(c:Conversation)
        RETURN a.id AS aid, m.type AS type, toString(m.day) AS day,
               m.importance AS imp, m.satisfaction AS sat,
               m.summary AS summary, m.source AS source,
               m.topic_type AS topic_type, m.topic_value AS topic_value,
               p.id AS poi_id, p.name AS poi_name,
               c.id AS conv_id, c.intent AS conv_intent
        ORDER BY m.day DESC, m.importance DESC
    """, aids=agent_ids):
        all_memories[r["aid"]].append({
            "type": r["type"], "day": r["day"],
            "imp": r["imp"], "sat": r["sat"],
            "summary": r["summary"], "source": r["source"],
            "topic_type": r["topic_type"], "topic_value": r["topic_value"],
            "poi_id": r["poi_id"], "poi_name": r["poi_name"] or "",
            "conv_id": r["conv_id"], "conv_intent": r["conv_intent"],
        })
    # 호환성을 위해 visited만 따로도 유지
    visited = defaultdict(list)
    for aid, mems in all_memories.items():
        for m in mems:
            if m["type"] == "visited":
                visited[aid].append(m)

    print(f"  total memories: {sum(len(v) for v in all_memories.values()):,} "
          f"(visited {sum(len(v) for v in visited.values()):,})")

    # KNOWS_POI Top 20 (affinity·visit_count 높은 것)
    knows_poi = defaultdict(list)
    for r in s.run("""
        MATCH (a:Agent)-[kp:KNOWS_POI]->(p:POI)
        WHERE a.id IN $aids AND kp.visit_count > 0
        OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
        WITH a, kp, p, c
        ORDER BY kp.affinity DESC, kp.visit_count DESC
        WITH a.id AS aid, collect({
            poi_id: p.id, poi_name: coalesce(p.name,''),
            cat: c.parent, sub: c.name,
            visit_count: kp.visit_count,
            avg_sat: kp.avg_satisfaction,
            affinity: kp.affinity,
            source: kp.source,
            last_visit: toString(kp.last_visit)
        })[..20] AS top
        RETURN aid, top
    """, aids=agent_ids):
        knows_poi[r["aid"]] = r["top"]
    print(f"  KNOWS_POI top sets: {len(knows_poi):,}")

    # State (어제 mood/fatigue/balance)
    state = defaultdict(dict)
    for d in DAYS:
        for r in s.run("""
            MATCH (a:Agent)-[:HAS_STATE]->(s:State)
            WHERE a.id IN $aids AND s.day = date($d)
            RETURN a.id AS aid, s.balance AS bal, s.mood AS mood,
                   s.fatigue AS fat, s.yesterday_satisfaction AS yest_sat
        """, aids=agent_ids, d=d):
            state[r["aid"]][d] = {
                "balance": r["bal"], "mood": r["mood"],
                "fatigue": r["fat"], "yest_sat": r["yest_sat"],
            }

    # 약속 (Conversation intent='약속') — 양쪽 agent 모두에게 표시되도록 PARTICIPATES_IN으로 fetch
    appointments = defaultdict(list)
    for r in s.run("""
        MATCH (a:Agent)-[part:PARTICIPATES_IN]->(c:Conversation {intent:'약속'})
        WHERE a.id IN $aids
        OPTIONAL MATCH (c)-[:MENTIONS_POI]->(meet:POI)
        WITH a, part, c, meet,
             CASE WHEN part.role = 'initiator' THEN c.recipient_id ELSE c.initiator_id END AS counterpart
        RETURN a.id AS aid,
               c.id AS conv_id,
               toString(c.day) AS day,
               c.target_day_offset AS offset_days,
               c.target_time AS target_time,
               c.meeting_location_hint AS hint,
               meet.id AS meeting_poi_id,
               meet.name AS meeting_poi_name,
               counterpart AS with_agent,
               part.role AS role,
               c.summary AS summary
        ORDER BY c.day ASC
    """, aids=agent_ids):
        appointments[r["aid"]].append({
            "conv_id": r["conv_id"],
            "day": r["day"],
            "offset_days": r["offset_days"],
            "target_time": r["target_time"],
            "hint": r["hint"],
            "meeting_poi_id": r["meeting_poi_id"],
            "meeting_poi_name": r["meeting_poi_name"] or "",
            "with_agent": r["with_agent"],
            "role": r["role"],
            "summary": r["summary"],
        })
    print(f"  appointments: {sum(len(v) for v in appointments.values()):,}")

    return {
        aid: {
            "memories": all_memories.get(aid, []),    # 전체 (visited + rumor + sns + policy)
            "visited": visited.get(aid, []),          # 호환성: visited만
            "knows_poi": knows_poi.get(aid, []),
            "state": state.get(aid, {}),
            "appointments": appointments.get(aid, []),
        }
        for aid in agent_ids
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    args = ap.parse_args()

    global DAYS
    start = date.fromisoformat(args.start)
    DAYS = [(start + timedelta(days=i)).isoformat() for i in range(args.days)]

    print(f"=== Visualization Export ({DAYS[0]} ~ {DAYS[-1]}, OUT_DIR={OUT_DIR}) ===")
    with driver_session() as s:
        print("[agents] fetching ...")
        agents = fetch_agents(s)
        agent_ids = [a["id"] for a in agents]
        print(f"  total agents: {len(agents)}")

        frames, by_agent_events = fetch_timeline(s, agent_ids)
        memories = fetch_memories(s, agent_ids)

    # JSON dump
    (OUT_DIR / "agents.json").write_text(
        json.dumps(agents, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT_DIR / "timeline.json").write_text(
        json.dumps(frames, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT_DIR / "memories.json").write_text(
        json.dumps(memories, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT_DIR / "events.json").write_text(
        json.dumps({aid: by_agent_events[aid] for aid in agent_ids if aid in by_agent_events},
                   ensure_ascii=False, default=str), encoding="utf-8")

    # 사이즈 보고
    for name in ["agents.json", "timeline.json", "memories.json", "events.json"]:
        p = OUT_DIR / name
        print(f"  → {name}: {p.stat().st_size / 1024:.0f} KB")

    print(f"\n[done] export → {OUT_DIR}")


if __name__ == "__main__":
    main()
