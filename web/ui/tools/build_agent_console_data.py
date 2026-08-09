#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
에이전트 1:1 화면(`/runs/:runId/agents`)이 읽을 데이터를 미리 잘라 둔다.

왜 필요한가
-----------
원본은 `web/viz_store/demo/` 에 있고 합계 110MB 가 넘는다.

    agents.json    1.5MB   1,825명 프로필
    memories.json  30.2MB  1,825명분 기억 48,474건
    events.json    44.2MB  1,823명분 활동 81,157건
    timeline.json  36.5MB  120프레임(5일 × 24시간) 시간대별 위치

화면이 쓰는 것은 **한 번에 한 명**이다. 통째로 import 하면 번들이 100MB 를 넘고
브라우저가 멈춘다(DESIGN_SPEC B5). 그래서 여기서 두 갈래로 미리 자른다.

    public/agent-console/roster.json        목록·필터·검색용 (한 명당 11개 필드)
    public/agent-console/agents/<idx>.json  그 한 명의 활동·기억·상태 전부

`public/` 은 Vite 가 손대지 않고 그대로 내보내는 자리라, 개발 서버와 빌드 산출물
양쪽에서 같은 주소(`/agent-console/...`)로 받을 수 있다. vite.config.ts 를 고칠 일이 없다.

timeline.json 을 쓰지 않는 이유
-------------------------------
timeline 은 events 를 시간대별로 다시 담은 것이다. 프레임 8(day0 08:00)의 1,823명을
events 에서 "그 시각 이전 마지막 활동"으로 재구성해 200명 표본을 대조한 결과
intent·좌표가 전부 일치했다(불일치 0). 같은 기록을 두 번 싣지 않는다 —
"그 시각에 어디 있었나"는 events 에서 그대로 답이 나온다.

실행
----
    python web/ui/tools/build_agent_console_data.py

원본은 읽기만 한다. 출력은 `web/ui/public/agent-console/` 아래에만 쓴다.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
UI = HERE.parent
SRC = (UI / ".." / "viz_store" / "demo").resolve()
OUT = UI / "public" / "agent-console"

# 원본 memories.json 의 visited 요약문은 인코딩이 깨져 있다 ("(직장) ??, ??? 0.69").
# 46,034건 중 18,876건. 깨진 문자열을 화면에 그대로 싣지 않는다 — 대신 구조화된
# 필드(날짜·장소·만족도·중요도)로 답하고, 몇 건이 손상됐는지 화면에 적는다.
DAMAGED = "?"

# poi_id 앞머리가 장소 종류를 말한다. 집·직장은 원본에 이름이 없어서(`poi_name: ""`)
# 이 값이 없으면 화면이 "이름 없는 어딘가"라고밖에 못 쓴다.
PLACE_KIND = {"R": "residence", "W": "workplace", "C": "commerce"}


def kind_of(poi_id: str | None) -> str | None:
    if not poi_id:
        return None
    return PLACE_KIND.get(poi_id[:1])


def label_in_parens(summary: str) -> str | None:
    """
    요약문 "상구네백반(한식) ??, ??? 0.68" 에서 괄호 안의 분류를 꺼낸다.
    글자가 깨진 요약문에서도 괄호 부분은 멀쩡해서, 손상된 기록의 분류를 여기서 되살린다.
    """
    a = summary.find("(")
    b = summary.find(")", a + 1)
    if a < 0 or b < 0:
        return None
    inner = summary[a + 1 : b].strip()
    return inner or None


def load(name: str):
    with (SRC / name).open(encoding="utf-8") as fh:
        return json.load(fh)


def jdump(path: Path, obj) -> int:
    """separators 로 공백을 없앤다. 사람이 읽을 파일이 아니라 화면이 받을 파일이다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def main() -> None:
    agents = load("agents.json")
    events = load("events.json")
    memories = load("memories.json")

    # 기록에 등장하는 날짜. 활동 기록에서만 뽑는다 — 달력을 지어내지 않는다.
    days = sorted({e["day"] for evs in events.values() for e in evs})
    day_ix = {d: i for i, d in enumerate(days)}

    # 소비 분위는 원본에 없다. 평일 하루 소비 예산(daily_wd) 순위로 여기서 만든다.
    # 시뮬레이션이 쓴 분위가 아니라 **이 화면이 계산한 값**이므로 화면에도 그렇게 적는다.
    ranked = sorted(agents, key=lambda a: a.get("daily_wd") or 0)
    decile = {}
    n = len(ranked)
    for pos, a in enumerate(ranked):
        decile[a["id"]] = min(10, pos * 10 // n + 1)

    if OUT.exists():
        shutil.rmtree(OUT)

    roster = []
    total_detail = 0

    for idx, prof in enumerate(agents):
        aid = prof["id"]
        raw_events = events.get(aid, [])
        mem = memories.get(aid) or {}

        # --- 활동 -------------------------------------------------------------
        evs = []
        for e in sorted(raw_events, key=lambda e: (e["day"], e["ord"])):
            evs.append(
                {
                    "d": day_ix[e["day"]],
                    "t": e["time"],
                    "cat": e["cat"],
                    "sub": e["sub"],
                    "l1": e["l1"],
                    "intent": e["intent"],
                    "sat": e["sat"],
                    "spent": e["spent"],
                    "trg": e["trigger"],
                    "why": e["reasoning"],
                    "pick": e["pick_reason"],
                    "pf": e["pick_factor"],
                    "poi": e["poi_name"],
                    "dong": e["dong"],
                    "pt": e["poi_type"],
                }
            )

        # --- 아는 장소 (기억에서 만들어진 장소별 누적) -------------------------
        knows = [
            {
                "poi": k["poi_name"],
                "cat": k["cat"],
                "sub": k["sub"],
                "pt": kind_of(k.get("poi_id")),
                "n": k["visit_count"],
                "sat": k["avg_sat"],
                "aff": k["affinity"],
                "last": k["last_visit"],
                "src": k["source"],
            }
            for k in (mem.get("knows_poi") or [])
        ]
        by_poi = {k["poi_id"]: k for k in (mem.get("knows_poi") or [])}

        # --- 기억 -------------------------------------------------------------
        visited, rumors, damaged = [], [], 0
        for m in mem.get("memories") or []:
            if m["type"] == "rumor":
                rumors.append(
                    {
                        "d": day_ix.get(m["day"], None),
                        "day": m["day"],
                        "imp": m["imp"],
                        "s": m["summary"],
                        "src": m["source"],
                        "tt": m["topic_type"],
                        "tv": m["topic_value"],
                        "ci": m["conv_intent"],
                    }
                )
                continue
            summary = m.get("summary") or ""
            broken = DAMAGED in summary
            if broken:
                damaged += 1
            ref = by_poi.get(m.get("poi_id")) or {}
            visited.append(
                {
                    "day": m["day"],
                    "imp": m["imp"],
                    "sat": m["sat"],
                    "poi": m["poi_name"] or (ref.get("poi_name") or ""),
                    # 분류는 knows_poi 를 먼저 보고, 없으면 요약문 괄호에서 되살린다
                    "cat": ref.get("cat") or label_in_parens(summary),
                    "sub": ref.get("sub"),
                    "pt": kind_of(m.get("poi_id")),
                    # 깨지지 않은 요약문만 싣는다
                    "s": None if broken else summary,
                }
            )

        detail = {
            "id": aid,
            "idx": idx,
            "profile": prof,
            "days": days,
            "events": evs,
            "visited": visited,
            "visitedDamaged": damaged,
            "rumors": rumors,
            "knows": knows,
            "state": mem.get("state") or {},
            "appointments": mem.get("appointments") or [],
        }
        total_detail += jdump(OUT / "agents" / f"{idx:04d}.json", detail)

        spent = sum(e["spent"] or 0 for e in raw_events)
        roster.append(
            {
                "i": idx,
                "id": aid,
                "gu": prof["district"],
                "dong": prof["home_dong"],
                "age": prof["age"],
                "sex": prof["gender"],
                "inc": prof["income"],
                "job": prof["job"],
                "dec": decile[aid],
                "wd": prof["daily_wd"],
                # 목록 한 줄에서 "기록이 있는가"를 바로 보이기 위한 두 수치
                "ev": len(raw_events),
                "spent": spent,
            }
        )

    meta = {
        "source": "web/viz_store/demo/{agents,events,memories}.json",
        "generatedFrom": {
            "agents": len(agents),
            "eventAgents": len(events),
            "events": sum(len(v) for v in events.values()),
            "memoryAgents": len(memories),
        },
        "days": days,
        "decileBasis": "daily_wd",
    }
    size = jdump(OUT / "roster.json", {"meta": meta, "items": roster})

    print(f"roster.json          {size/1024:8.1f} KB  ({len(roster)}명)")
    print(f"agents/*.json  총    {total_detail/1024/1024:8.1f} MB  (1인 평균 {total_detail/len(roster)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
