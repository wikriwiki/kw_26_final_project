"""정책 본런을 콘솔의 '실행'으로 세운다 — 정책마다 개요·모니터·결과·3D 지도·시민 문답·보고서를 보게.

한 정책 = 한 실행. 정책 전 기간과 정책 있음 기간을 이어 붙인 것이다(실제로 시민이 겪은 순서).
정책 없음 쪽은 실험 화면(같은 시민·같은 날 비교)에서 본다.

입력 (모두 읽기만 한다)
* 실험 정의 ``web/experiments/<실험>.json`` — ``run_id``·``source``·팔
* 실험 미러 ``<SIM_EXPERIMENTS_ROOT>/<실험>/arms/{shared-pre,post-on}/metrics/day_*.jsonl``
  (``import_policy_runs.py`` 가 꺼내 둔 일자 기록)
* ``extract_policy_runs.py`` 가 꺼내 둔 ``<run_src>/<실험>/{pre,on}/``: 일자 요약·단계 시간·체크포인트·
  응답 오류 기록, ``on/dossier.jsonl``(사람별 계획·기억·상태, 정책 전 기간 포함)
* 장소 표 ``poi_all.txt``·시민 거주지/직장 표 ``agent_anchor.txt`` — 본런이 쓴 그래프 덤프에서
  읽기 전용으로 뽑은 것 (cypher-shell --format plain)

출력
* ``<runs>/out_<run_id>/`` — 콘솔이 '실행'으로 읽는 폴더. 일자 기록은 하드링크(복사하지 않는다).
  ``summary.json`` 은 하루씩 돌린 본런의 일자 요약(``day_*.json``)을 모아 만든다.
  ``events.jsonl`` 은 dossier 의 결제 계획 항목(결제 완료분)에서 만든다.
* ``<viz_src>/<run_id>/{agents,timeline,memories,events}.json`` — 3D 지도 팩·시민 문답 데이터의 원본.
  ``export_visualization.py`` 가 Neo4j 에서 만들던 것과 같은 모양이다. 약속(대화) 기록은 dossier 에
  없어서 싣지 않는다(``appointments`` 키를 두지 않는다 — '없음'과 구별).

    python web/tools/build_policy_runs.py --run-src <폴더> --ref <장소 표 폴더> --runs <폴더> --viz-src <폴더>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DAY_FILE = re.compile(r"^day_(\d{4}-\d{2}-\d{2})\.json$")
MEM_POI = re.compile(r"_([CRW]_[A-Za-z0-9]+)_\d{4}-\d{2}-\d{2}")


def experiments_root() -> Path:
    if os.environ.get("SIM_EXPERIMENTS_ROOT"):
        return Path(os.environ["SIM_EXPERIMENTS_ROOT"])
    if os.environ.get("SIM_DATA_ROOT"):
        return Path(os.environ["SIM_DATA_ROOT"]) / "experiments"
    return Path.home() / "sim_data" / "experiments"


def plain_rows(path: Path):
    """cypher-shell --format plain 출력: 따옴표 CSV, 구분자 뒤 공백, 없음은 NULL."""
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, skipinitialspace=True):
            yield {k: (None if v == "NULL" else v) for k, v in row.items()}


def num(value, cast=float):
    if value in (None, "", "None", "null"):
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def hhmm(value: str | None) -> str:
    return (value or "00:00")[:5]


# ─────────────────────────────────────────────────────────────
# 실행 폴더
# ─────────────────────────────────────────────────────────────
def build_run_folder(definition: dict, src: Path, out: Path, mirror: Path) -> dict:
    pre_arm = next(a for a in definition["arms"] if a["role"] == "shared_pre")
    on_arm = next(a for a in definition["arms"] if a["role"] == "treatment")
    out.mkdir(parents=True, exist_ok=True)

    days: list[str] = []
    for arm_id, arm_src in ((pre_arm["arm_id"], src / "pre"), (on_arm["arm_id"], src / "on")):
        for path in sorted((mirror / "arms" / arm_id / "metrics").glob("day_*.jsonl")):
            link_or_copy(path, out / "metrics" / path.name)
            days.append(path.name[4:-6])
        for sub in ("timing", "checkpoints"):
            for path in sorted((arm_src / sub).glob("*")):
                link_or_copy(path, out / sub / path.name)
    days = sorted(set(days))

    # 일자 요약: 하루씩 돌린 본런이라 summary.json 은 마지막 날만 담는다. day_*.json 을 모은다.
    summaries = []
    for arm_src in (src / "pre", src / "on"):
        for path in sorted(arm_src.glob("day_*.json")):
            if DAY_FILE.match(path.name):
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
    summaries.sort(key=lambda s: s.get("day") or "")
    last = json.loads((src / "on" / "summary.json").read_text(encoding="utf-8"))
    workers = (last.get("args") or {}).get("workers")
    summary = {
        "summary": summaries,
        "args": {"start": days[0], "days": len(days), "limit": definition.get("cohort_size"), "workers": workers},
        "completed_at": last.get("completed_at"),
        "updated_at": last.get("completed_at"),
        "assembled_from": "본런 일자 요약 day_*.json (정책 전 + 정책 있음)",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    with (out / "stage1_failures.jsonl").open("wb") as dst:
        for arm_src in (src / "pre", src / "on"):
            path = arm_src / "stage1_failures.jsonl"
            if path.exists():
                with path.open("rb") as fh:
                    shutil.copyfileobj(fh, dst, 1 << 20)

    meta = {
        "run_id": definition["run_id"],
        "experiment_id": definition["experiment_id"],
        "title": definition.get("title"),
        "policy_id": definition.get("policy_id"),
        "effective_date": (definition.get("window") or {}).get("effective_date"),
        "cohort_size": definition.get("cohort_size"),
        "days": days,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


# ─────────────────────────────────────────────────────────────
# dossier → 결제 기록·3D 지도·시민 문답 원본
# ─────────────────────────────────────────────────────────────
def build_from_dossier(definition: dict, src: Path, out: Path, viz_out: Path, pois: dict, anchors: dict, days: list[str]) -> dict:
    viz_out.mkdir(parents=True, exist_ok=True)
    day_set = set(days)
    agents, memories, events = [], {}, {}
    totals = Counter()
    with (src / "on" / "dossier.jsonl").open(encoding="utf-8") as fh, (out / "events.jsonl").open("w", encoding="utf-8") as ev_out:
        for line in fh:
            d = json.loads(line)
            aid = d["aid"]
            prof = d.get("profile") or {}
            anchor = anchors.get(aid) or {}
            plan_events = []
            has_appt = False
            for plan in d.get("plans") or []:
                day = str(plan.get("day"))
                if day not in day_set:
                    continue
                for it in sorted(plan.get("items") or [], key=lambda x: (hhmm(x.get("time")), x.get("order") or 0)):
                    pid = it.get("poi_id") or ""
                    poi = pois.get(pid) or {}
                    if it.get("trigger") in ("약속", "appointment") or (it.get("with_agents") not in (None, "", "[]")):
                        has_appt = True
                    purchased = it.get("purchase_status") == "purchased"
                    spent = num(it.get("actual_spent")) or 0
                    l1 = poi.get("l1") or it.get("category")
                    plan_events.append({
                        "day": day, "ord": it.get("order"), "time": hhmm(it.get("time")),
                        "anchor": "residence" if pid.startswith("R_") else "workplace" if pid.startswith("W_") else f"zone:{it.get('dong_code') or ''}",
                        "cat": it.get("category"), "sub": it.get("sub_category"), "intent": it.get("intent"),
                        "sat": num(it.get("actual_satisfaction")), "spent": spent if purchased else 0,
                        "reasoning": it.get("reasoning"), "trigger": it.get("trigger"),
                        "pick_reason": it.get("pick_reason"), "pick_factor": None,
                        "poi_id": pid, "poi_name": poi.get("name") or it.get("poi") or "",
                        "lon": num(poi.get("lon")), "lat": num(poi.get("lat")),
                        "poi_type": poi.get("type"), "l1": l1, "dong": poi.get("dong"),
                    })
                    if purchased and spent > 0:
                        ev_out.write(json.dumps({
                            "aid": aid, "day": day, "time": hhmm(it.get("time")), "day_type": plan.get("day_type"),
                            "poi_id": pid, "cat": it.get("category"), "sub": it.get("sub_category"), "l1": l1,
                            "amt": spent, "sp": it.get("spent_from_policy") or "{}",
                            "rebate": it.get("policy_rebate") or "{}", "disc": it.get("discount_total"),
                            "elig": bool(it.get("coupon_eligible")),
                            # 엔진 기록에 '정책이 없었다면 안 썼을 돈'은 없다. 0 이 아니라 비워 둔다
                            "ex": None, "wba": None,
                        }, ensure_ascii=False) + "\n")
                        totals["events"] += 1
                        totals["amt"] += spent
            plan_events.sort(key=lambda e: (e["day"], e["time"], e["ord"] or 0))
            events[aid] = plan_events

            agents.append({
                "id": aid,
                "age": prof.get("age_group"), "age_years": num(prof.get("age"), int), "gender": prof.get("gender"),
                "income": prof.get("income_level"), "life_stage": prof.get("life_stage"),
                "job": prof.get("job"), "tendency": prof.get("spending_tendency"),
                "daily_wd": num(prof.get("s_daily_wd"), int), "daily_we": None,
                "top_wd": prof.get("spending_top_wd_json"), "lifestyle": prof.get("lifestyle"),
                "district": prof.get("gu"), "dist_code": aid.split("_")[1][:5] if aid.count("_") >= 1 else None,
                "home_dong": prof.get("residence_dong"),
                "home_poi_id": anchor.get("home_id"), "home_poi_name": anchor.get("home_name"),
                "home_lon": num(anchor.get("home_lon")), "home_lat": num(anchor.get("home_lat")),
                "work_poi_id": anchor.get("work_id"), "work_poi_name": anchor.get("work_name"),
                "work_lon": num(anchor.get("work_lon")), "work_lat": num(anchor.get("work_lat")),
                "commute": num(prof.get("commute_min"), int), "has_appointment": has_appt,
            })

            mems, visited = [], []
            for m in d.get("memories") or []:
                match = MEM_POI.search(m.get("id") or "")
                pid = match.group(1) if match else None
                item = {
                    "type": m.get("type"), "day": m.get("day"),
                    "imp": num(m.get("importance")), "sat": num(m.get("satisfaction")),
                    "summary": m.get("summary"), "source": None,
                    "topic_type": None, "topic_value": None,
                    "poi_id": pid, "poi_name": m.get("store") or (pois.get(pid) or {}).get("name") or "",
                    "conv_id": None, "conv_intent": None,
                }
                mems.append(item)
                if item["type"] == "visited":
                    visited.append(item)
            mems.sort(key=lambda x: (x["day"] or "", x["imp"] or 0), reverse=True)
            by_poi: dict[str, dict] = {}
            for v in visited:
                if not v["poi_id"]:
                    continue
                k = by_poi.setdefault(v["poi_id"], {"poi_id": v["poi_id"], "poi_name": v["poi_name"], "n": 0, "sat": [], "last": v["day"]})
                k["n"] += 1
                if v["sat"] is not None:
                    k["sat"].append(v["sat"])
                k["last"] = max(k["last"] or "", v["day"] or "")
            knows = sorted(by_poi.values(), key=lambda k: -k["n"])[:20]
            state = {}
            for s in d.get("states") or []:
                state[str(s.get("day"))] = {
                    "balance": num(s.get("balance")), "mood": num(s.get("mood")),
                    "fatigue": num(s.get("fatigue")), "yest_sat": num(s.get("yesterday_satisfaction")),
                }
            memories[aid] = {
                "memories": mems, "visited": visited,
                "knows_poi": [{
                    "poi_id": k["poi_id"], "poi_name": k["poi_name"],
                    "cat": (pois.get(k["poi_id"]) or {}).get("l1"), "sub": (pois.get(k["poi_id"]) or {}).get("cat"),
                    "visit_count": k["n"], "avg_sat": (sum(k["sat"]) / len(k["sat"])) if k["sat"] else None,
                    "affinity": None, "source": "visited", "last_visit": k["last"],
                } for k in knows],
                "state": state,
            }

    timeline = build_timeline(agents, events, days)
    for name, payload in (("agents.json", agents), ("timeline.json", timeline), ("memories.json", memories), ("events.json", events)):
        (viz_out / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"agents": len(agents), "frames": len(timeline), **totals}


def build_timeline(agents: list[dict], events: dict, days: list[str]) -> list[dict]:
    """시간대별 위치 — export_visualization.fetch_timeline 과 같은 규칙.

    그 시각까지 시작한 마지막 일정의 장소에 있다. 그날 아직 일정이 없으면 전날 마지막 일정,
    그것도 없으면(첫날 이른 시각) 집.
    """
    by_agent_day = {a["id"]: defaultdict(list) for a in agents}
    for aid, evs in events.items():
        for e in evs:
            if e["lon"] is not None and e["lat"] is not None:
                by_agent_day[aid][e["day"]].append(e)
    home = {a["id"]: a for a in agents}
    frames = []
    for day_idx, day in enumerate(days):
        for hour in range(24):
            frame_agents = []
            for aid in by_agent_day:
                current = None
                for e in by_agent_day[aid].get(day, []):
                    if int(e["time"][:2]) <= hour:
                        current = e
                    else:
                        break
                if current is None and day_idx > 0:
                    prev = by_agent_day[aid].get(days[day_idx - 1]) or []
                    current = prev[-1] if prev else None
                if current is not None:
                    frame_agents.append({
                        "id": aid, "lon": current["lon"], "lat": current["lat"],
                        "cat": current.get("cat") or "집", "l1": current.get("l1"), "dong": current.get("dong"),
                        "intent": current.get("intent"), "sat": current.get("sat"),
                        "spent": current.get("spent") or 0, "anchor": current.get("anchor"),
                    })
                elif home[aid].get("home_lon") is not None:
                    frame_agents.append({
                        "id": aid, "lon": home[aid]["home_lon"], "lat": home[aid]["home_lat"],
                        "cat": "집", "l1": None, "dong": home[aid].get("home_dong"),
                        "intent": None, "sat": None, "spent": 0, "anchor": "residence",
                    })
            frames.append({"day_idx": day_idx, "day": day, "hour": hour,
                           "label": f"{day_idx + 1}일째 {day} {hour:02d}:00", "agents": frame_agents})
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-src", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path, help="poi_all.txt · agent_anchor.txt 가 있는 폴더")
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--viz-src", required=True, type=Path)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    pois = {r["id"]: r for r in plain_rows(args.ref / "poi_all.txt")}
    anchors = {r["id"]: r for r in plain_rows(args.ref / "agent_anchor.txt")}
    print(f"장소 {len(pois):,} · 시민 {len(anchors):,}", flush=True)
    for path in sorted((REPO / "web" / "experiments").glob("*.json")):
        definition = json.loads(path.read_text(encoding="utf-8"))
        if not definition.get("run_id") or not definition.get("source"):
            continue
        if args.only and definition["experiment_id"] not in args.only:
            continue
        exp_id, run_id = definition["experiment_id"], definition["run_id"]
        src = args.run_src / exp_id
        out = args.runs / f"out_{run_id}"
        meta = build_run_folder(definition, src, out, experiments_root() / exp_id)
        stats = build_from_dossier(definition, src, out, args.viz_src / run_id, pois, anchors, meta["days"])
        # 보고서 스냅숏이 요구하는 파일. 장소 수는 본런 그래프 덤프의 표에서 센다.
        # 사용 가능 매장 수는 그 표에 없어 지어내지 않고 비운다.
        (out / "poi_summary.json").write_text(
            json.dumps({"poi_total": len(pois), "poi_eligible": None}, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{exp_id} → {out.name}: {len(meta['days'])}일, {stats}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
