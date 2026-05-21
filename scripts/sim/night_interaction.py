"""Night 상호작용 대상 선정 — Neo4j 포팅 (Python-side 점수 계산).

설계 출처:
  - prototype/src/night_interaction.py (in-memory 원본; 가중치·임계값·로직 유지)
  - docs/schedule_generation_plan/runtime_ontology.md §5.2 Night Phase 2

전략: Neo4j는 가벼운 read-only fetch만 (각 노드의 속성·관계 카운트).
모든 점수 계산은 Python in-memory dict로 — Neo4j 부담 최소화 + 대규모 cross-matching 회피.

InteractionScore(A, B) = 0.4 × Exposure + 0.3 × Relationship + 0.3 × Urgency
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
import math

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))
from _common import driver_session  # noqa: E402


# ═══════════════════════════════════════════
# 가중치 (prototype과 동일)
# ═══════════════════════════════════════════
W_EXPOSURE = 0.4
W_RELATION = 0.3
W_URGENCY = 0.3

THRESHOLD = 0.3
MAX_PAIRS_PER_AGENT = 2

# 시간 감쇠 상수: ln(2)/7 ≈ 0.099 → 7일 미대화 시 친밀도 반감
DECAY_LAMBDA = math.log(2) / 7.0

# 확률적 매칭 기본 온도 (낮을수록 greedy에 가까움)
DEFAULT_TEMPERATURE = 0.5


# ═══════════════════════════════════════════
# 1. Neo4j → Python 가벼운 fetch
# ═══════════════════════════════════════════
FETCH_VISITS_CYPHER = """
MATCH (a:Agent)-[:HAS_PLAN {day:date($d)}]->(:Plan)-[i:INCLUDES]->(p:POI)-[:IN_DONG]->(dd:Dong)
WHERE p.type IN ['commerce','workplace']
RETURN a.id AS aid, dd.code AS dong, i.time.hour AS hr
"""

FETCH_KNOWS_CYPHER = """
MATCH (a:Agent)-[k:KNOWS]->(b:Agent)
RETURN a.id AS aid, b.id AS bid, k.relation AS rel
"""

FETCH_CONV_HISTORY_CYPHER = """
MATCH (a:Agent)-[:PARTICIPATES_IN]->(c:Conversation)<-[:PARTICIPATES_IN]-(b:Agent)
WHERE a.id < b.id AND c.day < date($d)
RETURN a.id AS aid, b.id AS bid, count(c) AS n, max(c.day) AS last_day
"""

FETCH_STATE_CYPHER = """
MATCH (a:Agent)-[:HAS_STATE {day:date($d)}]->(s:State)
RETURN a.id AS aid, s.mood AS mood, s.fatigue AS fatigue,
       s.policy_lifecycle AS lc
"""

FETCH_INFO_MEMORY_CYPHER = """
MATCH (a:Agent)-[:REMEMBERS]->(m:Memory)
WHERE m.type IN ['policy','sns'] AND m.day >= date($d) - duration({days:7})
RETURN a.id AS aid, count(m) AS info_count
"""


def fetch_all(day: date) -> dict:
    """Night 점수 계산에 필요한 모든 데이터를 단계별로 fetch.
    각 fetch마다 새 session — connection pool 압박 회피.
    """
    data: dict = {
        "visits": defaultdict(list),
        "outed": set(),
        "knows": {},
        "conv_history": {},
        "state": {},
        "info_count": {},
    }
    t0 = time.time()

    print(f"  [fetch] visits ...")
    with driver_session() as s:
        for r in s.run(FETCH_VISITS_CYPHER, d=day.isoformat()):
            aid = r["aid"]
            data["visits"][aid].append((r["dong"], r["hr"]))
            data["outed"].add(aid)
    print(f"    visits: {sum(len(v) for v in data['visits'].values()):,} rows, outed agents: {len(data['outed']):,}")

    print(f"  [fetch] KNOWS ...")
    with driver_session() as s:
        for r in s.run(FETCH_KNOWS_CYPHER):
            a, b = sorted([r["aid"], r["bid"]])
            data["knows"][(a, b)] = r["rel"]
    print(f"    knows: {len(data['knows']):,}")

    print(f"  [fetch] Conversation history ...")
    with driver_session() as s:
        for r in s.run(FETCH_CONV_HISTORY_CYPHER, d=day.isoformat()):
            a, b = sorted([r["aid"], r["bid"]])
            last_day = r["last_day"]
            # Neo4j date → Python date 변환 (neo4j.time.Date 또는 str)
            if last_day is not None and not isinstance(last_day, date):
                try:
                    last_day = date.fromisoformat(str(last_day))
                except (ValueError, TypeError):
                    last_day = None
            data["conv_history"][(a, b)] = {
                "n": r["n"],
                "last_day": last_day,
            }
    print(f"    conv_history: {len(data['conv_history']):,}")

    print(f"  [fetch] State ...")
    with driver_session() as s:
        for r in s.run(FETCH_STATE_CYPHER, d=day.isoformat()):
            lc = r["lc"]
            knows_policy = lc is not None and lc not in ("", "{}", None)
            data["state"][r["aid"]] = {
                "mood": r["mood"] if r["mood"] is not None else 0.5,
                "fatigue": r["fatigue"] if r["fatigue"] is not None else 0.3,
                "knows_policy": knows_policy,
            }
    print(f"    state: {len(data['state']):,}")

    print(f"  [fetch] info Memory ...")
    with driver_session() as s:
        for r in s.run(FETCH_INFO_MEMORY_CYPHER, d=day.isoformat()):
            data["info_count"][r["aid"]] = r["info_count"]
    print(f"    info_count: {len(data['info_count']):,}")

    print(f"  [fetch] total {time.time()-t0:.1f}s")
    return data


# ═══════════════════════════════════════════
# 2. 후보 쌍 추출 (Python)
# ═══════════════════════════════════════════
def find_candidate_pairs(data: dict) -> set[tuple[str, str]]:
    """같은 (dong, hr) bucket + 인접 hr + KNOWS 이웃 (양쪽 외출시)."""
    # (dong, hr) bucket 구성
    bucket: dict[tuple[str, int], set[str]] = defaultdict(set)
    for aid, vs in data["visits"].items():
        for (dong, hr) in vs:
            if dong is None or hr is None: continue
            bucket[(dong, hr)].add(aid)

    # 동별 hour 분포
    by_dong: dict[str, dict[int, set[str]]] = defaultdict(dict)
    for (dong, hr), aids in bucket.items():
        by_dong[dong][hr] = aids

    pairs: set[tuple[str, str]] = set()
    for dong, by_hr in by_dong.items():
        hours = sorted(by_hr.keys())
        for hr in hours:
            here = list(by_hr[hr])
            # same hour: all pairs
            for i in range(len(here)):
                for j in range(i + 1, len(here)):
                    a, b = sorted([here[i], here[j]])
                    pairs.add((a, b))
            # adjacent hour
            if hr + 1 in by_hr:
                next_set = by_hr[hr + 1]
                for a1 in here:
                    for a2 in next_set:
                        if a1 == a2: continue
                        a, b = sorted([a1, a2])
                        pairs.add((a, b))

    # KNOWS 이웃 (양쪽 외출)
    outed = data["outed"]
    for (a, b) in data["knows"]:
        if a in outed and b in outed:
            pairs.add((a, b))
    return pairs


# ═══════════════════════════════════════════
# 3. 3축 점수 (Python — prototype 알고리즘 그대로)
# ═══════════════════════════════════════════
def calc_exposure(a: str, b: str, data: dict) -> float:
    va = data["visits"].get(a, [])
    vb = data["visits"].get(b, [])
    co_visits = []
    for (dong_a, hr_a) in va:
        for (dong_b, hr_b) in vb:
            if dong_a == dong_b and dong_a is not None:
                if hr_a is None or hr_b is None: continue
                diff = abs(hr_a - hr_b)
                if diff <= 1:
                    co_visits.append(1.0 - diff * 0.5)
    if not co_visits:
        return 0.0
    freq = min(len(co_visits), 5) / 5.0
    avg_overlap = sum(co_visits) / len(co_visits)
    return min(freq * 0.6 + avg_overlap * 0.4, 1.0)


def calc_relationship(a: str, b: str, data: dict, current_day: date | None = None) -> float:
    pair = (a, b)
    rel = data["knows"].get(pair)
    base = 0.0
    if rel == "colleague":
        base = 0.6
    elif rel == "neighbor":
        base = 0.4
    elif rel:
        base = 0.3

    hist = data["conv_history"].get(pair)
    if hist is None:
        past, last_day = 0, None
    elif isinstance(hist, dict):
        past, last_day = hist["n"], hist.get("last_day")
    else:
        # 하위 호환: 기존 int 형태
        past, last_day = hist, None

    intimacy = min(math.log(past + 1) / math.log(11), 1.0)

    # 시간 감쇠: 마지막 대화 이후 경과 일수에 따라 친밀도 감소
    if last_day is not None and current_day is not None:
        days_since = (current_day - last_day).days
        if days_since > 0:
            decay = math.exp(-DECAY_LAMBDA * days_since)
            intimacy *= decay

    return min(base * 0.5 + intimacy * 0.5, 1.0)


def calc_urgency(a: str, b: str, data: dict) -> float:
    sa = data["state"].get(a) or {"mood": 0.5, "fatigue": 0.3, "knows_policy": False}
    sb = data["state"].get(b) or {"mood": 0.5, "fatigue": 0.3, "knows_policy": False}

    urgency_a = 0.0
    urgency_b = 0.0

    # 정보 희소성 (정책 인지 비대칭)
    if sa["knows_policy"] and not sb["knows_policy"]:
        urgency_a += 0.5
    if sb["knows_policy"] and not sa["knows_policy"]:
        urgency_b += 0.5

    # 정책/SNS Memory 누적 비대칭
    ia = data["info_count"].get(a, 0)
    ib = data["info_count"].get(b, 0)
    if ia > ib:
        urgency_a += min((ia - ib) * 0.15, 0.4)
    elif ib > ia:
        urgency_b += min((ib - ia) * 0.15, 0.4)

    # 감정 임계치
    mood_a = sa["mood"]; mood_b = sb["mood"]
    emo_a = (0.8 - mood_a) if mood_a < 0.3 else max(0, mood_a - 0.7) * 1.5
    emo_b = (0.8 - mood_b) if mood_b < 0.3 else max(0, mood_b - 0.7) * 1.5
    urgency_a += min(max(emo_a, 0), 0.4)
    urgency_b += min(max(emo_b, 0), 0.4)

    # 피로 보정
    urgency_a *= (1.0 - sa["fatigue"] * 0.3)
    urgency_b *= (1.0 - sb["fatigue"] * 0.3)

    return min(max(urgency_a, urgency_b), 1.0)


# ═══════════════════════════════════════════
# 4. 메인 — 매칭 알고리즘
# ═══════════════════════════════════════════
def _softmax_select(
    scored: list[dict],
    max_pairs_per_agent: int,
    temperature: float,
    rng: random.Random,
) -> list[dict]:
    """Softmax 확률 기반 매칭: 점수가 높을수록 선택 확률이 높지만,
    낮은 점수의 쌍도 기회를 가짐 → 네트워크 다양성 확보.
    temperature → 0: greedy와 동일, temperature ↑: 균등 분포에 가까움.
    """
    remaining = list(scored)  # 복사
    cnt: dict[str, int] = defaultdict(int)
    selected: list[dict] = []

    while remaining:
        # 현재 선택 가능한 쌍만 필터
        eligible = [
            r for r in remaining
            if cnt[r["aid_a"]] < max_pairs_per_agent
            and cnt[r["aid_b"]] < max_pairs_per_agent
        ]
        if not eligible:
            break

        # Softmax 확률 계산 (수치 안정성을 위해 max 빼기)
        scores = [r["score"] / temperature for r in eligible]
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        total_exp = sum(exps)
        probs = [e / total_exp for e in exps]

        # 가중 랜덤 선택
        chosen_idx = rng.choices(range(len(eligible)), weights=probs, k=1)[0]
        chosen = eligible[chosen_idx]

        selected.append(chosen)
        cnt[chosen["aid_a"]] += 1
        cnt[chosen["aid_b"]] += 1

        # 선택된 항목 제거
        remaining.remove(chosen)

    return selected


def select_interaction_pairs(
    day: date,
    threshold: float = THRESHOLD,
    max_pairs_per_agent: int = MAX_PAIRS_PER_AGENT,
    weights: tuple[float, float, float] = (W_EXPOSURE, W_RELATION, W_URGENCY),
    stochastic: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Night Phase 2 — 후보 추출 + 3축 점수 + 매칭.

    stochastic=True (기본): Softmax 확률적 매칭 — 점수 높은 쌍이 선택될
        확률이 높지만 다양성 보장. temperature로 무작위성 조절.
    stochastic=False: 기존 그리디 매칭 (점수 내림차순 결정론적 선택).
    """
    w_e, w_r, w_u = weights
    t0 = time.time()
    if verbose:
        mode = f"stochastic(T={temperature})" if stochastic else "greedy"
        print(f"[Night] {day} 시작 (matching={mode})")

    # 1) 데이터 fetch
    data = fetch_all(day)

    # 2) 후보 쌍
    if verbose:
        print(f"  [candidates] 추출 중 ...")
    cands = find_candidate_pairs(data)
    if verbose:
        print(f"  candidates: {len(cands):,}")
    if not cands:
        return []

    # 3) 점수 계산 (시간 감쇠 적용된 calc_relationship 사용)
    if verbose:
        print(f"  [scores] 계산 중 ...")
    t1 = time.time()
    scored = []
    for (a, b) in cands:
        exp = calc_exposure(a, b, data)
        rel = calc_relationship(a, b, data, current_day=day)
        urg = calc_urgency(a, b, data)
        total = w_e * exp + w_r * rel + w_u * urg
        if total >= threshold:
            scored.append({
                "aid_a": a, "aid_b": b, "score": round(total, 4),
                "exposure": round(exp, 4),
                "relationship": round(rel, 4),
                "urgency": round(urg, 4),
            })
    if verbose:
        print(f"  scoring: {time.time()-t1:.1f}s, above threshold: {len(scored):,}")

    # 4) 매칭
    if stochastic:
        rng = random.Random(seed)
        selected = _softmax_select(scored, max_pairs_per_agent, temperature, rng)
    else:
        # 기존 그리디 매칭 (하위 호환)
        scored.sort(key=lambda x: x["score"], reverse=True)
        cnt: dict[str, int] = defaultdict(int)
        selected = []
        for r in scored:
            a, b = r["aid_a"], r["aid_b"]
            if cnt[a] < max_pairs_per_agent and cnt[b] < max_pairs_per_agent:
                selected.append(r); cnt[a] += 1; cnt[b] += 1

    if verbose:
        label = f"stochastic(T={temperature})" if stochastic else "greedy"
        print(f"  selected ({label}, max={max_pairs_per_agent}/agent): {len(selected):,}")
        print(f"  total elapsed: {time.time()-t0:.1f}s")
    return selected


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    import os as _os
    DEFAULT_SIM_OUT = _os.environ.get("SIM_OUTPUT_DIR",
                                       _os.path.expanduser("~/sim_output"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2026-05-02")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_AGENT)
    ap.add_argument("--greedy", action="store_true",
                    help="확률적 매칭 대신 기존 그리디 매칭 사용")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help=f"Softmax 온도 (기본 {DEFAULT_TEMPERATURE}). "
                         "낮을수록 점수 상위 쌍 집중, 높을수록 다양성 증가")
    ap.add_argument("--seed", type=int, default=None,
                    help="확률적 매칭 시드 (재현성 확보용)")
    ap.add_argument("--dump", default=None,
                    help=f"dump path. 기본: $SIM_OUTPUT_DIR/interactions_<day>.json "
                         f"(현재 {DEFAULT_SIM_OUT})")
    args = ap.parse_args()
    if args.dump is None:
        args.dump = f"{DEFAULT_SIM_OUT}/interactions_{args.day}.json"

    day = date.fromisoformat(args.day)
    pairs = select_interaction_pairs(
        day, threshold=args.threshold,
        max_pairs_per_agent=args.max_pairs,
        stochastic=not args.greedy,
        temperature=args.temperature,
        seed=args.seed,
    )

    print(f"\n=== Top 15 상호작용 쌍 ({day}) ===")
    for r in pairs[:15]:
        print(f"  {r['aid_a']:35} ↔ {r['aid_b']:35} score={r['score']:.3f} "
              f"[exp={r['exposure']:.2f} rel={r['relationship']:.2f} urg={r['urgency']:.2f}]")

    print(f"\n=== 점수 분포 ===")
    if pairs:
        scores = [r["score"] for r in pairs]
        print(f"  count={len(scores):,}, min={min(scores):.3f}, max={max(scores):.3f}, "
              f"mean={sum(scores)/len(scores):.3f}")

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump).write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[dump] → {args.dump}")
