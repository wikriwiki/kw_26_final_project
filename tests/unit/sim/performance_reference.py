"""Frozen pre-optimization algorithms from c684f83 for differential tests.

Keep these independent of optimized helpers: ordering, duplicate rows, sampling
state, and frame payloads are part of the compatibility contract.
"""
from collections import defaultdict


def social_pairs(work_group, home_group, rng):
    pairs = set()
    for members in work_group.values():
        if len(members) < 2:
            continue
        for a in members:
            others = [m for m in members if m != a]
            k = min(5, len(others))
            if not k:
                continue
            for b in rng.sample(others, k):
                key = (a, b) if a < b else (b, a)
                pairs.add((key, 0.6, "colleague"))
    for members in home_group.values():
        if len(members) < 2:
            continue
        for a in members:
            others = [m for m in members if m != a]
            k = min(3, len(others))
            if not k:
                continue
            for b in rng.sample(others, k):
                key = (a, b) if a < b else (b, a)
                if any(p[0] == key for p in pairs):
                    continue
                pairs.add((key, 0.4, "neighbor"))
    return pairs


def exposure(a, b, data):
    co_visits = []
    for dong_a, hr_a in data["visits"].get(a, []):
        for dong_b, hr_b in data["visits"].get(b, []):
            if dong_a == dong_b and dong_a is not None:
                if hr_a is None or hr_b is None:
                    continue
                diff = abs(hr_a - hr_b)
                if diff <= 1:
                    co_visits.append(1.0 - diff * 0.5)
    if not co_visits:
        return 0.0
    freq = min(len(co_visits), 5) / 5.0
    avg_overlap = sum(co_visits) / len(co_visits)
    return min(freq * 0.6 + avg_overlap * 0.4, 1.0)


def timeline_frames(by_agent, days):
    frames = []
    for day_idx, day_str in enumerate(days):
        for hour in range(24):
            agents = []
            for aid, events in by_agent.items():
                current = None
                for e in [e for e in events if e["day"] == day_str]:
                    if int(e["time"][:2]) <= hour:
                        current = e
                    else:
                        break
                if current is None and day_idx > 0:
                    previous = [e for e in events if e["day"] == days[day_idx - 1]]
                    if previous:
                        current = previous[-1]
                if current and current.get("lon") and current.get("lat"):
                    agents.append({
                        "id": aid, "lon": current["lon"], "lat": current["lat"],
                        "cat": current.get("cat") or "집", "l1": current.get("l1"),
                        "dong": current.get("dong"), "intent": current["intent"],
                        "sat": current["sat"], "spent": current.get("spent") or 0,
                        "anchor": current["anchor"],
                    })
            frames.append({"day_idx": day_idx, "day": day_str, "hour": hour,
                           "label": f"Day {day_idx+1} {day_str} {hour:02d}:00", "agents": agents})
    return frames
