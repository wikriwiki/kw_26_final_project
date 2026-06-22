from __future__ import annotations

from datetime import date, timedelta
from typing import Any


def lerp_position(a: list[float], b: list[float], t: float) -> list[float]:
    t = max(0.0, min(1.0, float(t)))
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def build_viz_meta(
    agents: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    memories: dict[str, dict[str, Any]],
    events: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    frame_by_day_hour = _frame_index_by_day_hour(timeline)
    frame_positions = _frame_position_lookup(timeline)
    poi_index = _poi_from_events(events)
    frame_summaries = _frame_summaries(timeline)

    return {
        "agent_count": len(agents),
        "frame_summaries": frame_summaries,
        "poi_index": poi_index,
        "spend_bursts": _build_spend_bursts(events, frame_by_day_hour),
        "meetups": _build_meetups(memories, poi_index, frame_by_day_hour, frame_positions),
        "rumor_edges": _build_rumor_edges(memories, frame_by_day_hour),
    }


def _frame_summaries(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for frame in timeline:
        active = frame.get("agents") or []
        summaries.append(
            {
                "day": frame.get("day"),
                "hour": frame.get("hour"),
                "label": frame.get("label"),
                "active_agents": len(active),
                "spend_total": sum(_int_or_zero(agent.get("spent")) for agent in active),
                "spenders": sum(1 for agent in active if _int_or_zero(agent.get("spent")) > 0),
            }
        )
    return summaries


def _poi_from_events(events: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    poi_index: dict[str, dict[str, Any]] = {}
    for agent_events in events.values():
        for event in agent_events:
            poi_id = event.get("poi_id")
            lon = event.get("lon")
            lat = event.get("lat")
            if not poi_id or lon is None or lat is None:
                continue

            poi = poi_index.setdefault(
                str(poi_id),
                {
                    "id": str(poi_id),
                    "name": event.get("poi_name") or str(poi_id),
                    "lon": float(lon),
                    "lat": float(lat),
                    "cat": event.get("cat") or event.get("l1") or "기타",
                    "l1": event.get("l1") or event.get("cat") or "기타",
                    "visit_count": 0,
                    "spent_total": 0,
                },
            )
            poi["visit_count"] += 1
            poi["spent_total"] += _int_or_zero(event.get("spent"))
    return poi_index


def _build_spend_bursts(
    events: dict[str, list[dict[str, Any]]],
    frame_by_day_hour: dict[tuple[str, int], int],
) -> list[dict[str, Any]]:
    bursts = []
    for agent_id, agent_events in events.items():
        for event in agent_events:
            amount = _int_or_zero(event.get("spent"))
            if amount <= 0 or event.get("lon") is None or event.get("lat") is None:
                continue

            hour = _hour_from_time(event.get("time"))
            bursts.append(
                {
                    "agent_id": agent_id,
                    "frame_index": _frame_for_day_hour(
                        frame_by_day_hour,
                        str(event.get("day") or ""),
                        hour,
                    ),
                    "day": event.get("day"),
                    "hour": hour,
                    "amount": amount,
                    "sat": event.get("sat"),
                    "poi_id": event.get("poi_id"),
                    "poi_name": event.get("poi_name") or "",
                    "cat": event.get("cat") or event.get("l1") or "기타",
                    "lon": float(event["lon"]),
                    "lat": float(event["lat"]),
                }
            )

    bursts.sort(key=lambda item: (item["frame_index"], item["agent_id"], item["poi_id"] or ""))
    return bursts


def _build_meetups(
    memories: dict[str, dict[str, Any]],
    poi_index: dict[str, dict[str, Any]],
    frame_by_day_hour: dict[tuple[str, int], int],
    frame_positions: dict[tuple[int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    poi_by_name = {_normalize_name(poi["name"]): poi for poi in poi_index.values() if poi.get("name")}
    meetups: dict[str, dict[str, Any]] = {}

    for agent_id, memory_bundle in memories.items():
        for appointment in memory_bundle.get("appointments") or []:
            conv_id = appointment.get("conv_id")
            other_agent_id = appointment.get("with_agent")
            if not conv_id or not other_agent_id:
                continue

            day = _target_day(appointment)
            if not _has_frame_day(frame_by_day_hour, day):
                continue
            hour = _hour_from_time(appointment.get("target_time"))
            frame_index = _frame_for_day_hour(frame_by_day_hour, day, hour)
            resolved = _resolve_appointment_location(
                appointment,
                poi_index,
                poi_by_name,
                frame_positions,
                frame_index,
                agent_id,
                other_agent_id,
            )
            if resolved is None:
                continue

            lon, lat, poi_id, poi_name = resolved
            meetups[str(conv_id)] = {
                "conv_id": conv_id,
                "frame_index": frame_index,
                "day": day,
                "hour": hour,
                "agents": sorted([agent_id, other_agent_id]),
                "lon": float(lon),
                "lat": float(lat),
                "poi_id": poi_id,
                "poi_name": poi_name,
                "summary": appointment.get("summary") or "",
            }

    return sorted(meetups.values(), key=lambda item: (item["frame_index"], item["conv_id"]))


def _resolve_appointment_location(
    appointment: dict[str, Any],
    poi_index: dict[str, dict[str, Any]],
    poi_by_name: dict[str, dict[str, Any]],
    frame_positions: dict[tuple[int, str], dict[str, Any]],
    frame_index: int,
    agent_id: str,
    other_agent_id: str,
) -> tuple[float, float, str | None, str] | None:
    meeting_poi_id = appointment.get("meeting_poi_id")
    if meeting_poi_id and str(meeting_poi_id) in poi_index:
        poi = poi_index[str(meeting_poi_id)]
        return poi["lon"], poi["lat"], poi["id"], poi["name"]

    for value in (appointment.get("meeting_poi_name"), appointment.get("hint")):
        poi = poi_by_name.get(_normalize_name(value))
        if poi is not None:
            return poi["lon"], poi["lat"], poi["id"], poi["name"]

    first = frame_positions.get((frame_index, agent_id))
    second = frame_positions.get((frame_index, other_agent_id))
    if not first or not second:
        return None
    if first.get("lon") is None or first.get("lat") is None:
        return None
    if second.get("lon") is None or second.get("lat") is None:
        return None

    lon, lat = lerp_position(
        [float(first["lon"]), float(first["lat"])],
        [float(second["lon"]), float(second["lat"])],
        0.5,
    )
    return lon, lat, None, appointment.get("meeting_poi_name") or appointment.get("hint") or "약속 장소"


def _build_rumor_edges(
    memories: dict[str, dict[str, Any]],
    frame_by_day_hour: dict[tuple[str, int], int],
) -> list[dict[str, Any]]:
    edges = []
    for target_id, memory_bundle in memories.items():
        for memory in memory_bundle.get("memories") or []:
            source_id = memory.get("source")
            if memory.get("type") != "rumor" or not source_id:
                continue

            day = str(memory.get("day") or "")
            edges.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "frame_index": _frame_for_day_hour(frame_by_day_hour, day, 12),
                    "day": day,
                    "topic_type": memory.get("topic_type") or "",
                    "topic_value": memory.get("topic_value") or "",
                    "poi_id": memory.get("poi_id"),
                    "poi_name": memory.get("poi_name") or "",
                }
            )

    edges.sort(key=lambda item: (item["frame_index"], item["source_id"], item["target_id"]))
    return edges


def _frame_index_by_day_hour(timeline: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    return {
        (str(frame.get("day") or ""), int(frame.get("hour") or 0)): index
        for index, frame in enumerate(timeline)
    }


def _frame_position_lookup(timeline: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    lookup = {}
    for frame_index, frame in enumerate(timeline):
        for item in frame.get("agents") or []:
            if item.get("id"):
                lookup[(frame_index, item["id"])] = item
    return lookup


def _frame_for_day_hour(
    frame_by_day_hour: dict[tuple[str, int], int],
    day: str,
    preferred_hour: int,
) -> int:
    if (day, preferred_hour) in frame_by_day_hour:
        return frame_by_day_hour[(day, preferred_hour)]

    candidates = [
        (abs(hour - preferred_hour), index)
        for (candidate_day, hour), index in frame_by_day_hour.items()
        if candidate_day == day
    ]
    if candidates:
        return min(candidates)[1]
    return 0


def _has_frame_day(frame_by_day_hour: dict[tuple[str, int], int], day: str) -> bool:
    return any(candidate_day == day for candidate_day, _hour in frame_by_day_hour)


def _target_day(appointment: dict[str, Any]) -> str:
    base_day = appointment.get("day")
    if not base_day:
        return ""

    offset = appointment.get("offset_days")
    if offset is None:
        return str(base_day)

    return (date.fromisoformat(str(base_day)) + timedelta(days=int(offset))).isoformat()


def _hour_from_time(value: str | None) -> int:
    if not value:
        return 12
    try:
        return max(0, min(23, int(str(value)[:2])))
    except ValueError:
        return 12


def _normalize_name(value: str | None) -> str:
    return "".join(ch for ch in (value or "").casefold() if ch.isalnum())


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
