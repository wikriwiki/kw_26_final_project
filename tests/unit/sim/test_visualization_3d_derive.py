from scripts.sim.visualization_3d.derive import build_viz_meta, lerp_position


AGENTS = [
    {
        "id": "AGT_A",
        "dist_code": "11110",
        "home_lon": 126.97,
        "home_lat": 37.57,
        "home_poi_name": "집A",
    },
    {
        "id": "AGT_B",
        "dist_code": "11140",
        "home_lon": 126.99,
        "home_lat": 37.56,
        "home_poi_name": "집B",
    },
]

TIMELINE = [
    {"day": "2026-05-01", "hour": 0, "label": "Day 1 00:00", "agents": []},
    {
        "day": "2026-05-01",
        "hour": 12,
        "label": "Day 1 12:00",
        "agents": [
            {
                "id": "AGT_A",
                "lon": 126.981,
                "lat": 37.566,
                "cat": "식사",
                "l1": "식사",
                "spent": 12000,
                "sat": 0.8,
                "anchor": "lunch",
            },
            {
                "id": "AGT_B",
                "lon": 126.982,
                "lat": 37.567,
                "cat": "카페",
                "l1": "카페",
                "spent": 0,
                "sat": 0.7,
                "anchor": "meeting",
            },
        ],
    },
]

MEMORIES = {
    "AGT_A": {
        "appointments": [
            {
                "conv_id": "conv_1",
                "day": "2026-05-01",
                "offset_days": 0,
                "target_time": "12:00",
                "hint": "테스트카페",
                "meeting_poi_id": None,
                "meeting_poi_name": "",
                "with_agent": "AGT_B",
                "role": "initiator",
                "summary": "점심 약속",
            }
        ],
        "memories": [
            {
                "type": "rumor",
                "day": "2026-05-01",
                "source": "AGT_B",
                "topic_type": "policy",
                "topic_value": "식사",
                "poi_id": None,
                "poi_name": "",
            }
        ],
        "visited": [],
        "knows_poi": [{"poi_id": "C_1", "poi_name": "테스트카페", "cat": "카페"}],
        "state": {},
    },
    "AGT_B": {"appointments": [], "memories": [], "visited": [], "knows_poi": [], "state": {}},
}

EVENTS = {
    "AGT_A": [
        {
            "day": "2026-05-01",
            "time": "12:00",
            "poi_id": "C_1",
            "poi_name": "테스트카페",
            "lon": 126.981,
            "lat": 37.566,
            "cat": "식사",
            "l1": "식사",
            "spent": 12000,
            "sat": 0.8,
        }
    ],
    "AGT_B": [],
}


def test_lerp_position_clamps_and_interpolates():
    assert lerp_position([0, 10], [10, 20], -1) == [0, 10]
    assert lerp_position([0, 10], [10, 20], 2) == [10, 20]
    assert lerp_position([0, 10], [10, 20], 0.25) == [2.5, 12.5]


def test_build_viz_meta_handles_empty_first_frame_and_derives_events():
    meta = build_viz_meta(AGENTS, TIMELINE, MEMORIES, EVENTS)

    assert set(meta) == {
        "agent_count",
        "frame_summaries",
        "poi_index",
        "spend_bursts",
        "meetups",
        "rumor_edges",
        "building_features",
    }
    assert meta["agent_count"] == 2

    assert meta["frame_summaries"][0]["active_agents"] == 0
    assert meta["frame_summaries"][1]["active_agents"] == 2
    assert meta["frame_summaries"][1]["spend_total"] == 12000

    assert meta["poi_index"]["C_1"]["name"] == "테스트카페"
    assert meta["poi_index"]["C_1"]["spent_total"] == 12000

    assert meta["spend_bursts"][0]["agent_id"] == "AGT_A"
    assert meta["spend_bursts"][0]["amount"] == 12000
    assert meta["spend_bursts"][0]["frame_index"] == 1

    assert meta["meetups"][0]["conv_id"] == "conv_1"
    assert meta["meetups"][0]["lon"] == 126.981
    assert meta["meetups"][0]["lat"] == 37.566
    assert meta["meetups"][0]["agents"] == ["AGT_A", "AGT_B"]

    assert meta["rumor_edges"][0]["source_id"] == "AGT_B"
    assert meta["rumor_edges"][0]["target_id"] == "AGT_A"
    assert meta["rumor_edges"][0]["topic_type"] == "policy"

    assert len(meta["building_features"]["features"]) == 1
    assert meta["building_features"]["features"][0]["properties"]["height"] >= 18


def test_build_viz_meta_skips_meetups_when_target_day_is_outside_timeline():
    memories = {
        **MEMORIES,
        "AGT_A": {
            **MEMORIES["AGT_A"],
            "appointments": [
                {
                    **MEMORIES["AGT_A"]["appointments"][0],
                    "conv_id": "conv_outside",
                    "day": "2026-05-03",
                }
            ],
        },
    }

    meta = build_viz_meta(AGENTS, TIMELINE, memories, EVENTS)

    assert meta["meetups"] == []
