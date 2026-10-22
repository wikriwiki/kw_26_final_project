"""시각화 팩(scripts/sim/visualization_3d/pack.py) — 원본 JSON 과 값이 그대로 왕복하는지."""
import gzip
import hashlib
import json
import math
import struct
from array import array

import pytest

from scripts.sim.visualization_3d import pack

TYPECODES = {"float32": "f", "int32": "i", "uint8": "B", "uint16": "H", "uint32": "I"}


def _fixture():
    agents = [
        {"id": "AGT_A", "dist_code": "11110", "home_lon": 126.97, "home_lat": 37.57},
        {"id": "AGT_B", "dist_code": "11140", "home_lon": 126.99, "home_lat": 37.56},
        {"id": "AGT_A", "dist_code": "99999"},  # 중복은 처음 것만 쓴다
    ]
    timeline = [
        {"day": "2026-05-01", "hour": 0, "label": "D1 00:00", "day_idx": 0, "agents": []},
        {
            "day": "2026-05-01",
            "hour": 1,
            "label": "D1 01:00",
            "day_idx": 0,
            "agents": [
                {"id": "AGT_A", "lon": 126.981, "lat": 37.566, "cat": "카페", "l1": "카페", "dong": "청운동",
                 "intent": "커피", "sat": 0.8, "spent": 4500, "anchor": "zone:1"},
                {"id": "AGT_C", "lon": 127.01, "lat": 37.5, "cat": None, "l1": "", "dong": None,
                 "intent": None, "sat": None, "spent": None, "anchor": "residence"},
            ],
        },
        {
            "day": "2026-05-02",
            "hour": 0,
            "label": "D2 00:00",
            "day_idx": 1,
            "agents": [
                {"id": "AGT_B", "lon": 126.95, "lat": 37.55, "cat": "식사", "l1": "식사", "dong": "중림동",
                 "intent": "점심", "sat": 0.61, "spent": 12000.4, "anchor": "zone:2"},
            ],
        },
    ]
    memories = {
        "AGT_A": {
            "memories": [
                {"type": "visited", "day": "2026-05-01", "poi_id": "C_1"},
                {"type": "rumor", "day": "2026-05-01", "source": "AGT_B"},
            ],
            "visited": [{"type": "visited", "day": "2026-05-01", "poi_id": "C_1"}],
            "appointments": [{"conv_id": "X", "with_agent": "AGT_B", "day": "2026-05-01"}],
            "knows_poi": [],
            "state": {"2026-05-01": {"balance": 1000}},
        },
        "AGT_B": {"memories": [], "visited": [], "appointments": [], "knows_poi": [], "state": {}},
    }
    events = {"AGT_A": [{"day": "2026-05-01", "time": "01:00", "poi_id": "C_1", "poi_name": "카페", "spent": 4500}]}
    return agents, timeline, memories, events


def _column(chunk_bytes, spec):
    code = TYPECODES[spec["dtype"]]
    values = array(code)
    width = values.itemsize
    values.frombytes(chunk_bytes[spec["offset"] : spec["offset"] + spec["length"] * width])
    return values


def _decode(manifest, files):
    n = manifest["agent_count"]
    out = {}
    for chunk in manifest["chunks"]:
        raw = gzip.decompress(files[chunk["file"]])
        assert len(raw) == chunk["raw_bytes"]
        cols = {name: _column(raw, spec) for name, spec in chunk["offsets"].items()}
        for offset in range(chunk["frame_count"]):
            frame = chunk["frame_start"] + offset
            for agent in range(n):
                slot = offset * n + agent
                out[(frame, agent)] = {name: cols[name][slot] for name in cols}
    return out


def test_roundtrip_matches_source_values():
    agents, timeline, memories, events = _fixture()
    manifest, files = pack.build_pack(agents, timeline, memories, events, policy_dongs=[])
    roster = json.loads(gzip.decompress(files["agents.json.gz"]))

    assert [a["id"] for a in roster] == ["AGT_A", "AGT_B", "AGT_C"]
    assert roster[0]["dist_code"] == "11110"
    assert manifest["agent_count"] == 3 and manifest["frame_count"] == 3
    assert [c["frame_count"] for c in manifest["chunks"]] == [2, 1]  # 날짜별 묶음
    assert manifest["first_active_frame"] == 1

    decoded = _decode(manifest, files)
    index = {agent["id"]: position for position, agent in enumerate(roster)}
    dicts = manifest["dicts"]

    def word(name, code):
        return dicts[name][code - 1] if code else None

    for frame_index, frame in enumerate(timeline):
        present = {item["id"]: item for item in frame["agents"]}
        for agent_id, position in index.items():
            values = decoded[(frame_index, position)]
            item = present.get(agent_id)
            if item is None:
                assert math.isnan(values["x"]) and math.isnan(values["y"])
                assert values["spent"] == 0 and values["cat"] == 0
                continue
            assert values["x"] == struct.unpack("f", struct.pack("f", item["lon"]))[0]
            assert values["y"] == struct.unpack("f", struct.pack("f", item["lat"]))[0]
            assert word("cat", values["cat"]) == (item["cat"] or None)
            assert word("l1", values["l1"]) == (item["l1"] or None)
            assert word("dong", values["dong"]) == item["dong"]
            assert word("intent", values["intent"]) == item["intent"]
            assert word("anchor", values["anchor"]) == item["anchor"]
            if item["sat"] is None:
                assert math.isnan(values["sat"])
            else:
                assert values["sat"] == pytest.approx(item["sat"], abs=1e-6)
            assert values["spent"] == round(item["spent"] or 0)


def test_totals_appointments_and_detail_shards():
    agents, timeline, memories, events = _fixture()
    manifest, files = pack.build_pack(agents, timeline, memories, events, policy_dongs=[], shard_size=2)

    # 예전 화면의 computeMemoryTotals 와 같은 규칙
    assert manifest["totals"] == {"visited": 1, "appointments": 1, "rumors": 1}
    assert manifest["appointment_agents"] == [0]

    shards = manifest["detail"]["shards"]
    assert [(s["agent_start"], s["agent_count"]) for s in shards] == [(0, 2), (2, 1)]
    first = json.loads(gzip.decompress(files[shards[0]["file"]]))
    assert set(first["agents"]) == {"AGT_A", "AGT_B"}
    bundle = first["agents"]["AGT_A"]
    assert "visited" not in bundle["memories"]  # memories 의 복사본이라 뺀다
    assert bundle["memories"]["memories"] == memories["AGT_A"]["memories"]
    assert bundle["events"] == events["AGT_A"]
    assert json.loads(gzip.decompress(files[shards[1]["file"]]))["agents"] == {}


def test_digests_are_recorded_and_output_is_deterministic(tmp_path):
    agents, timeline, memories, events = _fixture()
    first_manifest, first_files = pack.build_pack(agents, timeline, memories, events, policy_dongs=[])
    second_manifest, second_files = pack.build_pack(agents, timeline, memories, events, policy_dongs=[])
    assert first_files == second_files
    for entry in [first_manifest["agents"], first_manifest["extra"], *first_manifest["chunks"], *first_manifest["detail"]["shards"]]:
        data = first_files[entry["file"]]
        assert entry["bytes"] == len(data)
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()

    out = pack.write_pack(tmp_path / "PACK1", first_manifest, first_files)
    assert pack.verify_pack(out) == []
    assert pack.read_manifest(out)["format"] == pack.PACK_FORMAT
    assert not list(tmp_path.glob(".PACK1.*"))  # 임시 폴더를 남기지 않는다

    (out / "timeline" / "000.bin.gz").write_bytes(b"corrupt")
    assert pack.verify_pack(out) == ["해시 불일치: timeline/000.bin.gz"]


def test_rewrite_replaces_previous_pack(tmp_path):
    agents, timeline, memories, events = _fixture()
    manifest, files = pack.build_pack(agents, timeline, memories, events, policy_dongs=[], pack_id="P")
    target = tmp_path / "P"
    pack.write_pack(target, manifest, files)
    (target / "stale.txt").write_text("old", encoding="utf-8")
    pack.write_pack(target, manifest, files)
    assert not (target / "stale.txt").exists()
    assert pack.verify_pack(target) == []


def test_dictionary_codes_widen_beyond_uint8():
    agents = [{"id": f"AGT_{i}"} for i in range(300)]
    timeline = [{"day": "2026-05-01", "hour": 0, "label": "x", "agents": [
        {"id": f"AGT_{i}", "lon": 127.0, "lat": 37.5, "intent": f"의도 {i}"} for i in range(300)
    ]}]
    manifest, files = pack.build_pack(agents, timeline, {}, {}, policy_dongs=[])
    assert manifest["columns"]["intent"]["dtype"] == "uint16"
    assert manifest["columns"]["cat"]["dtype"] == "uint8"
    decoded = _decode(manifest, files)
    assert manifest["dicts"]["intent"][decoded[(0, 299)]["intent"] - 1] == "의도 299"


def test_empty_timeline_is_rejected():
    with pytest.raises(pack.PackError):
        pack.build_pack([{"id": "A"}], [], {}, {}, policy_dongs=[])
